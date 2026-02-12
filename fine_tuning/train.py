"""
Standalone LoRA fine-tuning script for Gemma 3 4B on legal Q&A.

Usage:
    python fine_tuning/train.py \
        --data fine_tuning/sample_data/legal_qa_pairs.jsonl \
        --output ./fine_tuned_model \
        --epochs 3 \
        --batch_size 4

Requirements (install separately):
    pip install torch transformers peft trl bitsandbytes datasets accelerate

After training, export to GGUF and register with Ollama:
    1. Merge adapter: python fine_tuning/train.py --merge --output ./merged_model
    2. Convert to GGUF using llama.cpp
    3. Create Modelfile and run: ollama create query-right-legal -f Modelfile
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 4B with LoRA")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL training data")
    parser.add_argument("--output", type=str, default="./fine_tuned_model")
    parser.add_argument("--base_model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()

    # 4-bit QLoRA quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA configuration targeting attention and MLP layers
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        r=args.lora_r,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # Load training data (JSONL with "messages" field)
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"Training samples: {len(dataset)}")

    training_args = SFTConfig(
        output_dir=args.output,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model()
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
