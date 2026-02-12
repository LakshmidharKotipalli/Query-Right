"""
Utility to convert raw Q&A data into the JSONL format expected by the trainer.

Input format (CSV or JSON):
    question, answer

Output format (JSONL):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import argparse
import csv
import json


def csv_to_jsonl(input_path: str, output_path: str):
    """Convert a CSV file with 'question' and 'answer' columns to training JSONL."""
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        count = 0
        for row in reader:
            entry = {
                "messages": [
                    {"role": "user", "content": row["question"].strip()},
                    {"role": "assistant", "content": row["answer"].strip()},
                ]
            }
            f_out.write(json.dumps(entry) + "\n")
            count += 1
    print(f"Converted {count} Q&A pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Q&A data to training format")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()
    csv_to_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()
