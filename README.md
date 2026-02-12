# Query Right - Legal Document Q&A System

A Retrieval-Augmented Generation (RAG) pipeline for legal and policy document analysis, built with Gemma 3 4B, LangChain, ChromaDB, and Streamlit. Users upload legal documents and ask questions, receiving answers grounded in the source material with precise citations for verification.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Pipeline Deep Dive](#pipeline-deep-dive)
  - [Document Ingestion](#1-document-ingestion)
  - [Embedding & Vector Storage](#2-embedding--vector-storage)
  - [RAG Retrieval Chain](#3-rag-retrieval-chain)
  - [Query Routing](#4-query-routing)
  - [Web Search Fallback](#5-web-search-fallback)
- [Fine-Tuning with LoRA](#fine-tuning-with-lora)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
                                    ┌─────────────────┐
                                    │   Streamlit UI   │
                                    │  (Upload + Chat) │
                                    └────────┬────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Query Router    │
                                    │  (LLM Classifier)│
                                    └───┬─────────┬───┘
                                        │         │
                               LOCAL ◄──┘         └──► WEB
                                        │                 │
                               ┌────────▼────────┐  ┌────▼──────────┐
                               │  RAG Retrieval   │  │  Tavily API   │
                               │  Chain           │  │  Web Search   │
                               └────────┬────────┘  └────┬──────────┘
                                        │                 │
                               ┌────────▼────────┐       │
                               │    ChromaDB      │       │
                               │  Vector Store    │       │
                               └────────┬────────┘       │
                                        │                 │
                               ┌────────▼────────┐  ┌────▼──────────┐
                               │   Gemma 3 4B    │  │   Gemma 3 4B  │
                               │  (via Ollama)    │  │  (via Ollama)  │
                               │  + Doc Context   │  │  + Web Context │
                               └────────┬────────┘  └────┬──────────┘
                                        │                 │
                                        └────────┬───────┘
                                                 │
                                    ┌────────────▼───────────┐
                                    │   Answer + Citations   │
                                    └────────────────────────┘
```

**Data Flow:**

1. User uploads PDF/TXT/DOCX documents via the Streamlit sidebar
2. Documents are loaded, chunked (1000 chars with 200 overlap), and embedded using `all-MiniLM-L6-v2`
3. Embeddings are stored persistently in ChromaDB
4. When a user asks a question, the query router classifies it as LOCAL (document) or WEB (internet)
5. LOCAL queries retrieve the top-5 most relevant chunks via MMR search, then Gemma 3 generates an answer citing sources
6. WEB queries are sent to Tavily API, results are passed to Gemma 3 for synthesis
7. The answer with expandable source citations is displayed in the chat interface

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Gemma 3 4B (via Ollama) | Answer generation, query classification |
| **Embeddings** | all-MiniLM-L6-v2 (sentence-transformers) | Semantic vector embeddings (384-dim) |
| **Vector DB** | ChromaDB 1.5 (PersistentClient) | Document chunk storage and similarity search |
| **Orchestration** | LangChain 1.2 | Prompt templates, chains, output parsing |
| **Web Search** | Tavily API | Real-time internet search fallback |
| **Document Parsing** | PyMuPDF (fitz), python-docx | PDF and DOCX text extraction |
| **UI** | Streamlit | Interactive web interface |
| **Fine-tuning** | HuggingFace PEFT + TRL + QLoRA | Domain adaptation on legal terminology |

---

## Project Structure

```
query-right/
│
├── app.py                          # Streamlit entry point
├── config.py                       # All tunable constants and env vars
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore
│
├── core/                           # Foundation layer
│   ├── embeddings.py               # LangChain-compatible embedding wrapper
│   ├── llm.py                      # Gemma 3 4B factory (ChatOllama)
│   └── vectorstore.py              # ChromaDB client and retriever
│
├── ingestion/                      # Document processing pipeline
│   ├── loader.py                   # Multi-format loaders (PDF, TXT, DOCX)
│   ├── chunker.py                  # Legal-tuned RecursiveCharacterTextSplitter
│   └── pipeline.py                 # Orchestrator: load → chunk → embed → store
│
├── retrieval/                      # Query answering
│   ├── retriever.py                # Chunk retrieval + citation extraction
│   └── chain.py                    # RAG chain with legal QA system prompt
│
├── routing/                        # Intelligent query routing
│   ├── classifier.py               # LLM-based LOCAL vs WEB classifier
│   ├── web_search.py               # Tavily search + LLM synthesis
│   └── router.py                   # Route orchestrator
│
├── ui/                             # Streamlit components
│   ├── sidebar.py                  # Document upload and management panel
│   └── chat.py                     # Chat interface with citation rendering
│
├── fine_tuning/                    # Domain adaptation (standalone)
│   ├── train.py                    # QLoRA training script
│   ├── prepare_data.py             # CSV → JSONL data converter
│   └── sample_data/
│       └── legal_qa_pairs.jsonl    # 16 sample legal Q&A training pairs
│
└── data/                           # Runtime data (gitignored)
    ├── chroma_db/                  # Persistent vector storage
    └── uploads/                    # Temporary uploaded files
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- (Optional) [Tavily API key](https://tavily.com/) for web search

### Step 1: Pull the Gemma 3 4B Model

```bash
ollama pull gemma3:4b
```

### Step 2: Install Python Dependencies

```bash
cd query-right
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
cp .env.example .env
# Edit .env and add your Tavily API key (optional)
```

### Step 4: Launch the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

### Uploading Documents

1. Use the sidebar **Upload Documents** section
2. Drag-and-drop or browse for PDF, TXT, or DOCX files
3. Click **Process Documents** — the pipeline loads, chunks, embeds, and indexes
4. The sidebar shows all indexed documents with chunk counts

### Asking Questions

Type a question in the chat input. Examples:

- *"What are the termination conditions in the agreement?"* → Routes to LOCAL, retrieves from your documents
- *"What are the latest GDPR enforcement actions in 2024?"* → Routes to WEB, searches Tavily

### Source Citations

Every answer includes an expandable **Sources** section showing:
- Document name and page number (for local docs)
- Title and URL (for web sources)
- A preview of the relevant text chunk

### Managing Documents

- View all indexed documents in the sidebar under **Indexed Documents**
- Click **X** next to a document to remove it from the index
- The **Total Chunks** metric shows how many chunks are stored

---

## Pipeline Deep Dive

### 1. Document Ingestion

**File:** `ingestion/loader.py`

Three format-specific loaders dispatch based on file extension:

| Format | Library | Strategy |
|--------|---------|----------|
| PDF | PyMuPDF (`fitz`) | Per-page extraction preserving page numbers in metadata |
| TXT | Built-in `open()` | Full file as single document |
| DOCX | python-docx | Concatenated paragraphs (skips empty) |

**Why PyMuPDF over pypdf:** PyMuPDF provides superior text extraction for complex legal layouts — multi-column pages, footnotes, tables of contents, and cross-references.

**File:** `ingestion/chunker.py`

Uses `RecursiveCharacterTextSplitter` with legal-tuned separators:

```python
separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
chunk_size = 1000    # Larger for legal context preservation
chunk_overlap = 200  # Preserves cross-boundary references
```

**Why 1000-char chunks:** Legal text contains long, complex sentences with dependent clauses. A 500-char chunk frequently splits mid-sentence, losing context like "notwithstanding the foregoing..." references. Five chunks of ~250 tokens each fit well within Gemma 3's 8192-token context window.

**Why semicolon separator:** Legal documents use semicolons heavily in statutory lists (e.g., "the parties shall: (a) ...; (b) ...; (c) ..."). Splitting at semicolons preserves individual list items as coherent units.

**File:** `ingestion/pipeline.py`

The `IngestionPipeline` class orchestrates the full flow and generates deterministic chunk IDs (`{filename}_{page}_{chunk_index}`) to prevent duplicate insertion on re-upload.

### 2. Embedding & Vector Storage

**File:** `core/embeddings.py`

Wraps `sentence-transformers` (`all-MiniLM-L6-v2`) into a LangChain-compatible `Embeddings` interface.

**Why sentence-transformers instead of Ollama embeddings:**
- `all-MiniLM-L6-v2` is purpose-built for semantic similarity (384-dim output) — fast and proven
- Gemma 3 via Ollama generates embeddings as a side effect of its LLM architecture, which is slower
- Separating embedding and generation lets Ollama focus resources on generation

**File:** `core/vectorstore.py`

Uses ChromaDB's `PersistentClient` for durable storage across app restarts. The retriever uses **Maximal Marginal Relevance (MMR)** search instead of plain similarity to ensure diversity in retrieved chunks — avoiding 5 nearly-identical chunks from the same paragraph.

### 3. RAG Retrieval Chain

**File:** `retrieval/chain.py`

The chain follows the pattern: **Retrieve → Format Context → Generate Answer**

The system prompt is carefully designed for legal accuracy:
- Instructs the model to cite sources using `[Source N: filename, Page X]` markers
- Explicitly refuses hallucination: *"If the context does not contain enough information, say so"*
- Requires precise legal language matching the source material
- Synthesizes across multiple sources when relevant

**Context formatting** (`retrieval/retriever.py`) injects source attribution markers before each chunk so the LLM can reference them naturally in its response.

### 4. Query Routing

**File:** `routing/classifier.py`

An LLM-based classifier (zero temperature for determinism) categorizes each query as LOCAL or WEB:

- **LOCAL:** Questions about specific documents, clauses, sections, terms, definitions
- **WEB:** Questions about current events, general legal concepts, public regulations

**Why LLM-based over keyword-based:** A keyword classifier looking for "current" or "latest" would misclassify *"What is the current termination clause?"* as a web query when it's clearly about a local document. The LLM understands semantic intent.

### 5. Web Search Fallback

**File:** `routing/web_search.py`

When a query is classified as WEB:
1. Tavily API searches the internet (3 results by default)
2. Results are formatted into a context string with `[Web Source N]` markers
3. Gemma 3 synthesizes an answer citing the web sources
4. Citations include title, URL, and content preview

---

## Fine-Tuning with LoRA

The `fine_tuning/` directory contains a standalone training pipeline for domain adaptation.

### Training Data Format

JSONL with chat-style messages:

```json
{"messages": [{"role": "user", "content": "What is force majeure?"}, {"role": "assistant", "content": "Force majeure is..."}]}
```

16 sample legal Q&A pairs are provided in `fine_tuning/sample_data/legal_qa_pairs.jsonl`.

### Converting Custom Data

If you have Q&A data in CSV format:

```bash
python fine_tuning/prepare_data.py --input your_data.csv --output training_data.jsonl
```

CSV must have `question` and `answer` columns.

### Running Fine-Tuning

Requires a GPU with 8GB+ VRAM:

```bash
pip install torch transformers peft trl bitsandbytes datasets accelerate

python fine_tuning/train.py \
    --data fine_tuning/sample_data/legal_qa_pairs.jsonl \
    --output ./fine_tuned_model \
    --epochs 3 \
    --batch_size 4
```

### Training Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--base_model` | google/gemma-3-4b-it | HuggingFace model ID |
| `--lora_r` | 16 | LoRA rank (higher = more capacity, more memory) |
| `--lora_alpha` | 32 | LoRA scaling factor (typically 2x rank) |
| `--learning_rate` | 2e-4 | AdamW learning rate |
| `--max_seq_length` | 1024 | Max tokens per training sample |

Uses **QLoRA** (4-bit NF4 quantization + LoRA) to fit training on consumer GPUs. Targets all attention and MLP projection layers: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

### Deploying the Fine-Tuned Model

After training:

1. Merge adapter weights and convert to GGUF format using `llama.cpp`
2. Create an Ollama Modelfile:
   ```
   FROM ./your_model.gguf
   PARAMETER temperature 0.3
   PARAMETER num_ctx 8192
   ```
3. Register: `ollama create query-right-legal -f Modelfile`
4. Update `LLM_MODEL` in `.env` to `query-right-legal`

---

## Configuration Reference

All configuration lives in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `gemma3:4b` | Ollama model name |
| `LLM_TEMPERATURE` | `0.3` | Generation temperature (low for factual) |
| `LLM_NUM_CTX` | `8192` | Context window size |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `RETRIEVAL_SEARCH_TYPE` | `mmr` | Search strategy (mmr or similarity) |
| `TAVILY_API_KEY` | *(empty)* | Tavily API key for web search |
| `TAVILY_MAX_RESULTS` | `3` | Web search results count |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ConnectionError: Ollama not running` | Start Ollama: `ollama serve` |
| `Model not found: gemma3:4b` | Pull it: `ollama pull gemma3:4b` |
| Slow first query | The embedding model downloads on first use (~90MB). Subsequent runs are instant. |
| Empty answers | Check that documents are indexed (sidebar shows chunk count > 0) |
| Web search not working | Add your Tavily API key in the sidebar settings or `.env` file |
| ChromaDB errors after upgrade | Delete `data/chroma_db/` to reset the vector store |
| Import errors | Run `pip install -r requirements.txt` to ensure all dependencies |
