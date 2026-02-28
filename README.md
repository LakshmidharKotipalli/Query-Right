# Query Right

A Retrieval-Augmented Generation (RAG) pipeline for document Q&A, built with Gemma 3 4B, LangChain, ChromaDB, and Streamlit. Upload any document and ask questions — answers are grounded in the source material with citations, never made up.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Setup & Installation](#setup--installation)
- [Configuration Reference](#configuration-reference)

---

## Architecture Overview

```
                              ┌─────────────────────┐
                              │     Streamlit UI    │
                              │   (Upload + Chat)   │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    gemma3:270m      │  ← small model, zero-latency routing
                              │ (Agentic Classifier)│
                              └──┬────────┬───────┬─┘
                                 │        │       │
                              LOCAL    COMBINED  WEB   DIRECT
                                 │        │       │      │
                    ┌────────────┘     ┌──┘       └──┐   └──────────────┐
                    │              ┌───┴───┐          │                 │
           ┌────────▼────────┐    LOCAL  WEB    ┌──────▼──────┐  ┌───────▼───────┐
           │    ChromaDB     │    in parallel   │  DuckDuckGo │  │  Gemma 3 4B   │
           │    Retrieval    │◄───┘       └────►│  Web Search │  │  (no context) │
           └────────┬────────┘                  └──────┬──────┘  └───────┬───────┘
                    │                                  │                 │
                    └──────────────┬───────────────────┘                 │
                                   │                                     │
                        ┌──────────▼──────────┐                          │
                        │    Merge Context    │  ← club results          │
                        └──────────┬──────────┘                          │
                                   │                                     │
                        ┌──────────▼──────────┐                          │
                        │     Gemma 3 4B      │                          │
                        │    (via Ollama)     │                          │
                        └──────────┬──────────┘                          │
                                   │                                     │
                                   └──────────────┬──────────────────────┘
                                                  │
                                       ┌──────────▼──────────┐
                                       │  Answer + Citations │
                                       └─────────────────────┘
```

**Data Flow:**

1. User uploads PDF/TXT/DOCX documents via the Streamlit sidebar
2. Documents are loaded, chunked (1000 chars with 200 overlap), and embedded using `all-MiniLM-L6-v2`
3. Embeddings are stored persistently in ChromaDB
4. Each query is first routed by `gemma3:270m` — a lightweight model used purely for classification (LOCAL, WEB, COMBINED, or DIRECT)
5. LOCAL queries retrieve the most relevant chunks from ChromaDB; WEB queries fetch results from DuckDuckGo
6. For COMBINED queries, both ChromaDB and DuckDuckGo run in parallel and their results are merged into a single context
7. For DIRECT queries (no documents needed, no live data needed), Gemma 3 4B answers straight from its own knowledge
8. The context (or none, for DIRECT) is passed to Gemma 3 4B which generates a grounded answer with source citations

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

### Step 1: Pull the Gemma 3 4B Model

```bash
ollama pull gemma3:4b
```

### Step 2: Install Python Dependencies

```bash
cd query-right
pip install -r requirements.txt
```

### Step 3: Launch the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

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
| `WEB_CRAWL_MAX_RESULTS` | `5` | DuckDuckGo results count |
