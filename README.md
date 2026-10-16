# Query Right

A local RAG app for document Q&A with agentic query routing. Upload documents and ask questions; answers are
grounded in your sources (or the web, when needed) and come with citations.

Built with Gemma 3 (via Ollama), LangChain, ChromaDB and Streamlit.

## How it works

A small router model (`gemma3:270m`) classifies every query, then Gemma 3 4B answers from the right context:

| Route | Context used |
|---|---|
| **LOCAL** | Top chunks from your documents in ChromaDB |
| **WEB** | DuckDuckGo search results |
| **COMBINED** | Documents and web searched in parallel, then merged |
| **DIRECT** | No retrieval; the model answers from its own knowledge |

```
Query → gemma3:270m router → LOCAL / WEB / COMBINED / DIRECT → Gemma 3 4B → answer + citations
```

Documents (PDF, TXT, DOCX) are chunked (1000 chars, 200 overlap), embedded with `all-MiniLM-L6-v2`, and stored
persistently in ChromaDB.

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com/) running.

```bash
ollama pull gemma3:4b
pip install -r requirements.txt
streamlit run app.py   # http://localhost:8501
```

## Configuration

Defaults live in `config.py` and can be overridden in `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemma3:4b` | Answering model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Chunking |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved |
| `RETRIEVAL_SEARCH_TYPE` | `mmr` | `mmr` or `similarity` |
| `WEB_CRAWL_MAX_RESULTS` | `5` | DuckDuckGo results |
