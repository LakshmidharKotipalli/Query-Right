# Query Right — Interview Preparation Guide

This document covers every technical concept, design decision, and potential interview question related to the Query Right project. Organized by topic for quick reference.

---

## Table of Contents

1. [Project Elevator Pitch](#1-project-elevator-pitch)
2. [RAG Fundamentals](#2-rag-fundamentals)
3. [Embedding & Vector Search](#3-embedding--vector-search)
4. [Document Processing & Chunking](#4-document-processing--chunking)
5. [LLM Integration (Gemma 3 + Ollama)](#5-llm-integration-gemma-3--ollama)
6. [LangChain Architecture](#6-langchain-architecture)
7. [ChromaDB & Vector Databases](#7-chromadb--vector-databases)
8. [Query Routing & Classification](#8-query-routing--classification)
9. [Web Search Integration (Tavily)](#9-web-search-integration-tavily)
10. [Fine-Tuning with LoRA/QLoRA](#10-fine-tuning-with-loraqLora)
11. [Prompt Engineering](#11-prompt-engineering)
12. [Streamlit & UI Design](#12-streamlit--ui-design)
13. [System Design & Scalability](#13-system-design--scalability)
14. [Common Interview Questions & Answers](#14-common-interview-questions--answers)
15. [Behavioral / Project Discussion Questions](#15-behavioral--project-discussion-questions)
16. [Transformer Architecture Deep Dive](#16-transformer-architecture-deep-dive)
17. [Tokenization & Text Processing](#17-tokenization--text-processing)
18. [Quantization Deep Dive](#18-quantization-deep-dive)
19. [Advanced RAG Patterns](#19-advanced-rag-patterns)
20. [Open-Source LLM Landscape](#20-open-source-llm-landscape)
21. [Vector Distance Metrics & Similarity Search](#21-vector-distance-metrics--similarity-search)
22. [Hallucination Detection & Mitigation](#22-hallucination-detection--mitigation)
23. [RAG Evaluation Frameworks](#23-rag-evaluation-frameworks)
24. [Security & Privacy in RAG Systems](#24-security--privacy-in-rag-systems)
25. [MLOps & Production Deployment](#25-mlops--production-deployment)
26. [Cost Analysis: Local vs API](#26-cost-analysis-local-vs-api)
27. [Ethical & Legal Considerations](#27-ethical--legal-considerations)
28. [Python & Software Engineering Patterns](#28-python--software-engineering-patterns)
29. [Coding Questions You May Be Asked](#29-coding-questions-you-may-be-asked)
30. [Failure Scenarios & Edge Cases](#30-failure-scenarios--edge-cases)
31. [Key Papers & References](#31-key-papers--references)

---

## 1. Project Elevator Pitch

> "I built Query Right, a RAG-based Q&A system for legal and policy documents. Users upload PDFs, TXT, or DOCX files through a Streamlit interface, and the pipeline automatically chunks, embeds, and indexes them in ChromaDB. When a user asks a question, an LLM-based query router determines whether to answer from the local document corpus or search the web via Tavily API. The system uses Gemma 3 4B served through Ollama for generation, sentence-transformers for embeddings, and returns answers with precise source citations — document name, page number, and text preview — so users can verify every claim. I also built a standalone LoRA fine-tuning pipeline to adapt the model to domain-specific legal terminology."

**Key numbers to mention:**
- Supports 3 document formats (PDF, TXT, DOCX)
- 384-dimensional embeddings via all-MiniLM-L6-v2
- 1000-char chunks with 200-char overlap, tuned for legal text
- Top-5 MMR retrieval for diverse, relevant context
- Automatic LOCAL/WEB routing with zero-temperature classification
- QLoRA fine-tuning targeting 7 projection layers with rank-16 adapters

---

## 2. RAG Fundamentals

### What is RAG?

Retrieval-Augmented Generation combines information retrieval with language model generation. Instead of relying solely on the LLM's parametric knowledge (which can hallucinate), RAG retrieves relevant documents from an external knowledge base and includes them as context in the prompt.

### RAG Pipeline Steps (as implemented in Query Right)

```
User Query
    │
    ▼
Query Embedding (all-MiniLM-L6-v2)
    │
    ▼
Vector Similarity Search (ChromaDB, MMR, top-5)
    │
    ▼
Context Formatting (source markers + chunk text)
    │
    ▼
Prompt Construction (system prompt + context + question)
    │
    ▼
LLM Generation (Gemma 3 4B via Ollama)
    │
    ▼
Answer with Source Citations
```

### Q: Why RAG instead of fine-tuning alone?

**Answer:** RAG and fine-tuning serve different purposes:
- **RAG** grounds the model in specific, up-to-date documents. It's ideal when the knowledge base changes frequently (new contracts, updated policies). No retraining needed — just re-index.
- **Fine-tuning** teaches the model domain-specific language patterns and reasoning styles. It improves *how* the model responds, not *what* it knows.
- **In Query Right**, we use both: RAG for document-grounded answers, and LoRA fine-tuning to improve the model's understanding of legal terminology. They're complementary, not competing approaches.

### Q: What are the limitations of RAG?

**Answer:**
- **Retrieval quality bottleneck**: If the retriever doesn't find the right chunks, the LLM can't generate a good answer. Garbage in, garbage out.
- **Context window limits**: You can only pass a finite number of chunks. With Gemma 3's 8K context, we're limited to ~5 chunks of 1000 chars each.
- **Lost in the middle**: LLMs tend to pay more attention to the beginning and end of context, potentially missing information in the middle.
- **No cross-document reasoning**: If the answer requires synthesizing information spread across many documents, single-hop retrieval may miss pieces.

---

## 3. Embedding & Vector Search

### Q: Why all-MiniLM-L6-v2 instead of Ollama embeddings or OpenAI?

**Answer:**
- **all-MiniLM-L6-v2** is a purpose-built semantic similarity model that produces 384-dimensional vectors. It's optimized for retrieval tasks with a proven track record on MTEB benchmarks.
- **Ollama embeddings** (using Gemma 3) generate embeddings as a side effect of the LLM architecture. They produce higher-dimensional vectors that don't necessarily outperform dedicated embedding models for retrieval. Also, using the same model for both embedding and generation creates resource contention.
- **OpenAI embeddings** require an API key, introduce latency from network calls, and cost money per token. For a local-first application, sentence-transformers is the better choice.

### Q: What is the difference between cosine similarity and MMR?

**Answer:**
- **Cosine similarity** returns the K most similar chunks to the query. Problem: if a paragraph is split into 3 chunks, all 3 might be retrieved, wasting 3 of your 5 retrieval slots on essentially the same information.
- **MMR (Maximal Marginal Relevance)** balances relevance and diversity. It iteratively selects chunks that are similar to the query BUT dissimilar to already-selected chunks. This ensures broader coverage of the document corpus.
- **In Query Right**, we use MMR because legal documents often have repetitive language (e.g., boilerplate clauses), and we need diverse chunks to cover different aspects of a question.

### Q: How do vector embeddings capture semantic meaning?

**Answer:** Transformer-based embedding models (like MiniLM) map text into a high-dimensional vector space where semantically similar texts are closer together. The model was pre-trained on large corpora and fine-tuned on sentence pairs for similarity tasks. The key insight is that the model learns contextual representations — "termination of employment" and "firing an employee" produce similar vectors despite different surface forms. This is fundamentally different from keyword matching (TF-IDF/BM25) which would treat these as unrelated.

### Q: What is the dimensionality of your embeddings and why does it matter?

**Answer:** 384 dimensions. Higher dimensions (e.g., OpenAI's 1536 or 3072) can capture more nuance but require more storage, memory, and compute for similarity calculations. For legal document retrieval, 384 dimensions provide an excellent quality-to-efficiency tradeoff. ChromaDB stores these vectors and uses HNSW (Hierarchical Navigable Small World) indexing for approximate nearest neighbor search.

---

## 4. Document Processing & Chunking

### Q: Why is chunking necessary? Why not embed entire documents?

**Answer:**
1. **Embedding models have token limits** — all-MiniLM-L6-v2 has a 512-token limit. A 50-page legal document far exceeds this.
2. **Granularity for retrieval** — if you embed an entire document, you can only retrieve "the whole document" or nothing. Chunking allows retrieving specific relevant sections.
3. **LLM context limits** — even if you could retrieve whole documents, Gemma 3's 8K context can't fit a 50-page document. Chunks keep context manageable.

### Q: How did you decide on 1000-char chunks with 200-char overlap?

**Answer:**
- **1000 chars (vs. typical 500)**: Legal text has long, complex sentences with dependent clauses ("notwithstanding the foregoing provisions of Section 4(a)..."). Smaller chunks frequently split mid-clause, losing critical cross-references. 1000 chars preserves most complete legal paragraphs.
- **200-char overlap**: Ensures that sentences or clauses spanning chunk boundaries are fully captured in at least one chunk. For legal text, this is crucial because a definition in the last line of chunk N may be essential context for understanding chunk N+1.
- **Token budget check**: 5 chunks × ~250 tokens each = ~1250 tokens of context. With the system prompt (~200 tokens) and question (~50 tokens), we're well within Gemma 3's 8192-token context.

### Q: Explain RecursiveCharacterTextSplitter and your separator hierarchy.

**Answer:** `RecursiveCharacterTextSplitter` attempts to split text using the first separator that creates chunks within the target size. It "recurses" through the separator list only when a higher-priority separator produces chunks that are still too large.

Our hierarchy:
1. `\n\n` — Paragraph breaks (most natural split point in legal docs, often delineates sections)
2. `\n` — Line breaks (clauses within a section)
3. `. ` — Sentence boundaries (preserves complete sentences)
4. `; ` — Semicolons (legal documents use these heavily in enumerated lists: "(a) ...; (b) ...; (c) ...")
5. `, ` — Clause boundaries (within a sentence)
6. ` ` — Word boundaries (last resort to avoid splitting mid-word)

### Q: Why PyMuPDF over pypdf for PDF extraction?

**Answer:** PyMuPDF (`fitz`) provides significantly better text extraction for complex legal document layouts:
- **Multi-column pages**: Legal briefs and court filings often use two columns. PyMuPDF correctly reads column order; pypdf sometimes interleaves columns.
- **Headers/footers**: PyMuPDF provides better spatial awareness for extracting meaningful text.
- **Tables**: Legal contracts contain tables (payment schedules, definitions). PyMuPDF handles these more gracefully.
- **Performance**: PyMuPDF is a C-based library (bindings to MuPDF), typically 3-5x faster than pypdf for large documents.

---

## 5. LLM Integration (Gemma 3 + Ollama)

### Q: Why Gemma 3 4B specifically?

**Answer:**
- **Quality**: Gemma 3 4B outperforms many 7B models from previous generations (Llama 2 7B, Mistral 7B) on instruction-following benchmarks. Google's training methodology (knowledge distillation from larger models) gives it outsized quality for its parameter count.
- **Efficiency**: At 4B parameters, it runs comfortably on 8GB+ VRAM or Apple Silicon Macs with minimal quantization. Inference is fast enough for interactive chat.
- **Instruction-tuned**: The `gemma3:4b` Ollama model is the instruction-tuned variant, meaning it follows instructions well out of the box — critical for legal QA where precise instruction adherence prevents hallucination.
- **Open weights**: Freely available under Google's terms, no API costs, fully local.

### Q: Why Ollama as the serving layer?

**Answer:**
- **One-command setup**: `ollama pull gemma3:4b` downloads and optimizes the model automatically.
- **Automatic quantization**: Ollama applies appropriate quantization (typically Q4_K_M) to balance quality and speed for the available hardware.
- **OpenAI-compatible API**: LangChain's `ChatOllama` integrates seamlessly.
- **Hot-swap models**: Changing `LLM_MODEL` in config switches to a different model without code changes.
- **Comparison to alternatives**: vLLM offers higher throughput for production but is heavier to set up. HuggingFace Transformers requires manual GPU memory management. Ollama is the best fit for a local development/demo tool.

### Q: What is the difference between ChatOllama and Ollama (LLM)?

**Answer:** `ChatOllama` exposes a chat-based interface (system/user/assistant messages), while `Ollama` (the LLM wrapper) exposes a text completion interface (single text input → text output). For RAG with a system prompt and multi-turn conversation, `ChatOllama` is the correct choice because it preserves message roles, which Gemma 3's instruction template uses for proper formatting.

---

## 6. LangChain Architecture

### Q: How does your RAG chain work in LangChain?

**Answer:** The chain uses LangChain Expression Language (LCEL):

```python
chain = LEGAL_QA_PROMPT | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": question})
```

This is a pipeline: prompt template → LLM → string parser. The `|` operator creates a `RunnableSequence` where each component's output becomes the next component's input:
1. `ChatPromptTemplate` formats context + question into a list of messages
2. `ChatOllama` sends messages to Gemma 3, receives an `AIMessage`
3. `StrOutputParser` extracts the string content from the `AIMessage`

### Q: Why not use RetrievalQA chain or ConversationalRetrievalChain?

**Answer:** Those are LangChain's legacy chain classes (deprecated in favor of LCEL). I built the retrieval and generation as separate steps rather than a monolithic chain because:
1. **Separation of concerns**: Retrieval, formatting, and generation are independent operations. Keeping them separate makes testing and debugging easier.
2. **Citation extraction**: I need access to the retrieved documents *after* retrieval but *before* returning to the user, to extract citation metadata. A monolithic chain hides the intermediate documents.
3. **Custom routing**: The query router needs to decide *before* retrieval whether to use local docs or web search. This doesn't fit the standard RetrievalQA pattern.

### Q: What is LCEL and how does it differ from legacy chains?

**Answer:** LangChain Expression Language (LCEL) is a declarative way to compose LangChain components using the `|` pipe operator. Key differences:
- **Legacy chains** (LLMChain, SequentialChain): Python classes with hardcoded flow, limited composability, difficult to customize.
- **LCEL**: Functional composition using `Runnable` protocol. Every component implements `.invoke()`, `.batch()`, `.stream()`. Components compose naturally with `|`, and you get streaming, async, and batch for free.
- **In Query Right**: All chains use LCEL for consistency and modern LangChain best practices.

---

## 7. ChromaDB & Vector Databases

### Q: Why ChromaDB over Pinecone, Weaviate, or FAISS?

**Answer:**
- **ChromaDB**: Embedded database, zero infrastructure, persistent storage to disk, Python-native. Perfect for local-first applications.
- **Pinecone**: Cloud-hosted, requires API key and internet. Better for production at scale but overkill for a local tool.
- **Weaviate**: Self-hosted with Docker, feature-rich but heavy setup. Better for microservice architectures.
- **FAISS**: Meta's similarity search library. Extremely fast but lacks metadata filtering, persistence is manual, and it's a library not a database.
- **For Query Right**: ChromaDB's `PersistentClient` gives us durable storage across app restarts with zero ops overhead.

### Q: How does ChromaDB store and search vectors?

**Answer:** ChromaDB uses **HNSW (Hierarchical Navigable Small World)** graphs for approximate nearest neighbor search:
1. **Storage**: Vectors are stored alongside metadata and original text in a SQLite-backed persistent store.
2. **Indexing**: HNSW builds a multi-layer graph where each node is a vector. Higher layers have fewer, more spread-out nodes (for coarse navigation), lower layers are denser (for fine-grained search).
3. **Search**: A query vector enters at the top layer, greedily navigates to the nearest node, then descends to the next layer for finer search. This gives O(log N) search complexity instead of O(N) brute force.
4. **Metadata**: ChromaDB supports `where` filters on metadata, which we use for `delete_source()` to remove all chunks from a specific document.

### Q: How do you prevent duplicate documents in ChromaDB?

**Answer:** We generate deterministic IDs: `{filename}_{page}_{chunk_index}`. When the same document is re-uploaded, `add_documents` with the same IDs either skips or updates existing entries rather than creating duplicates. This is simpler and more reliable than checking for existing documents before insertion.

---

## 8. Query Routing & Classification

### Q: How does your query router work?

**Answer:** The router is a two-stage system:

1. **Classification**: An LLM-based classifier (Gemma 3 at temperature=0 for deterministic output) reads the query and responds with exactly one word: "LOCAL" or "WEB".
2. **Dispatch**: Based on the classification, the router calls either `rag_query()` (local ChromaDB retrieval) or `web_search_query()` (Tavily API).

### Q: Why LLM-based classification instead of keyword matching or a fine-tuned classifier?

**Answer:**
- **Keyword matching** is brittle. "What is the *current* termination clause?" contains the word "current" but is a LOCAL query about a document. "What are the *force majeure* provisions in recent court rulings?" is a WEB query despite containing legal terminology.
- **Fine-tuned classifier** (like a BERT classifier) would require labeled training data and a separate model to maintain. Overkill for this use case.
- **LLM-based classification** leverages Gemma 3's language understanding to interpret semantic intent. The cost is one extra inference call (~200ms) per query, which is negligible compared to the generation step.

### Q: What if the classifier makes a wrong decision?

**Answer:** We default to LOCAL on any unexpected output (the fallback in the classifier). This is a conservative choice — it's better to try local retrieval and get "no relevant documents found" than to accidentally send a confidential legal question to a web search API. In a production system, you could add user feedback to improve routing, or show both LOCAL and WEB results and let the user choose.

---

## 9. Web Search Integration (Tavily)

### Q: Why Tavily over Google Search API or SerpAPI?

**Answer:**
- **Tavily** is purpose-built for AI agents. It returns clean, parsed content (not raw HTML) and includes an optional `include_answer` field with a pre-generated summary.
- **Google Custom Search API**: Returns snippets and URLs but not full page content. Requires additional scraping.
- **SerpAPI**: Parses Google results but is more expensive and provides raw SERP data, not AI-ready content.
- **For RAG**: Tavily's clean content output integrates directly into the LLM's context without HTML parsing or cleaning.

### Q: How does the web search fallback integrate with the RAG chain?

**Answer:** The web search path mirrors the local RAG chain's structure:
1. Search Tavily (analogous to ChromaDB retrieval)
2. Format results with `[Web Source N]` markers (analogous to `format_context`)
3. Pass to Gemma 3 with a web-specific system prompt (analogous to `LEGAL_QA_PROMPT`)
4. Return answer + web citations (same output structure)

Both paths return the same `dict` shape (`answer`, `citations`, `source_documents`, `route`), so the UI renders them identically with the only difference being citation format (page numbers vs. URLs).

---

## 10. Fine-Tuning with LoRA/QLoRA

### Q: What is LoRA and how does it work?

**Answer:** **Low-Rank Adaptation (LoRA)** freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into each transformer layer. Instead of updating a full weight matrix W (d×d), LoRA learns two small matrices A (d×r) and B (r×d) where r << d. The effective weight becomes W + BA.

Key concepts:
- **Rank (r=16)**: Controls adapter capacity. Rank 16 means each adapter adds r×(d₁+d₂) parameters instead of d₁×d₂. For Gemma 3 4B, this reduces trainable parameters from 4B to ~20M.
- **Alpha (α=32)**: Scaling factor. The adapter output is scaled by α/r. Higher α increases the adapter's influence.
- **Target modules**: We target all 7 projection layers in each transformer block (q/k/v/o in attention, gate/up/down in MLP) for comprehensive adaptation.

### Q: What is QLoRA and why use it?

**Answer:** **QLoRA** (Quantized LoRA) combines 4-bit quantization of the base model with LoRA adapters:
1. The base model (Gemma 3 4B) is loaded in 4-bit NF4 quantization — reducing memory from ~8GB to ~2.5GB.
2. LoRA adapters are trained in bfloat16 precision on top of the frozen quantized model.
3. Forward pass: quantized weights are dequantized on-the-fly, adapter outputs are added.
4. Backward pass: only adapter weights receive gradients.

**Result**: Fine-tune a 4B model on an 8GB GPU that would normally require 16GB+ for full-precision training.

### Q: Why these specific target modules?

**Answer:**
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                  "gate_proj", "up_proj", "down_proj"]       # MLP
```

- **Attention projections (q/k/v/o)**: Adapt how the model attends to legal terminology and document structure. Critical for understanding cross-references and clause dependencies.
- **MLP projections (gate/up/down)**: Adapt the model's knowledge representations. Important for learning new legal concepts and terminology.
- **Why all 7 instead of just q/v (common default)**: Legal language is structurally different from general text. Adapting all projection layers gives the model more capacity to learn domain-specific patterns. The parameter increase is modest since r=16 is relatively low.

### Q: Explain the training configuration choices.

**Answer:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate: 2e-4 | Standard for QLoRA fine-tuning; higher than full fine-tuning because we're training far fewer parameters |
| Batch size: 4, gradient accumulation: 4 | Effective batch size of 16; small per-device batch fits in memory, accumulation increases effective batch |
| Warmup ratio: 0.03 | Short warmup (3% of steps) prevents early instability without wasting training budget |
| Cosine LR scheduler | Gradually reduces learning rate following a cosine curve; smoother than step decay, avoids sudden drops |
| Gradient checkpointing | Trades compute for memory; recomputes activations during backward pass instead of storing them |
| Max grad norm: 0.3 | Aggressive clipping prevents training instabilities common with quantized models |
| BF16 compute dtype | Better dynamic range than FP16 for training stability; supported on modern GPUs and Apple Silicon |

### Q: How would you evaluate the fine-tuned model?

**Answer:**
1. **Held-out test set**: Reserve 20% of legal Q&A pairs for evaluation. Compare base vs. fine-tuned model on answer quality.
2. **Domain-specific metrics**: Measure legal terminology accuracy — does the model correctly use terms like "indemnification," "force majeure," "severability"?
3. **RAG integration test**: Run the same queries through the full RAG pipeline with base vs. fine-tuned model. Compare citation accuracy and answer relevance.
4. **Human evaluation**: Have a legal professional rate answers for accuracy, completeness, and appropriate use of legal language.

---

## 11. Prompt Engineering

### Q: Walk through your legal QA system prompt.

**Answer:**
```
You are a legal document assistant. Answer questions based ONLY on the
provided context from legal/policy documents. Follow these rules strictly:

1. If the answer is found in the context, provide it with specific references
   to the source documents (e.g., "According to [Source 1: contract.pdf, Page 3]...").
2. If the context does not contain enough information to answer, say:
   "I could not find sufficient information in the uploaded documents..."
3. Do NOT make up or infer legal conclusions beyond what the documents state.
4. Use precise legal language when the source documents use it.
5. If multiple sources address the question, synthesize them and cite each.
```

**Design principles:**
- **Rule 1 (Citation format)**: Tells the model exactly how to reference sources, matching the `[Source N: file, Page X]` markers injected in the context.
- **Rule 2 (Refusal)**: Explicitly provides the refusal phrase. LLMs are less likely to hallucinate when given a specific alternative to "I don't know."
- **Rule 3 (No inference)**: Critical for legal — the model should report what documents say, not draw legal conclusions. "The contract states X" is safe; "Therefore, Party A is liable" is dangerous.
- **Rule 4 (Legal language)**: Prevents the model from "simplifying" legal terms, which could change their meaning (e.g., "indemnify" ≠ "compensate").
- **Rule 5 (Multi-source synthesis)**: Encourages comprehensive answers that cross-reference multiple documents.

### Q: How do you handle context formatting for the LLM?

**Answer:** Each retrieved chunk is wrapped with a source attribution marker:

```
[Source 1: contract.pdf, Page 3]
<chunk text here>

---

[Source 2: policy.pdf, Page 7]
<chunk text here>
```

This format serves two purposes:
1. Gives the LLM explicit source markers to reference in its response
2. Separates chunks with `---` dividers so the LLM recognizes boundary between different sources

---

## 12. Streamlit & UI Design

### Q: How does the chat interface maintain conversation history?

**Answer:** Streamlit reruns the entire script on every interaction. We use `st.session_state` to persist data across reruns:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

Each message is stored as a dict with `role`, `content`, `citations`, and `route`. On every rerun, the history is replayed to rebuild the chat display.

### Q: How do file uploads work in your pipeline?

**Answer:**
1. `st.file_uploader` returns `UploadedFile` objects (in-memory byte buffers)
2. We write each file to disk (`data/uploads/`) because PyMuPDF and python-docx require file paths
3. `IngestionPipeline.ingest_file()` processes the file: load → chunk → embed → store
4. A progress bar updates per-file
5. Success messages show pages loaded and chunks created

### Q: Why Streamlit over Gradio or Flask?

**Answer:**
- **Streamlit**: Rapid prototyping with pure Python. Built-in chat components (`st.chat_message`, `st.chat_input`), file uploaders, sidebars. Ideal for ML demos and internal tools.
- **Gradio**: Better for model demos (sliders, inputs → outputs). Lacks the layout flexibility for a full application with sidebar document management.
- **Flask/FastAPI**: Requires building HTML/CSS/JS frontend separately. Much more work for a prototype. Better for production APIs.

---

## 13. System Design & Scalability

### Q: How would you scale this system for production?

**Answer:**

| Component | Current (Local) | Production Scale |
|-----------|-----------------|-----------------|
| **LLM** | Ollama (single instance) | vLLM cluster with GPU auto-scaling, or API (Claude/GPT-4) |
| **Embeddings** | In-process sentence-transformers | Dedicated embedding service (GPU), or API-based embeddings |
| **Vector DB** | ChromaDB PersistentClient (local disk) | Pinecone, Weaviate, or Qdrant (managed cluster) |
| **Document Storage** | Local filesystem | S3/GCS with CDN |
| **UI** | Streamlit (single user) | React/Next.js frontend + FastAPI backend |
| **Caching** | None | Redis for embedding cache + query result cache |
| **Auth** | None | OAuth2/JWT with per-user document isolation |

### Q: How would you handle multi-tenancy?

**Answer:**
- **Collection per tenant**: Each user/organization gets a separate ChromaDB collection (or Pinecone namespace). Queries only search the tenant's collection.
- **Metadata filtering**: Single collection with a `tenant_id` metadata field. Filter on retrieval. Simpler but less isolated.
- **Document-level access control**: Each chunk stores `allowed_users` metadata. Pre-filter at retrieval time.

### Q: What if the document corpus grows to millions of documents?

**Answer:**
1. **Hybrid search**: Combine dense vector search (semantic) with sparse search (BM25/keyword). LangChain supports `EnsembleRetriever` for this.
2. **Hierarchical retrieval**: First retrieve relevant documents (document-level embeddings), then retrieve relevant chunks within those documents.
3. **Re-ranking**: Use a cross-encoder model (e.g., `ms-marco-MiniLM-L-12-v2`) to re-rank retrieved chunks for higher precision.
4. **Index sharding**: Distribute the vector index across multiple machines using Qdrant or Milvus.

### Q: How would you add conversation memory / multi-turn support?

**Answer:** Currently each query is independent. For multi-turn:
1. **Contextualize the question**: Before retrieval, use the LLM to rewrite the user's question incorporating chat history. E.g., "What about section 5?" → "What does section 5 of the employment agreement say about non-compete?"
2. **Implementation**: LangChain's `create_history_aware_retriever` does exactly this — it takes chat history and reformulates the retrieval query.
3. **Session memory**: Store conversation history in `st.session_state` (already done) and pass the last N messages as context.

---

## 14. Common Interview Questions & Answers

### General RAG Questions

**Q: What is the difference between parametric and non-parametric knowledge in LLMs?**
A: Parametric knowledge is encoded in the model's weights during training (fixed, can be outdated, may hallucinate). Non-parametric knowledge is retrieved from external sources at inference time (dynamic, verifiable, citation-ready). RAG combines both: the LLM's parametric knowledge for language understanding and reasoning, plus non-parametric retrieval for factual grounding.

**Q: How do you evaluate a RAG system?**
A: Key metrics:
- **Retrieval quality**: Precision@K (what fraction of retrieved chunks are relevant), Recall@K (what fraction of relevant chunks are retrieved), MRR (Mean Reciprocal Rank)
- **Generation quality**: Faithfulness (does the answer only contain information from the context?), Answer Relevance (does it address the question?), Citation Accuracy (are citations correct?)
- **End-to-end**: Answer Correctness (compared to ground truth), F1 score, human evaluation
- **Framework**: RAGAS is a popular framework that automates RAG evaluation using LLM-as-judge.

**Q: What is the "lost in the middle" problem?**
A: Research shows LLMs pay disproportionate attention to information at the beginning and end of their context window, while under-attending to information in the middle. Mitigation strategies: (1) place the most relevant chunk first, (2) keep total context short, (3) use models specifically trained to handle long contexts, (4) use re-ranking to ensure the most relevant information is positioned at the start.

### Embedding Questions

**Q: Can you use the same model for embedding and generation?**
A: Technically yes (some models like Instructor or E5-Mistral are built for this), but it's generally not recommended. Embedding models are trained with contrastive learning objectives (making similar texts close in vector space), while generation models are trained with next-token prediction. These are different optimization targets. In Query Right, separating them also prevents resource contention on the Ollama server.

**Q: What happens if the embedding model doesn't understand legal terminology?**
A: The `all-MiniLM-L6-v2` model is trained on general English. For highly specialized terminology, you could: (1) use a domain-specific embedding model like `legal-bert-base-uncased`, (2) fine-tune the embedding model on legal text using contrastive learning, or (3) augment queries with term expansions. In practice, MiniLM handles legal text reasonably well because legal English shares enough vocabulary with general English.

### LangChain Questions

**Q: What is a Runnable in LangChain?**
A: A `Runnable` is the base protocol in LCEL. Any object implementing `invoke()`, `ainvoke()`, `batch()`, and `stream()` is a Runnable. This includes prompt templates, LLMs, output parsers, retrievers, and custom functions. Runnables compose with `|` (sequence) and `RunnableParallel` (parallel execution).

**Q: How does LangChain handle streaming?**
A: When you call `chain.stream(input)`, each component in the chain yields partial outputs as they become available. For LLMs, this means tokens stream one at a time. `StrOutputParser` passes through each token. This allows showing the response to the user as it's generated, rather than waiting for the full response.

---

## 15. Behavioral / Project Discussion Questions

**Q: What was the most challenging part of this project?**
A: "The most technically challenging aspect was tuning the chunking strategy for legal documents. Legal text has unique structural properties — long sentences with nested clauses, cross-references between sections, and defined terms that span paragraphs. My first attempt with 500-character chunks produced retrieval results that were often mid-sentence fragments, leading to poor answers. I iterated on chunk size (landing on 1000 chars), overlap (200 chars), and the separator hierarchy (adding semicolons specifically for legal lists). The improvement in answer quality was significant — going from generic responses to precise, citation-rich answers."

**Q: If you had more time, what would you improve?**
A: "Three things: (1) **Hybrid search** — combining vector similarity with BM25 keyword search for better retrieval on legal terms of art that have precise meanings. (2) **Re-ranking** — adding a cross-encoder re-ranker after initial retrieval to improve precision. (3) **Evaluation pipeline** — building an automated evaluation suite with RAGAS to measure faithfulness, answer relevance, and citation accuracy as I iterate on the prompt and retrieval parameters."

**Q: How did you ensure the system doesn't hallucinate?**
A: "Multiple layers: (1) The system prompt explicitly instructs the model to refuse rather than guess when context is insufficient. (2) RAG itself grounds answers in retrieved text, reducing the model's reliance on parametric knowledge. (3) Source citations let users verify every claim — the UI shows the exact text chunk used to generate each part of the answer. (4) Low temperature (0.3) reduces randomness in generation. (5) The prompt prohibits inferring legal conclusions beyond what documents state."

**Q: Walk me through a request from user input to final response.**
A: "When a user types 'What are the termination conditions?':
1. The query router sends this to Gemma 3 at temperature=0, which classifies it as LOCAL (it's about document content).
2. The query is embedded using all-MiniLM-L6-v2 into a 384-dim vector.
3. ChromaDB performs MMR search, returning the 5 most relevant-yet-diverse chunks, each with metadata (source file, page number).
4. The chunks are formatted with [Source N] markers and assembled into the prompt alongside the system instructions.
5. Gemma 3 generates an answer, naturally referencing the source markers because the prompt and context structure guide it.
6. Citation metadata is extracted from the retrieved documents for UI display.
7. The answer appears in the chat with an expandable Sources section showing document name, page number, and a text preview for each source."

**Q: Why did you choose to build this project?**
A: "I wanted to solve a real problem I observed — legal professionals spend significant time manually searching through policy documents and contracts to find specific clauses or verify information. A RAG system dramatically reduces this search time while the citation system ensures every answer can be verified against the source material. I chose the legal domain specifically because it demands high accuracy and citation, which stress-tests the RAG pipeline in ways that simpler use cases don't."

---

## Quick Reference: Key Metrics & Numbers

| Metric | Value |
|--------|-------|
| LLM parameters | 4 billion (Gemma 3 4B) |
| Embedding dimensions | 384 (all-MiniLM-L6-v2) |
| Chunk size / overlap | 1000 / 200 characters |
| Retrieval top-K | 5 (MMR) |
| LLM context window | 8192 tokens |
| Generation temperature | 0.3 (factual) |
| Classification temperature | 0.0 (deterministic) |
| LoRA rank / alpha | 16 / 32 |
| LoRA trainable params | ~20M (out of 4B) |
| Quantization | 4-bit NF4 (QLoRA training) |
| Supported formats | PDF, TXT, DOCX |
| Vector DB | ChromaDB with HNSW indexing |
| Web search results | 3 (Tavily) |
