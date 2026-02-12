from typing import Any, Dict, List

from langchain_core.documents import Document

from core.vectorstore import get_retriever


def retrieve_with_sources(query: str, k: int = 5) -> List[Document]:
    """Retrieve relevant chunks with full metadata for citation."""
    retriever = get_retriever(k=k)
    return retriever.invoke(query)


def format_context(docs: List[Document]) -> str:
    """Format retrieved documents into a context string for the LLM prompt."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


def extract_citations(docs: List[Document]) -> List[Dict[str, Any]]:
    """Extract citation metadata for display in the UI."""
    citations = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        key = f"{source}_p{page}"
        if key not in seen:
            citations.append({
                "source": source,
                "page": page,
                "file_type": doc.metadata.get("file_type", "unknown"),
                "preview": doc.page_content[:150] + "...",
            })
            seen.add(key)
    return citations
