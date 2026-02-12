import os
from typing import List

from ingestion.loader import load_document
from ingestion.chunker import chunk_documents
from core.vectorstore import get_vectorstore
import config


class IngestionPipeline:
    """Orchestrates: file -> load -> chunk -> embed -> store in ChromaDB."""

    def __init__(self):
        self.vectorstore = get_vectorstore()

    def ingest_file(self, file_path: str) -> dict:
        """Ingest a single file into the vector store."""
        documents = load_document(file_path)
        chunks = chunk_documents(documents)

        ids = [
            f"{os.path.basename(file_path)}_{chunk.metadata.get('page', 0)}_{chunk.metadata['chunk_index']}"
            for chunk in chunks
        ]

        self.vectorstore.add_documents(documents=chunks, ids=ids)

        return {
            "file": os.path.basename(file_path),
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
        }

    def ingest_files(self, file_paths: List[str]) -> List[dict]:
        """Ingest multiple files, returning per-file summaries."""
        results = []
        for path in file_paths:
            result = self.ingest_file(path)
            results.append(result)
        return results

    def get_document_count(self) -> int:
        """Returns the total number of chunks in the vector store."""
        collection = self.vectorstore._collection
        return collection.count()

    def list_ingested_sources(self) -> List[str]:
        """Returns a list of unique source document names."""
        collection = self.vectorstore._collection
        results = collection.get(include=["metadatas"])
        sources = set()
        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                if meta and "source" in meta:
                    sources.add(meta["source"])
        return sorted(sources)

    def delete_source(self, source_name: str):
        """Delete all chunks belonging to a specific source document."""
        collection = self.vectorstore._collection
        collection.delete(where={"source": source_name})
