from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Returns a text splitter tuned for legal documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=[
            "\n\n",   # Paragraph breaks
            "\n",     # Line breaks
            ". ",     # Sentence boundaries
            "; ",     # Semicolons (common in legal lists)
            ", ",     # Clause boundaries
            " ",      # Word boundaries (last resort)
        ],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks, preserving and enriching metadata."""
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
