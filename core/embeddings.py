from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

import config


class LocalEmbeddings(Embeddings):
    """Wraps sentence-transformers to conform to LangChain Embeddings interface."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()
