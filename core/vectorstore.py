import chromadb
from langchain_chroma import Chroma

from core.embeddings import LocalEmbeddings
import config


def get_vectorstore() -> Chroma:
    """Returns a persistent Chroma vector store instance."""
    embeddings = LocalEmbeddings()
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return Chroma(
        client=client,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
    )


def get_retriever(search_type: str = config.RETRIEVAL_SEARCH_TYPE, k: int = config.RETRIEVAL_TOP_K):
    """Returns a LangChain retriever from the vector store."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
