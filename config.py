import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")
UPLOAD_DIR = str(DATA_DIR / "uploads")

# Ollama / LLM
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
LLM_TEMPERATURE = 0.3
LLM_NUM_CTX = 8192

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ChromaDB
CHROMA_COLLECTION_NAME = "legal_documents"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
RETRIEVAL_TOP_K = 5
RETRIEVAL_SEARCH_TYPE = "mmr"

# Web Search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = 3

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}
