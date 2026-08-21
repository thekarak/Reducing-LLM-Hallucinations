import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Directories
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
QUESTIONS_FILE = DATA_DIR / "questions.csv"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_FILE = RESULTS_DIR / "results.csv"
SUMMARY_FILE = RESULTS_DIR / "summary.json"
MANUAL_REVIEW_FILE = RESULTS_DIR / "manual_review.csv"
VECTOR_STORE_DIR = BASE_DIR / "vector_db"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Embeddings & Vector Store Config
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 550
CHUNK_OVERLAP = 90
TOP_K_DEFAULT = 3
TOP_K_EXTENDED = 5

# Retrieval Quality Config
# Chunks whose dense cosine similarity falls below this are discarded.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
# "hybrid" = BM25 (keyword) + dense (semantic) fusion; "dense" = semantic only.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
BM25_CANDIDATES = 9  # keyword candidates considered before rank fusion

# LLM Providers Configuration
# Supported: 'opencode_zen', 'groq', 'gemini', 'openai', 'local_mock'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local_mock").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

OPENCODE_ZEN_API_KEY = os.getenv("OPENCODE_ZEN_API_KEY", "")
OPENCODE_ZEN_BASE_URL = os.getenv("OPENCODE_ZEN_BASE_URL", "https://api.opencodezen.com/v1")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
