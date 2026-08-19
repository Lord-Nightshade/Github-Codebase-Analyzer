import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Model Definitions
MODEL_NAME = "gemini-3.6-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Local ChromaDB Storage Paths
PERSIST_DIR = "./chroma_db_data"
GUIDELINES_COLLECTION_NAME = "enterprise_architecture_guidelines"

# Whitelisted File Extensions for Ingestion
ALLOWED_EXTENSIONS = {
    # Backend & Systems
    ".java", ".kt", ".py", ".ts", ".js", ".go", ".cs", ".cpp", ".c", ".h", ".hpp", ".rs",
    
    # Database
    ".sql",
    
    # Web & UI
    ".html", ".css", ".jsx", ".tsx",
    
    # Configuration & Build Manifests
    ".xml", ".yaml", ".yml", ".json", ".properties", ".gradle", "Dockerfile"
}

# Ingestion & Chunking Parameters
MAX_FILE_SIZE_KB = 500  # Ignore minified or generated binary/large files
CHUNK_SIZE = 1500       # Character size per code chunk
CHUNK_OVERLAP = 200     # Overlap characters between chunks