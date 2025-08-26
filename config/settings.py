import os
from pathlib import Path
from typing import List, Optional
import time


# Handle the pydantic compatibility issue
try:
    from pydantic.v1 import BaseSettings, Field
except ImportError:
    try:
        from pydantic import BaseSettings, Field
    except ImportError:
        from pydantic_settings import BaseSettings
        from pydantic import Field

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Domain-Specific Semantic RAG System"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    # Path settings
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = BASE_DIR / "cache"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    # Document processing settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Vector store settings
    VECTOR_STORE_TYPE: str = "chroma"  # Options: chroma, pinecone, weaviate
    COLLECTION_NAME: str = "rag_documents"
    SIMILARITY_THRESHOLD: float = 0.7
    TOP_K_RETRIEVAL: int = 10
    
    # Query processing settings
    MAX_QUERY_LENGTH: int = 256
    QUERY_EXPANSION_ENABLED: bool = True
    
    # Reranker settings
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    TOP_K_RERANK: int = 3
    
    # LLM settings
    # LLM_PROVIDER: str = "openai"  # Options: openai, anthropic, huggingface, bedrock
    # LLM_MODEL: str = "gpt-3.5"
    # MAX_TOKENS: int = 1000
    # TEMPERATURE: float = 0.1

    LLM_PROVIDER:str = "gemini"
    LLM_MODEL: str = "gemini-pro"
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.1

 



    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    CORS_ORIGINS: List[str] = ["*"]
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./rag_system.db"
    
    # Cache settings
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600  # 1 hour
    
    # Monitoring settings
    ENABLE_MONITORING: bool = True
    PROMETHEUS_PORT: int = 8001
    
    # Authentication
    API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)

settings = Settings()