import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolves to the 'backend' folder where .env is stored
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    NVIDIA_API_KEY: str
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION: str = "multimodal_rag"
    
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_KEY: str
    
    GROQ_API_KEY: str = ""
    
    CORS_ORIGINS: str = "*"
    MAX_RETRIEVAL_RETRIES: int = 3
    TOP_K_RETRIEVE: int = 20
    TOP_K_RERANK: int = 5
    RELEVANCE_THRESHOLD: float = 0.5
    
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
