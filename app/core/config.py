from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings."""
    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Google OAuth Configuration
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str = "http://127.0.0.1:8000/auth/callback"
    
    # AI Processing
    AI_PROVIDER: str = "local"  # Options: "local", "openai", "multimodal", "openai_multimodal"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    # Database
    DATABASE_URL: str = "sqlite:///./remarkable_ai.db"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Email
    EMAIL_FROM: str = "noreply@remarkableai.com"
    EMAIL_TO: str = "user@example.com"
    
    # Storage
    STORAGE_PATH: str = "storage"
    PDFS_PATH: str = "storage/pdfs"
    RESULTS_PATH: str = "storage/results"
    
    # Processing Configuration
    OCR_ENABLED: bool = True
    AUTO_SYNC: bool = False
    AUTO_PROCESS: bool = False
    MAX_RESULTS: int = 50
    RETENTION_DAYS: int = 30
    
    SESSION_SECRET: str

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

def get_google_oauth_config():
    settings = get_settings()
    return {
        'client_id': settings.GOOGLE_CLIENT_ID,
        'client_secret': settings.GOOGLE_CLIENT_SECRET,
        'redirect_uri': settings.GOOGLE_REDIRECT_URI
    } 