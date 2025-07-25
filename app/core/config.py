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
    
    # Gmail API
    GMAIL_CLIENT_ID: str = ""
    GMAIL_CLIENT_SECRET: str = ""
    GMAIL_REDIRECT_URI: str = "http://localhost:8000/gmail/callback"
    
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
    
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str
    SESSION_SECRET: str

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET')

def get_google_oauth_config():
    return {
        'client_id': GOOGLE_CLIENT_ID,
        'client_secret': GOOGLE_CLIENT_SECRET,
        'redirect_uri': os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:5000/oauth2callback')
    } 