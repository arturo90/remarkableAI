import pytest
from app.core.config import Settings, get_settings
import os

def test_settings_default_values():
    """Test that settings have correct default values."""
    settings = Settings()
    
    assert settings.ENVIRONMENT == "development"
    assert settings.DEBUG is True
    assert settings.LOG_LEVEL == "INFO"
    assert settings.ALGORITHM == "HS256"
    assert settings.ACCESS_TOKEN_EXPIRE_MINUTES == 30
    assert settings.PDFS_PATH == "storage/pdfs"
    assert settings.STORAGE_PATH == "storage"
    assert settings.RESULTS_PATH == "storage/results"

def test_get_settings_caching():
    """Test that get_settings returns cached instance."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2

def test_settings_env_loading():
    """Test that settings can be loaded from environment variables."""
    os.environ["GMAIL_CLIENT_ID"] = "env_test"
    os.environ["GMAIL_CLIENT_SECRET"] = "env_test"
    os.environ["GMAIL_REDIRECT_URI"] = "env_test"
    os.environ["OPENAI_API_KEY"] = "env_test"
    os.environ["DATABASE_URL"] = "env_test"
    os.environ["SECRET_KEY"] = "env_test"
    os.environ["EMAIL_FROM"] = "env_test"
    os.environ["EMAIL_TO"] = "env_test"
    
    settings = Settings()
    assert settings.GMAIL_CLIENT_ID == "env_test"
    
    # Cleanup
    for key in [
        "GMAIL_CLIENT_ID",
        "GMAIL_CLIENT_SECRET",
        "GMAIL_REDIRECT_URI",
        "OPENAI_API_KEY",
        "DATABASE_URL",
        "SECRET_KEY",
        "EMAIL_FROM",
        "EMAIL_TO"
    ]:
        os.environ.pop(key, None) 