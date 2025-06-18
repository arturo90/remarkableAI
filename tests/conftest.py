import pytest
from pathlib import Path
import os
import sys

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["ENVIRONMENT"] = "test"
    yield
    # Cleanup
    os.environ.pop("TESTING", None)
    os.environ.pop("ENVIRONMENT", None)

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Return the test data directory."""
    return project_root / "tests" / "data" 