import pytest
from pathlib import Path

def test_project_structure(project_root):
    """Test that the project has the correct directory structure."""
    # Required directories
    required_dirs = [
        "app",
        "app/api",
        "app/core",
        "app/db",
        "app/services",
        "app/utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "docs",
        "scripts",
        "config"
    ]
    
    for dir_path in required_dirs:
        assert (project_root / dir_path).is_dir(), f"Directory {dir_path} does not exist"

def test_required_files(project_root):
    """Test that all required files exist."""
    required_files = [
        "requirements.txt",
        "README.md",
        "tests/conftest.py",
        ".env.example"
    ]
    
    for file_path in required_files:
        assert (project_root / file_path).is_file(), f"File {file_path} does not exist"

def test_python_files(project_root):
    """Test that Python files follow naming conventions."""
    python_files = list(project_root.rglob("*.py"))
    
    for file_path in python_files:
        # Check that Python files use snake_case
        assert "_" in file_path.stem or file_path.stem.islower(), \
            f"Python file {file_path} should use snake_case naming"
        
        # Check that test files start with test_
        if "tests" in str(file_path):
            assert file_path.stem.startswith("test_"), \
                f"Test file {file_path} should start with 'test_'" 