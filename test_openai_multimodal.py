#!/usr/bin/env python3
"""
Test script for OpenAI multimodal functionality.
This script tests the new process_with_openai_multimodal method.
"""

import os
import sys
import json
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.ai_processor import AIProcessor
from core.config import get_settings

def test_openai_multimodal():
    """Test the OpenAI multimodal processing functionality."""
    
    # Get settings
    settings = get_settings()
    
    # Check if OpenAI API key is configured
    if not settings.OPENAI_API_KEY:
        print("❌ OpenAI API key not configured. Please set OPENAI_API_KEY in your environment.")
        return False
    
    # Initialize AI processor
    ai_processor = AIProcessor()
    
    # Set provider to openai_multimodal
    ai_processor.provider = "openai_multimodal"
    
    # Check if we have a test PDF
    test_pdf_path = "storage/pdfs"
    if not os.path.exists(test_pdf_path):
        print("❌ No PDFs found in storage/pdfs. Please sync some PDFs first.")
        return False
    
    # Find the first PDF file
    pdf_files = list(Path(test_pdf_path).glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in storage/pdfs. Please sync some PDFs first.")
        return False
    
    test_pdf = str(pdf_files[0])
    print(f"📄 Testing with PDF: {test_pdf}")
    
    try:
        # Test the OpenAI multimodal processing
        print("🔄 Testing OpenAI multimodal processing...")
        result = ai_processor.process_with_openai_multimodal(test_pdf)
        
        # Check if we got a valid result
        if result and isinstance(result, dict):
            print("✅ OpenAI multimodal processing completed successfully!")
            print(f"📊 Summary: {result.get('summary', 'No summary')}")
            print(f"📝 Tasks found: {len(result.get('tasks', []))}")
            print(f"🏷️ Topics found: {len(result.get('topics', []))}")
            print(f"📅 Dates found: {len(result.get('dates', []))}")
            print(f"🎯 Method: {result.get('method', 'Unknown')}")
            print(f"📄 Pages processed: {result.get('pages_processed', 0)}")
            print(f"📏 Total text length: {result.get('total_length', 0)}")
            
            return True
        else:
            print("❌ Invalid result returned from OpenAI multimodal processing")
            return False
            
    except Exception as e:
        print(f"❌ Error during OpenAI multimodal processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test the configuration for OpenAI multimodal."""
    
    settings = get_settings()
    
    print("🔧 Configuration Test:")
    print(f"   AI Provider: {settings.AI_PROVIDER}")
    print(f"   OpenAI API Key: {'✅ Set' if settings.OPENAI_API_KEY else '❌ Not set'}")
    print(f"   OpenAI Model: {settings.OPENAI_MODEL}")
    
    return bool(settings.OPENAI_API_KEY)

if __name__ == "__main__":
    print("🧪 Testing OpenAI Multimodal Functionality")
    print("=" * 50)
    
    # Test configuration first
    if not test_configuration():
        print("\n❌ Configuration test failed. Please check your settings.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    
    # Test the actual functionality
    if test_openai_multimodal():
        print("\n🎉 All tests passed! OpenAI multimodal functionality is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1) 