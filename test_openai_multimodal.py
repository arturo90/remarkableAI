#!/usr/bin/env python3
"""
Test script for OpenAI multimodal functionality.
This script tests the OpenAI multimodal PDF processing capabilities.
"""

import os
import sys
import json
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.services.ai_processor import AIProcessor
from app.core.config import get_settings

def test_openai_multimodal():
    """Test OpenAI multimodal processing functionality."""
    
    # Check if OpenAI API key is set
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        return False
    
    print("✅ OpenAI API key found")
    
    # Initialize AI processor
    try:
        ai_processor = AIProcessor()
        print("✅ AI processor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize AI processor: {e}")
        return False
    
    # Test that the provider is set correctly
    if ai_processor.provider not in ["openai", "openai_multimodal"]:
        print(f"❌ Invalid AI provider: {ai_processor.provider}")
        print("   Expected: 'openai' or 'openai_multimodal'")
        return False
    
    print(f"✅ AI provider set to: {ai_processor.provider}")
    
    # Test OpenAI text processing
    try:
        test_text = "Meeting notes: Discuss project timeline. Tasks: 1. Review requirements 2. Set up development environment. Topics: Project planning, Development. Dates: Next week, March 15th."
        
        print("🧪 Testing OpenAI text processing...")
        result = ai_processor.process_with_openai(test_text)
        
        if result and isinstance(result, dict):
            print("✅ OpenAI text processing successful")
            print(f"   Summary: {result.get('summary', 'N/A')[:100]}...")
            print(f"   Tasks found: {len(result.get('tasks', []))}")
            print(f"   Topics found: {len(result.get('topics', []))}")
            print(f"   Dates found: {len(result.get('dates', []))}")
        else:
            print("❌ OpenAI text processing failed - invalid result format")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI text processing failed: {e}")
        return False
    
    # Test OpenAI multimodal processing (requires a PDF file)
    pdf_files = list(Path("storage/pdfs").glob("*.pdf"))
    if pdf_files:
        test_pdf = str(pdf_files[0])
        print(f"🧪 Testing OpenAI multimodal processing with: {test_pdf}")
        
        try:
            # Temporarily set provider to openai_multimodal
            original_provider = ai_processor.provider
            ai_processor.provider = "openai_multimodal"
            
            result = ai_processor.process_with_openai_multimodal(test_pdf)
            
            if result and isinstance(result, dict):
                print("✅ OpenAI multimodal processing successful")
                print(f"   Summary: {result.get('summary', 'N/A')[:100]}...")
                print(f"   Tasks found: {len(result.get('tasks', []))}")
                print(f"   Topics found: {len(result.get('topics', []))}")
                print(f"   Dates found: {len(result.get('dates', []))}")
                print(f"   Method: {result.get('method', 'N/A')}")
                print(f"   Pages processed: {result.get('pages_processed', 'N/A')}")
            else:
                print("❌ OpenAI multimodal processing failed - invalid result format")
                return False
                
        except Exception as e:
            print(f"❌ OpenAI multimodal processing failed: {e}")
            return False
        finally:
            # Restore original provider
            ai_processor.provider = original_provider
    else:
        print("⚠️  No PDF files found in storage/pdfs/ - skipping multimodal test")
        print("   To test multimodal processing, add a PDF file to storage/pdfs/")
    
    print("\n🎉 All tests completed successfully!")
    print("\n📋 Summary:")
    print("   ✅ OpenAI API key configured")
    print("   ✅ AI processor initialized")
    print("   ✅ OpenAI text processing working")
    if pdf_files:
        print("   ✅ OpenAI multimodal processing working")
    else:
        print("   ⚠️  OpenAI multimodal processing not tested (no PDF files)")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing OpenAI Multimodal Functionality")
    print("=" * 50)
    
    success = test_openai_multimodal()
    
    if success:
        print("\n✅ All tests passed! OpenAI multimodal functionality is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the configuration and try again.")
        sys.exit(1) 