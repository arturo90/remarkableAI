#!/usr/bin/env python3
"""
Check Gmail API setup and provide guidance for missing credentials.
"""

import os
import json
from pathlib import Path

def check_gmail_setup():
    """Check if Gmail API credentials are properly configured."""
    print("🔍 Checking Gmail API setup...")
    
    # Check if the application is running and has tokens
    try:
        from app.main import _tokens
        if _tokens:
            print("✅ Gmail authentication tokens found")
            return True
        else:
            print("❌ No Gmail authentication tokens found")
            print("\n📋 To authenticate with Gmail:")
            print("1. Start the application: python -m uvicorn app.main:app --reload")
            print("2. Go to http://localhost:8000/auth")
            print("3. Sign in with your Google account")
            print("4. Grant the necessary permissions")
            return False
    except ImportError:
        print("❌ Could not import application modules")
        print("Make sure you're running this from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Error checking authentication: {e}")
        return False

def test_gmail_connection():
    """Test the Gmail API connection."""
    print("\n🧪 Testing Gmail API connection...")
    
    try:
        from app.main import _tokens
        if not _tokens:
            print("❌ No authentication tokens available")
            return False
        
        from app.services.gmail_service import GmailService
        gmail_service = GmailService()
        gmail_service.set_tokens(_tokens)
        
        # Try to authenticate
        if gmail_service.authenticate():
            print("✅ Gmail authentication successful")
            
            # Try to fetch a small number of PDFs
            try:
                attachments = gmail_service.get_pdf_attachments(max_results=1)
                print(f"✅ Found {len(attachments)} PDF attachments")
                return True
            except Exception as e:
                print(f"⚠️  Could not fetch PDFs: {e}")
                return False
        else:
            print("❌ Gmail authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gmail connection: {e}")
        return False

if __name__ == "__main__":
    print("🚀 RemarkableAI Gmail Setup Checker")
    print("=" * 50)
    
    # Check setup
    setup_ok = check_gmail_setup()
    
    if setup_ok:
        # Test connection
        connection_ok = test_gmail_connection()
        
        if connection_ok:
            print("\n🎉 Gmail setup is complete and working!")
            print("You can now use the 'Sync from Gmail' button in the web interface.")
        else:
            print("\n⚠️  Setup appears complete but connection test failed.")
            print("Try signing in again at http://localhost:8000/auth")
    else:
        print("\n❌ Gmail setup is incomplete.")
        print("Please follow the instructions above to authenticate with Gmail.")
