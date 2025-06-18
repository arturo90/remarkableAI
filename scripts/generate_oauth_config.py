import json
import os
from pathlib import Path

def generate_oauth_config():
    """Generate OAuth configuration for Gmail API."""
    config = {
        "web": {
            "client_id": "YOUR_CLIENT_ID",
            "project_id": "YOUR_PROJECT_ID",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "YOUR_CLIENT_SECRET",
            "redirect_uris": ["http://localhost:8000/oauth2callback"],
            "javascript_origins": ["http://localhost:8000"]
        }
    }
    
    print("\nTo set up Gmail API credentials:")
    print("\n1. Go to https://console.cloud.google.com/")
    print("2. Create a new project or select an existing one")
    print("3. Enable the Gmail API for your project")
    print("4. Go to 'APIs & Services' > 'OAuth consent screen'")
    print("   - Set User Type to 'External'")
    print("   - Add your email as a test user")
    print("   - Add the scope: https://www.googleapis.com/auth/gmail.readonly")
    print("5. Go to 'APIs & Services' > 'Credentials'")
    print("   - Click 'Create Credentials' > 'OAuth client ID'")
    print("   - Choose 'Web application' as the application type")
    print("   - Add 'http://localhost:8000' as an authorized JavaScript origin")
    print("   - Add 'http://localhost:8000/oauth2callback' as an authorized redirect URI")
    print("6. Download the credentials and save as 'credentials.json' in the project root")
    print("\nAfter downloading the credentials, run:")
    print("python scripts/setup_gmail.py")

if __name__ == "__main__":
    generate_oauth_config() 