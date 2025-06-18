import os
import json
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import pickle

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def setup_gmail_credentials():
    """Set up Gmail API credentials."""
    credentials_path = Path("credentials.json")
    token_path = Path("token.pickle")
    
    if not credentials_path.exists():
        print("\nError: credentials.json not found!")
        print("\nTo set up Gmail API credentials:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select an existing one")
        print("3. Enable the Gmail API for your project")
        print("4. Go to 'Credentials' and create an OAuth 2.0 Client ID")
        print("5. Choose 'Desktop app' as the application type")
        print("6. Download the credentials and save as 'credentials.json' in the project root")
        return False
    
    try:
        credentials = None
        if token_path.exists():
            with open(token_path, 'rb') as token:
                credentials = pickle.load(token)
        
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path),
                    SCOPES
                )
                credentials = flow.run_local_server(port=0)
            
            with open(token_path, 'wb') as token:
                pickle.dump(credentials, token)
        
        print("\nGmail API credentials set up successfully!")
        print("You can now use the application to access your Gmail account.")
        return True
    
    except Exception as e:
        print(f"\nError setting up Gmail credentials: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have enabled the Gmail API in your Google Cloud Console")
        print("2. Verify that your OAuth consent screen is properly configured")
        print("3. Check that your application is using the correct credentials")
        print("4. If testing, make sure you've added your email as a test user")
        return False

if __name__ == "__main__":
    setup_gmail_credentials() 