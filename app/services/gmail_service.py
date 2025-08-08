from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
import pickle
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.core.config import get_settings, get_google_oauth_config
from app.services.pdf_service import PDFService
from app.services.ai_processor import AIProcessor
from fastapi import HTTPException
import hashlib
import requests
import time

# Constants for token-based authentication
GOOGLE_AUTH_BASE = 'https://accounts.google.com/o/oauth2/auth'
GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token'
GOOGLE_API_BASE = 'https://gmail.googleapis.com/gmail/v1/users/me/'
GOOGLE_CALENDAR_API_BASE = 'https://www.googleapis.com/calendar/v3/'

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar']

class GmailService:
    """Service for interacting with Gmail API using token-based authentication."""
    
    def __init__(self):
        self.settings = get_settings()
        self.pdf_service = PDFService()
        self.ai_processor = AIProcessor()
        self._tokens = None
    
    def set_tokens(self, tokens: Dict[str, Any]):
        """Set the authentication tokens for this service instance."""
        self._tokens = tokens
    
    def get_tokens(self) -> Optional[Dict[str, Any]]:
        """Get the current authentication tokens."""
        return self._tokens
    
    def authenticate(self) -> bool:
        """Check if we have valid tokens for authentication."""
        if not self._tokens:
            raise HTTPException(
                status_code=401,
                detail="No authentication tokens available. Please authenticate first."
            )
        
        # Check if tokens are valid
        access_token, self._tokens = get_access_token(self._tokens)
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail="Authentication tokens are invalid or expired. Please re-authenticate."
            )
        
        return True
    
    def get_pdf_attachments(self, max_results: int = 10, email_filter: str = None) -> List[Dict[str, Any]]:
        """Fetch PDF attachments from Gmail using the specified email filter."""
        try:
            # Ensure we're authenticated
            self.authenticate()
            
            # Build the search query
            if email_filter and email_filter.strip():
                # Use the custom email filter if provided
                query = f'{email_filter.strip()} has:attachment filename:pdf'
            else:
                # More flexible default filter - look for any PDF attachments
                query = 'has:attachment filename:pdf'
            
            print(f"[DEBUG] Using Gmail query: {query}")
            
            # Use the token-based API call
            endpoint = f"messages?q={query}&maxResults={max_results}"
            response, self._tokens = gmail_api_get(endpoint, self._tokens)
            
            if not response or 'messages' not in response:
                print("[DEBUG] No messages found")
                return []
            
            messages = response['messages']
            pdf_attachments = []
            
            for message in messages:
                # Get message details
                msg_endpoint = f"messages/{message['id']}"
                msg_response, self._tokens = gmail_api_get(msg_endpoint, self._tokens)
                
                if not msg_response or 'payload' not in msg_response:
                    continue
                
                msg = msg_response
                
                if 'parts' in msg['payload']:
                    for part in msg['payload']['parts']:
                        if part.get('filename', '').lower().endswith('.pdf'):
                            attachment = {
                                'message_id': message['id'],
                                'attachment_id': part['body']['attachmentId'],
                                'filename': part['filename'],
                                'date': msg['internalDate'],
                                'subject': next(
                                    (header['value'] for header in msg['payload']['headers'] 
                                     if header['name'].lower() == 'subject'),
                                    'No Subject'
                                )
                            }
                            pdf_attachments.append(attachment)
            
            print(f"[DEBUG] Found {len(pdf_attachments)} PDF attachments")
            return pdf_attachments
        except Exception as e:
            print(f"[ERROR] Failed to fetch PDF attachments: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch PDF attachments: {str(e)}"
            )
    
    def download_attachment(self, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Download a PDF attachment using token-based authentication."""
        try:
            # Ensure we're authenticated
            self.authenticate()
            
            # Download the attachment using token-based API
            endpoint = f"messages/{message_id}/attachments/{attachment_id}"
            attachment_response, self._tokens = gmail_api_get(endpoint, self._tokens)
            
            if not attachment_response or 'data' not in attachment_response:
                raise HTTPException(
                    status_code=404,
                    detail="Attachment not found"
                )
            
            # Get message details for metadata
            msg_endpoint = f"messages/{message_id}"
            msg_response, self._tokens = gmail_api_get(msg_endpoint, self._tokens)
            
            if not msg_response:
                raise HTTPException(
                    status_code=404,
                    detail="Message not found"
                )
            
            metadata = {
                'message_id': message_id,
                'attachment_id': attachment_id,
                'subject': next(
                    (header['value'] for header in msg_response['payload']['headers'] 
                     if header['name'].lower() == 'subject'),
                    'No Subject'
                ),
                'date': msg_response['internalDate'],
                'from': next(
                    (header['value'] for header in msg_response['payload']['headers'] 
                     if header['name'].lower() == 'from'),
                    'Unknown'
                )
            }
            
            return {
                'data': attachment_response['data'],
                'metadata': metadata
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download attachment: {str(e)}"
            )
    
    def download_and_store_pdf(self, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Download a PDF attachment and store it locally."""
        try:
            # Download the attachment
            attachment_data = self.download_attachment(message_id, attachment_id)
            
            if not attachment_data or 'data' not in attachment_data:
                raise HTTPException(
                    status_code=404,
                    detail="Attachment not found"
                )
            
            # Decode the attachment data
            pdf_data = base64.urlsafe_b64decode(attachment_data['data'])
            
            # Store the PDF locally
            stored_path = self.pdf_service.store_pdf(
                pdf_data, 
                attachment_data['metadata']['subject'] + '.pdf',
                attachment_data['metadata']
            )
            
            return {
                'stored_path': stored_path,
                'metadata': attachment_data['metadata'],
                'size_bytes': len(pdf_data)
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download and store PDF: {str(e)}"
            )
    
    def process_pdf_with_ai(self, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Download, store, and process a PDF with AI."""
        try:
            # Download and store the PDF
            storage_result = self.download_and_store_pdf(message_id, attachment_id)
            
            # Process with AI
            ai_result = self.ai_processor.process_pdf(storage_result['stored_path'])
            
            return {
                'storage': storage_result,
                'analysis': ai_result,
                'processed_at': storage_result['metadata'].get('stored_at')
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process PDF with AI: {str(e)}"
            )
    
    def get_attachment_path(self, message_id: str, attachment_id: str) -> Optional[str]:
        """Get the local file path of a stored attachment."""
        try:
            # Create a hash-based filename
            file_hash = self._get_attachment_hash(attachment_id)
            storage_path = Path(self.settings.PDFS_PATH)
            
            # Look for files with this hash
            for pdf_file in storage_path.glob("*.pdf"):
                if file_hash in pdf_file.name:
                    return str(pdf_file)
            
            return None
        except Exception:
            return None
    
    def store_ai_result(self, message_id: str, attachment_id: str, result: Dict[str, Any]) -> None:
        """Store AI processing results for an attachment."""
        try:
            # Create results directory
            results_dir = Path("storage/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a unique filename for the result
            result_filename = f"{message_id}_{attachment_id}_result.json"
            result_path = results_dir / result_filename
            
            # Store the result
            import json
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"Failed to store AI result: {e}")
    
    def _get_attachment_hash(self, attachment_id: str) -> str:
        """Generate a hash for the attachment ID."""
        return hashlib.md5(attachment_id.encode()).hexdigest()[:8]

# Token-based authentication functions
def get_auth_url(state=None):
    config = get_google_oauth_config()
    params = {
        'client_id': config['client_id'],
        'redirect_uri': config['redirect_uri'],
        'response_type': 'code',
        'scope': ' '.join(SCOPES),
        'access_type': 'offline',
        'prompt': 'consent',
    }
    if state:
        params['state'] = state
    from urllib.parse import urlencode
    return f"{GOOGLE_AUTH_BASE}?{urlencode(params)}"

def exchange_code_for_tokens(code):
    config = get_google_oauth_config()
    data = {
        'code': code,
        'client_id': config['client_id'],
        'client_secret': config['client_secret'],
        'redirect_uri': config['redirect_uri'],
        'grant_type': 'authorization_code',
    }
    resp = requests.post(GOOGLE_TOKEN_URL, data=data)
    resp.raise_for_status()
    tokens = resp.json()
    return {
        'access_token': tokens['access_token'],
        'refresh_token': tokens.get('refresh_token'),
        'expires_at': int(time.time()) + tokens.get('expires_in', 3600)
    }

def refresh_access_token(tokens):
    config = get_google_oauth_config()
    if not tokens or not tokens.get('refresh_token'):
        return None
    data = {
        'client_id': config['client_id'],
        'client_secret': config['client_secret'],
        'refresh_token': tokens['refresh_token'],
        'grant_type': 'refresh_token',
    }
    resp = requests.post(GOOGLE_TOKEN_URL, data=data)
    resp.raise_for_status()
    new_tokens = resp.json()
    tokens['access_token'] = new_tokens['access_token']
    tokens['expires_at'] = int(time.time()) + new_tokens.get('expires_in', 3600)
    return tokens

def get_access_token(tokens):
    if not tokens:
        return None, tokens
    if tokens['expires_at'] < int(time.time()):
        tokens = refresh_access_token(tokens)
    return tokens['access_token'] if tokens else None, tokens

def gmail_api_get(endpoint, tokens):
    access_token, tokens = get_access_token(tokens)
    if not access_token:
        return None, tokens
    headers = {'Authorization': f'Bearer {access_token}'}
    resp = requests.get(f'{GOOGLE_API_BASE}{endpoint}', headers=headers)
    resp.raise_for_status()
    return resp.json(), tokens

def revoke_tokens(tokens):
    if tokens and tokens.get('access_token'):
        requests.post('https://oauth2.googleapis.com/revoke', params={'token': tokens['access_token']})

def get_calendar_auth_url(state=None):
    config = get_google_oauth_config()
    params = {
        'client_id': config['client_id'],
        'redirect_uri': config['redirect_uri'],
        'response_type': 'code',
        'scope': ' '.join(CALENDAR_SCOPES),
        'access_type': 'offline',
        'prompt': 'consent',
    }
    if state:
        params['state'] = state
    from urllib.parse import urlencode
    return f"{GOOGLE_AUTH_BASE}?{urlencode(params)}"

def calendar_list_calendars(tokens):
    access_token, tokens = get_access_token(tokens)
    if not access_token:
        return None, tokens
    headers = {'Authorization': f'Bearer {access_token}'}
    resp = requests.get(f'{GOOGLE_CALENDAR_API_BASE}users/me/calendarList', headers=headers)
    resp.raise_for_status()
    return resp.json(), tokens

def calendar_create_event(tokens, calendar_id, event):
    access_token, tokens = get_access_token(tokens)
    if not access_token:
        return None, tokens
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    resp = requests.post(f'{GOOGLE_CALENDAR_API_BASE}calendars/{calendar_id}/events', headers=headers, json=event)
    resp.raise_for_status()
    return resp.json(), tokens 