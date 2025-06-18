from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
import pickle
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.core.config import get_settings
from app.services.pdf_service import PDFService
from app.services.ai_processor import AIProcessor
from fastapi import HTTPException
import hashlib

class GmailService:
    """Service for interacting with Gmail API."""
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    
    def __init__(self):
        self.settings = get_settings()
        self.credentials = None
        self.service = None
        self.token_path = Path("token.pickle")
        self.credentials_path = Path("credentials.json")
        self.pdf_service = PDFService()
        self.ai_processor = AIProcessor()
    
    def authenticate(self) -> bool:
        """Authenticate with Gmail API."""
        if not self.credentials_path.exists():
            raise HTTPException(
                status_code=400,
                detail="Gmail credentials not found. Please set up credentials.json"
            )
        
        try:
            if self.token_path.exists():
                with open(self.token_path, 'rb') as token:
                    self.credentials = pickle.load(token)
            
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    self.credentials.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path),
                        self.SCOPES
                    )
                    self.credentials = flow.run_local_server(port=0)
                
                with open(self.token_path, 'wb') as token:
                    pickle.dump(self.credentials, token)
            
            self.service = build('gmail', 'v1', credentials=self.credentials)
            return True
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to authenticate with Gmail: {str(e)}"
            )
    
    def get_pdf_attachments(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """Fetch PDF attachments from Gmail, only from my@remarkable.com."""
        try:
            if not self.service:
                self.authenticate()
            
            # Search for emails with PDF attachments from a specific sender
            query = 'from:my@remarkable.com has:attachment filename:pdf'
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            pdf_attachments = []
            
            for message in messages:
                msg = self.service.users().messages().get(
                    userId='me',
                    id=message['id']
                ).execute()
                
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
            
            return pdf_attachments
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch PDF attachments: {str(e)}"
            )
    
    def download_and_store_pdf(self, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Download a PDF attachment and store it locally."""
        try:
            if not self.service:
                self.authenticate()
            
            # Download the attachment
            attachment = self.service.users().messages().attachments().get(
                userId='me',
                messageId=message_id,
                id=attachment_id
            ).execute()
            
            if not attachment or 'data' not in attachment:
                raise HTTPException(
                    status_code=404,
                    detail="Attachment not found"
                )
            
            # Decode the attachment data
            pdf_data = base64.urlsafe_b64decode(attachment['data'])
            
            # Get message details for metadata
            msg = self.service.users().messages().get(
                userId='me',
                id=message_id
            ).execute()
            
            metadata = {
                'message_id': message_id,
                'attachment_id': attachment_id,
                'subject': next(
                    (header['value'] for header in msg['payload']['headers'] 
                     if header['name'].lower() == 'subject'),
                    'No Subject'
                ),
                'date': msg['internalDate'],
                'from': next(
                    (header['value'] for header in msg['payload']['headers'] 
                     if header['name'].lower() == 'from'),
                    'Unknown'
                )
            }
            
            # Store the PDF locally
            stored_path = self.pdf_service.store_pdf(
                pdf_data, 
                metadata['subject'] + '.pdf',
                metadata
            )
            
            return {
                'stored_path': stored_path,
                'metadata': metadata,
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
    
    def download_attachment(self, message_id: str, attachment_id: str) -> Dict[str, Any]:
        """Download a specific attachment."""
        try:
            if not self.service:
                self.authenticate()
            
            attachment = self.service.users().messages().attachments().get(
                userId='me',
                messageId=message_id,
                id=attachment_id
            ).execute()
            
            if not attachment or 'data' not in attachment:
                raise HTTPException(
                    status_code=404,
                    detail="Attachment not found"
                )
            
            return {
                'data': attachment['data'],
                'content_type': 'application/pdf'
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download attachment: {str(e)}"
            )
    
    def get_attachment_path(self, message_id: str, attachment_id: str) -> Optional[str]:
        """Get the stored path of a downloaded PDF attachment."""
        try:
            # Check if the PDF was already downloaded and stored
            storage_dir = Path("storage/pdfs")
            if not storage_dir.exists():
                return None
            
            # Look for files with matching metadata
            for pdf_file in storage_dir.glob("*.pdf"):
                metadata_file = pdf_file.with_suffix('.json')
                if metadata_file.exists():
                    try:
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        if (metadata.get('message_id') == message_id and 
                            metadata.get('attachment_id') == attachment_id):
                            return str(pdf_file)
                    except Exception:
                        continue
            
            return None
        except Exception as e:
            print(f"Error getting attachment path: {str(e)}")
            return None
    
    def store_ai_result(self, message_id: str, attachment_id: str, result: Dict[str, Any]) -> None:
        """Store AI analysis result for an attachment."""
        try:
            import json
            from datetime import datetime
            
            # Create results directory if it doesn't exist
            results_dir = Path("storage/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a shorter filename using hash of attachment_id
            attachment_hash = hashlib.md5(attachment_id.encode()).hexdigest()[:8]
            result_filename = f"{message_id}_{attachment_hash}_ai_result.json"
            result_path = results_dir / result_filename
            
            # Add metadata to result
            result_with_metadata = {
                "message_id": message_id,
                "attachment_id": attachment_id,
                "attachment_hash": attachment_hash,
                "processed_at": datetime.now().isoformat(),
                "result": result
            }
            
            # Save result
            with open(result_path, 'w') as f:
                json.dump(result_with_metadata, f, indent=2)
                
            print(f"AI result stored at: {result_path}")
            
        except Exception as e:
            print(f"Failed to store AI result: {str(e)}")
    
    def _get_attachment_hash(self, attachment_id: str) -> str:
        """Get a short hash of the attachment ID for filename generation."""
        return hashlib.md5(attachment_id.encode()).hexdigest()[:8] 