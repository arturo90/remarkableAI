import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
from fastapi import HTTPException
from app.core.config import get_settings

class PDFService:
    """Service for handling PDF storage and processing."""
    
    def __init__(self):
        self.settings = get_settings()
        self.storage_path = Path(self.settings.PDFS_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def store_pdf(self, pdf_data: bytes, filename: str, metadata: Dict[str, Any]) -> str:
        """Store a PDF file locally and return the file path."""
        try:
            # Create a unique filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(pdf_data).hexdigest()[:8]
            safe_filename = f"{timestamp}_{file_hash}_{filename}"
            
            file_path = self.storage_path / safe_filename
            
            # Write the PDF data to file
            with open(file_path, 'wb') as f:
                f.write(pdf_data)
            
            # Store metadata in a companion JSON file
            metadata_path = file_path.with_suffix('.json')
            metadata['stored_at'] = timestamp
            metadata['file_path'] = str(file_path)
            metadata['original_filename'] = filename
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return str(file_path)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store PDF: {str(e)}"
            )
    
    def get_pdf_path(self, filename: str) -> Optional[Path]:
        """Get the path of a stored PDF file."""
        file_path = self.storage_path / filename
        return file_path if file_path.exists() else None
    
    def list_stored_pdfs(self) -> list:
        """List all stored PDF files with their metadata."""
        pdfs = []
        for file_path in self.storage_path.glob("*.pdf"):
            metadata_path = file_path.with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                pdfs.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'metadata': metadata
                })
        return sorted(pdfs, key=lambda x: x['metadata']['stored_at'], reverse=True)
    
    def delete_pdf(self, filename: str) -> bool:
        """Delete a stored PDF file and its metadata."""
        try:
            file_path = self.storage_path / filename
            metadata_path = file_path.with_suffix('.json')
            
            if file_path.exists():
                file_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            return True
        except Exception:
            return False 