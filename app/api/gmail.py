from fastapi import APIRouter, HTTPException, Response, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from app.services.gmail_service import GmailService
from app.services.pdf_service import PDFService
from app.services.ai_processor import AIProcessor
from typing import List, Dict
import base64
import io
import os
import traceback
from fastapi import BackgroundTasks
from pathlib import Path
import json
from datetime import datetime
from fastapi.responses import FileResponse

router = APIRouter(prefix="/gmail", tags=["gmail"])

# Initialize services
gmail_service = GmailService()
pdf_service = PDFService()
ai_processor = AIProcessor()

def get_gmail_service_with_tokens():
    """Get Gmail service instance with authentication tokens."""
    from app.main import _tokens
    if not _tokens:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated with Gmail. Please sign in first."
        )
    
    service = GmailService()
    service.set_tokens(_tokens)
    return service

@router.get("/auth-status")
async def check_auth_status():
    """Check if Gmail API is authenticated."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        is_authenticated = gmail_service_with_tokens.authenticate()
        return {
            "authenticated": is_authenticated,
            "status": "connected" if is_authenticated else "disconnected"
        }
    except HTTPException as e:
        return {
            "authenticated": False,
            "status": "error",
            "error": str(e.detail)
        }

@router.get("/fetch-pdfs")
async def fetch_pdfs(max_results: int = 10, email_filter: str = None) -> List[Dict]:
    """Fetch PDF attachments from Gmail."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        attachments = gmail_service_with_tokens.get_pdf_attachments(max_results, email_filter)
        return attachments
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download-and-store/{message_id}/{attachment_id}")
async def download_and_store_pdf(message_id: str, attachment_id: str):
    """Download and store a PDF attachment from Gmail."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        result = gmail_service_with_tokens.download_and_store_pdf(message_id, attachment_id)
        return {
            "status": "success",
            "message": "PDF downloaded and stored successfully",
            "data": result
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-with-ai/{message_id}/{attachment_id}")
async def process_pdf_with_ai(
    message_id: str,
    attachment_id: str,
    background_tasks: BackgroundTasks = None
):
    """Download, store, and process a PDF with AI."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        
        # Check if PDF is already stored locally
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        
        if not file_path or not os.path.exists(file_path):
            # Download and store the PDF first
            download_result = gmail_service_with_tokens.download_and_store_pdf(message_id, attachment_id)
            file_path = download_result['stored_path']
        
        # Process with AI
        if background_tasks:
            background_tasks.add_task(
                gmail_service_with_tokens.store_ai_result,
                message_id, attachment_id, ai_processor.process_pdf(file_path)
            )
            return {
                "status": "processing",
                "message": "PDF processing started in background"
            }
        else:
            result = ai_processor.process_pdf(file_path)
            gmail_service_with_tokens.store_ai_result(message_id, attachment_id, result)
            return {
                "status": "completed",
                "message": "PDF processed successfully",
                "result": result
            }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stored-pdfs")
async def list_stored_pdfs():
    """List all stored PDF files."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        pdfs = gmail_service_with_tokens.list_stored_pdfs()
        return {"pdfs": pdfs}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/stored-pdfs/{filename}")
async def delete_stored_pdf(filename: str):
    """Delete a stored PDF file."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        success = gmail_service_with_tokens.delete_stored_pdf(filename)
        if success:
            return {"message": "PDF deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="PDF not found")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{message_id}/{attachment_id}")
async def download_pdf(message_id: str, attachment_id: str):
    """Download a PDF attachment from Gmail."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        
        # Get the file path
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        
        if not file_path or not os.path.exists(file_path):
            # Download and store the PDF first
            download_result = gmail_service_with_tokens.download_and_store_pdf(message_id, attachment_id)
            file_path = download_result['stored_path']
        
        # Return the file
        return FileResponse(
            path=file_path,
            media_type='application/pdf',
            filename=os.path.basename(file_path)
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync-pdfs")
async def sync_pdfs_from_gmail(max_results: int = 10, email_filter: str = None):
    """Sync PDF attachments from Gmail to local storage."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        
        # Get PDF attachments from Gmail
        attachments = gmail_service_with_tokens.get_pdf_attachments(max_results, email_filter)
        
        synced_count = 0
        failed_count = 0
        results = []
        
        for attachment in attachments:
            try:
                message_id = attachment['message_id']
                attachment_id = attachment['attachment_id']
                
                # Check if already stored
                existing_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
                if existing_path and os.path.exists(existing_path):
                    results.append({
                        'message_id': message_id,
                        'attachment_id': attachment_id,
                        'status': 'already_exists',
                        'path': str(existing_path)
                    })
                    continue
                
                # Download and store
                download_result = gmail_service_with_tokens.download_and_store_pdf(message_id, attachment_id)
                
                results.append({
                    'message_id': message_id,
                    'attachment_id': attachment_id,
                    'status': 'synced',
                    'path': download_result['stored_path']
                })
                synced_count += 1
                
            except Exception as e:
                print(f"Failed to sync attachment {attachment.get('attachment_id', 'unknown')}: {str(e)}")
                results.append({
                    'message_id': attachment.get('message_id', 'unknown'),
                    'attachment_id': attachment.get('attachment_id', 'unknown'),
                    'status': 'failed',
                    'error': str(e)
                })
                failed_count += 1
        
        return {
            "message": f"Sync completed. {synced_count} synced, {failed_count} failed",
            "synced_count": synced_count,
            "failed_count": failed_count,
            "total_processed": len(attachments),
            "results": results
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Local PDF processing endpoints
@router.get("/local/list-pdfs")
async def local_list_pdfs():
    """List all locally stored PDFs."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        pdfs = gmail_service_with_tokens.list_stored_pdfs()
        return {"pdfs": pdfs}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/local/process-with-ai/{message_id}/{attachment_id}")
async def local_process_with_ai(message_id: str, attachment_id: str, background_tasks: BackgroundTasks = None):
    """Process a local PDF by Gmail IDs with OpenAI."""
    try:
        print(f"Local processing - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        print("Starting PDF processing...")
        result = ai_processor.process_pdf(str(file_path))
        print(f"PDF processing completed successfully")
        
        if background_tasks:
            background_tasks.add_task(
                gmail_service_with_tokens.store_ai_result,
                message_id,
                attachment_id,
                result
            )
        
        return {"message": "Local PDF processed successfully", "result": result, "file_path": str(file_path)}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in local process-with-ai: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process local PDF with AI: {str(e)}"
        )



@router.post("/local/process-with-openai-multimodal/{message_id}/{attachment_id}")
async def process_pdf_with_openai_multimodal(message_id: str, attachment_id: str):
    """Process a local PDF using OpenAI multimodal (GPT-4 Vision)."""
    try:
        print(f"OpenAI multimodal processing - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        # Temporarily set provider to openai_multimodal
        original_provider = ai_processor.provider
        ai_processor.provider = "openai_multimodal"
        
        try:
            print("Starting OpenAI multimodal PDF processing...")
            result = ai_processor.process_pdf(str(file_path))
            print(f"OpenAI multimodal PDF processing completed successfully")
            
            return {"message": "OpenAI multimodal PDF processing completed", "result": result, "file_path": str(file_path)}
            
        finally:
            # Restore original provider
            ai_processor.provider = original_provider
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in OpenAI multimodal processing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF with OpenAI multimodal: {str(e)}"
        )

# Results management endpoints
@router.get("/api/results/list")
async def list_results():
    """List all AI processing results."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        results = gmail_service_with_tokens.list_ai_results()
        return {"results": results}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/results/{message_id}/{attachment_id}")
async def get_result(message_id: str, attachment_id: str):
    """Get AI processing result for a specific PDF."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        result = gmail_service_with_tokens.get_ai_result(message_id, attachment_id)
        if result:
            return result
        else:
            raise HTTPException(status_code=404, detail="Result not found")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/local/view-pdf/{message_id}/{attachment_id}")
async def view_pdf_local(message_id: str, attachment_id: str):
    """View a locally stored PDF file."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="PDF not found locally. Please sync first.")
        
        # Return the file
        return FileResponse(
            path=file_path,
            media_type='application/pdf',
            filename=os.path.basename(file_path)
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoints
@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a PDF file using OpenAI multimodal for optimal PDF reading."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create storage directory if it doesn't exist
        storage_path = Path("storage/pdfs")
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{file.filename}"
        file_path = storage_path / filename
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"PDF uploaded and saved to: {file_path}")
        
        # Process with OpenAI multimodal AI for optimal PDF reading
        if background_tasks:
            background_tasks.add_task(
                process_uploaded_pdf_openai_multimodal,
                str(file_path),
                file.filename
            )
            return {
                "status": "processing",
                "message": "PDF uploaded and OpenAI multimodal processing started in background",
                "filename": filename,
                "file_path": str(file_path)
            }
        else:
            result = process_uploaded_pdf_openai_multimodal(str(file_path), file.filename)
            return {
                "status": "completed",
                "message": "PDF uploaded and OpenAI multimodal processing completed",
                "filename": filename,
                "file_path": str(file_path),
                "result": result
            }
            
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")



@router.post("/upload-pdf-openai-multimodal")
async def upload_pdf_openai_multimodal(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a PDF file using OpenAI multimodal (GPT-4 Vision)."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Create storage directory if it doesn't exist
        storage_path = Path("storage/pdfs")
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_openai_multimodal_{timestamp}_{file.filename}"
        file_path = storage_path / filename
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"PDF uploaded and saved to: {file_path}")
        
        # Process with OpenAI multimodal AI
        if background_tasks:
            background_tasks.add_task(
                process_uploaded_pdf_openai_multimodal,
                str(file_path),
                file.filename
            )
            return {
                "status": "processing",
                "message": "PDF uploaded and OpenAI multimodal processing started in background",
                "filename": filename,
                "file_path": str(file_path)
            }
        else:
            result = process_uploaded_pdf_openai_multimodal(str(file_path), file.filename)
            return {
                "status": "completed",
                "message": "PDF uploaded and OpenAI multimodal processing completed",
                "filename": filename,
                "file_path": str(file_path),
                "result": result
            }
            
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Helper functions for background processing
def process_uploaded_pdf(file_path: str, original_filename: str):
    """Process an uploaded PDF file using OpenAI multimodal for optimal PDF reading."""
    try:
        # Always use OpenAI multimodal for PDF processing
        result = ai_processor.process_pdf(file_path)
        
        # Store result with a unique identifier
        result_id = f"upload_{Path(file_path).stem}"
        
        # Save result to storage
        results_path = Path("storage/results")
        results_path.mkdir(parents=True, exist_ok=True)
        
        result_file = results_path / f"{result_id}.json"
        with open(result_file, 'w') as f:
            json.dump({
                "result": result,
                "original_filename": original_filename,
                "file_path": file_path,
                "processed_at": datetime.now().isoformat(),
                "processing_method": "openai_multimodal"
            }, f, indent=2)
        
        print(f"Upload processing completed for: {file_path}")
        return result
        
    except Exception as e:
        print(f"Upload processing failed: {str(e)}")
        raise



def process_uploaded_pdf_openai_multimodal(file_path: str, original_filename: str):
    """Process an uploaded PDF file using OpenAI multimodal."""
    try:
        # Use OpenAI multimodal processing (now the default for PDFs)
        result = ai_processor.process_pdf(file_path)
        
        # Store result with a unique identifier
        result_id = f"upload_openai_multimodal_{Path(file_path).stem}"
        
        # Save result to storage
        results_path = Path("storage/results")
        results_path.mkdir(parents=True, exist_ok=True)
        
        result_file = results_path / f"{result_id}.json"
        with open(result_file, 'w') as f:
            json.dump({
                "result": result,
                "original_filename": original_filename,
                "file_path": file_path,
                "processed_at": datetime.now().isoformat(),
                "method": "openai_multimodal"
            }, f, indent=2)
        
        print(f"OpenAI multimodal upload processing completed for: {file_path}")
        return result
        
    except Exception as e:
        print(f"OpenAI multimodal upload processing failed: {str(e)}")
        raise 