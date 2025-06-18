from fastapi import APIRouter, HTTPException, Response
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

router = APIRouter(prefix="/gmail", tags=["gmail"])
gmail_service = GmailService()
pdf_service = PDFService()

@router.get("/auth-status")
async def check_auth_status():
    """Check Gmail authentication status."""
    try:
        is_authenticated = gmail_service.authenticate()
        return {"status": "authenticated" if is_authenticated else "not_authenticated"}
    except HTTPException as e:
        return {"status": "not_authenticated", "error": e.detail}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fetch-pdfs")
async def fetch_pdfs(max_results: int = 10) -> List[Dict]:
    """Fetch PDF attachments from Gmail."""
    try:
        attachments = gmail_service.get_pdf_attachments(max_results)
        return attachments
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download-and-store/{message_id}/{attachment_id}")
async def download_and_store_pdf(message_id: str, attachment_id: str):
    """Download and store a PDF attachment locally."""
    try:
        result = gmail_service.download_and_store_pdf(message_id, attachment_id)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-with-ai/{message_id}/{attachment_id}")
async def process_pdf_with_ai(
    message_id: str,
    attachment_id: str,
    use_ocr: bool = True,
    background_tasks: BackgroundTasks = None
):
    """Process a downloaded PDF with AI analysis."""
    try:
        print(f"Processing PDF with AI - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        # Get the file path
        file_path = gmail_service.get_attachment_path(message_id, attachment_id)
        
        # If file not found, download it first
        if not file_path or not os.path.exists(file_path):
            print(f"PDF not found locally, downloading first...")
            download_result = gmail_service.download_and_store_pdf(message_id, attachment_id)
            file_path = download_result['stored_path']
            print(f"PDF downloaded and stored at: {file_path}")
        
        print(f"Found PDF file at: {file_path}")
        
        # Initialize AI processor
        ai_processor = AIProcessor()
        print(f"AI processor initialized with provider: {ai_processor.provider}")
        
        # Process the PDF
        print("Starting PDF processing...")
        result = ai_processor.process_pdf(str(file_path), use_ocr=use_ocr)
        print(f"PDF processing completed, result keys: {list(result.keys())}")
        
        # Store the result
        if background_tasks:
            background_tasks.add_task(
                gmail_service.store_ai_result,
                message_id,
                attachment_id,
                result
            )
        
        return {
            "message": "PDF processed successfully",
            "result": result,
            "file_path": str(file_path)
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in process-with-ai: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF with AI: {str(e)}"
        )

@router.get("/stored-pdfs")
async def list_stored_pdfs():
    """List all stored PDF files."""
    try:
        pdfs = pdf_service.list_stored_pdfs()
        return {"pdfs": pdfs, "count": len(pdfs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/stored-pdfs/{filename}")
async def delete_stored_pdf(filename: str):
    """Delete a stored PDF file."""
    try:
        success = pdf_service.delete_pdf(filename)
        if success:
            return {"message": f"PDF {filename} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="PDF not found")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{message_id}/{attachment_id}")
async def download_pdf(message_id: str, attachment_id: str):
    """Download a specific PDF attachment."""
    try:
        attachment = gmail_service.download_attachment(message_id, attachment_id)
        
        # Decode base64 data
        pdf_data = base64.urlsafe_b64decode(attachment['data'])
        
        # Create a streaming response
        return StreamingResponse(
            io.BytesIO(pdf_data),
            media_type=attachment['content_type'],
            headers={
                'Content-Disposition': f'attachment; filename="document.pdf"'
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync-pdfs")
async def sync_pdfs():
    """Sync all relevant Gmail PDFs to local storage."""
    try:
        attachments = gmail_service.get_pdf_attachments(50)
        synced = []
        for att in attachments:
            # Try to find local file by message_id and attachment_id
            local_path = gmail_service.get_attachment_path(att['message_id'], att['attachment_id'])
            if not local_path or not os.path.exists(local_path):
                # Download and store if not present
                result = gmail_service.download_and_store_pdf(att['message_id'], att['attachment_id'])
                synced.append({"message_id": att['message_id'], "attachment_id": att['attachment_id'], "status": "downloaded", "path": result['stored_path']})
            else:
                synced.append({"message_id": att['message_id'], "attachment_id": att['attachment_id'], "status": "already_present", "path": local_path})
        return {"synced": synced, "count": len(synced)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync PDFs: {str(e)}")

@router.get("/local/list-pdfs")
async def local_list_pdfs():
    """List all locally stored PDFs and their metadata."""
    try:
        import glob, json
        from pathlib import Path
        pdf_dir = Path("storage/pdfs")
        pdfs = []
        for pdf_file in pdf_dir.glob("*.pdf"):
            meta_file = pdf_file.with_suffix('.json')
            meta = {}
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
            pdfs.append({"pdf": str(pdf_file), "metadata": meta})
        return {"pdfs": pdfs, "count": len(pdfs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list local PDFs: {str(e)}")

@router.post("/local/process-with-ai/{message_id}/{attachment_id}")
async def local_process_with_ai(message_id: str, attachment_id: str, use_ocr: bool = True, background_tasks: BackgroundTasks = None):
    """Process a local PDF by Gmail IDs with OCR and OpenAI."""
    try:
        print(f"Local processing - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        file_path = gmail_service.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        ai_processor = AIProcessor()
        print(f"AI processor initialized with provider: {ai_processor.provider}")
        
        print("Starting PDF processing...")
        result = ai_processor.process_pdf(str(file_path), use_ocr=use_ocr)
        print(f"PDF processing completed successfully")
        
        if background_tasks:
            background_tasks.add_task(
                gmail_service.store_ai_result,
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