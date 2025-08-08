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
    use_ocr: bool = True,
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
    """Download a PDF attachment from Gmail."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        attachment = gmail_service_with_tokens.download_attachment(message_id, attachment_id)
        
        if not attachment or 'data' not in attachment:
            raise HTTPException(status_code=404, detail="Attachment not found")
        
        # Decode the attachment data
        pdf_data = base64.urlsafe_b64decode(attachment['data'])
        
        # Return the PDF as a file response
        from fastapi.responses import Response
        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={attachment['metadata']['subject']}.pdf"
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync-pdfs")
async def sync_pdfs_from_gmail(max_results: int = 10, email_filter: str = None):
    """Sync PDF attachments from Gmail to local storage."""
    import traceback
    try:
        print("[DEBUG] Starting Gmail sync...")
        
        # Get the global tokens from main.py
        from app.main import _tokens
        if not _tokens:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated with Gmail. Please sign in first."
            )
        
        # Create Gmail service instance and set tokens
        gmail_service = GmailService()
        gmail_service.set_tokens(_tokens)
        
        # Check if Gmail service can authenticate
        try:
            attachments = gmail_service.get_pdf_attachments(max_results, email_filter)
        except HTTPException as e:
            if "No authentication tokens" in str(e.detail):
                raise HTTPException(
                    status_code=401,
                    detail="Not authenticated with Gmail. Please sign in first."
                )
            elif "Authentication tokens are invalid" in str(e.detail):
                raise HTTPException(
                    status_code=401,
                    detail="Gmail authentication expired. Please sign in again."
                )
            else:
                raise e
        
        print(f"[DEBUG] Attachments found: {len(attachments)}")
        for i, att in enumerate(attachments):
            print(f"[DEBUG] Attachment {i+1}: {att}")
        synced_count = 0
        
        for attachment in attachments:
            try:
                print(f"[DEBUG] Attempting to download attachment: {attachment}")
                # Download and store the PDF
                pdf_info = gmail_service.download_attachment(
                    attachment['message_id'], 
                    attachment['attachment_id']
                )
                print(f"[DEBUG] Downloaded attachment: {attachment['filename']}")
                
                if pdf_info and 'data' in pdf_info:
                    print(f"[DEBUG] pdf_info type: {type(pdf_info)}, keys: {list(pdf_info.keys())}")
                    print(f"[DEBUG] pdf_info['data'] type: {type(pdf_info['data'])}")
                    pdf_data = base64.urlsafe_b64decode(pdf_info['data'])
                    print(f"[DEBUG] pdf_data type: {type(pdf_data)}, length: {len(pdf_data)}")
                    # Store the PDF locally
                    filename = f"{attachment['subject']}.pdf"
                    file_path = pdf_service.store_pdf(pdf_data, filename, attachment)
                    synced_count += 1
                    print(f"[DEBUG] Synced PDF: {filename} at {file_path}")
                else:
                    print(f"[DEBUG] No data returned for attachment: {attachment}")
                    print(f"[DEBUG] pdf_info: {pdf_info}")
                    
            except Exception as e:
                print(f"[ERROR] Failed to sync PDF {attachment.get('subject', 'unknown')}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "message": f"Successfully synced {synced_count} PDFs from Gmail",
            "synced_count": synced_count,
            "total_found": len(attachments)
        }
        
    except Exception as e:
        print(f"[ERROR] Exception in sync_pdfs_from_gmail: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync PDFs from Gmail: {str(e)}"
        )

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
        
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        print("Starting PDF processing...")
        result = ai_processor.process_pdf(str(file_path), use_ocr=use_ocr)
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

@router.get("/local/extract-ocr/{message_id}/{attachment_id}")
async def extract_ocr_text(message_id: str, attachment_id: str):
    """Extract OCR text from a local PDF without AI processing."""
    try:
        print(f"Extracting OCR text - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        # Extract text using OCR
        print("Starting OCR text extraction...")
        ocr_text = ai_processor.extract_text_with_ocr(str(file_path))
        print(f"OCR text extraction completed, extracted {len(ocr_text)} characters")
        
        # Get metadata
        metadata = {}
        meta_file = Path(file_path).with_suffix('.json')
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
        
        return {
            "message": "OCR text extracted successfully",
            "ocr_text": ocr_text,
            "text_length": len(ocr_text),
            "file_path": str(file_path),
            "metadata": metadata
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in extract-ocr: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract OCR text: {str(e)}"
        )

@router.get("/local/test-ocr-engines/{message_id}/{attachment_id}")
async def test_ocr_engines(message_id: str, attachment_id: str):
    """Test different OCR engines on a local PDF."""
    try:
        print(f"Testing OCR engines - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        # Test different OCR engines
        results = {}
        
        # Test Tesseract
        try:
            print("Testing Tesseract OCR...")
            tesseract_text = ai_processor.extract_text_with_tesseract(str(file_path))
            results['tesseract'] = {
                'text': tesseract_text,
                'length': len(tesseract_text),
                'status': 'success'
            }
        except Exception as e:
            results['tesseract'] = {
                'text': '',
                'length': 0,
                'status': 'error',
                'error': str(e)
            }
        
        # Test EasyOCR
        try:
            print("Testing EasyOCR...")
            easyocr_text = ai_processor.extract_text_with_easyocr(str(file_path))
            results['easyocr'] = {
                'text': easyocr_text,
                'length': len(easyocr_text),
                'status': 'success'
            }
        except Exception as e:
            results['easyocr'] = {
                'text': '',
                'length': 0,
                'status': 'error',
                'error': str(e)
            }
        
        # Test PaddleOCR
        try:
            print("Testing PaddleOCR...")
            paddleocr_text = ai_processor.extract_text_with_paddleocr(str(file_path))
            results['paddleocr'] = {
                'text': paddleocr_text,
                'length': len(paddleocr_text),
                'status': 'success'
            }
        except Exception as e:
            results['paddleocr'] = {
                'text': '',
                'length': 0,
                'status': 'error',
                'error': str(e)
            }
        
        return {
            "message": "OCR engine testing completed",
            "file_path": str(file_path),
            "results": results
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in OCR engine testing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test OCR engines: {str(e)}"
        )

@router.post("/local/process-with-multimodal/{message_id}/{attachment_id}")
async def process_pdf_with_multimodal(message_id: str, attachment_id: str):
    """Process a local PDF using multimodal LLM (LLaVA) for better handwritten text recognition."""
    try:
        print(f"Multimodal processing - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        # Initialize AI processor with multimodal provider
        # Temporarily set provider to multimodal for this request
        original_provider = ai_processor.provider
        ai_processor.provider = "multimodal"
        
        print("Starting multimodal PDF processing...")
        result = ai_processor.process_with_multimodal_llm(str(file_path))
        print(f"Multimodal PDF processing completed successfully")
        
        # Restore original provider
        ai_processor.provider = original_provider
        
        # Store the result
        gmail_service_with_tokens.store_ai_result(message_id, attachment_id, result)
        
        return {
            "message": "Multimodal PDF processing completed successfully", 
            "result": result, 
            "file_path": str(file_path),
            "method": "multimodal_llava"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in multimodal processing: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF with multimodal LLM: {str(e)}"
        )

@router.post("/local/process-with-openai-multimodal/{message_id}/{attachment_id}")
async def process_pdf_with_openai_multimodal(message_id: str, attachment_id: str):
    """Process a local PDF using OpenAI's multimodal capabilities (GPT-4 Vision) for better handwritten text recognition."""
    try:
        print(f"OpenAI Multimodal processing - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        # Initialize AI processor with OpenAI multimodal provider
        # Temporarily set provider to openai_multimodal for this request
        original_provider = ai_processor.provider
        ai_processor.provider = "openai_multimodal"
        
        print("Starting OpenAI multimodal PDF processing...")
        result = ai_processor.process_with_openai_multimodal(str(file_path))
        print(f"OpenAI multimodal PDF processing completed successfully")
        
        # Restore original provider
        ai_processor.provider = original_provider
        
        # Store the result
        gmail_service_with_tokens.store_ai_result(message_id, attachment_id, result)
        
        return {
            "message": "OpenAI multimodal PDF processing completed successfully", 
            "result": result, 
            "file_path": str(file_path),
            "method": "openai_vision"
        }
        
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

# Results API endpoints
@router.get("/api/results/list")
async def list_results():
    """List all AI processing results."""
    try:
        results_dir = Path("storage/results")
        if not results_dir.exists():
            return {"results": [], "count": 0}
        
        results = []
        for result_file in results_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                    results.append(result_data)
            except Exception as e:
                print(f"Error reading result file {result_file}: {str(e)}")
                continue
        
        # Sort by processed_at date (newest first)
        results.sort(key=lambda x: x.get('processed_at', ''), reverse=True)
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")

@router.get("/api/results/{message_id}/{attachment_id}")
async def get_result(message_id: str, attachment_id: str):
    """Get a specific AI processing result."""
    try:
        results_dir = Path("storage/results")
        if not results_dir.exists():
            raise HTTPException(status_code=404, detail="No results found")
        
        # Find the result file
        attachment_hash = gmail_service._get_attachment_hash(attachment_id)
        result_filename = f"{message_id}_{attachment_hash}_ai_result.json"
        result_path = results_dir / result_filename
        
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Result not found")
        
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        return result_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get result: {str(e)}")

@router.get("/local/view-pdf/{message_id}/{attachment_id}")
async def view_pdf_local(message_id: str, attachment_id: str):
    """View a locally stored PDF file."""
    try:
        gmail_service_with_tokens = get_gmail_service_with_tokens()
        file_path = gmail_service_with_tokens.get_attachment_path(message_id, attachment_id)
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="PDF not found locally. Please sync first.")
        
        # Return the PDF file
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={os.path.basename(file_path)}"
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in view-pdf: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to view PDF: {str(e)}"
        )

@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a PDF file manually and process it with AI."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )
        
        # Read the uploaded file
        pdf_data = await file.read()
        
        if len(pdf_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Create metadata for the uploaded file
        metadata = {
            'message_id': f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'attachment_id': f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'subject': f"Manual Upload: {file.filename}",
            'date': str(int(datetime.now().timestamp() * 1000)),
            'from': 'Manual Upload',
            'filename': file.filename,
            'upload_type': 'manual'
        }
        
        # Store the PDF locally
        file_path = pdf_service.store_pdf(pdf_data, file.filename, metadata)
        
        # Initialize AI processor
        ai_processor = AIProcessor()
        
        # Process the PDF with AI
        print(f"Processing uploaded PDF: {file_path}")
        result = ai_processor.process_pdf(file_path)
        
        # Store the AI result
        if background_tasks:
            background_tasks.add_task(
                gmail_service.store_ai_result,
                metadata['message_id'],
                metadata['attachment_id'],
                result
            )
        
        return {
            "message": "PDF uploaded and processed successfully",
            "result": result,
            "file_path": file_path,
            "metadata": metadata,
            "uploaded_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in upload_pdf: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload and process PDF: {str(e)}"
        )

@router.post("/upload-pdf-multimodal")
async def upload_pdf_multimodal(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a PDF file manually and process it with multimodal LLM."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )
        
        # Read the uploaded file
        pdf_data = await file.read()
        
        if len(pdf_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Create metadata for the uploaded file
        metadata = {
            'message_id': f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'attachment_id': f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'subject': f"Manual Upload: {file.filename}",
            'date': str(int(datetime.now().timestamp() * 1000)),
            'from': 'Manual Upload',
            'filename': file.filename,
            'upload_type': 'manual'
        }
        
        # Store the PDF locally
        file_path = pdf_service.store_pdf(pdf_data, file.filename, metadata)
        
        # Initialize AI processor with multimodal provider
        ai_processor = AIProcessor()
        # Temporarily set provider to multimodal for this request
        original_provider = ai_processor.provider
        ai_processor.provider = "multimodal"
        
        # Process the PDF with multimodal LLM
        print(f"Processing uploaded PDF with multimodal LLM: {file_path}")
        result = ai_processor.process_with_multimodal_llm(file_path)
        
        # Restore original provider
        ai_processor.provider = original_provider
        
        # Store the AI result
        if background_tasks:
            background_tasks.add_task(
                gmail_service.store_ai_result,
                metadata['message_id'],
                metadata['attachment_id'],
                result
            )
        
        return {
            "message": "PDF uploaded and processed with multimodal LLM successfully",
            "result": result,
            "file_path": file_path,
            "metadata": metadata,
            "uploaded_at": datetime.now().isoformat(),
            "method": "multimodal_llava"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in upload_pdf_multimodal: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload and process PDF with multimodal LLM: {str(e)}"
        )

@router.post("/upload-pdf-openai-multimodal")
async def upload_pdf_openai_multimodal(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a PDF file manually and process it with OpenAI's multimodal capabilities (GPT-4 Vision)."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )
        
        # Read the uploaded file
        pdf_data = await file.read()
        
        if len(pdf_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Create metadata for the uploaded file
        metadata = {
            'message_id': f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'attachment_id': f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            'subject': f"Manual Upload: {file.filename}",
            'date': str(int(datetime.now().timestamp() * 1000)),
            'from': 'Manual Upload',
            'filename': file.filename,
            'upload_type': 'manual'
        }
        
        # Store the PDF locally
        file_path = pdf_service.store_pdf(pdf_data, file.filename, metadata)
        
        # Initialize AI processor with OpenAI multimodal provider
        ai_processor = AIProcessor()
        # Temporarily set provider to openai_multimodal for this request
        original_provider = ai_processor.provider
        ai_processor.provider = "openai_multimodal"
        
        # Process the PDF with OpenAI multimodal
        print(f"Processing uploaded PDF with OpenAI multimodal: {file_path}")
        result = ai_processor.process_with_openai_multimodal(file_path)
        
        # Restore original provider
        ai_processor.provider = original_provider
        
        # Store the AI result
        if background_tasks:
            background_tasks.add_task(
                gmail_service.store_ai_result,
                metadata['message_id'],
                metadata['attachment_id'],
                result
            )
        
        return {
            "message": "PDF uploaded and processed with OpenAI multimodal successfully",
            "result": result,
            "file_path": file_path,
            "metadata": metadata,
            "uploaded_at": datetime.now().isoformat(),
            "method": "openai_vision"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in upload_pdf_openai_multimodal: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload and process PDF with OpenAI multimodal: {str(e)}"
        ) 