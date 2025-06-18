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
from pathlib import Path
import json
from datetime import datetime

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
async def sync_pdfs_from_gmail(max_results: int = 10):
    """Sync PDF attachments from Gmail to local storage."""
    import traceback
    try:
        print("[DEBUG] Starting Gmail sync...")
        attachments = gmail_service.get_pdf_attachments(max_results)
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
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                continue
        
        print(f"[DEBUG] Sync complete. Synced {synced_count} of {len(attachments)} attachments.")
        return {
            "message": f"Successfully synced {synced_count} PDFs from Gmail",
            "synced_count": synced_count,
            "total_attachments": len(attachments)
        }
        
    except Exception as e:
        print(f"[ERROR] Exception in sync_pdfs_from_gmail: {str(e)}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

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

@router.get("/local/extract-ocr/{message_id}/{attachment_id}")
async def extract_ocr_text(message_id: str, attachment_id: str):
    """Extract OCR text from a local PDF without AI processing."""
    try:
        print(f"Extracting OCR text - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        file_path = gmail_service.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        # Initialize AI processor just for OCR
        ai_processor = AIProcessor()
        
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
            "metadata": metadata,
            "extracted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in OCR extraction: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract OCR text: {str(e)}"
        )

@router.get("/local/test-ocr-engines/{message_id}/{attachment_id}")
async def test_ocr_engines(message_id: str, attachment_id: str):
    """Test different OCR engines on the same PDF to compare performance."""
    try:
        print(f"Testing OCR engines - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        file_path = gmail_service.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        # Initialize AI processor
        ai_processor = AIProcessor()
        
        # Test both OCR engines
        results = {}
        
        # Test EasyOCR
        try:
            print("Testing EasyOCR...")
            easyocr_text = ai_processor._extract_with_easyocr(str(file_path))
            results["easyocr"] = {
                "text": easyocr_text,
                "length": len(easyocr_text),
                "status": "success"
            }
        except Exception as e:
            results["easyocr"] = {
                "text": "",
                "length": 0,
                "status": "failed",
                "error": str(e)
            }
        
        # Test Tesseract
        try:
            print("Testing Tesseract...")
            tesseract_text = ai_processor._extract_with_tesseract_from_path(str(file_path))
            results["tesseract"] = {
                "text": tesseract_text,
                "length": len(tesseract_text),
                "status": "success"
            }
        except Exception as e:
            results["tesseract"] = {
                "text": "",
                "length": 0,
                "status": "failed",
                "error": str(e)
            }
        
        return {
            "message": "OCR engine comparison completed",
            "file_path": str(file_path),
            "results": results,
            "comparison": {
                "easyocr_length": results["easyocr"]["length"],
                "tesseract_length": results["tesseract"]["length"],
                "recommended": "easyocr" if results["easyocr"]["length"] > results["tesseract"]["length"] else "tesseract"
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in OCR engine test: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test OCR engines: {str(e)}"
        )

@router.post("/local/process-with-multimodal/{message_id}/{attachment_id}")
async def process_pdf_with_multimodal(message_id: str, attachment_id: str):
    """Process a local PDF using multimodal LLM (Ollama + LLaVA) for better handwritten text recognition."""
    try:
        print(f"Multimodal processing - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        file_path = gmail_service.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        # Initialize AI processor with multimodal provider
        ai_processor = AIProcessor()
        # Temporarily set provider to multimodal for this request
        original_provider = ai_processor.provider
        ai_processor.provider = "multimodal"
        
        print("Starting multimodal PDF processing...")
        result = ai_processor.process_with_multimodal_llm(str(file_path))
        print(f"Multimodal PDF processing completed successfully")
        
        # Restore original provider
        ai_processor.provider = original_provider
        
        # Store the result
        gmail_service.store_ai_result(message_id, attachment_id, result)
        
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
    """View a local PDF by Gmail message_id and attachment_id."""
    try:
        print(f"Viewing local PDF - Message ID: {message_id}, Attachment ID: {attachment_id}")
        
        # Find the local file path by looking up the metadata
        file_path = gmail_service.get_attachment_path(message_id, attachment_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Local PDF not found: {file_path}")
            raise HTTPException(status_code=404, detail="Local PDF not found. Please sync first.")
        
        print(f"Found local PDF at: {file_path}")
        
        # Read the PDF file
        with open(file_path, 'rb') as f:
            pdf_data = f.read()
        
        # Get the original filename from metadata
        meta_file = Path(file_path).with_suffix('.json')
        filename = "document.pdf"
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                filename = metadata.get('filename', 'document.pdf')
        
        # Create a streaming response
        return StreamingResponse(
            io.BytesIO(pdf_data),
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'inline; filename="{filename}"'
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"Unexpected error in view_pdf_local: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to view local PDF: {str(e)}"
        ) 