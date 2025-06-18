import os
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import HTTPException
from app.core.config import get_settings
import json

class AIProcessor:
    """Service for processing PDFs with AI models."""
    
    def __init__(self):
        self.settings = get_settings()
        self.provider = self.settings.AI_PROVIDER.lower()  # 'openai' or 'local'
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = True) -> str:
        """Extract text content from a PDF file using OCR if needed."""
        try:
            import PyPDF2
            
            print(f"Attempting to extract text from: {pdf_path}")
            
            # First try to extract text normally
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                
                print(f"Extracted {len(text)} characters via PyPDF2")
                
                # If we got meaningful text, return it
                if text.strip() and len(text.strip()) > 50:
                    return text.strip()
                
                # If no text or very little text, use OCR
                if use_ocr:
                    print("No text found via PyPDF2, attempting OCR...")
                    return self.extract_text_with_ocr(pdf_path)
                else:
                    return text.strip()
                    
        except Exception as e:
            print(f"PyPDF2 extraction failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            if use_ocr:
                # Fallback to OCR if normal extraction fails
                print("Falling back to OCR...")
                return self.extract_text_with_ocr(pdf_path)
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to extract text from PDF: {str(e)}"
                )
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """Extract text from PDF using OCR (Tesseract)."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            from PIL import Image
            
            print(f"Starting OCR process for: {pdf_path}")
            
            # Convert PDF to images
            print("Converting PDF to images...")
            images = convert_from_path(pdf_path, dpi=300)
            print(f"Converted {len(images)} pages to images")
            
            extracted_text = ""
            
            # Process each page
            for i, image in enumerate(images):
                print(f"Processing page {i+1} with OCR...")
                # Use Tesseract to extract text from the image
                page_text = pytesseract.image_to_string(image, lang='eng')
                extracted_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                print(f"Page {i+1} OCR completed, extracted {len(page_text)} characters")
            
            print(f"OCR completed, total extracted text length: {len(extracted_text)}")
            return extracted_text.strip()
            
        except ImportError as e:
            print(f"Import error in OCR: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="OCR dependencies not installed. Please install tesseract-ocr and pdf2image."
            )
        except Exception as e:
            print(f"OCR extraction failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"OCR extraction failed: {str(e)}"
            )
    
    def process_with_openai(self, text: str) -> Dict[str, Any]:
        """Process text using OpenAI API."""
        try:
            import openai
            
            print(f"Processing text with OpenAI, text length: {len(text)}")
            
            openai.api_key = self.settings.OPENAI_API_KEY
            
            prompt = f"""
            Analyze the following text extracted from a handwritten note or document and provide:
            1. A concise summary (2-3 sentences)
            2. A list of actionable tasks or action items
            3. Key topics or themes mentioned
            4. Any important dates or deadlines
            
            Text to analyze:
            {text}
            
            Please respond in JSON format:
            {{
                "summary": "brief summary here",
                "tasks": ["task 1", "task 2", "task 3"],
                "topics": ["topic 1", "topic 2"],
                "dates": ["date 1", "date 2"],
                "confidence": 0.95
            }}
            """
            
            print("Sending request to OpenAI...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes handwritten notes and documents to extract key information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            print("OpenAI response received")
            
            # Parse the JSON response
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
                print("Successfully parsed OpenAI JSON response")
                return result
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {str(e)}")
                # Fallback if JSON parsing fails
                return {
                    "summary": content,
                    "tasks": [],
                    "topics": [],
                    "dates": [],
                    "confidence": 0.8
                }
                
        except Exception as e:
            print(f"OpenAI processing failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Handle specific OpenAI errors
            if "quota" in str(e).lower() or "billing" in str(e).lower():
                raise HTTPException(
                    status_code=402,  # Payment Required
                    detail=f"OpenAI quota exceeded or billing issue: {str(e)}. Please check your OpenAI account."
                )
            elif "rate limit" in str(e).lower():
                raise HTTPException(
                    status_code=429,  # Too Many Requests
                    detail=f"OpenAI rate limit exceeded: {str(e)}. Please try again later."
                )
            elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                raise HTTPException(
                    status_code=401,  # Unauthorized
                    detail=f"OpenAI authentication failed: {str(e)}. Please check your API key."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"OpenAI processing failed: {str(e)}"
                )
    
    def process_with_local_model(self, text: str) -> Dict[str, Any]:
        """Process text using a local open-source model."""
        try:
            print(f"Processing text with local model, text length: {len(text)}")
            
            # For now, we'll use a simple rule-based approach
            # In the future, you can integrate with models like:
            # - Ollama (llama2, mistral, etc.)
            # - Local transformers models
            # - Hugging Face models
            
            # Simple keyword-based analysis
            text_lower = text.lower()
            
            # Extract potential tasks (lines with action words)
            action_words = ['todo', 'task', 'action', 'need to', 'must', 'should', 'will', 'going to']
            lines = text.split('\n')
            tasks = []
            for line in lines:
                if any(word in line.lower() for word in action_words):
                    tasks.append(line.strip())
            
            # Extract potential dates
            import re
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b'
            dates = re.findall(date_pattern, text)
            
            # Simple summary (first few sentences)
            sentences = text.split('.')
            summary = '. '.join(sentences[:3]) + '.'
            
            result = {
                "summary": summary,
                "tasks": tasks[:5],  # Limit to 5 tasks
                "topics": [],  # Would need more sophisticated analysis
                "dates": dates,
                "confidence": 0.6,  # Lower confidence for rule-based approach
                "method": "rule_based"
            }
            
            print(f"Local processing completed, found {len(tasks)} tasks, {len(dates)} dates")
            return result
            
        except Exception as e:
            print(f"Local processing failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Local processing failed: {str(e)}"
            )
    
    def process_pdf(self, pdf_path: str, use_ocr: bool = True) -> Dict[str, Any]:
        """Process a PDF file and return analysis results."""
        try:
            print(f"Starting PDF processing for: {pdf_path}")
            
            # Extract text from PDF (with OCR if needed)
            text = self.extract_text_from_pdf(pdf_path, use_ocr)
            
            if not text.strip():
                print("No text extracted from PDF")
                return {
                    "summary": "No text content found in PDF",
                    "tasks": [],
                    "topics": [],
                    "dates": [],
                    "confidence": 0.0,
                    "error": "Empty PDF or OCR failed to extract text"
                }
            
            print(f"Text extraction completed, processing with {self.provider}")
            
            # Process with selected AI provider
            if self.provider == "openai":
                return self.process_with_openai(text)
            else:
                return self.process_with_local_model(text)
                
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            print(f"PDF processing failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"PDF processing failed: {str(e)}"
            ) 