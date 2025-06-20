import os
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import HTTPException
from app.core.config import get_settings
import json

# Fix for PIL.Image.ANTIALIAS compatibility issue with EasyOCR
try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
except ImportError:
    pass

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
        """Extract text from PDF using OCR (EasyOCR with Tesseract fallback)."""
        try:
            from pdf2image import convert_from_path
            from PIL import Image
            
            print(f"Starting OCR process for: {pdf_path}")
            
            # Convert PDF to images
            print("Converting PDF to images...")
            images = convert_from_path(pdf_path, dpi=300)
            print(f"Converted {len(images)} pages to images")
            
            extracted_text = ""
            
            # Try EasyOCR first (better for handwritten text)
            try:
                import easyocr
                print("Using EasyOCR for better handwritten text recognition...")
                reader = easyocr.Reader(['en'])
                
                # Process each page
                for i, image in enumerate(images):
                    print(f"Processing page {i+1} with EasyOCR...")
                    # Convert PIL image to numpy array for EasyOCR
                    import numpy as np
                    img_array = np.array(image)
                    
                    # Extract text with EasyOCR
                    results = reader.readtext(img_array)
                    page_text = ""
                    
                    for (bbox, text, confidence) in results:
                        if confidence > 0.3:  # Filter low confidence results
                            page_text += text + " "
                    
                    extracted_text += f"\n--- Page {i+1} ---\n{page_text.strip()}\n"
                    print(f"Page {i+1} EasyOCR completed, extracted {len(page_text)} characters")
                
                # If EasyOCR didn't extract much text, try Tesseract as fallback
                """ if len(extracted_text.strip()) < 100:
                    print("EasyOCR extracted little text, trying Tesseract fallback...")
                    return self._extract_with_tesseract(images) """
                
                print(f"EasyOCR completed, total extracted text length: {len(extracted_text)}")
                return extracted_text.strip()
                
            except ImportError:
                print("EasyOCR not available, falling back to Tesseract...")
                return self._extract_with_tesseract(images)
            except Exception as e:
                print(f"EasyOCR failed: {str(e)}, falling back to Tesseract...")
                return self._extract_with_tesseract(images)
            
        except ImportError as e:
            print(f"Import error in OCR: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="OCR dependencies not installed. Please install easyocr, pytesseract and pdf2image."
            )
        except Exception as e:
            print(f"OCR extraction failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"OCR extraction failed: {str(e)}"
            )
    
    def _extract_with_tesseract(self, images) -> str:
        """Extract text using Tesseract OCR (fallback method)."""
        try:
            import pytesseract
            
            extracted_text = ""
            
            # Process each page with Tesseract
            for i, image in enumerate(images):
                print(f"Processing page {i+1} with Tesseract...")
                # Use Tesseract to extract text from the image
                #pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR"
                page_text = pytesseract.image_to_string(image, lang='eng')
                extracted_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                print(f"Page {i+1} Tesseract completed, extracted {len(page_text)} characters")
            
            print(f"Tesseract completed, total extracted text length: {len(extracted_text)}")
            return extracted_text.strip()
            
        except Exception as e:
            print(f"Tesseract extraction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Tesseract OCR extraction failed: {str(e)}"
            )
    
    def _extract_with_easyocr(self, pdf_path: str) -> str:
        """Extract text using EasyOCR only."""
        try:
            import easyocr
            from pdf2image import convert_from_path
            import numpy as np
            
            print(f"Starting EasyOCR extraction for: {pdf_path}")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            print(f"Converted {len(images)} pages to images")
            
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'])
            
            extracted_text = ""
            
            # Process each page
            for i, image in enumerate(images):
                print(f"Processing page {i+1} with EasyOCR...")
                # Convert PIL image to numpy array for EasyOCR
                img_array = np.array(image)
                
                # Extract text with EasyOCR
                results = reader.readtext(img_array)
                page_text = ""
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.3:  # Filter low confidence results
                        page_text += text + " "
                
                extracted_text += f"\n--- Page {i+1} ---\n{page_text.strip()}\n"
                print(f"Page {i+1} EasyOCR completed, extracted {len(page_text)} characters")
            
            print(f"EasyOCR completed, total extracted text length: {len(extracted_text)}")
            return extracted_text.strip()
            
        except Exception as e:
            print(f"EasyOCR extraction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"EasyOCR extraction failed: {str(e)}"
            )
    
    def _extract_with_tesseract_from_path(self, pdf_path: str) -> str:
        """Extract text using Tesseract OCR only."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            print(f"Starting Tesseract extraction for: {pdf_path}")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            print(f"Converted {len(images)} pages to images")
            
            return self._extract_with_tesseract(images)
            
        except Exception as e:
            print(f"Tesseract extraction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Tesseract extraction failed: {str(e)}"
            )
    
    def _get_additional_context(self) -> str:
        """Get additional context for the request."""
        from datetime import date
        todays_date = date.today()

        return f"Todays date is {todays_date}"


    
    def process_with_openai(self, text: str) -> Dict[str, Any]:
        """Process text using OpenAI API."""
        try:
            import openai
            
            print(f"Processing text with OpenAI, text length: {len(text)}")
            
            openai.api_key = self.settings.OPENAI_API_KEY

            context = self._get_additional_context()
            
            system_prompt = f"""You are a highly intelligent assistant that specializes in analyzing handwritten notes from photos, scans, or transcriptions. These notes are often from meetings, brainstorming sessions, strategy discussions, or personal planning. They may contain:
• Unstructured or bullet-point text
• Incomplete thoughts or shorthand
• Crossed-out or corrected items
• Diagrams, arrows, or visual cues
• Non-standard formatting or grammar

Your goal is to provide the user with clean, professional, and actionable outputs, in two steps:

1. **Accurate Transcription (Clean-Up)**
• Convert all handwritten content into legible, well-formatted digital text.
• Do not modify the meaning, but feel free to improve clarity (fix spelling, grammar, and punctuation).
• Preserve the structure or flow (e.g., bulleted lists, sections, subtopics) as much as possible.
• If any part of the handwriting is unreadable, mark it clearly as [UNREADABLE].

2. **Insightful Summary**
• Summarize the key points, insights, and actions from the notes.
• Group similar ideas, remove duplicates, and eliminate irrelevant information.
• Use clear, skimmable formatting such as bullet points and short paragraphs.
• Highlight any questions, feedback, or follow-up items noted in the text.
• If appropriate, infer intent or implications behind the notes to support decision-making.

**Tone & Output Guidelines:**
• Use a professional, neutral tone.
• Be concise but informative.
• If you're unsure about the content, use phrases like "Possibly refers to…" or mark it with [Ambiguous].
• Always return structured information in JSON format with the following fields:
  - summary: A concise 2-3 sentence summary of the main content
  - tasks: Array of actionable tasks, to-dos, or action items mentioned
  - topics: Array of key topics or themes discussed
  - dates: Array of important dates, deadlines, or time references
  - transcription: Clean, formatted version of the original text (if applicable)
  - confidence: Confidence score between 0.0 and 1.0

**Additional Context**
Here is additional context about the current state to help in guiding your response: {context}
  
"""

            user_prompt = f"""Analyze the following text extracted from a handwritten note or document:

{text}

Please respond in JSON format:
{{
    "summary": "brief summary here",
    "tasks": ["task 1", "task 2", "task 3"],
    "topics": ["topic 1", "topic 2"],
    "dates": ["date 1", "date 2"],
    "transcription": "clean formatted version of the text",
    "confidence": 0.95
}}"""
            
            print("Sending request to OpenAI...")
            response = openai.ChatCompletion.create(
                model=self.settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            print("OpenAI response received")
            
            # Parse the JSON response
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
                print("Successfully parsed OpenAI JSON response")
                
                # Apply cleanup to the result
                cleaned_result = self._cleanup_ai_output(result)
                return cleaned_result
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {str(e)}")
                # Fallback if JSON parsing fails
                fallback_result = {
                    "summary": content,
                    "tasks": [],
                    "topics": [],
                    "dates": [],
                    "transcription": text,
                    "confidence": 0.8
                }
                # Apply cleanup to fallback result as well
                return self._cleanup_ai_output(fallback_result)
                
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
            
            # Enhanced rule-based analysis with better prompt understanding
            text_lower = text.lower()
            
            # Extract potential tasks (lines with action words)
            action_words = [
                'todo', 'task', 'action', 'need to', 'must', 'should', 'will', 'going to',
                'do', 'complete', 'finish', 'start', 'prepare', 'review', 'check',
                'follow up', 'call', 'email', 'meet', 'schedule', 'book', 'arrange',
                'update', 'create', 'build', 'design', 'implement', 'test', 'deploy'
            ]
            
            lines = text.split('\n')
            tasks = []
            for line in lines:
                line_clean = line.strip()
                if line_clean and any(word in line_clean.lower() for word in action_words):
                    # Clean up the task text
                    task_text = line_clean
                    # Remove common prefixes
                    for prefix in ['•', '-', '*', '→', '>', 'todo:', 'task:', 'action:']:
                        if task_text.lower().startswith(prefix.lower()):
                            task_text = task_text[len(prefix):].strip()
                    if task_text:
                        tasks.append(task_text)
            
            # Extract potential dates with enhanced patterns
            import re
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
                r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
                r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4}\b',  # Abbreviated months
                r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',  # DD Month YYYY
                r'\btoday\b', r'\btomorrow\b', r'\bnext week\b', r'\bthis week\b', r'\bnext month\b'
            ]
            
            dates = []
            for pattern in date_patterns:
                found_dates = re.findall(pattern, text, re.IGNORECASE)
                dates.extend(found_dates)
            
            # Remove duplicates and clean up dates
            dates = list(set(dates))
            
            # Extract topics (look for common topic indicators and section headers)
            topic_indicators = [
                'topic', 'subject', 'theme', 'about', 'regarding', 'concerning',
                'project', 'meeting', 'discussion', 'agenda', 'notes on', 'summary of'
            ]
            
            topics = []
            for line in lines:
                line_clean = line.strip()
                if line_clean and any(indicator in line_clean.lower() for indicator in topic_indicators):
                    # Clean up topic text
                    topic_text = line_clean
                    for prefix in ['•', '-', '*', '→', '>', 'topic:', 'subject:', 'theme:']:
                        if topic_text.lower().startswith(prefix.lower()):
                            topic_text = topic_text[len(prefix):].strip()
                    if topic_text and len(topic_text) > 3:
                        topics.append(topic_text)
            
            # Generate a better summary
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences:
                # Take first 2-3 meaningful sentences
                summary_sentences = []
                for sentence in sentences[:3]:
                    if len(sentence) > 20:  # Only include substantial sentences
                        summary_sentences.append(sentence)
                
                if summary_sentences:
                    summary = '. '.join(summary_sentences) + '.'
                else:
                    summary = "Handwritten notes containing various topics and action items."
            else:
                summary = "Handwritten notes requiring further analysis."
            
            # Clean transcription (basic formatting)
            transcription_lines = []
            for line in lines:
                line_clean = line.strip()
                if line_clean:
                    # Basic formatting improvements
                    if line_clean.startswith(('•', '-', '*', '→', '>')):
                        transcription_lines.append(f"• {line_clean[1:].strip()}")
                    else:
                        transcription_lines.append(line_clean)
            
            transcription = '\n'.join(transcription_lines)
            
            result = {
                "summary": summary,
                "tasks": tasks[:10],  # Limit to 10 tasks
                "topics": topics[:5],  # Limit to 5 topics
                "dates": dates,
                "transcription": transcription,
                "confidence": 0.7,  # Improved confidence for enhanced rule-based approach
                "method": "enhanced_rule_based"
            }
            
            print(f"Local processing completed, found {len(tasks)} tasks, {len(dates)} dates, {len(topics)} topics")
            
            # Apply cleanup to the result
            cleaned_result = self._cleanup_ai_output(result)
            return cleaned_result
            
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
            
            # Check if multimodal LLM is requested
            if self.provider == "multimodal":
                return self.process_with_multimodal_llm(pdf_path)

            # Check if OpenAI multimodal is requested
            if self.provider == "openai_multimodal":
                return self.process_with_openai_multimodal(pdf_path)
            
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
    
    def process_with_multimodal_llm(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF using multimodal LLM (Ollama + LLaVA) by sending images directly."""
        try:
            import requests
            import base64
            from pdf2image import convert_from_path
            from PIL import Image
            import io
            
            print(f"Starting multimodal LLM processing for: {pdf_path}")
            
            # Convert PDF to images
            print("Converting PDF to images...")
            images = convert_from_path(pdf_path, dpi=300)
            print(f"Converted {len(images)} pages to images")
            
            # Process each page with LLaVA
            all_results = []
            
            for i, image in enumerate(images):
                print(f"Processing page {i+1} with LLaVA...")
                
                # Convert PIL image to base64
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Prepare the prompt for handwritten note analysis
                prompt = f"""You are a highly intelligent assistant that specializes in analyzing handwritten notes from photos, scans, or transcriptions. These notes are often from meetings, brainstorming sessions, strategy discussions, or personal planning. They may contain:
• Unstructured or bullet-point text
• Incomplete thoughts or shorthand
• Crossed-out or corrected items
• Diagrams, arrows, or visual cues
• Non-standard formatting or grammar

Your goal is to provide the user with clean, professional, and actionable outputs, in two steps:

1. **Accurate Transcription (Clean-Up)**
• Convert all handwritten content into legible, well-formatted digital text.
• Do not modify the meaning, but feel free to improve clarity (fix spelling, grammar, and punctuation).
• Preserve the structure or flow (e.g., bulleted lists, sections, subtopics) as much as possible.
• If any part of the handwriting is unreadable, mark it clearly as [UNREADABLE].

2. **Insightful Summary**
• Summarize the key points, insights, and actions from the notes.
• Group similar ideas, remove duplicates, and eliminate irrelevant information.
• Use clear, skimmable formatting such as bullet points and short paragraphs.
• Highlight any questions, feedback, or follow-up items noted in the text.
• If appropriate, infer intent or implications behind the notes to support decision-making.

**Tone & Output Guidelines:**
• Use a professional, neutral tone.
• Be concise but informative.
• If you're unsure about the content, use phrases like "Possibly refers to…" or mark it with [Ambiguous].

Analyze this handwritten note image and provide:

**Transcription:**
[Provide a clean, formatted version of the handwritten text]

**Summary:**
[2-3 sentence summary of the main content]

**Tasks/Action Items:**
[List any tasks, to-dos, or action items mentioned]

**Key Topics:**
[Identify the main topics or themes discussed]

**Important Dates:**
[Extract any dates, deadlines, or time references]

**Additional Notes:**
[Any other important information or insights]

Focus on extracting actionable information and key insights from the handwritten content. Be thorough but concise."""
                
                # Call Ollama LLaVA API
                try:
                    response = requests.post(
                        'http://localhost:11434/api/generate',
                        json={
                            'model': 'llava',
                            'prompt': prompt,
                            'images': [img_str],
                            'stream': False
                        },
                        timeout=120  # 2 minute timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        page_text = result.get('response', '')
                        print(f"Page {i+1} LLaVA completed, extracted {len(page_text)} characters")
                        
                        all_results.append({
                            'page': i + 1,
                            'text': page_text,
                            'length': len(page_text)
                        })
                    else:
                        print(f"LLaVA API error for page {i+1}: {response.status_code}")
                        all_results.append({
                            'page': i + 1,
                            'text': f"Error: API returned status {response.status_code}",
                            'length': 0
                        })
                        
                except requests.exceptions.RequestException as e:
                    print(f"Request error for page {i+1}: {str(e)}")
                    all_results.append({
                        'page': i + 1,
                        'text': f"Error: {str(e)}",
                        'length': 0
                    })
            
            # Aggregate results
            total_text = "\n\n--- Page Separator ---\n\n".join([r['text'] for r in all_results])
            total_length = sum([r['length'] for r in all_results])
            
            print(f"Multimodal LLM completed, total extracted text length: {total_length}")
            
            # Parse the aggregated response to extract structured information
            parsed_result = self._parse_multimodal_response(total_text)
            
            return {
                "summary": parsed_result.get("summary", "No summary available"),
                "tasks": parsed_result.get("tasks", []),
                "topics": parsed_result.get("topics", []),
                "dates": parsed_result.get("dates", []),
                "transcription": parsed_result.get("transcription", ""),
                "confidence": 0.9,  # High confidence for multimodal approach
                "method": "multimodal_llava",
                "raw_text": total_text,
                "pages_processed": len(all_results),
                "total_length": total_length
            }
            
        except Exception as e:
            print(f"Multimodal LLM processing failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Multimodal LLM processing failed: {str(e)}"
            )
    
    def _parse_multimodal_response(self, text: str) -> Dict[str, Any]:
        """Parse the multimodal LLM response to extract structured information."""
        try:
            # First, try to parse as JSON if it looks like JSON
            text_clean = text.strip()
            if text_clean.startswith('{') and text_clean.endswith('}'):
                try:
                    import json
                    json_result = json.loads(text_clean)
                    print("Successfully parsed JSON response from multimodal model")
                    
                    # Apply cleanup to the JSON result
                    cleaned_result = self._cleanup_ai_output(json_result)
                    return cleaned_result
                except json.JSONDecodeError as e:
                    print(f"JSON parsing failed, falling back to text parsing: {str(e)}")
            
            # Try to extract JSON from the text if it contains JSON-like content
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, text)
            
            if json_matches:
                for json_str in json_matches:
                    try:
                        import json
                        json_result = json.loads(json_str)
                        print("Successfully extracted and parsed JSON from text")
                        
                        # Apply cleanup to the JSON result
                        cleaned_result = self._cleanup_ai_output(json_result)
                        return cleaned_result
                    except json.JSONDecodeError:
                        continue
            
            # Enhanced parsing logic for structured responses
            text_lower = text.lower()
            
            # Try to extract structured sections first
            sections = {
                'transcription': '',
                'summary': '',
                'tasks': [],
                'topics': [],
                'dates': [],
                'additional_notes': ''
            }
            
            # Look for structured sections in the response
            current_section = None
            lines = text.split('\n')
            
            for line in lines:
                line_clean = line.strip()
                if not line_clean:
                    continue
                
                # Check for section headers
                if 'transcription:' in line_clean.lower():
                    current_section = 'transcription'
                    continue
                elif 'summary:' in line_clean.lower():
                    current_section = 'summary'
                    continue
                elif 'tasks' in line_clean.lower() or 'action items' in line_clean.lower():
                    current_section = 'tasks'
                    continue
                elif 'topics' in line_clean.lower() or 'key topics' in line_clean.lower():
                    current_section = 'topics'
                    continue
                elif 'dates' in line_clean.lower() or 'important dates' in line_clean.lower():
                    current_section = 'dates'
                    continue
                elif 'additional notes' in line_clean.lower() or 'notes:' in line_clean.lower():
                    current_section = 'additional_notes'
                    continue
                
                # Add content to current section
                if current_section:
                    if current_section in ['tasks', 'topics', 'dates']:
                        # For list sections, look for bullet points or numbered items
                        if line_clean.startswith(('•', '-', '*', '→', '>', '1.', '2.', '3.')):
                            content = line_clean
                            for prefix in ['•', '-', '*', '→', '>', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']:
                                if content.startswith(prefix):
                                    content = content[len(prefix):].strip()
                                    break
                            if content:
                                sections[current_section].append(content)
                        elif line_clean and not line_clean.startswith('['):  # Skip placeholder text
                            sections[current_section].append(line_clean)
                    else:
                        # For text sections, append to existing content
                        if sections[current_section]:
                            sections[current_section] += ' ' + line_clean
                        else:
                            sections[current_section] = line_clean
            
            # Fallback extraction if structured parsing didn't work well
            if not sections['tasks']:
                # Extract potential tasks (lines with action words)
                action_words = [
                    'todo', 'task', 'action', 'need to', 'must', 'should', 'will', 'going to', 'do', 'complete',
                    'finish', 'start', 'prepare', 'review', 'check', 'follow up', 'call', 'email', 'meet',
                    'schedule', 'book', 'arrange', 'update', 'create', 'build', 'design', 'implement', 'test', 'deploy'
                ]
                
                for line in lines:
                    line_clean = line.strip()
                    if line_clean and any(word in line_clean.lower() for word in action_words):
                        # Clean up the task text
                        task_text = line_clean
                        for prefix in ['•', '-', '*', '→', '>', 'todo:', 'task:', 'action:']:
                            if task_text.lower().startswith(prefix.lower()):
                                task_text = task_text[len(prefix):].strip()
                        if task_text and task_text not in sections['tasks']:
                            sections['tasks'].append(task_text)
            
            if not sections['dates']:
                # Extract potential dates with enhanced patterns
                import re
                date_patterns = [
                    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
                    r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
                    r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
                    r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4}\b',  # Abbreviated months
                    r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',  # DD Month YYYY
                    r'\btoday\b', r'\btomorrow\b', r'\bnext week\b', r'\bthis week\b', r'\bnext month\b'
                ]
                
                for pattern in date_patterns:
                    found_dates = re.findall(pattern, text, re.IGNORECASE)
                    sections['dates'].extend(found_dates)
                
                # Remove duplicates
                sections['dates'] = list(set(sections['dates']))
            
            if not sections['topics']:
                # Extract topics (look for common topic indicators and section headers)
                topic_indicators = [
                    'topic', 'subject', 'theme', 'about', 'regarding', 'concerning',
                    'project', 'meeting', 'discussion', 'agenda', 'notes on', 'summary of'
                ]
                
                for line in lines:
                    line_clean = line.strip()
                    if line_clean and any(indicator in line_clean.lower() for indicator in topic_indicators):
                        # Clean up topic text
                        topic_text = line_clean
                        for prefix in ['•', '-', '*', '→', '>', 'topic:', 'subject:', 'theme:']:
                            if topic_text.lower().startswith(prefix.lower()):
                                topic_text = topic_text[len(prefix):].strip()
                        if topic_text and len(topic_text) > 3 and topic_text not in sections['topics']:
                            sections['topics'].append(topic_text)
            
            # Generate summary if not found
            if not sections['summary']:
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                if sentences:
                    # Take first 2-3 meaningful sentences
                    summary_sentences = []
                    for sentence in sentences[:3]:
                        if len(sentence) > 20:  # Only include substantial sentences
                            summary_sentences.append(sentence)
                    
                    if summary_sentences:
                        sections['summary'] = '. '.join(summary_sentences) + '.'
                    else:
                        sections['summary'] = "Handwritten notes containing various topics and action items."
                else:
                    sections['summary'] = "Handwritten notes requiring further analysis."
            
            # Clean transcription if not found
            if not sections['transcription']:
                # Use the original text as fallback transcription
                transcription_lines = []
                for line in lines:
                    line_clean = line.strip()
                    if line_clean and not line_clean.startswith('['):  # Skip placeholder text
                        # Basic formatting improvements
                        if line_clean.startswith(('•', '-', '*', '→', '>')):
                            transcription_lines.append(f"• {line_clean[1:].strip()}")
                        else:
                            transcription_lines.append(line_clean)
                
                sections['transcription'] = '\n'.join(transcription_lines)
            
            result = {
                "summary": sections['summary'],
                "tasks": sections['tasks'][:10],  # Limit to 10 tasks
                "topics": sections['topics'][:5],  # Limit to 5 topics
                "dates": sections['dates'],
                "transcription": sections['transcription'],
                "additional_notes": sections['additional_notes']
            }
            
            # Apply cleanup to the parsed result
            cleaned_result = self._cleanup_ai_output(result)
            return cleaned_result
            
        except Exception as e:
            print(f"Error parsing multimodal response: {str(e)}")
            
            # Create a safe fallback that doesn't include raw JSON in summary
            safe_text = text
            # Remove any JSON-like content from the text for the summary
            import re
            # Remove JSON objects from the text
            safe_text = re.sub(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', safe_text)
            # Remove any remaining JSON-like patterns
            safe_text = re.sub(r'"[^"]*":\s*"[^"]*"', '', safe_text)
            safe_text = re.sub(r'"[^"]*":\s*\[[^\]]*\]', '', safe_text)
            
            # Clean up the text
            safe_text = safe_text.replace('{', '').replace('}', '').replace('"', '').replace(',', ' ')
            safe_text = ' '.join(safe_text.split())  # Normalize whitespace
            
            fallback_result = {
                "summary": safe_text[:200] + "..." if len(safe_text) > 200 else safe_text if safe_text.strip() else "Handwritten notes requiring further analysis.",
                "tasks": [],
                "topics": [],
                "dates": [],
                "transcription": text,
                "additional_notes": ""
            }
            # Apply cleanup to fallback result as well
            return self._cleanup_ai_output(fallback_result)
    
    def _cleanup_ai_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up formatting issues in AI model outputs."""
        try:
            cleaned_result = result.copy()
            
            # Clean up tasks - remove quotation marks and parentheses
            if 'tasks' in cleaned_result and isinstance(cleaned_result['tasks'], list):
                cleaned_tasks = []
                for task in cleaned_result['tasks']:
                    if isinstance(task, str):
                        # Remove quotation marks (both single and double)
                        cleaned_task = task.strip()
                        if (cleaned_task.startswith('"') and cleaned_task.endswith('"')) or \
                           (cleaned_task.startswith("'") and cleaned_task.endswith("'")):
                            cleaned_task = cleaned_task[1:-1].strip()
                        
                        # Remove parentheses (both round and square brackets)
                        if cleaned_task.startswith('(') and cleaned_task.endswith(')'):
                            cleaned_task = cleaned_task[1:-1].strip()
                        if cleaned_task.startswith('[') and cleaned_task.endswith(']'):
                            cleaned_task = cleaned_task[1:-1].strip()
                        
                        # Remove any remaining quotation marks or parentheses from the middle
                        cleaned_task = cleaned_task.replace('"', '').replace("'", '')
                        cleaned_task = cleaned_task.replace('(', '').replace(')', '')
                        cleaned_task = cleaned_task.replace('[', '').replace(']', '')
                        
                        # Only add non-empty tasks
                        if cleaned_task and len(cleaned_task.strip()) > 0:
                            cleaned_tasks.append(cleaned_task.strip())
                
                cleaned_result['tasks'] = cleaned_tasks
            
            # Clean up topics - similar cleanup
            if 'topics' in cleaned_result and isinstance(cleaned_result['topics'], list):
                cleaned_topics = []
                for topic in cleaned_result['topics']:
                    if isinstance(topic, str):
                        # Remove quotation marks and parentheses
                        cleaned_topic = topic.strip()
                        if (cleaned_topic.startswith('"') and cleaned_topic.endswith('"')) or \
                           (cleaned_topic.startswith("'") and cleaned_topic.endswith("'")):
                            cleaned_topic = cleaned_topic[1:-1].strip()
                        
                        if cleaned_topic.startswith('(') and cleaned_topic.endswith(')'):
                            cleaned_topic = cleaned_topic[1:-1].strip()
                        if cleaned_topic.startswith('[') and cleaned_topic.endswith(']'):
                            cleaned_topic = cleaned_topic[1:-1].strip()
                        
                        # Remove any remaining quotation marks or parentheses
                        cleaned_topic = cleaned_topic.replace('"', '').replace("'", '')
                        cleaned_topic = cleaned_topic.replace('(', '').replace(')', '')
                        cleaned_topic = cleaned_topic.replace('[', '').replace(']', '')
                        
                        # Only add non-empty topics
                        if cleaned_topic and len(cleaned_topic.strip()) > 0:
                            cleaned_topics.append(cleaned_topic.strip())
                
                cleaned_result['topics'] = cleaned_topics
            
            # Clean up dates - similar cleanup
            if 'dates' in cleaned_result and isinstance(cleaned_result['dates'], list):
                cleaned_dates = []
                for date_item in cleaned_result['dates']:
                    if isinstance(date_item, str):
                        # Remove quotation marks and parentheses
                        cleaned_date = date_item.strip()
                        if (cleaned_date.startswith('"') and cleaned_date.endswith('"')) or \
                           (cleaned_date.startswith("'") and cleaned_date.endswith("'")):
                            cleaned_date = cleaned_date[1:-1].strip()
                        
                        if cleaned_date.startswith('(') and cleaned_date.endswith(')'):
                            cleaned_date = cleaned_date[1:-1].strip()
                        if cleaned_date.startswith('[') and cleaned_date.endswith(']'):
                            cleaned_date = cleaned_date[1:-1].strip()
                        
                        # Remove any remaining quotation marks or parentheses
                        cleaned_date = cleaned_date.replace('"', '').replace("'", '')
                        cleaned_date = cleaned_date.replace('(', '').replace(')', '')
                        cleaned_date = cleaned_date.replace('[', '').replace(']', '')
                        
                        # Only add non-empty dates
                        if cleaned_date and len(cleaned_date.strip()) > 0:
                            cleaned_dates.append(cleaned_date.strip())
                
                cleaned_result['dates'] = cleaned_dates
            
            # Clean up summary and transcription - remove extra quotation marks
            for field in ['summary', 'transcription']:
                if field in cleaned_result and isinstance(cleaned_result[field], str):
                    cleaned_text = cleaned_result[field].strip()
                    # Remove surrounding quotation marks
                    if (cleaned_text.startswith('"') and cleaned_text.endswith('"')) or \
                       (cleaned_text.startswith("'") and cleaned_text.endswith("'")):
                        cleaned_text = cleaned_text[1:-1].strip()
                    
                    cleaned_result[field] = cleaned_text
            
            print(f"Cleaned up AI output: {len(cleaned_result.get('tasks', []))} tasks, {len(cleaned_result.get('topics', []))} topics, {len(cleaned_result.get('dates', []))} dates")
            return cleaned_result
            
        except Exception as e:
            print(f"Error cleaning up AI output: {str(e)}")
            return result

    def process_with_openai_multimodal(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF using OpenAI's multimodal capabilities (GPT-4 Vision) by sending images directly."""
        try:
            import openai
            import base64
            from pdf2image import convert_from_path
            from PIL import Image
            import io
            
            print(f"Starting OpenAI multimodal processing for: {pdf_path}")
            
            # Set up OpenAI API
            openai.api_key = self.settings.OPENAI_API_KEY

            date = self._get_additional_context()
            
            # Convert PDF to images
            print("Converting PDF to images...")
            images = convert_from_path(pdf_path, dpi=300)
            print(f"Converted {len(images)} pages to images")
            
            # Process each page with OpenAI Vision
            all_results = []
            
            for i, image in enumerate(images):
                print(f"Processing page {i+1} with OpenAI Vision...")
                
                # Convert PIL image to base64
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Prepare the prompt for handwritten note analysis
                system_prompt = f"""You are a highly intelligent assistant that specializes in analyzing handwritten notes from photos, scans, or transcriptions. These notes are often from meetings, brainstorming sessions, strategy discussions, or personal planning. They may contain:
• Unstructured or bullet-point text
• Incomplete thoughts or shorthand
• Crossed-out or corrected items
• Diagrams, arrows, or visual cues
• Non-standard formatting or grammar

Your goal is to provide the user with clean, professional, and actionable outputs, in two steps:

1. **Accurate Transcription (Clean-Up)**
• Convert all handwritten content into legible, well-formatted digital text.
• Do not modify the meaning, but feel free to improve clarity (fix spelling, grammar, and punctuation).
• Preserve the structure or flow (e.g., bulleted lists, sections, subtopics) as much as possible.
• If any part of the handwriting is unreadable, mark it clearly as [UNREADABLE].

2. **Insightful Summary**
• Summarize the key points, insights, and actions from the notes.
• Group similar ideas, remove duplicates, and eliminate irrelevant information.
• Use clear, skimmable formatting such as bullet points and short paragraphs.
• Highlight any questions, feedback, or follow-up items noted in the text.
• If appropriate, infer intent or implications behind the notes to support decision-making.

**Tone & Output Guidelines:**
• Use a professional, neutral tone.
• Be concise but informative.
• If you're unsure about the content, use phrases like "Possibly refers to…" or mark it with [Ambiguous].
• Always return structured information in JSON format with the following fields:
  - summary: A concise 2-3 sentence summary of the main content
  - tasks: Array of actionable tasks, to-dos, or action items mentioned
  - topics: Array of key topics or themes discussed
  - dates: Array of important dates, deadlines, or time references
  - transcription: Clean, formatted version of the original text
  - confidence: Confidence score between 0.0 and 1.0

**Additional Context:**
{date}

Analyze this handwritten note image and provide the information in the specified JSON format."""

                user_prompt = """Please analyze this handwritten note image and provide the information in the following JSON format:

{
    "summary": "brief summary here",
    "tasks": ["task 1", "task 2", "task 3"],
    "topics": ["topic 1", "topic 2"],
    "dates": ["date 1", "date 2"],
    "transcription": "clean formatted version of the text",
    "confidence": 0.95
}

Focus on extracting actionable information and key insights from the handwritten content. Be thorough but concise."""
                
                # Call OpenAI Vision API
                try:
                    response = openai.ChatCompletion.create(
                        model=self.settings.OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": user_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_str}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=2000,
                        temperature=0.3
                    )
                    
                    if response.choices and response.choices[0].message:
                        page_text = response.choices[0].message.content
                        print(f"Page {i+1} OpenAI Vision completed, extracted {len(page_text)} characters")
                        
                        all_results.append({
                            'page': i + 1,
                            'text': page_text,
                            'length': len(page_text)
                        })
                    else:
                        print(f"OpenAI Vision API error for page {i+1}: No response content")
                        all_results.append({
                            'page': i + 1,
                            'text': f"Error: No response content from OpenAI Vision",
                            'length': 0
                        })
                        
                except Exception as e:
                    print(f"OpenAI Vision API error for page {i+1}: {str(e)}")
                    all_results.append({
                        'page': i + 1,
                        'text': f"Error: {str(e)}",
                        'length': 0
                    })
            
            # Aggregate results
            total_text = "\n\n--- Page Separator ---\n\n".join([r['text'] for r in all_results])
            total_length = sum([r['length'] for r in all_results])
            
            print(f"OpenAI multimodal completed, total extracted text length: {total_length}")
            
            # Parse the aggregated response to extract structured information
            parsed_result = self._parse_multimodal_response(total_text)
            
            # Clean up the result
            cleaned_result = self._cleanup_ai_output(parsed_result)
            
            return {
                "summary": cleaned_result.get("summary", "No summary available"),
                "tasks": cleaned_result.get("tasks", []),
                "topics": cleaned_result.get("topics", []),
                "dates": cleaned_result.get("dates", []),
                "transcription": cleaned_result.get("transcription", ""),
                "confidence": 0.95,  # High confidence for OpenAI Vision approach
                "method": "openai_vision",
                "raw_text": total_text,
                "pages_processed": len(all_results),
                "total_length": total_length
            }
            
        except Exception as e:
            print(f"OpenAI multimodal processing failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI multimodal processing failed: {str(e)}"
            ) 