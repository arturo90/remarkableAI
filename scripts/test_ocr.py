import sys
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract

if len(sys.argv) < 2:
    print("Usage: python scripts/test_ocr.py <path_to_pdf>")
    sys.exit(1)

pdf_path = Path(sys.argv[1])
if not pdf_path.exists():
    print(f"File not found: {pdf_path}")
    sys.exit(1)

print(f"Converting first page of {pdf_path} to image...")
images = convert_from_path(str(pdf_path), dpi=300)
first_image = images[0]

print("Running OCR on the first page...")
text = pytesseract.image_to_string(first_image, lang='eng')

print("\n--- OCR Extracted Text ---\n")
print(text) 