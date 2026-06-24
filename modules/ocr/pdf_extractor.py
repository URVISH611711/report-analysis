import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a digitally-created (native) PDF using PyMuPDF.
    If the PDF is scanned (no native text), it returns an empty string or minimal noise.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    text_content = []
    try:
        doc = fitz.open(str(path))
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text_content.append(page_text)
        doc.close()
    except Exception as e:
        print(f"[WARN] PyMuPDF failed to extract text from {pdf_path}: {e}")
        
    return "\n".join(text_content).strip()

def is_native_pdf(pdf_path: str) -> bool:
    """
    Checks if a PDF contains actual digital text (not scanned images).
    """
    text = extract_text_from_pdf(pdf_path)
    # If we extracted more than 50 characters of non-whitespace text, it's likely a native PDF
    return len(text) > 50
