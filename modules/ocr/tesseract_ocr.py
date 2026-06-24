import sys
from PIL import Image
from pathlib import Path

try:
    import pytesseract
except ImportError:
    pytesseract = None

# On Windows, commonly installed at C:\Program Files\Tesseract-OCR\tesseract.exe
# If pytesseract is not configured in environment variables, we try to set it.
TESSERACT_CMD_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]

def _configure_tesseract():
    if pytesseract is None:
        return False
        
    # Check if tesseract is already in PATH or configured
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        # Try candidate paths on Windows
        for cmd in TESSERACT_CMD_CANDIDATES:
            if Path(cmd).exists():
                pytesseract.pytesseract.tesseract_cmd = cmd
                try:
                    pytesseract.get_tesseract_version()
                    return True
                except Exception:
                    pass
        return False

def extract_text_using_tesseract(image_path: str) -> str:
    """
    Extracts text from an image or scanned page using Tesseract OCR.
    """
    if pytesseract is None:
        return "[ERROR] pytesseract library is not installed."
        
    if not _configure_tesseract():
        return "[WARN] Tesseract-OCR engine is not found or configured on this system."

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"[ERROR] Tesseract OCR extraction failed: {e}"
