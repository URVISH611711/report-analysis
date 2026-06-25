import sys
from PIL import Image
from pathlib import Path

try:
    import pytesseract
except ImportError:
    pytesseract = None

from modules.ocr.image_preprocessor import preprocess_image

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
    
    Uses a dual-pass strategy with image preprocessing:
    1. Preprocesses the image (deskew, contrast enhancement, thresholding)
    2. First pass with PSM 6 (uniform block of text)
    3. If yield is low, second pass with PSM 4 (multi-column layout)
    4. Merges and deduplicates results from both passes
    """
    if pytesseract is None:
        return "[ERROR] pytesseract library is not installed."
        
    if not _configure_tesseract():
        return "[WARN] Tesseract-OCR engine is not found or configured on this system."

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    try:
        # Step 1: Preprocess image for optimal OCR quality
        print("[OCR] Applying image preprocessing (deskew, contrast, threshold)...")
        preprocessed_img = preprocess_image(str(path))
        
        # Step 2: First pass — PSM 6 (assumes uniform block of text)
        # OEM 3 = default (LSTM + legacy combined)
        config_psm6 = r'--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.:,/-% ()'
        text_psm6 = pytesseract.image_to_string(preprocessed_img, config=config_psm6)
        text_psm6 = text_psm6.strip()
        
        print(f"[OCR] PSM 6 pass: extracted {len(text_psm6)} chars")
        
        # Step 3: If first pass yields too little, try PSM 4 (multi-column)
        text_psm4 = ""
        if len(text_psm6) < 50:
            print("[OCR] PSM 6 yield low, retrying with PSM 4 (multi-column)...")
            config_psm4 = r'--psm 4 --oem 3'
            text_psm4 = pytesseract.image_to_string(preprocessed_img, config=config_psm4)
            text_psm4 = text_psm4.strip()
            print(f"[OCR] PSM 4 pass: extracted {len(text_psm4)} chars")
        
        # Step 4: Also try on the original (unprocessed) image as fallback
        # Sometimes preprocessing removes useful information
        original_img = Image.open(path)
        config_original = r'--psm 6 --oem 3'
        text_original = pytesseract.image_to_string(original_img, config=config_original)
        text_original = text_original.strip()
        
        # Step 5: Merge results — take the best output
        # Use the longest result as primary, merge unique lines from others
        candidates = [
            ("preprocessed-psm6", text_psm6),
            ("preprocessed-psm4", text_psm4),
            ("original-psm6", text_original)
        ]
        
        # Sort by length (longest first — more extracted text is generally better)
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        best_name, best_text = candidates[0]
        print(f"[OCR] Best result from {best_name}: {len(best_text)} chars")
        
        # Merge unique lines from other passes into the best result
        best_lines = set(best_text.split("\n"))
        merged_lines = list(best_text.split("\n"))
        
        for name, text in candidates[1:]:
            if not text:
                continue
            for line in text.split("\n"):
                line_stripped = line.strip()
                # Add lines that contain digits (likely lab values) and aren't already present
                if line_stripped and line_stripped not in best_lines and _has_lab_value(line_stripped):
                    merged_lines.append(line_stripped)
                    best_lines.add(line_stripped)
        
        final_text = "\n".join(merged_lines).strip()
        
        if not final_text:
            return "[WARN] Tesseract OCR extracted no text from image."
            
        return final_text
        
    except Exception as e:
        return f"[ERROR] Tesseract OCR extraction failed: {e}"


def _has_lab_value(line: str) -> bool:
    """
    Checks if a line likely contains a lab value (has both text and numbers).
    Used for intelligent merge of multi-pass OCR results.
    """
    import re
    has_text = bool(re.search(r'[a-zA-Z]{2,}', line))
    has_number = bool(re.search(r'\d+(?:\.\d+)?', line))
    return has_text and has_number
