import os
import sys
from pathlib import Path

# Lazy loading of PaddleOCR as it imports paddle which is heavy
_ocr_engine = None

def _get_paddle_ocr_engine():
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    try:
        from paddleocr import PaddleOCR as OriginalPaddleOCR
        # Initialize PaddleOCR:
        # use_angle_cls=True detects text orientation
        # lang="en" for English reports
        # show_log=False keeps console output clean
        _ocr_engine = OriginalPaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        return _ocr_engine
    except ImportError as e:
        print(f"[WARN] paddleocr or paddlepaddle is not installed: {e}")
        return None
    except Exception as e:
        print(f"[WARN] Failed to initialize PaddleOCR: {e}")
        return None

def extract_text_using_paddle(image_path: str) -> str:
    """
    Extracts text from an image or scanned document using PaddleOCR.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    engine = _get_paddle_ocr_engine()
    if engine is None:
        return "[WARN] PaddleOCR engine could not be loaded."

    try:
        # Run OCR on the image
        result = engine.ocr(str(path), cls=True)
        if not result or result[0] is None:
            return ""

        # Extract text strings and assemble page
        lines = []
        for line in result[0]:
            # result structure is [[[coord_coords], (text_str, confidence_float)]]
            text_str = line[1][0]
            lines.append(text_str)
            
        return "\n".join(lines).strip()
    except Exception as e:
        return f"[ERROR] PaddleOCR extraction failed: {e}"
