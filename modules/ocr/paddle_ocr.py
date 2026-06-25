import os
import sys
from pathlib import Path

from modules.ocr.image_preprocessor import preprocess_image

# Lazy loading of PaddleOCR as it imports paddle which is heavy
_ocr_engine = None

def _get_paddle_ocr_engine():
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    try:
        from paddleocr import PaddleOCR as OriginalPaddleOCR
        
        # Try with show_log parameter first (newer PaddleOCR versions)
        try:
            _ocr_engine = OriginalPaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False
            )
            return _ocr_engine
        except TypeError:
            # Older version doesn't support show_log parameter
            print("[OCR] PaddleOCR version doesn't support show_log, initializing without it...")
            pass
        
        # Fallback: initialize without show_log (older PaddleOCR versions)
        try:
            _ocr_engine = OriginalPaddleOCR(
                use_angle_cls=True,
                lang="en"
            )
            return _ocr_engine
        except Exception as e:
            print(f"[WARN] PaddleOCR fallback init also failed: {e}")
            return None
            
    except ImportError as e:
        print(f"[WARN] paddleocr or paddlepaddle is not installed: {e}")
        return None
    except Exception as e:
        print(f"[WARN] Failed to initialize PaddleOCR: {e}")
        return None

def extract_text_using_paddle(image_path: str) -> str:
    """
    Extracts text from an image or scanned document using PaddleOCR.
    
    Enhancements:
    - Applies image preprocessing before OCR
    - Sorts detected text by position (top-to-bottom, left-to-right)
    - Filters out low-confidence detections (< 0.6)
    - Reconstructs proper reading order for table layouts
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found at: {image_path}")

    engine = _get_paddle_ocr_engine()
    if engine is None:
        return "[WARN] PaddleOCR engine could not be loaded."

    try:
        # Step 1: Preprocess image for better OCR accuracy
        print("[OCR] Preprocessing image for PaddleOCR...")
        preprocessed_img = preprocess_image(str(path))
        
        # Convert PIL Image to numpy array for PaddleOCR
        import numpy as np
        img_array = np.array(preprocessed_img)
        
        # If image is grayscale (2D), convert to 3-channel for PaddleOCR
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Step 2: Run OCR on preprocessed image
        result = engine.ocr(img_array, cls=True)
        
        if not result or result[0] is None:
            # Fallback: try on original image
            print("[OCR] PaddleOCR returned no results on preprocessed image, trying original...")
            result = engine.ocr(str(path), cls=True)
            
        if not result or result[0] is None:
            return ""

        # Step 3: Extract text with confidence filtering and position sorting
        detections = []
        for line in result[0]:
            # result structure: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text_str, confidence_float)]
            coords = line[0]
            text_str = line[1][0]
            confidence = line[1][1]
            
            # Filter out low-confidence detections
            if confidence < 0.6:
                continue
            
            # Calculate center Y and min X for sorting (top-to-bottom, left-to-right)
            center_y = (coords[0][1] + coords[2][1]) / 2
            min_x = min(c[0] for c in coords)
            
            detections.append({
                "text": text_str,
                "confidence": confidence,
                "center_y": center_y,
                "min_x": min_x
            })
        
        if not detections:
            return ""
        
        # Step 4: Sort by position — group into rows, then sort within rows
        # Group detections into rows (similar Y coordinate within tolerance)
        detections.sort(key=lambda d: d["center_y"])
        
        rows = []
        current_row = [detections[0]]
        row_y = detections[0]["center_y"]
        
        # Y tolerance: detections within this pixel range are on the same row
        y_tolerance = 15
        
        for det in detections[1:]:
            if abs(det["center_y"] - row_y) <= y_tolerance:
                current_row.append(det)
            else:
                rows.append(current_row)
                current_row = [det]
                row_y = det["center_y"]
        rows.append(current_row)
        
        # Sort each row left-to-right, then join
        lines = []
        for row in rows:
            row.sort(key=lambda d: d["min_x"])
            row_text = "  ".join(d["text"] for d in row)
            lines.append(row_text)
            
        return "\n".join(lines).strip()
        
    except Exception as e:
        return f"[ERROR] PaddleOCR extraction failed: {e}"
