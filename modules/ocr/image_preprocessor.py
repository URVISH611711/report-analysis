"""
Image Pre-processing Module for Medical Report OCR.

Applies a series of enhancements to improve OCR accuracy on real-world
medical report images (phone photos, WhatsApp images, skewed scans).

Steps: Grayscale → Deskew → CLAHE contrast → Adaptive threshold →
       Noise reduction → Upscale if needed → Shadow removal.
"""

import numpy as np
from PIL import Image
from pathlib import Path

# Lazy-load OpenCV (heavy import)
_cv2 = None

def _get_cv2():
    global _cv2
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
        except ImportError:
            print("[WARN] opencv-python is not installed. Image preprocessing will be skipped.")
            print("[WARN] Install with: pip install opencv-python>=4.8.0")
            return None
    return _cv2


def preprocess_image(image_path: str, target_dpi: int = 300) -> Image.Image:
    """
    Applies full preprocessing pipeline to a medical report image.

    Args:
        image_path: Path to the input image file.
        target_dpi: Minimum effective DPI to upscale to.

    Returns:
        PIL.Image.Image: The preprocessed image ready for OCR.
    """
    cv2 = _get_cv2()
    if cv2 is None:
        # Fallback: return original image without enhancement
        return Image.open(image_path)

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read image with OpenCV
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image file: {image_path}")

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Upscale if image is too small (low resolution phone photos)
    height, width = gray.shape[:2]
    # Assume standard A4 report width ~210mm; if image width < 1500px, upscale
    min_width = 1500
    if width < min_width:
        scale_factor = min_width / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"[PREPROCESS] Upscaled image from {width}x{height} to {new_width}x{new_height}")

    # Step 3: Deskew using Hough line transform
    gray = _deskew(gray, cv2)

    # Step 4: Shadow removal (for photos taken on tables/floors)
    gray = _remove_shadows(gray, cv2)

    # Step 5: CLAHE contrast enhancement (handles uneven lighting in WhatsApp photos)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Step 6: Noise reduction with bilateral filter (preserves edges better than Gaussian)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Step 7: Adaptive thresholding for binarization
    # This handles uneven lighting much better than global thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=10
    )

    # Step 8: Morphological operations to clean up noise and connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Convert back to PIL Image for OCR engines
    pil_image = Image.fromarray(binary)
    return pil_image


def _deskew(image: np.ndarray, cv2) -> np.ndarray:
    """
    Automatically detect and correct skew/rotation in the image.
    Uses Hough Line Transform to find dominant text line angles.
    """
    try:
        # Edge detection for line finding
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return image

        # Calculate angles of all detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines (within ±30 degrees)
            if abs(angle) < 30:
                angles.append(angle)

        if not angles:
            return image

        # Use median angle (robust to outliers)
        median_angle = np.median(angles)

        # Only correct if skew is significant (> 0.5 degrees) but not extreme (< 15 degrees)
        if abs(median_angle) < 0.5 or abs(median_angle) > 15:
            return image

        print(f"[PREPROCESS] Detected skew of {median_angle:.1f}°, correcting...")

        # Rotate image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    except Exception as e:
        print(f"[PREPROCESS] Deskew failed (non-critical): {e}")
        return image


def _remove_shadows(image: np.ndarray, cv2) -> np.ndarray:
    """
    Removes shadows from images taken on tables/floors/uneven surfaces.
    Uses morphological operations to estimate and subtract the background.
    """
    try:
        # Dilate to create a background model (large kernel fills in text)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        background = cv2.dilate(image, kernel)
        background = cv2.medianBlur(background, 21)

        # Subtract background to normalize illumination
        # This removes shadows while preserving text
        normalized = 255 - cv2.absdiff(image, background)

        # Normalize the result to full 0-255 range
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

        return normalized.astype(np.uint8)

    except Exception as e:
        print(f"[PREPROCESS] Shadow removal failed (non-critical): {e}")
        return image


def get_image_quality_score(image_path: str) -> float:
    """
    Returns a quality score (0.0 to 1.0) estimating how suitable an image is for OCR.
    Based on contrast, sharpness, and resolution.
    """
    cv2 = _get_cv2()
    if cv2 is None:
        return 0.5  # Unknown quality

    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0

        height, width = img.shape[:2]

        # Resolution score (higher is better, cap at 1.0)
        resolution_score = min(1.0, (width * height) / (1500 * 2000))

        # Contrast score (standard deviation of pixel values)
        contrast_score = min(1.0, np.std(img) / 80.0)

        # Sharpness score (Laplacian variance)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500.0)

        # Weighted average
        quality = 0.3 * resolution_score + 0.3 * contrast_score + 0.4 * sharpness_score
        return round(quality, 3)

    except Exception:
        return 0.5
