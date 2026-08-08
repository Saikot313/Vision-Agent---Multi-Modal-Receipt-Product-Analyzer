"""OpenCV preprocessing helpers to make phone-photo receipts more OCR-friendly.

Kept as small, independently-testable functions so they can be unit tested
without needing Tesseract installed.
"""
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def deskew(image: np.ndarray) -> np.ndarray:
    """Estimate and correct small rotation angles (typical of phone photos)."""
    gray = to_grayscale(image)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 10:
        return image  # not enough signal to estimate an angle reliably

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip correction for negligible angles — avoids introducing blur on
    # already-straight images.
    if abs(angle) < 0.5:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    gray = to_grayscale(image)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return thresh


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Full pipeline: deskew -> adaptive threshold. Returns a binarized image
    ready for Tesseract."""
    deskewed = deskew(image)
    return adaptive_threshold(deskewed)
