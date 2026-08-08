"""Lightweight receipt-vs-product-photo classifier.

Doesn't need a trained model: receipts/documents have a very distinctive
signature under OpenCV — mostly white/near-white background with a high
density of small dark text strokes, and usually a portrait aspect ratio.
Product photos have much more color variance and lower edge density.

This is a heuristic, not a CNN classifier, which is a deliberate and easy-to
-explain design choice for a demo (fast, no training data needed). Swapping
in a small trained CNN later is a natural extension.
"""
import cv2
import numpy as np

from app.core.image_preprocessing import to_grayscale


def classify_image(image: np.ndarray) -> dict:
    gray = to_grayscale(image)
    h, w = gray.shape[:2]

    # 1. Colorfulness: receipts are near-grayscale, product photos are colorful
    if len(image.shape) == 3:
        b, g, r = cv2.split(image.astype("float32"))
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        colorfulness = float(np.std(rg) + np.std(yb))
    else:
        colorfulness = 0.0

    # 2. Edge density: printed text produces lots of small edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / (h * w)

    # 3. Background brightness: receipts are usually on white/light background
    mean_brightness = float(np.mean(gray))

    is_receipt = colorfulness < 25 and edge_density > 0.02 and mean_brightness > 120

    return {
        "image_type": "receipt" if is_receipt else "product",
        "colorfulness": round(colorfulness, 2),
        "edge_density": round(edge_density, 4),
        "mean_brightness": round(mean_brightness, 1),
    }


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "classify_image",
        "description": (
            "Classify an uploaded image as either a 'receipt' (document with "
            "printed text, e.g. invoice/receipt) or a 'product' photo (a "
            "physical item). Call this FIRST on any new image, before "
            "deciding whether to run OCR or object detection."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
