"""Object detection tool using pretrained YOLOv8 (COCO classes).

Used for the "plain product photo" path, as opposed to the receipt/OCR path.
Model weights auto-download from Ultralytics' GitHub releases on first use.
"""
from functools import lru_cache

import numpy as np

from app.config import settings


@lru_cache(maxsize=1)
def _get_model():
    from ultralytics import YOLO  # local import: keeps torch out of the

    # import path for users who only use the OCR/receipt features.
    return YOLO(settings.yolo_model)


def detect_objects(image: np.ndarray, confidence: float = 0.35) -> dict:
    model = _get_model()
    results = model.predict(image, conf=confidence, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"label": label, "confidence": round(conf, 3)})

    return {"detections": detections, "count": len(detections)}


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_objects",
        "description": (
            "Run object detection (YOLOv8, pretrained on COCO) on an "
            "uploaded image to identify physical objects/products in it. "
            "Call this when classify_image says the image is a 'product' "
            "photo, not a receipt."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}
