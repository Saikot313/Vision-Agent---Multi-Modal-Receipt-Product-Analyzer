"""Tests for the deterministic/CV parts that don't need a Groq API key or
downloaded models — safe to run in CI.

Run with: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.image_preprocessing import adaptive_threshold, deskew, to_grayscale
from app.tools.price_check_tool import check_price_mismatch


def _blank_image(h=200, w=200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_to_grayscale_shape():
    img = _blank_image()
    gray = to_grayscale(img)
    assert gray.shape == (200, 200)


def test_deskew_returns_same_shape():
    img = _blank_image()
    out = deskew(img)
    assert out.shape[:2] == img.shape[:2]


def test_adaptive_threshold_binary_output():
    img = _blank_image()
    out = adaptive_threshold(img)
    unique_values = set(np.unique(out).tolist())
    assert unique_values.issubset({0, 255})


def test_price_mismatch_detected():
    items = [
        {"name": "Rice 5kg", "quantity": 1, "line_total": 450.0},
        {"name": "Oil 1L", "quantity": 2, "unit_price": 180.0},
    ]
    result = check_price_mismatch(items, printed_total=900.0)
    assert result["computed_total"] == 810.0
    assert result["mismatch"] is True
    assert result["difference"] == -90.0


def test_price_mismatch_within_tolerance():
    items = [{"name": "Item A", "quantity": 1, "line_total": 100.0}]
    result = check_price_mismatch(items, printed_total=100.5)
    assert result["mismatch"] is False


def test_price_check_no_printed_total():
    items = [{"name": "Item A", "quantity": 1, "line_total": 50.0}]
    result = check_price_mismatch(items, printed_total=None)
    assert result["printed_total"] is None
    assert result["mismatch"] is False
