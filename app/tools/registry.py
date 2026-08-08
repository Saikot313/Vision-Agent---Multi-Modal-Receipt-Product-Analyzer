"""Tool registry.

Image-taking tools (classify_image, ocr_receipt, detect_objects) don't take
the image as an LLM-supplied argument — the LLM can't produce raw pixels.
Instead, `build_tool_dispatch(image)` closes over the current image so the
LLM only ever has to say "call classify_image" with no args, while the tool
still gets the right image. Tools that operate on text/structured data
(parse_line_items, check_price_mismatch) DO take LLM-supplied arguments
normally, since the LLM has already seen that data in the conversation.
"""
import json

import numpy as np

from app.tools.classify_image import TOOL_SCHEMA as CLASSIFY_SCHEMA
from app.tools.classify_image import classify_image
from app.tools.line_item_parser import TOOL_SCHEMA as PARSE_SCHEMA
from app.tools.line_item_parser import parse_line_items
from app.tools.object_detection import TOOL_SCHEMA as DETECT_SCHEMA
from app.tools.object_detection import detect_objects
from app.tools.ocr_tool import TOOL_SCHEMA as OCR_SCHEMA
from app.tools.ocr_tool import ocr_receipt
from app.tools.price_check_tool import TOOL_SCHEMA as PRICE_CHECK_SCHEMA
from app.tools.price_check_tool import check_price_mismatch

ALL_SCHEMAS = [CLASSIFY_SCHEMA, OCR_SCHEMA, PARSE_SCHEMA, PRICE_CHECK_SCHEMA, DETECT_SCHEMA]


def build_tool_dispatch(image: np.ndarray) -> dict:
    """Returns {tool_name: callable(arguments_dict) -> dict} bound to `image`."""

    return {
        "classify_image": lambda args: classify_image(image),
        "ocr_receipt": lambda args: ocr_receipt(image),
        "detect_objects": lambda args: detect_objects(image),
        "parse_line_items": lambda args: parse_line_items(args.get("raw_text", "")),
        "check_price_mismatch": lambda args: check_price_mismatch(
            args.get("items", []), args.get("printed_total")
        ),
    }


def parse_tool_call_arguments(raw_arguments: str) -> dict:
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    # Some models emit the literal string "null" (valid JSON) for tools that
    # take no arguments, instead of "{}" — json.loads("null") is None, not a
    # dict, which then fails downstream (ToolCallLog requires a dict, and
    # args.get(...) calls would break too). Normalize any non-dict result.
    return parsed if isinstance(parsed, dict) else {}
