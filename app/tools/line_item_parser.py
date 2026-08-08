"""Line-item parser: turns messy OCR text into structured {name, qty, price} rows.

This is itself an LLM call (structured extraction is exactly where an LLM
beats regex on noisy OCR output), but it's wrapped as a *tool* the main agent
calls — the main agent doesn't do free-form parsing itself, it delegates.
This keeps the main agent's job simple ("orchestrate") and this tool's job
narrow ("extract"), which is easier to test and debug independently.
"""
import json

from app.agent.llm_client import simple_generate

_SYSTEM_PROMPT = """You extract structured line items from noisy OCR text of \
a Bangladeshi retail receipt (text may be in Bangla, English, or a mix, and \
may contain OCR errors — use judgement to fix obvious digit/character \
mis-reads).

Respond with ONLY a JSON object:
{
  "items": [
    {"name": "...", "quantity": 1, "unit_price": 0.0, "line_total": 0.0}
  ],
  "printed_total": 0.0 or null
}

Rules:
- quantity defaults to 1 if not stated.
- unit_price and line_total may be null if genuinely unreadable — don't guess.
- printed_total is whatever the receipt states as its own grand total \
  (look for words like "Total", "মোট", "সর্বমোট"). null if not found.
- Keep item names as they appear (don't translate them).
"""


def parse_line_items(raw_text: str) -> dict:
    if not raw_text.strip():
        return {"items": [], "printed_total": None}

    raw = simple_generate(_SYSTEM_PROMPT, raw_text, temperature=0.0)
    cleaned = _strip_code_fence(raw)
    try:
        parsed = json.loads(cleaned)
        return {
            "items": parsed.get("items", []),
            "printed_total": parsed.get("printed_total"),
        }
    except (json.JSONDecodeError, AttributeError):
        # Surface the raw LLM output rather than silently returning an empty
        # result — makes this failure mode visible in the agent trace instead
        # of looking like "nothing happened".
        return {
            "items": [],
            "printed_total": None,
            "parse_error": True,
            "raw_llm_output": (raw or "")[:500],
        }


def _strip_code_fence(text: str | None) -> str:
    """LLMs frequently wrap JSON in ```json ... ``` even when told not to.
    Strip that before parsing instead of failing on it."""
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1] if stripped.count("```") >= 2 else stripped.strip("`")
        stripped = stripped.removeprefix("json").strip() if stripped.lower().startswith("json") else stripped
    return stripped.strip()


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "parse_line_items",
        "description": (
            "Parse raw OCR text from a receipt into structured line items "
            "(name, quantity, unit price, line total) and the printed grand "
            "total. Call this after ocr_receipt has produced raw_text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raw_text": {
                    "type": "string",
                    "description": "The raw OCR text (from ocr_receipt's output) to parse.",
                }
            },
            "required": ["raw_text"],
        },
    },
}
