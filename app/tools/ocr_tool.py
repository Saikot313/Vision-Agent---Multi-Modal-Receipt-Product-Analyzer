"""OCR tool: OpenCV preprocessing + Tesseract, bilingual (Bangla + English).

Requires the system tesseract-ocr binary + the Bangla language pack:
    Linux:   sudo apt-get install tesseract-ocr tesseract-ocr-ben
    Windows: install from https://github.com/UB-Mannheim/tesseract/wiki,
             then set TESSERACT_CMD_PATH in .env if it's not on PATH
             (e.g. TESSERACT_CMD_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe)
"""
import pytesseract

from app.config import settings
from app.core.image_preprocessing import preprocess_for_ocr

if settings.tesseract_cmd_path:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd_path


def ocr_receipt(image) -> dict:
    processed = preprocess_for_ocr(image)
    try:
        text = pytesseract.image_to_string(processed, lang=settings.tesseract_lang)
    except pytesseract.TesseractNotFoundError:
        # The tesseract-ocr system binary isn't installed (common when running
        # locally without Docker). Surface a clear, actionable message instead
        # of a generic tool error the LLM has no way to explain to the user.
        return {
            "error": (
                "Tesseract OCR is not installed on this machine. Install it with "
                "'sudo apt-get install tesseract-ocr tesseract-ocr-ben' (Linux) or "
                "'brew install tesseract tesseract-lang' (Mac), or run this project "
                "via Docker where it's preinstalled."
            )
        }
    except pytesseract.TesseractError as e:
        # Common cause: the "ben" language pack isn't installed. Fall back to
        # English-only rather than hard-failing the whole pipeline.
        if "ben" in str(e).lower() or "failed loading language" in str(e).lower():
            text = pytesseract.image_to_string(processed, lang="eng")
        else:
            raise

    return {"raw_text": text.strip(), "char_count": len(text.strip())}


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ocr_receipt",
        "description": (
            "Run OCR (optical character recognition) on the uploaded receipt/"
            "invoice image to extract raw text, in Bangla and/or English. "
            "Call this after classify_image confirms the image is a "
            "'receipt'."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}