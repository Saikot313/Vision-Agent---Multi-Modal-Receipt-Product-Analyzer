import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.agent.tool_agent import run_agent
from app.schemas import AnalyzeResponse, LineItem, PriceCheckResult
from app.tools.registry import build_tool_dispatch  # noqa: F401 (re-export convenience)

app = FastAPI(
    title="Multi-Modal Vision + LLM Agent",
    description="Tool-using LLM agent combining OCR/YOLO with reasoning for receipt/product analysis.",
    version="0.1.0",
)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...)):
    contents = await image.read()
    np_arr = np.frombuffer(contents, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if cv_image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    result = run_agent(cv_image)

    classify_out = result.tool_outputs.get("classify_image", {})
    ocr_out = result.tool_outputs.get("ocr_receipt", {})
    parse_out = result.tool_outputs.get("parse_line_items", {})
    price_out = result.tool_outputs.get("check_price_mismatch", {})
    detect_out = result.tool_outputs.get("detect_objects", {})

    line_items = [LineItem(**item) for item in parse_out.get("items", [])] if parse_out else []
    price_check = PriceCheckResult(**price_out) if price_out and "computed_total" in price_out else None
    detected_objects = [d["label"] for d in detect_out.get("detections", [])] if detect_out else []

    return AnalyzeResponse(
        image_type=classify_out.get("image_type", "unknown"),
        ocr_text=ocr_out.get("raw_text"),
        line_items=line_items,
        price_check=price_check,
        detected_objects=detected_objects,
        summary=result.final_answer,
        agent_trace=result.trace,
    )
