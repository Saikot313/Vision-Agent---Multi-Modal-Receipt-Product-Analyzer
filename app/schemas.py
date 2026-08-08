from typing import Optional

from pydantic import BaseModel


class LineItem(BaseModel):
    name: str
    quantity: float = 1
    unit_price: Optional[float] = None
    line_total: Optional[float] = None


class PriceCheckResult(BaseModel):
    computed_total: float
    printed_total: Optional[float]
    mismatch: bool
    difference: Optional[float]
    note: str


class ToolCallLog(BaseModel):
    tool: str
    arguments: dict
    result_preview: str


class AnalyzeResponse(BaseModel):
    image_type: str
    ocr_text: Optional[str] = None
    line_items: list[LineItem] = []
    price_check: Optional[PriceCheckResult] = None
    detected_objects: list[str] = []
    summary: str
    agent_trace: list[ToolCallLog]
