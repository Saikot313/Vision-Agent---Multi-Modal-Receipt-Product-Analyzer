"""Price mismatch check — deliberately plain Python, not an LLM call.

Arithmetic is exactly the kind of thing LLMs get subtly wrong on longer
receipts, and a wrong "no mismatch found" from the LLM would defeat the
purpose of the whole feature. So this tool sums the parsed line items in
Python and compares to the printed total; the LLM only narrates the result.
"""

_TOLERANCE = 1.0  # taka — allow small rounding/OCR noise before flagging


def check_price_mismatch(items: list[dict], printed_total: float | None) -> dict:
    computed_total = 0.0
    for item in items:
        line_total = item.get("line_total")
        if line_total is not None:
            computed_total += float(line_total)
        else:
            qty = float(item.get("quantity", 1) or 1)
            unit_price = item.get("unit_price")
            if unit_price is not None:
                computed_total += qty * float(unit_price)

    computed_total = round(computed_total, 2)

    if printed_total is None:
        return {
            "computed_total": computed_total,
            "printed_total": None,
            "mismatch": False,
            "difference": None,
            "note": "No printed total found on the receipt to compare against.",
        }

    difference = round(computed_total - float(printed_total), 2)
    mismatch = abs(difference) > _TOLERANCE

    note = (
        f"Line items sum to {computed_total}, printed total is {printed_total} "
        f"— {'MISMATCH of ' + str(abs(difference)) if mismatch else 'matches within tolerance'}."
    )

    return {
        "computed_total": computed_total,
        "printed_total": float(printed_total),
        "mismatch": mismatch,
        "difference": difference,
        "note": note,
    }


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_price_mismatch",
        "description": (
            "Given parsed line items and the printed grand total from a "
            "receipt, deterministically compute whether the numbers add up. "
            "Always call this after parse_line_items when checking a "
            "receipt for billing errors — do not do this arithmetic "
            "yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "List of line items with quantity/unit_price/line_total.",
                    "items": {"type": "object"},
                },
                "printed_total": {
                    "type": ["number", "null"],
                    "description": "The grand total printed on the receipt, or null if none found.",
                },
            },
            "required": ["items", "printed_total"],
        },
    },
}
