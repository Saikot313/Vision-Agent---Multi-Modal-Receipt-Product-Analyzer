"""The tool-calling agent loop.

Standard function-calling agent pattern:
1. Send the conversation + tool schemas to the LLM.
2. If the LLM responds with tool_calls, execute each one and append the
   result as a "tool" message.
3. Repeat until the LLM responds with plain text (no more tool_calls) — that
   is treated as the final answer.
4. Hard step cap so a confused LLM can't loop forever.
"""
import json
from dataclasses import dataclass, field

import numpy as np

from app.agent.llm_client import chat_completion
from app.schemas import ToolCallLog
from app.tools.registry import ALL_SCHEMAS, build_tool_dispatch, parse_tool_call_arguments

_SYSTEM_PROMPT = """You are a vision-and-language assistant for Bangladeshi \
retail document/image analysis. You have tools available to inspect an \
uploaded image (you cannot see the raw image yourself — you can only learn \
about it by calling tools).

Workflow you should generally follow:
1. Call classify_image first to find out whether the image is a 'receipt' \
   or a 'product' photo.
2. If it's a receipt: call ocr_receipt, then parse_line_items on the \
   resulting raw_text, then check_price_mismatch on the parsed items and \
   printed_total.
3. If it's a product photo: call detect_objects.
4. Once you have enough information, respond with a final natural-language \
   summary (no more tool calls). For receipts, explicitly mention whether a \
   price mismatch was found. Write the summary in Bangla, then a short \
   English translation.

Only call a tool once per purpose — don't repeat a tool call with the same \
arguments.
"""

_MAX_STEPS = 8


@dataclass
class AgentRunResult:
    final_answer: str
    trace: list[ToolCallLog] = field(default_factory=list)
    tool_outputs: dict = field(default_factory=dict)  # last output per tool name


def run_agent(image: np.ndarray) -> AgentRunResult:
    dispatch = build_tool_dispatch(image)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "An image has been uploaded. Analyze it using the available tools.",
        },
    ]

    trace: list[ToolCallLog] = []
    tool_outputs: dict = {}

    for _ in range(_MAX_STEPS):
        response = chat_completion(messages=messages, tools=ALL_SCHEMAS)
        message = response.choices[0].message

        if not message.tool_calls:
            # No more tool calls -> this is the final natural-language answer.
            answer = message.content or ""
            if not answer.strip():
                # The LLM finished without calling more tools but also
                # returned empty content (a real Groq/tool-calling edge
                # case). Rather than showing a blank result, fall back to a
                # summary built from whatever tool outputs we already have.
                answer = _fallback_summary(tool_outputs)
            return AgentRunResult(final_answer=answer, trace=trace, tool_outputs=tool_outputs)

        # Append the assistant's tool-call message, then execute each call.
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = parse_tool_call_arguments(tool_call.function.arguments)

            if name not in dispatch:
                result = {"error": f"unknown tool '{name}'"}
            else:
                try:
                    result = dispatch[name](args)
                except Exception as e:  # noqa: BLE001 — surface any tool failure to the LLM
                    result = {"error": str(e)}

            tool_outputs[name] = result

            trace.append(
                ToolCallLog(
                    tool=name,
                    arguments=args,
                    result_preview=json.dumps(result, ensure_ascii=False)[:300],
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    # Hit the step cap without a final answer — surface what we have rather
    # than silently failing.
    return AgentRunResult(
        final_answer="(Agent stopped after reaching the maximum number of steps.)",
        trace=trace,
        tool_outputs=tool_outputs,
    )


def _fallback_summary(tool_outputs: dict) -> str:
    """Best-effort plain-text summary built from raw tool outputs, used only
    when the LLM itself returns empty final content."""
    if not tool_outputs:
        return "The agent didn't produce a final answer and no tool results were captured — check GROQ_API_KEY and the agent trace."

    parts = ["(Auto-generated from tool results — the LLM returned an empty final answer.)"]
    if "classify_image" in tool_outputs:
        parts.append(f"Image type: {tool_outputs['classify_image'].get('image_type', 'unknown')}")
    if "ocr_receipt" in tool_outputs:
        chars = tool_outputs["ocr_receipt"].get("char_count", 0)
        parts.append(f"OCR extracted {chars} characters.")
    if "parse_line_items" in tool_outputs:
        n = len(tool_outputs["parse_line_items"].get("items", []))
        parts.append(f"Parsed {n} line item(s).")
        if tool_outputs["parse_line_items"].get("parse_error"):
            parts.append("Warning: line-item parsing failed to produce valid JSON.")
    if "check_price_mismatch" in tool_outputs:
        parts.append(tool_outputs["check_price_mismatch"].get("note", ""))
    if "detect_objects" in tool_outputs:
        labels = [d["label"] for d in tool_outputs["detect_objects"].get("detections", [])]
        parts.append(f"Detected objects: {', '.join(labels) if labels else 'none'}")
    return "\n".join(p for p in parts if p)
