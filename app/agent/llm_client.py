"""Groq LLM client with OpenAI-style tool/function calling.

Groq's API is OpenAI-compatible, so this uses the same `tools=[...]` /
`tool_calls` interface you'd use with OpenAI's function calling — makes it
easy to swap providers later if needed.
"""
from functools import lru_cache

from groq import Groq

from app.config import settings


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and put it in your .env file."
        )
    return Groq(api_key=settings.groq_api_key)


def chat_completion(messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.1):
    """Thin wrapper around Groq's chat.completions.create, kept in one place
    so the rest of the app doesn't need to know which SDK is being used."""
    client = get_groq_client()
    kwargs = {"model": settings.groq_model, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


def simple_generate(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """For LLM sub-calls that don't need tool-calling (e.g. line-item parsing,
    final summary rewriting)."""
    response = chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content
