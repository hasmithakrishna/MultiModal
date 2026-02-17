#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Optional
from loguru import logger
def generate_text(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 180,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> Optional[str]:
    try:
        from openai import OpenAI  
    except Exception as exc:
        logger.warning(f"OpenAI SDK not installed. Run: pip install openai. Error: {exc}")
        return None
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        logger.warning("OPENAI_API_KEY not set, skipping LLM explanation.")
        return None
    try:
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for industrial pump maintenance."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text if text else None

    except Exception as exc:
        logger.warning(f"OpenAI call failed, skipping LLM explanation: {exc}")
        return None

