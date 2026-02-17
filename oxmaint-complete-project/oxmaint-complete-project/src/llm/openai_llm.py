#!/usr/bin/env python3
"""
OpenAI LLM helper (optional).

- Generates a short explanation string.
- Does NOT affect predictions.
- If OPENAI_API_KEY is missing or request fails, returns None.

Security note:
- Prefer environment variable OPENAI_API_KEY.
- You *can* hardcode as a fallback, but DO NOT commit that to GitHub.
"""

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
        # OpenAI SDK (per OpenAI docs style)
        from openai import OpenAI  # pip install openai
    except Exception as exc:
        logger.warning(f"OpenAI SDK not installed. Run: pip install openai. Error: {exc}")
        return None

    # Prefer env var. Optional explicit api_key arg overrides.
    key = api_key or os.getenv("OPENAI_API_KEY")

    # OPTIONAL fallback (not recommended):
    # key = key or "PASTE_YOUR_KEY_HERE"

    if not key:
        logger.warning("OPENAI_API_KEY not set, skipping LLM explanation.")
        return None

    try:
        client = OpenAI(api_key=key)

        # Simple chat completion
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
