#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Optional
from loguru import logger
def _ollama_generate(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    host: str,
    timeout_s: int = 30,
) -> Optional[str]:
    try:
        import requests  
    except Exception as exc:
        logger.warning(f"requests not installed. Run: pip install requests. Error: {exc}")
        return None
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            logger.warning(f"Ollama call failed: HTTP {r.status_code} - {r.text[:200]}")
            return None

        data = r.json()
        text = (data.get("response") or "").strip()
        return text if text else None

    except Exception as exc:
        logger.warning(f"Ollama call failed: {exc}")
        return None
def generate_text(
    prompt: str,
    model: str = "llama3.1:8b",
    max_tokens: int = 180,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
) -> Optional[str]:
    backend = (os.getenv("LOCAL_LLM_BACKEND") or "").strip().lower()
    if backend in ("", "none", "disabled", "off"):
        return None

    if backend == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        env_model = os.getenv("OLLAMA_MODEL")
        use_model = env_model.strip() if env_model else model
        final_prompt = (
            "You are a helpful assistant for industrial pump maintenance.\n\n"
            + prompt.strip()
        )
        return _ollama_generate(
            prompt=final_prompt,
            model=use_model,
            max_tokens=max_tokens,
            temperature=temperature,
            host=host,
        )
    logger.warning(f"Unknown LOCAL_LLM_BACKEND='{backend}'. Supported: ollama. Returning None.")
    return None
