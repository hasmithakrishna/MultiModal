#!/usr/bin/env python3
from __future__ import annotations
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from loguru import logger
from src.orchestrator.inference_orchestrator import InferenceOrchestrator
class TextDoc(BaseModel):
    content: str = Field(..., min_length=1, description="Free text from manual/work order/request")
class PredictRequest(BaseModel):
    asset_id: str = Field(..., min_length=1)
    sensor_window: List[List[float]] = Field(
        ...,
        description="Sensor window shaped [T][F] (e.g., 50x52). Sensor modality is required.",
    )
    text_docs: Optional[List[TextDoc]] = Field(default=None)
    image_path: Optional[str] = Field(default=None, description="Path to image file")
    image_data: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional raw grayscale image array HxW (values 0..255 or 0..1)",
    )
    include_llm_explanation: Optional[bool] = Field(
        default=False,
        description="If true, adds llm_explanation (OpenAI) to the response. Does not affect prediction.",
    )
    @field_validator("sensor_window")
    @classmethod
    def validate_sensor_window(cls, v: List[List[float]]):
        if not isinstance(v, list) or len(v) < 2:
            raise ValueError("sensor_window must be a 2D list with shape [T][F]")

        row_lens = [len(r) for r in v if isinstance(r, list)]
        if len(row_lens) != len(v):
            raise ValueError("sensor_window must be a 2D list of numeric rows")

        if min(row_lens) == 0:
            raise ValueError("sensor_window rows cannot be empty")

        if len(set(row_lens)) != 1:
            raise ValueError("sensor_window must have consistent feature dimension per timestep")
        return v
    @field_validator("image_data")
    @classmethod
    def validate_image_data(cls, v):
        if v is None:
            return v
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("image_data must be a 2D list (HxW)")
        row_lens = [len(r) for r in v if isinstance(r, list)]
        if len(row_lens) != len(v) or min(row_lens) == 0:
            raise ValueError("image_data must be a 2D list (HxW)")
        if len(set(row_lens)) != 1:
            raise ValueError("image_data rows must have consistent width")
        return v
class BatchPredictRequest(BaseModel):
    items: List[PredictRequest] = Field(..., min_length=1)
app = FastAPI(title="Oxmaint Predictive Agent API", version="1.0")

BASE_DIR = Path(__file__).parent.parent.parent
ORCH = InferenceOrchestrator(BASE_DIR / "models")
ORCH.load_models()
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": bool(getattr(ORCH, "models_loaded", False)),
    }
def _safe_prepare_input(req: PredictRequest) -> Dict[str, Any]:
    inp: Dict[str, Any] = {
        "asset_id": req.asset_id,
        "sensor_window": req.sensor_window,
        "include_llm_explanation": bool(req.include_llm_explanation),
    }
    if req.text_docs is not None and len(req.text_docs) > 0:
        inp["text_docs"] = [{"content": d.content} for d in req.text_docs]
    if req.image_path is not None and req.image_path.strip() != "":
        inp["image_path"] = req.image_path.strip()
    elif req.image_data is not None:
        inp["image_data"] = np.array(req.image_data, dtype=np.float32)

    return inp
@app.post("/predict")
def predict(req: PredictRequest):
    if not getattr(ORCH, "models_loaded", False):
        raise HTTPException(status_code=503, detail="Models are not loaded")

    t0 = time.perf_counter()
    try:
        inp = _safe_prepare_input(req)
        out = ORCH.predict(inp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    t1 = time.perf_counter()

    out["inference_time_ms"] = round((t1 - t0) * 1000.0, 3)
    return out
@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    if not getattr(ORCH, "models_loaded", False):
        raise HTTPException(status_code=503, detail="Models are not loaded")
    t0 = time.perf_counter()
    results = []
    for i, item in enumerate(req.items):
        try:
            inp = _safe_prepare_input(item)
            out = ORCH.predict(inp)
            results.append({"index": i, "ok": True, "result": out})
        except Exception as e:
            results.append({"index": i, "ok": False, "error": str(e)})

    t1 = time.perf_counter()
    return {
        "count": len(req.items),
        "inference_time_ms": round((t1 - t0) * 1000.0, 3),
        "results": results,
    }
