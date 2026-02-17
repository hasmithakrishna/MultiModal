#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, List

import cv2
import numpy as np
import torch
from loguru import logger


class InferenceOrchestrator:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)

        self.sensor_model = None
        self.image_model = None

        self.text_clf = None
        self.text_vectorizer = None

        self.models_loaded = False

        self.fault_types = {0: "normal", 1: "abnormal"}

        self.risk_threshold = 0.5

    def load_models(self):
        logger.info("Loading models...")

        try:
            sensor_model_path = self.model_dir / "sensor_model" / "sensor_model.pth"
            if sensor_model_path.exists():
                checkpoint = torch.load(sensor_model_path, map_location="cpu")

                import sys
                base_dir = self.model_dir.parent
                src_dir = base_dir / "src"
                if str(src_dir) not in sys.path:
                    sys.path.append(str(src_dir))

                from modeling.train_sensor_model import SmallLSTM

                self.sensor_model = SmallLSTM(
                    input_size=int(checkpoint["input_size"]),
                    hidden_size=int(checkpoint.get("hidden_size", 32)),
                    num_layers=1,
                    dropout=float(checkpoint.get("dropout", 0.2)),
                )
                self.sensor_model.load_state_dict(checkpoint["model_state_dict"])
                self.sensor_model.eval()
                logger.success("✓ Sensor model loaded")
            else:
                logger.error("Sensor model not found (sensor is mandatory).")
                self.sensor_model = None

            image_model_path = self.model_dir / "image_model" / "image_model.pth"
            if image_model_path.exists():
                checkpoint = torch.load(image_model_path, map_location="cpu")

                import sys
                base_dir = self.model_dir.parent
                src_dir = base_dir / "src"
                if str(src_dir) not in sys.path:
                    sys.path.append(str(src_dir))

                from modeling.train_image_model import DefectCNN

                self.image_model = DefectCNN(
                    num_classes=int(checkpoint["num_classes"]),
                    dropout=float(checkpoint.get("dropout", 0.5)),
                )
                self.image_model.load_state_dict(checkpoint["model_state_dict"])
                self.image_model.eval()
                logger.success("✓ Image model loaded")
            else:
                logger.warning("Image model not found — skipping image predictions.")
                self.image_model = None

            text_dir = self.model_dir / "text_model"
            clf_path = text_dir / "text_risk_clf.pkl"
            vec_path = text_dir / "text_risk_vectorizer.pkl"

            if clf_path.exists() and vec_path.exists():
                with open(clf_path, "rb") as f:
                    self.text_clf = pickle.load(f)
                with open(vec_path, "rb") as f:
                    self.text_vectorizer = pickle.load(f)
                logger.success("✓ Text risk model loaded")
            else:
                logger.warning("Text model not found — skipping text predictions.")
                self.text_clf = None
                self.text_vectorizer = None

            self.models_loaded = self.sensor_model is not None
            if not self.models_loaded:
                logger.error("Models not loaded: sensor model is required.")
        except Exception as exc:
            logger.error(f"Model loading failed: {exc}")
            self.models_loaded = False

    def _safe_softmax(self, outputs: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(outputs, dim=1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return probs / row_sum

    def _risk_from_prob(self, p_abnormal: float, predicted_class: int) -> float:
        p = float(np.clip(np.nan_to_num(p_abnormal), 0.0, 1.0))
        thr = float(np.clip(self.risk_threshold, 0.0, 0.99))
        risk = (p - thr) / (1.0 - thr)
        if predicted_class == 0:
            risk *= 0.4
        return float(np.clip(risk, 0.0, 1.0))

    def _estimate_ttb_hours(self, failure_probability: float) -> float:
        fp = float(np.clip(failure_probability, 0.0, 1.0))
        if fp < 0.2:
            return 1200.0
        if fp < 0.4:
            return 400.0
        if fp < 0.6:
            return 120.0
        if fp < 0.8:
            return 48.0
        return 12.0

    def generate_llm_explanation(
        self,
        failure_probability: float,
        fault_type: str,
        top_signals: List[str],
        sensor_analysis: Dict[str, Any],
        image_analysis: Optional[Dict[str, Any]],
        text_analysis: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        try:
            from llm.openai_llm import generate_text
        except Exception as exc:
            logger.warning(f"LLM module not available: {exc}")
            return None

        prompt = f"""
Explain the prediction in 2-4 short bullet points using simple language.

- failure_probability: {failure_probability:.3f}
- predicted_fault_type: {fault_type}
- top_signals: {top_signals}

Sensor analysis: {sensor_analysis}
Image analysis: {image_analysis}
Text analysis: {text_analysis}

Rules:
- Only bullet points.
- Don’t invent missing info.
- If a modality is missing, mention it is missing.
""".strip()

        return generate_text(prompt)

    def predict_sensor(self, sensor_window) -> Optional[Dict[str, Any]]:
        if self.sensor_model is None:
            return None
        try:
            x = np.nan_to_num(np.array(sensor_window, dtype=np.float32))
            if x.ndim == 2:
                x = x[np.newaxis, :, :]
            tensor = torch.tensor(x, dtype=torch.float32)

            with torch.no_grad():
                outputs = self.sensor_model(tensor)
                probs = self._safe_softmax(outputs)
                cls = int(torch.argmax(probs, dim=1).item())
                conf = float(probs[0, cls].item())

            return {
                "predicted_class": cls,
                "confidence": float(np.clip(conf, 0.0, 1.0)),
                "probabilities": probs[0].cpu().numpy().tolist(),
            }
        except Exception as exc:
            logger.error(f"Sensor prediction error: {exc}")
            return None

    def predict_image(self, image_path_or_array) -> Optional[Dict[str, Any]]:
        if self.image_model is None:
            return None
        try:
            if isinstance(image_path_or_array, str):
                img = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return None
            else:
                img = np.array(image_path_or_array)

            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                outputs = self.image_model(tensor)
                probs = self._safe_softmax(outputs)
                cls = int(torch.argmax(probs, dim=1).item())
                conf = float(probs[0, cls].item())

            return {
                "predicted_class": cls,
                "confidence": float(np.clip(conf, 0.0, 1.0)),
                "probabilities": probs[0].cpu().numpy().tolist(),
            }
        except Exception as exc:
            logger.error(f"Image prediction error: {exc}")
            return None

    def predict_text(self, text_docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not text_docs:
            return None
        if self.text_clf is None or self.text_vectorizer is None:
            return None

        try:
            texts = []
            for d in text_docs:
                t = (d.get("content") or d.get("text") or "").strip()
                if t:
                    texts.append(t)
            if not texts:
                return None

            combined = " ".join(texts)
            X = self.text_vectorizer.transform([combined])
            proba = self.text_clf.predict_proba(X)[0]
            p_abnormal = float(proba[1])
            conf = float(np.clip(abs(p_abnormal - 0.5) * 2.0, 0.0, 1.0))

            return {"abnormal_probability": p_abnormal, "confidence": conf}
        except Exception as exc:
            logger.error(f"Text prediction error: {exc}")
            return None

    def fuse_predictions(self, sensor_pred, image_pred=None, text_pred=None) -> float:
        elems = []

        p_abnormal = float(sensor_pred["probabilities"][1])
        sensor_risk = self._risk_from_prob(p_abnormal, sensor_pred["predicted_class"])
        elems.append((sensor_risk, float(sensor_pred["confidence"]), "sensor"))

        if image_pred is not None:
            img_p_def = float(np.clip(float(image_pred["probabilities"][1]), 0.0, 1.0))
            elems.append((img_p_def, float(image_pred["confidence"]), "image"))

        if text_pred is not None:
            txt_risk = float(np.clip(float(text_pred["abnormal_probability"]), 0.0, 1.0))
            elems.append((txt_risk, float(text_pred["confidence"]), "text"))

        total_conf = sum(conf for _, conf, _ in elems) + 1e-12
        fused = sum(risk * (conf / total_conf) for (risk, conf, _) in elems)
        return float(np.clip(np.nan_to_num(fused), 0.0, 1.0))

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.models_loaded:
            raise ValueError("Models not loaded (sensor required).")
        if input_data.get("sensor_window") is None:
            raise ValueError("sensor_window is required")
        sensor_pred = self.predict_sensor(input_data["sensor_window"])
        if sensor_pred is None:
            raise ValueError("sensor prediction failed")
        image_pred = None
        image_data = input_data.get("image_data", None)
        image_path = input_data.get("image_path", None)
        if image_data is not None:
            image_pred = self.predict_image(image_data)
        elif image_path is not None:
            image_pred = self.predict_image(image_path)

        text_pred = None
        if input_data.get("text_docs"):
            text_pred = self.predict_text(input_data.get("text_docs"))

        fp = self.fuse_predictions(sensor_pred, image_pred, text_pred)

        fault_type = self.fault_types.get(sensor_pred["predicted_class"], "normal")
        if text_pred is not None and text_pred.get("abnormal_probability", 0.0) >= 0.6 and fault_type == "normal":
            fault_type = "abnormal_text_signal"

        signals = []
        if sensor_pred["predicted_class"] == 1:
            signals.append("sensor_abnormal")
        if image_pred and image_pred["predicted_class"] == 1:
            signals.append("visual_defect")
        if text_pred and text_pred["abnormal_probability"] >= 0.5:
            signals.append("text_risk")

        modalities_used = ["sensor"]
        if image_pred is not None:
            modalities_used.append("image")
        if text_pred is not None:
            modalities_used.append("text")
        sensor_analysis = {
            "predicted_class": int(sensor_pred["predicted_class"]),
            "confidence": round(float(sensor_pred["confidence"]), 3),
            "p_abnormal": round(float(sensor_pred["probabilities"][1]), 3),
        }
        res: Dict[str, Any] = {
            "failure_probability": round(float(fp), 3),
            "estimated_time_to_breakdown_hours": float(self._estimate_ttb_hours(fp)),
            "predicted_fault_type": fault_type,
            "top_signals": signals,
            "modalities_used": modalities_used,
            "sensor_analysis": sensor_analysis,
        }
        image_analysis = None
        if image_pred is not None:
            image_analysis = {
                "defect_detected": bool(image_pred["predicted_class"] == 1),
                "confidence": round(float(image_pred["confidence"]), 3),
                "p_defect": round(float(image_pred["probabilities"][1]), 3),
            }
            res["image_analysis"] = image_analysis
        text_analysis = None
        if text_pred is not None:
            text_analysis = {
                "abnormal_probability": round(float(text_pred["abnormal_probability"]), 3),
                "confidence": round(float(text_pred["confidence"]), 3),
            }
            res["text_analysis"] = text_analysis
        if bool(input_data.get("include_llm_explanation", False)):
            expl = self.generate_llm_explanation(
                failure_probability=float(fp),
                fault_type=fault_type,
                top_signals=signals,
                sensor_analysis=sensor_analysis,
                image_analysis=image_analysis,
                text_analysis=text_analysis,
            )
            if expl:
                res["llm_explanation"] = expl
        return res
def main():
    base_dir = Path(__file__).parent.parent.parent
    orchestrator = InferenceOrchestrator(base_dir / "models")
    orchestrator.load_models()
    X = np.load(base_dir / "data" / "processed" / "sensor" / "X_test.npy")
    example = np.nan_to_num(X[0], nan=0.0, posinf=0.0, neginf=0.0).tolist()
    inp = {
        "asset_id": "pump_001",
        "sensor_window": example,
        "text_docs": [{"content": "severe bearing noise and overheating detected"}],
        "include_llm_explanation": True,
    }
    print(json.dumps(orchestrator.predict(inp), indent=2))
if __name__ == "__main__":
    main()