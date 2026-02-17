#!/usr/bin/env python3
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
class TextPreprocessor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.fault_keywords = {
            "bearing_failure": ["bearing", "grinding", "rattling", "friction", "lubrication"],
            "seal_leak": ["seal", "leak", "leaking", "drip", "shaft"],
            "impeller_damage": ["impeller", "flow", "cavitation", "vane", "prime"],
            "motor_overload": ["motor", "overload", "current", "overheating", "thermal"],
            "normal": ["normal", "routine", "preventive", "scheduled", "inspection"],
        }

    def load_model(self):
        if self.model is None:
            logger.info(f"Loading text embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.success("Text model loaded")

    def clean_text(self, text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\s\.\,\-:]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_fault_indicators(self, text: str) -> Dict[str, float]:
        text_clean = self.clean_text(text)
        scores = {}
        for fault_type, keywords in self.fault_keywords.items():
            score = sum(1 for kw in keywords if kw in text_clean)
            scores[fault_type] = score / max(1, len(keywords))
        return scores

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        self.load_model()
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        emb = self.model.encode(texts, show_progress_bar=True)
        logger.success(f"Created embeddings with shape {emb.shape}")
        return emb

    def process_manuals(self, text_dir: Path) -> List[Dict]:
        manuals = []
        for manual_path in sorted(text_dir.glob("*.txt")):
            content = manual_path.read_text(encoding="utf-8", errors="ignore")
            manuals.append(
                {
                    "type": "manual",
                    "manual_name": manual_path.name,
                    "full_text": content,
                }
            )
        return manuals

    def process_work_orders(self, wo_path: Path) -> List[Dict]:
        work_orders = json.loads(wo_path.read_text(encoding="utf-8"))
        processed = []
        for wo in work_orders:
            created = wo.get("created_date")
            processed.append(
                {
                    "record_type": "work_order",
                    "work_order_id": wo.get("work_order_id"),
                    "asset_id": wo.get("asset_id"),
                    "timestamp": created,
                    "issue_type": wo.get("issue_type"),
                    "description": wo.get("description", ""),
                    "priority": wo.get("priority"),
                    "status": wo.get("status"),
                    "labor_hours": wo.get("labor_hours", 0),
                    "text": f"{wo.get('issue_type','')} {wo.get('description','')}".strip(),
                    "fault_indicators": self.extract_fault_indicators(wo.get("description", "")),
                }
            )
        return processed

    def process_maintenance_requests(self, mr_path: Path) -> List[Dict]:
        reqs = json.loads(mr_path.read_text(encoding="utf-8"))
        processed = []
        for r in reqs:
            ts = r.get("date_reported")
            processed.append(
                {
                    "record_type": "maintenance_request",
                    "request_id": r.get("request_id"),
                    "asset_id": r.get("asset_id"),
                    "timestamp": ts,
                    "issue": r.get("issue", ""),
                    "severity": r.get("severity"),
                    "text": r.get("issue", ""),
                    "fault_indicators": self.extract_fault_indicators(r.get("issue", "")),
                }
            )
        return processed


def main():
    base_dir = Path(__file__).parent.parent.parent

    text_dir = base_dir / "data" / "mock" / "text"
    processed_path = base_dir / "data" / "processed" / "text"
    processed_path.mkdir(parents=True, exist_ok=True)

    if not text_dir.exists() or not any(text_dir.iterdir()):
        logger.error(f"Text data not found at {text_dir}")
        logger.info("Run generate_mock_text.py first")
        return

    pre = TextPreprocessor()

    manuals = pre.process_manuals(text_dir)

    wo_file = text_dir / "work_orders.json"
    mr_file = text_dir / "maintenance_requests.json"
    work_orders = pre.process_work_orders(wo_file) if wo_file.exists() else []
    maint_reqs = pre.process_maintenance_requests(mr_file) if mr_file.exists() else []

    embeddings = {}

    if manuals:
        manual_texts = [m["full_text"] for m in manuals]
        embeddings["manuals"] = pre.create_embeddings(manual_texts)

    if work_orders:
        wo_texts = [w["text"] for w in work_orders]
        embeddings["work_orders"] = pre.create_embeddings(wo_texts)

    if maint_reqs:
        mr_texts = [m["text"] for m in maint_reqs]
        embeddings["maintenance_requests"] = pre.create_embeddings(mr_texts)

    processed_data = {
        "manuals": manuals,
        "work_orders": work_orders,
        "maintenance_requests": maint_reqs,
        "embeddings": embeddings,
        "model_name": pre.model_name,
        "fault_keywords": pre.fault_keywords,
    }

    with open(processed_path / "processed_text.pkl", "wb") as f:
        pickle.dump(processed_data, f)

    records_path = processed_path / "text_records.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        for m in manuals:
            rec = {
                "asset_id": "GLOBAL",
                "timestamp": "1970-01-01T00:00:00",
                "modality": "text",
                "record_type": "manual",
                "label": None,
                "payload": {
                    "manual_name": m["manual_name"],
                    "text": pre.clean_text(m["full_text"])[:2000],
                },
            }
            f.write(json.dumps(rec) + "\n")

        for w in work_orders:
            rec = {
                "asset_id": w["asset_id"],
                "timestamp": w["timestamp"],
                "modality": "text",
                "record_type": "work_order",
                "label": None,
                "payload": {
                    "work_order_id": w["work_order_id"],
                    "issue_type": w["issue_type"],
                    "priority": w["priority"],
                    "status": w["status"],
                    "fault_indicators": w["fault_indicators"],
                    "text": pre.clean_text(w["text"]),
                },
            }
            f.write(json.dumps(rec) + "\n")

        for r in maint_reqs:
            rec = {
                "asset_id": r["asset_id"],
                "timestamp": r["timestamp"],
                "modality": "text",
                "record_type": "maintenance_request",
                "label": None,
                "payload": {
                    "request_id": r["request_id"],
                    "severity": r["severity"],
                    "fault_indicators": r["fault_indicators"],
                    "text": pre.clean_text(r["text"]),
                },
            }
            f.write(json.dumps(rec) + "\n")
    meta = {
        "model_name": pre.model_name,
        "n_manuals": len(manuals),
        "n_work_orders": len(work_orders),
        "n_maintenance_requests": len(maint_reqs),
    }
    with open(processed_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.success(" Text preprocessing complete!")
    logger.info(f"Saved: {processed_path / 'processed_text.pkl'}")
    logger.info(f"Unified records: {records_path}")

if __name__ == "__main__":
    main()