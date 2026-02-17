#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def label_to_binary_from_issue_type(issue_type: str) -> int:
    it = (issue_type or "").strip().lower()

    normal_set = {
        "routine_maintenance",
        "inspection",
        "preventive",
        "scheduled",
        "routine",
        "normal",
        "",
        "unknown",
    }

    return 0 if it in normal_set else 1
def main():
    base_dir = Path(__file__).parent.parent.parent
    text_dir = base_dir / "data" / "mock" / "text"
    out_dir = base_dir / "models" / "text_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    wo_path = text_dir / "work_orders.json"
    mr_path = text_dir / "maintenance_requests.json"

    if not wo_path.exists() and not mr_path.exists():
        logger.error("No mock text found.")
        logger.info("Run your mock text generator first (generate_mock_text.py).")
        return

    texts = []
    y = []

    if wo_path.exists():
        work_orders = json.loads(wo_path.read_text(encoding="utf-8"))
        for wo in work_orders:
            issue_type = wo.get("issue_type", "unknown")
            desc = wo.get("description", "")
            t = f"{issue_type} {desc}".strip()
            if not t:
                continue
            texts.append(t)
            y.append(label_to_binary_from_issue_type(issue_type))

    if mr_path.exists():
        reqs = json.loads(mr_path.read_text(encoding="utf-8"))
        for r in reqs:
            issue = (r.get("issue", "") or "").strip()
            if not issue:
                continue
            texts.append(issue)
            y.append(1)

    if len(texts) < 30:
        logger.error(f"Not enough text samples to train. Found only {len(texts)}.")
        logger.info("Generate more mock text first.")
        return

    y = np.array(y, dtype=np.int64)

    logger.info(f"Text samples: {len(texts)}")
    logger.info(f"Label counts (0=normal, 1=abnormal): {np.bincount(y, minlength=2).tolist()}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)

    strat = y if len(np.unique(y)) == 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        n_jobs=None,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    pr_auc = 0.0
    if len(np.unique(y_test)) == 2:
        pr_auc = float(average_precision_score(y_test, y_prob))

    logger.info("Text risk model evaluation (binary):")
    print(classification_report(y_test, y_pred, target_names=["normal(0)", "abnormal(1)"], zero_division=0))
    logger.info(f"PR-AUC: {pr_auc:.4f}")

    with open(out_dir / "text_risk_clf.pkl", "wb") as f:
        pickle.dump(clf, f)

    with open(out_dir / "text_risk_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    meta = {
        "model_type": "TF-IDF + LogisticRegression (binary risk)",
        "n_samples": int(len(texts)),
        "label_counts": np.bincount(y, minlength=2).tolist(),
        "positive_label": "abnormal(1)",
        "note": "Predicts abnormal probability from maintenance text for fusion with sensor+image.",
        "pr_auc_test": pr_auc,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.success(f"Saved text risk model to: {out_dir}")
    logger.success("Files saved: text_risk_clf.pkl, text_risk_vectorizer.pkl, metadata.json")

if __name__ == "__main__":
    main()