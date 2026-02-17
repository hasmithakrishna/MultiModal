#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
)


class SensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            x = x + torch.randn_like(x) * 0.02
            if torch.rand(1).item() < 0.2:
                mask = (torch.rand_like(x) > 0.05).float()
                x = x * mask

        return x, y


class SmallLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits


def remap_labels_to_binary(y: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    mapping: Dict[int, int] = {}

    if y.dtype.kind in ("U", "S", "O"):
        y_str = y.astype(str)
        y_bin = np.array([0 if s.upper() == "NORMAL" else 1 for s in y_str], dtype=np.int64)
        return y_bin, mapping

    y_int = y.astype(np.int64)
    uniq = np.unique(y_int)

    if len(uniq) == 1:
        mapping[int(uniq[0])] = 0
        return np.zeros_like(y_int, dtype=np.int64), mapping

    normal_label = int(uniq.min())
    for u in uniq:
        mapping[int(u)] = 0 if int(u) == normal_label else 1

    y_bin = np.array([mapping[int(v)] for v in y_int], dtype=np.int64)
    return y_bin, mapping


def compute_class_weights_binary(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=2).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv / inv.sum() * 2.0
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_y = []
    all_pred = []
    all_prob1 = []
    total_loss = 0.0
    n_batches = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        all_y.append(yb.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_prob1.append(probs[:, 1].cpu().numpy())

    y_true = np.concatenate(all_y) if all_y else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    p1 = np.concatenate(all_prob1) if all_prob1 else np.array([], dtype=np.float32)

    if y_true.size == 0:
        return {"f1": 0.0, "pr_auc": 0.0, "y_true": y_true, "y_pred": y_pred, "p1": p1}

    f1 = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) == 2:
        pr_auc = float(average_precision_score(y_true, p1))
    else:
        pr_auc = 0.0

    return {"f1": float(f1), "pr_auc": float(pr_auc), "y_true": y_true, "y_pred": y_pred, "p1": p1}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running += float(loss.item())
        n += 1

    return running / max(n, 1)


def main():
    base_dir = Path(__file__).parent.parent.parent
    data_path = base_dir / "data" / "processed" / "sensor"
    model_save_path = base_dir / "models" / "sensor_model"
    model_save_path.mkdir(parents=True, exist_ok=True)

    if not (data_path / "X_train.npy").exists():
        logger.error("Processed sensor data not found. Run preprocessing first.")
        return

    logger.info("Loading sensor arrays...")
    X_train = np.load(data_path / "X_train.npy").astype(np.float32)
    X_test = np.load(data_path / "X_test.npy").astype(np.float32)
    y_train_raw = np.load(data_path / "y_train.npy")
    y_test_raw = np.load(data_path / "y_test.npy")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    y_train, map_train = remap_labels_to_binary(y_train_raw)
    y_test, map_test = remap_labels_to_binary(y_test_raw)

    uniq_all = np.unique(np.concatenate([y_train, y_test]))
    counts_all = np.bincount(np.concatenate([y_train, y_test]), minlength=2).astype(np.float32)
    logger.info(f"All-split unique labels (binary): {uniq_all.tolist()}")
    logger.info(f"All-split class counts (0=normal,1=abnormal): {counts_all.tolist()}")

    class_weights = compute_class_weights_binary(y_train)
    logger.info(f"Class weights (len=2): {class_weights.numpy().tolist()}")

    input_size = int(X_train.shape[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = SmallLSTM(input_size=input_size, hidden_size=32, num_layers=1, dropout=0.2).to(device)

    train_ds = SensorDataset(X_train, y_train, augment=True)
    test_ds = SensorDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)

    logger.info("Training sensor baseline (early stop on PR-AUC)...")
    best_pr = -1.0
    best_state = None
    patience = 4
    patience_left = patience

    history = []

    for epoch in range(1, 16):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = eval_model(model, test_loader, device)
        f1 = metrics["f1"]
        pr = metrics["pr_auc"]

        logger.info(
            f"Epoch {epoch}/15 | train loss {tr_loss:.4f} | test F1 {f1:.4f} | test PR-AUC {pr:.4f}"
        )

        history.append({"epoch": epoch, "train_loss": tr_loss, "test_f1": f1, "test_pr_auc": pr})

        score = pr if pr > 0 else f1
        if score > best_pr + 1e-4:
            best_pr = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = eval_model(model, test_loader, device)
    y_true, y_pred = metrics["y_true"], metrics["y_pred"]
    logger.success(f"Final Test F1: {metrics['f1']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")

    if y_true.size > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        logger.info(f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")

        logger.info("Classification report (binary labels):")
        print(classification_report(y_true, y_pred, target_names=["normal(0)", "abnormal(1)"], zero_division=0))

    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "num_classes": 2,
        "hidden_size": 32,
        "dropout": 0.2,
        "label_mapping_train_raw_to_bin": map_train,
        "label_mapping_test_raw_to_bin": map_test,
    }
    torch.save(ckpt, model_save_path / "sensor_model.pth")

    with open(model_save_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(model_save_path / "metadata.json", "w") as f:
        json.dump(
            {
                "model_type": "SmallLSTM",
                "input_size": input_size,
                "num_classes": 2,
                "final_test_f1": metrics["f1"],
                "final_test_pr_auc": metrics["pr_auc"],
                "class_counts_all": counts_all.tolist(),
                "class_weights_used": class_weights.numpy().tolist(),
                "note": "Binary task: normal vs abnormal (recovering/broken mapped to abnormal).",
            },
            f,
            indent=2,
        )
    logger.success(f"Saved sensor model to: {model_save_path}")
if __name__ == "__main__":
    main()