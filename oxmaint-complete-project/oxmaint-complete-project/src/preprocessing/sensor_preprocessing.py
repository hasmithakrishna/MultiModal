#!/usr/bin/env python3
import json
import pickle
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def assign_asset_ids(n_rows: int, num_assets: int = 50, method: str = "chunk") -> list[str]:
    if method == "round_robin":
        return [f"pump_{(i % num_assets) + 1:03d}" for i in range(n_rows)]

    chunk_size = max(1, n_rows // num_assets)
    asset_ids = []
    for i in range(n_rows):
        idx = min(i // chunk_size, num_assets - 1)
        asset_ids.append(f"pump_{idx + 1:03d}")
    return asset_ids


def main():
    base_dir = Path(__file__).parent.parent.parent
    raw_data_path = base_dir / "data" / "raw" / "sensor" / "sensor.csv"
    processed_path = base_dir / "data" / "processed" / "sensor"
    processed_path.mkdir(parents=True, exist_ok=True)

    if not raw_data_path.exists():
        logger.error(f"Sensor data not found at {raw_data_path}")
        return
    logger.info("Loading sensor data...")
    df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded {len(df)} records")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        logger.error("No 'timestamp' column found in sensor.csv")
        return

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if not sensor_cols:
        sensor_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {"machine_status", "Unnamed: 0"}
        sensor_cols = [c for c in sensor_cols if c not in exclude]

    logger.info(f"Found {len(sensor_cols)} sensor columns")

    df[sensor_cols] = df[sensor_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    NUM_ASSETS = 50
    ASSET_ASSIGN_METHOD = "chunk"
    df["asset_id"] = assign_asset_ids(len(df), num_assets=NUM_ASSETS, method=ASSET_ASSIGN_METHOD)

    label_col = "machine_status" if "machine_status" in df.columns else None
    if label_col is None:
        logger.warning("No machine_status found; using 0 labels.")
        df["machine_status"] = "normal"
        label_col = "machine_status"
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    classes = sorted(df[label_col].unique().tolist())
    class_to_id = {c: i for i, c in enumerate(classes)}
    y_all = df[label_col].map(class_to_id).astype(int)
    logger.info(f"Classes: {class_to_id}")
    logger.info("Creating windows...")
    window_size = 50
    stride = 10
    X_windows = []
    y_labels = []
    win_asset_ids = []
    win_timestamps = []
    ts_values = df["timestamp"].values
    asset_values = df["asset_id"].values
    for i in range(0, len(df) - window_size, stride):
        window_df = df.iloc[i : i + window_size]

        window = window_df[sensor_cols].values.astype(np.float32)

        if np.any(np.isnan(window)) or np.any(np.isinf(window)):
            continue

        end_idx = i + window_size - 1
        label_int = int(y_all.iloc[end_idx])

        asset_id = str(asset_values[end_idx])

        ts = ts_values[end_idx]
        if pd.isna(ts):
            ts_out = str(df.iloc[end_idx]["timestamp"])
        else:
            ts_out = pd.Timestamp(ts).isoformat()

        X_windows.append(window)
        y_labels.append(label_int)
        win_asset_ids.append(asset_id)
        win_timestamps.append(ts_out)

    X = np.array(X_windows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int64)

    logger.info(f"Created {len(X)} windows with shape {X.shape}")

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_asset = [win_asset_ids[i] for i in train_idx]
    test_asset = [win_asset_ids[i] for i in test_idx]
    train_ts = [win_timestamps[i] for i in train_idx]
    test_ts = [win_timestamps[i] for i in test_idx]

    logger.info("Normalizing (StandardScaler)...")
    scaler = StandardScaler()

    n_samples, n_timesteps, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(n_samples, n_timesteps, n_features)

    X_test_flat = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(-1, n_timesteps, n_features)

    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    np.save(processed_path / "X_train.npy", X_train_scaled.astype(np.float32))
    np.save(processed_path / "X_test.npy", X_test_scaled.astype(np.float32))
    np.save(processed_path / "y_train.npy", y_train)
    np.save(processed_path / "y_test.npy", y_test)

    with open(processed_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save(processed_path / "train_asset_id.npy", np.array(train_asset, dtype=object))
    np.save(processed_path / "test_asset_id.npy", np.array(test_asset, dtype=object))
    np.save(processed_path / "train_timestamp.npy", np.array(train_ts, dtype=object))
    np.save(processed_path / "test_timestamp.npy", np.array(test_ts, dtype=object))
    records_path = processed_path / "sensor_windows_records.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        for split_name, assets, tss, labels in [
            ("train", train_asset, train_ts, y_train),
            ("test", test_asset, test_ts, y_test),
        ]:
            for a, t, lab in zip(assets, tss, labels):
                rec = {
                    "asset_id": a,
                    "timestamp": t,
                    "modality": "sensor",
                    "split": split_name,
                    "label": int(lab),
                    "payload": {
                        "window_size": window_size,
                        "stride": stride,
                        "n_features": n_features,
                    },
                }
                f.write(json.dumps(rec) + "\n")
    metadata = {
        "n_features": n_features,
        "window_size": window_size,
        "stride": stride,
        "classes": class_to_id,
        "asset_assign_method": ASSET_ASSIGN_METHOD,
        "num_assets": NUM_ASSETS,
        "n_windows": int(len(X)),
    }
    with open(processed_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.success(" Sensor preprocessing complete!")
    logger.info(f"  Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    logger.info(f"  Unified records: {records_path}")
if __name__ == "__main__":
    main()
