#!/usr/bin/env python3
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
def assign_asset_id_for_index(i: int, num_assets: int = 50) -> str:
    return f"pump_{(i % num_assets) + 1:03d}"
def pseudo_timestamp_for_index(i: int) -> str:
    base = np.datetime64("2025-01-01T00:00:00")
    ts = base + np.timedelta64(i, "m")
    return str(ts).replace(" ", "T")

class ImagePreprocessor:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size

    def discover_ok_def_dirs(self, base_path: Path):
        all_dirs = [d for d in base_path.rglob("*") if d.is_dir()]
        ok_dirs, def_dirs = [], []
        for d in all_dirs:
            name = d.name.lower()
            if "ok" in name and "def" not in name:
                ok_dirs.append(d)
            elif "def" in name:
                def_dirs.append(d)
        return ok_dirs, def_dirs

    def list_images(self, d: Path):
        return list(d.glob("*.jpeg")) + list(d.glob("*.jpg")) + list(d.glob("*.png"))

    def load_images_with_paths(self, base_path: Path, limit_per_class: int = 1000):
        base_path = Path(base_path)
        ok_dirs, def_dirs = self.discover_ok_def_dirs(base_path)

        logger.info(f"Found {len(ok_dirs)} OK dirs, {len(def_dirs)} Defect dirs")

        images = []
        labels = []
        paths = []

        for ok_dir in ok_dirs:
            files = self.list_images(ok_dir)
            for img_file in tqdm(files[:limit_per_class], desc=f"OK {ok_dir.name}"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_resized = cv2.resize(img, self.img_size)
                images.append(img_resized)
                labels.append(0)
                paths.append(str(img_file))

        for def_dir in def_dirs:
            files = self.list_images(def_dir)
            for img_file in tqdm(files[:limit_per_class], desc=f"DEF {def_dir.name}"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_resized = cv2.resize(img, self.img_size)
                images.append(img_resized)
                labels.append(1)
                paths.append(str(img_file))

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64), paths

    def preprocess_batch(self, images: np.ndarray):
        images = images / 255.0
        if len(images.shape) == 3:
            images = images[..., np.newaxis]
        return images
def main():
    base_dir = Path(__file__).parent.parent.parent
    image_base = base_dir / "data" / "raw" / "images"
    if not image_base.exists():
        logger.error(f"Image directory not found: {image_base}")
        return
    possible_paths = [
        image_base / "casting_data" / "casting_data",
        image_base / "casting_data",
        image_base / "casting_512x512",
        image_base,
    ]
    raw_data_path = None
    for p in possible_paths:
        if p.exists() and any(p.rglob("*.jpg")) or any(p.rglob("*.png")) or any(p.rglob("*.jpeg")):
            raw_data_path = p
            break

    if raw_data_path is None:
        logger.error("Could not find image data in expected folders.")
        return
    processed_path = base_dir / "data" / "processed" / "images"
    processed_path.mkdir(parents=True, exist_ok=True)

    pre = ImagePreprocessor(img_size=(224, 224))
    logger.info(f"Loading images from: {raw_data_path}")
    images, labels, paths = pre.load_images_with_paths(raw_data_path, limit_per_class=1000)

    if len(images) == 0:
        logger.error("No images loaded.")
        return
    logger.info(f"Loaded {len(images)} images. OK={int((labels==0).sum())}, DEF={int((labels==1).sum())}")
    idx = np.arange(len(images))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    X_train = images[train_idx]
    y_train = labels[train_idx]
    train_paths = [paths[i] for i in train_idx]
    X_test = images[test_idx]
    y_test = labels[test_idx]
    test_paths = [paths[i] for i in test_idx]
    X_train_proc = pre.preprocess_batch(X_train)
    X_test_proc = pre.preprocess_batch(X_test)
    np.save(processed_path / "X_train.npy", X_train_proc.astype(np.float32))
    np.save(processed_path / "X_test.npy", X_test_proc.astype(np.float32))
    np.save(processed_path / "y_train.npy", y_train)
    np.save(processed_path / "y_test.npy", y_test)
    records_path = processed_path / "image_records.jsonl"
    with open(records_path, "w", encoding="utf-8") as f:
        for split_name, pths, labs in [
            ("train", train_paths, y_train),
            ("test", test_paths, y_test),
        ]:
            for i, (img_path, lab) in enumerate(zip(pths, labs)):
                global_index = i
                rec = {
                    "asset_id": assign_asset_id_for_index(global_index, num_assets=50),
                    "timestamp": pseudo_timestamp_for_index(global_index),
                    "modality": "image",
                    "split": split_name,
                    "label": int(lab),
                    "payload": {
                        "image_path": img_path,
                        "img_size": list(pre.img_size),
                        "grayscale": True,
                    },
                }
                f.write(json.dumps(rec) + "\n")
    metadata = {
        "n_images_train": int(len(X_train_proc)),
        "n_images_test": int(len(X_test_proc)),
        "image_size": list(pre.img_size),
        "n_classes": 2,
    }
    with open(processed_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.success(" Image preprocessing complete!")
    logger.info(f"Saved arrays to: {processed_path}")
    logger.info(f"Unified records: {records_path}")
if __name__ == "__main__":
    main()
