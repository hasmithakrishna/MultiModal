#!/usr/bin/env python3
import json
from pathlib import Path
from loguru import logger
def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists():
            return p
    return Path(__file__).resolve().parents[2]

def list_files(base: Path, exts=None, limit=50):
    if not base.exists():
        return []
    exts = set(e.lower() for e in (exts or []))
    out = []
    for p in base.rglob("*"):
        if p.is_file():
            if not exts or p.suffix.lower() in exts:
                out.append(str(p.relative_to(base)))
                if len(out) >= limit:
                    break
    return out

def main():
    repo_root = find_repo_root(Path(__file__).parent)
    data_root = repo_root / "data"

    sensor_dir = data_root / "raw" / "sensor"
    image_dir = data_root / "raw" / "images"
    text_dir = data_root / "mock" / "text"

    manifest = {
        "version": "v1",
        "datasets": [
            {
                "name": "Pump Sensor Data",
                "source": "kaggle",
                "kaggle_id": "nphantawee/pump-sensor-data",
                "modality": "sensor",
                "synthetic": False,
                "license_note": "Kaggle dataset rules apply; accept license on Kaggle page.",
                "local_path": str(sensor_dir.relative_to(repo_root)),
                "files_preview": list_files(sensor_dir, exts=[".csv"], limit=50),
            },
            {
                "name": "Casting Manufacturing Defects",
                "source": "kaggle",
                "kaggle_id": "ravirajsinh45/real-life-industrial-dataset-of-casting-product",
                "modality": "image",
                "synthetic": False,
                "license_note": "Kaggle dataset rules apply; accept license on Kaggle page.",
                "local_path": str(image_dir.relative_to(repo_root)),
                "files_preview": list_files(image_dir, exts=[".png", ".jpg", ".jpeg", ".bmp"], limit=50),
            },
            {
                "name": "Mock Pump Manuals + Work Orders",
                "source": "generated",
                "generator": "src/data_pipeline/generate_mock_text.py",
                "modality": "text",
                "synthetic": True,
                "license_note": "Synthetic content generated for project use.",
                "local_path": str(text_dir.relative_to(repo_root)),
                "files_preview": list_files(text_dir, exts=[".txt", ".json"], limit=50),
            },
        ],
    }

    out_path = data_root / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.success(f" Wrote manifest: {out_path}")


if __name__ == "__main__":
    main()

