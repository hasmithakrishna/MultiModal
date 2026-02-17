#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path
from loguru import logger
SENSOR_DATASET = "nphantawee/pump-sensor-data"
IMAGE_DATASET = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists():
            return p
    return Path(__file__).resolve().parents[2]
def setup_kaggle_credentials(repo_root: Path) -> bool:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json_dest = kaggle_dir / "kaggle.json"
    candidates = [
        repo_root / "kaggle.json",
        Path.cwd() / "kaggle.json",
    ]
    local_kaggle = next((p for p in candidates if p.exists()), None)
    if local_kaggle:
        logger.info(f"Found kaggle.json at: {local_kaggle}")
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_kaggle, kaggle_json_dest)
        try:
            os.chmod(kaggle_json_dest, 0o600)
        except Exception as e:
            logger.warning(f"Could not chmod kaggle.json (OK on Windows): {e}")

        logger.success(f" Kaggle credentials installed to {kaggle_json_dest}")
        return True

    if kaggle_json_dest.exists():
        logger.success(f" Kaggle credentials already exist at {kaggle_json_dest}")
        return True
    logger.error(" kaggle.json not found!")
    logger.info("Place kaggle.json in your repo root (same level as 'src') or current directory.")
    logger.info("How to get it: Kaggle -> Settings -> API -> Create New API Token")
    return False
def download_dataset(api, dataset: str, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(f"Downloading: {dataset}")
        api.dataset_download_files(dataset, path=str(out_dir), unzip=True)
        logger.success(f"Downloaded to: {out_dir}")
        return True
    except Exception as e:
        logger.error(f"✗ Download failed for {dataset}: {e}")
        logger.info("Possible issues:")
        logger.info("  1) You haven't accepted the dataset rules/license on Kaggle (open dataset page once).")
        logger.info("  2) kaggle.json username/key invalid.")
        logger.info("  3) Network/proxy issues.")
        return False
def verify_sensor_download(sensor_dir: Path) -> bool:
    csvs = list(sensor_dir.glob("*.csv"))
    if not csvs:
        csvs = list(sensor_dir.rglob("*.csv"))

    if not csvs:
        logger.error(f" No CSV found under: {sensor_dir}")
        return False

    main_csv = max(csvs, key=lambda p: p.stat().st_size)
    size_mb = main_csv.stat().st_size / (1024 * 1024)
    logger.success(f" Sensor CSV found: {main_csv.relative_to(sensor_dir)} ({size_mb:.1f} MB)")
    return True


def verify_image_download(image_dir: Path) -> bool:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    imgs = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    if not imgs:
        logger.error(f" No image files found under: {image_dir}")
        return False

    logger.success(f" Images found: {len(imgs)} files (sample: {imgs[0].relative_to(image_dir)})")
    return True


def main():
    logger.info("=" * 60)
    logger.info("KAGGLE SETUP AND DATA DOWNLOAD")
    logger.info("=" * 60)

    repo_root = find_repo_root(Path(__file__).parent)
    logger.info(f"Repo root detected: {repo_root}")

    logger.info("\nStep 1: Setting up Kaggle credentials...")
    if not setup_kaggle_credentials(repo_root):
        sys.exit(1)

    logger.info("\nStep 2: Downloading datasets from Kaggle...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        logger.error("✗ Kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    sensor_dir = repo_root / "data" / "raw" / "sensor"
    image_dir = repo_root / "data" / "raw" / "images"

    ok1 = download_dataset(api, SENSOR_DATASET, sensor_dir)
    ok2 = download_dataset(api, IMAGE_DATASET, image_dir)

    if not (ok1 and ok2):
        logger.error("✗ One or more downloads failed.")
        sys.exit(1)

    logger.info("\nStep 3: Verifying downloads...")
    v1 = verify_sensor_download(sensor_dir)
    v2 = verify_image_download(image_dir)

    if v1 and v2:
        logger.success("\n ALL DATA DOWNLOADED + VERIFIED ")
        logger.info("\nNext:")
        logger.info("  - Run: python src/data_pipeline/generate_mock_text.py")
        logger.info("  - (Recommended) Run: python src/data_pipeline/build_manifest.py")
        sys.exit(0)

    logger.error("\n Verification failed (download might still exist but structure differs).")
    sys.exit(1)

if __name__ == "__main__":
    main()
