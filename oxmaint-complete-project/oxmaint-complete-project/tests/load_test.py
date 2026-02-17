#!/usr/bin/env python3
import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import json
from pathlib import Path
import psutil
API_URL = "http://127.0.0.1:8000"
BASE_DIR = Path(__file__).resolve().parent.parent
X_TEST_PATH = BASE_DIR / "data" / "processed" / "sensor" / "X_test.npy"
def load_test_data():
    if not X_TEST_PATH.exists():
        raise FileNotFoundError(f"Missing: {X_TEST_PATH}. Run preprocessing first.")
    X = np.load(X_TEST_PATH)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X.shape) != 3:
        raise ValueError(f"Expected X_test shape (N, 50, 52). Got: {X.shape}")
    return X
def find_uvicorn_process():
    for p in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if "uvicorn" in cmd and "src.api.main:app" in cmd:
                return p
        except Exception:
            continue
    return None
def build_request_from_index(X, idx):
    i = idx % len(X)
    sensor_window = X[i].tolist()
    return {
        "asset_id": f"pump_{idx:06d}",
        "timestamp": "2026-02-14T10:30:00Z",
        "sensor_window": sensor_window,
        "text_docs": [],
        "image_refs": [],
        "video_refs": [],
        "audio_refs": [],
        "transactional": {},
        "environmental": {}
    }
def make_prediction_request(session, X, idx):
    request_data = build_request_from_index(X, idx)

    t0 = time.perf_counter()
    try:
        resp = session.post(f"{API_URL}/predict", json=request_data, timeout=30)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            return True, latency_ms
        return False, latency_ms
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return False, latency_ms
def run_load_test(X, num_requests, num_workers):
    logger.info(f"Running load test: {num_requests} requests with {num_workers} workers")

    uvicorn_proc = find_uvicorn_process()
    if uvicorn_proc is None:
        logger.warning("Could not find uvicorn process for resource usage. CPU/RAM will be missing.")

    latencies = []
    ok = 0
    fail = 0

    cpu_samples = []
    rss_samples = []

    if uvicorn_proc is not None:
        try:
            uvicorn_proc.cpu_percent(interval=None)
        except Exception:
            pass

    start = time.perf_counter()
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(make_prediction_request, session, X, i) for i in range(num_requests)]

            pending = set(futures)
            while pending:
                done_now = {f for f in pending if f.done()}
                for f in done_now:
                    pending.remove(f)
                    success, lat = f.result()
                    latencies.append(lat)
                    if success:
                        ok += 1
                    else:
                        fail += 1

                if uvicorn_proc is not None:
                    try:
                        cpu_samples.append(uvicorn_proc.cpu_percent(interval=None))
                        rss_samples.append(uvicorn_proc.memory_info().rss / (1024 * 1024))
                    except Exception:
                        pass

                time.sleep(0.05)

    total_time_s = time.perf_counter() - start
    lat_sorted = sorted(latencies)

    def pct(p):
        if not lat_sorted:
            return None
        k = (len(lat_sorted) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(lat_sorted) - 1)
        if f == c:
            return lat_sorted[f]
        return lat_sorted[f] * (c - k) + lat_sorted[c] * (k - f)

    metrics = {
        "total_requests": num_requests,
        "successful": ok,
        "failed": fail,
        "success_rate": (ok / max(num_requests, 1)) * 100.0,
        "total_time_s": total_time_s,
        "throughput_req_s": num_requests / max(total_time_s, 1e-9),
        "latency_mean_ms": float(np.mean(latencies)) if latencies else None,
        "latency_p50_ms": float(pct(50)) if latencies else None,
        "latency_p95_ms": float(pct(95)) if latencies else None,
        "latency_p99_ms": float(pct(99)) if latencies else None,
        "latency_min_ms": float(np.min(latencies)) if latencies else None,
        "latency_max_ms": float(np.max(latencies)) if latencies else None,
        "resource": {
            "cpu_mean_percent": float(np.mean(cpu_samples)) if cpu_samples else None,
            "cpu_max_percent": float(np.max(cpu_samples)) if cpu_samples else None,
            "rss_mean_mb": float(np.mean(rss_samples)) if rss_samples else None,
            "rss_max_mb": float(np.max(rss_samples)) if rss_samples else None,
        }
    }
    return metrics
def test_health_endpoint():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False
def main():
    logger.info("=== Oxmaint API Load Testing (Deterministic + Resource Usage) ===")
    X = load_test_data()

    if not test_health_endpoint():
        logger.error("API is not healthy. Start API first.")
        logger.info("python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000")
        return
    test_scenarios = [
        {"name": "Low Load", "requests": 50, "workers": 5},
        {"name": "Medium Load", "requests": 200, "workers": 20},
        {"name": "High Load", "requests": 500, "workers": 50},
    ]

    results = {}
    for s in test_scenarios:
        logger.info(f"\n{'='*50}\nScenario: {s['name']}\n{'='*50}")
        m = run_load_test(X, s["requests"], s["workers"])
        results[s["name"]] = m
        logger.info(f"Throughput: {m['throughput_req_s']:.2f} req/s")
        logger.info(f"p50: {m['latency_p50_ms']:.2f} ms")
        logger.info(f"p95: {m['latency_p95_ms']:.2f} ms")
        logger.info(f"OK/Fail: {m['successful']}/{m['failed']}")
        r = m["resource"]
        if r["cpu_mean_percent"] is not None:
            logger.info(f"CPU mean/max: {r['cpu_mean_percent']:.1f}% / {r['cpu_max_percent']:.1f}%")
            logger.info(f"RAM mean/max: {r['rss_mean_mb']:.1f} MB / {r['rss_max_mb']:.1f} MB")

        time.sleep(2)

    out_path = BASE_DIR / "reports" / "load_test_results_with_resources.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"Saved report: {out_path}")
if __name__ == "__main__":
    main()
