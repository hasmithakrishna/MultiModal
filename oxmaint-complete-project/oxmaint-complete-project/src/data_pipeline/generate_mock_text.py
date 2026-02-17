#!/usr/bin/env python3
"""
Creates pseudo assets pump_001..pump_050 by assigning rows in chunks
"""
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from loguru import logger
def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists():
            return p
    return Path(__file__).resolve().parents[2]
def find_largest_csv(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    csvs = list(folder.rglob("*.csv"))
    if not csvs:
        return None
    return max(csvs, key=lambda p: p.stat().st_size)
PUMP_MANUALS = {
    "troubleshooting_guide.txt": """
CENTRIFUGAL PUMP TROUBLESHOOTING GUIDE

BEARING FAILURE SYMPTOMS:
- Excessive vibration
- High bearing temperature
- Unusual grinding / rattling noise
Resolution: Replace bearings, check alignment, verify lubrication schedule

SEAL LEAK SYMPTOMS:
- Visible leakage around shaft
- Pressure drop in discharge line
Resolution: Replace mechanical seal, check seal faces

IMPELLER DAMAGE SYMPTOMS:
- Flow rate reduction
- Increased vibration at running speed
Resolution: Replace impeller, check suction conditions

MOTOR OVERLOAD SYMPTOMS:
- High current draw
- Motor overheating
Resolution: Check blockages, verify voltage supply

CAVITATION SYMPTOMS:
- Popping / crackling sounds
- Erratic flow and pressure
Resolution: Reduce suction lift, check for air leaks
""",
    "maintenance_schedule.txt": """
PREVENTIVE MAINTENANCE SCHEDULE

DAILY:
- Visual inspection for leaks
- Monitor bearing temperature and vibration

WEEKLY:
- Lubricate bearings
- Check oil level and condition

MONTHLY:
- Detailed vibration analysis
- Inspect seals and clean filters

QUARTERLY:
- Replace oil (if oil-lubricated)
- Calibrate instrumentation

ANNUAL:
- Complete pump overhaul (seals, bearings, alignment)
""",
    "warranty_info.txt": """
PUMP WARRANTY INFORMATION

Warranty coverage is void if:
- Preventive maintenance is skipped
- Operation is outside recommended parameters
- Cavitation damage occurs

Operating parameters (example):
- Temperature: 0-80°C
- Pressure: Max 150 PSI
"""
}


BROKEN_TEMPLATES = [
    {
        "type": "corrective",
        "issue": "bearing_failure",
        "description": "Excessive vibration detected; bearing temperature elevated to {temp}°C",
        "priority": "high",
        "parts": ["bearing_6309", "lubricant_oil"],
        "labor_hours": 6,
    },
    {
        "type": "corrective",
        "issue": "seal_leak",
        "description": "Mechanical seal leaking; visible fluid around shaft",
        "priority": "high",
        "parts": ["mechanical_seal_kit", "gasket_set"],
        "labor_hours": 4,
    },
    {
        "type": "corrective",
        "issue": "motor_overload",
        "description": "Motor current high at {current}A; above rated {rated}A",
        "priority": "high",
        "parts": ["motor_coupling", "alignment_shims"],
        "labor_hours": 3,
    },
    {
        "type": "corrective",
        "issue": "impeller_damage",
        "description": "Flow reduced by {reduction}%; suspected impeller damage",
        "priority": "medium",
        "parts": ["impeller_assembly", "wear_ring"],
        "labor_hours": 8,
    },
]

RECOVERING_TEMPLATES = [
    {
        "type": "follow_up",
        "issue": "post_repair_monitoring",
        "description": "Post-repair monitoring: confirm vibration trending down and temperature stable",
        "priority": "medium",
        "parts": [],
        "labor_hours": 1,
    },
    {
        "type": "follow_up",
        "issue": "alignment_check",
        "description": "Re-check coupling alignment and confirm no abnormal noise after recent maintenance",
        "priority": "medium",
        "parts": ["alignment_shims"],
        "labor_hours": 2,
    },
]

NORMAL_TEMPLATES = [
    {
        "type": "preventive",
        "issue": "routine_maintenance",
        "description": "Scheduled {interval} preventive maintenance: lubrication, inspection, and basic checks",
        "priority": "medium",
        "parts": ["oil_filter", "lubricant_oil"],
        "labor_hours": 2,
    },
    {
        "type": "preventive",
        "issue": "inspection",
        "description": "Routine inspection: check for leaks, verify pressure/flow within expected range",
        "priority": "low",
        "parts": [],
        "labor_hours": 1,
    },
]


def choose_template_for_status(status: str):
    s = (status or "").strip().lower()
    if "broken" in s or "fault" in s or "fail" in s:
        return random.choice(BROKEN_TEMPLATES)
    if "recover" in s:
        return random.choice(RECOVERING_TEMPLATES)
    return random.choice(NORMAL_TEMPLATES)


def load_sensor_status_per_asset(sensor_csv: Path, num_assets: int = 50) -> dict:
    import pandas as pd

    df = pd.read_csv(sensor_csv)

    if "machine_status" not in df.columns:
        logger.warning("machine_status column not found. Falling back to 'normal' for all pumps.")
        return {f"pump_{i:03d}": "normal" for i in range(1, num_assets + 1)}

    df["machine_status"] = df["machine_status"].astype(str).str.strip().str.lower()
    n = len(df)
    chunk_size = max(1, n // num_assets)
    asset_ids = []
    for i in range(n):
        idx = min(i // chunk_size, num_assets - 1)  # 0..num_assets-1
        asset_ids.append(f"pump_{idx + 1:03d}")
    df["_asset_id"] = asset_ids
    dominant = {}
    for aid, g in df.groupby("_asset_id"):
        top = g["machine_status"].value_counts().idxmax()
        dominant[aid] = str(top).strip().lower()
    return dominant
def generate_work_orders(asset_status: dict, num_orders: int = 100) -> list:
    work_orders = []
    start_date = datetime.now() - timedelta(days=730)

    asset_ids = sorted(asset_status.keys())
    if not asset_ids:
        asset_ids = [f"pump_{i:03d}" for i in range(1, 51)]
        asset_status = {aid: "normal" for aid in asset_ids}

    for i in range(num_orders):
        asset_id = random.choice(asset_ids)
        status = asset_status.get(asset_id, "normal")
        template = choose_template_for_status(status)
        date = start_date + timedelta(days=random.randint(0, 730))

        desc = template["description"].format(
            temp=random.randint(85, 110),
            reduction=random.randint(15, 40),
            interval=random.choice(["monthly", "quarterly", "annual"]),
            current=round(random.uniform(45, 60), 2),
            rated=42.5,
        )

        wo = {
            "work_order_id": f"WO-{100000 + i}",
            "asset_id": asset_id,
            "sensor_dominant_status": status,
            "type": template["type"],
            "issue_type": template["issue"],
            "description": desc,
            "priority": template["priority"],
            "status": random.choice(["completed", "completed", "in_progress", "scheduled"]),
            "created_date": date.isoformat(),
            "completed_date": (date + timedelta(hours=random.randint(2, 48))).isoformat()
            if random.random() > 0.25
            else None,
            "parts_used": template["parts"],
            "labor_hours": template["labor_hours"],
            "technician": f"Tech-{random.randint(1, 10):02d}",
            "cost": template["labor_hours"] * 75 + random.randint(100, 500),
        }
        work_orders.append(wo)

    return work_orders
def generate_maintenance_requests(asset_status: dict, num_requests: int = 50) -> list:
    requests = []
    start_date = datetime.now() - timedelta(days=365)

    asset_ids = sorted(asset_status.keys())
    if not asset_ids:
        asset_ids = [f"pump_{i:03d}" for i in range(1, 51)]
        asset_status = {aid: "normal" for aid in asset_ids}

    issues_for_broken = [
        "Unusual grinding noise from pump",
        "Vibration levels increasing rapidly",
        "Temperature running unusually high",
        "Pressure fluctuations and instability",
        "Seal dripping observed",
    ]
    issues_for_recovering = [
        "Post-repair check: mild vibration observed",
        "Monitor temperature after maintenance",
        "Confirm pressure stable after service",
    ]
    issues_for_normal = [
        "Routine inspection requested",
        "Minor vibration noted (monitor)",
        "Preventive lubrication request",
    ]

    for i in range(num_requests):
        asset_id = random.choice(asset_ids)
        status = str(asset_status.get(asset_id, "normal")).strip().lower()
        date = start_date + timedelta(days=random.randint(0, 365))

        if "broken" in status or "fail" in status or "fault" in status:
            issue = random.choice(issues_for_broken)
            severity = random.choice(["medium", "high", "high"])
        elif "recover" in status:
            issue = random.choice(issues_for_recovering)
            severity = random.choice(["low", "medium"])
        else:
            issue = random.choice(issues_for_normal)
            severity = random.choice(["low", "low", "medium"])

        req = {
            "request_id": f"MR-{50000 + i}",
            "asset_id": asset_id,
            "sensor_dominant_status": status,
            "reported_by": f"Operator-{random.randint(1, 20):02d}",
            "issue": issue,
            "severity": severity,
            "date_reported": date.isoformat(),
            "status": random.choice(["open", "assigned", "closed"]),
            "notes": "Reported during routine checks",
        }
        requests.append(req)

    return requests


def main():
    # Make output deterministic (same results every run)
    random.seed(42)

    repo_root = find_repo_root(Path(__file__).parent)
    text_dir = repo_root / "data" / "mock" / "text"
    sensor_dir = repo_root / "data" / "raw" / "sensor"
    text_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating mock text data (ALIGNED to sensor labels)...")
    logger.info("Creating pump manuals...")
    for filename, content in PUMP_MANUALS.items():
        (text_dir / filename).write_text(content.strip() + "\n", encoding="utf-8")
        logger.success(f"Created {filename}")
    sensor_csv = find_largest_csv(sensor_dir)
    if sensor_csv is None:
        logger.warning(f"No sensor CSV found under {sensor_dir}. Using 'normal' for all pumps.")
        asset_status = {f"pump_{i:03d}": "normal" for i in range(1, 51)}
    else:
        logger.info(f"Using sensor CSV: {sensor_csv}")
        asset_status = load_sensor_status_per_asset(sensor_csv, num_assets=50)

    logger.info(f"Asset status distribution: {Counter(asset_status.values())}")
    logger.info("Generating aligned work orders...")
    work_orders = generate_work_orders(asset_status, num_orders=100)
    with open(text_dir / "work_orders.json", "w", encoding="utf-8") as f:
        json.dump(work_orders, f, indent=2)
    logger.success(f"Generated {len(work_orders)} work orders")

    logger.info("Generating aligned maintenance requests...")
    requests = generate_maintenance_requests(asset_status, num_requests=50)
    with open(text_dir / "maintenance_requests.json", "w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2)
    logger.success(f"Generated {len(requests)} maintenance requests")

    logger.success(f"All aligned mock text data created in {text_dir}")
    logger.info("Open data/mock/text/work_orders.json to verify pump statuses.")
if __name__ == "__main__":
    main()
