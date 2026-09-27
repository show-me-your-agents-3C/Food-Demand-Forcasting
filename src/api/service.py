"""Read-only service layer over the precomputed forecast and replenishment artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.food_forecast.config import OUTPUT_DIR


def output_dir() -> Path:
    return Path(OUTPUT_DIR)


def _read_json(name: str):
    return json.loads((output_dir() / name).read_text(encoding="utf-8"))


def _round_up(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple) if value > 0 else 0


def _records(frame: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for row in frame.to_dict("records"):
        clean: dict = {}
        for key, value in row.items():
            if value is None:
                clean[key] = None
            elif isinstance(value, float):
                clean[key] = None if math.isnan(value) else round(value, 4)
            elif hasattr(value, "item"):
                converted = value.item()
                clean[key] = round(converted, 4) if isinstance(converted, float) else converted
            else:
                clean[key] = value
        records.append(clean)
    return records


def forecast_frame() -> pd.DataFrame:
    frame = pd.read_csv(output_dir() / "forecast.csv", parse_dates=["date"])
    return frame.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def plan_frame() -> pd.DataFrame:
    frame = pd.read_csv(output_dir() / "replenishment_plan.csv")
    frame["_priority"] = frame["action"].map({"order_today": 0, "monitor": 1}).fillna(2)
    frame = frame.sort_values(["_priority", "forecast_7d"], ascending=[True, False])
    return frame.drop(columns="_priority").reset_index(drop=True)


def get_scope() -> dict:
    plan = plan_frame()
    return {
        "stores": sorted(int(value) for value in plan["store_nbr"].unique()),
        "families": sorted(str(value) for value in plan["family"].unique()),
    }


def get_metrics() -> dict:
    return _read_json("metrics.json")


def get_summary() -> dict:
    return _read_json("dashboard_summary.json")


def get_forecast(store_nbr: int | None = None, family: str | None = None, horizon_days: int | None = None, limit: int | None = None) -> list[dict]:
    frame = forecast_frame()
    if store_nbr is not None:
        frame = frame[frame["store_nbr"] == int(store_nbr)]
    if family is not None:
        frame = frame[frame["family"].str.upper() == str(family).upper()]
    if horizon_days:
        frame = frame.groupby(["store_nbr", "family"]).head(int(horizon_days))
    if limit:
        frame = frame.head(int(limit))
    return _records(frame)


def get_replenishment(store_nbr: int | None = None, family: str | None = None, action: str | None = None, limit: int | None = None) -> list[dict]:
    frame = plan_frame()
    if store_nbr is not None:
        frame = frame[frame["store_nbr"] == int(store_nbr)]
    if family is not None:
        frame = frame[frame["family"].str.upper() == str(family).upper()]
    if action is not None:
        frame = frame[frame["action"] == action]
    if limit:
        frame = frame.head(int(limit))
    return _records(frame)


def get_action(store_nbr: int, family: str) -> dict | None:
    rows = get_replenishment(store_nbr=store_nbr, family=family, limit=1)
    return rows[0] if rows else None


def get_explanations(store_nbr: int | None = None, family: str | None = None) -> list[dict]:
    rows = _read_json("action_explanations.json")
    if store_nbr is not None:
        rows = [row for row in rows if int(row["store_nbr"]) == int(store_nbr)]
    if family is not None:
        rows = [row for row in rows if str(row["family"]).upper() == str(family).upper()]
    return rows


def simulate_promotion(store_nbr: int, family: str, uplift_pct: float) -> dict:
    row = get_action(store_nbr, family)
    if row is None:
        raise LookupError(f"no replenishment row for store {store_nbr} family {family}")
    uplift = float(uplift_pct) / 100.0
    baseline_daily = float(row["forecast_daily"])
    baseline_7d = float(row["forecast_7d"])
    uplifted_daily = baseline_daily * (1 + uplift)
    uplifted_7d = baseline_7d * (1 + uplift)
    lead = int(row["lead_time_days"])
    moq = int(row["moq"])
    shelf = int(row["shelf_life_days"])
    safety = float(row["safety_stock"])
    stock = float(row["current_stock_simulated"])
    sellable = uplifted_daily * min(shelf, 7)
    desired = min(uplifted_7d + safety, sellable + safety)
    order = _round_up(max(0.0, desired - stock), moq)
    return {
        "store_nbr": int(row["store_nbr"]),
        "family": row["family"],
        "uplift_pct": round(float(uplift_pct), 2),
        "baseline_forecast_7d": round(baseline_7d, 2),
        "simulated_forecast_7d": round(uplifted_7d, 2),
        "extra_units_7d": round(uplifted_7d - baseline_7d, 2),
        "baseline_order_qty": int(row["recommended_order_qty"]),
        "simulated_order_qty": order,
        "order_delta": order - int(row["recommended_order_qty"]),
        "lead_time_days": lead,
        "shelf_life_days": shelf,
        "baseline_waste_risk": row["waste_risk"],
        "simulated_waste_risk": "high" if (stock + order) > sellable * 1.15 else "low",
        "assumption_note": row["assumption_note"],
    }
