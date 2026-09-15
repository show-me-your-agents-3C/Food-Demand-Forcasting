"""Scenario-driven replenishment decisions. All operational values are simulated."""

import math
import pandas as pd


SCENARIOS = {
    "PRODUCE": {"shelf_life_days": 4, "lead_time_days": 1, "moq": 10},
    "BREAD/BAKERY": {"shelf_life_days": 3, "lead_time_days": 1, "moq": 10},
    "DAIRY": {"shelf_life_days": 7, "lead_time_days": 2, "moq": 20},
    "SEAFOOD": {"shelf_life_days": 3, "lead_time_days": 2, "moq": 10},
    "MEATS": {"shelf_life_days": 6, "lead_time_days": 2, "moq": 10},
    "POULTRY": {"shelf_life_days": 5, "lead_time_days": 2, "moq": 10},
    "FROZEN FOODS": {"shelf_life_days": 90, "lead_time_days": 3, "moq": 20},
    "BEVERAGES": {"shelf_life_days": 180, "lead_time_days": 3, "moq": 20},
}
DEFAULT_SCENARIO = {"shelf_life_days": 14, "lead_time_days": 2, "moq": 10}


def _round_up(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple) if value > 0 else 0


def make_plan(forecast: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a 7-day forecast and turn it into transparent order actions."""
    recent = history.groupby(["store_nbr", "family"], as_index=False).tail(7)
    recent_demand = recent.groupby(["store_nbr", "family"], as_index=False)["sales"].mean().rename(columns={"sales": "recent_daily_sales"})
    plan = forecast.groupby(["store_nbr", "family"], as_index=False).agg(
        forecast_7d=("forecast_sales", "sum"),
        forecast_daily=("forecast_sales", "mean"),
        demand_std_28d=("demand_std_28d", "first"),
    ).merge(recent_demand, on=["store_nbr", "family"], how="left")

    records = []
    for row in plan.to_dict("records"):
        scenario = SCENARIOS.get(row["family"], DEFAULT_SCENARIO)
        daily = row["forecast_daily"]
        safety_stock = 1.28 * row["demand_std_28d"] * math.sqrt(scenario["lead_time_days"])
        reorder_point = daily * scenario["lead_time_days"] + safety_stock
        # Deterministic scenario: inventory starts near 1.5 days of recent demand.
        current_stock = max(0.0, row["recent_daily_sales"] * 1.5)
        sellable_before_expiry = daily * min(scenario["shelf_life_days"], 7)
        desired_stock = min(row["forecast_7d"] + safety_stock, sellable_before_expiry + safety_stock)
        order_quantity = _round_up(max(0.0, desired_stock - current_stock), scenario["moq"])
        post_order_stock = current_stock + order_quantity
        waste_risk = "high" if post_order_stock > sellable_before_expiry * 1.15 else "low"
        stockout_risk = "high" if current_stock < reorder_point else "low"
        priority = "order_today" if stockout_risk == "high" else "monitor"
        records.append({
            **row, **scenario,
            "current_stock_simulated": round(current_stock, 2),
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2),
            "recommended_order_qty": order_quantity,
            "stockout_risk": stockout_risk,
            "waste_risk": waste_risk,
            "action": priority,
            "assumption_note": "Inventory, shelf-life, lead-time and MOQ are deterministic demo scenarios.",
        })
    result = pd.DataFrame(records)
    result["priority_rank"] = result["action"].map({"order_today": 0, "monitor": 1})
    return result.sort_values(["priority_rank", "forecast_7d"], ascending=[True, False]).drop(columns="priority_rank")
