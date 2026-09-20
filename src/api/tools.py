"""Plain-Python tools the agent can call over the demand-planning service."""

from __future__ import annotations

from src.api import service


def get_demand_forecast(store_nbr: int | None = None, family: str | None = None, horizon_days: int = 7) -> dict:
    rows = service.get_forecast(store_nbr=store_nbr, family=family, horizon_days=horizon_days)
    return {
        "tool": "get_demand_forecast",
        "row_count": len(rows),
        "forecast": rows,
        "source": "outputs/forecast.csv",
        "scope": {"store_nbr": store_nbr, "family": family, "horizon_days": horizon_days},
    }


def get_inventory_status(store_nbr: int, family: str) -> dict:
    row = service.get_action(store_nbr, family)
    if row is None:
        return {"tool": "get_inventory_status", "found": False, "store_nbr": store_nbr, "family": family}
    keys = [
        "store_nbr", "family", "current_stock_simulated", "safety_stock", "reorder_point",
        "lead_time_days", "shelf_life_days", "moq", "stockout_risk", "waste_risk", "assumption_note",
    ]
    return {
        "tool": "get_inventory_status",
        "found": True,
        "inventory": {key: row[key] for key in keys},
        "source": "outputs/replenishment_plan.csv",
    }


def get_replenishment_plan(store_nbr: int | None = None, family: str | None = None, action: str | None = None, limit: int = 10) -> dict:
    rows = service.get_replenishment(store_nbr=store_nbr, family=family, action=action, limit=limit)
    return {
        "tool": "get_replenishment_plan",
        "row_count": len(rows),
        "actions": rows,
        "source": "outputs/replenishment_plan.csv",
    }


def get_forecast_metrics() -> dict:
    return {
        "tool": "get_forecast_metrics",
        "metrics": service.get_metrics(),
        "source": "outputs/metrics.json",
    }


def get_dashboard_summary() -> dict:
    return {
        "tool": "get_dashboard_summary",
        "summary": service.get_summary(),
        "source": "outputs/dashboard_summary.json",
    }


def list_scope() -> dict:
    return {
        "tool": "list_scope",
        "scope": service.get_scope(),
        "source": "outputs/replenishment_plan.csv",
    }


def simulate_promotion(store_nbr: int, family: str, uplift_pct: float) -> dict:
    return {
        "tool": "simulate_promotion",
        "simulation": service.simulate_promotion(store_nbr, family, uplift_pct),
        "source": "outputs/forecast.csv + outputs/replenishment_plan.csv",
    }


TOOL_REGISTRY = {
    "get_demand_forecast": get_demand_forecast,
    "get_inventory_status": get_inventory_status,
    "get_replenishment_plan": get_replenishment_plan,
    "get_forecast_metrics": get_forecast_metrics,
    "get_dashboard_summary": get_dashboard_summary,
    "list_scope": list_scope,
    "simulate_promotion": simulate_promotion,
}

TOOL_SPECS = [
    {
        "name": "get_demand_forecast",
        "args": {"store_nbr": "int|null", "family": "str|null", "horizon_days": "int"},
        "description": "Daily sales forecast for a store and food family.",
    },
    {
        "name": "get_inventory_status",
        "args": {"store_nbr": "int", "family": "str"},
        "description": "Simulated on-hand stock, reorder point, safety stock, shelf life and risk for one store-family.",
    },
    {
        "name": "get_replenishment_plan",
        "args": {"store_nbr": "int|null", "family": "str|null", "action": "str|null", "limit": "int"},
        "description": "Recommended reorder quantities and risk flags, highest priority first.",
    },
    {
        "name": "get_forecast_metrics",
        "args": {},
        "description": "Backtest metrics (MAE, WAPE) for the seasonal-naive baseline.",
    },
    {
        "name": "get_dashboard_summary",
        "args": {},
        "description": "Headline KPIs and highest priority actions for the current scope.",
    },
    {
        "name": "list_scope",
        "args": {},
        "description": "Stores and food families covered by the current forecast run.",
    },
    {
        "name": "simulate_promotion",
        "args": {"store_nbr": "int", "family": "str", "uplift_pct": "float"},
        "description": "Estimate demand and reorder impact of a promotion uplift on one store-family.",
    },
]
