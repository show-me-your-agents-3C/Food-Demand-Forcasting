"""CLI entry point: forecast demand, backtest, produce a replenishment plan."""

import json

from .config import FORECAST_DAYS, OUTPUT_DIR
from .agent import explain_actions
from .data import load_sales, split_train_validation
from .forecasting import backtest, seasonal_naive
from .inventory import make_plan


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = load_sales()
    train, validation = split_train_validation(history)
    metrics = backtest(train, validation)
    # Forecast from all known history after evaluating on an earlier time split.
    forecast = seasonal_naive(history, horizon_days=FORECAST_DAYS)
    plan = make_plan(forecast, history)
    forecast.to_csv(OUTPUT_DIR / "forecast.csv", index=False)
    plan.to_csv(OUTPUT_DIR / "replenishment_plan.csv", index=False)
    explanations = explain_actions(plan)
    summary = {
        "scope": {"stores": sorted(history.store_nbr.unique().tolist()), "families": sorted(history.family.unique().tolist())},
        "forecast_rows": len(forecast),
        "order_today_count": int((plan.action == "order_today").sum()),
        "high_waste_risk_count": int((plan.waste_risk == "high").sum()),
        "top_actions": plan.head(10).to_dict("records"),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUT_DIR / "dashboard_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUTPUT_DIR / "action_explanations.json").write_text(json.dumps(explanations, indent=2, default=str))
    print(f"Created {len(forecast)} forecast rows and {len(plan)} replenishment actions in {OUTPUT_DIR}")
    print(f"Backtest metrics: {metrics}")


if __name__ == "__main__":
    main()
