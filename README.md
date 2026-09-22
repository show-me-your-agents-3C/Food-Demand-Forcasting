# FreshFlow: Demand Forecasting & Replenishment Decision Support

## Local dashboard

The Chinese dashboard is available at `/` when running the API. On this Mac,
double-click `start-local.command`, then open `http://127.0.0.1:8001`.
See `docs/FRONTEND_HANDOFF.md` for setup, changed files and the demo walkthrough.

The frontend uses the existing forecast and replenishment artifacts. No frontend
build step or external CDN is required. Without gateway credentials, the chat
panel visibly uses the scoped rule-based fallback.

FreshFlow is a portfolio-ready prototype for food retail planning. It uses the official Favorita Store Sales data for store-family demand forecasting and transparently simulated operational constraints for replenishment decisions.

## Outputs

Running the pipeline writes the following evidence-backed artifacts to `outputs/`:

| File | Purpose |
| --- | --- |
| `forecast.csv` | 7-day store-family sales forecast, plus a baseline prediction interval |
| `replenishment_plan.csv` | current stock, reorder point, recommended order quantity, stockout and waste risk |
| `dashboard_summary.json` | headline KPIs and the highest-priority actions for a UI or agent |
| `action_explanations.json` | evidence-bound, plain-language explanations for each action |
| `metrics.json` | time-based backtest metrics for the seasonal-naive baseline |

## Scope and data boundary

The input is `data/favorita/train.csv`, the official Kaggle Store Sales competition data. It has real-world historical sales but no inventory, shelf-life, supplier lead-time, or MOQ fields. `src/food_forecast/inventory.py` creates deterministic, category-level scenario parameters for those missing fields. They are simulations, not claims about Favorita operations.

The prototype focuses on food families only: BREAD/BAKERY, BEVERAGES, DAIRY, DELI, EGGS, FROZEN FOODS, GROCERY I/II, MEATS, POULTRY, PREPARED FOODS, PRODUCE, and SEAFOOD.

## Architecture

```text
Favorita sales + calendar/promotion signals
                |
                v
          Food-family filter
                |
                v
    Seasonal-naive forecast + backtest
                |
                v
    Inventory scenario + replenishment rules
                |
                v
    CSV / JSON outputs -> dashboard or explanation agent
```

## Pseudocode

```python
history = load_sales(food_families, selected_stores, lookback_days)
forecast = forecast_next_7_days(history, method="seasonal_naive_7d")

for store_family in forecast:
    scenario = build_inventory_scenario(store_family, recent_demand)
    reorder_point = demand_during(scenario.lead_time) + safety_stock
    desired_stock = min(
        forecast_7d + safety_stock,
        demand_before_expiry(scenario.shelf_life_days),
    )
    order_qty = round_up(max(0, desired_stock - current_stock), scenario.moq)
    risks = classify(current_stock, reorder_point, order_qty, shelf_life_days)
    explanation_facts = {forecast, scenario, order_qty, risks}

write_forecasts(); write_replenishment_plan(); write_dashboard_summary()
```

An explanation agent must only verbalize `explanation_facts`; it must not invent demand, costs, or supplier events.

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m src.food_forecast.pipeline
```

The first implementation intentionally uses a small, deterministic sample (three stores, eight food families, 120 days) so it runs quickly. Change limits in `src/food_forecast/config.py` as the project grows.

## API service

```bash
uvicorn src.api.main:app --reload
```

Endpoints: `GET /health`, `GET /api/scope`, `GET /api/metrics`, `GET /api/summary`, `GET /api/forecast`, `GET /api/replenishment`, `GET /api/explanations`, `POST /api/chat`. Interactive docs at `/docs`.

## Deploy

See `docs/HANDOFF.md` and `deploy/` for the bare-metal Lightsail workflow.
