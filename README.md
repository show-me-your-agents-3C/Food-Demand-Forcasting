# FreshFlow: Demand Forecasting & Replenishment Decision Support

## Local dashboard

The Chinese dashboard is available at `/` when running the API. On this Mac,
double-click `start-local.command`, then open `http://127.0.0.1:8001`.
See `docs/FRONTEND_HANDOFF.md` for setup, changed files and the demo walkthrough.

The frontend uses the existing forecast and replenishment artifacts. No frontend
build step or external CDN is required. Without gateway credentials, the chat
panel visibly uses the scoped rule-based fallback.

FreshFlow is a portfolio-ready prototype for food retail planning. It uses the official Favorita Store Sales data for store-family demand forecasting and transparently simulated operational constraints for replenishment decisions.

## Current model: v3 (see MODEL_CARD.md)

Direct 7-day LightGBM (Tweedie p50 + quantile p10/p90), retrained and evaluated on
five historical windows: **11.7% mean WAPE vs 19.9% seasonal naive and 15.3% for v2**.
The v3 Agent reads artifacts from `outputs/final/`; the root-level baseline pipeline and dashboard remain separate. Agent tools (forecast, replenishment, explain, what-if promotion, reliability) are in `src/food_forecast/forecast_tools.py` - see `AGENT_INTEGRATION.md`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-model.txt
.venv/bin/python -m src.food_forecast.train_final    # refresh outputs/final/; optional, takes several minutes
.venv/bin/python -m src.food_forecast.forecast_tools # demo every registered tool
.venv/bin/python -m src.food_forecast.chat_agent     # English multi-turn replenishment assistant
.venv/bin/python -m pytest tests -q
```

The chat Agent uses the installed LangGraph cycle and the gateway's text-JSON protocol (not native tool calling). Set `LLM_GATEWAY_URL`, `LLM_GATEWAY_API_KEY`, and `LLM_MODEL` in the environment or a local `.env` file before starting it. The key is sent only as the `X-API-Key` request header. `/clear` clears the current conversation; `/exit` exits. Conversation state is in memory only and is discarded when the process exits.

The Agent sends a bounded JSON-mode request (512 output tokens, 45-second HTTP timeout, one retry) and requires one complete raw JSON protocol object. The protocol reminder precedes the user request; numeric claims, dates, and required historical/simulation disclosures are still validated against tool evidence. In a live CLI check, the gateway returned a valid tool request, the Agent called `get_forecast` against `outputs/final/`, and the final answer passed validation with status `answered`. Earlier `invalid_protocol` responses were caused by Markdown-fenced/extra JSON and a final answer truncated by the former 256-token output cap. A separate minimal probe returned `OK`; all checks used the configured `X-API-Key` header without displaying or logging the key.

### Data and replenishment Agent

The Agent reads only the published v3 snapshot in `outputs/final/`; it does not train models during chat. The checked-in snapshot was trained on Favorita data through **2017-08-15**, with forecast origin **2017-08-16** and forecast dates **2017-08-16..2017-08-22**. These are historical dates, not today's sales or live inventory. The artifact metadata identifies the point model as `v3.0:v3_tweedie`; p10 and p90 are separate quantile LightGBM artifacts.

Replenishment is recomputed from those same v3 p50/p90 rows and `future_features.csv.gz` (`mean_7`). Current stock, lead time, shelf life, and MOQ are deterministic assumptions from `src/food_forecast/inventory.py`; no inventory, supplier, cost, or purchase-order feed exists. Safety stock uses the p90-p50 weekly uncertainty buffer scaled by simulated lead time. The priority rule is `order_today` before `monitor`, then higher p50 demand. This is decision support only and never places an order.

The original `python -m src.food_forecast.pipeline` remains a separate seasonal-naive baseline that writes root-level `outputs/forecast.csv` and `outputs/replenishment_plan.csv`. It is not connected to the v3 artifacts; do not join those files into one answer. Refreshing v3 requires the explicit `train_final` command above; chat queries only read existing artifacts.

Install and test the Agent dependencies in the active environment:

```bash
.venv/bin/python -m pip install -r requirements-model.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python -m src.food_forecast.chat_agent
```

The gateway variables can be exported in the shell or placed in a local, untracked `.env` file:

```bash
export LLM_GATEWAY_URL="<gateway-base-url>"
export LLM_GATEWAY_API_KEY="<your-key>"
export LLM_MODEL="<model-name>"
.venv/bin/python -m src.food_forecast.chat_agent
```

Example conversations:

```text
You> Which categories should Store 3 replenish first?
You> What is the seven-day forecast for beverages at Store 3?
You> How much should we order?
You> Why?

You> What is the seven-day forecast for beverages at Store 3?
You> What about dairy?

You> How much should we order?
```

If a required store or family is missing, the assistant asks for it in English. If the LLM gateway is unavailable or returns malformed/unsupported output, the Agent labels its evidence-bound response `[Template fallback]`. Every successful tool call emits a trace with tool name, parameters, status, and source files; it never records the API key.

Example responses from the checked-in artifacts (illustrative output, not live business data):

```text
FreshFlow [fallback]> Store 3 BEVERAGES forecast for 7 days: p50 57563.2; p10-p90 range 50858.4 to 62574.8. The data snapshot ends 2017-08-15, forecast origin is 2017-08-16, and forecast dates are 2017-08-16 through 2017-08-22. This is historical data, not live inventory or today's actual sales.

FreshFlow [fallback]> Store 3 BEVERAGES: simulated action order today; recommended order 59720 units; simulated stock 0.0; simulated reorder point 26817.8; stockout risk high; waste risk low. Inventory, lead time (3 days), shelf life (180 days), and MOQ (20) are simulated assumptions. This Agent provides recommendations only and does not place orders.
```

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
Favorita sales + promotion/calendar data
            |
            +--> v3 LightGBM artifacts in outputs/final/
            |        |
            |        +--> English Agent: forecast + simulated replenishment advice
            |
            +--> separate seasonal-naive baseline in outputs/
```

## Pseudocode

The following pseudocode describes the separate baseline pipeline, not the v3 Agent path.

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
