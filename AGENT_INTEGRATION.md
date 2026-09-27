# Agent integration guide (model v3)

The model team publishes everything the agent, replenishment rules and
dashboard need under `outputs/final/`. Nothing downstream should retrain or
copy files by hand.

```bash
# regenerate (≈4 min on the 2-vCPU box); needs lightgbm, pandas, numpy
python -m src.food_forecast.train_final
# see every tool's output once
python -m src.food_forecast.forecast_tools
```

## Files

| File | Use |
|---|---|
| `forecast.csv` | 7 days × 702 store-family series: `p10`, `p50`, `p90`, `onpromotion`, `route`, `forecast_method` |
| `metrics.json` | backtest headline (WAPE / bias per model, windows, calibration) |
| `error_analysis.csv` | accuracy by family, promotion, holiday, volume tier, horizon day |
| `feature_importance.csv` | global drivers of the p50 model |
| `model_metadata.json` | version, data ranges, features, parameters, `forecast_schema`, limitations |
| `models/p10.txt`, `p50.txt`, `p90.txt` | LightGBM boosters (used by the what-if tool) |
| `future_features.csv.gz` | feature rows of the forecast week (used by the what-if tool) |

**Replenishment:** use `p50` as expected demand and `p90 − p50` as the
uncertainty buffer (safety stock at ~90% service level) instead of a
28-day sales standard deviation.

## Tools for the agent

`src/food_forecast/forecast_tools.py` exposes `TOOL_SPECS` (JSON schema) and
`call_tool(name, args)`, which always returns a JSON-able dict (errors come back
as `{"error": ...}` so the agent can retry).

| Tool | Answers |
|---|---|
| `get_forecast(store_nbr, family)` | "How much DAIRY will store 44 sell next week?" |
| `get_forecast_overview(store_nbr?, top_n?)` | "What changes most next week?" |
| `explain_forecast(store_nbr, family)` | "Why is it higher than last week?" (SHAP contributions) |
| `what_if_promotion(store_nbr, family, onpromotion, dates?)` | "What if we promote 60 items on Fri–Sat?" / "What if we cancel the promotion?" |
| `get_model_reliability(family?)` | "Can I trust this?" (backtest WAPE, bias, verdict) |

The LLM gateway ignores native tool calling, so keep the JSON-text protocol
from `test_llm_gateway_langgraph.py`: put `TOOL_SPECS` in the prompt, parse
`{"tool": ..., "args": {...}}` from the reply, run `call_tool`, feed the
result back.

```python
from src.food_forecast.forecast_tools import TOOL_SPECS, call_tool

result = call_tool("what_if_promotion",
                   {"store_nbr": 44, "family": "DAIRY", "onpromotion": 60,
                    "dates": ["2017-08-18", "2017-08-19"]})
```

## Prompt rules we recommend

1. Every number in an answer must come from a tool result in this conversation.
2. When a family's reliability verdict is not "reliable", say so and suggest a planner review.
3. What-if results are model estimates learned from past promotions; state this caveat.
4. `route` = `cold_start` (new store/line) uses a rule, so it has no explanation or what-if; say why.
   `route` = `intermittent` is model-based but sparse: warn that percentage errors are large.

## Switching the existing backend (`feat/backend-agent-api`, `src/api/`) to v3

The current API reads the old seasonal-naive artifacts in `outputs/`. Minimal changes:

| Current (`src/api`) | Change to |
|---|---|
| `service.forecast_frame()` reads `outputs/forecast.csv` (`forecast_sales`, `lower_bound`, `upper_bound`) | read `outputs/final/forecast.csv` (`p50`, `p10`, `p90`, `route`, `onpromotion`) |
| `service.get_metrics()` reads `outputs/metrics.json` (7-day naive backtest) | `outputs/final/metrics.json` (5-window backtest) |
| `simulate_promotion(uplift_pct)` multiplies the forecast by a fixed % | `forecast_tools.what_if_promotion(...)` (model re-scores the planned promotion count) |
| — | add `explain_forecast` and `get_model_reliability` to `TOOL_REGISTRY` |
| `replenishment_plan.csv` built from the old forecast and a 28-day std | rebuild from `p50`; per-family buffers in `outputs/business_value/learned_settings.csv` (policy `v3_tweedie`, latest window) |

The `outputs/final/` files are small (≈10 MB incl. models), so the "server only reads
artifacts" rule in `docs/HANDOFF.md` still holds. The what-if tool needs `lightgbm` at runtime.
