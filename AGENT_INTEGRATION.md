# FreshFlow replenishment Agent

## Data contract

The chat Agent reads the published v3 artifacts under `outputs/final/` only. The checked-in snapshot has model version `v3.0`, point model `v3_tweedie`, training data through `2017-08-15`, forecast origin `2017-08-16`, and target dates `2017-08-16..2017-08-22`. This is archived competition data, not a current or live business feed. The data source is the Kaggle Favorita Store Sales dataset and its promotion, holiday, and store metadata.

Do not combine these rows with root-level `outputs/forecast.csv` or `outputs/replenishment_plan.csv`: `src.food_forecast.pipeline` independently generates those using a seasonal-naive baseline. `src.food_forecast.train_final` is the separate v3 artifact publisher; chat does not train models.

`outputs/final/forecast.csv` supplies p10/p50/p90 demand. `future_features.csv.gz` supplies historical `mean_7` for the deterministic stock scenario. Lead time, current stock, shelf life, and MOQ are simulated using `src/food_forecast/inventory.py`; they are not measured Favorita inventory facts. The replenishment tool scales the weekly p90-p50 buffer by simulated lead time, uses the existing reorder/expiry/MOQ logic, and ranks `order_today` first, then higher p50 demand. No real order is placed.

## Tools

`src/food_forecast/forecast_tools.py` exports `TOOL_SPECS` and `call_tool(name, args)`. Results are JSON-compatible. Errors are returned as `{"error": ...}`; store, family, dates, and top_n are validated. English family names are the default; legacy Chinese family aliases remain accepted. Ambiguous meat-family requests are rejected so the assistant can clarify in English.

| Tool | Purpose |
|---|---|
| `get_forecast` | Daily and aggregate forecast by store/family; accepts an optional date range. |
| `get_replenishment` | v3-based suggested quantity, risks, and simulated assumptions for a store/family. |
| `get_priority_replenishments` | Prioritized items for one store or the available snapshot. |
| `get_forecast_overview` | Forecast totals and biggest changes. |
| `explain_forecast` | Feature contributions for a forecast. |
| `what_if_promotion` | Re-score a promotion scenario. |
| `get_model_reliability` | Backtest reliability facts. |

Every forecast-derived response identifies source files, training cutoff, forecast dates, and model id when available. Replenishment responses include the feature artifact used for the simulated stock input. Unknown metadata must remain unknown rather than inferred.

## LLM and sessions

Run the interactive terminal Agent:

```bash
.venv/bin/python -m pip install -r requirements-model.txt
.venv/bin/python -m src.food_forecast.chat_agent
```

Configure `LLM_GATEWAY_URL`, `LLM_GATEWAY_API_KEY`, and `LLM_MODEL` in the process environment or a local `.env`. The gateway example in `test_llm_gateway_langgraph.py` uses `ChatOllama`, `X-API-Key`, and plain-text JSON because native tool calling is ignored. The Agent follows that pattern while requiring a single JSON protocol object, executing only registered tools, limiting tool calls/retries, and applying a request timeout. Secrets are not included in prompts or traces.

`ChatSession` retains messages, confirmed store/family, snapshot/date context, and tool evidence only in memory. Separate instances do not share state. `/clear` erases the current state and `/exit` exits; process shutdown discards the session. Final answers, clarifications, and error messages are in English. Malformed protocol, unavailable gateway, and unsupported numeric claims use a clearly labeled `[Template fallback]` response. Each trace entry includes tool, arguments, status, and data source.

The latest bounded live CLI check reached the configured gateway but returned a response that could not be accepted as a registered tool request, clarification, or answer (`invalid_protocol`). The CLI returned `[Template fallback]` in English; this is not a successful live LLM validation. The raw gateway response is not retained or printed. Retry after confirming the gateway's model follows the plain-text JSON protocol. The request uses the existing `ChatOllama` configuration, `X-API-Key` header, 20-second timeout, and one retry; API keys are never included in prompts or traces.

Sample English exchange:

```text
You> What is the seven-day forecast for beverages at Store 3?
FreshFlow [answered]> Store 3 BEVERAGES is forecast at 57,563.2 units (p50) for 2017-08-16 through 2017-08-22. The training snapshot ends 2017-08-15; this is historical, not live sales.
You> How much should we order?
FreshFlow [answered]> The simulated recommendation is 59,720 units. Simulated stock is 0 and the simulated reorder point is 26,817.8. Inventory, lead time, shelf life, and MOQ are scenario assumptions; no order is placed.
You> What about dairy?
FreshFlow [answered]> ...
```

Numbers above are illustrative and must be generated from the current tool results at runtime. The Agent retains Store 3 and the same data snapshot when handling follow-ups, while changing the family to dairy.

The Agent provides recommendations only. It does not claim live inventory, supplier incidents, cost/profit, or model accuracy beyond the reported backtest artifacts.