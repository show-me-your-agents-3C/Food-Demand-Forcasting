# Demo script (≈5 minutes)

Site: https://56-10-18-105.sslip.io (backup: `ssh -N -L 8000:127.0.0.1:8000 lightsail-ubuntu`, then http://localhost:8000)

## 0. Set-up (before the demo)

- Open the site. The status in the bottom-left should read **Data connected**, and the badge in the chat panel should read **Ready**.
- Keep a terminal on the server ready for step 6:

```bash
cd ~/freshflow-app && .venv/bin/python -m src.food_forecast.chat_agent
```

## 1. The dashboard (45 s)

- The KPI cards show how many store × category pairs need an order today and how many have high waste risk, plus the model's backtest error: **11.7% WAPE vs 19.9%** for “same day last week”.
- The chart shows the 7-day forecast (p50) with its 80% range (p10–p90).
- Click a row in the replenishment table: the chart and the assistant follow the selected store and category.

## 2. Chat, in one session

Choose **Store 1** in the store filter, then ask in the chat box:

| # | Question | What to point out |
|---|---|---|
| 1 | Give me the seven-day forecast for PRODUCE at Store 1. | p50 with its p10–p90 range; open **View evidence** to show the `get_forecast` tool call |
| 2 | Is there a stockout risk? | It reuses the previous context (Store 1, produce) and calls `get_replenishment` |
| 3 | How many units should I order? | The order quantity, with simulated-inventory and no-order disclaimers |
| 4 | Which categories should Store 3 replenish first? | `get_priority_replenishments`: high stock-out risk first |
| 5 | (Select Store 44 · Dairy in the table) **What if we cancel all promotions next week?** | The model re-scores the promotion plan (−2.3%) |
| 6 | **Why?** | Last week vs the 4-week average, the promotion plan, and the main drivers |

If an answer shows **[Template fallback]**: its numbers are still correct. It means the LLM's wording failed a fact check, so the product did not show an unverified answer. That is a safety feature worth mentioning.

## 3. Asking for missing information (terminal)

The web chat always knows the selected row, so this part is shown in the CLI started in step 0:

```text
You> How many units should I order?
```

The agent asks which store and category to use instead of guessing.

## 4. Business value (45 s)

Show the table in `BUSINESS_VALUE.md`:
- total cost **$4.62M → $2.28M per year (−51%)**;
- of which **$0.61M** comes from forecast accuracy alone;
- bread waste falls from 18% to 3.7%.
