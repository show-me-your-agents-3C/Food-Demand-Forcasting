# Business value — from forecast accuracy to money

`python -m src.food_forecast.business_value` (≈20 s) replays the backtest day by
day for 689 store × family series (53 stores) with **real demand** and each
method's forecasts, then counts lost sales, spoilage and holding cost.
Outputs: `outputs/business_value/`.

## How the simulation works

* Forecasts are fixed at the Wednesday order date (no peeking inside the week).
* Deliveries every 1–7 days by family (daily for bread, prepared food, seafood and
  produce; weekly for dry goods). Order-up-to = forecast for the cover period + buffer.
* First-in-first-out selling. Unmet demand is lost. Stock older than its sellable life is thrown away.
* **Buffers are learned, not hand-picked.** Each family uses the buffer that minimised
  cost in the *previous* window (Christmas → Carnival → May → July). Every method gets the same
  rule, and only the last three windows (84 days) are scored, then annualised (× 4.35).

## Results (per year, 53 stores)

| Policy | Fill rate | Spoiled share | Lost margin | Waste cost | Holding | **Total cost** |
|---|---|---|---|---|---|---|
| **Current practice**: 4-week same-weekday average + fixed 20% buffer | 99.08% | 1.07% | $1.30M | $2.79M | $0.52M | **$4.62M** |
| Same planner rule + learned buffer | 99.14% | 0.24% | $1.18M | $0.90M | $0.81M | $2.89M |
| Seasonal naive + learned buffer | 98.88% | 0.31% | $1.77M | $1.13M | $0.78M | $3.68M |
| LightGBM v2 + learned buffer | 99.00% | 0.16% | $1.44M | $0.50M | $0.74M | $2.67M |
| **LightGBM v3 + learned buffer** | **99.30%** | 0.19% | $1.04M | $0.60M | $0.64M | **$2.28M** |
| LightGBM v3 p10/p50/p90 + learned safety factor | 99.30% | 0.29% | $0.99M | $0.95M | $0.50M | $2.44M |

**Headline: −$2.34M a year (−51%) against current practice.** Fill rate rises from
99.08% to 99.30%, spoiled units fall 82%, and lost-sales units fall 23%.

Where the saving comes from:

| Step | Saving / year | |
|---|---|---|
| Fixed 20% buffer → buffer learned per family | $1.73M | smarter replenishment policy (the agent's job) |
| Planner rule → v3 forecast (same learned-buffer rule) | $0.61M (−21%) | forecast accuracy (the model's job) |
| v2 → v3 (same rule) | $0.40M (−15%) | this round of model work |

The biggest wins are in fast-perishing families. Bread goes from 18% of deliveries
spoiled to 3.7%, and prepared foods from 21% to 4.8%, because a fixed +20% buffer is
far too much for products that must sell the same day.

The quantile (p10/p90) policy was tested and was **not** better than p50 + learned
buffer (the 80% band is slightly narrow, see MODEL_CARD). We keep p50 + learned buffer
for replenishment and use p10/p90 for communicating risk.

## Assumptions (Favorita has no prices or stock)

| Family | Cost | Price | Sellable days | Delivery every |
|---|---|---|---|---|
| Prepared foods | $2.20 | $3.40 | 1 | 1 day |
| Bread/bakery | $0.60 | $0.95 | 1 | 1 |
| Seafood | $4.50 | $6.50 | 2 | 1 |
| Produce | $1.00 | $1.45 | 3 | 1 |
| Poultry | $2.60 | $3.60 | 3 | 2 |
| Meats | $3.50 | $4.90 | 4 | 2 |
| Deli | $2.50 | $3.60 | 4 | 2 |
| Dairy | $0.85 | $1.20 | 7 | 2 |
| Eggs | $2.00 | $2.80 | 14 | 3 |
| Frozen | $2.00 | $2.90 | 60 | 7 |
| Beverages, Grocery I/II | $0.70–1.80 | $1.00–2.60 | 120 | 7 |

Stockout cost = lost gross margin (no goodwill penalty, so this is conservative).
Waste cost = unit cost. Holding cost = 25% per year. Stock starts at zero in each window
for every policy. Perfect FIFO and on-time deliveries make waste lower than in real
stores, so the percentages matter more than the absolute dollars.
