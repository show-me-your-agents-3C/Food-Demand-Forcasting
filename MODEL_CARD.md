# Model card — FreshFlow demand forecaster v3.0

## What it does

Forecasts daily unit demand for the next **7 days** for every **store × food
family** (54 stores × 13 families = 702 series), once a week on the Wednesday
ordering day. Outputs a median (p50) and an 80% range (p10–p90) that the
replenishment rules and the planning agent consume from `outputs/final/`.

## Data

| Item | Value |
|---|---|
| Source | Kaggle *Store Sales – Favorita* (Ecuador): `train.csv`, `test.csv` (future promotion plan), `holidays_events.csv`, `stores.csv` |
| Scope | 13 food families; all 54 stores |
| History used | 2013-01-01 → 2017-08-15 (final model trains on 2013-03-05 → 2017-08-15, 1,034,109 rows) |
| Forecast week | 2017-08-16 → 2017-08-22, promotions taken from `test.csv` |
| Cleaning | No raw rows deleted. Full daily calendar rebuilt, so lags stay correct across the store-closed days (25 Dec) missing from the file. Promotion **count** kept (v1/v2 collapsed it to yes/no). |

## Method

* **Direct, non-recursive formulation.** Horizon *h* = 1..7 is fixed by weekday (Wed = 1 … Tue = 7). Every sales feature reads only days ≤ *t − h*, i.e. sales already closed when the order is placed. One model covers all 7 horizons; no forecast is fed back as input (v2's recursion compounded errors). A unit test changes all sales from the order date on and asserts no feature changes.
* **Features (42):** store/family/type/cluster/city; horizon, weekday, day of month, month, payday distance (Ecuador public-sector paydays: 15th and month end); holidays matched to each store's city/region (national/regional/local, transferred days handled), days to/since holiday, national events; planned promotion count, 7/14-day promotion intensity, promotion vs the series' usual level; last known day, 7/14/28/56-day means, 28-day volatility, zero-share, trend, same-weekday lags (1–4 weeks).
* **Models:** global LightGBM. Point forecast = **Tweedie** objective (chosen over L1 by lower backtest WAPE); p10/p90 = quantile objectives.
* **Routing (decided at the order date):**
  * < 56 trading days (new store / new line) → *same weekday last week*. In backtest: 17.4% WAPE vs 22.1% for LightGBM on these rows (store 52 opened 2017-04-20).
  * Sparse demand in an open store → LightGBM with an `intermittent` low-reliability flag. TSB was tested (90.0% vs 87.5% WAPE for LightGBM on these tiny series), so it was not kept.
  * Stores reopening after a closure stay on LightGBM (a zero-share rule would have wrongly sent store 18 to TSB).

## Evaluation

For each window, all models are **retrained on data before the window**, then issue 4 weekly Wednesday forecasts × 7 days. WAPE = Σ|error| / Σ actual; bias > 0 = over-forecast.

| Model | Earthquake 2016-04 *(stress)* | Christmas 2016-12 | Carnival 2017-02 | Regular 2017-05 | Latest 2017-07 | **Mean of 4 regular** |
|---|---|---|---|---|---|---|
| Seasonal naive (last week) | 32.9% | 24.5% | 20.0% | 18.7% | 16.3% | **19.9%** |
| Planner rule: avg of last 4 same weekdays | 23.9% | 21.6% | 16.0% | 14.0% | 13.7% | **16.3%** |
| v2 LightGBM, recursive | 21.4% | 19.4% | 16.2% | 12.1% | 13.4% | **15.3%** |
| v3 LightGBM, L1 | 21.1% | 15.7% | 12.6% | 10.4% | 10.5% | **12.3%** |
| **v3 LightGBM, Tweedie (final)** | 21.3% | 15.4% | 11.3% | 10.0% | 10.1% | **11.7%** |

* v3 cuts error by **41% vs seasonal naive** and **23% vs v2** (mean of regular windows). Every window improves.
* Full system on all 702 series (with routing): **11.8% WAPE, bias −2.1%**.
* Promotion as count vs yes/no (ablation, L1): 12.29% vs 12.36%. The fix is correct, but at family level the gain is small because most of the promotion signal is already captured by recent sales.

### Where it is reliable, and where it is not (regular windows, Tweedie)

| Situation | v3 WAPE | Seasonal naive | v3 bias |
|---|---|---|---|
| Normal days | 11.1% | 18.8% | −1.4% |
| Holidays / national events | 16.8% | 28.6% | **−7.1%** (under-forecasts) |
| On promotion / no promotion | 11.7% / 14.0% | 20.0% / 20.9% | −2.3% / +0.6% |
| High / mid / low volume series | 11.0% / 15.5% / 27.5% | 19.1% / 24.6% / 39.8% | |
| Horizon day 1 → 7 | 10.3% → 13.2% | 19.8% → 25.3% | |
| Best families | PRODUCE 8.5%, DAIRY 10.1%, BREAD/BAKERY 10.8% | | |
| Weakest families | GROCERY II 35.4%, FROZEN FOODS 28.5% (bias −9.2%), SEAFOOD 24.3% | | |

**Uncertainty calibration:** 77.3% of actuals fall inside p10–p90 (target 80%); 88.1% fall at or below p90 (target 90%). The band is usable for safety stock, slightly narrow.

**Stress test:** after the April 2016 earthquake all models degrade (v3 21.3%, naive 32.9%). Shocks are not predictable from history; the agent should flag them for planner override.

## Most important features (gain share)

Last week's mean 40%, same weekday last week 35%, last known day 13%, same-weekday 4-week mean 8%; calendar, store, family, holidays and promotions make up the rest.

## Known limitations

* Inventory, lead time, shelf life and MOQ are **simulated downstream**; the model only forecasts demand.
* Promotion plans are assumed known in advance (retailer promotion calendar / `test.csv`).
* Granularity is **product family**, not SKU. SKU-level data would add genuinely intermittent items; the routing and TSB code are ready for that.
* Holidays are under-forecast (bias −7%). Planners should review holiday weeks, and FROZEN FOODS / GROCERY II / SEAFOOD orders in particular.
* One model per window is reused for all 4 weekly origins in the backtest; weekly retraining in production should be slightly better.

## Reproduce

```bash
python -m src.food_forecast.backtest_v3 --part v2       # ~5 min
python -m src.food_forecast.backtest_v3 --part v3       # ~30 min, cached per window/model
python -m src.food_forecast.backtest_v3 --part report
python -m src.food_forecast.train_final                 # ~4 min -> outputs/final/
python -m pytest tests -q
```
