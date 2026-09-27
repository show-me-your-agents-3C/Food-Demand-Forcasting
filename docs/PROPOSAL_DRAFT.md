# FreshFlow — an AI planning agent for food-distributor demand forecasting

*Proposal draft. Figures come from `MODEL_CARD.md` and `BUSINESS_VALUE.md`; live demo: https://56-10-18-105.sslip.io*

## 1. The need, and how we address it

Food distributors estimate demand for hundreds of products from past sales and planner experience. Seasonality, promotions and shifting buying patterns are handled by gut feel, so they over-order perishables (waste) or under-order staples (stock-outs).

FreshFlow puts a planning **agent** on top of a backtested forecasting model:

| Step | What happens | Component |
|---|---|---|
| Forecast | Every Wednesday, forecast the next 7 days of demand for every store × category, with an 80% range (p10–p90) | LightGBM v3 (`train_final.py`) |
| Decide | Turn the forecast into an order for the next delivery: forecast × (1 + a per-category safety buffer learned from backtests) − stock on hand | replenishment tool |
| Explain and explore | A planner asks in plain English: *“What's the forecast?”, “Why?”, “Is there a stock-out risk?”, “How much should I order?”, “Which categories first?”, “What if we cancel the promotion?”* | LangGraph agent + Claude Sonnet 4.5 |
| Stay trustworthy | Every number in an answer must come from a tool result. Unsupported answers are replaced by a labelled template. The agent never places orders | fact checks in `chat_agent.py` |

## 2. Domain we chose (the statement was broad)

A **regional fresh-food distributor supplying 54 supermarkets** across Ecuador. It covers 13 food categories (produce, dairy, bread, meats, poultry, seafood, beverages, grocery, …), so **702 store × category demand series**. Fresh categories are delivered daily or every 2–3 days, and dry goods weekly. Orders are planned weekly on Wednesday.

## 3. Assumptions

- **Historical replay.** Sales data ends 2017-08-15, and we forecast 2017-08-16 → 08-22 as if it were “next week”.
- **Promotions are planned in advance.** The number of items on promotion per category and day is known when ordering, as in a retailer's promotion calendar.
- **Granularity is category, not SKU.** The same pipeline extends to SKUs.
- **Operations are simulated.** Stock on hand, shelf life, lead time, delivery frequency, MOQ, cost and price are not in the dataset (see §4). They are disclosed everywhere in the product.

## 4. Data: real vs mock

| Real (Kaggle *Favorita Store Sales*) | Mock / simulated (our assumptions) |
|---|---|
| 1.18M daily food sales rows, 2013–2017 | Stock on hand = 1.5 days of recent sales |
| Items on promotion per store, category and day | Sellable days: bread / prepared food 1, seafood 2, produce 3, meat 3–4, dairy 7, dry goods 60–120 |
| National, regional and local holidays; store city and type | Delivery interval: 1 day (fresh) to 7 days (dry) |
| Forecast-week promotion plan (`test.csv`) | Unit cost and price per category (≈30% gross margin), holding cost 25% per year, MOQ |

No raw sales rows are deleted during cleaning. The 13 series of store 52, which opened in April 2017, are handled as a cold start rather than dropped.

## 5. Method, and why

**Forecasting model.**
- One *global* LightGBM learns across all 702 series. It suits hundreds of products, shares signal between sparse and rich series, and trains in about 2 minutes on the 2-vCPU server.
- **Direct 7-day design.** Features use only sales known on the order date, so the model never feeds its own forecasts back as inputs. A unit test changes all future sales and asserts that no feature changes.
- **Features target the three causes named in the brief:**
  - *seasonality*: weekday, month, paydays, store-matched holidays and days to/from them;
  - *promotions*: planned promotion count, 7- and 14-day promotion intensity, deviation from the usual level;
  - *changing buying patterns*: last day, 7/14/28/56-day means, trend and volatility, plus retraining every week.
- **Tweedie objective** for skewed sales (lower error than L1). **Quantile models** give the p10–p90 range.

**Evaluation.** Five historical windows, each with the model **retrained on data before the window**, then 4 weekly forecasts × 7 days:

| Method | Christmas 2016 | Carnival 2017 | May 2017 | Jul 2017 | **Mean WAPE** | Earthquake 2016 (stress) |
|---|---|---|---|---|---|---|
| Last week's same day | 24.5% | 20.0% | 18.7% | 16.3% | **19.9%** | 32.9% |
| Planner rule (avg of last 4 same weekdays) | 21.6% | 16.0% | 14.0% | 13.7% | **16.3%** | 23.9% |
| Previous model v2 | 19.4% | 16.2% | 12.1% | 13.4% | **15.3%** | 21.4% |
| **FreshFlow v3** | **15.4%** | **11.3%** | **10.0%** | **10.1%** | **11.7%** | 21.3% |

- v3 is better in every window.
- It is weakest on holidays (16.8% WAPE, under-forecasts by 7%) and on low-volume categories; the agent flags these for planner review.
- 77% of actual values fall inside the p10–p90 range (target 80%).

**Decisions that were tested, not assumed.**
- A cold-start rule for new stores beat the model on those series (17.4% vs 22.1%).
- The TSB method for intermittent demand did *not* beat the model, so it was dropped.
- Keeping promotion *counts* instead of yes/no flags helped only slightly.
- The quantile-based order rule did *not* beat p50 + learned buffer, so the simpler rule was kept.

**Agent.**
- LangGraph loop with Claude Sonnet 4.5 through the AWS LLM gateway (text-JSON tool protocol).
- 7 tools over the published model outputs: forecast, overview, explanation (SHAP), what-if promotion, reliability, replenishment, priorities.
- Answers are checked against tool evidence (numbers, dates, risk levels, required disclosures), and failures fall back to a labelled template.
- Multi-turn context (for example, “Is there a stock-out risk?” reuses the store and category), and the agent asks for clarification when information is missing.

## 6. Business value

We replayed 84 days (3 backtest windows) day by day with **real demand**:
- first-in-first-out selling, spoilage after the sellable days, and lost sales when out of stock;
- each category's safety buffer learned from the *previous* window, using the same rule for every method.

| Policy (per year, 53 stores) | Fill rate | Share spoiled | **Total cost** (lost margin + waste + holding) |
|---|---|---|---|
| Current practice: planner rule + fixed 20% buffer | 99.08% | 1.07% | **$4.62M** |
| Planner rule + learned buffer | 99.14% | 0.24% | $2.89M |
| **FreshFlow: v3 forecast + learned buffer** | **99.30%** | 0.19% | **$2.28M** |

- **−$2.34M a year (−51%)** against current practice, split as:
  - $1.73M from category-specific buffers (the decision layer);
  - $0.61M (−21%) from forecast accuracy alone.
- Spoiled units fall 82% and lost-sales units 23%.
- Bread waste drops from 18% to 3.7% of deliveries, and prepared foods from 21% to 4.8%.
- Prices and shelf lives are assumptions, so the **percentages matter more than the dollar amounts**.

## 7. Risks and next steps

- **Real operations data.** Connect stock, supplier lead times and prices. The policy code is already parameterised.
- **Operations.** Weekly retraining (currently one command) and drift monitoring.
- **Human in the loop.** Log planner overrides with reasons, and measure whether they add value (forecast value added).
- **Scope.** Move to SKU level (the TSB and cold-start routes are ready) and model holidays better, since they are under-forecast by 7%.
