# FreshFlow demo script (about 5 minutes, English)

Each part lists **On screen** (what to show or click) and **Say** (narration to read or adapt).
Numbers under **Check** are what the screen should show. The LLM words its answers differently each
time, so paraphrase if needed, but keep the numbers the screen shows.

| # | Part | Time | On screen |
|---|---|---|---|
| 1 | Problem | 0:30 | Dashboard overview |
| 2 | Solution overview | 0:40 | Architecture diagram (proposal doc) |
| 3 | Model performance | 0:40 | Backtest chart (proposal doc) |
| 4 | Agent demo | 2:00 | Dashboard chat |
| 5 | Safety and reliability | 0:45 | Dashboard evidence + terminal |
| 6 | Business impact | 0:30 | Business value table |

## Before recording

- [ ] Open https://56-10-18-105.sslip.io. The bottom-left status reads **Data connected**, and the chat badge reads **Ready**.
- [ ] Open the proposal doc in a second tab and scroll to the architecture diagram.
- [ ] Open a terminal on the server with the CLI agent running (used in part 5):

```bash
cd ~/freshflow-app && .venv/bin/python -m src.food_forecast.chat_agent
```

- [ ] Do one warm-up question in the chat, then click the **Clear chat** icon.
- [ ] Backup if the site is unreachable: `ssh -N -L 8000:127.0.0.1:8000 lightsail-ubuntu`, then open http://localhost:8000.
- [ ] Record each part separately. If an answer times out, ask again.
- [ ] The chat allows 30 questions per 5 minutes per visitor, so leave time between retakes.

## 1. Problem (0:30)

**On screen:** the dashboard overview (KPI cards and the forecast chart).

**Say:**
> Food distributors order hundreds of products every week, mostly from last week's sales and a planner's experience. Seasonality, promotions and shifting demand are judged by gut feel. The result is fresh food thrown away and staples out of stock. FreshFlow is an AI planning agent that turns a tested demand forecast into replenishment decisions. In a replay with real demand, it cut stock-out, waste and holding costs by about half.

## 2. Solution overview (0:40)

**On screen:** the architecture diagram in the proposal doc, section "How it works".

**Say:**
> FreshFlow has two layers. Every Wednesday, a LightGBM model retrains on sales, promotion plans and holidays, and forecasts the next seven days for 702 store-and-category series: 54 stores and 13 food categories. It publishes forecasts, an 80% range, backtest metrics and learned order buffers. On top of that, an LLM agent (LangGraph with Claude Sonnet 4.5) answers the planner's questions. It can only use seven tools that read those published results, so every number traces back to the model.

## 3. Model performance (0:40)

**On screen:** the backtest chart in the proposal doc, section "Forecasting method and why".

**Say:**
> We tested the model the way it would be used. For each of five historical periods we retrained on earlier data only, then forecast four weeks ahead. Average error is about 11.7%, against 19.9% for "same day last week", 16.3% for the planner's four-week average, and 15.3% for our previous model. It wins in every period, including Christmas and Carnival. It is not perfect: it under-forecasts holidays by about 7%, and after the 2016 earthquake every method struggled. The agent reports these weak spots instead of hiding them.

## 4. Agent demo (2:00)

**On screen:** the dashboard. In the **Store** filter choose **Store 1**, then type each question into the chat box.

| # | Type in the chat | Check | Say |
|---|---|---|---|
| 4.1 | `Give me the seven-day forecast for PRODUCE at Store 1.` | p50 about 16,629 units, range about 13,851 to 18,532 | "The agent calls the forecast tool: about 16,600 units next week, with an 80% range." |
| 4.2 | `Is there a stockout risk?` | Tool `get_replenishment` for Store 1 · PRODUCE; risk high | "I didn't repeat the store or product. The agent keeps the context and checks replenishment: current stock won't cover demand until the next delivery." |
| 4.3 | `How many units should I order?` | 2,910 units | "It recommends 2,910 units for the next delivery. That is the p50 forecast plus a buffer learned for produce in our backtest, minus stock on hand." |
| 4.4 | `Which categories should Store 3 replenish first?` | Beverages 63,820, Grocery I 56,690, Dairy 2,920 first | "Now a different store. It ranks categories: high stock-out risk first, then by demand." |

Then click the **Store 44 · Dairy** row in the replenishment table. The chat context changes to it.

| # | Type in the chat | Check | Say |
|---|---|---|---|
| 4.5 | `What if we cancel all promotions next week?` | 17,332 → 16,941 units, about -391 (-2.3%) | "This isn't a fixed percentage. The model re-scores the week with the promotion plan changed: about 391 fewer units." |
| 4.6 | `Why?` | Last week 16,266 units; 4-week average 17,517 | "It explains the forecast: last week's level, the four-week average, the promotion plan and the model's top drivers." |

## 5. Safety and reliability (0:45)

**On screen:** under any answer, open **View evidence** and expand **Show raw result**. Then switch to the terminal.

**Say (dashboard):**
> Every answer shows its evidence: the exact tool call and the raw result it used. Before an answer is shown, a checker compares its numbers, dates and risk levels with the tool results. If the model says something the tools don't support, the planner gets a clearly labelled template built from the same verified data instead.

*If a "[Template fallback]" answer appears during recording, point at it: "That label is this safety check at work. The numbers are still the tool's numbers."*

**In the terminal, type:** `How many units should I order?`
**Check:** `Which store number and product family should I use?`

**Say (terminal):**
> When information is missing, it asks rather than guesses. On the dashboard the selected row provides the context, so this is shown here in the CLI. And FreshFlow never places an order. Every recommendation says that stock and lead times are simulated, and the decision stays with the planner.

## 6. Business impact (0:30)

**On screen:** the "Business value" table in the proposal doc (or `BUSINESS_VALUE.md`).

**Say:**
> We replayed 84 days of real demand through the same ordering process. Today's rule of thumb, a four-week average plus 20%, costs about 4.6 million dollars a year in lost margin, waste and holding. FreshFlow brings that to 2.3 million, a 51% cut, while the fill rate rises from 99.08% to 99.30%. About 0.6 million of the saving comes from better forecasts alone, and the rest from per-category order buffers. Spoiled units drop 82%; bread waste falls from 18% to under 4%. Next, we would connect real stock and supplier data and log planner overrides. Thank you.

## Reference: expected numbers

| Item | Value |
|---|---|
| Backtest WAPE (mean of 4 windows) | 11.7% (dashboard shows 11.69%); naive 19.9%, planner rule 16.3%, v2 15.3% |
| Store 1 · Produce | p50 16,628.8 (p10 13,850.5, p90 18,532.1); order 2,910; stock-out risk high |
| Store 3 priorities | Beverages 63,820 · Grocery I 56,690 · Dairy 2,920 · Poultry 560 · Meats 560 |
| Store 44 · Dairy what-if (no promotions) | 17,332.0 → 16,941.0 units (-391.0, -2.3%) |
| Store 44 · Dairy why | last week 16,266 units; 4-week average 17,517 units |
| Business value (per year) | $4.62M → $2.28M (-51%); forecast accuracy alone $0.61M; spoiled units -82% |
