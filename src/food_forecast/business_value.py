"""Translate backtest forecasts into stock-outs, waste and money.

Replays the regular backtest windows day by day for every store x family with
real demand. Each method's forecasts (made at the Wednesday origin, never
updated with actuals inside the week) drive the same replenishment process:

* deliveries every `interval` days (daily for bread/produce, weekly for dry goods);
  order-up-to level = demand forecast over the cover period + safety buffer
* first-in-first-out selling; unmet demand is lost (no back-orders)
* units older than their sellable life are thrown away at the end of the day

Costs: lost gross margin per unit short, unit cost per unit spoiled, 25%/year
holding cost on stock carried overnight. Unit economics and sellable lives are
**assumptions** (Favorita has no prices or inventory); see ECONOMICS.

Safety buffers are set the way a planner could in practice: per family, the
buffer that minimised cost in the *previous* window is used in the next one
(Christmas -> Carnival -> May -> July), identically for every method. Only
the three later windows are scored. Policies:

* current practice: planner rule (avg of last 4 same weekdays) + fixed 20% buffer
* <method> + learned buffer: forecast x (1 + buffer)
* v3 quantile + learned safety factor: p50 + z x sigma, sigma from p10/p90

    python -m src.food_forecast.business_value
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from .backtest_v3 import OUT as BACKTEST_DIR, STRESS_WINDOWS
from .config import PROJECT_ROOT


OUT = PROJECT_ROOT / "outputs" / "business_value"
HOLDING_RATE_PER_DAY = 0.25 / 365
# family: unit cost, selling price (USD per sales unit), sellable days after delivery, delivery every N days
ECONOMICS = {
    "PREPARED FOODS": (2.20, 3.40, 1, 1),
    "BREAD/BAKERY": (0.60, 0.95, 1, 1),
    "SEAFOOD": (4.50, 6.50, 2, 1),
    "PRODUCE": (1.00, 1.45, 3, 1),
    "POULTRY": (2.60, 3.60, 3, 2),
    "MEATS": (3.50, 4.90, 4, 2),
    "DELI": (2.50, 3.60, 4, 2),
    "DAIRY": (0.85, 1.20, 7, 2),
    "EGGS": (2.00, 2.80, 14, 3),
    "FROZEN FOODS": (2.00, 2.90, 60, 7),
    "BEVERAGES": (0.70, 1.00, 120, 7),
    "GROCERY I": (1.10, 1.55, 120, 7),
    "GROCERY II": (1.80, 2.60, 120, 7),
}
POINT_METHODS = {
    "same_weekday_avg_4w": "Planner rule (avg of last 4 same weekdays)",
    "seasonal_naive": "Seasonal naive (last week's same day)",
    "v2_recursive": "LightGBM v2 (recursive)",
    "v3_tweedie": "LightGBM v3 (p50)",
}
BUFFERS = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
SAFETY_Z = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
CURRENT_PRACTICE = ("same_weekday_avg_4w", 0.2)
WINDOW_ORDER = ["2016-12_christmas", "2017-02_carnival", "2017-05_regular", "2017-07_latest"]


def simulate(actual: np.ndarray, order_up_to: np.ndarray, delivery: np.ndarray, shelf: int) -> dict:
    """One series-window. order_up_to[i] is used on delivery days (delivery[i] True)."""
    batches: list[list[float]] = []  # [units, age]
    sold = lost = wasted = delivered = unit_days = 0.0
    for day in range(len(actual)):
        if delivery[day]:
            on_hand = sum(b[0] for b in batches)
            quantity = max(0.0, order_up_to[day] - on_hand)
            if quantity > 0:
                batches.append([quantity, 0])
                delivered += quantity
        demand = actual[day]
        for batch in batches:  # FIFO: oldest first
            if demand <= 0:
                break
            take = min(batch[0], demand)
            batch[0] -= take
            demand -= take
            sold += take
        lost += demand
        kept = []
        for units, age in batches:
            if units <= 1e-9:
                continue
            if age + 1 >= shelf:
                wasted += units
            else:
                kept.append([units, age + 1])
                unit_days += units
        batches = kept
    return {"demand": float(actual.sum()), "sold": sold, "lost": lost, "wasted": wasted,
            "delivered": delivered, "unit_days": unit_days}


def cover_sums(values: np.ndarray, block: np.ndarray, delivery: np.ndarray) -> np.ndarray:
    """Sum of `values` from each delivery day up to the next delivery or the end of the forecast week."""
    out = np.zeros(len(values))
    starts = np.flatnonzero(delivery)
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(values)
        end = min(end, start + int(np.argmax(np.r_[block[start:] != block[start], True])))
        out[start] = values[start:end].sum()
    return out


def load_rows() -> pd.DataFrame:
    data = pd.read_csv(BACKTEST_DIR / "backtest_predictions.csv.gz", parse_dates=["date", "forecast_origin"])
    data = data[~data["window"].isin(STRESS_WINDOWS) & (data["series_status"] == "active")]
    needed = list(POINT_METHODS) + ["v3_q10", "v3_q90"]
    complete = data.groupby(["window", "store_nbr", "family"])[needed].transform(lambda s: s.notna().all()).all(axis=1)
    return data[complete].sort_values(["window", "store_nbr", "family", "date"])


def run(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (window, store, family), part in rows.groupby(["window", "store_nbr", "family"], sort=False):
        cost, price, shelf, interval = ECONOMICS[family]
        actual = part["actual"].to_numpy(float)
        block = part["forecast_origin"].to_numpy()
        delivery = (part["horizon_day"].to_numpy(int) - 1) % interval == 0
        base = {"window": window, "store_nbr": store, "family": family}

        for method in POINT_METHODS:
            cover = cover_sums(part[method].to_numpy(float), block, delivery)
            for buffer in BUFFERS:
                records.append({**base, "policy": method, "setting": buffer,
                                **simulate(actual, cover * (1 + buffer), delivery, shelf)})

        p50 = part["v3_tweedie"].to_numpy(float)
        sigma = (np.maximum(part["v3_q90"].to_numpy(float), p50) - np.minimum(part["v3_q10"].to_numpy(float), p50)) / (2 * norm.ppf(0.9))
        mean_cover = cover_sums(p50, block, delivery)
        sigma_cover = np.sqrt(cover_sums(sigma ** 2, block, delivery))
        for z in SAFETY_Z:
            records.append({**base, "policy": "v3_quantile", "setting": z,
                            **simulate(actual, mean_cover + z * sigma_cover, delivery, shelf)})

    frame = pd.DataFrame(records)
    econ = frame["family"].map(ECONOMICS)
    frame["lost_margin_usd"] = frame["lost"] * econ.map(lambda e: e[1] - e[0])
    frame["waste_cost_usd"] = frame["wasted"] * econ.map(lambda e: e[0])
    frame["holding_cost_usd"] = frame["unit_days"] * econ.map(lambda e: e[0]) * HOLDING_RATE_PER_DAY
    frame["revenue_usd"] = frame["sold"] * econ.map(lambda e: e[1])
    frame["total_cost_usd"] = frame[["lost_margin_usd", "waste_cost_usd", "holding_cost_usd"]].sum(axis=1)
    return frame


def summarize(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    sums = frame.groupby(group, dropna=False)[
        ["demand", "sold", "lost", "wasted", "delivered", "lost_margin_usd", "waste_cost_usd",
         "holding_cost_usd", "total_cost_usd", "revenue_usd"]].sum()
    sums["fill_rate"] = sums["sold"] / sums["demand"]
    sums["waste_share_of_delivered"] = sums["wasted"] / sums["delivered"]
    return sums.reset_index()


def learned(frame: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per family, use the setting that minimised cost in the previous window (no hindsight)."""
    part = frame[frame["policy"] == policy]
    cost = part.groupby(["window", "family", "setting"])["total_cost_usd"].sum().reset_index()
    chosen, picks = [], []
    for previous, current in zip(WINDOW_ORDER[:-1], WINDOW_ORDER[1:]):
        prev = cost[cost["window"] == previous]
        best = prev.loc[prev.groupby("family")["total_cost_usd"].idxmin(), ["family", "setting"]]
        picks.append(best.assign(window=current))
        chosen.append(part[part["window"] == current].merge(best, on=["family", "setting"]))
    return pd.concat(chosen, ignore_index=True), pd.concat(picks, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tic = time.time()
    rows = load_rows()
    frame = run(rows)
    frame.to_csv(OUT / "simulation_detail.csv.gz", index=False)
    summarize(frame, ["window", "policy", "setting"]).to_csv(OUT / "policy_grid.csv", index=False)

    scored = WINDOW_ORDER[1:]
    sim_days = int(rows[rows["window"].isin(scored)].groupby("window")["date"].nunique().sum())
    annualize = 365 / sim_days

    current_label = "Current practice: planner rule + fixed 20% buffer"
    scenarios = {current_label: frame[(frame["policy"] == CURRENT_PRACTICE[0]) & (frame["setting"] == CURRENT_PRACTICE[1]) & frame["window"].isin(scored)]}
    settings = []
    for method, label in POINT_METHODS.items():
        chosen, picks = learned(frame, method)
        scenarios[f"{label} + learned buffer"] = chosen
        settings.append(picks.assign(policy=method))
    chosen, picks = learned(frame, "v3_quantile")
    scenarios["LightGBM v3 p10/p50/p90 + learned safety factor"] = chosen
    settings.append(picks.assign(policy="v3_quantile"))
    pd.concat(settings).to_csv(OUT / "learned_settings.csv", index=False)
    # Buffers for the live forecast week: per family, the v3 setting that cost least in the latest window.
    latest = frame[(frame["policy"] == "v3_tweedie") & (frame["window"] == WINDOW_ORDER[-1])]
    latest = latest.groupby(["family", "setting"])["total_cost_usd"].sum().reset_index()
    latest.loc[latest.groupby("family")["total_cost_usd"].idxmin(), ["family", "setting"]].rename(
        columns={"setting": "buffer"}).assign(learned_on=WINDOW_ORDER[-1]).to_csv(OUT / "production_buffers.csv", index=False)

    comparison = pd.DataFrame([summarize(part.assign(scenario=name), ["scenario"]).iloc[0] for name, part in scenarios.items()])
    for column in ["lost_margin_usd", "waste_cost_usd", "holding_cost_usd", "total_cost_usd", "revenue_usd"]:
        comparison[f"{column}_per_year"] = comparison[column] * annualize
    comparison.to_csv(OUT / "scenario_comparison.csv", index=False)
    table = comparison.set_index("scenario")

    v3_options = ["LightGBM v3 (p50) + learned buffer", "LightGBM v3 p10/p50/p90 + learned safety factor"]
    best_v3 = min(v3_options, key=lambda s: table.loc[s, "total_cost_usd"])
    fair_baseline = "Planner rule (avg of last 4 same weekdays) + learned buffer"
    by_family = pd.concat([
        summarize(scenarios[current_label].assign(scenario="current_practice"), ["scenario", "family"]),
        summarize(scenarios[fair_baseline].assign(scenario="planner_rule_learned_buffer"), ["scenario", "family"]),
        summarize(scenarios[best_v3].assign(scenario="v3_best"), ["scenario", "family"]),
    ])
    by_family.to_csv(OUT / "by_family.csv", index=False)

    def delta(a: str, b: str) -> dict:
        x, y = table.loc[a], table.loc[b]
        return {
            "from": a, "to": b,
            "total_cost_per_year_usd": [round(float(x["total_cost_usd_per_year"])), round(float(y["total_cost_usd_per_year"]))],
            "saving_per_year_usd": round(float(x["total_cost_usd_per_year"] - y["total_cost_usd_per_year"])),
            "total_cost_reduction_pct": round(float(1 - y["total_cost_usd"] / x["total_cost_usd"]) * 100, 1),
            "lost_sales_units_reduction_pct": round(float(1 - y["lost"] / x["lost"]) * 100, 1),
            "waste_units_reduction_pct": round(float(1 - y["wasted"] / x["wasted"]) * 100, 1) if x["wasted"] else None,
            "fill_rate": [round(float(x["fill_rate"]), 4), round(float(y["fill_rate"]), 4)],
            "waste_share_of_delivered": [round(float(x["waste_share_of_delivered"]), 4), round(float(y["waste_share_of_delivered"]), 4)],
        }

    report = {
        "scope": {
            "series": int(rows.groupby(["store_nbr", "family"]).ngroups), "stores": int(rows["store_nbr"].nunique()),
            "calibration_window": WINDOW_ORDER[0], "scored_windows": scored,
            "simulated_days": sim_days, "annualisation_factor": round(annualize, 3),
            "annual_revenue_simulated_usd": round(float(table.loc[best_v3, "revenue_usd_per_year"])),
        },
        "best_v3_policy": best_v3,
        "headline_vs_current_practice": delta(current_label, best_v3),
        "fair_comparison_same_learned_buffer_rule": delta(fair_baseline, best_v3),
        "v2_to_v3": delta("LightGBM v2 (recursive) + learned buffer", best_v3),
        "assumptions": {
            "economics_usd_per_unit": {f: {"cost": c, "price": p, "sellable_days": s, "delivery_every_days": i}
                                       for f, (c, p, s, i) in ECONOMICS.items()},
            "holding_cost": "25% of unit cost per year",
            "stockout_cost": "lost gross margin (price - cost); no back-orders, no goodwill penalty",
            "waste_cost": "unit cost of spoiled units",
            "ordering": "forecasts fixed at the Wednesday origin; deliveries arrive in the morning; FIFO",
            "start_stock": "zero at each window start (same for all policies)",
            "buffers": "learned per family from the previous window, identical rule for every method",
        },
        "runtime_seconds": round(time.time() - tic, 1),
    }
    (OUT / "business_value.json").write_text(json.dumps(report, indent=2))
    show = ["scenario", "fill_rate", "waste_share_of_delivered", "lost_margin_usd_per_year",
            "waste_cost_usd_per_year", "holding_cost_usd_per_year", "total_cost_usd_per_year"]
    print(comparison[show].round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in report.items() if k != "assumptions"}, indent=2))


if __name__ == "__main__":
    main()
