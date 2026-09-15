"""Explainable seasonal-naive demand forecasting and time-based evaluation."""

import numpy as np
import pandas as pd


KEYS = ["store_nbr", "family"]


def seasonal_naive(history: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """Repeat the most recent complete 7-day pattern for each store-family."""
    rows: list[dict] = []
    for key, group in history.groupby(KEYS):
        group = group.sort_values("date")
        recent = group.tail(7)
        if len(recent) < 7:
            continue
        demand_std = float(group.tail(28)["sales"].std(ddof=0) or 0.0)
        last_date = group["date"].max()
        for step, (_, source) in enumerate(recent.iterrows(), start=1):
            predicted = max(0.0, float(source["sales"]))
            rows.append({
                "date": last_date + pd.Timedelta(days=step),
                "store_nbr": key[0], "family": key[1],
                "forecast_sales": predicted,
                "lower_bound": max(0.0, predicted - 1.28 * demand_std),
                "upper_bound": predicted + 1.28 * demand_std,
                "demand_std_28d": demand_std,
                "forecast_method": "seasonal_naive_7d",
            })
    return pd.DataFrame(rows)


def backtest(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    prediction = seasonal_naive(train, horizon_days=7)
    actual = validation.rename(columns={"sales": "actual_sales"})
    merged = prediction.merge(actual[["date", *KEYS, "actual_sales"]], on=["date", *KEYS], how="inner")
    if merged.empty:
        return {"mae": None, "wape": None, "n_predictions": 0}
    absolute_error = (merged["actual_sales"] - merged["forecast_sales"]).abs()
    denominator = merged["actual_sales"].abs().sum()
    return {
        "mae": round(float(absolute_error.mean()), 4),
        "wape": round(float(absolute_error.sum() / denominator), 4) if denominator else None,
        "n_predictions": int(len(merged)),
    }
