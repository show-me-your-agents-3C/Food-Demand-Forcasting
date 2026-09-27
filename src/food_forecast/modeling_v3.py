"""Shared model definitions, metrics and the intermittent-demand fallback for v3."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .features_v3 import CATEGORICAL, FEATURES


BASE_PARAMS = dict(
    learning_rate=0.05, num_leaves=63, min_child_samples=50,
    subsample=0.85, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0,
    random_state=42, n_jobs=2, verbosity=-1,
)
MODEL_SPECS = {
    # name: (LightGBM objective params, number of trees)
    "v3_l1": ({"objective": "regression_l1"}, 700),
    "v3_tweedie": ({"objective": "tweedie", "tweedie_variance_power": 1.2}, 700),
    "v3_q10": ({"objective": "quantile", "alpha": 0.1}, 400),
    "v3_q90": ({"objective": "quantile", "alpha": 0.9}, 400),
}


def fit(train: pd.DataFrame, spec: str, features: list[str] = FEATURES) -> LGBMRegressor:
    params, trees = MODEL_SPECS[spec]
    model = LGBMRegressor(n_estimators=trees, **BASE_PARAMS, **params)
    model.fit(train[features], train["sales"], categorical_feature=[c for c in CATEGORICAL if c in features])
    return model


def predict(model: LGBMRegressor, frame: pd.DataFrame, features: list[str] = FEATURES) -> np.ndarray:
    return np.clip(model.predict(frame[features]), 0, None)


def score(actual, predicted) -> dict[str, float]:
    """WAPE = sum|e| / sum(actual); bias = sum(pred - actual) / sum(actual) (+ = over-forecast)."""
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    total = actual.sum()
    return {
        "n": int(len(actual)),
        "actual_units": round(float(total), 1),
        "mae": round(float(np.abs(predicted - actual).mean()), 4),
        "wape": round(float(np.abs(predicted - actual).sum() / total), 6) if total else None,
        "bias": round(float((predicted - actual).sum() / total), 6) if total else None,
    }


COLD_START_DAYS = 56
INTERMITTENT_ZERO_SHARE = 0.5
STORE_OPEN_SHARE = 0.8


def route(trading_days, zero_frac_28, store_open_frac_28) -> np.ndarray:
    """Pick the forecaster per row from facts known at the order date.

    - cold_start: fewer than 56 trading days (new store / new line) -> same weekday last week
    - intermittent: >= 50% zero-sales days in the last 28 while the store itself was
      open >= 80% of them (slow demand, not a store closure). Still forecast by
      LightGBM (it beat TSB in the backtest) but flagged as low reliability.
    - lightgbm: everything else, including stores reopening after a closure
    """
    trading_days = np.asarray(trading_days)
    sparse = (np.asarray(zero_frac_28) >= INTERMITTENT_ZERO_SHARE) & (np.asarray(store_open_frac_28) >= STORE_OPEN_SHARE)
    return np.select(
        [trading_days < COLD_START_DAYS, sparse],
        ["cold_start", "intermittent"], default="lightgbm",
    )


def tsb_forecast(history: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> float:
    """Teunter-Syntetos-Babai: smoothed demand size x smoothed demand probability.

    Used for intermittent series (and returns 0 if the last 28 days were all zero),
    so slow movers still receive a forecast instead of being silently dropped.
    """
    history = np.nan_to_num(np.asarray(history, float))
    if len(history) == 0 or history[-28:].sum() == 0:
        return 0.0
    nonzero = history[history > 0]
    size = nonzero[0] if len(nonzero) else 0.0
    prob = float(history[0] > 0)
    for value in history[1:]:
        occurred = value > 0
        prob += beta * (occurred - prob)
        if occurred:
            size += alpha * (value - size)
    return float(size * prob)


def intermittent_forecast(panel_sales: np.ndarray, rows: np.ndarray, anchor: int) -> pd.DataFrame:
    """Flat TSB forecast plus an empirical 10-90% band from the trailing 56 days."""
    records = []
    for row in rows:
        history = panel_sales[row, : anchor + 1]
        recent = np.nan_to_num(history[-56:])
        p50 = tsb_forecast(history)
        records.append({
            "row": int(row), "p50": p50,
            "p10": float(np.quantile(recent, 0.1)) if p50 else 0.0,
            "p90": max(p50, float(np.quantile(recent, 0.9))) if p50 else 0.0,
        })
    return pd.DataFrame(records)
