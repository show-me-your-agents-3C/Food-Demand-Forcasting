"""Validate LightGBM in the actual seven-day replenishment setting.

Unlike v1's daily-update test, each 7-day forecast block hides all sales in
the block. Predicted demand is fed back into the rolling-mean feature for the
following days, avoiding target leakage.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_training_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_lightgbm_v2_7day"
TARGET = "sales"
CATEGORICAL = ["store_nbr", "family"]
FEATURES = [
    "store_nbr", "family", "onpromotion", "holiday_event_count",
    "is_national_holiday", "is_national_event", "day_of_week", "month",
    "is_weekend", "sales_lag_7", "sales_lag_14", "sales_rolling_mean_7",
]
HORIZON = 7
ORIGINS = pd.to_datetime(["2017-07-19", "2017-07-26", "2017-08-02", "2017-08-09"])


def metric(actual: pd.Series, prediction: pd.Series | np.ndarray) -> dict[str, float]:
    error = np.asarray(actual, dtype=float) - np.asarray(prediction, dtype=float)
    total = float(np.abs(actual).sum())
    return {
        "mae": round(float(np.abs(error).mean()), 4),
        "rmse": round(float(np.sqrt(np.mean(error ** 2))), 4),
        "wape": round(float(np.abs(error).sum() / total), 6) if total else None,
    }


def feature_frame(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows[FEATURES].copy()
    for col in ("onpromotion", "is_national_holiday", "is_national_event", "is_weekend"):
        result[col] = result[col].astype("int8")
    for col in CATEGORICAL:
        # Keep a shared string representation so train and forecast categories align.
        result[col] = result[col].astype(str).astype("category")
    return result


def fit_model(train: pd.DataFrame) -> LGBMRegressor:
    x_train = feature_frame(train)
    model = LGBMRegressor(
        objective="regression_l1", n_estimators=700, learning_rate=0.05,
        num_leaves=63, min_child_samples=50, subsample=0.85,
        colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbosity=-1,
    )
    model.fit(x_train, train[TARGET], categorical_feature=CATEGORICAL)
    return model


def predict_block(model: LGBMRegressor, known: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Forecast one block without reading actual sales inside that block."""
    end = origin + pd.Timedelta(days=HORIZON - 1)
    past = known[known["date"] < origin]
    future = known[(known["date"] >= origin) & (known["date"] <= end)].copy()
    history: dict[tuple[int, str], deque[float]] = {}
    for key, part in past.groupby(["store_nbr", "family"], sort=False):
        values = part.sort_values("date")[TARGET].tail(14).astype(float).tolist()
        if len(values) != 14:
            raise ValueError(f"Insufficient history for {key} at {origin.date()}")
        history[(int(key[0]), str(key[1]))] = deque(values, maxlen=14)

    predicted_parts = []
    for current_date, day in future.groupby("date", sort=True):
        day = day.copy()
        lag7, lag14, rolling = [], [], []
        for row in day.itertuples(index=False):
            values = history[(int(row.store_nbr), str(row.family))]
            lag7.append(values[-7])
            lag14.append(values[0])
            rolling.append(float(np.mean(list(values)[-7:])))
        day["sales_lag_7"] = lag7
        day["sales_lag_14"] = lag14
        day["sales_rolling_mean_7"] = rolling
        x_day = feature_frame(day)
        # Use the same categories model saw at training time.
        for col in CATEGORICAL:
            categories = model.booster_.pandas_categorical[CATEGORICAL.index(col)]
            x_day[col] = pd.Categorical(x_day[col].astype(str), categories=categories)
        day["seasonal_naive_prediction"] = np.clip(day["sales_lag_7"].to_numpy(float), 0, None)
        day["lightgbm_prediction"] = np.clip(model.predict(x_day), 0, None)
        predicted_parts.append(day)
        for row in day.itertuples(index=False):
            key = (int(row.store_nbr), str(row.family))
            history[key].append(float(row.lightgbm_prediction))
    return pd.concat(predicted_parts, ignore_index=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_PATH, parse_dates=["date"])
    frame = frame.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    train_cutoff = ORIGINS.min() - pd.Timedelta(days=1)
    train = frame[frame["date"] <= train_cutoff].copy()
    model = fit_model(train)

    blocks = [predict_block(model, frame, origin) for origin in ORIGINS]
    result = pd.concat(blocks, ignore_index=True)
    result = result.rename(columns={TARGET: "actual_sales"})
    result["forecast_origin"] = result["date"].map(
        lambda date: max(origin for origin in ORIGINS if origin <= date)
    )
    result["horizon_day"] = (result["date"] - result["forecast_origin"]).dt.days + 1
    result["seasonal_naive_abs_error"] = np.abs(result["actual_sales"] - result["seasonal_naive_prediction"])
    result["lightgbm_abs_error"] = np.abs(result["actual_sales"] - result["lightgbm_prediction"])
    keep = ["forecast_origin", "horizon_day", "date", "store_nbr", "family", "actual_sales", "onpromotion", "seasonal_naive_prediction", "lightgbm_prediction", "seasonal_naive_abs_error", "lightgbm_abs_error"]
    result[keep].to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    by_horizon = []
    for horizon, part in result.groupby("horizon_day"):
        by_horizon.append({
            "horizon_day": int(horizon),
            "n_predictions": int(len(part)),
            "seasonal_naive_wape": metric(part["actual_sales"], part["seasonal_naive_prediction"])["wape"],
            "lightgbm_wape": metric(part["actual_sales"], part["lightgbm_prediction"])["wape"],
        })
    pd.DataFrame(by_horizon).to_csv(OUTPUT_DIR / "metrics_by_horizon.csv", index=False)
    baseline = metric(result["actual_sales"], result["seasonal_naive_prediction"])
    lightgbm = metric(result["actual_sales"], result["lightgbm_prediction"])
    report = {
        "experiment": "model_lightgbm_v2_7day",
        "evaluation": "Four independent 7-day recursive forecast blocks. All sales within each block are hidden; model predictions, not actual sales, update rolling features.",
        "source": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "train_end_date": train_cutoff.strftime("%Y-%m-%d"),
        "forecast_origins": [date.strftime("%Y-%m-%d") for date in ORIGINS],
        "horizon_days": HORIZON,
        "train_rows": int(len(train)),
        "validation_rows": int(len(result)),
        "baseline": {"name": "seasonal_naive_7_day_lag", **baseline},
        "lightgbm": {"name": "global_lightgbm_l1_recursive", **lightgbm},
        "wape_relative_improvement_vs_baseline": round((baseline["wape"] - lightgbm["wape"]) / baseline["wape"], 6),
        "caveat": "Promotion values are treated as known planned inputs, consistent with a retailer's promotion calendar.",
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
