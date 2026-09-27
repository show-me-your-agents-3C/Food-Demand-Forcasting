"""Run the first reproducible LightGBM demand-forecasting backtest.

The evaluation mimics a daily planning job: for each validation day, only
signals available by that morning are used.  Lag features therefore include
observed earlier days, as they would after the previous day's sales close.
The old weekly seasonal-naive method (sales_lag_7) remains the benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_training_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_lightgbm_v1"

TARGET = "sales"
CATEGORICAL = ["store_nbr", "family"]
FEATURES = [
    "store_nbr",
    "family",
    "onpromotion",
    "holiday_event_count",
    "is_national_holiday",
    "is_national_event",
    "day_of_week",
    "month",
    "is_weekend",
    "sales_lag_7",
    "sales_lag_14",
    "sales_rolling_mean_7",
]
VALIDATION_DAYS = 28


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    denominator = float(np.abs(actual).sum())
    return {
        "mae": round(float(np.abs(error).mean()), 4),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
        "wape": round(float(np.abs(error).sum() / denominator), 6) if denominator else None,
    }


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[FEATURES].copy()
    for col in ("onpromotion", "is_national_holiday", "is_national_event", "is_weekend"):
        result[col] = result[col].astype("int8")
    result["store_nbr"] = result["store_nbr"].astype("category")
    result["family"] = result["family"].astype("category")
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    usecols = ["date", TARGET, *FEATURES]
    frame = pd.read_csv(DATA_PATH, usecols=usecols, parse_dates=["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    unique_dates = np.sort(frame["date"].unique())
    if len(unique_dates) <= VALIDATION_DAYS:
        raise ValueError("Not enough dates for the requested validation window.")
    cutoff = pd.Timestamp(unique_dates[-VALIDATION_DAYS - 1])
    train = frame[frame["date"] <= cutoff].copy()
    valid = frame[frame["date"] > cutoff].copy()

    x_train = prepare_features(train)
    x_valid = prepare_features(valid)
    # Align category codes across train/validation for LightGBM.
    for col in CATEGORICAL:
        categories = pd.Index(sorted(set(x_train[col].astype(str)) | set(x_valid[col].astype(str))))
        x_train[col] = pd.Categorical(x_train[col].astype(str), categories=categories)
        x_valid[col] = pd.Categorical(x_valid[col].astype(str), categories=categories)

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, train[TARGET], categorical_feature=CATEGORICAL)

    baseline = np.clip(valid["sales_lag_7"].to_numpy(dtype=float), 0, None)
    prediction = np.clip(model.predict(x_valid), 0, None)
    baseline_result = metrics(valid[TARGET], baseline)
    model_result = metrics(valid[TARGET], prediction)
    wape_improvement = (baseline_result["wape"] - model_result["wape"]) / baseline_result["wape"]

    result = valid[["date", "store_nbr", "family", TARGET, "sales_lag_7", "onpromotion"]].copy()
    result = result.rename(columns={TARGET: "actual_sales", "sales_lag_7": "seasonal_naive_prediction"})
    result["lightgbm_prediction"] = prediction
    result["seasonal_naive_abs_error"] = np.abs(result["actual_sales"] - result["seasonal_naive_prediction"])
    result["lightgbm_abs_error"] = np.abs(result["actual_sales"] - result["lightgbm_prediction"])
    result.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    daily = result.groupby("date", as_index=False).apply(
        lambda part: pd.Series(
            {
                "seasonal_naive_wape": metrics(part["actual_sales"], part["seasonal_naive_prediction"])["wape"],
                "lightgbm_wape": metrics(part["actual_sales"], part["lightgbm_prediction"])["wape"],
            }
        ),
        include_groups=False,
    )
    daily.to_csv(OUTPUT_DIR / "daily_metrics.csv", index=False)

    importance = pd.DataFrame(
        {"feature": FEATURES, "importance_gain": model.booster_.feature_importance(importance_type="gain")}
    ).sort_values("importance_gain", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    report = {
        "experiment": "model_lightgbm_v1",
        "evaluation": "28-day rolling one-day-ahead backtest; validation lag features use sales available after each preceding day closes",
        "source": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "train_end_date": cutoff.strftime("%Y-%m-%d"),
        "validation_start_date": valid["date"].min().strftime("%Y-%m-%d"),
        "validation_end_date": valid["date"].max().strftime("%Y-%m-%d"),
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "baseline": {"name": "seasonal_naive_7_day_lag", **baseline_result},
        "lightgbm": {"name": "global_lightgbm_l1", **model_result},
        "wape_relative_improvement_vs_baseline": round(float(wape_improvement), 6),
        "feature_count": len(FEATURES),
        "model_parameters": model.get_params(),
        "caveat": "This is a daily-update evaluation, not a fully recursive seven-day-ahead simulation.",
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
