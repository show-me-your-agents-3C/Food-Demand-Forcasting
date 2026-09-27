"""Train the final v3 models on all history and publish the stable model interface.

    python -m src.food_forecast.train_final

Writes outputs/final/ (read by the replenishment rules, agent and dashboard):
    forecast.csv            next 7 days per store x family: p10 / p50 / p90
    metrics.json            backtest headline + final training facts
    backtest_summary.csv    per-window x per-model scores (copied from model_v3)
    error_analysis.csv      grouped errors (family, promotion, holiday, volume, horizon)
    feature_importance.csv  gain importance of the p50 model
    model_metadata.json     versions, data ranges, features, parameters, schema
    models/*.txt            LightGBM boosters
    future_features.csv.gz  feature rows of the forecast week (for what-if re-scoring)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .backtest_v3 import OUT as BACKTEST_DIR, STRESS_WINDOWS
from .config import PROJECT_ROOT
from .features_v3 import FEATURES, HORIZON, ORIGIN_WEEKDAY, build_features, load_panel
from .modeling_v3 import BASE_PARAMS, COLD_START_DAYS, INTERMITTENT_ZERO_SHARE, MODEL_SPECS, STORE_OPEN_SHARE, fit, intermittent_forecast, predict, route


FINAL = PROJECT_ROOT / "outputs" / "final"
MODEL_VERSION = "v3.0"
# Backtest: TSB 90.0% vs LightGBM 87.5% WAPE on sparse rows -> keep LightGBM, flag the route.
USE_TSB_FOR_INTERMITTENT = False
SCHEMA = {
    "forecast_origin": "Wednesday the order is placed (first forecast day)",
    "date": "target day",
    "horizon_day": "1..7 days ahead of the origin",
    "store_nbr": "store id (Favorita)",
    "family": "food product family",
    "route": "lightgbm | cold_start (<56 trading days: same weekday last week) | intermittent (sparse demand in an open store: low reliability flag)",
    "onpromotion": "planned number of promoted items (from the promotion calendar)",
    "p10": "10th percentile of demand (units)",
    "p50": "median demand forecast (units) - use as the point forecast",
    "p90": "90th percentile of demand (units) - use for safety stock / service level",
    "forecast_method": "model that produced the row",
    "model_version": MODEL_VERSION,
}


def choose_primary() -> tuple[str, dict]:
    """Pick the point model with the lower mean WAPE over the regular backtest windows."""
    metrics = json.loads((BACKTEST_DIR / "metrics.json").read_text())
    candidates = {m: v["mean_wape_regular_windows"] for m, v in metrics["headline"].items() if m in ("v3_l1", "v3_tweedie")}
    return min(candidates, key=candidates.get), metrics


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "models").mkdir(exist_ok=True)
    primary, backtest = choose_primary()

    panel = load_panel(include_future=True)
    frame = build_features(panel)
    last_actual = panel.dates[~np.isnan(panel.sales).all(axis=0)].max()
    origin = last_actual + pd.Timedelta(days=1)
    if origin.dayofweek != ORIGIN_WEEKDAY:
        raise ValueError(f"Forecast origin {origin.date()} is not the ordering weekday")
    end = origin + pd.Timedelta(days=HORIZON - 1)

    train = frame[frame["train_ok"] & (frame["date"] <= last_actual)]
    future = frame[frame["date"].between(origin, end)].copy()
    models = {name: fit(train, spec) for name, spec in (("p50", primary), ("p10", "v3_q10"), ("p90", "v3_q90"))}
    for name, model in models.items():
        future[name] = predict(model, future)
        model.booster_.save_model(str(FINAL / "models" / f"{name}.txt"))

    # Route per series from facts known at the order date (same rule as the backtest).
    future["route"] = route(future["trading_days"], future["zero_frac_28"], future["store_open_frac_28"])
    cold = (future["route"] == "cold_start").to_numpy()
    future.loc[cold, "p50"] = future.loc[cold, "lag_7"]
    future.loc[cold, "p10"] = 0.7 * future.loc[cold, "lag_7"]  # heuristic band: +/-30%
    future.loc[cold, "p90"] = 1.3 * future.loc[cold, "lag_7"]
    intermittent = (future["route"] == "intermittent").to_numpy()
    if USE_TSB_FOR_INTERMITTENT and intermittent.any():
        anchor = int((last_actual - panel.dates[0]).days)
        tsb = intermittent_forecast(panel.sales, np.arange(len(panel.series)), anchor)
        tsb["store_nbr"] = panel.series["store_nbr"].astype(str).to_numpy()
        tsb["family"] = panel.series["family"].to_numpy()
        tsb = tsb.set_index(["store_nbr", "family"])
        keys = list(zip(future.loc[intermittent, "store_nbr"].astype(str), future.loc[intermittent, "family"].astype(str)))
        for column in ("p10", "p50", "p90"):
            future.loc[intermittent, column] = tsb.loc[keys, column].to_numpy()

    # Quantile models are trained separately; enforce p10 <= p50 <= p90.
    stacked = np.sort(future[["p10", "p50", "p90"]].to_numpy(), axis=1)
    stacked[:, 1] = future["p50"].to_numpy()
    stacked[:, 0] = np.minimum(stacked[:, 0], stacked[:, 1])
    stacked[:, 2] = np.maximum(stacked[:, 2], stacked[:, 1])
    future[["p10", "p50", "p90"]] = stacked

    forecast = pd.DataFrame({
        "forecast_origin": origin.strftime("%Y-%m-%d"),
        "date": future["date"].dt.strftime("%Y-%m-%d"),
        "horizon_day": future["horizon_day"].astype(int),
        "store_nbr": future["store_nbr"].astype(int),
        "family": future["family"].astype(str),
        "route": future["route"],
        "onpromotion": future["onpromotion"].astype(int),
        "p10": future["p10"].round(3),
        "p50": future["p50"].round(3),
        "p90": future["p90"].round(3),
        "forecast_method": np.select([cold, intermittent & USE_TSB_FOR_INTERMITTENT], ["same_weekday_last_week", "tsb_intermittent"], default=f"lightgbm_{primary}"),
        "model_version": MODEL_VERSION,
    }).sort_values(["store_nbr", "family", "date"])
    forecast.to_csv(FINAL / "forecast.csv", index=False)
    # Feature rows of the forecast week: lets forecast_tools re-score what-if scenarios.
    future[["date", "route", "trading_days", *FEATURES]].to_csv(FINAL / "future_features.csv.gz", index=False)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance_gain": models["p50"].booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance_gain", ascending=False)
    importance["share"] = (importance["importance_gain"] / importance["importance_gain"].sum()).round(4)
    importance.to_csv(FINAL / "feature_importance.csv", index=False)

    for name in ("backtest_summary.csv", "error_analysis.csv"):
        shutil.copy(BACKTEST_DIR / name, FINAL / name)

    head = backtest["headline"]
    metrics = {
        "model_version": MODEL_VERSION,
        "primary_model": primary,
        "selection_rule": "lower mean WAPE over regular backtest windows among v3_l1 / v3_tweedie",
        "backtest": backtest,
        "headline_mean_wape_regular_windows": {m: v["mean_wape_regular_windows"] for m, v in head.items()},
        "forecast": {
            "origin": origin.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"),
            "rows": len(forecast), "series": int(forecast[["store_nbr", "family"]].drop_duplicates().shape[0]),
            "total_p50_units": round(float(forecast["p50"].sum()), 1),
        },
    }
    (FINAL / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "data_source": "Kaggle Favorita Store Sales (train.csv, test.csv promotions, holidays_events.csv, stores.csv)",
        "granularity": "store x food family x day",
        "series": int(forecast[["store_nbr", "family"]].drop_duplicates().shape[0]),
        "series_by_route": forecast.drop_duplicates(["store_nbr", "family"])["route"].value_counts().to_dict(),
        "training_range": [str(train["date"].min().date()), str(last_actual.date())],
        "training_rows": int(len(train)),
        "forecast_origin": origin.strftime("%Y-%m-%d"),
        "horizon_days": HORIZON,
        "ordering_cycle": "weekly, Wednesday; features use only sales up to the day before the origin",
        "features": FEATURES,
        "models": {
            "p50": {"spec": primary, "objective": MODEL_SPECS[primary][0], "n_estimators": MODEL_SPECS[primary][1]},
            "p10": {"spec": "v3_q10", "objective": MODEL_SPECS["v3_q10"][0], "n_estimators": MODEL_SPECS["v3_q10"][1]},
            "p90": {"spec": "v3_q90", "objective": MODEL_SPECS["v3_q90"][0], "n_estimators": MODEL_SPECS["v3_q90"][1]},
            "shared_params": BASE_PARAMS,
            "routing": {
                "cold_start": f"< {COLD_START_DAYS} trading days at the order date -> same weekday last week (+/-30% band)",
                "intermittent": f">= {INTERMITTENT_ZERO_SHARE:.0%} zero-sales days in last 28 while the store was open "
                                f">= {STORE_OPEN_SHARE:.0%} of them -> LightGBM, flagged low reliability "
                                "(TSB tested in backtest and not better)",
                "lightgbm": "all other series",
            },
        },
        "backtest_windows": backtest["windows"],
        "stress_windows": sorted(STRESS_WINDOWS),
        "forecast_schema": SCHEMA,
        "known_limitations": [
            "Inventory, lead time, shelf life and MOQ are simulated downstream; the model only forecasts demand.",
            "Promotion counts are assumed known in advance (retailer promotion calendar / Kaggle test.csv).",
            "Granularity is product family, not SKU.",
            "Unforeseen shocks (e.g. the 2016 earthquake) are not predictable from history; see stress-test window.",
        ],
    }
    (FINAL / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Wrote {len(forecast)} rows for {origin.date()}..{end.date()} with primary={primary} to {FINAL}")


if __name__ == "__main__":
    main()
