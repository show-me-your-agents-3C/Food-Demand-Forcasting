"""Multi-window seven-day backtest: baselines vs LightGBM v2 (recursive) vs v3 (direct).

Each window retrains every model on data strictly before its first forecast
origin, then issues four weekly Wednesday forecasts of seven days each. Results
are cached per window and model under outputs/model_v3/backtest/, so a crashed
or interrupted run resumes where it stopped.

    python -m src.food_forecast.backtest_v3 --part v2      # recursive v2 reference
    python -m src.food_forecast.backtest_v3 --part v3      # baselines + v3 models
    python -m src.food_forecast.backtest_v3 --part report  # merge, score, analyse
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .features_v3 import FEATURES, HORIZON, ORIGIN_WEEKDAY, PROMO_FEATURES, build_features, load_panel, promo_features
from .modeling_v3 import fit, intermittent_forecast, predict, route, score


OUT = PROJECT_ROOT / "outputs" / "model_v3"
CACHE = OUT / "backtest"
N_BLOCKS = 4
WINDOWS = {
    "2016-04_earthquake": "2016-04-13",  # stress test: 7.8 earthquake on 2016-04-16
    "2016-12_christmas": "2016-11-30",
    "2017-02_carnival": "2017-02-22",
    "2017-05_regular": "2017-05-03",
    "2017-07_latest": "2017-07-19",
}
STRESS_WINDOWS = {"2016-04_earthquake"}
V3_MODELS = ["v3_l1", "v3_tweedie", "v3_q10", "v3_q90", "v3_l1_promo_bool"]
KEYS = ["date", "store_nbr", "family"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def window_dates(name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(WINDOWS[name])
    if start.dayofweek != ORIGIN_WEEKDAY:
        raise ValueError(f"{name}: forecast origins must fall on the ordering weekday")
    return start, start + pd.Timedelta(days=N_BLOCKS * HORIZON - 1)


def cache_path(window: str, model: str):
    return CACHE / f"{window}__{model}.csv.gz"


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[KEYS].copy()
    keys["store_nbr"] = keys["store_nbr"].astype(str)
    keys["family"] = keys["family"].astype(str)
    return keys


def run_v2() -> None:
    from .train_lightgbm_v2_7day import DATA_PATH, fit_model, predict_block

    frame = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values(KEYS).reset_index(drop=True)
    for window in WINDOWS:
        path = cache_path(window, "v2_recursive")
        if path.exists():
            continue
        start, _ = window_dates(window)
        tic = time.time()
        model = fit_model(frame[frame["date"] < start])
        origins = [start + pd.Timedelta(days=HORIZON * b) for b in range(N_BLOCKS)]
        result = pd.concat([predict_block(model, frame, origin) for origin in origins], ignore_index=True)
        out = _key_frame(result)
        out["v2_recursive"] = result["lightgbm_prediction"].to_numpy()
        out.to_csv(path, index=False)
        log(f"v2 {window}: {len(out)} rows in {time.time() - tic:.0f}s")


def run_v3(models: list[str]) -> None:
    panel = load_panel(include_future=False)
    frame = build_features(panel)
    bool_promo = promo_features((panel.promo > 0).astype(float), *_day_anchor(panel))

    for window in WINDOWS:
        start, end = window_dates(window)
        in_window = frame["date"].between(start, end) & frame["sales"].notna()
        evaluation = frame[in_window]
        base_path = cache_path(window, "base")
        if not base_path.exists():
            base = _key_frame(evaluation)
            base["forecast_origin"] = evaluation["date"] - pd.to_timedelta(evaluation["horizon_day"].astype(int) - 1, unit="D")
            base["route"] = route(evaluation["trading_days"], evaluation["zero_frac_28"], evaluation["store_open_frac_28"])
            for column in ["series_status", "trading_days", "horizon_day", "onpromotion", "is_holiday", "is_national_event", "sales"]:
                base[column] = evaluation[column].to_numpy()
            base = base.rename(columns={"sales": "actual"})
            # Volume tier uses only sales before the window (per series, 56-day mean).
            first_rows = evaluation[evaluation["date"] == start]
            pre_mean = dict(zip(zip(first_rows["store_nbr"].astype(str), first_rows["family"].astype(str)), first_rows["mean_56"]))
            base["pre_window_daily_mean"] = [pre_mean.get(k, np.nan) for k in zip(base["store_nbr"], base["family"])]
            base["seasonal_naive"] = evaluation["lag_7"].to_numpy()
            base["same_weekday_avg_4w"] = evaluation["same_dow_mean_4"].to_numpy()
            base["tsb_intermittent"] = _intermittent_window(panel, evaluation, start)
            base.to_csv(base_path, index=False)
            log(f"base {window}: {len(base)} rows")

        # All series with 56+ trading days (includes new stores once established).
        train_mask = frame["train_ok"] & (frame["date"] < start)
        for model_name in models:
            path = cache_path(window, model_name)
            if path.exists():
                continue
            tic = time.time()
            if model_name == "v3_l1_promo_bool":
                # Ablation: promotions as a yes/no flag, as in the v1/v2 cleaning step.
                swapped = frame[PROMO_FEATURES].copy()
                for column, values in bool_promo.items():
                    frame[column] = values.ravel().astype("float32")
                model = fit(frame[train_mask], "v3_l1")
                prediction = predict(model, frame[in_window])
                frame[PROMO_FEATURES] = swapped
            else:
                model = fit(frame[train_mask], model_name)
                prediction = predict(model, evaluation)
            out = _key_frame(evaluation)
            out[model_name] = prediction
            out.to_csv(path, index=False)
            log(f"{model_name} {window}: trained on {int(train_mask.sum())} rows in {time.time() - tic:.0f}s")


def _day_anchor(panel) -> tuple[np.ndarray, np.ndarray]:
    from .features_v3 import MIN_HISTORY_DAYS
    days = np.arange(MIN_HISTORY_DAYS, panel.sales.shape[1])
    horizon = ((panel.dates[days].dayofweek.to_numpy() - ORIGIN_WEEKDAY) % 7) + 1
    return days, days - horizon


def _intermittent_window(panel, evaluation: pd.DataFrame, start: pd.Timestamp) -> np.ndarray:
    """TSB forecast for every eval row (used for intermittent series in the full system)."""
    key_rows = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(panel.series["store_nbr"], panel.series["family"]))}
    rows = np.arange(len(panel.series))
    result = np.zeros(len(evaluation))
    origin_of_row = evaluation["date"] - pd.to_timedelta(evaluation["horizon_day"].astype(int) - 1, unit="D")
    series_row = np.array([key_rows[(str(s), str(f))] for s, f in zip(evaluation["store_nbr"], evaluation["family"])])
    for origin in origin_of_row.unique():
        anchor = int((pd.Timestamp(origin) - panel.dates[0]).days) - 1
        forecast = intermittent_forecast(panel.sales, rows, anchor).set_index("row")["p50"]
        mask = (origin_of_row == origin).to_numpy()
        result[mask] = forecast.loc[series_row[mask]].to_numpy()
    return result


def _groups(frame: pd.DataFrame) -> dict[str, pd.Series]:
    # Terciles of series (not rows) by their pre-window daily mean, per window.
    series = frame[["window", "store_nbr", "family", "pre_window_daily_mean"]].drop_duplicates(["window", "store_nbr", "family"])
    series["volume_tier"] = series.groupby("window")["pre_window_daily_mean"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    )
    tiers = frame[["window", "store_nbr", "family"]].merge(series, on=["window", "store_nbr", "family"], how="left")["volume_tier"].to_numpy()
    return {
        "window": frame["window"],
        "family": frame["family"],
        "promotion": np.where(frame["onpromotion"] > 0, "on_promotion", "no_promotion"),
        "holiday": np.where((frame["is_holiday"] > 0) | (frame["is_national_event"] > 0), "holiday_or_event", "normal_day"),
        "volume_tier": tiers,
        "horizon_day": frame["horizon_day"].astype(int).astype(str),
    }


def run_report() -> None:
    frames = []
    for window in WINDOWS:
        merged = pd.read_csv(cache_path(window, "base"), parse_dates=["date", "forecast_origin"], dtype={"store_nbr": str})
        for path in sorted(CACHE.glob(f"{window}__*.csv.gz")):
            model = path.name.split("__")[1].removesuffix(".csv.gz")
            if model == "base":
                continue
            part = pd.read_csv(path, parse_dates=["date"], dtype={"store_nbr": str})
            merged = merged.merge(part, on=KEYS, how="left", validate="one_to_one")
        merged.insert(0, "window", window)
        frames.append(merged)
    data = pd.concat(frames, ignore_index=True)
    data["store_nbr"] = data["store_nbr"].astype(str)
    point_models = [m for m in ["seasonal_naive", "same_weekday_avg_4w", "v2_recursive", "v3_l1", "v3_tweedie", "v3_l1_promo_bool"] if m in data]

    # Full system: route per series at the order date (see modeling_v3.route).
    intermittent = data["series_status"] != "active"  # registry label; v2 only covers "active"
    # The point model inside the system is chosen like train_final: lower mean WAPE (regular windows).
    point = [m for m in ("v3_l1", "v3_tweedie") if m in data]
    if point:
        regular_rows = ~data["window"].isin(STRESS_WINDOWS) & ~intermittent
        def mean_wape(model):
            return np.mean([score(p["actual"], p[model])["wape"] for _, p in data[regular_rows].groupby("window")])
        system_point = min(point, key=mean_wape)
        # Intermittent rows stay on the model: TSB did not beat it in this backtest (see route_intermittent).
        data["v3_system"] = np.where(data["route"] == "cold_start", data["seasonal_naive"], data[system_point])
    data.to_csv(OUT / "backtest_predictions.csv.gz", index=False)

    active = data[~intermittent]
    summary = []
    for window, part in active.groupby("window", sort=False):
        for model in point_models:
            if part[model].notna().all():
                summary.append({"window": window, "stress_test": window in STRESS_WINDOWS, "model": model, **score(part["actual"], part[model])})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "backtest_summary.csv", index=False)

    regular = active[~active["window"].isin(STRESS_WINDOWS)]
    analysis = []
    for group_type, labels in _groups(regular).items():
        for label, part in regular.groupby(np.asarray(labels)):
            for model in point_models:
                analysis.append({"group_type": group_type, "group": label, "model": model, **score(part["actual"], part[model])})
    pd.DataFrame(analysis).to_csv(OUT / "error_analysis.csv", index=False)

    headline = {}
    for model in point_models:
        rows = summary[(summary["model"] == model) & ~summary["stress_test"]]
        stress = summary[(summary["model"] == model) & summary["stress_test"]]
        if rows.empty:
            continue
        headline[model] = {
            "mean_wape_regular_windows": round(float(rows["wape"].mean()), 6),
            "best_window": rows.loc[rows["wape"].idxmin(), ["window", "wape"]].to_dict(),
            "worst_window": rows.loc[rows["wape"].idxmax(), ["window", "wape"]].to_dict(),
            "pooled_regular": score(regular["actual"], regular[model]),
            "stress_test_wape": None if stress.empty else float(stress["wape"].iloc[0]),
        }
    coverage = {}
    if {"v3_q10", "v3_q90"} <= set(data):
        regular_all = data[~data["window"].isin(STRESS_WINDOWS) & ~intermittent]
        low = np.minimum(regular_all["v3_q10"], regular_all["v3_q90"])
        high = np.maximum(regular_all["v3_q10"], regular_all["v3_q90"])
        coverage = {
            "p10_p90_interval_coverage": round(float(((regular_all["actual"] >= low) & (regular_all["actual"] <= high)).mean()), 4),
            "share_actual_below_p90": round(float((regular_all["actual"] <= high).mean()), 4),
            "share_actual_below_p10": round(float((regular_all["actual"] < low).mean()), 4),
            "target": {"interval": 0.8, "below_p90": 0.9, "below_p10": 0.1},
        }
    system = {}
    if "v3_system" in data:
        regular_all = data[~data["window"].isin(STRESS_WINDOWS)]
        system = {
            "system_point_model": system_point,
            "all_702_series": {m: score(regular_all["actual"], regular_all[m]) for m in ("seasonal_naive", system_point, "v3_system")},
            "rows_by_route": regular_all["route"].value_counts().to_dict(),
        }
        for name, part in regular_all.groupby("route"):
            system[f"route_{name}"] = {m: score(part["actual"], part[m]) for m in ("seasonal_naive", "tsb_intermittent", system_point, "v3_system")}
    report = {
        "evaluation": (
            "Per window: retrain on data before the first origin, then four weekly Wednesday "
            "origins x 7-day horizon. v3 uses only sales known at the origin (direct, non-recursive). "
            "Scores on active series unless stated; stress window reported separately."
        ),
        "windows": {name: [str(d.date()) for d in window_dates(name)] for name in WINDOWS},
        "stress_windows": sorted(STRESS_WINDOWS),
        "headline": headline,
        "quantile_calibration": coverage,
        "full_system": system,
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    print(summary.pivot(index="model", columns="window", values="wape").to_string())
    print(json.dumps(report["headline"], indent=2, default=str))
    print(json.dumps({"coverage": coverage, "system": system}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["v2", "v3", "report"], required=True)
    parser.add_argument("--models", nargs="*", default=V3_MODELS)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    {"v2": run_v2, "v3": lambda: run_v3(args.models), "report": run_report}[args.part]()


if __name__ == "__main__":
    main()
