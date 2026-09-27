"""Agent-facing tools over the published v3 model artifacts (outputs/final/).

Every function takes plain JSON-able arguments and returns a JSON-able dict,
so an agent (LangGraph, plain tool-calling, or the JSON-text protocol used with
the LLM gateway) can call them directly. Numbers come only from the model
artifacts; the agent should verbalise them, never invent them.

    from src.food_forecast.forecast_tools import TOOLS, TOOL_SPECS, call_tool
    call_tool("what_if_promotion", {"store_nbr": 44, "family": "DAIRY", "onpromotion": 40})

CLI demo:  python -m src.food_forecast.forecast_tools
"""

from __future__ import annotations

import json
from functools import lru_cache

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .features_v3 import CATEGORICAL, FEATURES
from .inventory import DEFAULT_SCENARIO, SCENARIOS, _round_up


FINAL = PROJECT_ROOT / "outputs" / "final"
QUANTILES = ("p10", "p50", "p90")
DRIVER_NAMES = {
    "onpromotion": "planned promotion (items on promotion that day)",
    "promo_sum_7": "promotion intensity over the last 7 days",
    "promo_sum_14": "promotion intensity over the last 14 days",
    "promo_vs_hist": "promotion level vs the series' usual level",
    "promo_mean_28_hist": "usual promotion level (last 28 days)",
    "promo_lag_7": "promotion one week earlier",
    "mean_7": "last week's average sales",
    "mean_14": "last 2 weeks' average sales",
    "mean_28": "last 4 weeks' average sales",
    "mean_56": "last 8 weeks' average sales",
    "last_sales": "latest known day's sales",
    "trend_7_28": "recent trend (last week vs last 4 weeks)",
    "same_dow_mean_4": "same weekday, last 4 weeks",
    "lag_7": "same weekday last week", "lag_14": "same weekday 2 weeks ago",
    "lag_21": "same weekday 3 weeks ago", "lag_28": "same weekday 4 weeks ago",
    "std_28": "recent volatility", "zero_frac_28": "share of zero-sales days",
    "day_of_week": "day of week", "day_of_month": "day of month", "month": "month / season",
    "days_since_payday": "days since payday (15th / month end)", "is_payday": "payday",
    "is_holiday": "local/regional/national holiday", "holiday_scope": "holiday scope",
    "days_to_holiday": "days until next holiday", "days_since_holiday": "days since last holiday",
    "is_national_event": "national event", "is_workday": "make-up work day",
    "horizon_day": "days ahead", "store_nbr": "store", "family": "product family",
    "store_type": "store type", "store_cluster": "store cluster", "city": "city",
}


# ---------------------------------------------------------------- artifacts
@lru_cache(maxsize=1)
def _forecast() -> pd.DataFrame:
    return pd.read_csv(FINAL / "forecast.csv", parse_dates=["date"])


@lru_cache(maxsize=1)
def _future_features() -> pd.DataFrame:
    frame = pd.read_csv(FINAL / "future_features.csv.gz", parse_dates=["date"])
    for column in CATEGORICAL:
        frame[column] = frame[column].astype(str).astype("category")
    return frame


@lru_cache(maxsize=1)
def _boosters() -> dict[str, lgb.Booster]:
    return {name: lgb.Booster(model_file=str(FINAL / "models" / f"{name}.txt")) for name in QUANTILES}


@lru_cache(maxsize=1)
def _errors() -> pd.DataFrame:
    return pd.read_csv(FINAL / "error_analysis.csv")


@lru_cache(maxsize=1)
def _metadata() -> dict:
    return json.loads((FINAL / "model_metadata.json").read_text())


FAMILY_ALIASES = {
    "饮料": "BEVERAGES", "饮品": "BEVERAGES", "乳制品": "DAIRY", "奶制品": "DAIRY",
    "乳品": "DAIRY", "面包烘焙": "BREAD/BAKERY", "烘焙": "BREAD/BAKERY",
    "蔬果": "PRODUCE", "果蔬": "PRODUCE", "海鲜": "SEAFOOD", "水产": "SEAFOOD",
    "冷冻食品": "FROZEN FOODS", "冻品": "FROZEN FOODS", "禽肉": "POULTRY",
    "熟食": "DELI", "鸡蛋": "EGGS", "预制食品": "PREPARED FOODS",
    "杂货一": "GROCERY I", "杂货二": "GROCERY II",
}


def _normalize_family(family: str) -> str:
    if not isinstance(family, str) or not family.strip():
        raise ValueError("family must be a non-empty food family name")
    if family.strip() in {"肉类", "肉"}:
        raise ValueError("Ambiguous family; choose MEATS or POULTRY (禽肉)")
    value = FAMILY_ALIASES.get(family.strip(), family.strip().upper())
    available = sorted(_forecast()["family"].astype(str).unique())
    if value not in available:
        raise ValueError(f"Unknown family {family!r}. Available families: {available}")
    return value


def _validate_store(store_nbr: int) -> int:
    if isinstance(store_nbr, bool) or not isinstance(store_nbr, int):
        raise ValueError("store_nbr must be an integer")
    available = sorted(_forecast()["store_nbr"].astype(int).unique().tolist())
    if store_nbr not in available:
        raise ValueError(f"No forecast for store {store_nbr}. Available stores: {available}")
    return store_nbr


def _filter_dates(rows: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    try:
        start = pd.Timestamp(start_date) if start_date else rows["date"].min()
        end = pd.Timestamp(end_date) if end_date else rows["date"].max()
    except (TypeError, ValueError) as error:
        raise ValueError("Dates must use YYYY-MM-DD format") from error
    for supplied, parsed in ((start_date, start), (end_date, end)):
        if supplied and parsed.strftime("%Y-%m-%d") != supplied:
            raise ValueError("Dates must use YYYY-MM-DD format")
    available_start, available_end = rows["date"].min(), rows["date"].max()
    if start > available_end or end < available_start:
        available = f"{available_start:%Y-%m-%d}..{available_end:%Y-%m-%d}"
        raise ValueError(f"No forecast rows in {start:%Y-%m-%d}..{end:%Y-%m-%d}; available range: {available}")
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    selected = rows[rows["date"].between(start, end)].copy()
    if selected.empty:
        available = f"{rows['date'].min():%Y-%m-%d}..{rows['date'].max():%Y-%m-%d}"
        raise ValueError(f"No forecast rows in {start:%Y-%m-%d}..{end:%Y-%m-%d}; available range: {available}")
    return selected


def _series(store_nbr: int, family: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    store_nbr = _validate_store(store_nbr)
    family = _normalize_family(family)
    forecast = _forecast()
    rows = forecast[(forecast["store_nbr"] == store_nbr) & (forecast["family"] == family)]
    if rows.empty:
        available = sorted(forecast[forecast["store_nbr"] == store_nbr]["family"].unique())
        raise ValueError(f"No forecast for store {store_nbr} / {family}. Available families for this store: {available}")
    return _filter_dates(rows.sort_values("date"), start_date, end_date)


def _predict(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    boosters = _boosters()
    values = {name: np.clip(boosters[name].predict(rows[FEATURES]), 0, None) for name in QUANTILES}
    values["p10"] = np.minimum(values["p10"], values["p50"])
    values["p90"] = np.maximum(values["p90"], values["p50"])
    return values


def _r(value: float) -> float:
    return round(float(value), 1)


# ---------------------------------------------------------------- tools
def _source_facts(rows: pd.DataFrame, sources: list[str] | None = None) -> dict:
    metadata = _metadata()
    spec = metadata.get("models", {}).get("p50", {}).get("spec", "unknown")
    origins = rows["forecast_origin"].astype(str).unique()
    versions = rows["model_version"].astype(str).unique()
    if len(origins) != 1 or len(versions) != 1:
        raise ValueError("Selected forecast rows contain inconsistent origins or model versions")
    if metadata.get("forecast_origin") and origins[0] != metadata["forecast_origin"]:
        raise ValueError("Forecast rows do not match the origin in model metadata")
    metrics = json.loads((FINAL / "metrics.json").read_text())
    coverage = metrics.get("backtest", {}).get("quantile_calibration", {}).get("p10_p90_interval_coverage")
    return {
        "source_files": sources or ["outputs/final/forecast.csv", "outputs/final/model_metadata.json", "outputs/final/metrics.json"],
        "data_snapshot_through": (metadata.get("training_range") or [None, None])[-1],
        "forecast_origin": origins[0],
        "forecast_date_range": [rows["date"].min().strftime("%Y-%m-%d"), rows["date"].max().strftime("%Y-%m-%d")],
        "model_id": f"{metadata.get('model_version', 'unknown')}:{spec}",
        "backtest_interval_coverage_pct": _r(coverage * 100) if coverage is not None else "unknown",
        "assumptions": "Forecast uses archived Favorita promotion/calendar inputs. Inventory, lead time, shelf life, and MOQ are simulated scenarios, not live operations.",
    }


def get_forecast(store_nbr: int, family: str, start_date: str | None = None, end_date: str | None = None) -> dict:
    """Daily 7-day forecast (median and 80% range) for one store x family."""
    family = _normalize_family(family)
    rows = _series(store_nbr, family, start_date, end_date)
    return {
        "store_nbr": int(store_nbr), "family": family,
        "forecast_origin": str(rows["forecast_origin"].iloc[0]),
        "method": rows["forecast_method"].iloc[0], "model_version": rows["model_version"].iloc[0],
        "route": rows["route"].iloc[0],
        **({"warning": "Sparse, low-volume series: expect large percentage errors."} if rows["route"].iloc[0] == "intermittent" else {}),
        "total": {"p10": _r(rows["p10"].sum()), "p50": _r(rows["p50"].sum()), "p90": _r(rows["p90"].sum())},
        "days": len(rows),
        "daily": [
            {"date": d.strftime("%Y-%m-%d"), "weekday": d.strftime("%a"), "onpromotion": int(p),
             "p10": _r(a), "p50": _r(b), "p90": _r(c)}
            for d, p, a, b, c in zip(rows["date"], rows["onpromotion"], rows["p10"], rows["p50"], rows["p90"])
        ],
        "note": "p50 is the point forecast. p10-p90 is a model interval; historical backtest coverage is reported in data.backtest_interval_coverage_pct and is not a guarantee.",
        "data": _source_facts(rows),
    }


def _replenishment_rows(
    store_nbr: int | None = None,
    family: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    forecast = _forecast().copy()
    features = _future_features()
    if store_nbr is not None:
        store_nbr = _validate_store(store_nbr)
        forecast = forecast[forecast["store_nbr"] == store_nbr]
    if family is not None:
        family = _normalize_family(family)
        forecast = forecast[forecast["family"] == family]
    if forecast.empty:
        raise ValueError("No matching forecast rows. Check store/family; query available forecast data.")
    if start_date or end_date:
        forecast = _filter_dates(forecast, start_date, end_date)
    forecast_keys = set(zip(
        forecast["store_nbr"].astype(int), forecast["family"].astype(str), forecast["date"].dt.strftime("%Y-%m-%d"),
    ))
    feature_keys = set(zip(
        features["store_nbr"].astype(int), features["family"].astype(str), features["date"].dt.strftime("%Y-%m-%d"),
    ))
    if not forecast_keys.issubset(feature_keys):
        raise ValueError("Forecast rows and future_features.csv.gz have inconsistent store/family/date keys")
    anchor = features[features["horizon_day"] == 1][["store_nbr", "family", "mean_7"]].copy()
    # Feature keys are categorical strings; align with forecast.csv before merging.
    anchor["store_nbr"] = anchor["store_nbr"].astype(int)
    anchor["family"] = anchor["family"].astype(str)
    rows = forecast.groupby(["store_nbr", "family"], as_index=False).agg(
        forecast_7d_p50=("p50", "sum"), forecast_7d_p90=("p90", "sum"),
        forecast_origin=("forecast_origin", "first"), start_date=("date", "min"), end_date=("date", "max"),
    ).merge(anchor, on=["store_nbr", "family"], how="left", validate="one_to_one")
    records = []
    for row in rows.to_dict("records"):
        scenario = SCENARIOS.get(row["family"], DEFAULT_SCENARIO)
        days = (pd.Timestamp(row["end_date"]) - pd.Timestamp(row["start_date"])).days + 1
        daily = row["forecast_7d_p50"] / days
        uncertainty_buffer = max(0.0, row["forecast_7d_p90"] - row["forecast_7d_p50"])
        safety_stock = uncertainty_buffer * min(scenario["lead_time_days"] / 7, 1)
        current_stock = max(0.0, float(row["mean_7"]) * 1.5)
        reorder_point = daily * scenario["lead_time_days"] + safety_stock
        sellable_before_expiry = daily * min(scenario["shelf_life_days"], days)
        desired_stock = min(row["forecast_7d_p50"] + safety_stock, sellable_before_expiry + safety_stock)
        order_qty = _round_up(max(0.0, desired_stock - current_stock), scenario["moq"])
        post_order_stock = current_stock + order_qty
        stockout_risk = "high" if current_stock < reorder_point else "low"
        waste_risk = "high" if post_order_stock > sellable_before_expiry * 1.15 else "low"
        item_forecast = forecast[(forecast["store_nbr"] == row["store_nbr"]) & (forecast["family"] == row["family"])]
        records.append({
            "store_nbr": int(row["store_nbr"]), "family": row["family"],
            "forecast_7d_p50": _r(row["forecast_7d_p50"]), "forecast_7d_p90": _r(row["forecast_7d_p90"]),
            "current_stock_simulated": _r(current_stock), "safety_stock_simulated": _r(safety_stock),
            "reorder_point_simulated": _r(reorder_point), "recommended_order_qty": int(order_qty),
            "stockout_risk_simulated": stockout_risk, "waste_risk_simulated": waste_risk,
            "action": "order_today" if stockout_risk == "high" else "monitor",
            "lead_time_days_simulated": scenario["lead_time_days"],
            "shelf_life_days_simulated": scenario["shelf_life_days"], "moq_simulated": scenario["moq"],
            "data": _source_facts(item_forecast, [
                "outputs/final/forecast.csv", "outputs/final/future_features.csv.gz",
                "outputs/final/model_metadata.json", "outputs/final/metrics.json",
            ]),
        })
    result = pd.DataFrame(records)
    result["priority_rank"] = result["action"].map({"order_today": 0, "monitor": 1})
    return result.sort_values(["priority_rank", "forecast_7d_p50"], ascending=[True, False]).drop(columns="priority_rank")


def get_replenishment(
    store_nbr: int,
    family: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Scenario-based order suggestion and forecast/inventory evidence for one store and family."""
    store_nbr = _validate_store(store_nbr)
    family = _normalize_family(family)
    matches = _replenishment_rows(store_nbr, family, start_date, end_date)
    if matches.empty:
        raise ValueError(f"No replenishment data for store {store_nbr} / {family}")
    return matches.iloc[0].to_dict()


def get_priority_replenishments(store_nbr: int | None = None, top_n: int = 5) -> dict:
    """Highest-priority simulated replenishment actions; order_today first, then higher p50 demand."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 20:
        raise ValueError("top_n must be an integer from 1 to 20")
    if store_nbr is not None:
        store_nbr = _validate_store(store_nbr)
    rows = _replenishment_rows(store_nbr=store_nbr).head(top_n)
    source = _forecast()
    if store_nbr is not None:
        source = source[source["store_nbr"] == store_nbr]
    return {
        "scope": "all available stores" if store_nbr is None else f"store {store_nbr}",
        "items": rows.to_dict("records"),
        "priority_rule": "order_today before monitor; within each action, higher v3 p50 forecast first.",
        "data": _source_facts(source),
    }


def get_forecast_overview(store_nbr: int | None = None, top_n: int = 5) -> dict:
    """Totals by family and the series with the biggest expected change vs last week."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 20:
        raise ValueError("top_n must be an integer from 1 to 20")
    forecast = _forecast()
    features = _future_features()
    if store_nbr is not None:
        store_nbr = _validate_store(store_nbr)
        forecast = forecast[forecast["store_nbr"] == int(store_nbr)]
    totals = forecast.groupby("family")[list(QUANTILES)].sum().round(1).sort_values("p50", ascending=False)
    # Last week's actual units per series = mean_7 x 7 (known at the origin, same for all horizon days).
    last_week = features[features["horizon_day"] == 1][["store_nbr", "family", "mean_7"]].copy()
    last_week["store_nbr"] = last_week["store_nbr"].astype(int)
    last_week["family"] = last_week["family"].astype(str)
    weekly = forecast.groupby(["store_nbr", "family"], as_index=False)["p50"].sum().merge(last_week, on=["store_nbr", "family"])
    weekly["last_week_units"] = weekly["mean_7"] * 7
    weekly = weekly[weekly["last_week_units"] >= 50]  # ignore tiny series for % changes
    weekly["change_pct"] = (weekly["p50"] / weekly["last_week_units"] - 1) * 100
    def pick(frame):
        return [{"store_nbr": int(r.store_nbr), "family": r.family, "forecast_7d": _r(r.p50),
                 "last_week": _r(r.last_week_units), "change_pct": _r(r.change_pct)} for r in frame.itertuples()]
    return {
        "scope": "all stores" if store_nbr is None else f"store {store_nbr}",
        "forecast_origin": str(forecast["forecast_origin"].iloc[0]),
        "total_7d_units": {q: _r(forecast[q].sum()) for q in QUANTILES},
        "by_family": totals.reset_index().to_dict("records"),
        "biggest_increases": pick(weekly.nlargest(top_n, "change_pct")),
        "biggest_decreases": pick(weekly.nsmallest(top_n, "change_pct")),
        "data": _source_facts(forecast),
    }


def get_model_reliability(family: str | None = None) -> dict:
    """Backtest accuracy (WAPE, bias) of the model vs baselines, overall and by situation."""
    errors = _errors()
    if family is not None:
        family = _normalize_family(family)
    primary = f"v3_{_metadata()['models']['p50']['spec'].removeprefix('v3_')}"
    compare = [m for m in (primary, "seasonal_naive", "v2_recursive") if m in set(errors["model"])]

    def table(group_type: str, groups=None) -> list[dict]:
        part = errors[(errors["group_type"] == group_type) & errors["model"].isin(compare)]
        if groups is not None:
            part = part[part["group"].isin(groups)]
        wide = part.pivot_table(index="group", columns="model", values=["wape", "bias"])
        records = []
        for group, row in wide.iterrows():
            item = {"group": group}
            for model in compare:
                item[f"{model}_wape_pct"] = _r(row[("wape", model)] * 100)
            item[f"{primary}_bias_pct"] = _r(row[("bias", primary)] * 100)
            records.append(item)
        return records

    result = {
        "model": primary,
        "how_to_read": "WAPE = total absolute error / total sales (lower is better). "
                       "Bias > 0 means over-forecast (waste risk), < 0 under-forecast (stock-out risk).",
        "by_situation": table("promotion") + table("holiday") + table("volume_tier"),
        "by_horizon_day": table("horizon_day"),
        "by_window": table("window"),
    }
    result["by_family"] = table("family", [family.upper()] if family else None)
    if family:
        row = result["by_family"][0] if result["by_family"] else None
        if row:
            wape = row[f"{primary}_wape_pct"]
            result["verdict"] = ("reliable" if wape < 12 else "usable - review big orders" if wape < 20
                                 else "low reliability - planner review recommended")
    return result


def explain_forecast(store_nbr: int, family: str, top_n: int = 5) -> dict:
    """Why the 7-day forecast is what it is: top feature contributions (LightGBM SHAP values)."""
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 20:
        raise ValueError("top_n must be an integer from 1 to 20")
    rows = _series(store_nbr, family)
    if rows["route"].iloc[0] == "cold_start":
        return {"store_nbr": int(store_nbr), "family": _normalize_family(family), "route": "cold_start",
                "explanation": "New series (under 56 trading days): forecast = same weekday last week, "
                       "because there is too little history for the model.", "data": _source_facts(rows)}
    features = _future_features()
    mask = (features["store_nbr"].astype(str) == str(int(store_nbr))) & (features["family"].astype(str) == family.upper())
    part = features[mask].sort_values("date")
    contributions = _boosters()["p50"].predict(part[FEATURES], pred_contrib=True)
    objective = _metadata()["models"]["p50"]["objective"]["objective"]
    first = part.iloc[0]
    if objective in ("tweedie", "poisson", "gamma"):
        # Log-link model: contributions add up in log space, i.e. they multiply the forecast.
        effects = pd.Series((np.exp(contributions[:, :-1]) - 1).mean(axis=0) * 100, index=FEATURES)
        ordered = effects.reindex(effects.abs().sort_values(ascending=False).index).head(top_n)
        drivers = [{"feature": f, "meaning": DRIVER_NAMES.get(f, f), "effect_pct": _r(v)} for f, v in ordered.items()]
        baseline = float(np.exp(contributions[:, -1]).sum())
        note = ("effect_pct: average % by which each factor multiplies the daily forecast relative to the "
                "model's typical series (effects combine multiplicatively).")
    else:
        effects = pd.Series(contributions[:, :-1].sum(axis=0), index=FEATURES)
        ordered = effects.reindex(effects.abs().sort_values(ascending=False).index).head(top_n)
        drivers = [{"feature": f, "meaning": DRIVER_NAMES.get(f, f), "effect_units_7d": _r(v)} for f, v in ordered.items()]
        baseline = float(contributions[:, -1].sum())
        note = "effect_units_7d: how much each factor pushes the week's forecast above (+) or below (-) the model average."
    return {
        "store_nbr": int(store_nbr), "family": _normalize_family(family),
        "forecast_7d_p50": _r(rows["p50"].sum()),
        "typical_series_7d": _r(baseline),
        "top_drivers": drivers,
        "context": {
            "last_week_units": _r(first["mean_7"] * 7),
            "last_4_weeks_avg_week_units": _r(first["mean_28"] * 7),
            "planned_promo_items_next_7d": int(part["onpromotion"].sum()),
            "usual_promo_items_per_day": _r(first["promo_mean_28_hist"]),
            "holiday_days_next_7d": int(part["is_holiday"].sum()),
        },
        "note": note,
        "data": _source_facts(rows, ["outputs/final/forecast.csv", "outputs/final/future_features.csv.gz",
                         "outputs/final/model_metadata.json", "outputs/final/metrics.json"]),
    }


def what_if_promotion(store_nbr: int, family: str, onpromotion: int, dates: list[str] | None = None) -> dict:
    """Re-forecast one store x family if the promotion plan changes.

    onpromotion: number of items of the family on promotion on each selected day
    (0 = cancel promotions). dates: 'YYYY-MM-DD' days to change; default = all 7 days.
    """
    store_nbr = _validate_store(store_nbr)
    family = _normalize_family(family)
    if isinstance(onpromotion, bool) or not isinstance(onpromotion, int) or onpromotion < 0:
        raise ValueError("onpromotion must be a non-negative integer")
    if dates is not None and (not isinstance(dates, list) or not dates):
        raise ValueError("dates must be a non-empty list of YYYY-MM-DD dates")
    if dates is not None:
        for date in dates:
            try:
                parsed = pd.Timestamp(date)
            except (TypeError, ValueError) as error:
                raise ValueError("Dates must use YYYY-MM-DD format") from error
            if not isinstance(date, str) or parsed.strftime("%Y-%m-%d") != date:
                raise ValueError("Dates must use YYYY-MM-DD format")
    features = _future_features()
    mask = (features["store_nbr"].astype(str) == str(int(store_nbr))) & (features["family"].astype(str) == family.upper())
    base = features[mask].sort_values("date").copy()
    if base.empty:
        raise ValueError(f"No forecast rows for store {store_nbr} / {family}")
    if (_series(store_nbr, family)["route"] == "cold_start").all():
        raise ValueError("What-if needs a model-based (lightgbm route) series; this one uses a rule-based fallback.")
    selected = pd.to_datetime(dates) if dates else base["date"]
    unknown = set(selected) - set(base["date"])
    if unknown:
        raise ValueError(f"Dates outside the forecast week: {sorted(d.strftime('%Y-%m-%d') for d in unknown)}")

    scenario = base.copy()
    delta = pd.Series(0.0, index=scenario["date"].to_numpy())
    change = scenario["date"].isin(selected).to_numpy()
    delta[change] = float(onpromotion) - scenario.loc[change, "onpromotion"].to_numpy()
    scenario.loc[change, "onpromotion"] = float(onpromotion)
    # Rolling promotion sums include the changed future days inside their window.
    for column, width in (("promo_sum_7", 7), ("promo_sum_14", 14)):
        shift = [delta[(delta.index > d - pd.Timedelta(days=width)) & (delta.index <= d)].sum() for d in scenario["date"]]
        scenario[column] = scenario[column].to_numpy() + np.asarray(shift, dtype="float32")
    scenario["promo_vs_hist"] = scenario["onpromotion"] - scenario["promo_mean_28_hist"]

    before, after = _predict(base), _predict(scenario)
    daily = [
        {"date": d.strftime("%Y-%m-%d"), "onpromotion_before": int(b0), "onpromotion_after": int(a0),
         "p50_before": _r(b), "p50_after": _r(a), "p90_after": _r(c)}
        for d, b0, a0, b, a, c in zip(base["date"], base["onpromotion"], scenario["onpromotion"], before["p50"], after["p50"], after["p90"])
    ]
    total_before, total_after = before["p50"].sum(), after["p50"].sum()
    return {
        "store_nbr": int(store_nbr), "family": family.upper(),
        "scenario": f"{int(onpromotion)} promoted items on {len(selected)} day(s)",
        "total_7d_p50_before": _r(total_before), "total_7d_p50_after": _r(total_after),
        "change_units": _r(total_after - total_before),
        "change_pct": _r((total_after / total_before - 1) * 100) if total_before else None,
        "total_7d_p90_after": _r(after["p90"].sum()),
        "daily": daily,
        "caveat": "Model-estimated response learned from historical promotions; promotion levels far outside "
                  "the series' history are extrapolations. Promotion backtest accuracy: see get_model_reliability.",
        "data": _source_facts(_series(store_nbr, family), [
            "outputs/final/forecast.csv", "outputs/final/future_features.csv.gz",
            "outputs/final/model_metadata.json", "outputs/final/metrics.json",
        ]),
    }


TOOLS = {
    "get_forecast": get_forecast,
    "get_replenishment": get_replenishment,
    "get_priority_replenishments": get_priority_replenishments,
    "get_forecast_overview": get_forecast_overview,
    "get_model_reliability": get_model_reliability,
    "explain_forecast": explain_forecast,
    "what_if_promotion": what_if_promotion,
}
_STORE = {"type": "integer", "description": "Favorita store number, 1-54"}
_FAMILY = {"type": "string", "description": "Food family, e.g. DAIRY, PRODUCE, BEVERAGES, BREAD/BAKERY"}
TOOL_SPECS = [
    {"name": "get_forecast", "description": get_forecast.__doc__.strip(),
     "parameters": {"type": "object", "properties": {"store_nbr": _STORE, "family": _FAMILY, "start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["store_nbr", "family"]}},
    {"name": "get_replenishment", "description": get_replenishment.__doc__.strip(),
    "parameters": {"type": "object", "properties": {"store_nbr": _STORE, "family": _FAMILY, "start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["store_nbr", "family"]}},
    {"name": "get_priority_replenishments", "description": get_priority_replenishments.__doc__.strip(),
     "parameters": {"type": "object", "properties": {"store_nbr": _STORE, "top_n": {"type": "integer", "minimum": 1, "maximum": 20}}}},
    {"name": "get_forecast_overview", "description": get_forecast_overview.__doc__.strip(),
     "parameters": {"type": "object", "properties": {"store_nbr": _STORE, "top_n": {"type": "integer"}}}},
    {"name": "get_model_reliability", "description": get_model_reliability.__doc__.strip(),
     "parameters": {"type": "object", "properties": {"family": _FAMILY}}},
    {"name": "explain_forecast", "description": explain_forecast.__doc__.strip(),
     "parameters": {"type": "object", "properties": {"store_nbr": _STORE, "family": _FAMILY, "top_n": {"type": "integer"}}, "required": ["store_nbr", "family"]}},
    {"name": "what_if_promotion", "description": what_if_promotion.__doc__.strip(),
     "parameters": {"type": "object", "properties": {
         "store_nbr": _STORE, "family": _FAMILY,
         "onpromotion": {"type": "integer", "description": "items on promotion per selected day; 0 cancels"},
         "dates": {"type": "array", "items": {"type": "string"}, "description": "YYYY-MM-DD days to change; omit for all 7"}},
         "required": ["store_nbr", "family", "onpromotion"]}},
]


def call_tool(name: str, args: dict) -> dict:
    """Dispatch helper for agent loops; errors come back as data so the agent can recover."""
    try:
        return TOOLS[name](**args)
    except Exception as error:  # noqa: BLE001 - surfaced to the agent, not swallowed
        return {"error": f"{type(error).__name__}: {error}"}


def main() -> None:
    demos = [
        ("get_forecast_overview", {"top_n": 3}),
        ("get_forecast", {"store_nbr": 44, "family": "DAIRY"}),
        ("explain_forecast", {"store_nbr": 44, "family": "DAIRY"}),
        ("what_if_promotion", {"store_nbr": 44, "family": "DAIRY", "onpromotion": 0}),
        ("what_if_promotion", {"store_nbr": 44, "family": "DAIRY", "onpromotion": 60, "dates": ["2017-08-18", "2017-08-19"]}),
        ("get_model_reliability", {"family": "DAIRY"}),
    ]
    for name, args in demos:
        print(f"\n=== {name}({args})")
        print(json.dumps(call_tool(name, args), indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
