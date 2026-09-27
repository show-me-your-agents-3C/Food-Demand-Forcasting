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


def _series(store_nbr: int, family: str) -> pd.DataFrame:
    forecast = _forecast()
    rows = forecast[(forecast["store_nbr"] == int(store_nbr)) & (forecast["family"] == family.upper())]
    if rows.empty:
        raise ValueError(f"No forecast for store {store_nbr} / {family}. Families: {sorted(forecast['family'].unique())}")
    return rows.sort_values("date")


def _predict(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    boosters = _boosters()
    values = {name: np.clip(boosters[name].predict(rows[FEATURES]), 0, None) for name in QUANTILES}
    values["p10"] = np.minimum(values["p10"], values["p50"])
    values["p90"] = np.maximum(values["p90"], values["p50"])
    return values


def _r(value: float) -> float:
    return round(float(value), 1)


# ---------------------------------------------------------------- tools
def get_forecast(store_nbr: int, family: str) -> dict:
    """Daily 7-day forecast (median and 80% range) for one store x family."""
    rows = _series(store_nbr, family)
    return {
        "store_nbr": int(store_nbr), "family": family.upper(),
        "forecast_origin": str(rows["forecast_origin"].iloc[0]),
        "method": rows["forecast_method"].iloc[0], "model_version": rows["model_version"].iloc[0],
        "route": rows["route"].iloc[0],
        **({"warning": "Sparse, low-volume series: expect large percentage errors."} if rows["route"].iloc[0] == "intermittent" else {}),
        "total_7d": {"p10": _r(rows["p10"].sum()), "p50": _r(rows["p50"].sum()), "p90": _r(rows["p90"].sum())},
        "daily": [
            {"date": d.strftime("%Y-%m-%d"), "weekday": d.strftime("%a"), "onpromotion": int(p),
             "p10": _r(a), "p50": _r(b), "p90": _r(c)}
            for d, p, a, b, c in zip(rows["date"], rows["onpromotion"], rows["p10"], rows["p50"], rows["p90"])
        ],
        "note": "p50 = expected demand; there is a ~80% chance demand falls between p10 and p90 (per backtest calibration).",
    }


def get_forecast_overview(store_nbr: int | None = None, top_n: int = 5) -> dict:
    """Totals by family and the series with the biggest expected change vs last week."""
    forecast = _forecast()
    features = _future_features()
    if store_nbr is not None:
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
    }


def get_model_reliability(family: str | None = None) -> dict:
    """Backtest accuracy (WAPE, bias) of the model vs baselines, overall and by situation."""
    errors = _errors()
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
    rows = _series(store_nbr, family)
    if rows["route"].iloc[0] == "cold_start":
        return {"store_nbr": int(store_nbr), "family": family.upper(), "route": "cold_start",
                "explanation": "New series (under 56 trading days): forecast = same weekday last week, "
                               "because there is too little history for the model."}
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
        "store_nbr": int(store_nbr), "family": family.upper(),
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
    }


def what_if_promotion(store_nbr: int, family: str, onpromotion: int, dates: list[str] | None = None) -> dict:
    """Re-forecast one store x family if the promotion plan changes.

    onpromotion: number of items of the family on promotion on each selected day
    (0 = cancel promotions). dates: 'YYYY-MM-DD' days to change; default = all 7 days.
    """
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
    }


TOOLS = {
    "get_forecast": get_forecast,
    "get_forecast_overview": get_forecast_overview,
    "get_model_reliability": get_model_reliability,
    "explain_forecast": explain_forecast,
    "what_if_promotion": what_if_promotion,
}
_STORE = {"type": "integer", "description": "Favorita store number, 1-54"}
_FAMILY = {"type": "string", "description": "Food family, e.g. DAIRY, PRODUCE, BEVERAGES, BREAD/BAKERY"}
TOOL_SPECS = [
    {"name": "get_forecast", "description": get_forecast.__doc__.strip(),
     "parameters": {"type": "object", "properties": {"store_nbr": _STORE, "family": _FAMILY}, "required": ["store_nbr", "family"]}},
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
