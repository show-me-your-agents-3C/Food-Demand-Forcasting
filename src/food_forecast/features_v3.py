"""Feature panel for the v3 direct (non-recursive) seven-day forecaster.

Orders are placed once a week on Wednesday morning (the Kaggle test period
also starts on Wednesday 2017-08-16). A target day t therefore has a fixed
horizon h = 1..7 set by its weekday, and every sales-derived feature reads only
days <= t - h, i.e. sales already closed when the order is placed. One model
serves all horizons with no recursive feedback of its own predictions, and
look-ahead leakage is ruled out by construction.

Sales are held as (series x day) matrices over a complete daily calendar, so
lags are true calendar lags even across the store-closed days (e.g. 25 Dec)
that are missing from the Kaggle file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FOOD_FAMILIES, PROJECT_ROOT


RAW = PROJECT_ROOT / "data" / "favorita"
REGISTRY = PROJECT_ROOT / "data" / "processed" / "series_registry.csv"
ORIGIN_WEEKDAY = 2  # Wednesday
HORIZON = 7
MIN_HISTORY_DAYS = 63
CELEBRATED = {"Holiday", "Transfer", "Additional", "Bridge"}

CATEGORICAL = ["store_nbr", "family", "store_type", "store_cluster", "city"]
PROMO_FEATURES = ["onpromotion", "promo_sum_7", "promo_sum_14", "promo_lag_7", "promo_mean_28_hist", "promo_vs_hist"]
FEATURES = CATEGORICAL + [
    "horizon_day", "day_of_week", "day_of_month", "month", "days_since_payday", "is_payday",
    "is_holiday", "holiday_scope", "days_to_holiday", "days_since_holiday", "is_national_event", "is_workday",
    *PROMO_FEATURES,
    "last_sales", "mean_7", "mean_14", "mean_28", "mean_56", "std_28", "zero_frac_28", "trend_7_28",
    "lag_7", "lag_14", "lag_21", "lag_28", "same_dow_mean_4",
]


@dataclass
class Panel:
    series: pd.DataFrame      # one row per store-family, aligned with matrix rows
    dates: pd.DatetimeIndex   # complete daily calendar
    sales: np.ndarray         # (n_series, n_days); NaN = unobserved or future
    promo: np.ndarray         # (n_series, n_days); items of the family on promotion
    store_open: np.ndarray    # (n_series, n_days); False when the store sold no food at all
    first_sale: np.ndarray    # (n_series,); index of the first non-zero sale
    store_calendar: dict      # name -> (n_series, n_days) store-matched holiday arrays


def _read(path, with_sales: bool) -> pd.DataFrame:
    cols = ["date", "store_nbr", "family", "onpromotion"] + (["sales"] if with_sales else [])
    frame = pd.read_csv(path, usecols=cols, parse_dates=["date"], dtype={"store_nbr": "int16", "onpromotion": "float32"})
    frame = frame[frame["family"].isin(FOOD_FAMILIES)]
    if not with_sales:
        frame["sales"] = np.nan
    return frame


def _store_calendar(stores: pd.DataFrame, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Holidays matched to each store's city/state; transferred days are not celebrated."""
    holidays = pd.read_csv(RAW / "holidays_events.csv", parse_dates=["date"])
    holidays = holidays[holidays["date"].between(dates[0], dates[-1])]
    n_stores, n_days = len(stores), len(dates)
    shape = (n_stores, n_days)
    is_holiday = np.zeros(shape, bool)
    scope = np.zeros(shape, np.int8)
    event = np.zeros(shape, bool)
    workday = np.zeros(shape, bool)
    day_index = (holidays["date"] - dates[0]).dt.days.to_numpy()
    for row, j in zip(holidays.itertuples(index=False), day_index):
        if row.locale == "National":
            match, level = np.ones(n_stores, bool), 3
        elif row.locale == "Regional":
            match, level = (stores["state"] == row.locale_name).to_numpy(), 2
        else:
            match, level = (stores["city"] == row.locale_name).to_numpy(), 1
        if row.type in CELEBRATED and not row.transferred:
            is_holiday[match, j] = True
            scope[match, j] = np.maximum(scope[match, j], level)
        elif row.type == "Event":
            event[match, j] = True
        elif row.type == "Work Day":
            workday[match, j] = True

    to_next = np.full(shape, 15, np.int8)
    since = np.full(shape, 15, np.int8)
    for j in range(n_days - 1, -1, -1):
        following = to_next[:, j + 1] + 1 if j + 1 < n_days else 15
        to_next[:, j] = np.where(is_holiday[:, j], 0, np.minimum(following, 15))
    for j in range(n_days):
        previous = since[:, j - 1] + 1 if j > 0 else 15
        since[:, j] = np.where(is_holiday[:, j], 0, np.minimum(previous, 15))
    return {"is_holiday": is_holiday, "holiday_scope": scope, "is_national_event": event,
            "is_workday": workday, "days_to_holiday": to_next, "days_since_holiday": since}


def load_panel(include_future: bool = True) -> Panel:
    """Food sales as matrices; `include_future` appends Kaggle test.csv promotion plans."""
    frames = [_read(RAW / "train.csv", with_sales=True)]
    if include_future:
        frames.append(_read(RAW / "test.csv", with_sales=False))
    data = pd.concat(frames, ignore_index=True)
    data["family"] = data["family"].astype(str)

    series = pd.read_csv(REGISTRY)[["store_nbr", "family", "series_status"]]
    series = series.sort_values(["store_nbr", "family"]).reset_index(drop=True)
    stores = pd.read_csv(RAW / "stores.csv").sort_values("store_nbr").reset_index(drop=True)
    series = series.merge(stores.rename(columns={"type": "store_type", "cluster": "store_cluster"}), on="store_nbr", how="left", validate="many_to_one")

    dates = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
    key_index = pd.MultiIndex.from_frame(series[["store_nbr", "family"]])
    rows = key_index.get_indexer(pd.MultiIndex.from_frame(data[["store_nbr", "family"]]))
    if (rows < 0).any():
        raise ValueError("Sales rows without a registry entry; rebuild series_registry.csv first.")
    cols = (data["date"] - dates[0]).dt.days.to_numpy()

    sales = np.full((len(series), len(dates)), np.nan)
    promo = np.zeros((len(series), len(dates)))
    sales[rows, cols] = data["sales"].to_numpy(float)
    promo[rows, cols] = data["onpromotion"].fillna(0).to_numpy(float)

    store_pos = pd.Index(stores["store_nbr"]).get_indexer(series["store_nbr"])
    filled = np.nan_to_num(sales)
    store_total = np.zeros((len(stores), len(dates)))
    np.add.at(store_total, store_pos, filled)
    observed = ~np.isnan(sales)
    store_open = (store_total[store_pos] > 0) | ~observed.any(axis=0)  # future days: assume open
    nonzero = filled > 0
    first_sale = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), len(dates))

    calendar = {name: values[store_pos] for name, values in _store_calendar(stores, dates).items()}
    return Panel(series, dates, sales, promo, store_open, first_sale, calendar)


def _window_sum(cumsum: np.ndarray, end: np.ndarray, width: int) -> np.ndarray:
    """Sum over the `width` days ending at (and including) column `end`."""
    return cumsum[:, end + 1] - cumsum[:, end + 1 - width]


def promo_features(promo: np.ndarray, days: np.ndarray, anchor: np.ndarray) -> dict[str, np.ndarray]:
    cum = np.concatenate([np.zeros((promo.shape[0], 1)), np.cumsum(promo, axis=1)], axis=1)
    hist = _window_sum(cum, anchor, 28) / 28
    return {
        "onpromotion": promo[:, days],
        "promo_sum_7": _window_sum(cum, days, 7),
        "promo_sum_14": _window_sum(cum, days, 14),
        "promo_lag_7": promo[:, days - 7],
        "promo_mean_28_hist": hist,
        "promo_vs_hist": promo[:, days] - hist,
    }


def build_features(panel: Panel) -> pd.DataFrame:
    """One row per (series, day) with at least MIN_HISTORY_DAYS of calendar history."""
    n_series, n_days = panel.sales.shape
    days = np.arange(MIN_HISTORY_DAYS, n_days)
    dates = panel.dates[days]
    horizon = ((dates.dayofweek.to_numpy() - ORIGIN_WEEKDAY) % 7) + 1
    anchor = days - horizon  # last day whose sales are known at order time

    sales = np.nan_to_num(panel.sales)
    cum = np.concatenate([np.zeros((n_series, 1)), np.cumsum(sales, axis=1)], axis=1)
    cum_sq = np.concatenate([np.zeros((n_series, 1)), np.cumsum(sales ** 2, axis=1)], axis=1)
    cum_zero = np.concatenate([np.zeros((n_series, 1)), np.cumsum(sales == 0, axis=1)], axis=1)
    mean = {w: _window_sum(cum, anchor, w) / w for w in (7, 14, 28, 56)}
    var_28 = _window_sum(cum_sq, anchor, 28) / 28 - mean[28] ** 2
    lags = {k: sales[:, days - k] for k in (7, 14, 21, 28)}

    dom = dates.day.to_numpy()
    days_since_payday = np.where(dates.is_month_end, 0, np.where(dom >= 15, dom - 15, dom))

    def per_day(values) -> np.ndarray:
        return np.tile(np.asarray(values), n_series)

    def per_series(column: str) -> np.ndarray:
        return np.repeat(panel.series[column].to_numpy(), len(days))

    matrices = {
        "last_sales": sales[:, anchor],
        **{f"mean_{w}": m for w, m in mean.items()},
        "std_28": np.sqrt(np.clip(var_28, 0, None)),
        "zero_frac_28": _window_sum(cum_zero, anchor, 28) / 28,
        "trend_7_28": mean[7] / (mean[28] + 1.0),
        **{f"lag_{k}": v for k, v in lags.items()},
        "same_dow_mean_4": sum(lags.values()) / 4,
        **promo_features(panel.promo, days, anchor),
        **{name: values[:, days] for name, values in panel.store_calendar.items()},
    }
    frame = pd.DataFrame({
        "date": per_day(dates.values),
        "store_nbr": per_series("store_nbr"),
        "family": per_series("family"),
        "store_type": per_series("store_type"),
        "store_cluster": per_series("store_cluster"),
        "city": per_series("city"),
        "series_status": per_series("series_status"),
        "horizon_day": per_day(horizon).astype("int8"),
        "day_of_week": per_day(dates.dayofweek.to_numpy()).astype("int8"),
        "day_of_month": per_day(dom).astype("int8"),
        "month": per_day(dates.month.to_numpy()).astype("int8"),
        "days_since_payday": per_day(days_since_payday).astype("int8"),
        "is_payday": per_day(days_since_payday == 0).astype("int8"),
        **{name: values.ravel().astype("float32") for name, values in matrices.items()},
        "sales": panel.sales[:, days].ravel(),
    })
    for flag in ("is_holiday", "is_national_event", "is_workday"):
        frame[flag] = frame[flag].astype("int8")
    for column in CATEGORICAL:
        frame[column] = frame[column].astype(str).astype("category")
    frame["series_status"] = frame["series_status"].astype("category")
    # Share of the last 28 days the store was open: separates store closures from slow demand.
    cum_open = np.concatenate([np.zeros((n_series, 1)), np.cumsum(panel.store_open, axis=1)], axis=1)
    frame["store_open_frac_28"] = (_window_sum(cum_open, anchor, 28) / 28).ravel().astype("float32")
    # Days the series has traded as of the order date; used to route cold starts.
    trading_days = anchor[None, :] - panel.first_sale[:, None] + 1
    frame["trading_days"] = np.clip(trading_days, 0, None).ravel().astype("int32")
    # Train only on open days once every rolling window (56 days) covers trading history.
    history_ok = days[None, :] - 56 >= panel.first_sale[:, None]
    frame["train_ok"] = (~np.isnan(panel.sales[:, days]) & panel.store_open[:, days] & history_ok).ravel()
    return frame
