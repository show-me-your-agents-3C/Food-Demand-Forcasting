"""Load and prepare a bounded, reproducible Favorita modeling slice."""

import pandas as pd

from .config import DATA_PATH, FOOD_FAMILIES, LOOKBACK_DAYS, SELECTED_FAMILIES, SELECTED_STORES


def load_sales() -> pd.DataFrame:
    usecols = ["date", "store_nbr", "family", "sales", "onpromotion"]
    frame = pd.read_csv(DATA_PATH, usecols=usecols, parse_dates=["date"])
    frame = frame[
        frame["store_nbr"].isin(SELECTED_STORES)
        & frame["family"].isin(FOOD_FAMILIES)
        & frame["family"].isin(SELECTED_FAMILIES)
    ].copy()
    end_date = frame["date"].max()
    start_date = end_date - pd.Timedelta(days=LOOKBACK_DAYS + 14)
    frame = frame[frame["date"] >= start_date].sort_values(["store_nbr", "family", "date"])
    if frame.empty:
        raise ValueError("No rows remain after food/store filters. Check config.py.")
    return frame


def split_train_validation(frame: pd.DataFrame, validation_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = frame["date"].max() - pd.Timedelta(days=validation_days - 1)
    return frame[frame["date"] < cutoff].copy(), frame[frame["date"] >= cutoff].copy()
