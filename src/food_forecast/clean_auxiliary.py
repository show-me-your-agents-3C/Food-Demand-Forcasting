"""Non-destructive cleaning for Favorita auxiliary tables."""

import json
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT


RAW = PROJECT_ROOT / "data" / "favorita"
OUT = PROJECT_ROOT / "data" / "processed"


def _write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    stores = pd.read_csv(RAW / "stores.csv")
    stores["store_nbr"] = pd.to_numeric(stores["store_nbr"], errors="raise").astype("int16")
    for column in ["city", "state", "type"]:
        stores[column] = stores[column].astype("string").str.strip()
    if stores["store_nbr"].duplicated().any():
        raise ValueError("stores.csv has duplicate store_nbr values; refusing to silently deduplicate.")
    _write(stores, "cleaned_stores.csv")
    report["stores"] = {"input_rows": len(stores), "output_rows": len(stores), "deleted_rows": 0}

    holidays = pd.read_csv(RAW / "holidays_events.csv")
    holidays["date"] = pd.to_datetime(holidays["date"], errors="raise").dt.strftime("%Y-%m-%d")
    for column in ["type", "locale", "locale_name", "description"]:
        holidays[column] = holidays[column].astype("string").str.strip()
    holidays["transferred"] = holidays["transferred"].fillna(False).astype(bool)
    _write(holidays, "cleaned_holidays_events.csv")
    national = holidays[(holidays["locale"] == "National") & ~holidays["transferred"]].copy()
    calendar = holidays.groupby("date", as_index=False).agg(
        holiday_event_count=("type", "size"),
        has_transferred_holiday=("transferred", "max"),
    )
    national_flags = national.groupby("date", as_index=False).agg(
        is_national_holiday=("type", lambda x: bool((x == "Holiday").any())),
        is_national_event=("type", lambda x: bool((x == "Event").any())),
    )
    calendar = calendar.merge(national_flags, on="date", how="left").fillna({"is_national_holiday": False, "is_national_event": False})
    _write(calendar, "holiday_features.csv")
    report["holidays"] = {"input_rows": len(holidays), "cleaned_rows": len(holidays), "feature_dates": len(calendar), "deleted_rows": 0}

    oil = pd.read_csv(RAW / "oil.csv")
    oil["date"] = pd.to_datetime(oil["date"], errors="raise")
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
    oil = oil.sort_values("date")
    observed_missing = int(oil["dcoilwtico"].isna().sum())
    daily_index = pd.date_range(oil["date"].min(), oil["date"].max(), freq="D")
    oil_daily = oil.set_index("date").reindex(daily_index).rename_axis("date").reset_index()
    oil_daily["is_observed_oil_date"] = oil_daily["dcoilwtico"].notna()
    oil_daily["dcoilwtico"] = oil_daily["dcoilwtico"].ffill().bfill()
    oil_daily["date"] = oil_daily["date"].dt.strftime("%Y-%m-%d")
    _write(oil_daily, "oil_daily_cleaned.csv")
    report["oil"] = {"input_rows": len(oil), "input_missing_values": observed_missing, "output_daily_rows": len(oil_daily), "deleted_rows": 0, "note": "Missing calendar dates and observed nulls are forward/back-filled in a derived historical series."}

    transactions = pd.read_csv(RAW / "transactions.csv")
    transactions["date"] = pd.to_datetime(transactions["date"], errors="raise").dt.strftime("%Y-%m-%d")
    transactions["store_nbr"] = pd.to_numeric(transactions["store_nbr"], errors="raise").astype("int16")
    transactions["transactions"] = pd.to_numeric(transactions["transactions"], errors="raise").astype("int32")
    if transactions.duplicated(["date", "store_nbr"]).any():
        raise ValueError("transactions.csv has duplicate date-store rows; refusing to silently deduplicate.")
    _write(transactions, "cleaned_transactions.csv")
    report["transactions"] = {"input_rows": len(transactions), "output_rows": len(transactions), "deleted_rows": 0, "note": "Preserved for analysis only; do not use same-day transactions as a future forecasting feature."}

    (OUT / "auxiliary_cleaning_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
