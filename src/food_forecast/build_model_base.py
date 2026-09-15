"""Build a food-only model base without dropping valid sales records."""

import json
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT


PROCESSED = PROJECT_ROOT / "data" / "processed"
INPUT = PROCESSED / "cleaned_food_sales.csv"
OUTPUT = PROCESSED / "model_base_food.csv"
REPORT = PROCESSED / "model_base_report.json"
CHUNK_SIZE = 100_000


def main() -> None:
    stores = pd.read_csv(PROCESSED / "cleaned_stores.csv")
    holidays = pd.read_csv(PROCESSED / "holiday_features.csv", parse_dates=["date"])
    temp = OUTPUT.with_suffix(".tmp")
    temp.unlink(missing_ok=True)
    total = unmatched_stores = rows_with_holiday_event = 0
    wrote_header = False

    for chunk in pd.read_csv(INPUT, chunksize=CHUNK_SIZE, parse_dates=["date"]):
        input_rows = len(chunk)
        chunk = chunk.merge(stores, on="store_nbr", how="left", validate="many_to_one", indicator="_store_merge")
        unmatched_stores += int((chunk["_store_merge"] != "both").sum())
        chunk = chunk.drop(columns="_store_merge")
        chunk = chunk.merge(holidays, on="date", how="left", validate="many_to_one")
        chunk["holiday_event_count"] = chunk["holiday_event_count"].fillna(0).astype("int16")
        for column in ["has_transferred_holiday", "is_national_holiday", "is_national_event"]:
            chunk[column] = chunk[column].fillna(False).astype(bool)
        rows_with_holiday_event += int((chunk["holiday_event_count"] > 0).sum())
        chunk["day_of_week"] = chunk["date"].dt.dayofweek.astype("int8")
        chunk["month"] = chunk["date"].dt.month.astype("int8")
        chunk["is_weekend"] = (chunk["day_of_week"] >= 5)
        chunk["date"] = chunk["date"].dt.strftime("%Y-%m-%d")
        if len(chunk) != input_rows:
            raise ValueError("A join changed the sales row count; refusing to write a lossy model base.")
        total += len(chunk)
        chunk.to_csv(temp, mode="a", index=False, header=not wrote_header)
        wrote_header = True

    if unmatched_stores:
        raise ValueError(f"{unmatched_stores} sales rows have no store metadata; no output promoted.")
    temp.replace(OUTPUT)
    report = {
        "input_food_rows": total,
        "output_rows": total,
        "deleted_rows": 0,
        "quality_deletion_rate": 0.0,
        "unmatched_store_rows": unmatched_stores,
        "rows_with_holiday_event": rows_with_holiday_event,
        "rule": "A missing holiday entry means no listed event and is filled with false/0; no sales row is removed.",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
