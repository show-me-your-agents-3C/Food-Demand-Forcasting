"""Low-memory, non-destructive cleaning for Favorita sales data.

The raw Kaggle files are never altered. This module first creates a full
cleaned copy with an `is_food` flag; a food-only view is created separately so
scope exclusions cannot be mistaken for quality deletions.
"""

import csv
import json
from pathlib import Path

import pandas as pd

from .config import FOOD_FAMILIES, PROJECT_ROOT


RAW_TRAIN = PROJECT_ROOT / "data" / "favorita" / "train.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
ALL_OUTPUT = PROCESSED / "cleaned_sales_all.csv"
FOOD_OUTPUT = PROCESSED / "cleaned_food_sales.csv"
REPORT_OUTPUT = PROCESSED / "cleaning_report.json"
CHUNK_SIZE = 100_000


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["date"] = pd.to_datetime(chunk["date"], errors="raise").dt.strftime("%Y-%m-%d")
    chunk["store_nbr"] = pd.to_numeric(chunk["store_nbr"], errors="raise").astype("int16")
    chunk["family"] = chunk["family"].astype("string").str.strip()
    chunk["sales"] = pd.to_numeric(chunk["sales"], errors="raise").astype("float32")
    if (chunk["sales"] < 0).any():
        raise ValueError("Negative sales found; do not silently alter raw demand values.")
    chunk["onpromotion"] = chunk["onpromotion"].fillna(False).astype("bool")
    chunk["is_food"] = chunk["family"].isin(FOOD_FAMILIES)
    return chunk


def clean_sales() -> dict:
    """Create full and food-only clean views while recording exact retention."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    all_temp = ALL_OUTPUT.with_suffix(".tmp")
    food_temp = FOOD_OUTPUT.with_suffix(".tmp")
    all_temp.unlink(missing_ok=True)
    food_temp.unlink(missing_ok=True)

    total = food_rows = 0
    wrote_all_header = wrote_food_header = False
    for chunk in pd.read_csv(RAW_TRAIN, chunksize=CHUNK_SIZE):
        cleaned = _clean_chunk(chunk)
        total += len(cleaned)
        cleaned.to_csv(all_temp, mode="a", index=False, header=not wrote_all_header, quoting=csv.QUOTE_MINIMAL)
        wrote_all_header = True
        food = cleaned[cleaned["is_food"]]
        food_rows += len(food)
        if not food.empty:
            food.to_csv(food_temp, mode="a", index=False, header=not wrote_food_header, quoting=csv.QUOTE_MINIMAL)
            wrote_food_header = True

    all_temp.replace(ALL_OUTPUT)
    food_temp.replace(FOOD_OUTPUT)
    report = {
        "raw_rows": total,
        "cleaned_all_rows": total,
        "quality_deleted_rows": 0,
        "quality_deletion_rate": 0.0,
        "food_view_rows": food_rows,
        "food_view_retention_rate": round(food_rows / total, 6),
        "nonfood_scope_excluded_rows": total - food_rows,
        "raw_source": str(RAW_TRAIN),
        "rule": "All valid raw rows are retained in cleaned_sales_all.csv; cleaned_food_sales.csv is a non-destructive business-scope view.",
    }
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    print(json.dumps(clean_sales(), indent=2))


if __name__ == "__main__":
    main()
