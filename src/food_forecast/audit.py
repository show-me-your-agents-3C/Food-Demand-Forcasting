"""Data-quality audit for the Favorita source files (read-only)."""

import json
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "data" / "favorita"


def _csv_audit(path: Path, date_column: str | None = "date") -> dict:
    frame = pd.read_csv(path)
    result = {
        "file": path.name,
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "missing_by_column": {key: int(value) for key, value in frame.isna().sum().items() if value},
        "duplicate_rows": int(frame.duplicated().sum()),
    }
    if date_column and date_column in frame:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        result["invalid_dates"] = int(dates.isna().sum())
        result["date_range"] = [str(dates.min().date()), str(dates.max().date())]
    return result


def run_audit() -> dict:
    files = ["train.csv", "test.csv", "stores.csv", "holidays_events.csv", "oil.csv", "transactions.csv"]
    report = {name: _csv_audit(DATA_DIR / name, None if name == "stores.csv" else "date") for name in files}

    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["id", "date", "store_nbr", "family", "sales", "onpromotion"])
    report["train.csv"]["duplicate_ids"] = int(train["id"].duplicated().sum())
    report["train.csv"]["negative_sales"] = int((train["sales"] < 0).sum())
    report["train.csv"]["unique_stores"] = int(train["store_nbr"].nunique())
    report["train.csv"]["unique_families"] = int(train["family"].nunique())

    return report


def main() -> None:
    report = run_audit()
    output = PROJECT_ROOT / "outputs" / "data_quality_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
