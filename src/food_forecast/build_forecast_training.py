"""Create a compact, leakage-safe first forecasting table.

This is a derived model view. Source and cleaned records remain untouched.
Zero sales are retained. Highly intermittent series are registered separately
for a future intermittent-demand model rather than discarded from the project.
"""

from collections import defaultdict, deque
import csv
import json
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT


PROCESSED = PROJECT_ROOT / "data" / "processed"
SOURCE = PROCESSED / "model_base_food.csv"
OUTPUT = PROCESSED / "forecast_training_v1.csv"
REGISTRY = PROCESSED / "series_registry.csv"
REPORT = PROCESSED / "forecast_training_v1_report.json"
CHUNK_SIZE = 100_000
MIN_NONZERO_DAYS = 30
MIN_NONZERO_RATE = 0.10


def series_statistics() -> pd.DataFrame:
    counts: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    for chunk in pd.read_csv(SOURCE, usecols=["store_nbr", "family", "sales"], chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            key = (int(row.store_nbr), str(row.family))
            counts[key][0] += 1
            counts[key][1] += int(float(row.sales) > 0)
    rows = []
    for (store_nbr, family), (observations, nonzero_days) in counts.items():
        rate = nonzero_days / observations
        status = "active" if nonzero_days >= MIN_NONZERO_DAYS and rate >= MIN_NONZERO_RATE else "intermittent_or_inactive"
        rows.append({
            "store_nbr": store_nbr,
            "family": family,
            "observations": observations,
            "nonzero_days": nonzero_days,
            "nonzero_rate": round(rate, 6),
            "series_status": status,
        })
    return pd.DataFrame(rows).sort_values(["series_status", "store_nbr", "family"])


def main() -> None:
    registry = series_statistics()
    registry.to_csv(REGISTRY, index=False)
    active = set(map(tuple, registry.loc[registry["series_status"] == "active", ["store_nbr", "family"]].to_records(index=False)))

    temp = OUTPUT.with_suffix(".tmp")
    temp.unlink(missing_ok=True)
    histories: dict[tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=14))
    wrote_header = False
    prior_date = None
    active_input_rows = output_rows = zero_rows_retained = warmup_rows = 0

    usecols = [
        "date", "store_nbr", "family", "sales", "onpromotion", "holiday_event_count",
        "is_national_holiday", "is_national_event", "day_of_week", "month", "is_weekend",
    ]
    for chunk in pd.read_csv(SOURCE, usecols=usecols, parse_dates=["date"], chunksize=CHUNK_SIZE):
        if prior_date is not None and chunk["date"].min() < prior_date:
            raise ValueError("Source is not ordered by date; refusing to compute potentially leaky lag features.")
        prior_date = chunk["date"].max()
        chunk = chunk[[(int(row.store_nbr), str(row.family)) in active for row in chunk[["store_nbr", "family"]].itertuples(index=False)]].copy()
        output = []
        for row in chunk.itertuples(index=False):
            key = (int(row.store_nbr), str(row.family))
            history = histories[key]
            active_input_rows += 1
            sales = float(row.sales)
            # Keep zero sales: they describe observed no-demand days.
            if len(history) >= 14:
                output.append({
                    "date": row.date.strftime("%Y-%m-%d"),
                    "store_nbr": key[0],
                    "family": key[1],
                    "sales": sales,
                    "onpromotion": bool(row.onpromotion),
                    "holiday_event_count": int(row.holiday_event_count),
                    "is_national_holiday": bool(row.is_national_holiday),
                    "is_national_event": bool(row.is_national_event),
                    "day_of_week": int(row.day_of_week),
                    "month": int(row.month),
                    "is_weekend": bool(row.is_weekend),
                    "sales_lag_7": history[-7],
                    "sales_lag_14": history[0],
                    "sales_rolling_mean_7": sum(list(history)[-7:]) / 7,
                })
                output_rows += 1
                zero_rows_retained += int(sales == 0)
            else:
                warmup_rows += 1
            history.append(sales)
        if output:
            pd.DataFrame(output).to_csv(temp, mode="a", index=False, header=not wrote_header, quoting=csv.QUOTE_MINIMAL)
            wrote_header = True

    temp.replace(OUTPUT)
    total_source_rows = int(registry["observations"].sum())
    inactive_rows = int(registry.loc[registry["series_status"] != "active", "observations"].sum())
    report = {
        "source_food_rows": total_source_rows,
        "source_series": int(len(registry)),
        "active_series": int((registry["series_status"] == "active").sum()),
        "intermittent_or_inactive_series": int((registry["series_status"] != "active").sum()),
        "active_input_rows": active_input_rows,
        "intermittent_scope_excluded_rows": inactive_rows,
        "warmup_rows_without_14_day_history": warmup_rows,
        "forecast_training_rows": output_rows,
        "forecast_training_retention_from_active": round(output_rows / active_input_rows, 6),
        "zero_sales_rows_retained": zero_rows_retained,
        "quality_deleted_rows": 0,
        "note": "Excluded intermittent series and 14-day feature warm-up rows remain preserved in upstream cleaned files; no raw data is deleted.",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
