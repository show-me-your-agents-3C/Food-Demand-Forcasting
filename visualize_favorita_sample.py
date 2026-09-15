"""Create lightweight exploratory plots from a chunk of Favorita train.csv."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("/home/ubuntu/demand-forecasting/data/favorita/train.csv")
OUTPUT = Path("/home/ubuntu/demand-forecasting/output/favorita_sample_visualization.png")


def main() -> None:
    # Read only the first 90 days and selected columns; this is a representative
    # slice and keeps memory use small for the 3M-row source file.
    # The file is sorted by date, so reading the first ~160k rows covers the
    # first 90 days while avoiding a full-file read.
    df = pd.read_csv(
        INPUT,
        nrows=180_000,
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < df["date"].min() + pd.Timedelta(days=90)].copy()

    daily = df.groupby("date", as_index=False)["sales"].sum()
    store = df.groupby("store_nbr", as_index=False)["sales"].sum().sort_values("sales", ascending=False).head(10)
    family = df.groupby("family", as_index=False)["sales"].sum().sort_values("sales", ascending=False).head(10)
    promo = df.groupby("date", as_index=False)["onpromotion"].sum()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Favorita sales — first 90 days", fontsize=16, fontweight="bold")

    axes[0, 0].plot(daily["date"], daily["sales"], color="#2563eb", linewidth=1.8)
    axes[0, 0].set_title("Daily total sales")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Sales units")
    axes[0, 0].tick_params(axis="x", rotation=35)

    axes[0, 1].bar(store["store_nbr"].astype(str), store["sales"], color="#16a34a")
    axes[0, 1].set_title("Top 10 stores by sales")
    axes[0, 1].set_xlabel("Store number")
    axes[0, 1].set_ylabel("Sales units")

    axes[1, 0].barh(family["family"].iloc[::-1], family["sales"].iloc[::-1], color="#f97316")
    axes[1, 0].set_title("Top 10 product families")
    axes[1, 0].set_xlabel("Sales units")
    axes[1, 0].set_ylabel("Family")

    axes[1, 1].plot(promo["date"], promo["onpromotion"], color="#9333ea", linewidth=1.8)
    axes[1, 1].set_title("Promotion records per day")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Promotion count")
    axes[1, 1].tick_params(axis="x", rotation=35)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160)
    print(f"Rows plotted: {len(df):,}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
