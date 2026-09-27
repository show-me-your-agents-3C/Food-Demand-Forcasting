from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "favorita" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FOOD_FAMILIES = [
    "BREAD/BAKERY", "BEVERAGES", "DAIRY", "DELI", "EGGS", "FROZEN FOODS",
    "GROCERY I", "GROCERY II", "MEATS", "POULTRY", "PREPARED FOODS", "PRODUCE", "SEAFOOD",
]

# Small default scope keeps local iteration fast and leaves a clear expansion path.
SELECTED_STORES = [1, 2, 3]
SELECTED_FAMILIES = ["BREAD/BAKERY", "BEVERAGES", "DAIRY", "MEATS", "POULTRY", "PRODUCE", "SEAFOOD", "FROZEN FOODS"]
LOOKBACK_DAYS = 120
FORECAST_DAYS = 7
SEASONAL_LAG_DAYS = 7
