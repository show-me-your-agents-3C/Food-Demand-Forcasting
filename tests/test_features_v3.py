"""Leakage and fallback tests for the v3 forecaster (synthetic data, runs in seconds)."""

import numpy as np
import pandas as pd

from src.food_forecast.features_v3 import FEATURES, PROMO_FEATURES, Panel, build_features
from src.food_forecast.modeling_v3 import score, tsb_forecast


def _panel(n_days: int = 140, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    series = pd.DataFrame({
        "store_nbr": [1, 1], "family": ["DAIRY", "PRODUCE"], "series_status": ["active", "active"],
        "city": ["Quito", "Quito"], "state": ["Pichincha", "Pichincha"], "store_type": ["D", "D"], "store_cluster": [13, 13],
    })
    dates = pd.date_range("2017-01-04", periods=n_days, freq="D")  # a Wednesday
    shape = (2, n_days)
    calendar = {name: np.zeros(shape, np.int8) for name in
                ("is_holiday", "holiday_scope", "is_national_event", "is_workday", "days_to_holiday", "days_since_holiday")}
    return Panel(series, dates, rng.poisson(50, shape).astype(float), rng.integers(0, 5, shape).astype(float),
                 np.ones(shape, bool), np.zeros(2, int), calendar)


def test_sales_features_ignore_sales_on_or_after_the_origin():
    panel = _panel()
    base = build_features(panel)
    origin = pd.Timestamp("2017-04-19")  # Wednesday
    assert origin.dayofweek == 2
    week = base["date"].between(origin, origin + pd.Timedelta(days=6))
    assert sorted(base.loc[week, "horizon_day"].unique()) == list(range(1, 8))

    shocked = _panel()
    shocked.sales[:, panel.dates >= origin] *= 100  # change everything the planner cannot know yet
    after = build_features(shocked)
    sales_features = [f for f in FEATURES if f not in PROMO_FEATURES]
    pd.testing.assert_frame_equal(base.loc[week, sales_features], after.loc[week, sales_features])


def test_promotion_count_is_kept_not_collapsed_to_flag():
    panel = _panel()
    panel.promo[:, :] = 7
    frame = build_features(panel)
    assert (frame["onpromotion"] == 7).all()
    assert (frame["promo_sum_7"] == 49).all()


def test_tsb_zero_when_recent_month_is_empty_and_positive_otherwise():
    assert tsb_forecast(np.r_[np.full(100, 5.0), np.zeros(28)]) == 0.0
    sporadic = np.tile([0, 0, 0, 8.0], 30)
    assert 1.0 < tsb_forecast(sporadic) < 4.0  # about 8 units x 25% occurrence


def test_score_sign_convention():
    result = score([10, 10], [12, 12])
    assert result["wape"] == 0.2 and result["bias"] == 0.2


def test_routing_sends_new_series_to_cold_start_and_sparse_ones_to_tsb():
    from src.food_forecast.modeling_v3 import route
    assert list(route([10, 200, 200, 200], [0.0, 0.8, 0.1, 0.8], [1.0, 1.0, 1.0, 0.3])) == [
        "cold_start", "intermittent", "lightgbm", "lightgbm"]  # last: store was closed, not slow demand


def test_simulation_counts_lost_sales_and_spoilage():
    from src.food_forecast.business_value import simulate
    # Deliver 10 on day 1 only; shelf life 2 days; demand 3 per day for 3 days.
    result = simulate(np.array([3.0, 3.0, 3.0]), np.array([10.0, 0, 0]), np.array([True, False, False]), shelf=2)
    assert result["sold"] == 6 and result["wasted"] == 4 and result["lost"] == 3
