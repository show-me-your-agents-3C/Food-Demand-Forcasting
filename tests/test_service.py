from src.api import service


def test_scope_nonempty():
    scope = service.get_scope()
    assert scope["stores"]
    assert scope["families"]


def test_metrics_keys():
    metrics = service.get_metrics()
    assert {"mae", "wape", "n_predictions"} <= set(metrics)


def test_forecast_filter():
    rows = service.get_forecast(store_nbr=1, family="BEVERAGES", horizon_days=7)
    assert len(rows) == 7
    assert all(row["store_nbr"] == 1 and row["family"] == "BEVERAGES" for row in rows)


def test_replenishment_sorted_and_filtered():
    rows = service.get_replenishment(limit=5)
    assert rows
    assert rows[0]["action"] in {"order_today", "monitor"}
    only_today = service.get_replenishment(action="order_today", limit=100)
    assert all(row["action"] == "order_today" for row in only_today)


def test_action_lookup():
    row = service.get_action(1, "BEVERAGES")
    assert row is not None
    assert row["recommended_order_qty"] >= 0


def test_simulate_promotion_raises_unknown():
    result = service.simulate_promotion(1, "BEVERAGES", 25)
    assert result["simulated_forecast_7d"] > result["baseline_forecast_7d"]
    assert result["extra_units_7d"] > 0
    assert result["simulated_order_qty"] >= result["baseline_order_qty"]
