from src.api import tools


def test_registry_covers_specs():
    spec_names = {spec["name"] for spec in tools.TOOL_SPECS}
    assert spec_names == set(tools.TOOL_REGISTRY)


def test_tools_are_json_serializable():
    import json

    for name, function in tools.TOOL_REGISTRY.items():
        if name == "simulate_promotion":
            result = function(store_nbr=1, family="BEVERAGES", uplift_pct=10)
        elif name in {"get_inventory_status",}:
            result = function(store_nbr=1, family="BEVERAGES")
        elif name == "get_replenishment_plan":
            result = function(limit=3)
        else:
            result = function()
        json.dumps(result, default=str)
        assert "source" in result


def test_forecast_tool_horizon():
    result = tools.get_demand_forecast(store_nbr=1, family="DAIRY", horizon_days=3)
    assert result["row_count"] == 3
