from src.api import agent


def test_fallback_replenishment_grounded():
    result = agent.run_fallback("What should I reorder today?", "test")
    assert result["mode"] == "fallback"
    assert result["tool_trace"]
    assert result["tool_trace"][0]["tool"] == "get_replenishment_plan"
    assert "order" in result["reply"].lower()


def test_fallback_family_alias_and_store():
    result = agent.run_fallback("Forecast demand for store 1 milk", "test")
    assert result["mode"] == "fallback"
    assert result["tool_trace"][0]["tool"] == "get_demand_forecast"
    assert "DAIRY" in result["reply"]


def test_fallback_promotion_defaults():
    result = agent.run_fallback("What if we run a 30% promotion?", "test")
    assert result["tool_trace"][0]["tool"] == "simulate_promotion"
    assert "+30%" in result["reply"]


def test_run_agent_without_gateway_uses_fallback(monkeypatch):
    for key in ("LLM_GATEWAY_URL", "LLM_GATEWAY_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    result = agent.run_agent("Forecast demand for store 1 DAIRY")
    assert result["mode"] == "fallback"
