from langchain_core.messages import AIMessage

from src.api import agent


def _configure_gateway(monkeypatch):
    monkeypatch.setenv("LLM_GATEWAY_URL", "http://localhost:11434")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")


def test_llm_graph_routes_tool_call(monkeypatch):
    _configure_gateway(monkeypatch)
    calls = {"count": 0}

    def fake_invoke(llm, messages, max_retries=3, backoff=2.0):
        calls["count"] += 1
        if calls["count"] == 1:
            return AIMessage(content='{"tool": "get_forecast_metrics", "args": {}}')
        return AIMessage(content="WAPE is 0.1742 based on the tool result.")

    monkeypatch.setattr(agent, "_invoke_with_retry", fake_invoke)
    result = agent._run_llm("What is the forecast accuracy?", [])
    assert result["mode"] == "llm"
    assert result["tool_trace"][0]["tool"] == "get_forecast_metrics"
    assert "0.1742" in result["reply"]


def test_run_agent_uses_llm_when_configured(monkeypatch):
    _configure_gateway(monkeypatch)

    def fake_invoke(llm, messages, max_retries=3, backoff=2.0):
        return AIMessage(content="No tool needed.")

    monkeypatch.setattr(agent, "_invoke_with_retry", fake_invoke)
    result = agent.run_agent("hello")
    assert result["mode"] == "llm"
    assert result["reply"] == "No tool needed."
