import json

import pandas as pd
import pytest

from src.food_forecast.chat_agent import ChatSession
from src.food_forecast.chat_agent import _fallback
from src.food_forecast.forecast_tools import (
    FINAL,
    call_tool,
    get_forecast,
    get_priority_replenishments,
    get_replenishment,
    what_if_promotion,
)


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        reply = next(self.replies)
        return reply(messages) if callable(reply) else reply


def _tool(name, args):
    return json.dumps({"type": "tool", "name": name, "args": args}, ensure_ascii=False)


def _answer(text="The result is based on the retrieved tool facts."):
    return json.dumps({"type": "answer", "text": text}, ensure_ascii=False)


def test_forecast_matches_published_v3_artifact_and_reports_snapshot():
    artifact = pd.read_csv(FINAL / "forecast.csv")
    rows = artifact[(artifact.store_nbr == 3) & (artifact.family == "BEVERAGES")]
    result = get_forecast(3, "饮料")
    assert result["total"]["p50"] == round(rows.p50.sum(), 1)
    assert result["daily"][0]["date"] == rows.sort_values("date").iloc[0].date
    assert result["data"]["model_id"] == "v3.0:v3_tweedie"
    assert result["data"]["data_snapshot_through"] == "2017-08-15"
    assert result["data"]["forecast_date_range"] == ["2017-08-16", "2017-08-22"]
    assert result["data"]["backtest_interval_coverage_pct"] == 77.3


def test_tools_validate_store_family_dates_and_limits():
    assert "Available stores" in call_tool("get_forecast", {"store_nbr": 999, "family": "DAIRY"})["error"]
    assert "Available families" in call_tool("get_forecast", {"store_nbr": 3, "family": "NOT_A_FAMILY"})["error"]
    assert "Ambiguous family" in call_tool("get_forecast", {"store_nbr": 3, "family": "肉类"})["error"]
    assert "YYYY-MM-DD" in call_tool("get_forecast", {"store_nbr": 3, "family": "DAIRY", "start_date": "bad"})["error"]
    assert "available range" in call_tool("get_forecast", {"store_nbr": 3, "family": "DAIRY", "start_date": "2026-01-01"})["error"]
    assert "top_n" in call_tool("get_priority_replenishments", {"top_n": 21})["error"]
    assert "missing" in call_tool("get_forecast", {"store_nbr": 3})["error"]
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        what_if_promotion(3, "DAIRY", 10, ["08-18-2017"])


def test_replenishment_uses_v3_forecast_and_explicit_simulation_inputs():
    result = get_replenishment(3, "BEVERAGES")
    assert result["recommended_order_qty"] % result["moq_simulated"] == 0
    assert result["action"] == ("order_today" if result["current_stock_simulated"] < result["reorder_point_simulated"] else "monitor")
    assert result["data"]["source_files"] == [
        "outputs/final/forecast.csv", "outputs/final/future_features.csv.gz",
        "outputs/final/model_metadata.json", "outputs/final/metrics.json",
    ]
    assert "simulated" in " ".join(result)
    priorities = get_priority_replenishments(store_nbr=3, top_n=2)
    assert len(priorities["items"]) == 2
    assert priorities["items"] == sorted(
        priorities["items"],
        key=lambda row: (row["action"] != "order_today", -row["forecast_7d_p50"]),
    )
    partial = get_replenishment(3, "BEVERAGES", "2017-08-17", "2017-08-19")
    assert partial["data"]["forecast_date_range"] == ["2017-08-17", "2017-08-19"]


def test_replenishment_rejects_mixed_forecast_and_feature_snapshots(monkeypatch):
    from src.food_forecast import forecast_tools

    features = forecast_tools._future_features()
    missing_key = (
        (features["store_nbr"].astype(int) == 3)
        & (features["family"].astype(str) == "BEVERAGES")
        & (features["date"] == pd.Timestamp("2017-08-22"))
    )
    assert missing_key.any()
    mismatched = features.loc[~missing_key].copy()
    monkeypatch.setattr(forecast_tools, "_future_features", lambda: mismatched)
    result = call_tool("get_replenishment", {"store_nbr": 3, "family": "BEVERAGES"})
    assert "inconsistent store/family/date keys" in result["error"]


def test_llm_tool_request_result_and_answer_complete_the_loop():
    def answer_with_fact(messages):
        result = json.loads(messages[-1]["content"].removeprefix("TOOL_RESULT_JSON "))
        data = result["data"]
        dates = data["forecast_date_range"]
        disclosure = (f"The snapshot ends {data['data_snapshot_through']}; forecast dates are "
                      f"{dates[0]} through {dates[1]}. This is historical data, not live sales.")
        return _answer(f"The forecast p50 is {result['total']['p50']} units. {disclosure}")

    model = ScriptedLLM(_tool("get_forecast", {"store_nbr": 3, "family": "BEVERAGES"}), answer_with_fact)
    result = ChatSession(llm=model).ask("3号店饮料未来七天卖多少？")
    assert result["status"] == "answered"
    assert result["tool_trace"][0]["tool"] == "get_forecast"
    assert result["tool_trace"][0]["status"] == "success"
    assert result["tool_trace"][0]["source_files"] == [
        "outputs/final/forecast.csv", "outputs/final/model_metadata.json", "outputs/final/metrics.json",
    ]
    assert str(result["evidence"][0]["result"]["total"]["p50"]) in result["text"]


def test_followups_keep_store_change_family_and_sessions_are_isolated():
    model = ScriptedLLM(
        _tool("get_forecast", {"store_nbr": 3, "family": "BEVERAGES"}), _answer(),
        _tool("get_replenishment", {}), _answer(),
        _tool("get_forecast", {"family": "DAIRY"}), _answer(),
    )
    session = ChatSession(llm=model)
    session.ask("3号店饮料未来七天卖多少？")
    replenishment = session.ask("应该补多少？")
    dairy = session.ask("那乳制品呢？")
    assert replenishment["tool_trace"][0]["args"] == {"store_nbr": 3, "family": "BEVERAGES"}
    assert dairy["tool_trace"][0]["args"] == {"store_nbr": 3, "family": "DAIRY"}
    assert session.context["family"] == "DAIRY"

    other = ChatSession(llm=ScriptedLLM('{"type":"clarify","text":"Which store should I use?"}'))
    response = other.ask("What about dairy?")
    assert response["status"] == "clarification"
    assert "store_nbr" not in other.context
    session.clear()
    assert not session.context and not session.messages and not session.tool_trace


@pytest.mark.parametrize("reply", ["not json", '{"type":"tool","name":"__import__","args":{}}'])
def test_invalid_json_and_unregistered_tools_never_execute(reply):
    session = ChatSession(llm=ScriptedLLM(reply))
    result = session.ask("查询补货")
    assert result["status"] == "fallback"
    assert not result["tool_trace"]
    assert "__import__" not in result["text"]


def test_gateway_timeout_retries_once_and_returns_safe_fallback():
    calls = []

    def timeout(_messages):
        calls.append(1)
        raise TimeoutError("secret-value-must-not-appear")

    result = ChatSession(llm=timeout, max_retries=1, retry_delay=0).ask("查一下")
    assert len(calls) == 2
    assert result["status"] == "fallback"
    assert result["fallback_reason"] == "gateway_timeout"
    assert "secret-value-must-not-appear" not in result["text"]


def test_tool_call_limit_stops_repeated_requests():
    model = ScriptedLLM(_tool("get_forecast", {"store_nbr": 3, "family": "DAIRY"}))
    result = ChatSession(llm=model, max_tool_calls=1).ask("3号店乳制品预测")
    assert len(result["tool_trace"]) == 1
    assert model.calls <= 3  # one request, one post-tool response, bounded by LangGraph recursion
    assert result["status"] == "fallback"


def test_numeric_claim_without_evidence_uses_template_fallback():
    model = ScriptedLLM(
        _tool("get_forecast", {"store_nbr": 3, "family": "DAIRY"}),
        _answer("销量是 999999 件。"),
    )
    result = ChatSession(llm=model).ask("3号店乳制品销量？")
    assert result["status"] == "fallback"
    assert "[Template fallback]" in result["text"]


def test_template_fallback_displays_historical_dates_and_simulated_inventory():
    model = ScriptedLLM(
        _tool("get_replenishment", {"store_nbr": 3, "family": "DAIRY"}),
        "malformed final response",
    )
    result = ChatSession(llm=model).ask("3号店乳制品应该补多少？")
    assert result["status"] == "fallback"
    assert "snapshot ends 2017-08-15" in result["text"]
    assert "forecast dates are 2017-08-16 through 2017-08-22" in result["text"]
    assert "simulated stock" in result["text"]
    assert "does not place orders" in result["text"]


def test_english_priority_and_forecast_questions_use_the_registered_tools():
    model = ScriptedLLM(
        _tool("get_priority_replenishments", {"store_nbr": 3, "top_n": 5}),
        _answer("Store 3 priority items are listed above. The snapshot ends 2017-08-15; forecast dates are 2017-08-16 through 2017-08-22. This is historical data, not live inventory. Inventory, lead time, shelf life, and MOQ are simulated assumptions. This Agent does not place orders."),
        _tool("get_forecast", {"store_nbr": 3, "family": "BEVERAGES"}),
        _answer("The seven-day beverages forecast is ready. The snapshot ends 2017-08-15; forecast dates are 2017-08-16 through 2017-08-22. This is historical data, not live sales."),
    )
    session = ChatSession(llm=model)
    priority = session.ask("Which categories should Store 3 replenish first?")
    forecast = session.ask("What is the seven-day forecast for beverages at Store 3?")
    assert priority["tool_trace"][0]["tool"] == "get_priority_replenishments"
    assert forecast["tool_trace"][0]["tool"] == "get_forecast"
    assert forecast["status"] == "answered"
    assert priority["context"]["store_nbr"] == 3


def test_english_multi_turn_order_why_and_family_change_keep_store_context():
    model = ScriptedLLM(
        _tool("get_forecast", {"store_nbr": 3, "family": "BEVERAGES"}), _answer(),
        _tool("get_replenishment", {}), _answer(),
        _tool("explain_forecast", {}), _answer(),
        _tool("get_forecast", {"family": "DAIRY"}), _answer(),
    )
    session = ChatSession(llm=model)
    session.ask("What is the seven-day forecast for beverages at Store 3?")
    order = session.ask("How much should we order?")
    why = session.ask("Why?")
    dairy = session.ask("What about dairy?")
    assert order["tool_trace"][0]["args"] == {"store_nbr": 3, "family": "BEVERAGES"}
    assert why["tool_trace"][0]["args"] == {"store_nbr": 3, "family": "BEVERAGES"}
    assert dairy["tool_trace"][0]["args"] == {"store_nbr": 3, "family": "DAIRY"}
    assert session.context["data_snapshot_through"] == "2017-08-15"


def test_english_explicit_store_change_overrides_previous_context():
    model = ScriptedLLM(
        _tool("get_forecast", {"store_nbr": 3, "family": "BEVERAGES"}), _answer(),
        _tool("get_forecast", {"family": "DAIRY"}), _answer(),
    )
    session = ChatSession(llm=model)
    session.ask("What is the forecast for beverages at Store 3?")
    changed = session.ask("What about dairy at Store 4?")
    assert changed["tool_trace"][0]["args"] == {"store_nbr": 4, "family": "DAIRY"}
    assert session.context["data_snapshot_through"] == "2017-08-15"


def test_english_missing_information_gets_an_english_clarification():
    session = ChatSession(llm=ScriptedLLM('{"type":"clarify","text":"Which store and product family should I use?"}'))
    response = session.ask("How much should we order?")
    assert response["status"] == "clarification"
    assert response["text"] == "Which store and product family should I use?"


def test_agent_does_not_execute_a_tool_with_model_guessed_entities():
    model = ScriptedLLM(_tool("get_replenishment", {"store_nbr": 44, "family": "DAIRY"}))
    response = ChatSession(llm=model).ask("How much should we order?")
    assert response["status"] == "clarification"
    assert response["text"] == "Which store number and product family should I use?"
    assert response["text"].isascii()
    assert response["tool_trace"] == []


def test_english_empty_input_and_cli_copy_are_english():
    response = ChatSession(llm=ScriptedLLM()).ask(" ")
    assert response["status"] == "clarification"
    assert response["text"].startswith("Please provide")


def test_prompt_preserves_protocol_and_requires_english_errors():
    def inspect_prompt(messages):
        system = messages[0]["content"]
        assert "Answer in English" in system
        assert "Ask a clarification in English" in system
        assert "errors in English" in system
        assert '"type":"tool"' in system
        assert '"name": "get_forecast"' in system
        return '{"type":"clarify","text":"Which store should I use?"}'

    assert ChatSession(llm=inspect_prompt).ask("How much should we order?")["status"] == "clarification"


def test_gateway_failure_fallback_is_english():
    def fail(_messages):
        raise ConnectionError("do-not-show-raw-error")

    result = ChatSession(llm=fail, max_retries=0).ask("What is the forecast?")
    assert result["text"].startswith("[Template fallback] Unable to retrieve tool data.")
    assert "do-not-show-raw-error" not in result["text"]


def test_fallback_uses_english_when_snapshot_metadata_is_missing():
    text = _fallback("get_forecast", {
        "store_nbr": 3,
        "family": "DAIRY",
        "total": {"p10": 10, "p50": 20, "p90": 30},
    })
    assert text.isascii()
    assert "forecast origin is unknown" in text


def test_non_english_model_answer_is_replaced_by_english_fallback():
    def answer_in_chinese(messages):
        result = json.loads(messages[-1]["content"].removeprefix("TOOL_RESULT_JSON "))
        return _answer(f"预测需求量为 {result['total']['p50']} 件。")

    model = ScriptedLLM(
        _tool("get_forecast", {"store_nbr": 3, "family": "BEVERAGES"}),
        answer_in_chinese,
    )
    result = ChatSession(llm=model).ask("What is the seven-day forecast for beverages at Store 3?")
    assert result["status"] == "fallback"
    assert result["text"].isascii()
    assert result["text"].startswith("[Template fallback]")


def test_unsupported_forecast_dates_are_replaced_by_tool_dates():
    model = ScriptedLLM(
        _tool("get_forecast", {"store_nbr": 3, "family": "BEVERAGES"}),
        lambda messages: _answer(
            "The forecast runs from 2017-08-23 through 2017-08-29. "
            "The snapshot ends 2017-08-15; this is historical data, not live sales."
        ),
    )
    result = ChatSession(llm=model).ask("What is the seven-day forecast for beverages at Store 3?")
    assert result["status"] == "fallback"
    assert "2017-08-16 through 2017-08-22" in result["text"]


def test_unsupported_risk_level_is_replaced_by_tool_risk():
    def contradict_risk(messages):
        payload = json.loads(messages[-1]["content"].removeprefix("TOOL_RESULT_JSON "))
        actual = payload["stockout_risk_simulated"]
        claimed = "low" if actual == "high" else "high"
        data = payload["data"]
        dates = data["forecast_date_range"]
        disclosure = (f"The snapshot ends {data['data_snapshot_through']}; forecast dates are "
                  f"{dates[0]} through {dates[1]}. This is historical data, not live sales. "
                  "Inventory, lead time, shelf life, and MOQ are simulated assumptions. "
                  "This Agent does not place orders.")
        return _answer(f"Stock-out risk is {claimed}. {disclosure}")

    model = ScriptedLLM(
        _tool("get_replenishment", {"store_nbr": 3, "family": "BEVERAGES"}),
        contradict_risk,
    )
    result = ChatSession(llm=model).ask("How much should we order for beverages at Store 3?")
    assert result["status"] == "fallback"
    assert "stockout risk" in result["text"]