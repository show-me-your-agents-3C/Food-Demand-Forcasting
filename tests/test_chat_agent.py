import json

import pandas as pd
import pytest

from src.food_forecast.chat_agent import (
    ASCII_EQUIVALENTS,
    ChatSession,
    _answer_facts_supported,
    _answer_numbers_supported,
    _answer_risks_supported,
    _complete_evidence_disclosures,
    _fallback,
    _requires_forecast_tool,
    _requires_replenishment_tool,
)
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


def _tool_result(messages):
    content = messages[-1]["content"].removeprefix("TOOL_RESULT_JSON ")
    return json.loads(content.split("\n\n", 1)[0])


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


def test_numeric_evidence_accepts_rounded_tool_values_and_date_range_duration():
    result = call_tool("get_replenishment", {"store_nbr": 3, "family": "BEVERAGES"})
    evidence = [{"tool": "get_replenishment", "result": result}]
    answer = (
        "Stockout risk is high. Current stock is 11,021; the reorder point is 57,563 units. "
        "The 7-day forecast is 57,563 units. Snapshot 2017-08-15, forecast dates "
        "2017-08-16 through 2017-08-22."
    )
    assert _answer_numbers_supported(answer, evidence)


def test_replenishment_evidence_accepts_rounded_numbers_bold_risk_and_full_dates():
    result = call_tool("get_replenishment", {"store_nbr": 3, "family": "BEVERAGES"})
    evidence = [{"tool": "get_replenishment", "result": result}]
    answer = (
        "Stockout risk is **high**. Current simulated stock is 11,021 units and reorder point is 57,563 units. "
        "The 7-day forecast is 57,563 units; recommended order is 63,820 units. "
        "Snapshot through 2017-08-15, forecast origin 2017-08-16, dates 2017-08-16 through 2017-08-22. "
        "This is historical data, not live sales. Inventory, lead time, shelf life, and MOQ are simulated assumptions. "
        "This Agent does not place orders."
    )
    assert _answer_risks_supported(answer, evidence)
    assert _answer_risks_supported("Stockout risk: high.", evidence)
    assert not _answer_risks_supported("Stockout risk: low.", evidence)
    assert _answer_facts_supported(answer, evidence)


def test_replenishment_no_order_phrase_accepts_equivalent_clear_wording():
    result = call_tool("get_replenishment", {"store_nbr": 3, "family": "BEVERAGES"})
    evidence = [{"tool": "get_replenishment", "result": result}]
    answer = (
        "Stockout risk for BEVERAGES at Store 3 is high. Snapshot through 2017-08-15, origin 2017-08-16, "
        "forecast dates 2017-08-16 through 2017-08-22. This is historical data, not live sales. "
        "Inventory, lead time, shelf life, and MOQ are simulated assumptions. No order is placed."
    )
    assert _answer_facts_supported(answer, evidence)


def test_historical_data_snapshot_disclosure_is_equivalent_to_not_live():
    result = call_tool("get_replenishment", {"store_nbr": 1, "family": "PRODUCE"})
    evidence = [{"tool": "get_replenishment", "result": result}]
    answer = (
        "Recommended order quantity for PRODUCE at Store 1 is 2910 units (historical data snapshot through 2017-08-15, "
        "forecast origin 2017-08-16, range 2017-08-16 to 2017-08-22). Simulated inventory is 3178.6, lead time 1 day, "
        "shelf life 3 days, MOQ 10. These are simulated assumptions. This agent provides decision support only and does not place orders."
    )
    assert _answer_facts_supported(answer, evidence)


@pytest.mark.parametrize("question", [
    "How many units should I order?",
    "How much should we replenish?",
    "What order quantity should I use?",
])
def test_replenishment_quantity_phrasings_require_fresh_tool_evidence(question):
    assert _requires_replenishment_tool(question)


def test_forecast_intent_requires_fresh_tool_evidence():
    assert _requires_forecast_tool("Give me the seven-day forecast for PRODUCE at Store 1.")
    assert not _requires_forecast_tool("Which categories should Store 3 replenish first?")


def test_priority_aggregate_risk_and_action_are_checked_against_every_item():
    result = call_tool("get_priority_replenishments", {"store_nbr": 3, "top_n": 5})
    assert result["items"]
    assert all(item["stockout_risk_simulated"] == "high" for item in result["items"])
    assert all(item["action"] == "order_today" for item in result["items"])
    evidence = [{"tool": "get_priority_replenishments", "result": result}]
    response = (
        "Top priorities for Store 3: " + ", ".join(
            f"{item['family']} ({item['forecast_7d_p50']} p50)" for item in result["items"]
        ) + ". All show high stockout risk and order_today action. "
        "Data snapshot through 2017-08-15, forecast origin 2017-08-16, forecast range 2017-08-16 to 2017-08-22. "
        "This is historical data, not live sales. Inventory, lead time, shelf life, and MOQ are simulated assumptions. "
        "No order is placed."
    )
    assert _answer_facts_supported(response, evidence)

    mismatched = {**result, "items": [dict(item) for item in result["items"]]}
    mismatched["items"][-1]["stockout_risk_simulated"] = "low"
    assert not _answer_risks_supported(
        "All show high stockout risk and order_today action.",
        [{"tool": "get_priority_replenishments", "result": mismatched}],
    )
    mismatched_action = {**result, "items": [dict(item) for item in result["items"]]}
    mismatched_action["items"][-1]["action"] = "monitor"
    assert not _answer_risks_supported(
        "All show high stockout risk and order_today action.",
        [{"tool": "get_priority_replenishments", "result": mismatched_action}],
    )


def test_missing_replenishment_disclaimer_is_completed_without_weakening_numbers():
    result = call_tool("get_replenishment", {"store_nbr": 1, "family": "PRODUCE"})
    evidence = [{"tool": "get_replenishment", "result": result}]
    response = (
        "Stockout risk for PRODUCE at Store 1 is high (data snapshot through 2017-08-15, "
        "forecast origin 2017-08-16, forecast dates 2017-08-16 to 2017-08-22). "
        "Simulated current stock is 3178.6 units against a reorder point of 4055.4. "
        "Recommended order quantity is 2910 units. This is historical data, not live sales. "
        "Inventory, lead time, shelf life, and MOQ are simulated assumptions."
    )
    completed = _complete_evidence_disclosures(response, evidence)
    assert "does not place orders" in completed.lower()
    assert _answer_facts_supported(completed, evidence)
    fabricated = completed.replace("2910 units", "2911 units")
    assert not _answer_numbers_supported(fabricated, evidence)


def test_priority_historical_wording_gets_explicit_boundary_and_safe_disclaimer():
    result = call_tool("get_priority_replenishments", {"store_nbr": 3, "top_n": 5})
    evidence = [{"tool": "get_priority_replenishments", "result": result}]
    item_list = ", ".join(
        f"{item['family']} ({item['recommended_order_qty']} units)" for item in result["items"]
    )
    response = (
        f"Top priorities for Store 3 (data snapshot through 2017-08-15, forecast origin 2017-08-16, "
        f"forecast dates 2017-08-16 to 2017-08-22): {item_list}. All have high stockout risk. "
        "This is decision support based on historical data. Inventory, lead time, shelf life, and MOQ are simulated assumptions."
    )
    completed = _complete_evidence_disclosures(response, evidence)
    assert "not live sales or inventory" in completed.lower()
    assert "does not place orders" in completed.lower()
    assert _answer_facts_supported(completed, evidence)
    fabricated = completed.replace("63820 units", "63821 units")
    assert not _answer_numbers_supported(fabricated, evidence)


def test_store_one_produce_four_turn_sequence_uses_fresh_evidence_and_priority_items():
    def forecast_answer(messages):
        result = _tool_result(messages)
        data = result["data"]
        start, end = data["forecast_date_range"]
        return _answer(
            f"PRODUCE forecast p50 is {result['total']['p50']} units. Snapshot {data['data_snapshot_through']}; "
            f"origin {data['forecast_origin']}; forecast dates {start} through {end}. Historical data, not live sales."
        )

    def replenishment_answer(messages):
        result = _tool_result(messages)
        data = result["data"]
        start, end = data["forecast_date_range"]
        return _answer(
            f"Stockout risk for {result['family']} at Store {result['store_nbr']} is {result['stockout_risk_simulated']}. "
            f"Recommended order is {result['recommended_order_qty']} units. Snapshot {data['data_snapshot_through']}; "
            f"origin {data['forecast_origin']}; forecast dates {start} through {end}. Historical data, not live sales. "
            "Inventory, lead time, shelf life, and MOQ are simulated assumptions. No order is placed."
        )

    def priority_answer(messages):
        result = _tool_result(messages)
        items = result["items"]
        data = result["data"]
        start, end = data["forecast_date_range"]
        listing = ", ".join(f"{item['family']} ({item['forecast_7d_p50']} p50)" for item in items)
        return _answer(
            f"Top priorities for Store 3: {listing}. All show high stockout risk and order_today action. "
            f"Data snapshot through {data['data_snapshot_through']}, forecast origin {data['forecast_origin']}, "
            f"forecast range {start} to {end}. Historical data, not live sales. "
            "Inventory, lead time, shelf life, and MOQ are simulated assumptions. No order is placed."
        )

    model = ScriptedLLM(
        _tool("get_forecast", {"store_nbr": 1, "family": "PRODUCE"}), forecast_answer,
        _tool("get_replenishment", {}), replenishment_answer,
        _answer("Recommended order quantity for PRODUCE at Store 1 is 9780 units."), replenishment_answer,
        _tool("get_priority_replenishments", {"store_nbr": 3, "top_n": 5}), priority_answer,
    )
    session = ChatSession(llm=model)
    forecast = session.ask("Give me the seven-day forecast for PRODUCE at Store 1.")
    risk = session.ask("Is there a stockout risk?")
    order = session.ask("How many units should I order?")
    priorities = session.ask("Which categories should Store 3 replenish first?")

    assert forecast["status"] == risk["status"] == order["status"] == priorities["status"] == "answered"
    assert risk["tool_trace"][0]["args"]["store_nbr"] == 1
    assert risk["tool_trace"][0]["args"]["family"] == "PRODUCE"
    assert order["tool_trace"][0]["tool"] == "get_replenishment"
    assert order["tool_trace"][0]["args"] == {"store_nbr": 1, "family": "PRODUCE"}
    assert str(order["evidence"][0]["result"]["recommended_order_qty"]) in order["text"]
    assert priorities["tool_trace"][0]["tool"] == "get_priority_replenishments"
    assert priorities["tool_trace"][0]["args"]["store_nbr"] == 3


def test_clear_then_missing_replenishment_context_clarifies_without_calling_llm():
    def forecast_answer(messages):
        result = _tool_result(messages)
        data = result["data"]
        start, end = data["forecast_date_range"]
        return _answer(
            f"Forecast p50 is {result['total']['p50']}. Snapshot {data['data_snapshot_through']}; "
            f"origin {data['forecast_origin']}; forecast dates {start} through {end}. "
            "This is historical data, not live sales."
        )

    model = ScriptedLLM(
        _tool("get_forecast", {"store_nbr": 1, "family": "PRODUCE"}),
        forecast_answer,
    )
    session = ChatSession(llm=model)
    assert session.ask("Give me the seven-day forecast for PRODUCE at Store 1.")["status"] == "answered"
    assert session.context["store_nbr"] == 1
    assert session.context["family"] == "PRODUCE"

    session.clear()
    response = session.ask("How many units should I order?")

    assert response["status"] == "clarification"
    assert response["text"] == "Which store number and product family should I use?"
    assert response["tool_trace"] == []
    assert response["evidence"] == []
    assert response["context"] == {}
    assert model.calls == 2


def test_repeated_identical_forecast_request_fetches_fresh_evidence():
    def forecast_answer(messages):
        result = _tool_result(messages)
        data = result["data"]
        start, end = data["forecast_date_range"]
        return _answer(
            f"PRODUCE at Store 1 has p50 forecast {result['total']['p50']} units for {start} through {end}. "
            f"Data snapshot through {data['data_snapshot_through']}; forecast origin {data['forecast_origin']}. "
            "This is historical data, not live sales."
        )

    model = ScriptedLLM(
        _tool("get_forecast", {"store_nbr": 1, "family": "PRODUCE"}), forecast_answer,
        _answer("The PRODUCE forecast at Store 1 is 16628.8 units."), forecast_answer,
    )
    session = ChatSession(llm=model)
    question = "Give me the seven-day forecast for PRODUCE at Store 1."
    first = session.ask(question)
    repeated = session.ask(question)

    assert first["status"] == "answered"
    assert repeated["status"] == "answered"
    assert repeated["fallback_reason"] is None
    assert repeated["tool_trace"][0]["tool"] == "get_forecast"
    assert repeated["tool_trace"][0]["status"] == "success"
    assert repeated["tool_trace"][0]["args"]["store_nbr"] == 1
    assert repeated["tool_trace"][0]["args"]["family"] == "PRODUCE"
    assert repeated["evidence"][0]["result"]["total"]["p50"] == first["evidence"][0]["result"]["total"]["p50"]
    assert model.calls == 4


@pytest.mark.parametrize("claim", [
    "The reorder point is 26,819 units.",
    "The forecast is 57,564 units.",
    "Current stock is 15 units.",
    "The forecast covers 8 days.",
])
def test_numeric_evidence_rejects_mismatched_and_date_component_numbers(claim):
    result = call_tool("get_replenishment", {"store_nbr": 3, "family": "BEVERAGES"})
    evidence = [{"tool": "get_replenishment", "result": result}]
    assert not _answer_numbers_supported(claim, evidence)


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
        result = _tool_result(messages)
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


def test_three_turn_followup_forces_fresh_replenishment_evidence():
    def forecast_answer(messages):
        result = _tool_result(messages)
        data = result["data"]
        start, end = data["forecast_date_range"]
        return _answer(
            f"Forecast p50 is {result['total']['p50']}. Snapshot {data['data_snapshot_through']}; "
            f"origin {data['forecast_origin']}; forecast range {start} through {end}. "
            "This is historical data, not live sales."
        )

    def replenishment_answer(messages):
        result = _tool_result(messages)
        data = result["data"]
        start, end = data["forecast_date_range"]
        return _answer(
            f"Stockout risk for {result['family']} at Store {result['store_nbr']} is {result['stockout_risk_simulated']}. "
            f"Recommended order is {result['recommended_order_qty']} units. Snapshot {data['data_snapshot_through']}; "
            f"origin {data['forecast_origin']}; forecast range {start} through {end}. "
            "This is historical data, not live sales. Inventory, lead time, shelf life, and MOQ are simulated assumptions. "
            "No order is placed."
        )

    model = ScriptedLLM(
        _tool("get_forecast", {"store_nbr": 3, "family": "BEVERAGES"}), forecast_answer,
        _tool("get_replenishment", {}), replenishment_answer,
        _answer("Recommended replenishment for BEVERAGES at Store 3: 63820 units."),
        replenishment_answer,
    )
    session = ChatSession(llm=model)
    forecast = session.ask("Give me the seven-day forecast for beverages at Store 3.")
    risk = session.ask("What is the stockout risk for beverages at Store 3?")
    replenish = session.ask("How much should I replenish?")

    assert forecast["status"] == "answered"
    assert forecast["tool_trace"][0]["tool"] == "get_forecast"
    assert risk["status"] == "answered"
    assert risk["tool_trace"][0]["tool"] == "get_replenishment"
    assert replenish["status"] == "answered"
    assert replenish["fallback_reason"] is None
    assert replenish["tool_trace"] == [{
        "tool": "get_replenishment",
        "args": {"store_nbr": 3, "family": "BEVERAGES"},
        "status": "success",
        "source_files": [
            "outputs/final/forecast.csv", "outputs/final/future_features.csv.gz",
            "outputs/final/model_metadata.json", "outputs/final/metrics.json",
        ],
    }]
    assert replenish["evidence"][0]["result"]["recommended_order_qty"] == 63820
    assert "63820" in replenish["text"]


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
    model = ScriptedLLM()
    session = ChatSession(llm=model)
    response = session.ask("How much should we order?")
    assert response["status"] == "clarification"
    assert response["text"] == "Which store number and product family should I use?"
    assert model.calls == 0


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

    result = ChatSession(llm=fail, max_retries=0).ask("What is the forecast for PRODUCE at Store 1?")
    assert result["status"] == "fallback"
    assert result["fallback_reason"] == "gateway_request_failed"
    assert result["text"].startswith("[Template fallback]")
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
        result = _tool_result(messages)
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
        payload = _tool_result(messages)
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


def test_markdown_emphasis_does_not_hide_mismatched_risk_level():
    result = call_tool("get_replenishment", {"store_nbr": 3, "family": "BEVERAGES"})
    evidence = [{"tool": "get_replenishment", "result": result}]
    assert not _answer_risks_supported("Stockout risk is **low**.", evidence)

def test_live_phrasings_are_accepted_but_fabrications_still_rejected():
    priority = call_tool("get_priority_replenishments", {"store_nbr": 3, "top_n": 3})
    evidence = [{"tool": "get_priority_replenishments", "result": priority}]
    listed = "Store 3 first: 1) BEVERAGES (high stockout risk, order 63820 units), 2) GROCERY I (order 56690 units)."
    assert _answer_numbers_supported(listed, evidence)
    assert _answer_risks_supported(listed, evidence)
    assert not _answer_risks_supported(listed.replace("high stockout", "low stockout"), evidence)
    assert not _answer_numbers_supported(listed.replace("63820", "63920"), evidence)

    what_if = call_tool("what_if_promotion", {"store_nbr": 44, "family": "DAIRY", "onpromotion": 0})
    evidence = [{"tool": "what_if_promotion", "result": what_if}]
    drop = f"Demand falls by {abs(what_if['change_units'])} units ({abs(what_if['change_pct'])}%)."
    assert _answer_numbers_supported(drop, evidence)
    assert not _answer_numbers_supported(drop.replace(str(abs(what_if["change_units"])), "999"), evidence)
    assert "p50 17332.0 – 17548.6".translate(ASCII_EQUIVALENTS).isascii()
