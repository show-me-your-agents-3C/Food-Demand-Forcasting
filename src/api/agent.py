"""LangGraph demand-planning agent with a deterministic offline fallback."""

from __future__ import annotations

import json
import operator
import os
import re
import time
from typing import Annotated, Any, Dict, List, TypedDict

from src.api import service
from src.api.tools import TOOL_REGISTRY, TOOL_SPECS


SYSTEM_PROMPT = """You are the FreshFlow demand-planning agent for a food distributor.

Rules:
- Base every number only on tool results. Never invent demand, stock, cost or supplier facts.
- If a store or family is missing, call list_scope or get_replenishment_plan first.
- Keep answers short: the recommended action, the key numbers, and any risk.
- Always state that inventory, shelf-life, lead-time and MOQ are simulated scenarios.
"""

TOOL_CALL_INSTRUCTIONS = """When you need data, reply with ONLY this JSON and nothing else:
{"tool": "<tool_name>", "args": {<arguments>}}

Available tools and arguments:
"""

class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    trace: Annotated[List[dict], operator.add]


FAMILY_ALIASES = {
    "BEVERAGES": ["beverage", "drink", "饮料", "饮品"],
    "BREAD/BAKERY": ["bread", "bakery", "面包", "烘焙"],
    "DAIRY": ["dairy", "milk", "乳", "牛奶", "奶"],
    "MEATS": ["meat", "肉"],
    "POULTRY": ["poultry", "chicken", "禽", "鸡"],
    "PRODUCE": ["produce", "vegetable", "fruit", "蔬菜", "水果", "生鲜"],
    "SEAFOOD": ["seafood", "fish", "海鲜", "鱼"],
    "FROZEN FOODS": ["frozen", "冷冻"],
}


def llm_configured() -> bool:
    return all(os.getenv(key) for key in ("LLM_GATEWAY_URL", "LLM_GATEWAY_API_KEY", "LLM_MODEL"))


def _tool_specs_text() -> str:
    return json.dumps(TOOL_SPECS, indent=2)


def _extract_tool_call(text: str):
    match = re.search(r'\{.*"tool".*\}', text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if payload.get("tool") and isinstance(payload.get("args", {}), dict):
        return payload
    return None


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")
    return str(content)


def _invoke_with_retry(llm, messages, max_retries: int = 3, backoff: float = 2.0):
    error = None
    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            error = exc
            if attempt < max_retries:
                time.sleep(backoff * attempt)
    raise error


def _extract_store(message: str, stores: list[int]) -> int | None:
    match = re.search(r"(?:store|shop|店|门店)\s*#?\s*(\d+)", message, re.IGNORECASE)
    if match and int(match.group(1)) in stores:
        return int(match.group(1))
    hits = [int(number) for number in re.findall(r"\b(\d+)\b", message) if int(number) in stores]
    if len(set(hits)) == 1:
        return hits[0]
    return None


def _extract_family(message: str, families: list[str]) -> str | None:
    upper = message.upper()
    for family in families:
        if family.upper() in upper:
            return family
    lower = message.lower()
    for family, aliases in FAMILY_ALIASES.items():
        if family in families and any(alias.lower() in lower for alias in aliases):
            return family
    return None


def _extract_uplift(message: str, default: float = 20.0) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", message)
    if match:
        return float(match.group(1))
    match = re.search(r"(?:uplift|increase|提升|涨|增加)\D{0,8}(\d+(?:\.\d+)?)", message, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default


def _top_target(store: int | None, family: str | None) -> tuple[int | None, str | None]:
    if store is not None and family is not None:
        return store, family
    rows = service.get_replenishment(limit=1)
    if not rows:
        return store, family
    top = rows[0]
    return (store if store is not None else int(top["store_nbr"]), family if family is not None else top["family"])


def _format_actions(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"- Store {row['store_nbr']} {row['family']}: {row['action']} -> order "
            f"{int(row['recommended_order_qty'])} units (7d forecast {row['forecast_7d']:.0f}, "
            f"stock {row['current_stock_simulated']:.0f}, reorder point {row['reorder_point']:.0f}, "
            f"lead time {row['lead_time_days']}d, stockout risk {row['stockout_risk']}, "
            f"waste risk {row['waste_risk']})"
        )
    return "\n".join(lines)


def _format_forecast(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"- {row['date']} store {row['store_nbr']} {row['family']}: {row['forecast_sales']:.0f} units "
            f"(range {row['lower_bound']:.0f}-{row['upper_bound']:.0f})"
        )
    return "\n".join(lines)


def run_fallback(message: str, reason: str) -> dict:
    scope = service.get_scope()
    stores, families = scope["stores"], scope["families"]
    store = _extract_store(message, stores)
    family = _extract_family(message, families)
    text = message.lower()
    trace: list[dict] = []
    DEFAULT_NOTE = "Inventory, shelf-life, lead-time and MOQ are simulated scenarios."

    def call(name: str, **args) -> dict:
        result = TOOL_REGISTRY[name](**args)
        trace.append({"tool": name, "args": args, "result": result})
        return result

    if any(word in text for word in ["uplift", "promotion", "promo", "what if", "what-if", "促销", "打折", "涨价", "提升需求"]):
        target_store, target_family = _top_target(store, family)
        uplift = _extract_uplift(message)
        result = call("simulate_promotion", store_nbr=int(target_store), family=target_family, uplift_pct=uplift)
        sim = result["simulation"]
        reply = (
            f"Promotion simulation for store {sim['store_nbr']} {sim['family']} at +{sim['uplift_pct']:.0f}%:\n"
            f"- 7-day demand: {sim['baseline_forecast_7d']:.0f} -> {sim['simulated_forecast_7d']:.0f} "
            f"(+{sim['extra_units_7d']:.0f} units)\n"
            f"- Recommended order: {sim['baseline_order_qty']} -> {sim['simulated_order_qty']} units "
            f"(delta {sim['order_delta']})\n"
            f"- Waste risk: {sim['baseline_waste_risk']} -> {sim['simulated_waste_risk']}\n"
            f"{DEFAULT_NOTE}"
        )
    elif any(word in text for word in ["metric", "mae", "wape", "accuracy", "backtest", "指标", "准确", "误差"]):
        result = call("get_forecast_metrics")
        metrics = result["metrics"]
        wape = metrics.get("wape")
        reply = (
            f"Backtest on the seasonal-naive baseline: MAE {metrics.get('mae')}, WAPE {wape}, "
            f"{metrics.get('n_predictions')} predictions."
            + (f" WAPE means about {wape * 100:.1f}% of unit demand is missed on average." if wape is not None else "")
        )
    elif any(word in text for word in ["scope", "coverage", "which store", "范围", "覆盖", "哪些门店", "有哪些"]):
        result = call("list_scope")
        reply = (
            f"Stores covered: {result['scope']['stores']}\n"
            f"Food families covered: {', '.join(result['scope']['families'])}"
        )
    elif any(word in text for word in ["inventory", "on hand", "库存", "存货"]):
        target_store, target_family = _top_target(store, family)
        result = call("get_inventory_status", store_nbr=int(target_store), family=target_family)
        inv = result["inventory"]
        reply = (
            f"Store {inv['store_nbr']} {inv['family']}: on hand {inv['current_stock_simulated']:.0f}, "
            f"reorder point {inv['reorder_point']:.0f}, safety stock {inv['safety_stock']:.0f}, "
            f"lead time {inv['lead_time_days']}d, shelf life {inv['shelf_life_days']}d, MOQ {inv['moq']}. "
            f"Stockout risk {inv['stockout_risk']}, waste risk {inv['waste_risk']}.\n{DEFAULT_NOTE}"
        )
    elif any(word in text for word in ["risk", "stockout", "waste", "shortage", "expire", "风险", "缺货", "报废", "过期"]):
        if any(word in text for word in ["today", "今天", "今日", "order today"]):
            result = call("get_replenishment_plan", action="order_today", limit=10)
        else:
            result = call("get_replenishment_plan", limit=10)
        reply = "Highest-risk replenishment actions:\n" + _format_actions(result["actions"]) + f"\n{DEFAULT_NOTE}"
    elif any(word in text for word in ["reorder", "order", "replenish", "补货", "订货", "采购", "下单"]):
        if any(word in text for word in ["today", "今天", "今日", "order today"]):
            result = call("get_replenishment_plan", action="order_today", limit=10)
        elif store is not None or family is not None:
            result = call("get_replenishment_plan", store_nbr=store, family=family, limit=10)
        else:
            result = call("get_replenishment_plan", limit=10)
        reply = "Recommended replenishment actions:\n" + _format_actions(result["actions"]) + f"\n{DEFAULT_NOTE}"
    elif any(word in text for word in ["forecast", "demand", "predict", "sales", "预测", "需求", "销量"]):
        target_store, target_family = _top_target(store, family)
        result = call("get_demand_forecast", store_nbr=int(target_store), family=target_family, horizon_days=7)
        reply = (
            f"7-day demand forecast for store {target_store} {target_family}:\n"
            + _format_forecast(result["forecast"])
            + f"\n{DEFAULT_NOTE}"
        )
    else:
        result = call("get_dashboard_summary")
        summary = result["summary"]
        reply = (
            f"Scope: {len(summary['scope']['stores'])} stores x {len(summary['scope']['families'])} food families, "
            f"{summary['forecast_rows']} forecast rows.\n"
            f"Actions due today: {summary['order_today_count']}. High waste risk: {summary['high_waste_risk_count']}.\n"
            "Top priorities:\n" + _format_actions(summary["top_actions"]) + f"\n{DEFAULT_NOTE}"
        )
    return {"reply": reply, "tool_trace": trace, "mode": "fallback", "fallback_reason": reason}


def _run_llm(message: str, history: list[dict], max_iterations: int = 4) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, StateGraph

    llm = ChatOllama(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_GATEWAY_URL"],
        temperature=0.2,
        num_predict=1500,
        client_kwargs={"headers": {"X-API-Key": os.environ["LLM_GATEWAY_API_KEY"]}},
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in history:
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=turn.get("content", "")))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=turn.get("content", "")))
    messages.append(HumanMessage(content=TOOL_CALL_INSTRUCTIONS + _tool_specs_text() + "\n\nUser question: " + message))

    def chatbot(state: AgentState):
        return {"messages": [_invoke_with_retry(llm, state["messages"])]}

    def route(state: AgentState) -> Any:
        return "tools" if _extract_tool_call(_message_text(state["messages"][-1])) else END

    def call_tool(state: AgentState):
        request = _extract_tool_call(_message_text(state["messages"][-1]))
        name = request["tool"]
        args = request.get("args", {})
        function = TOOL_REGISTRY.get(name)
        if function is None:
            result: Dict[str, Any] = {"error": f"unknown tool '{name}'"}
        else:
            try:
                result = function(**args)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "messages": [HumanMessage(content="Tool result: " + json.dumps(result, default=str) + "\nNow answer the user using only these facts.")],
            "trace": [{"tool": name, "args": args, "result": result}],
        }

    builder = StateGraph(AgentState)
    builder.add_node("chatbot", chatbot)
    builder.add_node("tools", call_tool)
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "chatbot")
    graph = builder.compile()

    final = graph.invoke(
        {"messages": messages, "trace": []},
        config={"recursion_limit": 2 * max_iterations + 1},
    )
    return {"reply": _message_text(final["messages"][-1]), "tool_trace": final.get("trace", []), "mode": "llm"}


def run_agent(message: str, history: list[dict] | None = None) -> dict:
    history = history or []
    if not llm_configured():
        return run_fallback(message, reason="LLM gateway not configured")
    try:
        return _run_llm(message, history)
    except Exception as exc:
        return run_fallback(message, reason=f"LLM unavailable ({type(exc).__name__})")
