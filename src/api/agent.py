"""LangGraph demand-planning agent with a deterministic offline fallback."""

from __future__ import annotations

import json
import operator
import os
import re
import time
from typing import Annotated, Any, Dict, List, TypedDict

from src.api.fallback import run_fallback
from src.api.tools import TOOL_REGISTRY, TOOL_SPECS


SYSTEM_PROMPT = """You are the FreshFlow demand-planning agent for a food distributor.

Rules:
- Base every number only on tool results. Never invent demand, stock, cost or supplier facts.
- If a store or family is missing, call list_scope or get_replenishment_plan first.
- Keep answers short: the recommended action, the key numbers, and any risk.
- Respond in the user's language. Keep their store/family scope. Use source-data sales units, not bottles or boxes.
- Dates are a historical replay. Demand uplift is an assumption, not a learned discount effect.
- Always state that inventory, shelf-life, lead-time and MOQ are simulated scenarios.
"""

TOOL_CALL_INSTRUCTIONS = """When you need data, reply with ONLY this JSON and nothing else:
{"tool": "<tool_name>", "args": {<arguments>}}

Available tools and arguments:
"""

class AgentState(TypedDict):
    messages: Annotated[List[Any], operator.add]
    trace: Annotated[List[dict], operator.add]



def llm_configured() -> bool:
    values = [os.getenv(key, "") for key in ("LLM_GATEWAY_URL", "LLM_GATEWAY_API_KEY", "LLM_MODEL")]
    return all(value and "replace-with-" not in value for value in values)


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


def _run_llm(message: str, history: list[dict], max_iterations: int = 4) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, StateGraph

    llm = ChatOllama(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_GATEWAY_URL"],
        temperature=0.2,
        num_predict=1500,
        client_kwargs={"headers": {"X-API-Key": os.environ["LLM_GATEWAY_API_KEY"]}, "timeout": 12.0},
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
        return run_fallback(message, reason="LLM gateway not configured", history=history)
    try:
        return _run_llm(message, history)
    except Exception as exc:
        return run_fallback(message, reason=f"LLM unavailable ({type(exc).__name__})", history=history)
