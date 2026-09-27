"""English-first replenishment decision assistant over the published v3 artifacts."""

from __future__ import annotations

import json
import operator
import os
import re
import time
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Annotated, Any, Callable, TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .config import FOOD_FAMILIES
from .forecast_tools import FAMILY_ALIASES, TOOL_SPECS, call_tool


MAX_TOOL_CALLS = 3
REQUEST_TIMEOUT_SECONDS = 45.0
MAX_RESPONSE_TOKENS = 512
MAX_RETRIES = 1
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?%?")
ISO_DATE_PATTERN = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
PROTOCOL_REMINDER = (
    "Return exactly one raw JSON object matching the required protocol. "
    "Do not use Markdown fences and do not add any text before or after the object."
)

SYSTEM_TEMPLATE = """You are FreshFlow, an English-language replenishment decision assistant. All sales, inventory, order-quantity, risk, and date facts must come from tool results in this turn or saved tool facts in this session. Never guess facts, invent live operations, or claim an order was placed.

Every response must be exactly one JSON object without Markdown, using one of these forms:
Tool request: {{"type":"tool","name":"registered_tool_name","args":{{}}}}
Clarification: {{"type":"clarify","text":"Ask for the required missing information in English."}}
Final answer: {{"type":"answer","text":"Answer in English."}}

Available tools and parameters:
{tool_specs}

Confirmed session context: {context}
Rules:
- For "why?", call explain_forecast using the current store and family. For "how much should we order?", call get_replenishment.
- Follow-ups such as "what about dairy?" retain the current store and change only the explicitly named family. Explicit values in the current message override older context.
- Ask a clarification in English when required information is missing; never guess. English family names are preferred. Clarify ambiguous categories such as meat.
- Explain tool and parameter-validation errors in English. Do not expose raw exception text or secrets; never present an error as a successful result.
- Forecasts come from archived data. State the returned data snapshot and forecast dates accurately; never describe them as current live sales.
- Keep final responses concise (normally under 80 words). Do not repeat the daily breakdown unless the user asks for it.
- Copy numeric values exactly from tool results; do not round, reformat, or calculate new quantities. State snapshot and forecast dates in the exact YYYY-MM-DD format returned by the tool.
- For a forecast summary, report only the returned p50, p10, and p90 fields and their exact dates. Do not describe p10-p90 as a confidence probability or add derived rankings unless requested.
- For forecast answers, explicitly say the data is historical and not live. For replenishment answers, explicitly say inventory, lead time, shelf life, and MOQ are simulated assumptions and that no order is placed.
- Every forecast-derived answer, including replenishment advice, must state the data snapshot cutoff, forecast origin, and full forecast date range exactly as returned by the tool.
- For replenishment risk, state the risk with its scope (for example, "Stockout risk for BEVERAGES at Store 3 is high") and include the exact snapshot date, forecast origin, and full forecast date range. Explicitly state "historical data, not live sales".
- Inventory, shelf life, lead time, and MOQ are simulated assumptions and must be disclosed in English. This Agent provides decision support only and never places orders.
- Never present a tool error as a successful result.
- Tool requests must use a registered tool name and a JSON object for args. Obtain tool facts before answering.
"""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], operator.add]
    current_text: str
    tool_count: int
    trace: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[dict[str, Any]], operator.add]
    llm_failed: bool
    clarification: bool


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return str(content)


def _parse_protocol(content: Any) -> dict[str, Any] | None:
    try:
        payload = json.loads(_content_text(content))
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("type") not in {"tool", "clarify", "answer"}:
        return None
    if payload["type"] == "tool":
        if not isinstance(payload.get("name"), str) or not isinstance(payload.get("args"), dict):
            return None
    elif not isinstance(payload.get("text"), str):
        return None
    return payload


def _explicit_entities(text: str) -> tuple[int | None, str | None, bool]:
    store = None
    match = re.search(r"(?:第\s*)?(\d+)\s*(?:号\s*)?(?:店|门店)", text)
    if not match:
        match = re.search(r"\bstore\s*(\d+)\b", text, re.IGNORECASE)
    if match:
        store = int(match.group(1))

    if "肉类" in text or re.search(r"(?<![\w])肉(?![\w])", text):
        return store, None, True
    for alias, family in sorted(FAMILY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in text:
            return store, family, False
    upper = text.upper()
    for family in sorted(FOOD_FAMILIES, key=len, reverse=True):
        if family in upper:
            return store, family, False
    return store, None, False


def _requires_replenishment_tool(text: str) -> bool:
    if re.search(r"\b(?:which|what)\s+(?:product\s+)?categories\b|\breplenish first\b", text, re.IGNORECASE):
        return False
    return bool(re.search(
        r"\b(?:replenish(?:ment|ing)?|stock[\s-]?out\s+risk|how much.{0,40}\b(?:order|replenish)|how many.{0,60}\b(?:order|replenish)|(?:what|which)\s+(?:is\s+)?(?:the\s+)?order\s+quantity|(?:what|which)\s+(?:is\s+)?(?:the\s+)?quantity.{0,60}\b(?:order|replenish)|(?:recommended|suggested)\s+order(?:\s+quantity)?)\b",
        text,
        re.IGNORECASE,
    ))


def _requires_forecast_tool(text: str) -> bool:
    if _requires_replenishment_tool(text) or re.search(
        r"\b(?:which|what)\s+(?:product\s+)?categories\b|\breplenish first\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return bool(re.search(r"\b(?:forecast(?:ing)?|predict(?:ion)?)\b", text, re.IGNORECASE))


def _required_series_tool(text: str) -> str | None:
    if _requires_replenishment_tool(text):
        return "get_replenishment"
    if _requires_forecast_tool(text):
        return "get_forecast"
    return None


def _normalize_number(value: str) -> str:
    value = value.replace(",", "").removesuffix("%")
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def _number_forms(value: str) -> set[str]:
    normalized = _normalize_number(value)
    forms = {normalized}
    try:
        number = Decimal(value.replace(",", "").removesuffix("%"))
    except InvalidOperation:
        return forms
    if number != number.to_integral_value():
        rounded = number.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        forms.add(_normalize_number(str(rounded)))
    return forms


def _numbers(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_numbers(item) for item in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_numbers(item) for item in value)) if value else set()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _number_forms(str(value))
    if isinstance(value, str):
        without_dates = ISO_DATE_PATTERN.sub(" ", value)
        tokens = NUMBER_PATTERN.findall(without_dates)
        return set().union(*(_number_forms(token) for token in tokens)) if tokens else set()
    return set()


def _derived_numeric_evidence(value: Any) -> set[str]:
    derived: set[str] = set()
    if isinstance(value, dict):
        date_range = value.get("forecast_date_range")
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            try:
                start, end = (date.fromisoformat(item) for item in date_range)
            except (TypeError, ValueError):
                pass
            else:
                days = (end - start).days + 1
                if days > 0:
                    derived.add(str(days))
        for item in value.values():
            derived.update(_derived_numeric_evidence(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            derived.update(_derived_numeric_evidence(item))
    return derived


def _answer_numbers_supported(text: str, evidence: list[dict[str, Any]]) -> bool:
    allowed = _numbers(evidence) | _derived_numeric_evidence(evidence)
    without_dates = ISO_DATE_PATTERN.sub(" ", text)
    return all(_normalize_number(token) in allowed for token in NUMBER_PATTERN.findall(without_dates))


def _evidence_dates(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_evidence_dates(item) for item in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_evidence_dates(item) for item in value)) if value else set()
    if isinstance(value, str):
        return set(re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", value))
    return set()


def _answer_risks_supported(text: str, evidence: list[dict[str, Any]]) -> bool:
    emphasis = r"[*_`]{0,2}"
    records: list[dict[str, Any]] = []
    priority_records: list[dict[str, Any]] = []
    for entry in evidence:
        result = entry.get("result", {})
        if not isinstance(result, dict):
            continue
        if entry.get("tool") == "get_priority_replenishments":
            items = result.get("items", [])
            if isinstance(items, list):
                priority_records.extend(item for item in items if isinstance(item, dict))
                records.extend(item for item in items if isinstance(item, dict))
        else:
            records.append(result)

    known_families = sorted(
        {record["family"] for record in records if isinstance(record.get("family"), str)},
        key=len,
        reverse=True,
    )
    family_pattern = "|".join(re.escape(family) for family in known_families) or r"(?!)"
    risk_pattern = re.compile(
        rf"\b(?P<kind>stock[\s-]?out|waste)\s+risk"
        rf"(?:\s+for\s+(?P<family>{family_pattern})(?:\s+at\s+Store\s+(?P<store>\d+))?)?"
        rf"(?:\s+is|\s*[:=])?\s*{emphasis}(?P<level>high|low){emphasis}\b",
        re.IGNORECASE,
    )
    claims = [
        (
            "stockout" if re.sub(r"[^a-z]", "", match.group("kind").lower()) == "stockout" else "waste",
            match.group("level").lower(),
            int(match.group("store")) if match.group("store") else None,
            match.group("family"),
        )
        for match in risk_pattern.finditer(text)
    ]
    aggregate_claims = [
        ("stockout" if re.sub(r"[^a-z]", "", risk_kind.lower()) == "stockout" else "waste", level.lower())
        for level, risk_kind in re.findall(
            rf"\b(?:all|both|each)\s+(?:show|have)\s+(high|low)\s+(stock[\s-]?out|waste)\s+risk\b",
            text,
            re.IGNORECASE,
        )
    ]
    aggregate_actions = re.findall(
        r"\b(?:all|both|each)\s+(?:show|have)\b[^.!?]{0,80}\b(order_today|monitor)\s+action\b",
        text,
        re.IGNORECASE,
    )
    if re.search(r"\b(?:stock[\s-]?out|waste)\s+risk\b", text, re.IGNORECASE) and not (claims or aggregate_claims):
        return False
    for kind, level, store_nbr, family in claims:
        matching = [
            record for record in records
            if (store_nbr is None or record.get("store_nbr") == store_nbr)
            and (family is None or record.get("family") == family)
            and isinstance(record.get("stockout_risk_simulated" if kind == "stockout" else "waste_risk_simulated"), str)
        ]
        if not matching or any(
            record.get("stockout_risk_simulated" if kind == "stockout" else "waste_risk_simulated") != level
            for record in matching
        ):
            return False
    if aggregate_claims:
        if not priority_records:
            return False
        for kind, level in aggregate_claims:
            field = "stockout_risk_simulated" if kind == "stockout" else "waste_risk_simulated"
            if any(record.get(field) != level for record in priority_records):
                return False
    if aggregate_actions and (
        not priority_records
        or any(record.get("action") != action.lower() for action in aggregate_actions for record in priority_records)
    ):
        return False
    return True


def _answer_facts_supported(text: str, evidence: list[dict[str, Any]]) -> bool:
    dates = re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", text)
    return (
        _answer_numbers_supported(text, evidence)
        and all(date in _evidence_dates(evidence) for date in dates)
        and _answer_risks_supported(text, evidence)
        and _required_disclosures_present(text, evidence)
    )


def _complete_evidence_disclosures(text: str, evidence: list[dict[str, Any]]) -> str:
    forecast_tools = {
        "get_forecast", "get_replenishment", "get_priority_replenishments",
        "get_forecast_overview", "explain_forecast", "what_if_promotion",
    }
    replenishment_tools = {"get_replenishment", "get_priority_replenishments"}
    lower_text = text.lower()
    tools = {item.get("tool") for item in evidence}

    if tools & forecast_tools and not any(
        term in lower_text
        for term in ("not live", "not current", "historical data snapshot", "archived data")
    ):
        text += " This is historical data from the published snapshot, not live sales or inventory."
        lower_text = text.lower()

    if tools & replenishment_tools:
        required_terms = ("inventory", "lead time", "shelf life", "moq", "simulated", "assumption")
        no_order_claim = re.search(
            r"\b(?:does not place (?:an? )?orders?|no orders? (?:is|are) placed)\b",
            lower_text,
        )
        if any(term not in lower_text for term in required_terms) or not no_order_claim:
            text += (
                " Inventory, lead time, shelf life, and MOQ are simulated assumptions. "
                "This Agent provides recommendations only and does not place orders."
            )
    return text


def _required_disclosures_present(text: str, evidence: list[dict[str, Any]]) -> bool:
    forecast_tools = {
        "get_forecast", "get_replenishment", "get_priority_replenishments",
        "get_forecast_overview", "explain_forecast", "what_if_promotion",
    }
    replenishment_tools = {"get_replenishment", "get_priority_replenishments"}
    lower_text = text.lower()
    for item in evidence:
        tool_name = item.get("tool")
        result = item.get("result", {})
        if tool_name in forecast_tools:
            data = result.get("data", {})
            dates = [data.get("data_snapshot_through"), *data.get("forecast_date_range", [])]
            if any(not date or date not in text for date in dates):
                return False
            historical_boundary = any(
                term in lower_text
                for term in ("not live", "not current", "historical data snapshot", "archived data")
            )
            if "historical" not in lower_text or not historical_boundary:
                return False
        if tool_name in replenishment_tools:
            required_terms = ("inventory", "lead time", "shelf life", "moq", "simulated", "assumption")
            if any(term not in lower_text for term in required_terms):
                return False
            if not re.search(r"\b(?:does not place (?:an? )?orders?|no orders? (?:is|are) placed)\b", lower_text):
                return False
    return True


def _fallback(tool_name: str | None, result: dict[str, Any] | None, reason: str | None = None) -> str:
    prefix = "[Template fallback] "
    if not result:
        detail = "No tool facts are available yet." if not reason else "Unable to retrieve tool data. Please try again later."
        return f"{prefix}{detail}"

    data = result.get("data", {})
    origin = data.get("forecast_origin", result.get("forecast_origin", "unknown"))
    date_range = data.get("forecast_date_range", [])
    date_text = f"forecast dates are {date_range[0]} through {date_range[-1]}" if len(date_range) == 2 else "the forecast date range is unknown"
    snapshot = data.get("data_snapshot_through", "unknown")
    note = f"The data snapshot ends {snapshot}, forecast origin is {origin}, and {date_text}. This is historical data, not live inventory or today's actual sales."
    if tool_name == "get_forecast":
        total = result.get("total", {})
        text = (f"Store {result['store_nbr']} {result['family']} forecast for {result.get('days', 7)} days: "
                f"p50 {total.get('p50', 'unknown')}; p10-p90 range {total.get('p10', 'unknown')} to {total.get('p90', 'unknown')}. ")
    elif tool_name == "get_replenishment":
        action = "order today" if result["action"] == "order_today" else "monitor"
        text = (f"Store {result['store_nbr']} {result['family']}: simulated action {action}; "
                f"recommended order {result['recommended_order_qty']} units; simulated stock {result['current_stock_simulated']}; "
                f"simulated reorder point {result['reorder_point_simulated']}; stockout risk {result['stockout_risk_simulated']}; "
                f"waste risk {result['waste_risk_simulated']}. ")
        note += (f"Inventory, lead time ({result['lead_time_days_simulated']} days), shelf life "
                 f"({result['shelf_life_days_simulated']} days), and MOQ ({result['moq_simulated']}) are simulated assumptions.")
    elif tool_name == "get_priority_replenishments":
        parts = [f"Store {item['store_nbr']} {item['family']}: {item['action']}, order {item['recommended_order_qty']} units"
                 for item in result.get("items", [])]
        text = "Priority items: " + ("; ".join(parts) if parts else "no matching items.") + " "
        note += " Inventory, lead time, shelf life, and MOQ are simulated assumptions."
    elif tool_name == "explain_forecast":
        drivers = result.get("top_drivers", [])
        text = f"Model explanation for Store {result.get('store_nbr')} {result.get('family')}: "
        if drivers:
            text += "; ".join(f"{item['meaning']}: {item.get('effect_pct', item.get('effect_units_7d'))}" for item in drivers)
        else:
            text += result.get("explanation", "The tool returned no explainable features.")
        note = f"{note} This is a v3 model feature-contribution explanation, not a causal claim."
    else:
        text = "Tool results are available in the evidence fields. "
    suffix = " This Agent provides recommendations only and does not place orders." if "replenishment" in (tool_name or "") else ""
    return f"{prefix}{text} {note}{suffix}"


class ChatSession:
    """One in-memory conversation. No state is written when the CLI exits."""

    def __init__(
        self,
        llm: Any | Callable[[list[dict[str, Any]]], str] | None = None,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_retries: int = MAX_RETRIES,
        retry_delay: float = 0.25,
    ) -> None:
        self.llm = llm
        self.max_tool_calls = max_tool_calls
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.messages: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}
        self.tool_trace: list[dict[str, Any]] = []
        self.latest_tool_name: str | None = None
        self.latest_result: dict[str, Any] | None = None
        self.last_gateway_error_type: str | None = None
        self._tool_calls_this_turn = 0
        self.graph = self._build_graph()

    def _gateway_llm(self) -> ChatOllama:
        if self.llm is not None:
            return self.llm
        load_dotenv()
        url = os.getenv("LLM_GATEWAY_URL")
        key = os.getenv("LLM_GATEWAY_API_KEY")
        model = os.getenv("LLM_MODEL")
        if not all((url, key, model)):
            raise RuntimeError("LLM gateway configuration is incomplete; set LLM_GATEWAY_URL, LLM_GATEWAY_API_KEY and LLM_MODEL")
        return ChatOllama(
            model=model,
            base_url=url,
            temperature=0,
            num_predict=MAX_RESPONSE_TOKENS,
            format="json",
            client_kwargs={"headers": {"X-API-Key": key}, "timeout": REQUEST_TIMEOUT_SECONDS},
        )

    def _invoke_llm(self, messages: list[dict[str, Any]]) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                model = self._gateway_llm()
                reply = model(messages) if callable(model) and not hasattr(model, "invoke") else model.invoke(messages)
                return _content_text(getattr(reply, "content", reply))
            except Exception as error:
                self.last_gateway_error_type = type(error).__name__
                if attempt >= self.max_retries:
                    raise RuntimeError("LLM gateway request failed or timed out") from None
                time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError("LLM gateway request failed")

    def _build_graph(self):
        def ask_model(state: AgentState) -> dict[str, Any]:
            try:
                text = self._invoke_llm(state["messages"])
                return {"messages": [{"role": "assistant", "content": text}]}
            except RuntimeError:
                return {"messages": [{"role": "assistant", "content": ""}], "llm_failed": True}

        def route_after_model(state: AgentState) -> str:
            protocol = _parse_protocol(state["messages"][-1].get("content", ""))
            required_tool = _required_series_tool(state.get("current_text", ""))
            if self._tool_calls_this_turn == 0 and required_tool:
                if protocol and protocol["type"] == "clarify":
                    return END
                if protocol and protocol["type"] == "tool" and protocol["name"] == required_tool:
                    return "execute_tool"
                if self._tool_calls_this_turn >= self.max_tool_calls:
                    return END
                return "execute_tool"
            if not protocol or protocol["type"] != "tool" or protocol["name"] not in self._registered_tools():
                return END
            if self._tool_calls_this_turn >= self.max_tool_calls:
                return END
            return "execute_tool"

        def execute_tool(state: AgentState) -> dict[str, Any]:
            request = _parse_protocol(state["messages"][-1]["content"])
            current_text = state.get("current_text", "")
            required_tool = _required_series_tool(current_text)
            forced_replenishment = (
                self._tool_calls_this_turn == 0
                and required_tool is not None
                and (not request or request["type"] != "tool" or request["name"] != required_tool)
            )
            tool_name = required_tool if forced_replenishment else request["name"]
            args = {} if forced_replenishment else dict(request["args"])
            explicit_store, explicit_family, ambiguous_family = _explicit_entities(current_text)
            clarification = None
            if ambiguous_family:
                clarification = "Did you mean MEATS or POULTRY?"
            elif tool_name in {"get_forecast", "get_replenishment", "explain_forecast"}:
                store_nbr = explicit_store if explicit_store is not None else self.context.get("store_nbr")
                family = explicit_family if explicit_family is not None else self.context.get("family")
                missing = []
                if store_nbr is None:
                    missing.append("store number")
                if family is None:
                    missing.append("product family")
                if missing:
                    clarification = "Which " + " and ".join(missing) + " should I use?"
                else:
                    args["store_nbr"] = store_nbr
                    args["family"] = family
            elif tool_name in {"get_priority_replenishments", "get_forecast_overview"}:
                store_nbr = explicit_store if explicit_store is not None else self.context.get("store_nbr")
                if store_nbr is None:
                    args.pop("store_nbr", None)
                else:
                    args["store_nbr"] = store_nbr
            elif tool_name == "get_model_reliability":
                family = explicit_family if explicit_family is not None else self.context.get("family")
                if family is None:
                    args.pop("family", None)
                else:
                    args["family"] = family

            if clarification:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": json.dumps({"type": "clarify", "text": clarification}),
                    }],
                    "clarification": True,
                }

            self._tool_calls_this_turn += 1
            result = call_tool(tool_name, args)

            success = isinstance(result, dict) and "error" not in result
            data = result.get("data", {}) if success else {}
            source_files = data.get("source_files", []) if isinstance(data, dict) else []
            trace_entry = {
                "tool": tool_name,
                "args": args,
                "status": "success" if success else "error",
                "source_files": source_files,
            }
            if success:
                self.latest_tool_name = tool_name
                self.latest_result = result
                self._update_context(tool_name, result)
                if explicit_store is not None:
                    self.context["store_nbr"] = explicit_store
                if explicit_family is not None:
                    self.context["family"] = explicit_family
            return {
                "tool_count": state.get("tool_count", 0) + 1,
                "trace": [trace_entry],
                "evidence": ([{"tool": tool_name, "result": result}] if success else []),
                    "messages": [{
                        "role": "user",
                        "content": "TOOL_RESULT_JSON " + json.dumps(result, ensure_ascii=False, default=str)
                        + "\n\n" + PROTOCOL_REMINDER,
                    }],
            }

        def route_after_tool(state: AgentState) -> str:
            return END if state.get("clarification") else "ask_model"

        builder = StateGraph(AgentState)
        builder.add_node("ask_model", ask_model)
        builder.add_node("execute_tool", execute_tool)
        builder.add_edge(START, "ask_model")
        builder.add_conditional_edges("ask_model", route_after_model, {"execute_tool": "execute_tool", END: END})
        builder.add_conditional_edges("execute_tool", route_after_tool, {"ask_model": "ask_model", END: END})
        return builder.compile()

    @staticmethod
    def _registered_tools() -> set[str]:
        return {spec["name"] for spec in TOOL_SPECS}

    def _update_context(self, tool_name: str, result: dict[str, Any]) -> None:
        if isinstance(result.get("store_nbr"), int):
            self.context["store_nbr"] = result["store_nbr"]
        if isinstance(result.get("family"), str):
            self.context["family"] = result["family"]
        data = result.get("data", {})
        if data:
            self.context.update({
                "data_snapshot_through": data.get("data_snapshot_through"),
                "date_range": data.get("forecast_date_range"),
                "model_id": data.get("model_id"),
            })
        self.context["last_tool"] = tool_name

    def ask(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            return {"text": "Please provide a store or product family to look up.", "status": "clarification", "tool_trace": [], "evidence": []}
        required_tool = _required_series_tool(text)
        explicit_store, explicit_family, ambiguous_family = _explicit_entities(text)
        if required_tool:
            if ambiguous_family:
                missing_text = "Did you mean MEATS or POULTRY?"
            else:
                store_nbr = explicit_store if explicit_store is not None else self.context.get("store_nbr")
                family = explicit_family if explicit_family is not None else self.context.get("family")
                missing = []
                if store_nbr is None:
                    missing.append("store number")
                if family is None:
                    missing.append("product family")
                missing_text = "Which " + " and ".join(missing) + " should I use?" if missing else None
            if missing_text:
                return {
                    "text": missing_text,
                    "status": "clarification",
                    "tool_trace": [],
                    "evidence": [],
                    "context": dict(self.context),
                    "fallback_reason": None,
                }
        self.latest_tool_name = None
        self.latest_result = None
        self.last_gateway_error_type = None
        self._tool_calls_this_turn = 0
        system = SYSTEM_TEMPLATE.format(
            tool_specs=json.dumps(TOOL_SPECS, ensure_ascii=False),
            context=json.dumps(self.context, ensure_ascii=False),
        )
        initial_messages = [
            {"role": "system", "content": system},
            *self.messages,
            {"role": "user", "content": f"{PROTOCOL_REMINDER}\n\nUser request: {text}"},
        ]
        try:
            state = self.graph.invoke(
                {"messages": initial_messages, "tool_count": 0, "trace": [], "evidence": [], "current_text": text},
                config={"recursion_limit": 2 * self.max_tool_calls + 3},
            )
        except Exception:
            state = {"messages": initial_messages, "trace": [], "evidence": [], "llm_failed": True}

        last = state.get("messages", [])[-1].get("content", "") if state.get("messages") else ""
        protocol = _parse_protocol(last)
        evidence = state.get("evidence", [])
        trace = state.get("trace", [])
        candidate_answer = (
            _complete_evidence_disclosures(protocol["text"], evidence)
            if protocol and protocol["type"] == "answer"
            else ""
        )
        if protocol and protocol["type"] == "answer" and candidate_answer.isascii() and _answer_facts_supported(candidate_answer, evidence):
            answer = candidate_answer
            status = "answered"
        elif protocol and protocol["type"] == "clarify" and protocol["text"].isascii() and _answer_facts_supported(protocol["text"], evidence):
            answer = protocol["text"]
            status = "clarification"
        else:
            if state.get("llm_failed"):
                reason = "gateway_timeout" if self.last_gateway_error_type and "timeout" in self.last_gateway_error_type.lower() else "gateway_request_failed"
            elif protocol and protocol.get("type") == "answer":
                reason = "unsupported_numeric_claim"
            else:
                reason = "invalid_protocol"
            answer = _fallback(self.latest_tool_name, self.latest_result, reason)
            status = "fallback"
        self.messages = state.get("messages", [])[1:]
        self.tool_trace.extend(trace)
        return {
            "text": answer,
            "status": status,
            "tool_trace": trace,
            "evidence": evidence,
            "context": dict(self.context),
            "fallback_reason": reason if status == "fallback" else None,
        }

    def clear(self) -> None:
        self.messages.clear()
        self.context.clear()
        self.tool_trace.clear()
        self.latest_tool_name = None
        self.latest_result = None
        self.last_gateway_error_type = None


def main() -> None:
    session = ChatSession()
    print("FreshFlow Replenishment Assistant (session is in memory; /clear resets it, /exit quits)")
    while True:
        try:
            text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            session.clear()
            print("\nSession ended and cleared.")
            break
        if text == "/exit":
            session.clear()
            print("Session cleared.")
            break
        if text == "/clear":
            session.clear()
            print("Session cleared.")
            continue
        if not text:
            continue
        response = session.ask(text)
        print(f"FreshFlow [{response['status']}]> {response['text']}")
        if response["fallback_reason"]:
            print("[fallback_reason] " + response["fallback_reason"])
        for item in response["tool_trace"]:
            print("[tool] " + json.dumps(item, ensure_ascii=True))


if __name__ == "__main__":
    main()