"""FastAPI application exposing the FreshFlow demand-planning agent."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from src.api import agent, service
from src.food_forecast.chat_agent import ChatSession
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse, MetricsResponse, ScopeResponse

app = FastAPI(
    title="FreshFlow Demand Planning API",
    description="Forecast, replenishment and agent chat over precomputed Favorita food-sales artifacts.",
    version="0.1.0",
)

_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        scope = service.get_scope()
        service.get_forecast(limit=1)
        service.get_metrics()
        service.get_summary()
        service.get_explanations()
        data_available = True
        data_detail = f"{len(scope['stores'])} stores, {len(scope['families'])} families"
    except Exception as exc:
        data_available = False
        data_detail = f"{type(exc).__name__}: {exc}"
    return HealthResponse(
        status="ok",
        version=app.version,
        data_available=data_available,
        data_detail=data_detail,
        llm_configured=agent.llm_configured(),
    )


@app.get("/api/scope", response_model=ScopeResponse)
def scope() -> ScopeResponse:
    return ScopeResponse(**service.get_scope())


@app.get("/api/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(**service.get_metrics())


@app.get("/api/summary")
def summary() -> dict:
    return service.get_summary()


@app.get("/api/forecast")
def forecast(
    store_nbr: int | None = Query(default=None),
    family: str | None = Query(default=None),
    horizon_days: int = Query(default=7, ge=1, le=30),
    limit: int | None = Query(default=None, ge=1),
) -> list[dict]:
    return service.get_forecast(store_nbr=store_nbr, family=family, horizon_days=horizon_days, limit=limit)


@app.get("/api/replenishment")
def replenishment(
    store_nbr: int | None = Query(default=None),
    family: str | None = Query(default=None),
    action: str | None = Query(default=None, pattern="^(order_today|monitor)$"),
    limit: int | None = Query(default=None, ge=1),
) -> list[dict]:
    return service.get_replenishment(store_nbr=store_nbr, family=family, action=action, limit=limit)


@app.get("/api/explanations")
def explanations(
    store_nbr: int | None = Query(default=None),
    family: str | None = Query(default=None),
) -> list[dict]:
    return service.get_explanations(store_nbr=store_nbr, family=family)


_sessions: "OrderedDict[str, ChatSession]" = OrderedDict()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # v3 chat agent (src/food_forecast/chat_agent.py); sessions live in memory only.
    session = _sessions.pop(request.session_id, None) if request.session_id else None
    if session is None:
        session = ChatSession()
        if request.store_nbr is not None:
            session.context["store_nbr"] = request.store_nbr
        if request.family:
            session.context["family"] = request.family
    if request.session_id:
        _sessions[request.session_id] = session
        while len(_sessions) > 200:
            _sessions.popitem(last=False)
    result = session.ask(request.message)
    evidence = iter(result["evidence"])
    trace = []
    for entry in result["tool_trace"]:
        facts = next(evidence)["result"] if entry["status"] == "success" else {"error": "tool call failed"}
        trace.append({"tool": entry["tool"], "args": entry["args"],
                      "result": {"source": ", ".join(entry["source_files"]), **facts}})
    return ChatResponse(
        reply=result["text"], tool_trace=trace,
        mode="llm" if result["status"] in ("answered", "clarification") else "fallback",
        fallback_reason=result["fallback_reason"],
    )


_frontend = Path(__file__).resolve().parents[2] / "frontend"
if (_frontend / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_frontend / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(_frontend / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
