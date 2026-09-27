"""FastAPI contract for the FreshFlow demand-planning agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    data_available: bool
    data_detail: str
    llm_configured: bool


class ScopeResponse(BaseModel):
    stores: list[int]
    families: list[str]


class MetricsResponse(BaseModel):
    mae: float | None = None
    wape: float | None = None
    n_predictions: int | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    session_id: str | None = Field(default=None, max_length=64)
    store_nbr: int | None = None
    family: str | None = None


class ToolTraceEntry(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    mode: str
    fallback_reason: str | None = None
