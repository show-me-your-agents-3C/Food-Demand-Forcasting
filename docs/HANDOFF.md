# Handoff — Backend, Agent & Deployment (Person A)

- **Branch:** `feat/backend-agent-api`
- **Owner:** Person A (backend / agent / deployment)
- **Status:** complete and verified locally (21 tests passing)
- **Next owner:** Person B (frontend + demo polish)
- **Date:** 2026-09-20

This document describes exactly what Person A built, how to run it, how it is
deployed, and what Person B should pick up.

---

## 1. Objective

Turn the existing Favorita forecast pipeline into a web-accessible
**AI Food Distribution Demand Planning Agent**:

```
historical sales + promo/holiday/calendar
        -> offline forecast + replenishment plan (already existed)
        -> REST API over the artifacts
        -> LangGraph agent that answers planning questions from tool output
        -> deploy on Lightsail, reachable over HTTPS
```

Core rule: **forecasting runs offline on a laptop and the results are committed
to `outputs/`; the server only reads artifacts and runs the agent.** No training
and no 500 MB of raw data on the server.

---

## 2. What was delivered (file map)

| Area | Files |
| --- | --- |
| Repo hygiene / config | `src/food_forecast/config.py`, `requirements.txt`, `.env.example`, `.gitignore`, `.gitattributes` |
| API contract | `src/api/main.py`, `src/api/schemas.py` |
| Data service | `src/api/service.py` |
| Agent tools | `src/api/tools.py` |
| Agent (LLM + fallback) | `src/api/agent.py` |
| Tests | `pytest.ini`, `tests/test_service.py`, `tests/test_tools.py`, `tests/test_agent.py`, `tests/test_agent_graph.py`, `tests/test_api.py` |
| Deployment | `deploy/foodflow.service`, `deploy/Caddyfile`, `deploy/bootstrap.sh`, `deploy/deploy.sh` |
| Docs | this file, `README.md` updates |

---

## 3. P0 — Repo hygiene and portability

### 3.1 Removed the hardcoded path

`src/food_forecast/config.py` previously hardcoded `/home/ubuntu/demand-forecasting`,
so nothing ran on any other machine. It now resolves relative to the repo and
supports env overrides:

```python
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("FOOD_FORECAST_ROOT") or Path(__file__).resolve().parents[2])
DATA_PATH = Path(os.environ.get("FOOD_FORECAST_DATA_PATH") or PROJECT_ROOT / "data" / "favorita" / "train.csv")
OUTPUT_DIR = Path(os.environ.get("FOOD_FORECAST_OUTPUT_DIR") or PROJECT_ROOT / "outputs")
```

### 3.2 Dependencies

`requirements.txt` pins the runtime (server target: Python 3.10–3.12, e.g.
Ubuntu 22.04/24.04):

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
python-dotenv==1.0.1
pandas==2.2.3
numpy==2.2.1
langgraph==0.2.62
langchain-core==0.3.29
langchain-ollama==0.2.2
httpx==0.28.1
pytest==8.3.4
```

### 3.3 Secrets

`.env.example` documents every required variable. Real `.env` is git-ignored and
lives only on the server.

```
LLM_GATEWAY_URL=...
LLM_GATEWAY_API_KEY=...
LLM_MODEL=...
APP_HOST=127.0.0.1
APP_PORT=8000
CORS_ORIGINS=*
FOOD_FORECAST_ROOT=
FOOD_FORECAST_OUTPUT_DIR=
```

`.gitignore` now also ignores `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`,
`*.egg-info/`.

### 3.4 Git LFS + line-ending root fix

The repo had two phantom-diff problems that blocked clean handoff:

1. Five large CSVs are Git LFS objects. On a machine without `git-lfs` they show
   as modified (pointer vs full file). `git-lfs` was installed locally and
   verified: `git lfs fetch --all` found all 5 objects and `git lfs fsck` is OK.
2. 41 text files were CRLF in the working tree while HEAD stores LF, producing
   millions of fake changed lines. `.gitattributes` now enforces
   `* text=auto eol=lf` and the working tree was renormalized to LF, so a fresh
   clone on any OS is stable.

> `.gitattributes` is currently **staged but not committed** on this branch
> (see section 12).

---

## 4. P1 — API contract

`src/api/main.py` exposes a FastAPI app. Response models live in
`src/api/schemas.py`. Interactive OpenAPI docs are served at `/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | liveness + whether artifacts and the LLM gateway are available |
| GET | `/api/scope` | stores and families in the current run |
| GET | `/api/metrics` | backtest MAE / WAPE / n_predictions |
| GET | `/api/summary` | dashboard KPIs and top actions |
| GET | `/api/forecast` | daily forecast rows (`store_nbr`, `family`, `horizon_days`, `limit`) |
| GET | `/api/replenishment` | plan rows (`store_nbr`, `family`, `action`, `limit`) |
| GET | `/api/explanations` | plain-language explanations per action |
| POST | `/api/chat` | agent chat with tool trace |

Example:

```bash
curl -s http://127.0.0.1:8000/health
curl -s "http://127.0.0.1:8000/api/forecast?store_nbr=1&family=DAIRY&horizon_days=3"
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What should I reorder today?"}'
```

Chat response shape:

```json
{
  "reply": "Recommended replenishment actions: ...",
  "tool_trace": [{"tool": "get_replenishment_plan", "args": {"limit": 10}, "result": {"...": "..."}}],
  "mode": "fallback",
  "fallback_reason": "LLM gateway not configured"
}
```

CORS is configurable with `CORS_ORIGINS` (default `*`). In production the
frontend is served same-origin through Caddy, so CORS is not needed.

---

## 5. P2 — Service layer, tools, agent

### 5.1 Service (`src/api/service.py`)

Read-only pandas access to `outputs/*.csv|json`. No network, no training.
Key functions: `get_scope`, `get_metrics`, `get_summary`, `get_forecast`,
`get_replenishment`, `get_action`, `get_explanations`, `simulate_promotion`.
`simulate_promotion` replays the replenishment math with an uplift percentage on
the committed forecast.

### 5.2 Tools (`src/api/tools.py`)

Plain Python functions (no framework dependency) that wrap the service and tag
every result with its `source` artifact:

`get_demand_forecast`, `get_inventory_status`, `get_replenishment_plan`,
`get_forecast_metrics`, `get_dashboard_summary`, `list_scope`,
`simulate_promotion`.

`TOOL_REGISTRY` maps names to callables; `TOOL_SPECS` is the machine-readable
description injected into the LLM prompt. Keeping tools framework-free means
the business logic is testable without LangGraph installed.

### 5.3 Agent (`src/api/agent.py`)

- `run_agent(message, history)` is the single entry point.
- If `LLM_GATEWAY_URL` / `LLM_GATEWAY_API_KEY` / `LLM_MODEL` are set, it runs a
  **LangGraph** `chatbot -> route -> tools -> chatbot` loop. The gateway does not
  support native tool calling, so the graph detects a JSON tool request in the
  model's text output (same workaround as `test_llm_gateway_langgraph.py`).
- Every tool call and result is recorded in `tool_trace`.
- If the gateway is missing or errors, it falls back to a **deterministic
  grounded agent** (`run_fallback`) that routes on intent keywords
  (reorder / forecast / risk / inventory / metrics / scope / promotion),
  extracts store and family (with Chinese + English aliases), calls the same
  tools, and formats the same numbers.

This guarantees the demo is fully usable offline, which is required because the
hackathon allows offline work and the gateway can be rate-limited.

### 5.4 Guardrails

- The LLM system prompt forbids inventing numbers; it may only use tool results.
- The fallback only quotes values returned by tools.
- Every tool result carries its `source` artifact and actions carry
  `assumption_note` so the UI can show where each number came from.

---

## 6. Assumptions (state these in the proposal)

1. **Proxy demand:** downstream retail sales are used as a proxy for the demand
   observed by the food distributor.
2. **Simulated operations:** inventory, shelf-life, lead-time, MOQ, unit cost and
   waste rate are deterministic mock scenarios, not Favorita operational data.
   `inventory.py` builds them and `assumption_note` is attached to every action.
3. **Scope:** the committed run covers 3 stores x 8 food families x a 7-day
   horizon. This is a demo scope; the pipeline is parameterized in `config.py`.

---

## 7. How to run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# optional: agent LLM mode
cp .env.example .env    # fill in the gateway values

# regenerate artifacts (needs data/, optional)
python -m src.food_forecast.pipeline

# start the API
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Without a `.env`, all endpoints still work; `/api/chat` reports
`"mode": "fallback"`.

## 8. How to test

```bash
python -m pytest -q
```

21 tests cover: service filtering and promotion math, tool registry and
JSON-serializability, the deterministic fallback, the LangGraph tool-routing
loop (with a mocked LLM), and every HTTP endpoint.

---

## 9. Deploy to Lightsail (no Docker)

Server requirements: Ubuntu with Python 3.10+, `git`, `git-lfs`, `caddy`.

### 9.1 First-time setup

```bash
# on the server
ssh ubuntu@<lightsail-static-ip>
sudo apt update && sudo apt install -y python3-venv git-lfs caddy

# read-only deploy key so the server can pull the private repo
ssh-keygen -t ed25519 -C "foodflow-deploy" -f ~/.ssh/foodflow_deploy -N ""
cat ~/.ssh/foodflow_deploy.pub      # add as a Deploy Key (read-only) in GitHub
```

Then run the bootstrap script from the repo (clone it first with LFS smudge
skipped so the server does not download 500 MB of data):

```bash
export GIT_LFS_SKIP_SMUDGE=1
git clone git@github.com:show-me-your-agents-3C/Food-Demand-Forcasting.git ~/Food-Demand-Forcasting
cd ~/Food-Demand-Forcasting
DOMAIN=your.domain.example.com ./deploy/bootstrap.sh
```

`bootstrap.sh` installs the venv, dependencies, `.env` (from the example),
the systemd unit and the Caddyfile, and starts the service. Then edit `.env`
with the real gateway credentials and restart:

```bash
sudo systemctl restart foodflow
```

### 9.2 Updates

```bash
cd ~/Food-Demand-Forcasting
./deploy/deploy.sh          # git pull + pip install + systemctl restart
```

Rollback:

```bash
git checkout v0.1 && sudo systemctl restart foodflow
```

### 9.3 Firewall / DNS

- Lightsail networking: allow 80 and 443; restrict 22 to your IP.
- Attach a static IP; point your domain's A record at it. Caddy gets the TLS
  certificate automatically.
- Caddy serves the frontend from `frontend/` (Person B) and reverse-proxies
  `/api/*` and `/health` to `127.0.0.1:8000`.

### 9.4 Why not `git pull` the data

`data/` is ~500 MB of LFS objects. The server runs
`git config lfs.fetchexclude '*'`, so pulls only fetch code and `outputs/`.
`outputs/` are normal small files and are what the API reads.

---

## 10. Handoff to Person B (frontend)

**Interface is frozen** — build against `/docs` (OpenAPI) and these endpoints:

- `GET /api/summary` for the landing dashboard (KPIs + top actions).
- `GET /api/forecast` and `GET /api/replenishment` for tables/charts.
- `POST /api/chat` for the agent panel. **Show `tool_trace` and `mode` in the
  UI** — visible tool calls and the fallback badge are the "show me your agents"
  evidence.
- `GET /health` for a status pill.

Suggested layout: top KPI row -> risk/reorder table -> 7-day forecast chart ->
chat panel with tool trace.

Frontend files go under `frontend/` so the two workstreams do not collide.
Person B does **not** need the gateway or raw data; the deployed API already
serves everything. Use `python -m http.server` or any static dev server and
point it at the deployed API (or run the API locally).

### Still to do (not Person A's scope)

- `frontend/` app and its build/serve step.
- Proposal document: problem, method justification, business value.
- Demo recording; agent trace on screen.
- Optional: enrich `outputs/` (more stores/families) by editing `config.py`
  and re-running the pipeline offline.

---

## 11. Known issues / limitations

- The LLM graph path was verified with a **mocked** model only (no gateway
  credentials available in this environment). The fallback path is fully tested.
- `requirements.txt` targets Python 3.10–3.12. The pinned `pandas`/`numpy` do
  not build on Python 3.14; do not use 3.14 on the server.
- The demo scope is small (3 stores x 8 food families). Widen it offline in
  `config.py` if more coverage is wanted; the API adapts automatically.
- `simulate_promotion` applies a flat uplift; it is a scenario tool, not a
  trained promotion-response model.
- CORS defaults to `*`. Tighten `CORS_ORIGINS` if the API is ever exposed
  cross-origin.

---

## 12. Commit / tag (needs explicit go-ahead)

Nothing has been committed. To finalize this branch:

```bash
git add -A
git commit -m "feat: backend API, LangGraph agent, Lightsail deploy, LF normalization"
git tag v0.1
git push -u origin feat/backend-agent-api
git push origin v0.1
```

After Person B merges `frontend/`, tag `v0.2` and deploy.
