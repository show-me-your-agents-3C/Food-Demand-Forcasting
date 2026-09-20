from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data_available"] is True


def test_scope():
    response = client.get("/api/scope")
    assert response.status_code == 200
    assert response.json()["stores"]


def test_metrics():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert "wape" in response.json()


def test_forecast_endpoint():
    response = client.get("/api/forecast", params={"store_nbr": 1, "family": "DAIRY", "horizon_days": 3})
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 3
    assert rows[0]["family"] == "DAIRY"


def test_replenishment_endpoint_validates_action():
    assert client.get("/api/replenishment", params={"action": "bogus"}).status_code == 422
    assert client.get("/api/replenishment", params={"action": "order_today"}).status_code == 200


def test_chat_endpoint_fallback():
    response = client.post("/api/chat", json={"message": "What should I reorder today?"})
    assert response.status_code == 200
    body = response.json()
    assert body["reply"]
    assert body["mode"] in {"fallback", "llm"}
