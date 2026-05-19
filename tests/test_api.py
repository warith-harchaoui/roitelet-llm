import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "roitelet" in response.json().get("service", "")


def test_root_serves_spa():
    """The vanilla JS client is mounted at '/' and must be served as HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>Roitelet</title>" in response.text

def test_v1_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert len(response.json()["data"]) > 0

def test_api_settings():
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert "local_synthesis_model" in response.json()
