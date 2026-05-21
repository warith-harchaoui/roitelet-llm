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


def test_conversation_path_traversal_rejected():
    """Malicious conversation ids must return 404, not leak files outside data/."""
    # Both raw and URL-encoded traversal payloads must fail closed.
    for evil in ("..%2F..%2Fetc%2Fpasswd", "not-a-uuid", "%00"):
        response = client.get(f"/api/conversations/{evil}")
        assert response.status_code in (404, 422), f"Unexpected status for {evil}: {response.status_code}"


def test_api_settings_masks_api_keys():
    """GET /api/settings must never echo real API keys to the client."""
    from core.schemas import SECRET_MASK, SECRET_FIELDS
    from core.storage import storage

    stored = storage.load_app_settings()
    real_secret = 'sk-or-test-real-secret-value'
    updated = stored.model_copy(update={'openrouter_api_key': real_secret})
    storage.save_app_settings(updated)
    try:
        body = client.get('/api/settings').json()
        # Any non-empty secret must be masked.
        assert body['openrouter_api_key'] == SECRET_MASK
        for field in SECRET_FIELDS:
            assert body[field] != real_secret
    finally:
        storage.save_app_settings(stored)


def test_api_settings_post_preserves_masked_secrets():
    """POSTing the mask sentinel must keep the stored key, not overwrite it."""
    from core.schemas import SECRET_MASK
    from core.storage import storage

    stored = storage.load_app_settings()
    real_secret = 'sk-or-test-keep-me'
    storage.save_app_settings(stored.model_copy(update={'openrouter_api_key': real_secret}))
    try:
        masked = client.get('/api/settings').json()
        assert masked['openrouter_api_key'] == SECRET_MASK
        # Round-trip the masked payload unchanged.
        response = client.post('/api/settings', json=masked)
        assert response.status_code == 200
        # The on-disk value must still be the real secret.
        assert storage.load_app_settings().openrouter_api_key == real_secret
    finally:
        storage.save_app_settings(stored)


def test_api_settings_post_accepts_new_secret():
    """A POST with a real (non-mask) secret value must actually overwrite."""
    from core.storage import storage

    stored = storage.load_app_settings()
    try:
        new_secret = 'sk-or-test-new-value'
        next_payload = stored.model_copy(update={'openrouter_api_key': new_secret}).model_dump()
        response = client.post('/api/settings', json=next_payload)
        assert response.status_code == 200
        assert storage.load_app_settings().openrouter_api_key == new_secret
    finally:
        storage.save_app_settings(stored)
