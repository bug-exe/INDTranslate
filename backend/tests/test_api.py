"""
API Integration Tests for IndTranslate FastAPI Endpoints
"""

import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app

@pytest.mark.asyncio
async def test_health_check_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

@pytest.mark.asyncio
async def test_languages_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Default / Universal: all 22 scheduled Indian languages + English = 23
        response = await client.get("/api/languages")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 23
        assert len(data["languages"]) == 23

        # 2. Filtered for Model 3 (Pralekha): exactly 12 official benchmark languages
        response_p = await client.get("/api/languages?engine=pralekha")
        assert response_p.status_code == 200
        data_p = response_p.json()
        assert data_p["total"] == 12
        assert len(data_p["languages"]) == 12
        assert "en" in [l["code"] for l in data_p["languages"]]
        assert "hi" in [l["code"] for l in data_p["languages"]]
        assert "sa" not in [l["code"] for l in data_p["languages"]]

@pytest.mark.asyncio
async def test_isl_dictionary_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/isl/dictionary")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 20
        assert "namaste" in [s["id"] for s in data["signs"]]

@pytest.mark.asyncio
async def test_isl_reverse_sign():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/isl/reverse-sign?query=namaste")
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is True
        assert data["primary_sign"]["id"] == "namaste"

@pytest.mark.asyncio
async def test_translation_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Universal engine
        response = await client.post(
            "/api/translate",
            json={
                "source_language": "en",
                "target_language": "hi",
                "text": "hello",
                "engine": "universal",
                "save_history": False
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "translated_text" in data
        assert data["engine"] == "universal"

        # 2. AI4Bharat Pralekha engine
        response_pralekha = await client.post(
            "/api/translate",
            json={
                "source_language": "en",
                "target_language": "hi",
                "text": "hello",
                "engine": "pralekha",
                "save_history": False
            }
        )
        assert response_pralekha.status_code == 200
        data_p = response_pralekha.json()
        assert "translated_text" in data_p
        assert data_p["engine"] == "pralekha"
        assert "Pralekha" in data_p["pipeline"]

@pytest.mark.asyncio
async def test_translation_models_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/translation/models")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["models"]) >= 2
        engine_ids = [m["id"] if "id" in m else m.get("engine_id") for m in data["models"]]
        assert "universal" in engine_ids
        assert "pralekha" in engine_ids
