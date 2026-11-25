"""
Unit Tests for Model Selection: NLLB-200, IndicTrans2, and Locally Trained Pralekha
===================================================================================
Tests:
- Meta NLLB-200 foundation model availability, metadata status, and translation pipeline.
- AI4Bharat IndicTrans2 specialized model availability, metadata status, and translation pipeline.
- Locally trained Pralekha Edge Transformer (52M) availability, metadata status, and translation pipeline.
- Unified translation_service engine routing across all models.
- FastAPI routes /api/translation/models and /api/translate with model switching.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.services.nllb_service import (
    is_nllb_available,
    get_nllb_status,
    get_nllb_base_status,
    translate_with_nllb
)
from backend.app.services.indictrans_service import (
    is_indictrans_available,
    get_indictrans_status,
    translate_with_indictrans
)
from backend.app.services.pralekha_service import (
    is_pralekha_trained,
    get_pralekha_status,
    translate_with_pralekha
)
from backend.app.services.translation_service import translate_text


def test_nllb_service_availability_and_status():
    """Verify NLLB-200 availability and metadata reporting."""
    assert is_nllb_available() is True
    status = get_nllb_status()
    assert status["engine_id"] == "nllb"
    assert status["status"] in ("operational", "operational_finetuned")
    assert status["total_parameters"] > 600_000_000
    assert status["languages_supported"] >= 20

    base_status = get_nllb_base_status()
    assert base_status["engine_id"] == "nllb_base"
    assert base_status["is_finetuned"] is False
    assert base_status["status"] == "operational"


def test_indictrans_service_availability_and_status():
    """Verify IndicTrans2 availability and metadata reporting."""
    assert is_indictrans_available() is True
    status = get_indictrans_status()
    assert status["engine_id"] == "indictrans"
    assert status["status"] == "operational"
    assert status["total_parameters"] > 200_000_000
    assert status["languages_supported"] == 22


def test_pralekha_service_availability_and_status():
    """Verify Locally Trained Pralekha model availability and metadata reporting."""
    assert is_pralekha_trained() is True
    status = get_pralekha_status()
    assert status["engine_id"] == "pralekha"
    assert status["status"] == "operational_trained"


@pytest.mark.asyncio
async def test_nllb_translation_pipeline():
    """Verify translation via Meta NLLB-200 engine."""
    res = await translate_with_nllb("en", "hi", "Please provide fresh drinking water.")
    assert res["engine"] == "nllb"
    assert "पानी" in res["translated_text"] or "जल" in res["translated_text"]
    assert res["confidence"] > 0.90


@pytest.mark.asyncio
async def test_indictrans_translation_pipeline():
    """Verify translation via AI4Bharat IndicTrans2 engine."""
    res = await translate_with_indictrans("en", "hi", "Please provide fresh drinking water.")
    assert res["engine"] == "indictrans"
    assert "पानी" in res["translated_text"] or "जल" in res["translated_text"]
    assert res["confidence"] > 0.90


@pytest.mark.asyncio
async def test_pralekha_translation_pipeline():
    """Verify translation via locally trained Pralekha model."""
    res = await translate_with_pralekha("en", "hi", "Please provide fresh drinking water.")
    assert res["engine"] == "pralekha"
    assert "पानी" in res["translated_text"] or "जल" in res["translated_text"]
    assert res["confidence"] > 0.90


@pytest.mark.asyncio
async def test_translation_service_engine_switching():
    """Verify translation_service correctly dispatches across different models."""
    text = "Please provide fresh drinking water."

    # 1. Locally Trained Pralekha
    r_local = await translate_text("en", "hi", text, engine="pralekha")
    assert r_local["engine"] == "pralekha"
    assert "पानी" in r_local["translated_text"] or "जल" in r_local["translated_text"]

    # 2. Meta NLLB-200 Fine-Tuned
    r_nllb = await translate_text("en", "hi", text, engine="nllb")
    assert r_nllb["engine"] == "nllb"
    assert "पानी" in r_nllb["translated_text"] or "जल" in r_nllb["translated_text"]

    # 3. Meta NLLB-200 Base Foundation
    r_nllb_base = await translate_text("en", "hi", text, engine="nllb_base")
    assert r_nllb_base["engine"] == "nllb_base"
    assert "पानी" in r_nllb_base["translated_text"] or "जल" in r_nllb_base["translated_text"]

    # 4. AI4Bharat IndicTrans2
    r_it = await translate_text("en", "hi", text, engine="indictrans")
    assert r_it["engine"] == "indictrans"
    assert "पानी" in r_it["translated_text"] or "जल" in r_it["translated_text"]


@pytest.mark.asyncio
async def test_api_models_endpoint_returns_all_models():
    """Verify GET /api/translation/models returns all 5 selectable models."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/translation/models")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        engines = data["available_engines"]
        assert "pralekha" in engines
        assert "nllb" in engines
        assert "nllb_base" in engines
        assert "indictrans" in engines
        assert "universal" in engines
        assert "google" in engines

        model_ids = [m.get("id") or m.get("engine_id") for m in data["models"]]
        assert "pralekha" in model_ids
        assert "nllb" in model_ids
        assert "nllb_base" in model_ids
        assert "indictrans" in model_ids


@pytest.mark.asyncio
async def test_api_translate_with_selected_model():
    """Verify POST /api/translate executes with user's selected engine parameter."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Test NLLB Fine-Tuned
        res_nllb = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "hi",
            "text": "Please provide fresh drinking water.",
            "engine": "nllb"
        })
        assert res_nllb.status_code == 200
        data_nllb = res_nllb.json()
        assert data_nllb["engine"] == "nllb"

        # Test NLLB Base Foundation
        res_nllb_base = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "hi",
            "text": "Please provide fresh drinking water.",
            "engine": "nllb_base"
        })
        assert res_nllb_base.status_code == 200
        data_nllb_base = res_nllb_base.json()
        assert data_nllb_base["engine"] == "nllb_base"

        # Test IndicTrans2
        res_it = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "hi",
            "text": "Please provide fresh drinking water.",
            "engine": "indictrans"
        })
        assert res_it.status_code == 200
        data_it = res_it.json()
        assert data_it["engine"] == "indictrans"

        # Test Local Model (Pralekha)
        res_pr = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "hi",
            "text": "Please provide fresh drinking water.",
            "engine": "pralekha"
        })
        assert res_pr.status_code == 200
        data_pr = res_pr.json()
        assert data_pr["engine"] == "pralekha"
