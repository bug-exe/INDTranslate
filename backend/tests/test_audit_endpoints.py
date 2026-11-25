"""
IndTranslate Comprehensive API and Service Audit Test Suite
Covers all endpoints in backend/app/api/:
- routes_translation
- routes_isl
- routes_speech
- routes_history
- routes_settings
- ws_isl

Validates critical edge cases:
- Offline translation fallback
- Invalid language codes (source & target)
- Empty inputs across all routes
- Pralekha engine toggle
- DB persistence and lifecycle
"""

import io
import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport
from starlette.testclient import TestClient

from backend.app.main import app
from backend.app.services.isl_engine import isl_engine


# ============================================================================
# 1. TRANSLATION API AUDIT & EDGE CASES
# ============================================================================

@pytest.mark.asyncio
async def test_translate_empty_text_raises_400():
    """Verify HTTP 400 when text is empty or purely whitespace."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Empty string
        res = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "hi",
            "text": ""
        })
        assert res.status_code == 400
        assert "Text cannot be empty" in res.json()["detail"]

        # Whitespace only
        res_ws = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "hi",
            "text": "    \n\t  "
        })
        assert res_ws.status_code == 400
        assert "Text cannot be empty" in res_ws.json()["detail"]


@pytest.mark.asyncio
async def test_translate_invalid_language_codes_raises_400():
    """Verify HTTP 400 when invalid/unsupported source or target language codes are provided."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Invalid source language
        res_src = await client.post("/api/translate", json={
            "source_language": "klingon_src",
            "target_language": "hi",
            "text": "hello"
        })
        assert res_src.status_code == 400
        assert "Unsupported source language" in res_src.json()["detail"]

        # Invalid target language
        res_tgt = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "valyrian_tgt",
            "text": "hello"
        })
        assert res_tgt.status_code == 400
        assert "Unsupported target language" in res_tgt.json()["detail"]


@pytest.mark.asyncio
async def test_translate_identity_same_language():
    """Verify identical source and target language returns source text directly."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/api/translate", json={
            "source_language": "hi",
            "target_language": "hi",
            "text": "नमस्ते भारत",
            "save_history": False
        })
        assert res.status_code == 200
        data = res.json()
        assert data["translated_text"] == "नमस्ते भारत"
        assert "Identity" in data["pipeline"]
        assert data["confidence"] == 1.0


@pytest.mark.asyncio
async def test_translate_pralekha_engine_toggle():
    """Verify developer switch to Google, Pralekha (Model 3), and Universal routes correctly."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Pralekha Engine
        res_pralekha = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "sa",
            "text": "hello",
            "engine": "pralekha",
            "save_history": False
        })
        assert res_pralekha.status_code == 200
        data_p = res_pralekha.json()
        assert data_p["engine"] == "pralekha"
        assert "Pralekha" in data_p["pipeline"]
        assert data_p["translated_text"] == "नमस्ते"
        assert data_p["model_status"] is not None

        # 2. Google Translate Engine
        res_google = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "hi",
            "text": "hello",
            "engine": "google",
            "save_history": False
        })
        assert res_google.status_code == 200
        data_g = res_google.json()
        assert data_g["engine"] == "google"
        assert data_g["translated_text"] != ""


@pytest.mark.asyncio
async def test_translate_offline_fallback():
    """
    Verify offline fallback when external network connectivity is lost.
    Should route to offline local lexicon service without raising exceptions.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("backend.app.services.translation_service.is_network_available", return_value=False):
            res = await client.post("/api/translate", json={
                "source_language": "en",
                "target_language": "sa",
                "text": "peace",
                "engine": "universal",
                "save_history": False
            })
            assert res.status_code == 200
            data = res.json()
            assert data["translated_text"] == "शान्तिः"
            assert data["confidence"] > 0.0


@pytest.mark.asyncio
async def test_translate_db_persistence():
    """Verify that save_history=True persists translation to DB and returns a valid history_id."""
    phrase = "hello"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Translate with save_history=True (uses dictionary for instant lookup without network lag)
        res = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "sa",
            "text": phrase,
            "engine": "universal",
            "module": "text",
            "save_history": True
        })
        assert res.status_code == 200
        data = res.json()
        history_id = data.get("history_id")
        assert history_id is not None
        assert isinstance(history_id, int)

        # 2. Query history endpoint to verify persistence
        hist_res = await client.get("/api/history?limit=10")
        assert hist_res.status_code == 200
        hist_data = hist_res.json()
        assert hist_data["count"] >= 1
        record = next((r for r in hist_data["history"] if r["id"] == history_id), None)
        assert record is not None
        assert record["source_text"] == phrase
        assert record["source_language"] == "en"
        assert record["target_language"] == "sa"

        # 3. Translate with save_history=False
        res_no_save = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "sa",
            "text": "water",
            "save_history": False
        })
        assert res_no_save.status_code == 200
        assert res_no_save.json()["history_id"] is None


@pytest.mark.asyncio
async def test_languages_endpoint_metadata():
    """Verify GET /api/languages returns all 22 scheduled Indian languages plus English."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/languages")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["total"] == 23
        languages = data["languages"]
        assert len(languages) == 23
        for lang in languages:
            assert "code" in lang
            assert "name" in lang
            assert "native_name" in lang
            assert "script" in lang


@pytest.mark.asyncio
async def test_translation_models_endpoint():
    """Verify GET /api/translation/models lists Google, Pralekha, and Universal models."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/translation/models")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["default_engine"] == "universal"
        model_ids = [m.get("id") or m.get("engine_id") for m in data["models"]]
        assert "google" in model_ids
        assert "pralekha" in model_ids
        assert "universal" in model_ids
        assert "sarvam" not in model_ids


@pytest.mark.asyncio
async def test_network_and_offline_status():
    """Verify GET /api/translation/network-status returns valid network & offline attributes."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/translation/network-status")
        assert res.status_code == 200
        data = res.json()
        assert "network_online" in data
        assert isinstance(data["network_online"], bool)
        assert "offline_ready" in data
        assert isinstance(data["offline_ready"], bool)


@pytest.mark.asyncio
async def test_detect_endpoint():
    """Verify POST /api/detect accurately detects Indian scripts and handles edge cases."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Tamil detection
        res_ta = await client.post("/api/detect", json={"text": "வணக்கம்"})
        assert res_ta.status_code == 200
        assert res_ta.json()["code"] == "ta"

        # Devanagari detection
        res_hi = await client.post("/api/detect", json={"text": "नमस्ते"})
        assert res_hi.status_code == 200
        assert res_hi.json()["script"] == "Devanagari"

        # Empty text fallback
        res_empty = await client.post("/api/detect", json={"text": ""})
        assert res_empty.status_code == 200
        assert res_empty.json()["code"] == "en"


@pytest.mark.asyncio
async def test_transliterate_endpoint():
    """Verify POST /api/transliterate supports cross-script Indic transliteration."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Devanagari to Bengali
        res = await client.post("/api/transliterate", json={
            "text": "कमल",
            "source_script": "devanagari",
            "target_script": "bengali"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["original"] == "कमल"
        assert data["transliterated"] != "कमल"

        # Unsupported script fallback returns original
        res_unsupp = await client.post("/api/transliterate", json={
            "text": "test",
            "source_script": "unknown_a",
            "target_script": "unknown_b"
        })
        assert res_unsupp.status_code == 200
        assert res_unsupp.json()["transliterated"] == "test"


# ============================================================================
# 2. ISL API AUDIT & EDGE CASES
# ============================================================================

@pytest.mark.asyncio
async def test_isl_dictionary_filters():
    """Verify ISL dictionary listing with category and search query filters."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # All signs
        res = await client.get("/api/isl/dictionary")
        assert res.status_code == 200
        assert res.json()["total"] >= 20

        # Filter by category
        res_cat = await client.get("/api/isl/dictionary?category=Greetings")
        assert res_cat.status_code == 200
        signs = res_cat.json()["signs"]
        assert len(signs) > 0
        assert all(s["category"] == "Greetings" for s in signs)

        # Filter by non-existent category
        res_none = await client.get("/api/isl/dictionary?category=NonExistentCategory")
        assert res_none.status_code == 200
        assert res_none.json()["total"] == 0

        # Filter by search
        res_search = await client.get("/api/isl/dictionary?search=water")
        assert res_search.status_code == 200
        assert len(res_search.json()["signs"]) >= 1


@pytest.mark.asyncio
async def test_isl_sign_detail_and_404():
    """Verify fetching valid sign and 404 for invalid sign."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Valid sign
        res_valid = await client.get("/api/isl/signs/namaste")
        assert res_valid.status_code == 200
        assert res_valid.json()["sign"]["id"] == "namaste"

        # Invalid sign
        res_invalid = await client.get("/api/isl/signs/non_existent_sign_xyz")
        assert res_invalid.status_code == 404
        assert "Sign not found" in res_invalid.json()["detail"]


@pytest.mark.asyncio
async def test_isl_recognize_landmarks_and_threshold_override():
    """Verify landmark ingestion and dynamic confidence threshold override."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Ingest frame with custom threshold
        res = await client.post("/api/isl/recognize", json={
            "left_hand_landmarks": None,
            "right_hand_landmarks": None,
            "pose_landmarks": None,
            "confidence_threshold": 0.88
        })
        assert res.status_code == 200
        assert isl_engine.confidence_threshold == 0.88
        assert res.json()["confidence"] == 0.0


@pytest.mark.asyncio
async def test_isl_simulate_sign_valid_and_404():
    """Verify simulating kinematic sequence of valid sign and 404 on invalid sign."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Valid sign simulation
        res_sim = await client.post("/api/isl/simulate-sign/namaste")
        assert res_sim.status_code == 200
        data = res_sim.json()
        assert "top_candidates" in data
        assert "sign_recognized" in data
        assert data["sign_recognized"] == "namaste"

        # Invalid sign simulation
        res_inv = await client.post("/api/isl/simulate-sign/not_a_real_sign")
        assert res_inv.status_code == 404


@pytest.mark.asyncio
async def test_isl_sentence_operations():
    """Verify sentence clear and undo operations."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Seed tokens
        isl_engine.sentence_tokens = ["namaste", "water"]

        # Undo last token
        res_undo = await client.post("/api/isl/sentence/undo")
        assert res_undo.status_code == 200
        assert res_undo.json()["sentence_tokens"] == ["namaste"]

        # Clear sentence
        res_clear = await client.post("/api/isl/sentence/clear")
        assert res_clear.status_code == 200
        assert res_clear.json()["sentence_tokens"] == []
        assert res_clear.json()["sentence"] == ""


@pytest.mark.asyncio
async def test_isl_reverse_sign_search():
    """Verify reverse sign lookup for non-signers."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Existing sign query
        res_found = await client.get("/api/isl/reverse-sign?query=water")
        assert res_found.status_code == 200
        assert res_found.json()["found"] is True
        assert res_found.json()["primary_sign"]["id"] == "water"

        # Non-matching query
        res_not_found = await client.get("/api/isl/reverse-sign?query=supercalifragilistic")
        assert res_not_found.status_code == 200
        assert res_not_found.json()["found"] is False
        assert res_not_found.json()["primary_sign"] is None


@pytest.mark.asyncio
async def test_isl_model_info():
    """Verify GET /api/isl/model-info provides deep learning specs."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/isl/model-info")
        assert res.status_code == 200
        data = res.json()
        assert "model_architecture" in data
        assert "input_shape" in data
        assert data["total_sign_classes"] >= 20
        assert "namaste" in data["classes"]


# ============================================================================
# 3. SPEECH API AUDIT & EDGE CASES
# ============================================================================

@pytest.mark.asyncio
async def test_speech_synthesize_empty_text_raises_400():
    """Verify speech synthesis raises HTTP 400 for empty or whitespace text."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/api/speech/synthesize", json={
            "text": "   ",
            "language": "hi",
            "speed": 1.0
        })
        assert res.status_code == 400
        assert "Text cannot be empty" in res.json()["detail"]


@pytest.mark.asyncio
async def test_speech_synthesize_invalid_speed_raises_422():
    """Verify speech rate boundaries (ge=0.5, le=2.0) are enforced by schema validation."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Speed too high
        res_high = await client.post("/api/speech/synthesize", json={
            "text": "नमस्ते",
            "speed": 3.5
        })
        assert res_high.status_code == 422

        # Speed too low
        res_low = await client.post("/api/speech/synthesize", json={
            "text": "नमस्ते",
            "speed": 0.2
        })
        assert res_low.status_code == 422


@pytest.mark.asyncio
async def test_speech_synthesize_success_mock():
    """Verify successful speech synthesis returns audio/mpeg stream."""
    mock_audio = b"ID3\x03\x00\x00\x00\x00\x00#TSSE\x00\x00\x00\x0f\x00\x00\x03Lavf58.29.100"
    with patch("backend.app.api.routes_speech.synthesize_speech", return_value=mock_audio):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.post("/api/speech/synthesize", json={
                "text": "नमस्ते भारत",
                "language": "hi",
                "speed": 1.0
            })
            assert res.status_code == 200
            assert res.headers["content-type"] == "audio/mpeg"
            assert res.content == mock_audio


@pytest.mark.asyncio
async def test_speech_transcribe_empty_file_raises_400():
    """Verify transcription endpoint rejects empty audio files with HTTP 400."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("empty.wav", b"", "audio/wav")}
        res = await client.post("/api/speech/transcribe", files=files)
        assert res.status_code == 400
        assert "Empty audio file" in res.json()["detail"]


@pytest.mark.asyncio
async def test_speech_transcribe_success_mock():
    """Verify speech transcription pipeline returns transcription structure."""
    mock_result = {
        "transcript": "नमस्ते",
        "language": "hi",
        "confidence": 0.95,
        "status": "success"
    }
    with patch("backend.app.api.routes_speech.transcribe_audio_file", return_value=mock_result):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("test.wav", b"RIFFfakeaudiodataWAVEfmt", "audio/wav")}
            res = await client.post("/api/speech/transcribe?language=hi", files=files)
            assert res.status_code == 200
            data = res.json()
            assert data["transcript"] == "नमस्ते"
            assert data["status"] == "success"


@pytest.mark.asyncio
async def test_speech_voices_endpoint():
    """Verify GET /api/speech/voices lists voice engines across all 22 scheduled languages + English."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/speech/voices")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        voices = data["voices"]
        assert len(voices) == 23
        assert all("tts_supported" in v and "asr_supported" in v for v in voices)


# ============================================================================
# 4. HISTORY API AUDIT & PERSISTENCE
# ============================================================================

@pytest.mark.asyncio
async def test_history_crud_and_filters():
    """Verify history creation, query filters, favorite toggle, and deletion."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Create a distinct history record via translation using dictionary lookup
        res_trans = await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "sa",
            "text": "water",
            "module": "text",
            "save_history": True
        })
        assert res_trans.status_code == 200
        hid = res_trans.json()["history_id"]
        assert hid is not None

        # 2. Filter by search
        res_search = await client.get("/api/history?search=water")
        assert res_search.status_code == 200
        assert res_search.json()["count"] >= 1

        # 3. Toggle favorite
        res_fav = await client.post(f"/api/history/{hid}/favorite")
        assert res_fav.status_code == 200
        assert res_fav.json()["is_favorite"] is True

        # Toggle favorite back
        res_unfav = await client.post(f"/api/history/{hid}/favorite")
        assert res_unfav.status_code == 200
        assert res_unfav.json()["is_favorite"] is False

        # Favorite 404 on non-existent item
        res_fav_404 = await client.post("/api/history/999999/favorite")
        assert res_fav_404.status_code == 404

        # 4. Delete item
        res_del = await client.delete(f"/api/history/{hid}")
        assert res_del.status_code == 200
        assert "Deleted record" in res_del.json()["message"]

        # Delete 404 on non-existent item
        res_del_404 = await client.delete("/api/history/999999")
        assert res_del_404.status_code == 404


@pytest.mark.asyncio
async def test_history_export_and_clear_all():
    """Verify history export endpoint and clear_all endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create a record first using dictionary lookup for instant resolution
        await client.post("/api/translate", json={
            "source_language": "en",
            "target_language": "sa",
            "text": "food",
            "save_history": True
        })

        # Export history
        res_exp = await client.get("/api/history/export")
        assert res_exp.status_code == 200
        assert res_exp.json()["total"] >= 1
        assert "data" in res_exp.json()

        # Clear all
        res_clear = await client.delete("/api/history")
        assert res_clear.status_code == 200
        assert "All history cleared" in res_clear.json()["message"]

        # Subsequent history count should be 0
        res_post_clear = await client.get("/api/history")
        assert res_post_clear.status_code == 200
        assert res_post_clear.json()["count"] == 0


# ============================================================================
# 5. SETTINGS API AUDIT & RUNTIME PROPAGATION
# ============================================================================

@pytest.mark.asyncio
async def test_settings_get_defaults():
    """Verify GET /api/settings provides all default keys."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/settings")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        settings = data["settings"]
        assert "isl_confidence_threshold" in settings
        assert "default_source_lang" in settings
        assert "default_target_lang" in settings
        assert "speech_speed" in settings


@pytest.mark.asyncio
async def test_settings_update_and_runtime_propagation():
    """Verify updating settings persists to DB and immediately updates runtime isl_engine."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        new_threshold = "0.82"
        res = await client.post("/api/settings", json={
            "settings": {
                "isl_confidence_threshold": new_threshold,
                "speech_speed": "1.3",
                "custom_audit_key": "audit_value"
            }
        })
        assert res.status_code == 200
        assert "Settings updated successfully" in res.json()["message"]

        # Verify runtime propagation to isl_engine
        assert isl_engine.confidence_threshold == float(new_threshold)

        # Verify GET returns updated values
        get_res = await client.get("/api/settings")
        assert get_res.status_code == 200
        settings = get_res.json()["settings"]
        assert settings["isl_confidence_threshold"] == new_threshold
        assert settings["speech_speed"] == "1.3"
        assert settings["custom_audit_key"] == "audit_value"


@pytest.mark.asyncio
async def test_settings_invalid_threshold_graceful_handling():
    """Verify non-numeric threshold strings do not crash the settings endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post("/api/settings", json={
            "settings": {
                "isl_confidence_threshold": "not_a_valid_float"
            }
        })
        assert res.status_code == 200


# ============================================================================
# 6. WEBSOCKET STREAMING ISL AUDIT (/ws/isl)
# ============================================================================

def test_websocket_isl_lifecycle_and_actions():
    """
    Test real-time ISL WebSocket streaming:
    - Connection
    - action: clear
    - action: undo
    - action: frame landmark processing
    - Invalid JSON packet handling
    """
    sync_client = TestClient(app)
    with sync_client.websocket_connect("/ws/isl") as ws:
        # 1. Action: clear
        ws.send_text('{"action": "clear"}')
        data_clear = ws.receive_json()
        assert data_clear["event"] == "sentence_cleared"
        assert data_clear["sentence"] == ""
        assert data_clear["sentence_tokens"] == []

        # 2. Action: frame
        mock_hand = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
        ws.send_json({
            "action": "frame",
            "left_hand": mock_hand,
            "right_hand": mock_hand,
            "pose": None
        })
        data_frame = ws.receive_json()
        assert data_frame["event"] == "prediction"
        assert "confidence" in data_frame
        assert "status" in data_frame

        # 3. Action: undo
        ws.send_text('{"action": "undo"}')
        data_undo = ws.receive_json()
        assert data_undo["event"] == "sentence_undone"
        assert "sentence" in data_undo

        # 4. Invalid JSON
        ws.send_text("not_valid_json_packet{{{")
        data_err = ws.receive_json()
        assert "error" in data_err
        assert data_err["error"] == "Invalid JSON packet"
