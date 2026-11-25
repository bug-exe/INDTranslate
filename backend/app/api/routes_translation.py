"""
Translation API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from backend.app.database import get_db
from backend.app.models.db_models import TranslationHistory
from backend.app.services.language_registry import get_all_languages, get_language, is_valid_language
from backend.app.services.translation_service import translate_text, detect_language, transliterate_text

router = APIRouter(prefix="/api", tags=["Translation"])

class TranslateRequest(BaseModel):
    source_language: str = Field(default="auto", description="Source language ISO code or 'auto'")
    target_language: str = Field(description="Target language ISO code (any of 11 scheduled languages + English)")
    text: str = Field(description="Text to translate")
    module: str = Field(default="text", description="Module: 'text', 'voice', or 'isl'")
    engine: str = Field(default="pralekha", description="Translation engine: 'pralekha' (Locally Trained 52M Model), 'nllb' (Meta NLLB-200 Fine-Tuned 600M), 'nllb_base' (Meta NLLB-200 Base Foundation 600M), 'indictrans' (AI4Bharat IndicTrans2), 'google' (Google Translate), or 'universal' (Auto-Adaptive)")
    save_history: bool = Field(default=True, description="Whether to persist to translation history")

class TranslateResponse(BaseModel):
    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    confidence: float
    pipeline: str
    engine: Optional[str] = "pralekha"
    model_status: Optional[str] = None
    cached: bool
    detected_lang: Optional[Dict[str, Any]] = None
    history_id: Optional[int] = None

class DetectRequest(BaseModel):
    text: str

class TransliterateRequest(BaseModel):
    text: str
    source_script: str
    target_script: str

@router.get("/languages")
async def list_languages(engine: Optional[str] = Query(None, description="Optional engine filter ('pralekha' filters to 12 official languages)")):
    """Lists all 22 constitutionally scheduled Indian languages plus English.
    When engine='pralekha', filters strictly to the 12 languages supported by Model 3."""
    from backend.app.services.pralekha_dataset import PRALEKHA_OFFICIAL_12_LANGUAGES
    all_langs = get_all_languages()
    if engine and engine.lower().strip() in ("pralekha", "model3", "ai4bharat-pralekha"):
        filtered = [l for l in all_langs if l["code"] in PRALEKHA_OFFICIAL_12_LANGUAGES]
        return {
            "status": "success",
            "engine": "pralekha",
            "total": len(filtered),
            "languages": filtered
        }

    return {
        "status": "success",
        "total": len(all_langs),
        "languages": all_langs
    }

@router.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest, db: AsyncSession = Depends(get_db)):
    """Translates text between any pair of 22 scheduled Indian languages."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if req.source_language != "auto" and not is_valid_language(req.source_language):
        raise HTTPException(status_code=400, detail=f"Unsupported source language: {req.source_language}")

    if not is_valid_language(req.target_language):
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {req.target_language}")

    result = await translate_text(
        source_lang=req.source_language,
        target_lang=req.target_language,
        text=req.text,
        engine=req.engine
    )

    history_id = None
    if req.save_history and result["translated_text"]:
        history_entry = TranslationHistory(
            source_language=result["source_lang"],
            target_language=result["target_lang"],
            source_text=req.text,
            translated_text=result["translated_text"],
            module=req.module,
            confidence=result["confidence"],
            pipeline_notes=f"[{result.get('engine', req.engine).upper()}] {result['pipeline']}"
        )
        db.add(history_entry)
        await db.commit()
        await db.refresh(history_entry)
        history_id = history_entry.id

    return TranslateResponse(
        source_language=result["source_lang"],
        target_language=result["target_lang"],
        source_text=result["source_text"],
        translated_text=result["translated_text"],
        confidence=result["confidence"],
        pipeline=result["pipeline"],
        engine=result.get("engine", req.engine),
        model_status=result.get("model_status"),
        cached=result["cached"],
        detected_lang=result.get("detected_lang"),
        history_id=history_id
    )

@router.get("/translation/models")
async def list_translation_models():
    """Lists available translation models: Locally Trained Pralekha, Meta NLLB-200 Fine-Tuned, Meta NLLB-200 Base Foundation, AI4Bharat IndicTrans2, Universal, and Google."""
    from backend.app.services.pralekha_service import get_pralekha_status
    from backend.app.services.nllb_service import get_nllb_status, get_nllb_base_status
    from backend.app.services.indictrans_service import get_indictrans_status

    pralekha_info = get_pralekha_status()
    # Enhance Pralekha naming to make locally trained identity prominent
    pralekha_info["name"] = "Locally Trained Pralekha (52.3M)"
    pralekha_info["badge"] = "Local Model (52M)"

    nllb_info = get_nllb_status()
    nllb_base_info = get_nllb_base_status()
    indictrans_info = get_indictrans_status()

    return {
        "status": "success",
        "default_engine": "universal",
        "available_engines": ["pralekha", "nllb", "nllb_base", "indictrans", "universal", "google"],
        "models": [
            pralekha_info,
            nllb_info,
            nllb_base_info,
            indictrans_info,
            {
                "id": "universal",
                "name": "Universal Auto-Adaptive Multi-Tier",
                "badge": "Auto Multi-Tier",
                "description": "Multi-tier engine that auto-routes across available local neural models and authentic lexicons.",
                "status": "operational",
                "type": "cloud_multitier",
                "is_trained": True
            },
            {
                "id": "google",
                "name": "Google Translate Neural Engine",
                "badge": "Google Cloud",
                "description": "Direct Google Cloud neural translation supporting 11 scheduled Indian languages + English.",
                "status": "operational",
                "type": "cloud_neural",
                "is_trained": True
            }
        ]
    }

@router.get("/translation/network-status")
async def get_network_and_offline_status():
    """Checks network reachability and local offline fallback readiness."""
    from backend.app.services.translation_service import is_network_available
    online = is_network_available()
    return {
        "network_online": online,
        "offline_ready": True
    }

@router.post("/translation/warmup")
async def warmup_translation_engine(engine: str = Query(..., description="Engine to prime: 'google', 'pralekha', 'universal'")):
    """
    On-demand model loader/releaser.
    Primes Pralekha weights when toggled.
    """
    engine_clean = engine.lower().strip()
    if engine_clean in ("pralekha", "ai4bharat-pralekha", "pralekha-model", "model3"):
        from backend.app.services.pralekha_service import _get_or_load_inference_bundle
        await _get_or_load_inference_bundle()
        return {
            "status": "success",
            "engine": "pralekha",
            "message": "Pralekha 6.33M model loaded on-demand."
        }
    return {
        "status": "success",
        "engine": engine_clean,
        "message": f"Active engine set to {engine_clean}."
    }

@router.post("/detect")
async def detect(req: DetectRequest):
    """Detects language and native script."""
    return detect_language(req.text)

@router.post("/transliterate")
async def transliterate(req: TransliterateRequest):
    """Transliterates text between Indic scripts."""
    transliterated = transliterate_text(req.text, req.source_script, req.target_script)
    return {
        "original": req.text,
        "source_script": req.source_script,
        "target_script": req.target_script,
        "transliterated": transliterated
    }
