"""
Speech Synthesis and Recognition API Routes
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Response
from pydantic import BaseModel, Field
from typing import Optional
from backend.app.services.speech_service import (
    synthesize_speech, transcribe_audio_file, get_supported_voice_profiles
)

router = APIRouter(prefix="/api/speech", tags=["Speech"])

class SynthesizeRequest(BaseModel):
    text: str = Field(description="Text to convert to speech")
    language: str = Field(default="hi", description="Language ISO code")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech rate (0.5 to 2.0)")

@router.post("/synthesize")
async def text_to_speech(req: SynthesizeRequest):
    """
    Synthesizes speech in Indian languages.
    Returns direct MP3 audio stream for playback.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        audio_bytes = synthesize_speech(
            text=req.text,
            lang_code=req.language,
            speed=req.speed
        )
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {str(e)}")

@router.post("/transcribe")
async def speech_to_text(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Optional expected language code")
):
    """
    Transcribes uploaded audio recording into text.
    """
    audio_content = await file.read()
    if not audio_content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    result = transcribe_audio_file(audio_content, source_language=language)
    return result

@router.get("/voices")
async def list_voices():
    """Lists voice engines and locale codes for 11 scheduled Indian languages + English."""
    return {
        "status": "success",
        "voices": get_supported_voice_profiles()
    }
