"""
User Preferences & System Settings API Routes
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any
from backend.app.database import get_db
from backend.app.models.db_models import UserSetting
from backend.app.services.isl_engine import isl_engine

router = APIRouter(prefix="/api/settings", tags=["Settings"])

DEFAULT_SETTINGS = {
    "isl_confidence_threshold": "0.70",
    "default_source_lang": "auto",
    "default_target_lang": "hi",
    "camera_show_landmarks": "true",
    "speech_speed": "1.0",
    "auto_speech_playback": "false",
    "privacy_local_processing": "true",
    "transliteration_enabled": "true"
}

class UpdateSettingsRequest(BaseModel):
    settings: Dict[str, str]

@router.get("")
async def get_settings(db: AsyncSession = Depends(get_db)):
    """Retrieves all user settings merged with default values."""
    res = await db.execute(select(UserSetting))
    records = res.scalars().all()
    
    current_settings = dict(DEFAULT_SETTINGS)
    for r in records:
        current_settings[r.key] = r.value

    return {
        "status": "success",
        "settings": current_settings
    }

@router.post("")
async def save_settings(req: UpdateSettingsRequest, db: AsyncSession = Depends(get_db)):
    """Updates user settings."""
    for key, value in req.settings.items():
        res = await db.execute(select(UserSetting).where(UserSetting.key == key))
        setting = res.scalar_one_or_none()
        if setting:
            setting.value = str(value)
        else:
            setting = UserSetting(key=key, value=str(value))
            db.add(setting)

        # Apply runtime settings immediately
        if key == "isl_confidence_threshold":
            try:
                thresh = float(value)
                isl_engine.confidence_threshold = thresh
            except ValueError:
                pass

    await db.commit()
    return {"status": "success", "message": "Settings updated successfully"}
