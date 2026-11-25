from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime
from datetime import datetime
from backend.app.database import Base

class TranslationHistory(Base):
    __tablename__ = "translation_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    source_language = Column(String(10), nullable=False)
    target_language = Column(String(10), nullable=False)
    source_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    module = Column(String(20), default="text") # 'text', 'voice', 'isl'
    confidence = Column(Float, default=1.0)
    pipeline_notes = Column(String(255), default="Direct Indic Neural Translation")
    is_favorite = Column(Boolean, default=False)

class UserSetting(Base):
    __tablename__ = "user_settings"

    key = Column(String(50), primary_key=True, index=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
