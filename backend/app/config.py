import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODELS_DIR = BACKEND_DIR / "saved_models"
CACHE_DIR = BACKEND_DIR / "cache"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{DATA_DIR / 'indtranslate.db'}"

# Application Settings
APP_NAME = "IndTranslate"
APP_TAGLINE = "Breaking Language Barriers Across India"
APP_VERSION = "1.0.0"

# ISL Recognition Parameters
ISL_SEQUENCE_LENGTH = 30       # 30 frames temporal window
ISL_CONFIDENCE_THRESHOLD = 0.70 # Minimum confidence to output a sign
ISL_DEBOUNCE_FRAMES = 5        # Frames required before re-triggering same sign
ISL_FEATURE_DIM = 126          # 21 landmarks * 3 coords * 2 hands

# Speech Settings
DEFAULT_TTS_SPEED = 1.0
DEFAULT_TTS_PITCH = 1.0
