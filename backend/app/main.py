"""
IndTranslate - Main FastAPI Application
AI-Powered Indian Sign Language and 22 Scheduled Indian Languages Translation Platform
Breaking Language Barriers Across India
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

# Eager PyTorch, Transformers, and BitsAndBytes initialization before TensorFlow
# Prevents LLVM symbol collision (bitcode-mdindex-threshold) between Torch CUDA and TF
import torch
import transformers
try:
    import bitsandbytes
except Exception:
    pass

# Dedicate GPU VRAM to PyTorch / Pralekha Seq2Seq model
# Run TensorFlow (ISL model) on CPU where it takes <2ms with zero VRAM consumption
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from backend.app.config import APP_NAME, APP_TAGLINE, APP_VERSION, FRONTEND_DIST
from backend.app.database import init_db
from backend.app.api.routes_translation import router as translation_router
from backend.app.api.routes_isl import router as isl_router
from backend.app.api.routes_speech import router as speech_router
from backend.app.api.routes_history import router as history_router
from backend.app.api.routes_settings import router as settings_router
from backend.app.api.routes_dataset import router as dataset_router
from backend.app.api.ws_isl import ws_router
from backend.app.services.isl_engine import isl_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB and ensure models are ready
    print(f"🚀 Starting {APP_NAME} v{APP_VERSION} - {APP_TAGLINE}")
    await init_db()
    print("✓ SQLite Database schema initialized.")
    print("✓ On-demand model loading active: ISL model loaded strictly when camera is opened; translation engines loaded on-demand.")

    yield
    print(f"Shutting down {APP_NAME} server.")

app = FastAPI(
    title=f"{APP_NAME} API",
    description=f"{APP_NAME}: {APP_TAGLINE}. Complete ISL & 22 Scheduled Indian Languages platform.",
    version=APP_VERSION,
    lifespan=lifespan
)

# CORS Middleware for React frontend and local tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(translation_router)
app.include_router(isl_router)
app.include_router(speech_router)
app.include_router(history_router)
app.include_router(settings_router)
app.include_router(dataset_router)
app.include_router(ws_router)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🇮🇳</text></svg>'
    return FileResponse(str(FRONTEND_DIST / "favicon.ico")) if (FRONTEND_DIST / "favicon.ico").exists() else Response(content=svg, media_type="image/svg+xml")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": isl_engine.model is not None,
        "confidence_threshold": isl_engine.confidence_threshold,
        "service": APP_NAME,
        "version": APP_VERSION
    }

# Mount Frontend static production build if available
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/")
    async def serve_root():
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Allow API and documentation routes to pass through to routers or raise 404
        if full_path.startswith("api") or full_path.startswith("ws") or full_path in ("docs", "openapi.json", "redoc"):
            raise HTTPException(status_code=404, detail="Endpoint not found")
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
else:
    @app.get("/")
    async def root():
        return {
            "platform": APP_NAME,
            "tagline": APP_TAGLINE,
            "version": APP_VERSION,
            "supported_scheduled_languages": 11,
            "isl_support": "Authentic ISLRTC Multimodal Sequence Recognition & Reverse Sign Synthesis",
            "documentation": "/docs",
            "status": "operational"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
