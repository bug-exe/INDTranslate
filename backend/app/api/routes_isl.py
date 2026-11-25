"""
Indian Sign Language (ISL) API Routes
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from backend.app.services.isl_vocabulary import (
    get_all_signs, get_sign_by_id, search_signs, ISL_SIGNS, TOTAL_SIGN_CLASSES
)
from backend.app.services.isl_engine import isl_engine

router = APIRouter(prefix="/api/isl", tags=["Indian Sign Language"])

class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: Optional[float] = 0.0

class RecognizeRequest(BaseModel):
    left_hand_landmarks: Optional[List[Dict[str, float]]] = None
    right_hand_landmarks: Optional[List[Dict[str, float]]] = None
    pose_landmarks: Optional[List[Dict[str, float]]] = None
    confidence_threshold: Optional[float] = None

class SentenceActionRequest(BaseModel):
    action: str = Field(description="'clear', 'undo', or 'add'")
    token: Optional[str] = None

@router.get("/dictionary")
async def get_dictionary(category: Optional[str] = None, search: Optional[str] = None):
    """Returns authentic ISL dictionary signs from ISLRTC curriculum."""
    signs = get_all_signs()
    if category:
        signs = [s for s in signs if s["category"].lower() == category.lower()]
    if search:
        signs = search_signs(search)
    return {
        "status": "success",
        "total": len(signs),
        "signs": signs
    }

@router.get("/signs/{sign_id}")
async def get_sign(sign_id: str):
    """Gets kinematic details and avatar animation frames for a sign."""
    sign = get_sign_by_id(sign_id)
    if not sign:
        raise HTTPException(status_code=404, detail="Sign not found in authentic ISL dictionary")
    return {"status": "success", "sign": sign}

@router.post("/recognize")
async def recognize_landmarks(req: RecognizeRequest):
    """
    Real-time sequence recognition endpoint.
    Receives current frame's landmarks, buffers, and infers gesture.
    """
    if req.confidence_threshold is not None:
        isl_engine.confidence_threshold = req.confidence_threshold

    result = isl_engine.process_frame_landmarks(
        left_hand_landmarks=req.left_hand_landmarks,
        right_hand_landmarks=req.right_hand_landmarks,
        pose_landmarks=req.pose_landmarks
    )
    return result

@router.post("/simulate-sign/{sign_id}")
async def simulate_sign_recognition(sign_id: str):
    """
    Feeds authentic ISLRTC kinematic sequence of a sign into the engine.
    Used for instant testing, verification, and interactive demoing.
    """
    from backend.app.services.isl_dataset_builder import generate_sign_sequence
    from backend.app.config import ISL_SEQUENCE_LENGTH, ISL_DEBOUNCE_FRAMES
    
    sign_key = sign_id.lower().strip()
    if sign_key not in ISL_SIGNS:
        raise HTTPException(status_code=404, detail=f"Sign '{sign_id}' not found in ISL dictionary")
        
    seq = generate_sign_sequence(sign_key, ISL_SEQUENCE_LENGTH)
    total_frames = ISL_SEQUENCE_LENGTH + ISL_DEBOUNCE_FRAMES + 1
    
    last_res = None
    for frame_idx in range(total_frames):
        f = min(frame_idx, ISL_SEQUENCE_LENGTH - 1)
        lh_data = [{"x": float(seq[f, i*3]), "y": float(seq[f, i*3+1]), "z": float(seq[f, i*3+2])} for i in range(21)]
        rh_data = [{"x": float(seq[f, 63+i*3]), "y": float(seq[f, 63+i*3+1]), "z": float(seq[f, 63+i*3+2])} for i in range(21)]
        last_res = isl_engine.process_frame_landmarks(left_hand_landmarks=lh_data, right_hand_landmarks=rh_data)
        
    return last_res

@router.post("/sentence/clear")
async def clear_sentence():
    """Resets the accumulated ISL sentence buffer."""
    isl_engine.clear_sentence()
    isl_engine.reset_buffer()
    return {"status": "success", "sentence": "", "sentence_tokens": []}

@router.post("/sentence/undo")
async def undo_sentence_token():
    """Removes the last recognized token from the sentence buffer."""
    isl_engine.remove_last_token()
    return {
        "status": "success",
        "sentence": isl_engine.build_natural_sentence(),
        "sentence_tokens": list(isl_engine.sentence_tokens)
    }

@router.post("/camera-active")
async def set_camera_active(active: bool = Query(..., description="Whether webcam is active (True loads ISL model, False puts in standby)")):
    """
    On-demand loader for the ISL Bi-GRU neural model.
    Only loads model weights when the webcam is actively opened by the user,
    and unloads/releases memory when the webcam is turned off.
    """
    if active:
        loaded = isl_engine.ensure_model_loaded()
        return {"status": "success", "action": "loaded", "model_loaded": loaded}
    else:
        isl_engine.unload_model()
        return {"status": "success", "action": "standby", "model_loaded": False}

@router.get("/reverse-sign")
async def reverse_sign_translate(query: str = Query(..., description="English or Hindi word to find ISL sign for")):
    """
    Reverse Communication Feature:
    Allows non-signers to type or speak in any language and view the corresponding authentic ISL sign
    with full visual instructions and animated keypoint frames.
    """
    matches = search_signs(query)
    if not matches:
        # Check partial or normalized
        q_norm = query.lower().strip()
        for s in ISL_SIGNS.values():
            if q_norm in s["id"] or q_norm in s["label"].lower():
                matches.append(s)
                break

    return {
        "status": "success",
        "query": query,
        "found": len(matches) > 0,
        "results": matches,
        "primary_sign": matches[0] if matches else None
    }

@router.get("/model-info")
async def get_model_info():
    """Returns ISL Deep Learning model specifications."""
    return {
        "model_architecture": "Bi-directional GRU with Spatial Feature Fusion & Temporal Pooling",
        "input_shape": [30, 126],
        "sequence_window_frames": 30,
        "confidence_threshold": isl_engine.confidence_threshold,
        "debounce_frames": 5,
        "total_sign_classes": TOTAL_SIGN_CLASSES,
        "classes": list(ISL_SIGNS.keys()),
        "is_model_loaded": isl_engine.model is not None,
        "islrtc_aligned": True
    }
