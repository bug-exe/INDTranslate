"""
WebSocket Handler for Real-Time Streaming ISL Gesture Recognition
"""

import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.app.services.isl_engine import isl_engine

logger = logging.getLogger("indtranslate.websocket")
ws_router = APIRouter(tags=["WebSocket"])

@ws_router.websocket("/ws/isl")
async def websocket_isl_stream(websocket: WebSocket):
    """
    Bi-directional streaming WebSocket for real-time ISL landmark processing.
    Receives landmark frames, responds with recognized gesture, confidence, and sentence status.
    """
    await websocket.accept()
    isl_engine.ensure_model_loaded()
    logger.info("ISL WebSocket client connected; ISL model loaded on-demand.")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                packet = json.loads(data)
                action = packet.get("action", "frame")

                if action == "clear":
                    isl_engine.clear_sentence()
                    isl_engine.reset_buffer()
                    await websocket.send_json({
                        "event": "sentence_cleared",
                        "sentence": "",
                        "sentence_tokens": []
                    })
                elif action == "undo":
                    isl_engine.remove_last_token()
                    await websocket.send_json({
                        "event": "sentence_undone",
                        "sentence": isl_engine.build_natural_sentence(),
                        "sentence_tokens": list(isl_engine.sentence_tokens)
                    })
                elif action == "frame":
                    lh = packet.get("left_hand")
                    rh = packet.get("right_hand")
                    pose = packet.get("pose")
                    
                    result = isl_engine.process_frame_landmarks(
                        left_hand_landmarks=lh,
                        right_hand_landmarks=rh,
                        pose_landmarks=pose
                    )
                    await websocket.send_json({
                        "event": "prediction",
                        **result
                    })
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON packet"})
            except Exception as e:
                logger.error(f"Error processing landmark frame: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("ISL WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket unhandled error: {e}")
