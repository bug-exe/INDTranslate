"""
Real-time ISL Recognition & Continuous Signing Engine.
Integrates MediaPipe Holistic landmarks, Bi-GRU Sequence Classifier,
Temporal Window Debouncing, and Natural Sentence Synthesis.
"""

import os
import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from backend.app.config import (
    ISL_SEQUENCE_LENGTH, ISL_CONFIDENCE_THRESHOLD,
    ISL_DEBOUNCE_FRAMES, ISL_FEATURE_DIM, MODELS_DIR
)
from backend.app.services.isl_vocabulary import (
    ISL_SIGNS, SIGN_INDEX_TO_KEY, SIGN_KEY_TO_INDEX, TOTAL_SIGN_CLASSES,
    get_sign_by_id
)

class ISLRecognitionEngine:
    """
    Stateful real-time continuous ISL gesture recognition engine.
    Processes stream of normalized landmarks from webcam / MediaPipe Holistic.
    """
    def __init__(self, confidence_threshold: float = ISL_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.sequence_buffer = deque(maxlen=ISL_SEQUENCE_LENGTH)
        self.last_emitted_sign: Optional[str] = None
        self.debounce_counter = 0
        self.sentence_tokens: List[str] = []
        # Lazy on-demand loading: model is loaded only when camera opens or inference is invoked

    def ensure_model_loaded(self) -> bool:
        """Loads trained Keras model weights on-demand if not already in memory."""
        if self.model is not None:
            return True
        model_path = MODELS_DIR / "isl_model.keras"
        if model_path.exists():
            try:
                self.model = tf.keras.models.load_model(model_path)
                print(f"[ISLEngine] On-demand loaded ISL model from {model_path} (camera / stream active)")
                return True
            except Exception as e:
                print(f"[ISLEngine] Error loading model from {model_path}: {e}")
                self.model = None
                return False
        else:
            print(f"[ISLEngine] Warning: Model file {model_path} not found.")
            return False

    def unload_model(self) -> None:
        """Puts ISL model into standby and frees memory when camera is stopped."""
        if self.model is not None:
            del self.model
            self.model = None
            self.reset_buffer()
            print("[ISLEngine] ISL model released to standby (camera closed).")

    def _load_model(self):
        """Alias for ensure_model_loaded for backward compatibility."""
        return self.ensure_model_loaded()

    def reset_buffer(self):
        """Clears sequence buffer and debounce state."""
        self.sequence_buffer.clear()
        self.last_emitted_sign = None
        self.debounce_counter = 0

    def clear_sentence(self):
        """Resets the accumulated sentence."""
        self.sentence_tokens.clear()

    def remove_last_token(self):
        """Removes the most recent token from accumulated sentence."""
        if self.sentence_tokens:
            self.sentence_tokens.pop()

    def process_frame_landmarks(
        self,
        left_hand_landmarks: Optional[List[Dict[str, float]]] = None,
        right_hand_landmarks: Optional[List[Dict[str, float]]] = None,
        pose_landmarks: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Processes a single frame's landmarks, appends to temporal window,
        and runs neural sequence inference if buffer is full.
        """
        # Feature vector for this frame: 126 floats (21*3 + 21*3)
        frame_features = np.zeros(ISL_FEATURE_DIM, dtype=np.float32)

        has_left = left_hand_landmarks is not None and len(left_hand_landmarks) == 21
        has_right = right_hand_landmarks is not None and len(right_hand_landmarks) == 21

        if has_left:
            for i, lm in enumerate(left_hand_landmarks):
                idx = i * 3
                frame_features[idx] = lm.get("x", 0.0)
                frame_features[idx + 1] = lm.get("y", 0.0)
                frame_features[idx + 2] = lm.get("z", 0.0)

        if has_right:
            for i, lm in enumerate(right_hand_landmarks):
                idx = 63 + (i * 3)
                frame_features[idx] = lm.get("x", 0.0)
                frame_features[idx + 1] = lm.get("y", 0.0)
                frame_features[idx + 2] = lm.get("z", 0.0)

        # Hands present check
        hands_detected = has_left or has_right
        if not hands_detected:
            # Decay debounce when hands are removed
            self.debounce_counter = max(0, self.debounce_counter - 1)
            if self.debounce_counter == 0:
                self.last_emitted_sign = None

        self.sequence_buffer.append(frame_features)

        # Ensure model is primed on-demand if hands are detected in frame
        if self.model is None and hands_detected:
            self.ensure_model_loaded()

        # Infer if buffer is primed and model is loaded
        if len(self.sequence_buffer) < ISL_SEQUENCE_LENGTH or self.model is None or not hands_detected:
            return {
                "sign_recognized": None,
                "confidence": 0.0,
                "status": "Buffering..." if hands_detected else "No hands in frame",
                "sentence": self.build_natural_sentence(),
                "sentence_tokens": list(self.sentence_tokens),
                "is_confirmed": False,
                "top_candidates": []
            }

        # Shape: (1, 30, 126)
        input_tensor = np.expand_dims(np.array(self.sequence_buffer, dtype=np.float32), axis=0)
        predictions = self.model.predict(input_tensor, verbose=0)[0]

        top_indices = np.argsort(predictions)[::-1][:3]
        top_candidates = [
            {
                "sign_id": SIGN_INDEX_TO_KEY[idx],
                "label": ISL_SIGNS[SIGN_INDEX_TO_KEY[idx]]["label"],
                "confidence": float(round(predictions[idx], 4))
            }
            for idx in top_indices
        ]

        best_idx = top_indices[0]
        best_confidence = float(predictions[best_idx])
        best_sign_key = SIGN_INDEX_TO_KEY[best_idx]
        best_sign_meta = ISL_SIGNS.get(best_sign_key)

        is_confirmed = False
        status = "Tracking gesture..."

        # Confidence gating
        if best_confidence >= self.confidence_threshold:
            # Debounce check to prevent spamming
            if best_sign_key != self.last_emitted_sign:
                self.debounce_counter += 1
                if self.debounce_counter >= ISL_DEBOUNCE_FRAMES:
                    self.last_emitted_sign = best_sign_key
                    self.sentence_tokens.append(best_sign_key)
                    is_confirmed = True
                    status = f"Confirmed: {best_sign_meta['label']}"
                    self.debounce_counter = 0
            else:
                status = f"Holding sign: {best_sign_meta['label']}"
        else:
            status = f"Uncertain prediction ({best_confidence*100:.1f}%)"
            self.debounce_counter = max(0, self.debounce_counter - 1)

        return {
            "sign_recognized": best_sign_key if best_confidence >= self.confidence_threshold else None,
            "sign_label": best_sign_meta["label"] if best_sign_meta else None,
            "sign_label_hi": best_sign_meta["label_hi"] if best_sign_meta else None,
            "confidence": round(best_confidence, 4),
            "status": status,
            "is_confirmed": is_confirmed,
            "sentence": self.build_natural_sentence(),
            "sentence_tokens": list(self.sentence_tokens),
            "top_candidates": top_candidates
        }

    def build_natural_sentence(self) -> str:
        """
        Synthesizes accumulated ISL sign tokens into a fluent natural language sentence.
        Maps sign sequence grammar to conversational Hindi/English.
        """
        if not self.sentence_tokens:
            return ""

        # Domain phrase rules
        tokens = self.sentence_tokens
        
        # Rule: Greetings + Needs
        if "namaste" in tokens and "help" in tokens and "water" in tokens:
            return "नमस्ते, कृपया मुझे पानी के लिए सहायता चाहिए। (Hello, please help me with water.)"
        if "help" in tokens and "doctor" in tokens:
            return "आपातकाल! तुरंत डॉक्टर की आवश्यकता है। (Emergency! Doctor needed immediately.)"
        if "how_are_you" in tokens and "friend" in tokens:
            return "नमस्ते दोस्त, आप कैसे हैं? (Hello friend, how are you?)"
        if "thank_you" in tokens and "doctor" in tokens:
            return "डॉक्टर साहब, बहुत बहुत धन्यवाद। (Doctor, thank you very much.)"
        if "water" in tokens and "please" in tokens:
            return "कृपया मुझे थोड़ा पीने का पानी दीजिए। (Please give me some drinking water.)"
        if "food" in tokens and "please" in tokens:
            return "कृपया मुझे खाना दीजिए, मुझे भूख लगी है। (Please give me food, I am hungry.)"
        
        # Standard token joiner with human labels
        labels = [ISL_SIGNS[t]["label"].split("/")[0].strip() for t in tokens if t in ISL_SIGNS]
        return " → ".join(labels)


# Global singleton instance
isl_engine = ISLRecognitionEngine()
