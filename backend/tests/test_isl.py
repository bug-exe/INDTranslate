"""
Unit Tests for Indian Sign Language (ISL) Recognition Engine
"""

import pytest
import numpy as np
from backend.app.services.isl_vocabulary import get_all_signs, get_sign_by_id, search_signs
from backend.app.services.isl_engine import isl_engine
from backend.app.config import ISL_SEQUENCE_LENGTH, ISL_FEATURE_DIM

def test_isl_vocabulary():
    """Verify authentic signs count and metadata."""
    signs = get_all_signs()
    assert len(signs) >= 20
    namaste = get_sign_by_id("namaste")
    assert namaste is not None
    assert namaste["category"] == "Greetings"
    assert namaste["is_two_handed"] is True
    assert len(namaste["avatar_frames"]) >= 2

def test_reverse_sign_search():
    """Verify non-signer query mapping for reverse sign communication."""
    results = search_signs("water")
    assert len(results) > 0
    assert results[0]["id"] == "water"

def test_isl_engine_frame_processing():
    """Verify landmark ingestion, sequence buffering, and prediction output shape."""
    isl_engine.reset_buffer()
    isl_engine.clear_sentence()
    
    # Fake 21 landmarks for left hand and right hand
    mock_hand = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    
    # Process 29 frames (under threshold 30)
    for _ in range(ISL_SEQUENCE_LENGTH - 1):
        res = isl_engine.process_frame_landmarks(left_hand_landmarks=mock_hand, right_hand_landmarks=mock_hand)
        assert res["status"] == "Buffering..."
        assert res["confidence"] == 0.0
        
    # Frame 30 triggers neural sequence inference
    res30 = isl_engine.process_frame_landmarks(left_hand_landmarks=mock_hand, right_hand_landmarks=mock_hand)
    assert "top_candidates" in res30
    assert len(res30["top_candidates"]) <= 3

def test_sentence_building_and_undo():
    """Verify sentence token management."""
    isl_engine.clear_sentence()
    isl_engine.sentence_tokens = ["namaste", "help", "water"]
    sentence = isl_engine.build_natural_sentence()
    assert "पानी" in sentence or "water" in sentence.lower()
    
    # Undo
    isl_engine.remove_last_token()
    assert isl_engine.sentence_tokens == ["namaste", "help"]
    
    # Clear
    isl_engine.clear_sentence()
    assert len(isl_engine.sentence_tokens) == 0
