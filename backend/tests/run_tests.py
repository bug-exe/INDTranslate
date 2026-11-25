"""
Standalone Test Runner for IndTranslate Backend
"""

import sys
import asyncio
from backend.tests.test_translation import (
    test_all_22_scheduled_languages_registered,
    test_language_detection,
    test_scheduled_special_dictionary_translation,
    test_transliteration
)
from backend.tests.test_isl import (
    test_isl_vocabulary,
    test_reverse_sign_search,
    test_isl_engine_frame_processing,
    test_sentence_building_and_undo
)
from backend.tests.test_api import (
    test_health_check_endpoint,
    test_languages_endpoint,
    test_isl_dictionary_endpoint,
    test_isl_reverse_sign,
    test_translation_endpoint
)

async def main():
    print("========================================")
    print(" Running IndTranslate Test Suite...")
    print("========================================")
    
    # 1. Translation Tests
    print("\n[1/3] Testing Translation & 22 Scheduled Languages Registry...")
    test_all_22_scheduled_languages_registered()
    print("  ✓ All 22 scheduled languages registered with complete metadata.")
    
    test_language_detection()
    print("  ✓ Heuristic script & language detection verified.")
    
    await test_scheduled_special_dictionary_translation()
    print("  ✓ Scheduled special dictionary and direct translations verified.")
    
    test_transliteration()
    print("  ✓ Script transliteration verified.")

    # 2. ISL Tests
    print("\n[2/3] Testing ISL Computer Vision & Neural Sequence Recognition...")
    test_isl_vocabulary()
    print("  ✓ Authentic ISLRTC vocabulary & avatar frames verified.")
    
    test_reverse_sign_search()
    print("  ✓ Reverse sign visualizer search verified.")
    
    test_isl_engine_frame_processing()
    print("  ✓ Landmark frame ingestion, buffering & neural prediction verified.")
    
    test_sentence_building_and_undo()
    print("  ✓ Sentence synthesis and token undo/clear verified.")

    # 3. API Endpoints Tests
    print("\n[3/3] Testing FastAPI HTTP & ASGI Endpoints...")
    await test_health_check_endpoint()
    print("  ✓ GET /api/health passed.")
    
    await test_languages_endpoint()
    print("  ✓ GET /api/languages passed.")
    
    await test_isl_dictionary_endpoint()
    print("  ✓ GET /api/isl/dictionary passed.")
    
    await test_isl_reverse_sign()
    print("  ✓ GET /api/isl/reverse-sign passed.")
    
    await test_translation_endpoint()
    print("  ✓ POST /api/translate passed.")

    print("\n========================================")
    print(" 🎉 ALL TESTS PASSED SUCCESSFULLY! (13/13)")
    print("========================================")

if __name__ == "__main__":
    asyncio.run(main())
