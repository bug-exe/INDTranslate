"""
Unit Tests for Translation Service and 22 Scheduled Languages
"""

import pytest
import asyncio
from backend.app.services.language_registry import get_all_languages, get_language, SCHEDULED_LANGUAGES
from backend.app.services.translation_service import translate_text, detect_language, transliterate_text

def test_all_22_scheduled_languages_registered():
    """Verify all 22 official scheduled languages are in the registry."""
    official_22 = [
        "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok",
        "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat",
        "sd", "ta", "te", "ur"
    ]
    for code in official_22:
        meta = get_language(code)
        assert meta is not None, f"Language {code} missing from registry"
        assert meta["name"] is not None
        assert meta["native_name"] is not None
        assert meta["script"] is not None

def test_language_detection():
    """Verify script and language detection heuristics."""
    # Devanagari Hindi
    res_hi = detect_language("नमस्ते, आप कैसे हैं?")
    assert res_hi["code"] == "hi" or res_hi["script"] == "Devanagari"
    
    # Bengali
    res_bn = detect_language("নমস্কার, কেমন আছেন?")
    assert res_bn["script"] == "Bengali"
    
    # Tamil
    res_ta = detect_language("வணக்கம், எப்படி இருக்கிறீர்கள்?")
    assert res_ta["code"] == "ta"
    
    # Gurmukhi Punjabi
    res_pa = detect_language("ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ")
    assert res_pa["code"] == "pa"

@pytest.mark.asyncio
async def test_scheduled_special_dictionary_translation():
    """Verify direct translation on authentic scheduled languages without English loss."""
    res_sa = await translate_text(source_lang="en", target_lang="sa", text="hello")
    assert res_sa["translated_text"] == "नमस्ते"
    assert "Authentic" in res_sa["pipeline"]

    res_brx = await translate_text(source_lang="en", target_lang="brx", text="hello")
    assert res_brx["translated_text"] == "खुलुमबाय"

    res_sat = await translate_text(source_lang="en", target_lang="sat", text="water")
    assert res_sat["translated_text"] == "ᱫᱟᱜ"

def test_transliteration():
    """Verify cross-script Indic transliteration."""
    # Devanagari to Bengali (same phonetic offset)
    dev_text = "कमल"
    beng_trans = transliterate_text(dev_text, "devanagari", "bengali")
    assert beng_trans != dev_text
    assert len(beng_trans) == len(dev_text)

@pytest.mark.asyncio
async def test_translation_engine_switching():
    """Verify developer switch across translation engines: 1. Google, 2. Pralekha (Model 3), and Auto Universal."""
    # 1. Google Translate Engine
    res_google = await translate_text(
        source_lang="en",
        target_lang="hi",
        text="hello",
        engine="google"
    )
    assert res_google["engine"] == "google"
    assert res_google["translated_text"] != ""

    # 2. AI4Bharat Pralekha Engine (Model 3)
    res_pralekha = await translate_text(
        source_lang="en",
        target_lang="hi",
        text="hello",
        engine="pralekha"
    )
    assert res_pralekha["engine"] == "pralekha"
    assert "Pralekha" in res_pralekha["pipeline"]
    assert res_pralekha["translated_text"] != ""

    # 3. Universal Engine (Auto-Adaptive)
    res_univ = await translate_text(
        source_lang="en",
        target_lang="hi",
        text="hello",
        engine="universal"
    )
    assert res_univ["engine"] == "universal"
    assert res_univ["translated_text"] != ""

@pytest.mark.asyncio
async def test_offline_fallback_translation():
    """Verify offline fallback translation mechanism."""
    from backend.app.services.translation_service import translate_offline, is_network_available
    assert isinstance(is_network_available(), bool)
    # Offline translation must succeed without throwing errors even when disconnected
    res = translate_offline(
        source_lang="en",
        target_lang="hi",
        text="hello"
    )
    assert res["is_offline"] is True
    assert res["translated_text"] != ""
    assert "Offline" in res["pipeline"]


@pytest.mark.asyncio
async def test_personal_name_and_typo_translation():
    """Verify that 'my name', typos like 'my nname', and name templates translate accurately."""
    # 1. Exact match 'my name'
    res_name = await translate_text("en", "hi", "my name", use_cache=False, engine="pralekha")
    assert res_name["translated_text"] == "मेरा नाम"
    assert res_name["engine"] == "pralekha"

    # 2. Typo tolerance: 'my nname' -> 'मेरा नाम'
    res_typo = await translate_text("en", "hi", "my nname", use_cache=False, engine="pralekha")
    assert res_typo["translated_text"] == "मेरा नाम"

    # 3. Dynamic template: 'my name is sagar' -> 'मेरा नाम सागर है'
    res_sagar = await translate_text("en", "hi", "my name is sagar", use_cache=False, engine="pralekha")
    assert res_sagar["translated_text"] == "मेरा नाम सागर है"

    # 4. Typo dynamic template: 'my nname is sagar' -> 'मेरा नाम सागर है'
    res_typo_sagar = await translate_text("en", "hi", "my nname is sagar", use_cache=False, engine="pralekha")
    assert res_typo_sagar["translated_text"] == "मेरा नाम सागर है"


def test_orthographic_validity_check():
    """Verify that tokens starting with combining marks or dependent vowels are flagged as degenerate."""
    from backend.app.services.pralekha_service import _is_degenerate_output

    # 'ामण्रिमण्डल' starts with dependent vowel sign \u093e (matra aa)
    assert _is_degenerate_output("ामण्रिमण्डल") is True
    assert _is_degenerate_output("ामंत्रिमण्डल") is True
    assert _is_degenerate_output("िहन्दी") is True

    # Valid Indic words
    assert _is_degenerate_output("मेरा नाम") is False
    assert _is_degenerate_output("हिन्दी") is False
    assert _is_degenerate_output("सागर") is False

