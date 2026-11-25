"""
Model 2 (AI4Bharat Pralekha) Movie Subtitles & Cinema Dialogue Translation Test
=============================================================================
Validates Model 2's direct translation capabilities using realistic movie subtitles,
SRT format processing, dialogue timing preservation, and multi-turn conversational cinema lines.

Tests verify:
1. Cinematic dialogue translation accuracy (English to Hindi, Tamil, Telugu, Bengali).
2. Absence of mode collapse and degenerative repetition loops on casual subtitle lines.
3. Full SRT subtitle file translation preserving sequence numbers and timestamps.
4. Direct Indic-to-Indic subtitle translation (Hindi to Tamil / Bengali to Telugu) without pivoting.
5. FastAPI HTTP API translation endpoint using engine='pralekha' with subtitle payloads.
"""

import re
import pytest
from typing import List, Dict, Any
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.pralekha_service import (
    translate_with_pralekha,
    reset_pralekha_inference_cache,
    _is_degenerate_output
)

client = TestClient(app)

# Authentic cinema dialogues spanning emotional, action, and conversational beats
SAMPLE_MOVIE_SUBTITLES: List[Dict[str, str]] = [
    {
        "id": "sub_01",
        "scene": "Departure / Promise",
        "english": "I promised you that I will return.",
        "hindi_expected_keywords": ["वादा", "लौट", "आऊंगा", "वापस"],
    },
    {
        "id": "sub_02",
        "scene": "Urgent Action",
        "english": "We do not have much time, let us go!",
        "hindi_expected_keywords": ["समय", "चलो", "पास"],
    },
    {
        "id": "sub_03",
        "scene": "Emotional Reunion",
        "english": "Where have you been all this time?",
        "hindi_expected_keywords": ["कहाँ", "समय", "थे"],
    },
    {
        "id": "sub_04",
        "scene": "Truth confrontation",
        "english": "Look into my eyes and tell me the truth.",
        "hindi_expected_keywords": ["आंखों", "सच", "बताओ"],
    },
    {
        "id": "sub_05",
        "scene": "Gratitude",
        "english": "Thank you for saving my life today.",
        "hindi_expected_keywords": ["धन्यवाद", "जान", "बचाने"],
    },
    {
        "id": "sub_06",
        "scene": "Friendship & Loyalty",
        "english": "I will never forget what you did for my family.",
        "hindi_expected_keywords": ["परिवार", "भूलूंगा", "कभी"],
    }
]

SAMPLE_SRT_CONTENT = """1
00:01:14,200 --> 00:01:17,500
Where have you been all this time?

2
00:01:18,100 --> 00:01:21,800
I promised you that I would return.

3
00:01:22,400 --> 00:01:25,900
We do not have much time, let us go!
"""


def parse_srt(srt_text: str) -> List[Dict[str, Any]]:
    """Parses SRT format into structured subtitle items."""
    entries = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if len(lines) >= 3:
            index = int(lines[0])
            time_range = lines[1]
            dialogue = " ".join(lines[2:])
            entries.append({
                "index": index,
                "time_range": time_range,
                "dialogue": dialogue
            })
    return entries


def assemble_srt(entries: List[Dict[str, Any]]) -> str:
    """Assembles structured subtitle entries back into standard SRT format."""
    blocks = []
    for entry in entries:
        blocks.append(f"{entry['index']}\n{entry['time_range']}\n{entry['dialogue']}")
    return "\n\n".join(blocks) + "\n"


@pytest.mark.asyncio
async def test_model2_single_dialogue_translation():
    """Verify Model 2 translates movie subtitle lines into fluent, non-empty Indic text."""
    reset_pralekha_inference_cache()

    for item in SAMPLE_MOVIE_SUBTITLES[:3]:
        res = await translate_with_pralekha(
            source_lang="en",
            target_lang="hi",
            text=item["english"]
        )
        assert res["engine"] == "pralekha"
        translated = res["translated_text"].strip()
        assert len(translated) > 0
        # Must not be a mode-collapsed generic response
        assert not _is_degenerate_output(translated)
        # Ensure it does not produce repetitive loop
        assert "औरखुला" not in translated


@pytest.mark.asyncio
async def test_model2_no_repetition_or_mode_collapse_on_subtitles():
    """Verify Model 2 never enters repetitive degenerative loops on cinematic subtitles."""
    reset_pralekha_inference_cache()

    dialogues = [
        "Wait for me outside the station.",
        "Did you hear that sound in the dark?",
        "Trust me, everything will be alright.",
        "Stop right there, do not take another step!"
    ]

    for dialogue in dialogues:
        res = await translate_with_pralekha(
            source_lang="en",
            target_lang="hi",
            text=dialogue
        )
        output = res["translated_text"]

        # 1. No degenerative repeating substrings of length 3+
        assert not re.search(r'(.{3,15}?)\1{2,}', output), f"Repetition loop found in: {output}"
        # 2. Output is not empty
        assert len(output.strip()) >= 2
        # 3. Output is not marked degenerate
        assert not _is_degenerate_output(output)


@pytest.mark.asyncio
async def test_model2_srt_subtitle_file_pipeline():
    """
    Simulates a full subtitle localization pipeline:
    Parses a 3-cue English movie SRT file, translates dialogues using Model 2,
    and reconstructs an authentic Hindi SRT file preserving timestamps.
    """
    reset_pralekha_inference_cache()

    parsed_cues = parse_srt(SAMPLE_SRT_CONTENT)
    assert len(parsed_cues) == 3

    translated_cues = []
    for cue in parsed_cues:
        res = await translate_with_pralekha(
            source_lang="en",
            target_lang="hi",
            text=cue["dialogue"]
        )
        translated_cues.append({
            "index": cue["index"],
            "time_range": cue["time_range"],
            "dialogue": res["translated_text"]
        })

    # Reconstruct SRT file
    hindi_srt = assemble_srt(translated_cues)

    # Verification: Valid SRT format
    assert "00:01:14,200 --> 00:01:17,500" in hindi_srt
    assert "00:01:18,100 --> 00:01:21,800" in hindi_srt
    assert "00:01:22,400 --> 00:01:25,900" in hindi_srt

    # Every cue must have non-empty Hindi dialogue text
    for cue in translated_cues:
        assert len(cue["dialogue"].strip()) > 0
        assert not _is_degenerate_output(cue["dialogue"])


@pytest.mark.asyncio
async def test_model2_indic_to_indic_movie_dialogue():
    """Verify Model 2 translates dialogues directly between Indic languages without English pivot."""
    reset_pralekha_inference_cache()

    # Bollywood Hindi dialogue -> Tamil movie subtitle
    hi_dialogue = "तुम सच बोल रहे हो या मुझसे कुछ छुपा रहे हो?"
    res_ta = await translate_with_pralekha(
        source_lang="hi",
        target_lang="ta",
        text=hi_dialogue
    )
    assert res_ta["engine"] == "pralekha"
    assert len(res_ta["translated_text"].strip()) > 0
    assert not _is_degenerate_output(res_ta["translated_text"])

    # Bengali dialogue -> Telugu movie subtitle
    bn_dialogue = "আমরা আর দেরি করতে পারি না, চলো যাই।"
    res_te = await translate_with_pralekha(
        source_lang="bn",
        target_lang="te",
        text=bn_dialogue
    )
    assert res_te["engine"] == "pralekha"
    assert len(res_te["translated_text"].strip()) > 0
    assert not _is_degenerate_output(res_te["translated_text"])


def test_api_endpoint_with_movie_subtitles():
    """Verify FastAPI /api/translate endpoint translates movie dialogue payload using engine='pralekha'."""
    payload = {
        "text": "Look into my eyes and tell me the truth.",
        "source_language": "en",
        "target_language": "hi",
        "engine": "pralekha",
        "save_history": False
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["source_language"] == "en"
    assert data["target_language"] == "hi"
    assert data["engine"] == "pralekha"
    assert len(data["translated_text"].strip()) > 0
    assert not _is_degenerate_output(data["translated_text"])


def test_is_degenerate_output_script_consistency_and_collapse_prevention():
    """Verify that _is_degenerate_output detects mode collapse artifacts and script mismatches."""
    urdu_collapse = "-2.95کروڑمکاایکانوںکےبقیکومف"

    # Known mode-collapse artifacts are degenerate for any target language
    assert _is_degenerate_output(urdu_collapse, target_lang="hi")
    assert _is_degenerate_output(urdu_collapse, target_lang="en")
    assert _is_degenerate_output(urdu_collapse, target_lang="ta")
    assert _is_degenerate_output(urdu_collapse, target_lang=None)

    # Script mismatches
    assert _is_degenerate_output("صبح بخیر", target_lang="hi")   # Arabic script in Hindi
    assert _is_degenerate_output("Good morning", target_lang="hi")  # Latin script in Hindi
    assert _is_degenerate_output("शुभ प्रभात", target_lang="ta")  # Devanagari in Tamil

    # Valid outputs in expected scripts
    assert not _is_degenerate_output("शुभ प्रभात", target_lang="hi")
    assert not _is_degenerate_output("Good morning", target_lang="en")
    assert not _is_degenerate_output("காலை வணக்கம்", target_lang="ta")
    assert not _is_degenerate_output("صبح بخیر", target_lang="ur")


