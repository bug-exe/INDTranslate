"""
Unit Tests for Aligned Multilingual Document Dataset & AI4Bharat Pralekha Scope
==============================================================================
Tests verify:
1. 100% N-way alignment completeness across all 22 Eighth Schedule languages + English (23 total).
2. Proper documentation and handling of the official AI4Bharat Pralekha 12-language scope (PIB/MKB).
3. Disk export generation (JSON, JSONL, per-language TXT, parallel pairs, Pralekha 12 subset).
4. FastAPI endpoints for dataset listing, single document retrieval, and parallel sentence pairs.
"""

import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.aligned_document_dataset import (
    get_aligned_documents,
    get_aligned_document_by_id,
    export_aligned_dataset_to_disk,
    ALL_23_LANGUAGES,
    ALL_22_SCHEDULED_LANGUAGES,
    PRALEKHA_OFFICIAL_12_LANGUAGES,
    PRALEKHA_OFFICIAL_11_INDIC,
    LANGUAGE_NAMES,
    DOCUMENTS_DATA_DIR
)

client = TestClient(app)


def test_pralekha_official_language_specification():
    """Verify official AI4Bharat Pralekha specification has exactly 12 languages."""
    assert len(PRALEKHA_OFFICIAL_12_LANGUAGES) == 12, "AI4Bharat Pralekha has exactly 12 languages"
    assert len(PRALEKHA_OFFICIAL_11_INDIC) == 11, "Pralekha has 11 Indic languages"
    assert "en" in PRALEKHA_OFFICIAL_12_LANGUAGES
    assert "hi" in PRALEKHA_OFFICIAL_12_LANGUAGES
    assert "bn" in PRALEKHA_OFFICIAL_12_LANGUAGES
    assert "ta" in PRALEKHA_OFFICIAL_12_LANGUAGES
    assert "te" in PRALEKHA_OFFICIAL_12_LANGUAGES

    # Excluded from official Pralekha (due to lack of PIB official releases)
    excluded_from_pralekha = set(ALL_22_SCHEDULED_LANGUAGES) - set(PRALEKHA_OFFICIAL_11_INDIC)
    assert len(excluded_from_pralekha) == 11
    assert "as" in excluded_from_pralekha  # Assamese
    assert "sa" in excluded_from_pralekha  # Sanskrit
    assert "brx" in excluded_from_pralekha # Bodo


def test_aligned_documents_universal_completeness():
    """Verify every document is 100% complete with non-empty text across all 23 languages."""
    docs = get_aligned_documents()
    assert len(docs) >= 6, f"Expected at least 6 aligned documents, got {len(docs)}"

    for doc in docs:
        assert doc.is_complete(), f"Document '{doc.doc_id}' should be complete across all 23 languages"
        assert len(doc.paragraphs) >= 1
        assert doc.get_sentence_count() >= 2

        # Check titles
        for l in ALL_23_LANGUAGES:
            assert bool(doc.title.get(l, "").strip()), f"Doc {doc.doc_id} missing title in language {l}"
            assert bool(doc.summary.get(l, "").strip()), f"Doc {doc.doc_id} missing summary in language {l}"

        # Check every sentence in every paragraph
        for p in doc.paragraphs:
            for s in p.sentences:
                for l in ALL_23_LANGUAGES:
                    text = s.get_text(l)
                    assert bool(text.strip()), f"Sentence {s.sentence_id} in {doc.doc_id} is empty for {l}"


def test_export_aligned_dataset_to_disk(tmp_path):
    """Verify exporting dataset to disk creates valid files for both universal 23 and Pralekha 12."""
    meta = export_aligned_dataset_to_disk(output_dir=tmp_path)
    assert meta["is_common_among_all_23_languages"] is True
    assert meta["universal_languages_count"] == 23
    assert meta["ai4bharat_pralekha_official_scope"]["official_languages_count"] == 12

    # Check files exist
    assert (tmp_path / "json" / "all_aligned_documents.json").exists()
    assert (tmp_path / "jsonl" / "aligned_documents.jsonl").exists()
    assert (tmp_path / "metadata.json").exists()

    # Check dedicated Pralekha 12 folder
    pralekha_json = tmp_path / "pralekha_official_12_languages" / "pralekha_12_documents.json"
    assert pralekha_json.exists()
    with open(pralekha_json, "r", encoding="utf-8") as f:
        p12_data = json.load(f)
        assert len(p12_data) == len(get_aligned_documents())
        for d in p12_data:
            assert set(d["title"].keys()) == set(PRALEKHA_OFFICIAL_12_LANGUAGES)

    # Check by_language text files
    for l in ALL_23_LANGUAGES:
        lang_dir = tmp_path / "by_language" / l
        assert lang_dir.exists()
        assert len(list(lang_dir.glob("*.txt"))) == len(get_aligned_documents())


def test_api_dataset_summary():
    """Verify GET /api/dataset/summary endpoint."""
    res = client.get("/api/dataset/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    assert data["total_documents"] >= 6
    assert data["is_common_among_all_23_languages"] is True
    assert data["scopes"]["ai4bharat_pralekha_official"]["language_count"] == 12
    assert data["scopes"]["universal_all_scheduled"]["language_count"] == 23


def test_api_list_documents_scopes():
    """Verify GET /api/dataset/documents supports all and pralekha scopes."""
    res_all = client.get("/api/dataset/documents?scope=all")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert len(data_all["languages_included"]) == 23

    res_p12 = client.get("/api/dataset/documents?scope=pralekha")
    assert res_p12.status_code == 200
    data_p12 = res_p12.json()
    assert len(data_p12["languages_included"]) == 12
    assert "as" not in data_p12["languages_included"]
    assert "en" in data_p12["languages_included"]
    assert "hi" in data_p12["languages_included"]


def test_api_get_document_by_id():
    """Verify GET /api/dataset/documents/{doc_id} with and without lang filter."""
    res = client.get("/api/dataset/documents/doc_01_constitution_preamble")
    assert res.status_code == 200
    data = res.json()
    assert data["doc_id"] == "doc_01_constitution_preamble"
    assert "paragraphs" in data

    # Test single language filter
    res_hi = client.get("/api/dataset/documents/doc_01_constitution_preamble?lang=hi")
    assert res_hi.status_code == 200
    data_hi = res_hi.json()
    assert data_hi["language"] == "hi"
    assert "भारत का संविधान" in data_hi["title"]
    assert len(data_hi["paragraphs"]) == 2


def test_api_get_parallel_pairs():
    """Verify GET /api/dataset/documents/{doc_id}/parallel produces exact aligned sentence pairs."""
    res = client.get("/api/dataset/documents/doc_01_constitution_preamble/parallel?src_lang=en&tgt_lang=hi")
    assert res.status_code == 200
    data = res.json()
    assert data["source_lang"] == "en"
    assert data["target_lang"] == "hi"
    assert data["sentence_pair_count"] == 4
    first_pair = data["sentence_pairs"][0]
    assert "We, the people of India" in first_pair["source_text"]
    assert "हम, भारत के लोग" in first_pair["target_text"]


def test_pralekha_strict_12way_database_endpoints():
    """Verify local SQLite database of strict 12-way documents and pivot-free pairs."""
    # 1. Summary
    res = client.get("/api/dataset/pralekha-strict-12way/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert data["total_documents"] >= 50
    assert data["total_training_pairs"] >= 50000
    assert data["language_pairs_covered"] == 132

    # 2. Documents listing
    res_docs = client.get("/api/dataset/pralekha-strict-12way/documents?limit=10")
    assert res_docs.status_code == 200
    docs_data = res_docs.json()
    assert len(docs_data["documents"]) == 10
    assert "doc_id" in docs_data["documents"][0]

    # 3. Direct Sample Pairs (Zero-Pivot Tamil -> Telugu)
    res_pairs = client.get("/api/dataset/pralekha-strict-12way/sample-pairs?source_lang=ta&target_lang=te&limit=5")
    assert res_pairs.status_code == 200
    pairs_data = res_pairs.json()
    assert pairs_data["is_direct_pivot_free"] is True
    assert pairs_data["source_lang"] == "ta"
    assert pairs_data["target_lang"] == "te"
    assert len(pairs_data["pairs"]) == 5
    assert len(pairs_data["pairs"][0]["source_text"]) > 0
    assert len(pairs_data["pairs"][0]["target_text"]) > 0

