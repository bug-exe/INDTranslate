"""
IndTranslate Aligned Multilingual Document Dataset API Routes
=============================================================
Provides endpoints to inspect, query, filter, and export the N-way
aligned document corpus. Supports both the full 23-language universal
corpus (22 Scheduled + English) and the official AI4Bharat Pralekha 12-language subset.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse

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

router = APIRouter(prefix="/api/dataset", tags=["Aligned Document Dataset"])


@router.get("/summary")
async def get_dataset_summary() -> Dict[str, Any]:
    """
    Returns high-level statistics and metadata about the aligned document corpus,
    including the official AI4Bharat Pralekha 12-language scope vs 23-language universal scope.
    """
    docs = get_aligned_documents()
    all_23_complete = all(d.is_complete() for d in docs)
    total_sentences = sum(d.get_sentence_count() for d in docs)

    return {
        "status": "available",
        "dataset_name": "IndTranslate Authentic Aligned Document Corpus",
        "total_documents": len(docs),
        "total_sentences": total_sentences,
        "is_common_among_all_23_languages": all_23_complete,
        "scopes": {
            "ai4bharat_pralekha_official": {
                "language_count": len(PRALEKHA_OFFICIAL_12_LANGUAGES),
                "languages": PRALEKHA_OFFICIAL_12_LANGUAGES,
                "indic_languages": PRALEKHA_OFFICIAL_11_INDIC,
                "description": "Matches official AI4Bharat Pralekha benchmark (PIB & Mann Ki Baat)",
                "source": "https://huggingface.co/datasets/ai4bharat/Pralekha"
            },
            "universal_all_scheduled": {
                "language_count": len(ALL_23_LANGUAGES),
                "languages": ALL_23_LANGUAGES,
                "description": "All 22 Eighth Schedule Indian Languages + English (100% N-way aligned)"
            }
        },
        "language_directory": {
            code: {
                "name": LANGUAGE_NAMES.get(code, code),
                "in_pralekha_official": code in PRALEKHA_OFFICIAL_12_LANGUAGES,
                "is_scheduled": code in ALL_22_SCHEDULED_LANGUAGES
            }
            for code in ALL_23_LANGUAGES
        },
        "disk_storage_path": str(DOCUMENTS_DATA_DIR)
    }


@router.get("/documents")
async def list_documents(
    scope: str = Query("all", description="Language scope: 'all' (23 languages) or 'pralekha' (12 official languages)"),
    domain: Optional[str] = Query(None, description="Optional domain filter (e.g. 'governance', 'healthcare', 'education')")
) -> Dict[str, Any]:
    """
    Lists all aligned documents with domain metadata, summaries, and word counts.
    """
    docs = get_aligned_documents()
    selected_langs = PRALEKHA_OFFICIAL_12_LANGUAGES if scope.lower() == "pralekha" else ALL_23_LANGUAGES

    results = []
    for doc in docs:
        if domain and doc.domain.lower() != domain.lower():
            continue

        results.append({
            "doc_id": doc.doc_id,
            "title": {l: doc.title.get(l, "") for l in selected_langs if l in doc.title},
            "domain": doc.domain,
            "summary": {l: doc.summary.get(l, "") for l in selected_langs if l in doc.summary},
            "paragraph_count": len(doc.paragraphs),
            "sentence_count": doc.get_sentence_count(),
            "metadata": doc.metadata
        })

    return {
        "scope": scope,
        "languages_included": selected_langs,
        "total_documents_matched": len(results),
        "documents": results
    }


@router.get("/documents/{doc_id}")
async def get_document_details(
    doc_id: str,
    lang: Optional[str] = Query(None, description="Optional single language code to retrieve full text only (e.g. 'hi', 'ta', 'en')")
) -> Dict[str, Any]:
    """
    Retrieves full content of an aligned document by ID, either for a single language or full N-way structure.
    """
    doc = get_aligned_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID '{doc_id}' not found.")

    if lang:
        lang_code = lang.lower().strip()
        if lang_code not in ALL_23_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{lang}'. Supported: {', '.join(ALL_23_LANGUAGES)}"
            )
        return {
            "doc_id": doc.doc_id,
            "language": lang_code,
            "language_name": LANGUAGE_NAMES.get(lang_code, lang_code),
            "title": doc.title.get(lang_code, ""),
            "domain": doc.domain,
            "summary": doc.summary.get(lang_code, ""),
            "full_text": doc.get_full_text(lang_code),
            "paragraphs": [
                {
                    "paragraph_id": p.paragraph_id,
                    "order_index": p.order_index,
                    "text": p.get_paragraph_text(lang_code),
                    "sentences": [s.get_text(lang_code) for s in p.sentences]
                }
                for p in doc.paragraphs
            ]
        }

    return doc.to_dict()


@router.get("/documents/{doc_id}/parallel")
async def get_parallel_document_pair(
    doc_id: str,
    src_lang: str = Query("en", description="Source language code (e.g. 'en')"),
    tgt_lang: str = Query("hi", description="Target language code (e.g. 'hi')")
) -> Dict[str, Any]:
    """
    Retrieves sentence-by-sentence parallel alignment between any two languages for a given document.
    """
    doc = get_aligned_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID '{doc_id}' not found.")

    src = src_lang.lower().strip()
    tgt = tgt_lang.lower().strip()

    if src not in ALL_23_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Invalid source language: '{src}'")
    if tgt not in ALL_23_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Invalid target language: '{tgt}'")

    aligned_pairs = []
    for p in doc.paragraphs:
        for s in p.sentences:
            aligned_pairs.append({
                "sentence_id": s.sentence_id,
                "paragraph_id": p.paragraph_id,
                "order_index": s.order_index,
                "source_text": s.get_text(src),
                "target_text": s.get_text(tgt)
            })

    return {
        "doc_id": doc.doc_id,
        "source_lang": src,
        "source_lang_name": LANGUAGE_NAMES.get(src, src),
        "target_lang": tgt,
        "target_lang_name": LANGUAGE_NAMES.get(tgt, tgt),
        "source_title": doc.title.get(src, ""),
        "target_title": doc.title.get(tgt, ""),
        "sentence_pair_count": len(aligned_pairs),
        "sentence_pairs": aligned_pairs
    }


@router.post("/export")
async def trigger_dataset_export() -> Dict[str, Any]:
    """
    Triggers physical export of the aligned documents dataset to disk in JSON, JSONL,
    by-language TXT, and parallel pair formats.
    """
    meta = export_aligned_dataset_to_disk()
    return {
        "status": "success",
        "message": "Aligned document corpus successfully exported to disk.",
        "export_metadata": meta
    }


@router.get("/pralekha-strict-12way/summary")
async def get_pralekha_strict_summary() -> Dict[str, Any]:
    """
    Returns summary statistics for the local database of strict 12-way aligned
    training documents from AI4Bharat Pralekha.
    """
    from backend.app.services.pralekha_strict_db import get_strict_database_summary
    return get_strict_database_summary()


@router.get("/pralekha-strict-12way/documents")
async def list_pralekha_strict_documents(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Lists documents stored in the local strict 12-way alignment database.
    """
    import sqlite3
    from backend.app.services.pralekha_strict_db import PRALEKHA_STRICT_DB_PATH
    conn = sqlite3.connect(str(PRALEKHA_STRICT_DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = "SELECT doc_id, title, domain, source_portal, sentence_count, created_at FROM documents"
    params = []
    if domain:
        query += " WHERE domain = ?"
        params.append(domain)
    query += " ORDER BY doc_id ASC LIMIT ? OFFSET ?;"
    params.extend([limit, offset])

    cur.execute(query, params)
    docs = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT COUNT(*) as total FROM documents" + (" WHERE domain = ?" if domain else ""), [domain] if domain else [])
    total = cur.fetchone()["total"]
    conn.close()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "documents": docs
    }


@router.get("/pralekha-strict-12way/sample-pairs")
async def get_pralekha_strict_sample_pairs(
    source_lang: str = Query(..., description="Source language ISO code, e.g. 'ta', 'mr', 'en'"),
    target_lang: str = Query(..., description="Target language ISO code, e.g. 'te', 'bn', 'hi'"),
    limit: int = Query(default=10, le=50)
) -> Dict[str, Any]:
    """
    Returns authentic direct (zero-pivot) sentence pairs for any of the 132 language directions
    from the local strict 12-way training database.
    """
    import sqlite3
    from backend.app.services.pralekha_strict_db import PRALEKHA_STRICT_DB_PATH, OFFICIAL_12_LANGUAGES
    src = source_lang.lower().strip()
    tgt = target_lang.lower().strip()

    if src not in OFFICIAL_12_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Source language '{src}' not in official 12 Pralekha languages.")
    if tgt not in OFFICIAL_12_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Target language '{tgt}' not in official 12 Pralekha languages.")

    conn = sqlite3.connect(str(PRALEKHA_STRICT_DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
    SELECT pair_id, doc_id, sentence_index, source_lang, target_lang, source_text, target_text, split
    FROM training_pairs
    WHERE source_lang = ? AND target_lang = ?
    LIMIT ?;
    """, (src, tgt, limit))
    pairs = [dict(r) for r in cur.fetchall()]
    conn.close()

    return {
        "source_lang": src,
        "target_lang": tgt,
        "count": len(pairs),
        "is_direct_pivot_free": True,
        "pairs": pairs
    }
