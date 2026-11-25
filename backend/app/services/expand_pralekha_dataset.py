"""
Pralekha Dataset Expansion Engine: Strict 12-Way Multi-Parallel Scaler.
Streams through official AI4Bharat Pralekha train shards, enforces authentic
Unicode script constraints across all 11 Indic languages + English, detects
strict 12-way aligned documents, and incrementally indexes them into the local
SQLite database (backend/data/pralekha_strict_12way.db).
"""

import os
import sys
import re
import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from backend.app.config import DATA_DIR
from backend.app.services.pralekha_strict_db import (
    PRALEKHA_STRICT_DB_PATH,
    PRALEKHA_STRICT_DIR,
    OFFICIAL_12_LANGUAGES,
    init_strict_db,
    get_db_connection,
    insert_strict_document,
    get_strict_database_summary
)
from backend.app.services.extract_pralekha_strict_12way import (
    PRALEKHA_LANG_MAP,
    REVERSE_LANG_MAP,
    split_into_coherent_chunks
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("indtranslate.dataset_expander")

# Authentic Unicode script ranges for strict language verification
SCRIPT_RANGES = {
    "hi": ("\u0900", "\u097F"),  # Devanagari
    "mr": ("\u0900", "\u097F"),  # Devanagari
    "bn": ("\u0980", "\u09FF"),  # Bengali
    "ta": ("\u0B80", "\u0BFF"),  # Tamil
    "te": ("\u0C00", "\u0C7F"),  # Telugu
    "kn": ("\u0C80", "\u0CFF"),  # Kannada
    "gu": ("\u0A80", "\u0AFF"),  # Gujarati
    "pa": ("\u0A00", "\u0A7F"),  # Gurmukhi
    "ml": ("\u0D00", "\u0D7F"),  # Malayalam
    "or": ("\u0B00", "\u0B7F"),  # Odia
    "ur": ("\u0600", "\u06FF"),  # Arabic/Urdu
}

# Shard counts in ai4bharat/Pralekha train/ split
TRAIN_SHARDS = {
    "ori": 1,
    "kan": 2,
    "guj": 2,
    "mal": 2,
    "ben": 2,
    "pan": 3,
    "tel": 3,
    "mar": 3,
    "tam": 3,
    "urd": 4,
    "hin": 5
}


def is_valid_indic_text(text: str, iso2_lang: str) -> bool:
    """Verifies that the target text contains authentic characters of that language's script."""
    if not text or len(text.strip()) < 10:
        return False
    if iso2_lang not in SCRIPT_RANGES:
        return True
    start_ch, end_ch = SCRIPT_RANGES[iso2_lang]
    return any(start_ch <= ch <= end_ch for ch in text)


def get_existing_docs_info(conn: sqlite3.Connection) -> Tuple[int, Set[str]]:
    """Returns the highest document index and a set of existing English source text snippets."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    total_docs = cur.fetchone()[0]

    cur.execute("SELECT en FROM sentences_aligned WHERE sentence_index = 0")
    existing_snippets = set(row[0][:100].strip() for row in cur.fetchall())
    return total_docs, existing_snippets


def expand_strict_12way_database(
    target_new_documents: int = 2000,
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Progressively streams through Pralekha train shards, discovers new 12-way
    aligned documents, and inserts them into the local SQLite database.
    """
    db_file = db_path or PRALEKHA_STRICT_DB_PATH
    init_strict_db(db_file)
    conn = get_db_connection(db_file)

    total_existing_docs, existing_snippets = get_existing_docs_info(conn)
    logger.info(f"Database currently contains {total_existing_docs} documents.")
    logger.info(f"Targeting addition of {target_new_documents} new 12-way aligned documents...")

    # Ordered list of languages to process (from smallest to largest shard count)
    ordered_iso3 = ["ori", "kan", "guj", "mal", "ben", "pan", "tel", "mar", "tam", "urd", "hin"]
    all_indic_iso2 = [PRALEKHA_LANG_MAP[iso3] for iso3 in ordered_iso3]

    # Step 1: Initialize candidate English texts from Odia (ori)
    logger.info("Phase 1: Downloading & loading initial candidates from Odia (ori)...")
    ori_file = hf_hub_download(
        repo_id="ai4bharat/Pralekha",
        filename="train/eng_ori-00000-of-00001.parquet",
        repo_type="dataset"
    )
    t_ori = pq.read_table(ori_file)
    
    # Candidate map: normalized English source text -> {lang: target_text}
    doc_store: Dict[str, Dict[str, str]] = {}

    for src, tgt in zip(t_ori["src_txt"].to_pylist(), t_ori["tgt_txt"].to_pylist()):
        s = src.strip()
        t = tgt.strip()
        # Verify Odia text validity and that it's not already in the database
        if s and is_valid_indic_text(t, "or"):
            snippet = s.split("\n")[0][:100].strip()
            if snippet not in existing_snippets:
                doc_store[s] = {"or": t}

    logger.info(f"Initial unique Odia candidates (non-duplicate): {len(doc_store)}")

    # Step 2: Progressively intersect each subsequent Indic language
    for iso3 in ordered_iso3[1:]:
        iso2 = PRALEKHA_LANG_MAP[iso3]
        total_shards = TRAIN_SHARDS[iso3]
        logger.info(f"Phase 2: Intersecting with {iso3.upper()} ({iso2}) across {total_shards} shards...")

        lang_matches: Dict[str, str] = {}
        candidate_keys = set(doc_store.keys())

        for shard_idx in range(total_shards):
            fname = f"train/eng_{iso3}-{shard_idx:05d}-of-{total_shards:05d}.parquet"
            logger.info(f"  Fetching shard {shard_idx + 1}/{total_shards}: {fname}...")
            fpath = hf_hub_download(
                repo_id="ai4bharat/Pralekha",
                filename=fname,
                repo_type="dataset"
            )
            table = pq.read_table(fpath)
            for src, tgt in zip(table["src_txt"].to_pylist(), table["tgt_txt"].to_pylist()):
                s = src.strip()
                if s in candidate_keys:
                    t = tgt.strip()
                    if is_valid_indic_text(t, iso2):
                        lang_matches[s] = t

        # Filter candidates: keep only those present and valid in this language
        matched_keys = set(lang_matches.keys())
        logger.info(f"  ✓ Found {len(matched_keys)} matching documents in {iso3.upper()}.")

        # Prune doc_store
        doc_store = {k: doc_store[k] for k in matched_keys}
        for k in matched_keys:
            doc_store[k][iso2] = lang_matches[k]

        logger.info(f"  Active 12-way candidates remaining: {len(doc_store)}")

        if len(doc_store) == 0:
            logger.warning(f"Intersection dropped to 0 at {iso3}!")
            break

    final_12way_docs = list(doc_store.keys())
    logger.info(f"✓ Found {len(final_12way_docs)} brand-new documents with 100% strict 12-way alignment!")

    # Step 3: Insert into SQLite database in efficient WAL batch transactions
    docs_to_insert = final_12way_docs[:target_new_documents]
    next_doc_idx = total_existing_docs + 1
    total_new_docs_inserted = 0
    total_new_pairs_generated = 0

    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = WAL;")

    for en_text in docs_to_insert:
        doc_id = f"pralekha_12way_doc_{next_doc_idx:04d}"
        next_doc_idx += 1

        first_line = en_text.split("\n")[0][:120].strip()
        title = first_line if first_line else f"Pralekha Multi-lingual Document {next_doc_idx}"

        # Classify domain
        domain = "governance"
        low = en_text.lower()
        if any(w in low for w in ["health", "vaccin", "patient", "disease", "medical", "ayush"]):
            domain = "health"
        elif any(w in low for w in ["science", "space", "isro", "satellite", "technol", "ai", "digital"]):
            domain = "science_tech"
        elif any(w in low for w in ["farmer", "agricultur", "crop", "irrigation", "soil"]):
            domain = "agriculture"
        elif any(w in low for w in ["culture", "temple", "festival", "heritage", "music", "art"]):
            domain = "culture"
        elif any(w in low for w in ["education", "school", "nep", "university", "student", "teacher"]):
            domain = "education"

        en_chunks = split_into_coherent_chunks(en_text, max_chars=200)
        num_chunks = len(en_chunks)
        if num_chunks == 0:
            continue

        lang_chunks: Dict[str, List[str]] = {}
        valid_alignment = True

        for lang in all_indic_iso2:
            tgt_full = doc_store[en_text].get(lang, "")
            t_chunks = split_into_coherent_chunks(tgt_full, max_chars=200)
            if len(t_chunks) == 0:
                valid_alignment = False
                break
            lang_chunks[lang] = t_chunks

        if not valid_alignment:
            continue

        min_chunks = min(num_chunks, min(len(lang_chunks[l]) for l in all_indic_iso2))
        effective_chunks = min(min_chunks, 10)

        sentences_matrix: List[Dict[str, str]] = []
        for c_idx in range(effective_chunks):
            s_dict = {"en": en_chunks[c_idx]}
            for lang in all_indic_iso2:
                s_dict[lang] = lang_chunks[lang][c_idx]
            sentences_matrix.append(s_dict)

        if sentences_matrix:
            pairs_added = insert_strict_document(
                doc_id=doc_id,
                title=title,
                domain=domain,
                sentences_matrix=sentences_matrix,
                source_portal="AI4Bharat Pralekha (Official Train Split)",
                db_path=db_file,
                conn=conn
            )
            if pairs_added > 0:
                total_new_docs_inserted += 1
                total_new_pairs_generated += pairs_added

        if total_new_docs_inserted % 50 == 0:
            conn.commit()
            logger.info(f"Progress: {total_new_docs_inserted}/{len(docs_to_insert)} new documents inserted ({total_new_pairs_generated} pairs)...")

    # Build / optimize lookup indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pairs_lookup ON training_pairs (source_lang, target_lang, source_text);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_en ON sentences_aligned (en);")
    conn.commit()
    conn.close()

    summary = get_strict_database_summary(db_file)
    logger.info("=== Dataset Expansion Completed Successfully! ===")
    logger.info(f"New Documents Added: {total_new_docs_inserted}")
    logger.info(f"New Pairs Generated: {total_new_pairs_generated}")
    logger.info(f"Total Documents in DB: {summary['total_documents']}")
    logger.info(f"Total Aligned Sentences: {summary['total_aligned_sentences']}")
    logger.info(f"Total Direct Pivot-Free Training Pairs: {summary['total_training_pairs']}")

    summary_file = PRALEKHA_STRICT_DIR / "database_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    expand_strict_12way_database(target_new_documents=count)
