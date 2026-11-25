"""
Pralekha Strict 12-Way Scaling Engine: Scale to 10,000 Documents.
Stage 1: Ingests all 4,209 verified 12-way parallel documents from local train shards (2,071 -> 6,280 docs).
Stage 2: Streams and intersects alignable shards by n_id to reach exactly 10,000 strict 12-way documents.
"""

import os
import sys
import json
import sqlite3
import hashlib
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
logger = logging.getLogger("indtranslate.scale_10k")

SCRIPT_RANGES = {
    "hi": ("\u0900", "\u097F"),
    "mr": ("\u0900", "\u097F"),
    "bn": ("\u0980", "\u09FF"),
    "ta": ("\u0B80", "\u0BFF"),
    "te": ("\u0C00", "\u0C7F"),
    "kn": ("\u0C80", "\u0CFF"),
    "gu": ("\u0A80", "\u0AFF"),
    "pa": ("\u0A00", "\u0A7F"),
    "ml": ("\u0D00", "\u0D7F"),
    "or": ("\u0B00", "\u0B7F"),
    "ur": ("\u0600", "\u06FF"),
}

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

ALIGNABLE_SHARDS = {
    "ori": 1,
    "mal": 1,
    "kan": 2,
    "guj": 2,
    "ben": 2,
    "mar": 2,
    "pan": 2,
    "tam": 2,
    "tel": 2,
    "urd": 3,
    "eng": 3,
    "hin": 4
}


def is_valid_indic_text(text: str, iso2_lang: str) -> bool:
    if not text or len(text.strip()) < 10:
        return False
    if iso2_lang not in SCRIPT_RANGES:
        return True
    start_ch, end_ch = SCRIPT_RANGES[iso2_lang]
    return any(start_ch <= ch <= end_ch for ch in text)


def get_existing_doc_hashes(conn: sqlite3.Connection) -> Tuple[int, Set[str]]:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    total_docs = cur.fetchone()[0]

    cur.execute("SELECT doc_id, GROUP_CONCAT(en, ' ') FROM sentences_aligned GROUP BY doc_id")
    hashes = set(hashlib.md5(r[1].encode('utf-8')).hexdigest() for r in cur.fetchall() if r[1])
    return total_docs, hashes


def classify_domain(en_text: str) -> str:
    low = en_text.lower()
    if any(w in low for w in ["health", "vaccin", "patient", "disease", "medical", "ayush", "hospital"]):
        return "health"
    elif any(w in low for w in ["science", "space", "isro", "satellite", "technol", "ai", "digital"]):
        return "science_tech"
    elif any(w in low for w in ["farmer", "agricultur", "crop", "irrigation", "soil", "kisan"]):
        return "agriculture"
    elif any(w in low for w in ["culture", "temple", "festival", "heritage", "music", "art", "monument"]):
        return "culture"
    elif any(w in low for w in ["education", "school", "nep", "university", "student", "teacher", "exam"]):
        return "education"
    elif any(w in low for w in ["railway", "train", "highway", "port", "airport", "infra"]):
        return "infrastructure"
    return "governance"


def run_stage1_train_ingestion(
    target_docs: int = 10000,
    db_path: Optional[Path] = None
) -> int:
    """
    Stage 1: Ingest all available 12-way parallel documents from local train shards.
    """
    db_file = db_path or PRALEKHA_STRICT_DB_PATH
    init_strict_db(db_file)
    conn = get_db_connection(db_file)

    total_existing, existing_hashes = get_existing_doc_hashes(conn)
    logger.info(f"=== Starting Stage 1: Local Train Shard Ingestion ===")
    logger.info(f"Current documents in database: {total_existing}")
    if total_existing >= target_docs:
        logger.info(f"Database already contains {total_existing} documents >= target {target_docs}.")
        conn.close()
        return total_existing

    # Step 1: Load Odia train candidates
    logger.info("Loading candidates from local train/eng_ori shard...")
    p_ori = hf_hub_download(repo_id="ai4bharat/Pralekha", filename="train/eng_ori-00000-of-00001.parquet", repo_type="dataset")
    t_ori = pq.read_table(p_ori)

    # doc_store: English source text -> {lang: target_text}
    doc_store: Dict[str, Dict[str, str]] = {}
    for src, tgt in zip(t_ori["src_txt"].to_pylist(), t_ori["tgt_txt"].to_pylist()):
        s = src.strip()
        t = tgt.strip()
        if s and is_valid_indic_text(t, "or"):
            if s not in doc_store:
                doc_store[s] = {"or": t}

    logger.info(f"Initial unique Odia candidates from train: {len(doc_store)}")

    # Step 2: Intersect across remaining 10 Indic languages
    ordered_iso3 = ["kan", "guj", "mal", "ben", "pan", "tel", "mar", "tam", "urd", "hin"]
    for iso3 in ordered_iso3:
        iso2 = PRALEKHA_LANG_MAP[iso3]
        total_shards = TRAIN_SHARDS[iso3]
        lang_matches: Dict[str, str] = {}
        cand_keys = set(doc_store.keys())

        for s_idx in range(total_shards):
            fname = f"train/eng_{iso3}-{s_idx:05d}-of-{total_shards:05d}.parquet"
            fpath = hf_hub_download(repo_id="ai4bharat/Pralekha", filename=fname, repo_type="dataset")
            table = pq.read_table(fpath)
            for src, tgt in zip(table["src_txt"].to_pylist(), table["tgt_txt"].to_pylist()):
                s = src.strip()
                if s in cand_keys and s not in lang_matches:
                    t = tgt.strip()
                    if is_valid_indic_text(t, iso2):
                        lang_matches[s] = t

        matched_keys = set(lang_matches.keys())
        doc_store = {k: doc_store[k] for k in matched_keys}
        for k in matched_keys:
            doc_store[k][iso2] = lang_matches[k]
        logger.info(f"After {iso3.upper()} ({iso2}): {len(doc_store)} 12-way candidates remain.")

    logger.info(f"Total strict 12-way documents found in train split: {len(doc_store)}")

    # Filter out documents already in DB
    all_indic_iso2 = [PRALEKHA_LANG_MAP[k] for k in ["ori", "kan", "guj", "mal", "ben", "pan", "tel", "mar", "tam", "urd", "hin"]]
    new_docs = []
    for s, trans in doc_store.items():
        chunks = split_into_coherent_chunks(s, max_chars=200)[:10]
        if chunks:
            h = hashlib.md5(" ".join(chunks).encode("utf-8")).hexdigest()
            if h not in existing_hashes:
                new_docs.append((s, trans, chunks))

    logger.info(f"Brand-new 12-way documents ready to insert: {len(new_docs)}")

    # Batch insert into SQLite
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = WAL;")

    next_doc_idx = total_existing + 1
    inserted_count = 0
    pairs_count = 0

    for en_text, trans_map, en_chunks in new_docs:
        doc_id = f"pralekha_12way_doc_{next_doc_idx:05d}"
        next_doc_idx += 1

        first_line = en_text.split("\n")[0][:120].strip()
        title = first_line if first_line else f"Pralekha Multi-lingual Document {next_doc_idx}"
        domain = classify_domain(en_text)

        # Chunk Indic translations
        lang_chunks: Dict[str, List[str]] = {}
        valid = True
        for lang in all_indic_iso2:
            tgt_text = trans_map.get(lang, "")
            t_chunks = split_into_coherent_chunks(tgt_text, max_chars=200)
            if not t_chunks:
                valid = False
                break
            lang_chunks[lang] = t_chunks

        if not valid:
            continue

        min_c = min(len(en_chunks), min(len(lang_chunks[l]) for l in all_indic_iso2))
        effective_chunks = min(min_c, 10)

        matrix: List[Dict[str, str]] = []
        for c in range(effective_chunks):
            row = {"en": en_chunks[c]}
            for l in all_indic_iso2:
                row[l] = lang_chunks[l][c]
            matrix.append(row)

        if matrix:
            added = insert_strict_document(
                doc_id=doc_id,
                title=title,
                domain=domain,
                sentences_matrix=matrix,
                source_portal="AI4Bharat Pralekha (Train Split)",
                db_path=db_file,
                conn=conn
            )
            if added > 0:
                inserted_count += 1
                pairs_count += added

        if inserted_count % 100 == 0:
            conn.commit()
            logger.info(f"[Stage 1 Progress] {inserted_count}/{len(new_docs)} documents inserted ({pairs_count} pairs)... Total in DB: {total_existing + inserted_count}")

    conn.commit()
    conn.close()
    logger.info(f"=== Stage 1 Complete! Total documents now: {total_existing + inserted_count} ===")
    return total_existing + inserted_count


def run_stage2_alignable_ingestion(
    current_docs: int,
    target_docs: int = 10000,
    db_path: Optional[Path] = None
) -> int:
    """
    Stage 2: Streams and intersects alignable shards by n_id to reach target_docs.
    """
    db_file = db_path or PRALEKHA_STRICT_DB_PATH
    needed = target_docs - current_docs
    if needed <= 0:
        logger.info(f"Target of {target_docs} documents already reached ({current_docs}). Skipping Stage 2.")
        return current_docs

    logger.info(f"=== Starting Stage 2: Alignable Subset Ingestion (Need {needed} more docs) ===")
    conn = get_db_connection(db_file)
    _, existing_hashes = get_existing_doc_hashes(conn)

    # Load cached alignable Odia (ori)
    p_ori = hf_hub_download(repo_id="ai4bharat/Pralekha", filename="alignable/ori-00000-of-00001.parquet", repo_type="dataset")
    t_ori = pq.read_table(p_ori)
    
    # Store candidates: n_id -> {lang: text}
    doc_store: Dict[str, Dict[str, str]] = {}
    for nid, txt in zip(t_ori["n_id"].to_pylist(), t_ori["text"].to_pylist()):
        t = txt.strip()
        if t and is_valid_indic_text(t, "or"):
            doc_store[nid] = {"or": t}

    logger.info(f"Stage 2: Initial Odia alignable candidates: {len(doc_store)}")

    # Sequential intersection by n_id
    ordered_langs = ["mal", "kan", "guj", "ben", "mar", "pan", "tam", "tel", "urd", "eng", "hin"]
    for lang_code in ordered_langs:
        shards_count = ALIGNABLE_SHARDS[lang_code]
        iso2 = PRALEKHA_LANG_MAP.get(lang_code, lang_code)
        lang_matches: Dict[str, str] = {}
        cand_keys = set(doc_store.keys())

        for s_idx in range(shards_count):
            fname = f"alignable/{lang_code}-{s_idx:05d}-of-{shards_count:05d}.parquet"
            logger.info(f"Stage 2: Fetching {fname}...")
            fpath = hf_hub_download(repo_id="ai4bharat/Pralekha", filename=fname, repo_type="dataset")
            table = pq.read_table(fpath)
            for nid, txt in zip(table["n_id"].to_pylist(), table["text"].to_pylist()):
                if nid in cand_keys and nid not in lang_matches:
                    t = txt.strip()
                    if iso2 == "en":
                        if any("a" <= ch.lower() <= "z" for ch in t):
                            lang_matches[nid] = t
                    else:
                        if is_valid_indic_text(t, iso2):
                            lang_matches[nid] = t

        matched_keys = set(lang_matches.keys())
        doc_store = {k: doc_store[k] for k in matched_keys}
        for k in matched_keys:
            doc_store[k][iso2] = lang_matches[k]
        logger.info(f"Stage 2: After {lang_code.upper()} ({iso2}): {len(doc_store)} 12-way candidates remain.")

    logger.info(f"Stage 2: Total 12-way alignable documents available: {len(doc_store)}")

    # Batch insert needed documents
    all_indic_iso2 = [PRALEKHA_LANG_MAP[k] for k in ["ori", "kan", "guj", "mal", "ben", "pan", "tel", "mar", "tam", "urd", "hin"]]
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = WAL;")

    next_doc_idx = current_docs + 1
    inserted_count = 0
    pairs_count = 0

    for nid, trans_map in doc_store.items():
        if current_docs + inserted_count >= target_docs:
            break

        en_text = trans_map.get("en", "")
        if not en_text:
            continue

        en_chunks = split_into_coherent_chunks(en_text, max_chars=200)[:10]
        if not en_chunks:
            continue

        h = hashlib.md5(" ".join(en_chunks).encode("utf-8")).hexdigest()
        if h in existing_hashes:
            continue
        existing_hashes.add(h)

        doc_id = f"pralekha_12way_doc_{next_doc_idx:05d}"
        next_doc_idx += 1

        first_line = en_text.split("\n")[0][:120].strip()
        title = first_line if first_line else f"Pralekha Multi-lingual Document {next_doc_idx} ({nid})"
        domain = classify_domain(en_text)

        lang_chunks: Dict[str, List[str]] = {}
        valid = True
        for lang in all_indic_iso2:
            tgt_text = trans_map.get(lang, "")
            t_chunks = split_into_coherent_chunks(tgt_text, max_chars=200)
            if not t_chunks:
                valid = False
                break
            lang_chunks[lang] = t_chunks

        if not valid:
            continue

        min_c = min(len(en_chunks), min(len(lang_chunks[l]) for l in all_indic_iso2))
        effective_chunks = min(min_c, 10)

        matrix: List[Dict[str, str]] = []
        for c in range(effective_chunks):
            row = {"en": en_chunks[c]}
            for l in all_indic_iso2:
                row[l] = lang_chunks[l][c]
            matrix.append(row)

        if matrix:
            added = insert_strict_document(
                doc_id=doc_id,
                title=title,
                domain=domain,
                sentences_matrix=matrix,
                source_portal="AI4Bharat Pralekha (Alignable Subset)",
                db_path=db_file,
                conn=conn
            )
            if added > 0:
                inserted_count += 1
                pairs_count += added

        if inserted_count % 100 == 0:
            conn.commit()
            logger.info(f"[Stage 2 Progress] Ingested {inserted_count}/{needed} docs ({pairs_count} pairs)... Total DB: {current_docs + inserted_count}")

    # Re-build indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pairs_lookup ON training_pairs (source_lang, target_lang, source_text);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sentences_en ON sentences_aligned (en);")
    conn.commit()
    conn.close()

    total_final = current_docs + inserted_count
    logger.info(f"=== Stage 2 Finished! Final Total Documents in Database: {total_final} ===")
    return total_final


def scale_database_to_target(target_documents: int = 10000, db_path: Optional[Path] = None) -> Dict[str, Any]:
    db_file = db_path or PRALEKHA_STRICT_DB_PATH
    logger.info(f"========== SCALING DATABASE TO {target_documents} DOCUMENTS ==========")
    
    # Run Stage 1 (Local Train)
    after_stage1 = run_stage1_train_ingestion(target_docs=target_documents, db_path=db_file)
    
    # Run Stage 2 (Alignable Streaming) if needed
    final_docs = run_stage2_alignable_ingestion(current_docs=after_stage1, target_docs=target_documents, db_path=db_file)

    summary = get_strict_database_summary(db_file)
    logger.info("========== DATABASE SCALING COMPLETED SUCCESSFULLY ==========")
    logger.info(f"Final Total Documents: {summary['total_documents']}")
    logger.info(f"Final Aligned Sentences: {summary['total_aligned_sentences']}")
    logger.info(f"Final Direct Pivot-Free Training Pairs: {summary['total_training_pairs']}")

    summary_file = PRALEKHA_STRICT_DIR / "database_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    scale_database_to_target(target_documents=target)
