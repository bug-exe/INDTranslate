"""
Pralekha Strict 12-Way Multi-Parallel Dataset Extractor & Database Builder.
Downloads official AI4Bharat Pralekha N-way parallel splits, verifies 100% coverage
across all 12 benchmark languages:
[en, hi, bn, gu, kn, ml, mr, or, pa, ta, te, ur]
and indexes them directly into the local SQLite database (pralekha_strict_12way.db).
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import logging

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pralekha_extractor")

# ISO-3 to ISO-2 language code mapping for Pralekha
PRALEKHA_LANG_MAP = {
    "eng": "en",
    "hin": "hi",
    "ben": "bn",
    "guj": "gu",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "ori": "or",
    "pan": "pa",
    "tam": "ta",
    "tel": "te",
    "urd": "ur"
}

REVERSE_LANG_MAP = {v: k for k, v in PRALEKHA_LANG_MAP.items()}


def split_into_coherent_chunks(text: str, max_chars: int = 250) -> List[str]:
    """Splits a document into sentence/paragraph units based on punctuation."""
    if not text:
        return []
    # Split by double newline or sentence terminal punctuation
    raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks = []
    for line in raw_lines:
        if len(line) <= max_chars:
            chunks.append(line)
        else:
            # Sub-split on period or purna viram
            parts = re.split(r'([.?!।॥]\s+)', line)
            buf = ""
            for p in parts:
                buf += p
                if len(buf) >= 60 or p.strip() in (".", "?", "!", "।", "॥"):
                    if buf.strip():
                        chunks.append(buf.strip())
                    buf = ""
            if buf.strip():
                chunks.append(buf.strip())
    return chunks


def build_strict_12way_database(
    max_documents: int = 2000,
    splits: List[str] = ["dev", "test"],
    db_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Downloads Pralekha N-way splits, finds strict 12-way intersections,
    and populates the local SQLite database.
    """
    init_strict_db(db_path)
    logger.info(f"Extracting strict 12-way aligned documents from Pralekha {splits}...")

    # Step 1: Download and load all 11 Indic language files for the splits
    tables_by_lang: Dict[str, Dict[str, str]] = {}

    for split in splits:
        for iso3, iso2 in PRALEKHA_LANG_MAP.items():
            if iso2 == "en":
                continue
            filename = f"{split}/eng_{iso3}-00000-of-00001.parquet"
            logger.info(f"Loading {filename} from Hugging Face...")
            fpath = hf_hub_download(
                repo_id="ai4bharat/Pralekha",
                filename=filename,
                repo_type="dataset"
            )
            table = pq.read_table(fpath)
            src_list = table["src_txt"].to_pylist()
            tgt_list = table["tgt_txt"].to_pylist()

            if iso2 not in tables_by_lang:
                tables_by_lang[iso2] = {}

            for src, tgt in zip(src_list, tgt_list):
                norm_src = src.strip()
                if norm_src:
                    tables_by_lang[iso2][norm_src] = tgt.strip()

    # Step 2: Compute strict intersection across ALL 11 Indic languages
    all_indic_langs = [l for l in OFFICIAL_12_LANGUAGES if l != "en"]
    common_en_texts: Set[str] = set(tables_by_lang[all_indic_langs[0]].keys())
    for lang in all_indic_langs[1:]:
        common_en_texts = common_en_texts.intersection(set(tables_by_lang[lang].keys()))

    logger.info(f"✓ Found {len(common_en_texts)} documents present in ALL 12 languages simultaneously!")

    # Sort documents deterministically and take up to max_documents
    selected_docs = sorted(list(common_en_texts))[:max_documents]

    # Step 3: Align, chunk, and insert into local SQLite database using fast batching
    total_docs_inserted = 0
    total_pairs_generated = 0

    conn = get_db_connection(db_path)
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = WAL;")

    for idx, en_text in enumerate(selected_docs, 1):
        doc_id = f"pralekha_12way_doc_{idx:04d}"
        
        # Determine title from first line of English text
        first_line = en_text.split("\n")[0][:120].strip()
        title = first_line if first_line else f"Pralekha Multi-lingual Document {idx}"

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

        # Chunk English text
        en_chunks = split_into_coherent_chunks(en_text, max_chars=200)
        num_chunks = len(en_chunks)
        if num_chunks == 0:
            continue

        # Prepare sentence matrix for the 12 languages
        sentences_matrix: List[Dict[str, str]] = []
        
        # Get chunks for each of the 11 target languages
        lang_chunks: Dict[str, List[str]] = {}
        valid_alignment = True

        for lang in all_indic_langs:
            tgt_full = tables_by_lang[lang][en_text]
            t_chunks = split_into_coherent_chunks(tgt_full, max_chars=200)
            if len(t_chunks) == 0:
                valid_alignment = False
                break
            lang_chunks[lang] = t_chunks

        if not valid_alignment:
            continue

        # Align chunk-by-chunk proportionally
        min_chunks = min(num_chunks, min(len(lang_chunks[l]) for l in all_indic_langs))
        # Keep high quality chunks up to min_chunks (capped at 10 chunks per document)
        effective_chunks = min(min_chunks, 10)

        for c_idx in range(effective_chunks):
            s_dict = {"en": en_chunks[c_idx]}
            for lang in all_indic_langs:
                s_dict[lang] = lang_chunks[lang][c_idx]
            sentences_matrix.append(s_dict)

        if sentences_matrix:
            pairs_added = insert_strict_document(
                doc_id=doc_id,
                title=title,
                domain=domain,
                sentences_matrix=sentences_matrix,
                source_portal="AI4Bharat Pralekha (Official N-Way Benchmark)",
                db_path=db_path,
                conn=conn
            )
            if pairs_added > 0:
                total_docs_inserted += 1
                total_pairs_generated += pairs_added

        if total_docs_inserted % 100 == 0:
            conn.commit()
            logger.info(f"Progress: {total_docs_inserted}/{len(selected_docs)} documents inserted ({total_pairs_generated} pairs)...")

    conn.commit()
    conn.close()

    summary = get_strict_database_summary(db_path)
    logger.info("Extraction and Local Database population completed!")
    logger.info(f"Total Documents in DB: {summary['total_documents']}")
    logger.info(f"Total Aligned Sentences: {summary['total_aligned_sentences']}")
    logger.info(f"Total Direct Pivot-Free Training Pairs: {summary['total_training_pairs']}")

    # Save summary json to disk
    summary_file = PRALEKHA_STRICT_DIR / "database_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    # Extract all 1,998 strict 12-way documents from dev + test splits
    build_strict_12way_database(max_documents=2000, splits=["dev", "test"])
