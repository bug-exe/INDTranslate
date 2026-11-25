#!/usr/bin/env python3
"""
Data Cleaning Pipeline for Pralekha Strict 12-Way Database
===========================================================
Filters and removes degenerate rows that contain ONLY punctuation marks,
symbols, combining characters, and whitespace (zero letters/numbers), such as
lone '.', '...', '---', '???', '।', '॥', etc.

Cleans:
1. `sentences_aligned` table
2. `training_pairs` table (chunked batch deletion across 8.4M rows)
3. Vacuums and verifies SQLite database integrity.
"""

import sys
import time
import sqlite3
import unicodedata
from pathlib import Path

DB_PATH = Path("backend/data/pralekha_strict_12way.db")


def is_only_punct(text: str) -> int:
    """
    Returns 1 if the string contains NO letters and NO numbers (i.e. only
    punctuation, whitespace, control symbols), 0 if it contains valid lexical tokens.
    """
    if not text or not text.strip():
        return 1
    for ch in text:
        cat = unicodedata.category(ch)
        # 'L' = Letter, 'N' = Number
        if cat[0] == 'L' or cat[0] == 'N':
            return 0
    return 1


def clean_database(db_file: Path = DB_PATH):
    print("=" * 70)
    print("🧹 INDTRANSLATE: PRALEKHA DATABASE PUNCTUATION CLEANING PIPELINE")
    print("=" * 70)
    print(f"Target Database: {db_file}")

    if not db_file.exists():
        print(f"Error: {db_file} does not exist!")
        sys.exit(1)

    t_start = time.time()
    conn = sqlite3.connect(str(db_file), timeout=60.0)
    conn.create_function("is_only_punct", 1, is_only_punct)

    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA cache_size = -100000;")  # 100 MB cache

    # 1. Inspect initial counts
    init_docs = cur.execute("SELECT count(*) FROM documents").fetchone()[0]
    init_aligned = cur.execute("SELECT count(*) FROM sentences_aligned").fetchone()[0]
    init_pairs = cur.execute("SELECT count(*) FROM training_pairs").fetchone()[0]

    print("\n[1/4] Initial Database State:")
    print(f"  • documents:         {init_docs:,} rows")
    print(f"  • sentences_aligned: {init_aligned:,} rows")
    print(f"  • training_pairs:    {init_pairs:,} rows")

    # 2. Clean sentences_aligned
    print("\n[2/4] Cleaning `sentences_aligned` table...")
    t0 = time.time()
    cur.execute("""
        DELETE FROM sentences_aligned 
        WHERE is_only_punct(en) = 1 OR is_only_punct(hi) = 1
    """)
    del_aligned = cur.rowcount
    conn.commit()
    print(f"✓ Removed {del_aligned:,} punctuation-only rows from sentences_aligned in {time.time() - t0:.2f}s")

    # 3. Clean training_pairs in chunks
    print("\n[3/4] Cleaning `training_pairs` table in chunks of 250,000 IDs...")
    min_id = cur.execute("SELECT MIN(pair_id) FROM training_pairs").fetchone()[0] or 0
    max_id = cur.execute("SELECT MAX(pair_id) FROM training_pairs").fetchone()[0] or 0

    chunk_size = 250000
    current_start = min_id
    total_deleted_pairs = 0
    chunk_count = 0
    total_chunks = ((max_id - min_id) // chunk_size) + 1

    t0_pairs = time.time()
    while current_start <= max_id:
        current_end = current_start + chunk_size
        chunk_count += 1
        t_chunk = time.time()

        cur.execute("""
            DELETE FROM training_pairs
            WHERE pair_id >= ? AND pair_id < ?
              AND (is_only_punct(source_text) = 1 OR is_only_punct(target_text) = 1)
        """, (current_start, current_end))

        chunk_deleted = cur.rowcount
        total_deleted_pairs += chunk_deleted
        conn.commit()

        if chunk_count % 5 == 0 or current_end > max_id:
            elapsed = time.time() - t0_pairs
            pct = min(100.0, (chunk_count / total_chunks) * 100.0)
            print(f"  Chunk [{chunk_count}/{total_chunks}] ({pct:.1f}%) | "
                  f"IDs {current_start:,}-{current_end:,} | "
                  f"Deleted chunk: {chunk_deleted:,} | Cumulative deleted: {total_deleted_pairs:,} ({elapsed:.1f}s)")

        current_start = current_end

    print(f"✓ Total training_pairs removed: {total_deleted_pairs:,} rows ({time.time() - t0_pairs:.2f}s)")

    # 4. Final verification
    final_aligned = cur.execute("SELECT count(*) FROM sentences_aligned").fetchone()[0]
    final_pairs = cur.execute("SELECT count(*) FROM training_pairs").fetchone()[0]

    print("\n[4/4] Verification & Compaction:")
    print(f"  • sentences_aligned: {final_aligned:,} remaining (was {init_aligned:,}, -{del_aligned:,})")
    print(f"  • training_pairs:    {final_pairs:,} remaining (was {init_pairs:,}, -{total_deleted_pairs:,})")

    # Quick check integrity
    print("  Running PRAGMA quick_check...")
    qc = cur.execute("PRAGMA quick_check;").fetchone()[0]
    print(f"  PRAGMA quick_check: {qc}")

    conn.close()
    total_sec = time.time() - t_start
    print(f"\n🎉 Data cleaning completed successfully in {total_sec:.2f}s!")
    print(f"Cleaned {total_deleted_pairs + del_aligned:,} degenerate punctuation-only records in total.")


if __name__ == "__main__":
    clean_database()
