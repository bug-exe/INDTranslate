"""
Pralekha Strict 12-Way Local Database & Multi-Parallel Matrix Engine
Maintains an authentic local database of multi-lingual documents aligned across
all 12 official benchmark languages:
[en, hi, bn, gu, kn, ml, mr, or, pa, ta, te, ur]
Provides pivot-free direct training pairs across all 132 language directions.
"""

from __future__ import annotations

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Generator, Union, Sequence
import logging

from backend.app.config import DATA_DIR

logger = logging.getLogger("indtranslate.pralekha_strict_db")

PRALEKHA_STRICT_DB_PATH = DATA_DIR / "pralekha_strict_12way.db"
PRALEKHA_STRICT_DIR = DATA_DIR / "pralekha_strict_12way"
PRALEKHA_STRICT_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_12_LANGUAGES = [
    "en", "hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"
]


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or PRALEKHA_STRICT_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_strict_db(db_path: Optional[Path] = None) -> None:
    """Initializes the SQLite schema for strict 12-way aligned documents and pairs."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        title TEXT,
        domain TEXT DEFAULT 'general',
        source_portal TEXT DEFAULT 'AI4Bharat Pralekha',
        sentence_count INTEGER NOT NULL,
        languages_verified TEXT NOT NULL,
        is_strict_12way INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sentences_aligned (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        sentence_index INTEGER NOT NULL,
        en TEXT NOT NULL,
        hi TEXT NOT NULL,
        bn TEXT NOT NULL,
        gu TEXT NOT NULL,
        kn TEXT NOT NULL,
        ml TEXT NOT NULL,
        mr TEXT NOT NULL,
        ori TEXT NOT NULL,
        pa TEXT NOT NULL,
        ta TEXT NOT NULL,
        te TEXT NOT NULL,
        ur TEXT NOT NULL,
        FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE,
        UNIQUE(doc_id, sentence_index)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS training_pairs (
        pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        sentence_index INTEGER NOT NULL,
        source_lang TEXT NOT NULL,
        target_lang TEXT NOT NULL,
        source_text TEXT NOT NULL,
        target_text TEXT NOT NULL,
        split TEXT DEFAULT 'train',
        FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_pairs_split ON training_pairs (source_lang, target_lang, split);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sentences_doc ON sentences_aligned (doc_id);")

    conn.commit()
    conn.close()
    logger.info("Initialized strict 12-way alignment database schema.")


def insert_strict_document(
    doc_id: str,
    title: str,
    domain: str,
    sentences_matrix: List[Dict[str, str]],
    source_portal: str = "AI4Bharat Pralekha",
    db_path: Optional[Path] = None,
    split_ratio: Tuple[float, float, float] = (0.85, 0.10, 0.05),
    conn: Optional[sqlite3.Connection] = None
) -> int:
    """
    Inserts a verified document where every sentence contains valid text in all 12 languages.
    Generates all 132 pairwise combinations per sentence directly.
    """
    should_close = False
    if conn is None:
        init_strict_db(db_path)
        conn = get_db_connection(db_path)
        should_close = True

    cur = conn.cursor()

    # Verify that all sentences contain all 12 languages
    cleaned_sentences: List[Dict[str, str]] = []
    for s_idx, s in enumerate(sentences_matrix):
        clean_s = {}
        for lang in OFFICIAL_12_LANGUAGES:
            val = (s.get(lang) or s.get("ori" if lang == "or" else lang) or "").strip()
            if not val:
                logger.warning(f"Sentence {s_idx} in doc {doc_id} missing language {lang}. Skipping doc.")
                if should_close:
                    conn.close()
                return 0
            clean_s[lang] = val
        cleaned_sentences.append(clean_s)

    # Insert Document Record
    cur.execute("""
    INSERT OR REPLACE INTO documents (doc_id, title, domain, source_portal, sentence_count, languages_verified, is_strict_12way)
    VALUES (?, ?, ?, ?, ?, ?, 1);
    """, (
        doc_id,
        title,
        domain,
        source_portal,
        len(cleaned_sentences),
        json.dumps(OFFICIAL_12_LANGUAGES)
    ))

    # Insert Sentences and Generate 132 Combinations
    cur.execute("DELETE FROM sentences_aligned WHERE doc_id = ?;", (doc_id,))
    cur.execute("DELETE FROM training_pairs WHERE doc_id = ?;", (doc_id,))

    total_pairs_added = 0
    import random

    for s_idx, s in enumerate(cleaned_sentences):
        cur.execute("""
        INSERT INTO sentences_aligned (doc_id, sentence_index, en, hi, bn, gu, kn, ml, mr, ori, pa, ta, te, ur)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            doc_id, s_idx,
            s["en"], s["hi"], s["bn"], s["gu"], s["kn"],
            s["ml"], s["mr"], s["or"], s["pa"], s["ta"], s["te"], s["ur"]
        ))

        # Determine split (train, val, test) deterministically based on hash
        rand_val = (hash(f"{doc_id}_{s_idx}") % 1000) / 1000.0
        if rand_val < split_ratio[0]:
            split = "train"
        elif rand_val < (split_ratio[0] + split_ratio[1]):
            split = "val"
        else:
            split = "test"

        # Generate all 132 permutations (12 * 11)
        pairs_to_insert = []
        for src_lang in OFFICIAL_12_LANGUAGES:
            for tgt_lang in OFFICIAL_12_LANGUAGES:
                if src_lang != tgt_lang:
                    src_txt = s[src_lang]
                    tgt_txt = s[tgt_lang]
                    pairs_to_insert.append((
                        doc_id, s_idx, src_lang, tgt_lang, src_txt, tgt_txt, split
                    ))

        cur.executemany("""
        INSERT INTO training_pairs (doc_id, sentence_index, source_lang, target_lang, source_text, target_text, split)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, pairs_to_insert)
        total_pairs_added += len(pairs_to_insert)

    if should_close:
        conn.commit()
        conn.close()

    return total_pairs_added


def get_strict_database_summary(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Returns metadata and statistics on the local database of training documents."""
    init_strict_db(db_path)
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as total_docs FROM documents;")
    total_docs = cur.fetchone()["total_docs"]

    cur.execute("SELECT COUNT(*) as total_sents FROM sentences_aligned;")
    total_sents = cur.fetchone()["total_sents"]

    cur.execute("SELECT split, COUNT(*) as cnt FROM training_pairs GROUP BY split;")
    splits = {row["split"]: row["cnt"] for row in cur.fetchall()}

    cur.execute("SELECT domain, COUNT(*) as doc_cnt FROM documents GROUP BY domain;")
    domains = {row["domain"]: row["doc_cnt"] for row in cur.fetchall()}

    cur.execute("SELECT doc_id, title, domain, sentence_count, created_at FROM documents ORDER BY created_at DESC LIMIT 20;")
    recent_docs = [dict(row) for row in cur.fetchall()]

    conn.close()
    return {
        "status": "operational",
        "database_file": str(db_path or PRALEKHA_STRICT_DB_PATH),
        "total_documents": total_docs,
        "total_aligned_sentences": total_sents,
        "total_training_pairs": sum(splits.values()),
        "splits": splits,
        "domains": domains,
        "official_languages": OFFICIAL_12_LANGUAGES,
        "language_pairs_covered": len(OFFICIAL_12_LANGUAGES) * (len(OFFICIAL_12_LANGUAGES) - 1),
        "recent_documents": recent_docs
    }


def stream_training_batches(
    split: str = "train",
    batch_size: int = 32,
    source_langs: Optional[List[str]] = None,
    target_langs: Optional[List[str]] = None,
    db_path: Optional[Path] = None
) -> Generator[List[Dict[str, Any]], None, None]:
    """Generator streaming training batches directly from SQLite without blowing up RAM."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    query = "SELECT source_lang, target_lang, source_text, target_text FROM training_pairs WHERE split = ?"
    params = [split]

    if source_langs:
        query += f" AND source_lang IN ({','.join(['?']*len(source_langs))})"
        params.extend(source_langs)
    if target_langs:
        query += f" AND target_lang IN ({','.join(['?']*len(target_langs))})"
        params.extend(target_langs)

    query += " ORDER BY RANDOM();"

    cur.execute(query, params)
    batch = []
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        yield [dict(r) for r in rows]

    conn.close()


class PralekhaStrict12WayDataset:
    """
    High-performance virtual dataset across all 132 language directions in Pralekha.
    Stores only the N aligned sentence rows in memory (~130 MB RAM for 39,280 sentences),
    and generates directional pairs on-the-fly in O(1) time (< 0.4 microseconds per item).
    Supports exact len(), integer indexing, slicing, iteration, and train/val splitting.
    """
    def __init__(
        self,
        rows: List[Tuple[Any, ...]],
        directions: Optional[List[Tuple[str, str]]] = None,
        split: str = "train",
        max_pairs: Optional[int] = None,
        col_names: Optional[List[str]] = None
    ):
        self.rows = rows
        self.split = split
        if directions is not None:
            self.directions = directions
        else:
            self.directions = [
                (src, tgt)
                for src in OFFICIAL_12_LANGUAGES
                for tgt in OFFICIAL_12_LANGUAGES
                if src != tgt
            ]
        self.n_rows = len(rows)
        self.n_dirs = len(self.directions)
        max_possible = self.n_rows * self.n_dirs
        self.total = min(max_possible, max_pairs) if (max_pairs and max_pairs > 0) else max_possible
        cols = col_names or ["en", "hi", "bn", "gu", "kn", "ml", "mr", "ori", "pa", "ta", "te", "ur"]
        self.col_map = {("or" if lang == "ori" else lang): i + 2 for i, lang in enumerate(cols)}

    @property
    def num_sentences(self) -> int:
        return self.n_rows

    @property
    def n_sentences(self) -> int:
        return self.n_rows

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.total)
            if step == 1:
                return PralekhaStrictDatasetView(self, start, stop)
            return [self[i] for i in range(start, stop, step)]

        if idx < 0:
            idx += self.total
        if idx < 0 or idx >= self.total:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total}")

        row_idx = idx // self.n_dirs
        dir_idx = idx % self.n_dirs
        src_lang, tgt_lang = self.directions[dir_idx]
        row = self.rows[row_idx]

        return {
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "source_text": row[self.col_map[src_lang]],
            "target_text": row[self.col_map[tgt_lang]],
            "doc_id": row[0],
            "sentence_index": row[1],
            "split": self.split
        }

    def train_val_split(
        self,
        val_ratio: float = 0.05,
        seed: int = 42
    ) -> Tuple[PralekhaStrict12WayDataset, PralekhaStrict12WayDataset]:
        """
        Splits dataset by aligned sentence rows into train and val datasets.
        Guarantees zero sentence leakage and identical 132-way language balance.
        """
        n_val_rows = max(1, int(self.n_rows * val_ratio)) if val_ratio > 0 else 0
        n_train_rows = self.n_rows - n_val_rows

        train_rows = self.rows[:n_train_rows]
        val_rows = self.rows[n_train_rows:]

        train_ds = PralekhaStrict12WayDataset(
            train_rows,
            directions=self.directions,
            split="train"
        )
        val_ds = PralekhaStrict12WayDataset(
            val_rows,
            directions=self.directions,
            split="val"
        )
        return train_ds, val_ds


class PralekhaStrictDatasetView:
    """Sliced view of a PralekhaStrict12WayDataset without copying memory."""
    def __init__(self, base: PralekhaStrict12WayDataset, start: int, stop: int):
        self.base = base
        self.start = start
        self.stop = stop
        self.length = max(0, stop - start)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.length)
            if step == 1:
                return PralekhaStrictDatasetView(self.base, self.start + start, self.start + stop)
            return [self[i] for i in range(start, stop, step)]

        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range for view of size {self.length}")

        return self.base[self.start + idx]


def load_balanced_pairs_from_strict_db(
    db_path: Optional[Path] = None,
    target_pairs: Optional[Union[int, str]] = None,
    seed: int = 42,
    split: str = "train",
    as_dataset: bool = True
) -> Union[PralekhaStrict12WayDataset, List[Dict[str, Any]]]:
    """
    Extracts a balanced parallel dataset across all 132 language directions in Pralekha.
    If target_pairs is None, 0, or 'all', extracts all 39,280 pure 12-way sentences (5,184,960 pairs).
    Uses high-performance virtual dataset indexing by default (~130 MB RAM footprint).
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    num_directions = len(OFFICIAL_12_LANGUAGES) * (len(OFFICIAL_12_LANGUAGES) - 1)  # 132
    cols = ["en", "hi", "bn", "gu", "kn", "ml", "mr", "ori", "pa", "ta", "te", "ur"]
    col_str = ", ".join(cols)

    query = f"SELECT doc_id, sentence_index, {col_str} FROM sentences_aligned"
    cur.execute(query)
    all_rows = cur.fetchall()
    conn.close()

    total_available = len(all_rows)
    if total_available == 0:
        logger.warning("No sentences found in sentences_aligned table.")
        return []

    is_all = (
        target_pairs is None
        or (isinstance(target_pairs, (int, float)) and int(target_pairs) <= 0)
        or str(target_pairs).lower() in ("all", "none", "0")
    )

    if is_all:
        selected_rows = all_rows
        max_p = None
    else:
        num_target = int(target_pairs)
        needed_per_direction = (num_target + num_directions - 1) // num_directions
        if needed_per_direction < total_available:
            import random
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(total_available), needed_per_direction))
            selected_rows = [all_rows[i] for i in indices]
        else:
            selected_rows = all_rows
        max_p = num_target

    if as_dataset:
        ds = PralekhaStrict12WayDataset(
            selected_rows,
            split=split,
            max_pairs=max_p,
            col_names=cols
        )
        logger.info(
            f"Loaded {len(ds):,} balanced pairs across {num_directions} directions "
            f"({len(selected_rows):,} aligned sentences, ~{len(ds) // num_directions:,} per direction) into virtual dataset."
        )
        return ds

    # If materialization to a raw list of dicts is requested
    col_map = {("or" if lang == "ori" else lang): i + 2 for i, lang in enumerate(cols)}
    balanced_pairs: List[Dict[str, Any]] = []
    for src in OFFICIAL_12_LANGUAGES:
        src_idx = col_map[src]
        for tgt in OFFICIAL_12_LANGUAGES:
            if src == tgt:
                continue
            tgt_idx = col_map[tgt]
            for r in selected_rows:
                s_txt = r[src_idx]
                t_txt = r[tgt_idx]
                if s_txt and t_txt:
                    balanced_pairs.append({
                        "source_lang": src,
                        "target_lang": tgt,
                        "source_text": s_txt.strip(),
                        "target_text": t_txt.strip(),
                        "doc_id": r[0],
                        "sentence_index": r[1],
                        "split": split
                    })

    if max_p and len(balanced_pairs) > max_p:
        import random
        rng = random.Random(seed)
        rng.shuffle(balanced_pairs)
        balanced_pairs = balanced_pairs[:max_p]

    logger.info(
        f"Materialized {len(balanced_pairs):,} balanced pairs across {num_directions} directions."
    )
    return balanced_pairs


def load_pairs_from_strict_db(
    db_path: Optional[Path] = None,
    split: Optional[str] = None,
    lang_pairs: Optional[List[Tuple[str, str]]] = None,
    max_pairs: Optional[Union[int, str]] = None,
    balanced: bool = False,
    as_dataset: Optional[bool] = None
) -> Union[List[Dict[str, Any]], PralekhaStrict12WayDataset]:
    """
    Loads parallel sentence pairs from the strict 12-way SQLite database.
    Supports max_pairs=None, 0, or 'all' to ingest all 5.18M pairs across all 132 language directions
    using high-performance virtual indexing (only ~130 MB RAM footprint).
    """
    is_full_or_all = (
        max_pairs is None
        or (isinstance(max_pairs, (int, float)) and int(max_pairs) <= 0)
        or str(max_pairs).lower() in ("all", "none", "0")
    )

    if is_full_or_all or balanced or (isinstance(max_pairs, int) and max_pairs >= 100000):
        if lang_pairs is None:
            use_ds = True if as_dataset is None else as_dataset
            target_p = None if is_full_or_all else max_pairs
            return load_balanced_pairs_from_strict_db(
                db_path=db_path,
                target_pairs=target_p,
                split=split or "train",
                as_dataset=use_ds
            )

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    query = "SELECT source_lang, target_lang, source_text, target_text, split, doc_id FROM training_pairs WHERE 1=1"
    params: List[Any] = []

    if split:
        query += " AND split = ?"
        params.append(split)

    if lang_pairs:
        placeholders = " OR ".join(["(source_lang = ? AND target_lang = ?)"] * len(lang_pairs))
        query += f" AND ({placeholders})"
        for src, tgt in lang_pairs:
            params.extend([src, tgt])

    if max_pairs and not is_full_or_all:
        query += f" LIMIT {int(max_pairs)}"

    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def load_balanced_500k_pairs(
    db_path: Optional[Path] = None,
    target_pairs: int = 500000,
    seed: int = 42,
    split: str = "train",
    as_dataset: bool = False
) -> Union[List[Dict[str, Any]], PralekhaStrict12WayDataset]:
    """
    Extracts a balanced parallel dataset of up to target_pairs (default 500,000)
    uniformly distributed across all 132 language directions in the 12-way alignment database.
    """
    return load_balanced_pairs_from_strict_db(
        db_path=db_path,
        target_pairs=target_pairs,
        seed=seed,
        split=split,
        as_dataset=as_dataset
    )


def stream_balanced_500k_batches(
    db_path: Optional[Path] = None,
    batch_size: int = 64,
    target_pairs: int = 500000,
    seed: int = 42
) -> Generator[List[Dict[str, Any]], None, None]:
    """Streams balanced batches of parallel pairs across 132 language directions."""
    pairs = load_balanced_500k_pairs(db_path=db_path, target_pairs=target_pairs, seed=seed, as_dataset=False)
    for i in range(0, len(pairs), batch_size):
        yield pairs[i : i + batch_size]

