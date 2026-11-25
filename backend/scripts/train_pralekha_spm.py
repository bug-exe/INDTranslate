"""
Train SentencePiece BPE Tokenizer for AI4Bharat Pralekha
=========================================================
Trains a 16,000-subword SentencePiece BPE model covering all 12 benchmark
languages (en, hi, bn, gu, kn, ml, mr, or, pa, ta, te, ur) using the authentic
strictly-aligned multi-parallel sentences from backend/data/pralekha_strict_12way.db.

Outputs:
  - backend/saved_models/pralekha/sentencepiece.bpe.model
  - backend/saved_models/pralekha/sentencepiece.bpe.vocab
  - backend/saved_models/pralekha/spm_vocab.json
"""

import os
import sys
import json
import sqlite3
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import sentencepiece as spm

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "backend" / "data"
MODELS_DIR = BASE_DIR / "backend" / "saved_models" / "pralekha"
STRICT_DB_PATH = DATA_DIR / "pralekha_strict_12way.db"

# 12 Benchmark Languages in Pralekha
OFFICIAL_12_LANGUAGES = [
    "en", "hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"
]
DB_COLS = [
    "en", "hi", "bn", "gu", "kn", "ml", "mr", "ori", "pa", "ta", "te", "ur"
]

# All 22 Scheduled Indian Languages + English
ALL_22_SCHEDULED_LANGUAGES = [
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai",
    "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"
]
ALL_LANGUAGES = ALL_22_SCHEDULED_LANGUAGES + ["en"]

# User defined symbols: mask token + target language routing tokens (<2en>, <2hi>, etc.)
MASK_TOKEN = "<mask" + ">"
USER_DEFINED_SYMBOLS = [MASK_TOKEN] + [f"<2{lang}>" for lang in sorted(ALL_LANGUAGES)]


def export_corpus_from_db(db_path: Path, output_file: Path, max_rows: Optional[int] = None) -> int:
    """Streams all sentences from sentences_aligned into a plain text corpus file."""
    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    query = f"SELECT {','.join(DB_COLS)} FROM sentences_aligned"
    if max_rows:
        query += f" LIMIT {int(max_rows)}"

    cur.execute(query)
    sentence_count = 0
    total_chars = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for row in cur:
            for text in row:
                if text:
                    clean = text.strip()
                    if len(clean) > 1:
                        out_f.write(clean + "\n")
                        sentence_count += 1
                        total_chars += len(clean)

    conn.close()
    print(f"✓ Exported {sentence_count:,} sentences ({total_chars / 1e6:.1f} MB) to {output_file}")
    return sentence_count


def train_sentencepiece_bpe(
    corpus_file: Path,
    model_prefix: Path,
    vocab_size: int = 16000,
    character_coverage: float = 0.9995,
    num_threads: int = 8
) -> Path:
    """Trains a 16,000 vocab SentencePiece BPE tokenizer."""
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"Training SentencePiece BPE (vocab_size={vocab_size}, coverage={character_coverage})...")
    print(f"User-defined symbols ({len(USER_DEFINED_SYMBOLS)}): {USER_DEFINED_SYMBOLS[:6]} ... {USER_DEFINED_SYMBOLS[-4:]}")

    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        byte_fallback=True,
        character_coverage=character_coverage,
        normalization_rule_name="nmt_nfkc",
        add_dummy_prefix=False,
        num_threads=num_threads,
        max_sentence_length=4192,
        hard_vocab_limit=False
    )

    model_file = Path(f"{model_prefix}.model")
    print(f"✓ SentencePiece model created at: {model_file}")
    return model_file


def export_spm_vocab(model_file: Path, json_output_file: Path) -> Dict[str, int]:
    """Extracts piece-to-id mapping from trained SentencePiece model and writes spm_vocab.json."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    vocab_size = sp.get_piece_size()
    vocab: Dict[str, int] = {}
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        vocab[piece] = i

    json_output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"✓ Exported vocabulary ({vocab_size:,} tokens) to {json_output_file}")
    return vocab


def compute_vocab_statistics(model_file: Path, vocab: Dict[str, int]) -> Dict[str, Any]:
    """Analyzes vocabulary token distribution by script and category."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    script_counts: Dict[str, int] = {}
    special_count = 0
    routing_count = 0
    byte_count = 0
    subword_count = 0

    for piece, tid in vocab.items():
        if tid in (0, 1, 2, 3, 4):
            special_count += 1
            script_counts["SPECIAL"] = script_counts.get("SPECIAL", 0) + 1
        elif piece.startswith("<2") and piece.endswith(">"):
            routing_count += 1
            script_counts["ROUTING"] = script_counts.get("ROUTING", 0) + 1
        elif piece.startswith("<0x") and piece.endswith(">"):
            byte_count += 1
            script_counts["BYTE_FALLBACK"] = script_counts.get("BYTE_FALLBACK", 0) + 1
        else:
            subword_count += 1
            # Classify script of primary character
            clean = piece.lstrip(" ").lstrip("▁")
            detected_script = "OTHER"
            if clean:
                for ch in clean:
                    name = unicodedata.name(ch, "")
                    if name:
                        first_word = name.split()[0]
                        if first_word in (
                            "DEVANAGARI", "TAMIL", "BENGALI", "TELUGU", "KANNADA",
                            "MALAYALAM", "GUJARATI", "GURMUKHI", "ORIYA", "ARABIC", "LATIN"
                        ):
                            detected_script = first_word
                            break
            script_counts[detected_script] = script_counts.get(detected_script, 0) + 1

    stats = {
        "vocab_size": sp.get_piece_size(),
        "special_tokens": special_count,
        "routing_tokens": routing_count,
        "byte_fallback_tokens": byte_count,
        "subword_tokens": subword_count,
        "script_distribution": script_counts
    }
    return stats


def verify_combining_mark_integrity(model_file: Path) -> bool:
    """Verifies that combining marks and routing tokens decode with complete integrity."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    test_phrases = {
        "Hindi": "नमस्ते भारत। यह एक सुंदर देश है।",
        "Tamil": "வணக்கம் தமிழ்நாடு! இது ஒரு அழகான நாடு.",
        "Bengali": "নমস্কার ভারত! এটি একটি সুন্দর দেশ।",
        "Telugu": "నమస్కారం భారతదేశం! ఇది ఒక అందమైన దేశం.",
        "Kannada": "ನಮಸ್ಕಾರ ಭಾರತ! ಇದು ಒಂದು ಸುಂದರ ದೇಶ.",
        "Malayalam": "നമസ്കാരം ഭാരതം! ഇത് മനോഹരമായ രാജ്യം.",
        "Gujarati": "નમસ્તે ભારત! આ એક સુંદર દેશ છે.",
        "Punjabi": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਭਾਰਤ! ਇਹ ਇੱਕ ਸੁੰਦਰ ਦੇਸ਼ ਹੈ।",
        "Odia": "ନମସ୍କାର ଭାରତ! ଏହା ଏକ ସୁନ୍ଦର ଦେଶ।",
        "Urdu": "سلام ہندوستان! یہ ایک خوبصورت ملک ہے۔",
        "English": "Hello India! This is a beautiful country."
    }

    all_passed = True
    print("\n--- Combining Mark & Detokenization Verification ---")
    for lang, phrase in test_phrases.items():
        pieces = sp.encode_as_pieces(phrase)
        ids = sp.encode_as_ids(phrase)
        decoded = sp.decode_ids(ids)

        roundtrip_match = (decoded.strip() == phrase.strip())
        status = "✓ PASS" if roundtrip_match else "✗ FAIL"
        if not roundtrip_match:
            all_passed = False

        print(f"[{status}] {lang:10s} | Tokens: {len(pieces):2d} | Decoded: '{decoded[:35]}...'")

    # Routing tokens verification: each must encode directly to its single piece ID
    print("\n--- Routing Tokens Verification ---")
    for lang in OFFICIAL_12_LANGUAGES:
        rtok = f"<2{lang}>"
        ids = sp.encode_as_ids(rtok)
        expected_id = sp.piece_to_id(rtok)
        assert len(ids) == 1 and ids[0] == expected_id, (
            f"Routing token {rtok} failed: ids={ids}, expected={expected_id}"
        )
        print(f"✓ {rtok:6s} -> ID {expected_id}")

    # Special tokens verification
    assert sp.piece_to_id("<pad>") == 0
    assert sp.piece_to_id("<unk>") == 1
    assert sp.piece_to_id("<bos>") == 2
    assert sp.piece_to_id("<eos>") == 3
    assert sp.piece_to_id(MASK_TOKEN) == 4
    print("✓ Special tokens (<pad>=0, <unk>=1, <bos>=2, <eos>=3, <mask >=4) verified.")

    return all_passed


def main():
    print("=" * 70)
    print("AI4Bharat Pralekha: 16k SentencePiece BPE Tokenizer Builder")
    print("=" * 70)

    corpus_temp = tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt")
    corpus_path = Path(corpus_temp.name)
    corpus_temp.close()

    try:
        # Step 1: Export sentences from SQLite DB
        export_corpus_from_db(STRICT_DB_PATH, corpus_path)

        # Step 2: Train SentencePiece BPE
        model_prefix = MODELS_DIR / "sentencepiece.bpe"
        model_file = train_sentencepiece_bpe(
            corpus_file=corpus_path,
            model_prefix=model_prefix,
            vocab_size=16000,
            character_coverage=0.9995
        )

        # Step 3: Export spm_vocab.json
        spm_vocab_path = MODELS_DIR / "spm_vocab.json"
        vocab = export_spm_vocab(model_file, spm_vocab_path)

        # Step 4: Verify combining mark integrity
        passed = verify_combining_mark_integrity(model_file)
        if not passed:
            print("WARNING: Some verification tests encountered issues.")

        # Step 5: Vocabulary Statistics
        stats = compute_vocab_statistics(model_file, vocab)
        print("\n--- Vocabulary Statistics ---")
        print(f"Total Vocabulary Size: {stats['vocab_size']:,}")
        print(f"Special Tokens:        {stats['special_tokens']}")
        print(f"Routing Tokens:        {stats['routing_tokens']}")
        print(f"Byte Fallback Tokens:  {stats['byte_fallback_tokens']}")
        print(f"Subword Tokens:        {stats['subword_tokens']}")
        print("\nScript Distribution:")
        for script, cnt in sorted(stats["script_distribution"].items(), key=lambda x: -x[1]):
            print(f"  - {script:15s}: {cnt:5d} tokens ({cnt / stats['vocab_size'] * 100:.1f}%)")

        # Step 6: Update tokenizer_config.json
        config_path = MODELS_DIR / "tokenizer_config.json"
        config_data = {
            "model_type": "sentencepiece_bpe",
            "vocab_size": stats["vocab_size"],
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "mask_token": MASK_TOKEN,
            "pad_token_id": 0,
            "unk_token_id": 1,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "mask_token_id": 4,
            "supported_languages": ALL_LANGUAGES,
            "benchmark_12_languages": OFFICIAL_12_LANGUAGES,
            "spm_model_file": "sentencepiece.bpe.model",
            "spm_vocab_file": "spm_vocab.json",
            "script_distribution": stats["script_distribution"]
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"✓ Updated tokenizer config: {config_path}")

        print("\n" + "=" * 70)
        print("✓ SentencePiece 16,000 BPE Tokenizer training & validation complete!")
        print("=" * 70)

    finally:
        if corpus_path.exists():
            corpus_path.unlink()


if __name__ == "__main__":
    main()
