"""
Unit tests for AI4Bharat Pralekha Dataset Engineering Pipeline
(backend/app/services/pralekha_dataset.py)
"""

import tempfile
from pathlib import Path
import pytest
import torch

from backend.app.services.pralekha_dataset import (
    ALL_22_SCHEDULED_LANGUAGES,
    ALL_LANGUAGES,
    DOMAINS,
    ParallelSentencePair,
    PralekhaTokenizer,
    PralekhaPyTorchDataset,
    PralekhaDataCollator,
    get_pralekha_parallel_pairs,
    get_common_aligned_matrix,
    get_common_aligned_sentences,
    get_common_aligned_parallel_pairs,
    load_ai4bharat_pralekha_dataset,
    PRALEKHA_ISO_MAP,
    load_pralekha_from_file,
    save_pralekha_dataset,
    train_val_test_split,
    get_pralekha_dataloaders,
    get_corpus_statistics
)


def test_all_22_scheduled_languages_and_english():
    """Verify that all 22 scheduled Indian languages plus English are present."""
    assert len(ALL_22_SCHEDULED_LANGUAGES) == 22
    assert len(ALL_LANGUAGES) == 23
    assert "en" in ALL_LANGUAGES
    assert "hi" in ALL_22_SCHEDULED_LANGUAGES
    assert "ta" in ALL_22_SCHEDULED_LANGUAGES
    assert "bn" in ALL_22_SCHEDULED_LANGUAGES
    assert "te" in ALL_22_SCHEDULED_LANGUAGES
    assert "mr" in ALL_22_SCHEDULED_LANGUAGES
    assert "gu" in ALL_22_SCHEDULED_LANGUAGES
    assert "kn" in ALL_22_SCHEDULED_LANGUAGES
    assert "ml" in ALL_22_SCHEDULED_LANGUAGES
    assert "pa" in ALL_22_SCHEDULED_LANGUAGES
    assert "or" in ALL_22_SCHEDULED_LANGUAGES
    assert "as" in ALL_22_SCHEDULED_LANGUAGES
    assert "ur" in ALL_22_SCHEDULED_LANGUAGES
    assert "sa" in ALL_22_SCHEDULED_LANGUAGES
    assert "mai" in ALL_22_SCHEDULED_LANGUAGES
    assert "kok" in ALL_22_SCHEDULED_LANGUAGES
    assert "doi" in ALL_22_SCHEDULED_LANGUAGES
    assert "ks" in ALL_22_SCHEDULED_LANGUAGES
    assert "ne" in ALL_22_SCHEDULED_LANGUAGES
    assert "sd" in ALL_22_SCHEDULED_LANGUAGES
    assert "brx" in ALL_22_SCHEDULED_LANGUAGES
    assert "sat" in ALL_22_SCHEDULED_LANGUAGES
    assert "mni" in ALL_22_SCHEDULED_LANGUAGES


def test_common_aligned_matrix_coverage():
    """Verify that common aligned matrix contains 100% representation across all 23 languages."""
    common_items = get_common_aligned_matrix()
    assert len(common_items) >= 24, "Expected at least 24 universally aligned sentences"

    all_langs_set = set(ALL_LANGUAGES)
    for item in common_items:
        # Every single item must contain every language without any missing
        item_langs = set(item["translations"].keys())
        missing = all_langs_set - item_langs
        assert len(missing) == 0, f"Item {item['id']} is missing languages: {missing}"
        for l, text in item["translations"].items():
            assert text.strip(), f"Item {item['id']} has empty translation for {l}"


def test_common_aligned_sentences_dict():
    """Verify dictionary retrieval of common aligned sentences."""
    sentences = get_common_aligned_sentences()
    assert isinstance(sentences, dict)
    assert len(sentences) >= 24
    sample_key = next(iter(sentences.keys()))
    assert len(sentences[sample_key]) == 23


def test_multidomain_parallel_corpus_generation():
    """Verify multi-domain parallel corpus pairs generation on common aligned data."""
    pairs = get_common_aligned_parallel_pairs()
    assert len(pairs) > 10000, "Expected over 10,000 parallel pairs from common aligned sentences"

    stats = get_corpus_statistics(pairs)
    assert stats["language_count"] == 23
    for d in DOMAINS:
        assert d in stats["domain_distribution"]
        assert stats["domain_distribution"][d] > 0


def test_filtered_pair_generation():
    """Verify pair generation filtered by specific language pairs and domains."""
    target_pairs = [("en", "hi"), ("hi", "bn"), ("ta", "te")]
    filtered = get_pralekha_parallel_pairs(
        lang_pairs=target_pairs,
        domains=["conversational", "governance"],
        require_common_among_all=True
    )
    assert len(filtered) > 0
    for p in filtered:
        assert (p.source_lang, p.target_lang) in target_pairs
        assert p.domain in ["conversational", "governance"]


def test_tokenizer_encoding_decoding():
    """Verify tokenizer encodes with target language prefix and decodes accurately."""
    tokenizer = PralekhaTokenizer()
    assert tokenizer.vocab_size > 1000

    # Check routing tokens for languages
    assert "<2hi>" in tokenizer.token2id
    assert "<2ta>" in tokenizer.token2id
    assert "<2en>" in tokenizer.token2id

    # Test encoding with target language tag
    src_text = "Hello, how are you?"
    encoded = tokenizer.encode(src_text, target_lang="hi")
    assert encoded[0] == tokenizer.token2id["<2hi>"]
    assert encoded[-1] == tokenizer.eos_token_id

    # Test decoding
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert len(decoded) > 0
    assert "<2hi>" not in decoded


def test_tokenizer_save_load_pretrained():
    """Verify saving tokenizer configuration and loading it back."""
    tokenizer = PralekhaTokenizer()
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = tokenizer.save_pretrained(tmp_dir)
        loaded = PralekhaTokenizer.from_pretrained(save_path)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.pad_token_id == tokenizer.pad_token_id
        assert loaded.eos_token_id == tokenizer.eos_token_id


def test_train_val_test_stratified_split():
    """Verify deterministic and stratified dataset splitting."""
    pairs = get_common_aligned_parallel_pairs(max_pairs=500)
    train_set, val_set, test_set = train_val_test_split(
        pairs, val_ratio=0.1, test_ratio=0.1, seed=42, stratify_by_domain=True
    )
    total_split = len(train_set) + len(val_set) + len(test_set)
    assert total_split == len(pairs)
    assert len(train_set) > len(val_set)
    assert len(val_set) > 0

    # Ensure no duplicate pair IDs between partitions
    train_ids = {p.id for p in train_set if p.id}
    val_ids = {p.id for p in val_set if p.id}
    test_ids = {p.id for p in test_set if p.id}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_pytorch_dataset_and_collator():
    """Verify PyTorch Dataset item generation and dynamic batch collator."""
    pairs = get_common_aligned_parallel_pairs(lang_pairs=[("en", "hi"), ("hi", "ta")], max_pairs=32)
    tokenizer = PralekhaTokenizer()
    dataset = PralekhaPyTorchDataset(pairs, tokenizer, max_source_length=64, max_target_length=64)
    assert len(dataset) == len(pairs)

    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert "decoder_input_ids" in item
    assert isinstance(item["input_ids"], torch.Tensor)

    # Test dynamic collator
    collator = PralekhaDataCollator(tokenizer=tokenizer)
    batch = collator([dataset[0], dataset[1], dataset[2]])

    assert batch["input_ids"].dim() == 2
    assert batch["input_ids"].shape[0] == 3
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape[0] == 3
    assert batch["decoder_input_ids"].shape[0] == 3

    # Check that padding in labels uses -100 for ignore_index
    if (batch["labels"] == -100).any():
        assert True


def test_dataloaders_factory_common_aligned():
    """Verify DataLoader generation produces working batch iterators on common aligned data."""
    train_loader, val_loader, test_loader, tokenizer = get_pralekha_dataloaders(
        batch_size=8,
        val_ratio=0.1,
        test_ratio=0.05,
        max_pairs=64,
        require_common_among_all=True
    )
    assert len(train_loader) > 0
    assert len(val_loader) > 0

    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].size(0) <= 8


def test_file_serialization_roundtrip():
    """Verify exporting pairs to JSON and loading them back."""
    pairs = get_common_aligned_parallel_pairs(lang_pairs=[("en", "hi"), ("hi", "bn")], max_pairs=10)
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_file = Path(tmp_dir) / "pralekha_sample.json"
        save_pralekha_dataset(pairs, json_file, format="json")
        loaded_pairs = load_pralekha_from_file(json_file)
        assert len(loaded_pairs) == len(pairs)
        assert loaded_pairs[0].source_text == pairs[0].source_text
        assert loaded_pairs[0].target_text == pairs[0].target_text

        csv_file = Path(tmp_dir) / "pralekha_sample.csv"
        save_pralekha_dataset(pairs, csv_file, format="csv")
        loaded_csv_pairs = load_pralekha_from_file(csv_file)
        assert len(loaded_csv_pairs) == len(pairs)


def test_ai4bharat_pralekha_loader():
    """Verify loading from AI4Bharat Pralekha loader with ISO mapping and fallback."""
    assert PRALEKHA_ISO_MAP["hin"] == "hi"
    assert PRALEKHA_ISO_MAP["tam"] == "ta"
    assert PRALEKHA_ISO_MAP["eng"] == "en"

    # Test loading from synthetic directory structured as Pralekha files
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_file = Path(tmp_dir) / "pralekha_hin_train.json"
        sample_data = [
            {"source_lang": "eng", "target_lang": "hin", "source_text": "Good morning.", "target_text": "सुप्रभात।", "domain": "pib"},
            {"source_lang": "eng", "target_lang": "tam", "source_text": "Thank you.", "target_text": "நன்றி.", "domain": "mkb"}
        ]
        import json
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        loaded = load_ai4bharat_pralekha_dataset(data_dir=tmp_dir)
        assert len(loaded) == 2
        assert loaded[0].source_lang == "en"
        assert loaded[0].target_lang == "hi"
        assert loaded[1].source_lang == "en"
        assert loaded[1].target_lang == "ta"

    # Test fallback to common aligned matrix when empty
    fallback_pairs = load_ai4bharat_pralekha_dataset(data_dir="/non/existent/path", max_samples=20)
    assert len(fallback_pairs) == 20


def test_strict_db_dataloaders():
    """Verify get_pralekha_dataloaders with use_strict_db=True against local strict 12-way DB."""
    from backend.app.services.pralekha_strict_db import PRALEKHA_STRICT_DB_PATH
    if PRALEKHA_STRICT_DB_PATH.exists():
        train_loader, val_loader, test_loader, tokenizer = get_pralekha_dataloaders(
            use_strict_db=True,
            batch_size=8,
            max_pairs=50
        )
        assert len(train_loader) > 0
        batch = next(iter(train_loader))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].size(0) <= 8


def test_sentencepiece_16k_tokenizer_integrity():
    """Verify 16k SentencePiece BPE tokenizer, special tokens, and combining mark preservation."""
    from backend.app.services.pralekha_dataset import PralekhaSentencePieceTokenizer, get_pralekha_spm_tokenizer
    from backend.app.config import MODELS_DIR

    spm_model = MODELS_DIR / "pralekha" / "sentencepiece.bpe.model"
    if not spm_model.exists():
        pytest.skip("SentencePiece model not yet generated.")

    tokenizer = PralekhaSentencePieceTokenizer(model_path=spm_model)
    assert tokenizer.vocab_size == 16000

    # Special token IDs
    assert tokenizer.pad_token_id == 0
    assert tokenizer.unk_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.eos_token_id == 3
    assert tokenizer.mask_token_id == 4

    # Routing tokens verification: each must be a single atomic token
    benchmark_langs = ["en", "hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]
    for lang in benchmark_langs:
        rtok = f"<2{lang}>"
        assert rtok in tokenizer.token2id
        ids = tokenizer.encode(f"{rtok} Test", add_special_tokens=False)
        assert ids[0] == tokenizer.token2id[rtok]

    # Combining mark / matra integrity & lossless detokenization test
    test_cases = {
        "hi": "नमस्ते भारत। यह एक सुंदर देश है।",
        "ta": "வணக்கம் தமிழ்நாடு! இது ஒரு அழகான நாடு.",
        "bn": "নমস্কার ভারত! এটি একটি সুন্দর দেশ।",
        "te": "నమస్కారం భారతదేశం! ఇది ఒక అందమైన దేశం.",
        "kn": "ನಮಸ್ಕಾರ ಭಾರತ! ಇದು ಒಂದು ಸುಂದರ ದೇಶ.",
        "ml": "നമസ്കാരം ഭാരതം! ഇത് മനോഹരമായ രാജ്യം.",
        "gu": "નમસ્તે ભારત! આ એક સુંદર દેશ છે.",
        "pa": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਭਾਰਤ! ਇਹ ਇੱਕ ਸੁੰਦਰ ਦੇਸ਼ ਹੈ।",
        "or": "ନମସ୍କାର ଭାରତ! ଏହା ଏକ ସୁନ୍ଦର ଦେଶ।",
        "ur": "سلام ہندوستان! یہ ایک خوبصورت ملک ہے۔",
        "en": "Hello India! This is a beautiful country."
    }

    for lang, text in test_cases.items():
        encoded = tokenizer.encode(text, target_lang=lang, add_special_tokens=True)
        assert encoded[0] == tokenizer.token2id[f"<2{lang}>"]
        assert encoded[-1] == tokenizer.eos_token_id

        # Decode without special tokens should recover original text
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        assert decoded == text, f"Detokenization mismatch in {lang}: '{decoded}' != '{text}'"

    # Script mask verification
    devanagari_mask = tokenizer.get_script_mask("hi")
    assert devanagari_mask.shape == (16000,)
    assert devanagari_mask[tokenizer.token2id["<2hi>"]] == 0.0 or devanagari_mask[tokenizer.pad_token_id] == 0.0

    # Combining mark suppression mask verification
    combining_mask = tokenizer.get_combining_marks_mask()
    assert combining_mask.shape == (16000,)


def test_balanced_500k_extraction_pipeline():
    """Verify balanced parallel pair extraction across all 132 language directions."""
    from backend.app.services.pralekha_strict_db import (
        PRALEKHA_STRICT_DB_PATH,
        load_balanced_500k_pairs,
        OFFICIAL_12_LANGUAGES
    )
    if not PRALEKHA_STRICT_DB_PATH.exists():
        pytest.skip("Strict 12-way DB not present.")

    # Test balanced extraction with 1,320 pairs (exactly 10 pairs per direction)
    sample_pairs = load_balanced_500k_pairs(
        db_path=PRALEKHA_STRICT_DB_PATH,
        target_pairs=1320,
        seed=42
    )
    assert len(sample_pairs) == 1320

    # Verify coverage across all 132 directions
    direction_counts = {}
    for p in sample_pairs:
        direction = (p["source_lang"], p["target_lang"])
        direction_counts[direction] = direction_counts.get(direction, 0) + 1

    expected_directions = len(OFFICIAL_12_LANGUAGES) * (len(OFFICIAL_12_LANGUAGES) - 1)
    assert len(direction_counts) == expected_directions == 132
    for d, cnt in direction_counts.items():
        assert cnt == 10, f"Direction {d} has {cnt} pairs, expected exactly 10"


def test_get_pralekha_dataloaders_with_balanced_pipeline():
    """Verify get_pralekha_dataloaders integration with balanced extraction and SentencePiece."""
    from backend.app.services.pralekha_strict_db import PRALEKHA_STRICT_DB_PATH
    from backend.app.services.pralekha_dataset import PralekhaSentencePieceTokenizer
    from backend.app.config import MODELS_DIR

    spm_model = MODELS_DIR / "pralekha" / "sentencepiece.bpe.model"
    if not PRALEKHA_STRICT_DB_PATH.exists() or not spm_model.exists():
        pytest.skip("DB or SPM model not available.")

    spm_tokenizer = PralekhaSentencePieceTokenizer(model_path=spm_model)

    train_l, val_l, test_l, tok = get_pralekha_dataloaders(
        use_strict_db=True,
        balanced_132way=True,
        max_pairs=264,  # exactly 2 per direction
        tokenizer=spm_tokenizer,
        batch_size=16
    )
    assert len(train_l) > 0
    batch = next(iter(train_l))
    assert batch["input_ids"].size(0) <= 16
    assert batch["input_ids"].dim() == 2
    assert batch["labels"].dim() == 2
    assert tok.vocab_size == 16000



