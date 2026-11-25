"""
Comprehensive Unit Tests for AI4Bharat Pralekha Trainer & Service Integration
=============================================================================
Tests:
- PralekhaSeq2SeqConfig serialization and deserialization
- PositionalEncoding buffer building and dynamic sequence extension
- PralekhaSeq2SeqTransformer forward pass, loss calculation, and weight tying
- PralekhaSeq2SeqTransformer autoregressive generation
- Checkpoint persistence (pralekha_checkpoint.pt, model.safetensors, config.json, vocab.json)
- Model loading from disk
- PralekhaService integration: is_pralekha_trained(), get_pralekha_status(), translate_with_pralekha()
"""

import os
import json
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from backend.app.services.pralekha_trainer import (
    PralekhaSeq2SeqConfig,
    PositionalEncoding,
    PralekhaSeq2SeqTransformer,
    PralekhaSentencePieceTokenizer,
    save_pralekha_checkpoint,
    load_pralekha_model,
    PRALEKHA_MODEL_DIR
)
from backend.app.services.pralekha_dataset import PralekhaTokenizer
from backend.app.services.pralekha_service import (
    is_pralekha_trained,
    get_pralekha_status,
    translate_with_pralekha,
    reset_pralekha_inference_cache
)


def test_pralekha_config_serialization():
    """Verify config dataclass serialization, deserialization, and json export."""
    cfg = PralekhaSeq2SeqConfig(
        vocab_size=3135,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.05,
        max_len=96
    )
    d = cfg.to_dict()
    assert d["vocab_size"] == 3135
    assert d["d_model"] == 128
    assert d["nhead"] == 4

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "config.json"
        cfg.save_json(p)
        assert p.exists()

        loaded_cfg = PralekhaSeq2SeqConfig.load_json(p)
        assert loaded_cfg.vocab_size == 3135
        assert loaded_cfg.d_model == 128
        assert loaded_cfg.dim_feedforward == 512


def test_positional_encoding():
    """Verify PositionalEncoding buffer shapes and dynamic sequence extension."""
    pe = PositionalEncoding(d_model=64, max_len=32)
    assert pe.pe.size(0) == 1
    assert pe.pe.size(1) >= 32
    assert pe.pe.size(2) == 64

    # Normal sequence
    x = torch.zeros(2, 20, 64)
    out = pe(x)
    assert out.shape == (2, 20, 64)

    # Sequence exceeding initial max_len triggers dynamic extension
    x_long = torch.zeros(2, 600, 64)
    out_long = pe(x_long)
    assert out_long.shape == (2, 600, 64)
    assert pe.pe.size(1) >= 600


def test_pralekha_seq2seq_transformer_forward():
    """Verify Seq2Seq Transformer forward pass, loss computation, and weight tying."""
    cfg = PralekhaSeq2SeqConfig(
        vocab_size=100,
        d_model=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        max_len=32,
        dropout=0.0
    )
    model = PralekhaSeq2SeqTransformer(cfg)

    # Weight tying verification
    assert model.lm_head.weight is model.embed.weight

    src = torch.randint(4, 99, (2, 10))
    tgt = torch.randint(4, 99, (2, 8))
    labels = tgt.clone()

    res = model(input_ids=src, decoder_input_ids=tgt, labels=labels)
    assert "loss" in res
    assert "logits" in res
    assert res["loss"] is not None
    assert res["loss"].item() > 0.0
    assert res["logits"].shape == (2, 8, 100)


def test_pralekha_seq2seq_transformer_generate():
    """Verify autoregressive greedy generation loop."""
    cfg = PralekhaSeq2SeqConfig(
        vocab_size=100,
        d_model=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        max_len=32,
        dropout=0.0
    )
    model = PralekhaSeq2SeqTransformer(cfg)
    model.eval()

    inp = torch.tensor([[10, 20, 30, 3]], dtype=torch.long)
    gen = model.generate(inp, max_length=15, bos_token_id=2, eos_token_id=3)
    assert gen.ndim == 2
    assert gen.size(0) == 1
    assert gen.size(1) <= 15
    # First token must be BOS
    assert gen[0, 0].item() == 2


def test_pralekha_target_lang_conditioning():
    """Verify generate() prefixes target language token ID for decoder conditioning."""
    cfg = PralekhaSeq2SeqConfig(
        vocab_size=100,
        d_model=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        max_len=32,
        dropout=0.0
    )
    model = PralekhaSeq2SeqTransformer(cfg)
    model.eval()

    inp = torch.tensor([[10, 20, 30, 3]], dtype=torch.long)

    # Condition on Hindi (<2hi> -> ID 11)
    gen_hi = model.generate(inp, max_length=10, target_lang="hi")
    assert gen_hi[0, 0].item() == 2  # BOS
    assert gen_hi[0, 1].item() == 11  # <2hi>

    # Condition on Tamil (<2ta> -> ID 25)
    gen_ta = model.generate(inp, max_length=10, target_lang="ta")
    assert gen_ta[0, 0].item() == 2  # BOS
    assert gen_ta[0, 1].item() == 25  # <2ta>

    # Condition on Urdu (<2ur> -> ID 27)
    gen_ur = model.generate(inp, max_length=10, target_lang_id=27)
    assert gen_ur[0, 0].item() == 2  # BOS
    assert gen_ur[0, 1].item() == 27  # <2ur>


def test_pralekha_beam_search_decoding():
    """Verify Beam Search decoding (k=4, length penalty alpha=0.6)."""
    cfg = PralekhaSeq2SeqConfig(
        vocab_size=100,
        d_model=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        max_len=32,
        dropout=0.0
    )
    model = PralekhaSeq2SeqTransformer(cfg)
    model.eval()

    inp = torch.tensor([[10, 20, 30, 3]], dtype=torch.long)

    # Standard beam search (k=4, alpha=0.6)
    gen_beam = model.generate(inp, max_length=15, num_beams=4, length_penalty=0.6)
    assert gen_beam.ndim == 2
    assert gen_beam.size(0) == 1
    assert gen_beam.size(1) <= 15
    assert gen_beam[0, 0].item() == 2  # BOS

    # Beam search with target language conditioning
    gen_beam_hi = model.generate(inp, max_length=12, num_beams=4, length_penalty=0.6, target_lang="hi")
    assert gen_beam_hi[0, 0].item() == 2  # BOS
    assert gen_beam_hi[0, 1].item() == 11  # <2hi>


def test_pralekha_beam_search_script_masking():
    """Verify target-script logit masking restricts forbidden tokens in Beam Search."""
    cfg = PralekhaSeq2SeqConfig(
        vocab_size=100,
        d_model=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        max_len=32,
        dropout=0.0
    )
    model = PralekhaSeq2SeqTransformer(cfg)
    model.eval()

    inp = torch.tensor([[50, 60, 70, 3]], dtype=torch.long)

    # First get unmasked generation to identify chosen tokens
    unmasked = model.generate(inp, max_length=12, num_beams=4, length_penalty=0.6)
    chosen_token = unmasked[0, 1].item() if unmasked.size(1) > 1 else None

    # Mask out the chosen token with -1e9
    if chosen_token is not None:
        mask = torch.zeros(100)
        mask[chosen_token] = -1e9
        masked_gen = model.generate(inp, max_length=12, num_beams=4, length_penalty=0.6, logit_mask=mask)
        # Forbidden token must never appear in generated output
        generated_tokens = masked_gen[0, 1:].tolist()  # skip BOS
        assert chosen_token not in generated_tokens, f"Token {chosen_token} should have been masked out"


def test_pralekha_edge_42m_config():
    """Verify 42M Edge Transformer configuration hyper-parameters and parameter counts."""
    cfg = PralekhaSeq2SeqConfig.edge_42m(vocab_size=16000)
    assert cfg.d_model == 512
    assert cfg.nhead == 8
    assert cfg.num_encoder_layers == 6
    assert cfg.num_decoder_layers == 6
    assert cfg.dim_feedforward == 2048
    assert cfg.vocab_size == 16000

    model = PralekhaSeq2SeqTransformer(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = total_params - (cfg.vocab_size * cfg.d_model)

    # Backbone: ~44.14M parameters
    assert 43_000_000 < backbone_params < 45_500_000, f"Backbone params: {backbone_params:,}"
    # Total with 16k vocab: ~52.33M parameters
    assert 50_000_000 < total_params < 54_000_000, f"Total params: {total_params:,}"


def test_pralekha_sentencepiece_tokenizer_wrapper():
    """Verify PralekhaSentencePieceTokenizer error handling and structure."""
    with pytest.raises(FileNotFoundError):
        PralekhaSentencePieceTokenizer("non_existent_spm_model_path.model")


def test_pralekha_checkpoint_files_exist():
    """Verify trained checkpoint and safetensors files exist in PRALEKHA_MODEL_DIR."""
    assert PRALEKHA_MODEL_DIR.exists()
    ckpt_file = PRALEKHA_MODEL_DIR / "pralekha_checkpoint.pt"
    st_file = PRALEKHA_MODEL_DIR / "model.safetensors"
    cfg_file = PRALEKHA_MODEL_DIR / "config.json"
    vocab_file = PRALEKHA_MODEL_DIR / "vocab.json"
    summary_file = PRALEKHA_MODEL_DIR / "training_summary.json"

    assert ckpt_file.exists(), "pralekha_checkpoint.pt must exist"
    assert st_file.exists(), "model.safetensors must exist"
    assert cfg_file.exists(), "config.json must exist"
    assert vocab_file.exists(), "vocab.json must exist"
    assert summary_file.exists(), "training_summary.json must exist"

    # Validate checkpoint file content
    ckpt = torch.load(ckpt_file, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "config" in ckpt
    assert "vocab_size" in ckpt
    assert ckpt["epoch"] >= 1


def test_load_pralekha_model():
    """Verify load_pralekha_model loads weights and tokenizer cleanly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, dev = load_pralekha_model(PRALEKHA_MODEL_DIR, device=device)

    assert isinstance(model, PralekhaSeq2SeqTransformer)
    assert isinstance(tokenizer, (PralekhaTokenizer, PralekhaSentencePieceTokenizer))
    assert str(dev) == device
    assert tokenizer.vocab_size > 2000
    assert model.config.vocab_size == tokenizer.vocab_size


def test_pralekha_service_is_trained():
    """Verify is_pralekha_trained() returns True when weights exist."""
    assert is_pralekha_trained() is True


def test_pralekha_service_status():
    """Verify get_pralekha_status() returns operational_trained with full details."""
    status = get_pralekha_status()
    assert status["engine_id"] == "pralekha"
    assert status["is_trained"] is True
    assert status["status"] == "operational_trained"
    assert "Custom trained weights active" in status["mode_description"]
    assert "summary" in status
    assert status["summary"]["architecture"] == "Seq2Seq Transformer (Encoder-Decoder)"


@pytest.mark.asyncio
async def test_pralekha_service_translate_lexicon():
    """Verify translate_with_pralekha uses trained checkpoint and preserves lexicon."""
    reset_pralekha_inference_cache()
    res = await translate_with_pralekha(
        source_lang="en",
        target_lang="sa",
        text="hello"
    )
    assert res["engine"] == "pralekha"
    assert "Pralekha" in res["pipeline"]
    assert res["model_status"] == "custom_trained_checkpoint"
    assert res["translated_text"] == "नमस्ते"
    assert res["confidence"] >= 0.90


@pytest.mark.asyncio
async def test_pralekha_service_translate_gpu_inference():
    """Verify live GPU neural inference executes across Indic languages."""
    reset_pralekha_inference_cache()
    res = await translate_with_pralekha(
        source_lang="en",
        target_lang="hi",
        text="Hello, how are you?"
    )
    assert res["engine"] == "pralekha"
    assert "Pralekha" in res["pipeline"]
    assert res["model_status"] == "custom_trained_checkpoint"
    assert isinstance(res["translated_text"], str)
    assert len(res["translated_text"]) > 0
