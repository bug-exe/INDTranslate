"""
AI4Bharat Pralekha Model Architecture & Training Pipeline
=========================================================
Encoder-Decoder Seq2Seq Transformer model implemented in PyTorch,
supporting all 22 Eighth Schedule Indian Languages + English (including the
strict 12-way direct pivot-free benchmark matrix across 132 language directions).
Optimized for NVIDIA RTX 4050 GPU (FP16 / BF16 mixed precision, AdamW optimizer).
Saves model checkpoints, config, safetensors, and vocabulary to backend/saved_models/pralekha/
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from backend.app.config import MODELS_DIR
from backend.app.services.pralekha_dataset import (
    PralekhaTokenizer,
    ParallelSentencePair,
    get_pralekha_dataloaders,
    ALL_LANGUAGES,
    ALL_22_SCHEDULED_LANGUAGES
)

logger = logging.getLogger("indtranslate.pralekha.trainer")

PRALEKHA_MODEL_DIR = MODELS_DIR / "pralekha"
PRALEKHA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_STRICT_12WAY_DB = Path("backend/data/pralekha_strict_12way.db")


# ---------------------------------------------------------------------------
# Strict 12-Way SQLite Loader Utility
# ---------------------------------------------------------------------------

def load_pairs_from_strict_db(
    db_path: Union[str, Path] = DEFAULT_STRICT_12WAY_DB,
    max_pairs: Optional[int] = 30000,
    min_len: int = 4
) -> List[ParallelSentencePair]:
    """
    Loads direct pivot-free parallel sentence pairs across all 132 language directions
    from the strict 12-way benchmark database (backend/data/pralekha_strict_12way.db).
    """
    path = Path(db_path)
    if not path.exists():
        logger.warning(f"Strict 12-way DB not found at {path}")
        return []

    import sqlite3
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()
    query = """
        SELECT source_text, target_text, source_lang, target_lang, doc_id
        FROM training_pairs
        WHERE split = 'train'
          AND length(source_text) >= ?
          AND length(target_text) >= ?
        ORDER BY pair_id ASC
    """
    if max_pairs:
        query += f" LIMIT {int(max_pairs)}"
    cursor.execute(query, (min_len, min_len))
    rows = cursor.fetchall()
    conn.close()

    pairs = [
        ParallelSentencePair(
            source_text=r[0],
            target_text=r[1],
            source_lang=r[2],
            target_lang=r[3],
            domain="governance_strict_12way",
            id=f"{r[4]}_{r[2]}_{r[3]}"
        )
        for r in rows
    ]
    logger.info(f"Loaded {len(pairs)} strict 12-way pairs from {path}")
    return pairs


# ---------------------------------------------------------------------------
# Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PralekhaSeq2SeqConfig:
    """Configuration hyper-parameters for Pralekha Seq2Seq Transformer."""
    vocab_size: int = 16000
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_len: int = 128
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    norm_first: bool = True
    model_type: str = "pralekha_seq2seq_transformer"
    num_languages: int = 23

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PralekhaSeq2SeqConfig:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> PralekhaSeq2SeqConfig:
        with open(Path(path), "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def edge_42m(cls, vocab_size: int = 16000, **kwargs) -> PralekhaSeq2SeqConfig:
        """42M Edge Transformer architecture (6e/6d, d_model=512, dim_ff=2048, nhead=8)."""
        return cls(
            vocab_size=vocab_size,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            **kwargs
        )

    @classmethod
    def tiny_demo(cls, vocab_size: int = 3135, **kwargs) -> PralekhaSeq2SeqConfig:
        """Lightweight 6M demo model for fast iteration and local unit testing."""
        return cls(
            vocab_size=vocab_size,
            d_model=256,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=1024,
            dropout=0.1,
            **kwargs
        )


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding with Dynamic Extension
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dropout and dynamic sequence length extension."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = self._build_pe(max(max_len, 512), d_model)
        self.register_buffer("pe", pe)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            new_pe = self._build_pe(max(seq_len + 64, 512), self.d_model).to(x.device)
            self.pe = new_pe
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Language Token ID Mappings & Resolution Helper
# ---------------------------------------------------------------------------

LANGUAGE_TOKEN_IDS: Dict[str, int] = {
    "as": 5, "assamese": 5, "<2as>": 5,
    "bn": 6, "bengali": 6, "<2bn>": 6,
    "brx": 7, "bodo": 7, "<2brx>": 7,
    "doi": 8, "dogri": 8, "<2doi>": 8,
    "en": 9, "english": 9, "<2en>": 9,
    "gu": 10, "gujarati": 10, "<2gu>": 10,
    "hi": 11, "hindi": 11, "<2hi>": 11,
    "kn": 12, "kannada": 12, "<2kn>": 12,
    "kok": 13, "konkani": 13, "<2kok>": 13,
    "ks": 14, "kashmiri": 14, "<2ks>": 14,
    "mai": 15, "maithili": 15, "<2mai>": 15,
    "ml": 16, "malayalam": 16, "<2ml>": 16,
    "mni": 17, "manipuri": 17, "<2mni>": 17,
    "mr": 18, "marathi": 18, "<2mr>": 18,
    "ne": 19, "nepali": 19, "<2ne>": 19,
    "or": 20, "odia": 20, "oriya": 20, "<2or>": 20,
    "pa": 21, "punjabi": 21, "<2pa>": 21,
    "sa": 22, "sanskrit": 22, "<2sa>": 22,
    "sat": 23, "santali": 23, "<2sat>": 23,
    "sd": 24, "sindhi": 24, "<2sd>": 24,
    "ta": 25, "tamil": 25, "<2ta>": 25,
    "te": 26, "telugu": 26, "<2te>": 26,
    "ur": 27, "urdu": 27, "<2ur>": 27,
}


def resolve_target_lang_id(target_lang: Optional[Union[str, int]] = None) -> Optional[int]:
    """Resolves language code, name, or token into integer token ID."""
    if target_lang is None:
        return None
    if isinstance(target_lang, int):
        return target_lang
    clean = str(target_lang).strip().lower()
    return LANGUAGE_TOKEN_IDS.get(clean)


# ---------------------------------------------------------------------------
# SentencePiece BPE Tokenizer Wrapper
# ---------------------------------------------------------------------------

class PralekhaSentencePieceTokenizer:
    """
    Subword tokenizer adapter wrapping Google SentencePiece BPE model.
    Matches the exact interface of PralekhaTokenizer:
    - encode(text, target_lang=None, max_length=128, truncation=True, add_special_tokens=True)
    - decode(token_ids, skip_special_tokens=True)
    - vocab_size, pad_token_id, unk_token_id, bos_token_id, eos_token_id
    """

    def __init__(self, model_path: Union[str, Path]):
        import sentencepiece as spm
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found at {self.model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))

        self.pad_token_id = self.sp.piece_to_id("<pad>") if self.sp.piece_to_id("<pad>") != -1 else 0
        self.unk_token_id = self.sp.piece_to_id("<unk>") if self.sp.piece_to_id("<unk>") != -1 else 1
        self.bos_token_id = self.sp.piece_to_id("<bos>") if self.sp.piece_to_id("<bos>") != -1 else 2
        self.eos_token_id = self.sp.piece_to_id("<eos>") if self.sp.piece_to_id("<eos>") != -1 else 3

        self.vocab_size = self.sp.get_piece_size()

    def encode(
        self,
        text: str,
        target_lang: Optional[str] = None,
        max_length: int = 128,
        truncation: bool = True,
        add_special_tokens: bool = True
    ) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            prefix: List[int] = []
            if target_lang:
                lang_tok = f"<2{target_lang}>"
                lid = self.sp.piece_to_id(lang_tok)
                if lid != -1:
                    prefix.append(lid)
                else:
                    mapped_id = LANGUAGE_TOKEN_IDS.get(str(target_lang).strip().lower())
                    if mapped_id is not None:
                        prefix.append(mapped_id)
                    else:
                        prefix.append(self.bos_token_id)
            else:
                prefix.append(self.bos_token_id)
            ids = prefix + ids + [self.eos_token_id]

        if truncation and len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_token_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special_ids = {self.pad_token_id, self.unk_token_id, self.bos_token_id, self.eos_token_id}
            filtered = []
            for tid in token_ids:
                if tid in special_ids:
                    continue
                piece = self.sp.id_to_piece(tid) if 0 <= tid < self.vocab_size else ""
                if piece.startswith("<2") and piece.endswith(">"):
                    continue
                filtered.append(tid)
            token_ids = filtered
        return self.sp.decode(token_ids)

    def get_vocab(self) -> Dict[str, int]:
        return {self.sp.id_to_piece(i): i for i in range(self.vocab_size)}

    def get_script_mask(self, target_lang: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Computes a target-script logit mask tensor (shape: [vocab_size])."""
        lang_clean = str(target_lang).lower().strip()
        if hasattr(self, "_script_masks") and lang_clean in self._script_masks:
            mask = self._script_masks[lang_clean]
            return mask.to(device) if device is not None else mask

        if not hasattr(self, "_script_masks"):
            self._script_masks = {}

        import unicodedata
        SCRIPT_BY_LANG: Dict[str, str] = {
            "hi": "DEVANAGARI", "mr": "DEVANAGARI", "sa": "DEVANAGARI", "ne": "DEVANAGARI",
            "kok": "DEVANAGARI", "mai": "DEVANAGARI", "doi": "DEVANAGARI", "brx": "DEVANAGARI",
            "bn": "BENGALI", "as": "BENGALI", "mni": "BENGALI",
            "pa": "GURMUKHI",
            "gu": "GUJARATI",
            "or": "ORIYA",
            "ta": "TAMIL",
            "te": "TELUGU",
            "kn": "KANNADA",
            "ml": "MALAYALAM",
            "ur": "ARABIC", "ks": "ARABIC", "sd": "ARABIC",
            "sat": "OL",
            "en": "LATIN"
        }
        target_script = SCRIPT_BY_LANG.get(lang_clean, "LATIN")
        mask = torch.full((self.vocab_size,), -float("inf"), dtype=torch.float32)

        # Always allow special control and routing tokens
        mask[self.pad_token_id] = 0.0
        mask[self.bos_token_id] = 0.0
        mask[self.eos_token_id] = 0.0
        mask[self.unk_token_id] = 0.0

        for i in range(self.vocab_size):
            piece = self.sp.id_to_piece(i)
            clean_piece = piece.lstrip(" ")
            if not clean_piece:
                mask[i] = 0.0
                continue
            is_allowed = True
            has_letters = False
            for ch in clean_piece:
                cat = unicodedata.category(ch)
                if cat.startswith("L") or cat.startswith("M"):
                    has_letters = True
                    try:
                        uname = unicodedata.name(ch, "")
                    except Exception:
                        uname = ""
                    if target_script not in uname:
                        is_allowed = False
                        break
                elif cat.startswith("P") or cat.startswith("Z") or cat.startswith("N"):
                    pass
            if is_allowed or not has_letters:
                mask[i] = 0.0

        self._script_masks[lang_clean] = mask
        return mask.to(device) if device is not None else mask

    def get_combining_marks_mask(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns mask suppressing standalone combining marks at step 0."""
        mask = torch.zeros(self.vocab_size, dtype=torch.float32)
        import unicodedata
        for i in range(self.vocab_size):
            piece = self.sp.id_to_piece(i).lstrip(" ")
            if piece and all(unicodedata.category(c).startswith("M") for c in piece):
                mask[i] = -float("inf")
        return mask.to(device) if device is not None else mask

    def save_pretrained(self, output_dir: Union[str, Path]) -> None:
        import shutil
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest_model = out / "sentencepiece.bpe.model"
        if self.model_path.resolve() != dest_model.resolve():
            shutil.copy2(self.model_path, dest_model)


# ---------------------------------------------------------------------------
# Pralekha Seq2Seq Transformer Model
# ---------------------------------------------------------------------------

class PralekhaSeq2SeqTransformer(nn.Module):
    """
    Direct Indic Seq2Seq Transformer architecture.
    Comprises shared token embeddings, sinusoidal positional encoding,
    PyTorch standard Transformer encoder-decoder stacks, and tied linear LM head.
    """

    def __init__(self, config: Optional[PralekhaSeq2SeqConfig] = None):
        super().__init__()
        self.config = config or PralekhaSeq2SeqConfig()

        self.embed = nn.Embedding(
            self.config.vocab_size,
            self.config.d_model,
            padding_idx=self.config.pad_token_id
        )
        # Scaled standard normal embedding initialization (std = 1 / sqrt(d_model))
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.config.d_model ** -0.5)
        self.pos_encoder = PositionalEncoding(
            self.config.d_model,
            max_len=max(self.config.max_len, 512),
            dropout=self.config.dropout
        )
        self.transformer = nn.Transformer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=getattr(self.config, "norm_first", True)
        )
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        # Weight tying between embedding and output projection
        self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        target_lang: Optional[str] = None,
        target_lang_id: Optional[int] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass for training or validation.
        Computes CrossEntropyLoss when labels are provided.
        """
        device = input_ids.device

        # Resolve target language ID if provided
        resolved_lang_id = resolve_target_lang_id(target_lang_id if target_lang_id is not None else target_lang)

        # If decoder_input_ids not explicitly passed, shift from labels with language conditioning
        if decoder_input_ids is None and labels is not None:
            if resolved_lang_id is not None:
                start_tokens = torch.tensor(
                    [[self.config.bos_token_id, resolved_lang_id]],
                    dtype=labels.dtype,
                    device=device
                ).repeat(labels.size(0), 1)
                clean_labels = labels.clone()
                clean_labels[clean_labels == -100] = self.config.pad_token_id
                decoder_input_ids = torch.cat([start_tokens, clean_labels[:, :-2]], dim=1) if clean_labels.size(1) >= 2 else start_tokens
            else:
                bos = torch.full(
                    (labels.size(0), 1),
                    self.config.bos_token_id,
                    dtype=labels.dtype,
                    device=device
                )
                clean_labels = labels.clone()
                clean_labels[clean_labels == -100] = self.config.pad_token_id
                decoder_input_ids = torch.cat([bos, clean_labels[:, :-1]], dim=1)

        if decoder_input_ids is None:
            raise ValueError("Either decoder_input_ids or labels must be supplied.")

        # Embeddings + Positional Encodings
        src = self.pos_encoder(self.embed(input_ids) * math.sqrt(self.config.d_model))
        tgt = self.pos_encoder(self.embed(decoder_input_ids) * math.sqrt(self.config.d_model))

        # Padding masks (True indicates key to IGNORE)
        src_key_padding_mask = (input_ids == self.config.pad_token_id)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        tgt_key_padding_mask = (decoder_input_ids == self.config.pad_token_id)

        # Causal subsequent mask for autoregressive decoder (0.0 for allow, -inf for forbidden future)
        tgt_len = decoder_input_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)

        # Transformer execution
        out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        logits = self.lm_head(out)
        loss = None

        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def _greedy_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        repetition_penalty: float = 1.25,
        no_repeat_ngram_size: int = 3,
        bos_id: int = 2,
        eos_id: int = 3,
        pad_id: int = 0,
        resolved_lang_id: Optional[int] = None,
        logit_mask: Optional[torch.Tensor] = None,
        combining_marks_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Greedy autoregressive decoding for Seq2Seq inference."""
        device = input_ids.device
        batch_size = input_ids.size(0)

        src_mask = (input_ids == pad_id)
        if attention_mask is not None:
            src_mask = (attention_mask == 0)

        # Initialize generation buffer with [BOS] matching decoder training format
        init_tokens = [bos_id]
        if resolved_lang_id is not None:
            init_tokens.append(resolved_lang_id)

        generated = torch.tensor([init_tokens], dtype=torch.long, device=device).repeat(batch_size, 1)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Encode source sequence once
        src_emb = self.pos_encoder(self.embed(input_ids) * math.sqrt(self.config.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

        max_steps = max(max_length - generated.size(1), 1)

        for step in range(max_steps):
            tgt_len = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
            tgt_pad_mask = (generated == pad_id)
            tgt_emb = self.pos_encoder(self.embed(generated) * math.sqrt(self.config.d_model))

            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_pad_mask
            )

            # Extract last token logits
            next_logits = self.lm_head(out[:, -1, :])

            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for b in range(batch_size):
                    unique_tokens = torch.unique(generated[b])
                    for t in unique_tokens:
                        if next_logits[b, t] > 0:
                            next_logits[b, t] /= repetition_penalty
                        else:
                            next_logits[b, t] *= repetition_penalty

            # Apply target script logit mask (restricts output to target language script)
            if logit_mask is not None:
                next_logits = next_logits + logit_mask.to(device)

            # Prevent regenerating BOS, PAD, or any special language routing tokens (<2as>..<2ur>)
            next_logits[:, bos_id] = float("-inf")
            next_logits[:, pad_id] = float("-inf")
            if next_logits.size(1) > 28:
                next_logits[:, 5:28] = float("-inf")

            # At step 0, prevent generating premature EOS or orphaned combining marks
            if step == 0:
                next_logits[:, eos_id] = float("-inf")
                if combining_marks_mask is not None:
                    next_logits = next_logits + combining_marks_mask.to(device)

            next_tokens = torch.argmax(next_logits, dim=-1, keepdim=True)

            # Mask out already finished sequences
            next_tokens = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_tokens, pad_id),
                next_tokens
            )

            generated = torch.cat([generated, next_tokens], dim=1)
            finished = finished | (next_tokens.squeeze(1) == eos_id)

            if finished.all():
                break

        return generated

    @torch.no_grad()
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 4,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.25,
        no_repeat_ngram_size: int = 3,
        bos_id: int = 2,
        eos_id: int = 3,
        pad_id: int = 0,
        resolved_lang_id: Optional[int] = None,
        logit_mask: Optional[torch.Tensor] = None,
        combining_marks_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Beam Search decoding (k=num_beams, length_penalty=alpha) with target script logit masking,
        repetition penalty, and combining mark prevention.
        """
        device = input_ids.device
        batch_size = input_ids.size(0)

        src_mask = (input_ids == pad_id)
        if attention_mask is not None:
            src_mask = (attention_mask == 0)

        # Initial tokens: [BOS] matching decoder training format
        init_tokens = [bos_id]
        if resolved_lang_id is not None:
            init_tokens.append(resolved_lang_id)
        prefix_len = len(init_tokens)

        # Encode source sequence once
        src_emb = self.pos_encoder(self.embed(input_ids) * math.sqrt(self.config.d_model))
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

        batch_results: List[torch.Tensor] = []

        for b in range(batch_size):
            mem_b = memory[b:b+1]
            src_mask_b = src_mask[b:b+1] if src_mask is not None else None

            cur_beams = torch.tensor([init_tokens], dtype=torch.long, device=device)
            beam_scores = torch.zeros(1, dtype=torch.float32, device=device)
            completed_hypotheses: List[Tuple[float, torch.Tensor]] = []

            max_steps = max(max_length - prefix_len, 1)

            for step in range(max_steps):
                num_active = cur_beams.size(0)
                expanded_mem = mem_b.expand(num_active, -1, -1)
                expanded_src_mask = src_mask_b.expand(num_active, -1) if src_mask_b is not None else None

                tgt_len = cur_beams.size(1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
                tgt_pad_mask = (cur_beams == pad_id)
                tgt_emb = self.pos_encoder(self.embed(cur_beams) * math.sqrt(self.config.d_model))

                out = self.transformer.decoder(
                    tgt_emb,
                    expanded_mem,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=expanded_src_mask,
                    tgt_key_padding_mask=tgt_pad_mask
                )

                next_logits = self.lm_head(out[:, -1, :])

                # 1. Repetition penalty
                if repetition_penalty > 1.0:
                    for i in range(num_active):
                        unique_tokens = torch.unique(cur_beams[i])
                        for t in unique_tokens:
                            if next_logits[i, t] > 0:
                                next_logits[i, t] /= repetition_penalty
                            else:
                                next_logits[i, t] *= repetition_penalty

                # 2. Target script logit mask
                if logit_mask is not None:
                    next_logits = next_logits + logit_mask.to(device)

                # 3. Special token suppression
                next_logits[:, bos_id] = float("-inf")
                next_logits[:, pad_id] = float("-inf")
                if next_logits.size(1) > 28:
                    next_logits[:, 5:28] = float("-inf")

                # 4. Combining marks suppression at step 0
                # At step 0, prevent generating premature EOS or orphaned combining marks
                if step == 0:
                    next_logits[:, eos_id] = float("-inf")
                    if combining_marks_mask is not None:
                        next_logits = next_logits + combining_marks_mask.to(device)

                log_probs = torch.log_softmax(next_logits, dim=-1)

                if step == 0:
                    cand_scores = log_probs[0]
                    topk_scores, topk_indices = torch.topk(cand_scores, k=min(2 * num_beams, cand_scores.size(-1)))

                    next_beams_list: List[torch.Tensor] = []
                    next_scores_list: List[float] = []

                    for sc, tok_id in zip(topk_scores, topk_indices):
                        tok = tok_id.item()
                        s = sc.item()
                        if tok == eos_id:
                            lp = ((5.0 + 1.0) / 6.0) ** length_penalty
                            completed_hypotheses.append((s / lp, torch.cat([cur_beams[0], tok_id.unsqueeze(0)])))
                        else:
                            next_beams_list.append(torch.cat([cur_beams[0], tok_id.unsqueeze(0)]))
                            next_scores_list.append(s)
                            if len(next_beams_list) == num_beams:
                                break

                    if not next_beams_list:
                        break
                    cur_beams = torch.stack(next_beams_list)
                    beam_scores = torch.tensor(next_scores_list, dtype=torch.float32, device=device)
                else:
                    cand_scores = beam_scores.unsqueeze(1) + log_probs
                    flat_scores = cand_scores.view(-1)
                    k_cands = min(2 * num_beams, flat_scores.numel())
                    topk_scores, topk_indices = torch.topk(flat_scores, k=k_cands)

                    beam_idx = topk_indices // self.config.vocab_size
                    token_idx = topk_indices % self.config.vocab_size

                    next_beams_list = []
                    next_scores_list = []

                    content_len = (cur_beams.size(1) - prefix_len) + 1
                    lp = ((5.0 + content_len) / 6.0) ** length_penalty

                    for sc_val, b_id, tok_id in zip(topk_scores, beam_idx, token_idx):
                        tok = tok_id.item()
                        b_i = b_id.item()
                        s = sc_val.item()

                        if tok == eos_id:
                            completed_hyp = torch.cat([cur_beams[b_i], tok_id.unsqueeze(0)])
                            completed_hypotheses.append((s / lp, completed_hyp))
                            if len(completed_hypotheses) >= num_beams:
                                break
                        else:
                            cand_hyp = torch.cat([cur_beams[b_i], tok_id.unsqueeze(0)])
                            next_beams_list.append(cand_hyp)
                            next_scores_list.append(s)
                            if len(next_beams_list) == num_beams:
                                break

                    if len(completed_hypotheses) >= num_beams or not next_beams_list:
                        break

                    cur_beams = torch.stack(next_beams_list)
                    beam_scores = torch.tensor(next_scores_list, dtype=torch.float32, device=device)

            if completed_hypotheses:
                best_hyp = max(completed_hypotheses, key=lambda x: x[0])[1]
            else:
                active_len = max(cur_beams.size(1) - prefix_len, 1)
                lp = ((5.0 + active_len) / 6.0) ** length_penalty
                norm_scores = beam_scores / lp
                best_idx = torch.argmax(norm_scores).item()
                best_hyp = cur_beams[best_idx]

            batch_results.append(best_hyp)

        max_res_len = max(res.size(0) for res in batch_results)
        padded_batch = []
        for res in batch_results:
            if res.size(0) < max_res_len:
                pad_tensor = torch.full((max_res_len - res.size(0),), pad_id, dtype=torch.long, device=device)
                padded_batch.append(torch.cat([res, pad_tensor]))
            else:
                padded_batch.append(res)

        return torch.stack(padded_batch)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 4,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.25,
        no_repeat_ngram_size: int = 3,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        target_lang: Optional[str] = None,
        target_lang_id: Optional[int] = None,
        logit_mask: Optional[torch.Tensor] = None,
        combining_marks_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressive Seq2Seq decoding for inference supporting Beam Search (num_beams > 1)
        and Greedy decoding (num_beams <= 1), with target language conditioning (<2lang>),
        target-script logit masking, length penalty, and repetition penalty.
        """
        self.eval()
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        bos_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        pad_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id

        # Resolve target language token ID
        resolved_lang_id = resolve_target_lang_id(target_lang_id if target_lang_id is not None else target_lang)

        # Fallback: if not explicitly passed, check if input_ids has language routing token at position 0
        if resolved_lang_id is None and input_ids.size(1) > 0:
            first_tok = input_ids[0, 0].item()
            if 5 <= first_tok <= 27:
                resolved_lang_id = first_tok

        if num_beams > 1:
            return self._beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                resolved_lang_id=resolved_lang_id,
                logit_mask=logit_mask,
                combining_marks_mask=combining_marks_mask
            )
        else:
            return self._greedy_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                resolved_lang_id=resolved_lang_id,
                logit_mask=logit_mask,
                combining_marks_mask=combining_marks_mask
            )


# ---------------------------------------------------------------------------
# Model Persistence & Loading Utilities
# ---------------------------------------------------------------------------

def save_pralekha_checkpoint(
    model: PralekhaSeq2SeqTransformer,
    tokenizer: PralekhaTokenizer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    train_loss: float = 0.0,
    val_loss: float = 0.0,
    output_dir: Union[str, Path] = PRALEKHA_MODEL_DIR,
    dataset_desc: Optional[str] = None
) -> Dict[str, Path]:
    """
    Saves model checkpoint (.pt), weights (.safetensors), config (.json),
    and vocabulary files into output_dir.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Config JSON
    config_path = out_dir / "config.json"
    model.config.save_json(config_path)

    # 2. Tokenizer & Vocab JSON
    tokenizer.save_pretrained(out_dir)
    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.get_vocab(), f, indent=2, ensure_ascii=False)

    # 3. Model safetensors
    safetensors_path = out_dir / "model.safetensors"
    state_dict_cpu = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict_cpu, str(safetensors_path))

    # 4. PyTorch comprehensive checkpoint (.pt)
    checkpoint_path = out_dir / "pralekha_checkpoint.pt"
    ckpt = {
        "model_state_dict": state_dict_cpu,
        "config": model.config.to_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "vocab_size": tokenizer.vocab_size,
        "model_type": model.config.model_type,
        "dataset": dataset_desc or "AI4Bharat Pralekha Direct Indic-to-Indic Dataset",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, checkpoint_path)

    # 5. Metadata summary
    summary_path = out_dir / "training_summary.json"
    summary = {
        "model_name": "AI4Bharat Pralekha Direct Indic-to-Indic Model",
        "architecture": "Seq2Seq Transformer (Encoder-Decoder)",
        "dataset": dataset_desc or "AI4Bharat Pralekha Strict 12-Way Benchmark (132 Directions)",
        "d_model": model.config.d_model,
        "nhead": model.config.nhead,
        "layers": f"{model.config.num_encoder_layers} enc / {model.config.num_decoder_layers} dec",
        "vocab_size": tokenizer.vocab_size,
        "languages": 23,
        "language_directions": 132,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "epochs_completed": epoch,
        "checkpoint_file": str(checkpoint_path.name),
        "safetensors_file": str(safetensors_path.name),
        "config_file": str(config_path.name),
        "status": "operational_trained"
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Pralekha checkpoint saved successfully to {out_dir}")
    return {
        "checkpoint": checkpoint_path,
        "safetensors": safetensors_path,
        "config": config_path,
        "vocab": vocab_path,
        "summary": summary_path
    }


def load_pralekha_model(
    model_dir: Union[str, Path] = PRALEKHA_MODEL_DIR,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[PralekhaSeq2SeqTransformer, PralekhaTokenizer, torch.device]:
    """
    Loads trained Pralekha Seq2Seq model and tokenizer from model_dir.
    Prefers model.safetensors, falls back to pralekha_checkpoint.pt.
    Moves model to specified device (or best available GPU/CPU).
    """
    m_dir = Path(model_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 1. Load config
    config_file = m_dir / "config.json"
    if config_file.exists():
        config = PralekhaSeq2SeqConfig.load_json(config_file)
    else:
        config = PralekhaSeq2SeqConfig()

    # 2. Load tokenizer
    spm_model_file = m_dir / "sentencepiece.bpe.model"
    if spm_model_file.exists() and (getattr(config, "vocab_size", 0) >= 10000 or not (m_dir / "pralekha_vocab.json").exists()):
        tokenizer: Any = PralekhaSentencePieceTokenizer(spm_model_file)
        logger.info(f"Loaded SentencePiece BPE tokenizer from {spm_model_file} (vocab={tokenizer.vocab_size})")
    else:
        tokenizer = PralekhaTokenizer.from_pretrained(m_dir)
        logger.info(f"Loaded Pralekha character tokenizer from {m_dir} (vocab={tokenizer.vocab_size})")
    config.vocab_size = tokenizer.vocab_size

    # 3. Instantiate model
    model = PralekhaSeq2SeqTransformer(config)

    # 4. Load weights
    safetensors_file = m_dir / "model.safetensors"
    pt_file = m_dir / "pralekha_checkpoint.pt"

    if safetensors_file.exists():
        state_dict = load_file(str(safetensors_file))
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {safetensors_file}")
    elif pt_file.exists():
        ckpt = torch.load(pt_file, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        logger.info(f"Loaded weights from {pt_file}")
    else:
        raise FileNotFoundError(f"No model weights found in {m_dir}")

    model.to(device)
    model.eval()
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Model Training Pipeline
# ---------------------------------------------------------------------------

def train_pralekha_model(
    dataset_path: Optional[str] = None,
    db_path: Optional[str] = "backend/data/pralekha_strict_12way.db",
    epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    max_len: int = 128,
    use_amp: bool = True,
    device: Optional[str] = None,
    output_dir: Union[str, Path] = PRALEKHA_MODEL_DIR,
    max_pairs: Optional[int] = 30000
) -> Dict[str, Any]:
    """
    Trains / fine-tunes the AI4Bharat Pralekha direct translation model.
    Optimized for RTX 4050 GPU using mixed precision FP16 and AdamW.
    """
    start_time = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    is_cuda = dev.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"

    # Determine data source
    pairs: Optional[List[ParallelSentencePair]] = None
    dataset_description = "AI4Bharat Pralekha Corpus"
    if db_path and Path(db_path).exists():
        pairs = load_pairs_from_strict_db(db_path=db_path, max_pairs=max_pairs)
        dataset_description = f"backend/data/pralekha_strict_12way.db ({len(pairs)} pairs across 132 directions)"

    print("=" * 65)
    print("🇮🇳 AI4Bharat Pralekha Seq2Seq Transformer Training Pipeline")
    print("=" * 65)
    print(f"Device: {dev} ({gpu_name})")
    print(f"Mixed Precision (AMP): {use_amp and is_cuda}")
    print(f"Target Output Directory: {out_dir}")
    print(f"Dataset: {dataset_description}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print(f"Architecture: d_model={d_model}, nhead={nhead}, layers={num_encoder_layers}e/{num_decoder_layers}d")

    # 1. Load DataLoaders and Tokenizer from pralekha_dataset
    print("\n[1/4] Loading Pralekha dataset and preparing PyTorch DataLoaders...")
    train_loader, val_loader, test_loader, tokenizer = get_pralekha_dataloaders(
        pairs=pairs,
        batch_size=batch_size,
        max_source_length=max_len,
        max_target_length=max_len,
        pin_memory=is_cuda
    )

    print(f"✓ Vocabulary Size: {tokenizer.vocab_size} tokens across 22 Scheduled Indic languages + English")
    print(f"✓ Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # 2. Build Model Architecture
    print("\n[2/4] Initializing Seq2Seq Transformer Model Architecture...")
    config = PralekhaSeq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_languages=len(ALL_LANGUAGES)
    )

    model = PralekhaSeq2SeqTransformer(config).to(dev)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total Trainable Parameters: {num_params:,} (~{num_params / 1e6:.2f}M)")

    # 3. Setup Optimizer & Mixed Precision Scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and is_cuda)

    # 4. Training Loop
    print("\n[3/4] Executing Training Epochs on GPU...")
    history: List[Dict[str, float]] = []

    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        t_epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            decoder_input_ids = batch["decoder_input_ids"].to(dev, non_blocking=True)
            labels = batch["labels"].to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if is_cuda and use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels
                    )
                    loss = outputs["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels
                )
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / max(len(train_loader), 1)

        # Validation step
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(dev, non_blocking=True)
                attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                decoder_input_ids = batch["decoder_input_ids"].to(dev, non_blocking=True)
                labels = batch["labels"].to(dev, non_blocking=True)

                if is_cuda and use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            labels=labels
                        )
                        val_loss_step = outputs["loss"]
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels
                    )
                    val_loss_step = outputs["loss"]

                epoch_val_loss += val_loss_step.item()

        avg_val_loss = epoch_val_loss / max(len(val_loader), 1)
        epoch_dur = time.time() - t_epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "duration_seconds": round(epoch_dur, 2)
        })

        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Time: {epoch_dur:.2f}s"
        )

        final_train_loss = avg_train_loss
        final_val_loss = avg_val_loss

    # 5. Save Final Artifacts
    print("\n[4/4] Saving Checkpoints and Model Weights...")
    saved_paths = save_pralekha_checkpoint(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        epoch=epochs,
        train_loss=final_train_loss,
        val_loss=final_val_loss,
        output_dir=out_dir,
        dataset_desc=dataset_description
    )

    total_time = time.time() - start_time
    print("=" * 65)
    print("✓ AI4Bharat Pralekha Model Training Complete!")
    print(f"Total Duration: {total_time:.2f}s")
    print(f"Checkpoint (.pt):        {saved_paths['checkpoint']}")
    print(f"Safetensors (.safetensors): {saved_paths['safetensors']}")
    print(f"Config (.json):          {saved_paths['config']}")
    print(f"Vocabulary (.json):      {saved_paths['vocab']}")
    print("=" * 65)

    return {
        "status": "success",
        "epochs": epochs,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "total_time_seconds": round(total_time, 2),
        "saved_paths": {k: str(v) for k, v in saved_paths.items()},
        "history": history
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train AI4Bharat Pralekha translation model")
    parser.add_argument("--dataset", type=str, default=None, help="Path to Pralekha parallel corpus")
    parser.add_argument("--db-path", type=str, default="backend/data/pralekha_strict_12way.db", help="Path to strict 12-way SQLite database")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=512, help="Embedding and hidden dimension")
    parser.add_argument("--max-len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--max-pairs", type=int, default=30000, help="Max sentence pairs to train on")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    train_pralekha_model(
        dataset_path=args.dataset,
        db_path=args.db_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        d_model=args.d_model,
        max_len=args.max_len,
        max_pairs=args.max_pairs,
        device=args.device
    )
