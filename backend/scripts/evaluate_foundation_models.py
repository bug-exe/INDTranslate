#!/usr/bin/env python3
"""
IndTranslate: Foundation Model Baseline Evaluation
==================================================
Evaluates Meta NLLB-200 Distilled 600M foundation model on the exact same
held-out test dataset (6,000 pairs across 132 directions, seed=42) from
backend/data/pralekha_strict_12way.db to establish foundation baseline scores.

Metrics Computed:
1. chrF++ (Character n-gram F-score with beta=2)
2. BLEU-1 & BLEU-2 scores with brevity penalty
3. Token Precision, Recall, F1, and Overlap Accuracy
4. Category Breakdown:
   - English -> Indic
   - Indic -> Indic
   - Indic -> English
5. Degeneracy Rate and Script Validity
6. Inference Latency on RTX 4050 GPU
"""

import os
import sys
import math
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import DATA_DIR, MODELS_DIR
from backend.app.services.pralekha_strict_db import load_pairs_from_strict_db
from backend.app.services.pralekha_dataset import (
    ParallelSentencePair,
    train_val_test_split,
    ALL_LANGUAGES
)
from backend.app.services.pralekha_service import _is_degenerate_output


# ---------------------------------------------------------------------------
# NLLB-200 Language Code Mapping
# ---------------------------------------------------------------------------

NLLB_LANG_CODES: Dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
    "as": "asm_Beng",
    "mai": "mai_Deva",
    "mni": "mni_Beng",
    "ne": "npi_Deva",
    "sa": "san_Deva",
    "sd": "snd_Arab",
    "ks": "kas_Arab"
}


# ---------------------------------------------------------------------------
# Metric Helper Functions
# ---------------------------------------------------------------------------

def tokenize_indic(text: str) -> List[str]:
    """Tokenize Indic and English text into normalized words."""
    cleaned = re.sub(r'[^\w\s\u0900-\u0D7F\u0600-\u06FF]', ' ', text)
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    return tokens


def compute_token_metrics(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float, float, float]:
    """Computes Token Precision, Recall, F1, and Token Overlap Accuracy."""
    if not ref_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0, 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    tp = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts if token in ref_counts)

    precision = tp / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = tp / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    total_distinct_mass = max(len(ref_tokens), len(pred_tokens))
    accuracy = tp / total_distinct_mass if total_distinct_mass > 0 else 0.0

    return precision, recall, f1, accuracy


def compute_chrf(pred_text: str, ref_text: str, n: int = 4, beta: float = 2.0) -> float:
    """Computes Character n-gram F-score (chrF++)."""
    def get_char_ngrams(s: str, n_val: int) -> Counter:
        norm = s.replace(" ", "")
        ngrams = [norm[i:i+n_val] for i in range(len(norm) - n_val + 1)]
        return Counter(ngrams)

    if not pred_text or not ref_text:
        return 0.0

    precisions = []
    recalls = []
    for k in range(1, n + 1):
        pred_ng = get_char_ngrams(pred_text, k)
        ref_ng = get_char_ngrams(ref_text, k)
        match = sum(min(pred_ng[ng], ref_ng[ng]) for ng in pred_ng if ng in ref_ng)
        total_p = sum(pred_ng.values())
        total_r = sum(ref_ng.values())
        p_k = match / total_p if total_p > 0 else 0.0
        r_k = match / total_r if total_r > 0 else 0.0
        precisions.append(p_k)
        recalls.append(r_k)

    avg_p = sum(precisions) / len(precisions) if precisions else 0.0
    avg_r = sum(recalls) / len(recalls) if recalls else 0.0

    if avg_p + avg_r == 0:
        return 0.0
    chrf_score = (1 + beta**2) * (avg_p * avg_r) / ((beta**2 * avg_p) + avg_r)
    return chrf_score * 100.0


def compute_bleu(pred_tokens: List[str], ref_tokens: List[str]) -> Tuple[float, float]:
    """Computes BLEU-1 and BLEU-2 scores with standard brevity penalty."""
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)

    # 1-gram
    tp1 = sum(min(pred_counts[t], ref_counts[t]) for t in pred_counts if t in ref_counts)
    p1 = tp1 / len(pred_tokens) if len(pred_tokens) > 0 else 0.0

    # 2-gram
    pred_2grams = [f"{pred_tokens[i]}_{pred_tokens[i+1]}" for i in range(len(pred_tokens) - 1)]
    ref_2grams = [f"{ref_tokens[i]}_{ref_tokens[i+1]}" for i in range(len(ref_tokens) - 1)]
    if pred_2grams and ref_2grams:
        p2_counts = Counter(pred_2grams)
        r2_counts = Counter(ref_2grams)
        tp2 = sum(min(p2_counts[g], r2_counts[g]) for g in p2_counts if g in r2_counts)
        p2 = tp2 / len(pred_2grams)
    else:
        p2 = 0.0

    # Brevity penalty
    c = len(pred_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else (math.exp(1 - r / c) if c > 0 else 0.0)

    bleu1 = bp * p1 * 100.0
    bleu2 = bp * math.sqrt(p1 * p2) * 100.0 if (p1 > 0 and p2 > 0) else 0.0

    return bleu1, bleu2


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------

def evaluate_nllb_foundation(
    model_dir: str = "backend/saved_models/foundation/nllb-200-distilled-600M",
    db_path: str = "backend/data/pralekha_strict_12way.db",
    max_pairs: Optional[int] = 120000,
    gen_samples: int = 300
) -> Dict[str, Any]:
    print("=" * 75)
    print("🇮🇳  INDTRANSLATE: FOUNDATION MODEL BASELINE EVALUATION (NLLB-200 600M)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    print(f"  • Hardware Device:       {device} ({gpu_name})")
    print(f"  • Model Directory:       {model_dir}")
    print(f"  • Target Database:       {db_path}")
    print(f"  • Evaluated Samples:     {gen_samples} stratified test sentences")
    print("=" * 75)

    # 1. Load NLLB-200 Model & Tokenizer
    print("\n[1/3] Loading NLLB-200 Distilled 600M in FP16...")
    t0_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        dtype=torch.float16 if is_cuda else torch.float32
    ).to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded in {time.time() - t0_load:.2f}s | Parameters: {total_params:,} ({total_params / 1e6:.1f}M) | Vocab: {len(tokenizer):,}")

    # 2. Isolate exact same held-out test dataset
    print("\n[2/3] Isolating held-out test split (seed=42, test_ratio=0.05)...")
    db_rows = load_pairs_from_strict_db(db_path=db_path, max_pairs=max_pairs)
    pairs = [
        ParallelSentencePair(
            source_text=r["source_text"],
            target_text=r["target_text"],
            source_lang=r["source_lang"],
            target_lang=r["target_lang"],
            domain=r.get("split", "strict_12way"),
            id=f"{r.get('doc_id', 'doc')}_{r['source_lang']}_{r['target_lang']}"
        )
        for r in db_rows
    ]
    print(f"✓ Loaded {len(pairs):,} parallel sentence pairs across all 132 language directions.")

    _, _, test_pairs = train_val_test_split(
        pairs,
        val_ratio=0.1,
        test_ratio=0.05,
        seed=42,
        shuffle=True,
        stratify_by_domain=True
    )
    total_test = len(test_pairs)
    print(f"✓ Isolated held-out test split: {total_test:,} sentence pairs.")

    # 3. Stratified Evaluation
    print(f"\n[3/3] Generating translations on {gen_samples} stratified test sentence pairs...")
    step_skip = max(1, total_test // gen_samples)
    sampled_indices = list(range(0, total_test, step_skip))[:gen_samples]

    gen_results = []
    chrf_scores = []
    bleu1_scores = []
    bleu2_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []
    degenerate_count = 0

    cat_gen_metrics: Dict[str, Dict[str, List[float]]] = {
        "English -> Indic": defaultdict(list),
        "Indic -> Indic": defaultdict(list),
        "Indic -> English": defaultdict(list),
    }

    t0_gen = time.time()
    for count, idx in enumerate(sampled_indices, 1):
        pair = test_pairs[idx]
        src_text = pair.source_text
        ref_text = pair.target_text
        src_lang = pair.source_lang
        tgt_lang = pair.target_lang

        if src_lang == "en":
            cat = "English -> Indic"
        elif tgt_lang == "en":
            cat = "Indic -> English"
        else:
            cat = "Indic -> Indic"

        # Resolve NLLB language codes
        src_nllb = NLLB_LANG_CODES.get(src_lang, "eng_Latn")
        tgt_nllb = NLLB_LANG_CODES.get(tgt_lang, "hin_Deva")

        # Set tokenizer source language
        tokenizer.src_lang = src_nllb
        tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_nllb)

        # Autoregressive generation
        inputs = tokenizer(src_text, return_tensors="pt", max_length=128, truncation=True).to(device)

        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_length=128,
                num_beams=1,
                repetition_penalty=1.15
            )

        pred_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()

        # Check degeneracy
        is_degen = _is_degenerate_output(pred_text, target_lang=tgt_lang)
        if is_degen:
            degenerate_count += 1

        # Calculate metrics
        pred_words = tokenize_indic(pred_text)
        ref_words = tokenize_indic(ref_text)

        chrf_val = compute_chrf(pred_text, ref_text)
        b1_val, b2_val = compute_bleu(pred_words, ref_words)
        prec, rec, f1, acc = compute_token_metrics(pred_words, ref_words)

        chrf_scores.append(chrf_val)
        bleu1_scores.append(b1_val)
        bleu2_scores.append(b2_val)
        precision_scores.append(prec * 100.0)
        recall_scores.append(rec * 100.0)
        f1_scores.append(f1 * 100.0)
        accuracy_scores.append(acc * 100.0)

        cat_gen_metrics[cat]["chrf"].append(chrf_val)
        cat_gen_metrics[cat]["bleu1"].append(b1_val)
        cat_gen_metrics[cat]["bleu2"].append(b2_val)
        cat_gen_metrics[cat]["f1"].append(f1 * 100.0)

        gen_results.append({
            "idx": idx,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "category": cat,
            "source_text": src_text,
            "reference_text": ref_text,
            "predicted_text": pred_text,
            "chrf": round(chrf_val, 2),
            "bleu1": round(b1_val, 2),
            "bleu2": round(b2_val, 2),
            "token_f1": round(f1 * 100.0, 2),
            "is_degenerate": is_degen
        })

        if count % 50 == 0 or count == len(sampled_indices):
            cur_chrf = sum(chrf_scores) / count
            cur_b1 = sum(bleu1_scores) / count
            print(f"  Evaluated [{count}/{len(sampled_indices)}] sentences | Avg chrF++: {cur_chrf:.2f} | Avg BLEU-1: {cur_b1:.2f}")

    gen_duration = time.time() - t0_gen

    # Aggregates
    mean_chrf = sum(chrf_scores) / max(len(chrf_scores), 1)
    mean_bleu1 = sum(bleu1_scores) / max(len(bleu1_scores), 1)
    mean_bleu2 = sum(bleu2_scores) / max(len(bleu2_scores), 1)
    mean_prec = sum(precision_scores) / max(len(precision_scores), 1)
    mean_rec = sum(recall_scores) / max(len(recall_scores), 1)
    mean_f1 = sum(f1_scores) / max(len(f1_scores), 1)
    mean_acc = sum(accuracy_scores) / max(len(accuracy_scores), 1)
    degen_rate = (degenerate_count / len(sampled_indices)) * 100.0

    print("\n" + "=" * 75)
    print("📈 NLLB-200 FOUNDATION BASELINE SCORES:")
    print("=" * 75)
    print(f"  • chrF++ Score:            {mean_chrf:.2f} / 100.0")
    print(f"  • BLEU-1 Score:            {mean_bleu1:.2f} / 100.0")
    print(f"  • BLEU-2 Score:            {mean_bleu2:.2f} / 100.0")
    print(f"  • Token Precision:         {mean_prec:.2f}%")
    print(f"  • Token Recall:            {mean_rec:.2f}%")
    print(f"  • Token F1 Score:          {mean_f1:.2f}%")
    print(f"  • Token Overlap Accuracy:  {mean_acc:.2f}%")
    print(f"  • Degeneracy Rate:         {degen_rate:.2f}% ({degenerate_count}/{len(sampled_indices)} invalid outputs)")
    print(f"  • Generation Latency:      {gen_duration / len(sampled_indices) * 1000:.1f} ms/sentence")
    print("=" * 75)

    print("\n🌐 CATEGORY BASELINE SCORES:")
    category_summary = {}
    for cat, m in cat_gen_metrics.items():
        if m["chrf"]:
            c_chrf = sum(m["chrf"]) / len(m["chrf"])
            c_b1 = sum(m["bleu1"]) / len(m["bleu1"])
            c_b2 = sum(m["bleu2"]) / len(m["bleu2"])
            c_f1 = sum(m["f1"]) / len(m["f1"])
            category_summary[cat] = {
                "chrf": round(c_chrf, 2),
                "bleu1": round(c_b1, 2),
                "bleu2": round(c_b2, 2),
                "token_f1": round(c_f1, 2),
                "samples": len(m["chrf"])
            }
            print(f"  • {cat:20s} | chrF++: {c_chrf:5.2f} | BLEU-1: {c_b1:5.2f} | BLEU-2: {c_b2:5.2f} | F1: {c_f1:5.2f}% (N={len(m['chrf'])})")

    # Sample predictions display
    print("\n📝 REPRESENTATIVE NLLB-200 BASELINE PREDICTIONS:")
    print("-" * 75)
    sample_display = [gen_results[i] for i in [0, 20, 50, 100, 150, 200, 250] if i < len(gen_results)]
    for s in sample_display:
        print(f"[{s['source_lang'].upper()} -> {s['target_lang'].upper()}] ({s['category']})")
        print(f"  Source:    {s['source_text'][:120]}...")
        print(f"  Reference: {s['reference_text'][:120]}...")
        print(f"  Generated: {s['predicted_text'][:120]}...")
        print(f"  chrF++:    {s['chrf']:.1f} | BLEU-1: {s['bleu1']:.1f} | F1: {s['token_f1']:.1f}%")
        print()

    # Save to JSON
    out_dir = Path("backend/saved_models/foundation")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = out_dir / "foundation_evaluation_results.json"

    evaluation_record = {
        "model_name": "Meta NLLB-200 Distilled 600M Foundation Model",
        "checkpoint_path": model_dir,
        "total_parameters": total_params,
        "dataset": db_path,
        "samples_evaluated": len(sampled_indices),
        "overall_metrics": {
            "chrf_score": round(mean_chrf, 2),
            "bleu1_score": round(mean_bleu1, 2),
            "bleu2_score": round(mean_bleu2, 2),
            "token_precision": round(mean_prec, 2),
            "token_recall": round(mean_rec, 2),
            "token_f1": round(mean_f1, 2),
            "token_accuracy": round(mean_acc, 2),
            "degeneracy_rate_pct": round(degen_rate, 2),
            "avg_latency_ms": round(gen_duration / len(sampled_indices) * 1000, 1)
        },
        "category_summary": category_summary,
        "sample_predictions": gen_results[:25],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_record, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved full foundation baseline evaluation results to: {output_json_path}")
    return evaluation_record


if __name__ == "__main__":
    evaluate_nllb_foundation()
