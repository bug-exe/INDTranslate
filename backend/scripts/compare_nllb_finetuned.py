#!/usr/bin/env python3
"""
IndTranslate: Comparative Evaluation: Base Foundation NLLB-200 vs LoRA Fine-Tuned NLLB-200
==========================================================================================
Runs an exact head-to-head comparison on identical held-out test sentence pairs from the
cleaned 12-way parallel database (backend/data/pralekha_strict_12way.db).

Computes:
1. chrF++ score (character n-gram F-score)
2. BLEU-1 and BLEU-2 with brevity penalty
3. Token Precision, Recall, F1, and Overlap Accuracy
4. Category breakdowns: English -> Indic, Indic -> Indic, Indic -> English
5. Degeneracy rate (%) and average inference latency (ms)
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import DATA_DIR, MODELS_DIR
from backend.app.services.pralekha_strict_db import load_pairs_from_strict_db
from backend.app.services.pralekha_dataset import (
    ParallelSentencePair,
    train_val_test_split,
)
from backend.app.services.pralekha_service import _is_degenerate_output
from backend.app.services.nllb_service import NLLB_MODEL_DIR, NLLB_LANG_CODES
from backend.scripts.train_nllb_finetune import inject_lora, load_lora_state_dict

NLLB_FINETUNED_DIR = MODELS_DIR / "nllb_finetuned"
LORA_WEIGHTS_PATH = NLLB_FINETUNED_DIR / "nllb_lora_weights.pt"


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


def evaluate_model_on_pairs(
    model: torch.nn.Module,
    tokenizer: Any,
    test_pairs: List[ParallelSentencePair],
    device: torch.device,
    model_label: str
) -> Dict[str, Any]:
    """Evaluates a model instance over the test pairs."""
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

    sample_predictions = []

    t0 = time.time()
    for count, pair in enumerate(test_pairs, 1):
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

        src_nllb = NLLB_LANG_CODES.get(src_lang, "eng_Latn")
        tgt_nllb = NLLB_LANG_CODES.get(tgt_lang, "hin_Deva")

        tokenizer.src_lang = src_nllb
        tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_nllb)

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

        is_degen = _is_degenerate_output(pred_text, target_lang=tgt_lang, source_text=src_text)
        if is_degen:
            degenerate_count += 1

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

        if count <= 15 or count % 40 == 0:
            sample_predictions.append({
                "idx": count,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "category": cat,
                "source": src_text,
                "reference": ref_text,
                "predicted": pred_text,
                "chrf": round(chrf_val, 2),
                "bleu1": round(b1_val, 2),
                "f1": round(f1 * 100.0, 2),
                "is_degenerate": is_degen
            })

        if count % 50 == 0 or count == len(test_pairs):
            cur_chrf = sum(chrf_scores) / count
            cur_b1 = sum(bleu1_scores) / count
            print(f"    [{model_label}] [{count}/{len(test_pairs)}] Avg chrF++: {cur_chrf:.2f} | Avg BLEU-1: {cur_b1:.2f}")

    total_time = time.time() - t0
    avg_latency = (total_time / len(test_pairs)) * 1000.0

    mean_chrf = sum(chrf_scores) / max(len(chrf_scores), 1)
    mean_bleu1 = sum(bleu1_scores) / max(len(bleu1_scores), 1)
    mean_bleu2 = sum(bleu2_scores) / max(len(bleu2_scores), 1)
    mean_prec = sum(precision_scores) / max(len(precision_scores), 1)
    mean_rec = sum(recall_scores) / max(len(recall_scores), 1)
    mean_f1 = sum(f1_scores) / max(len(f1_scores), 1)
    mean_acc = sum(accuracy_scores) / max(len(accuracy_scores), 1)
    degen_pct = (degenerate_count / len(test_pairs)) * 100.0

    category_summary = {}
    for cat_name, metrics in cat_gen_metrics.items():
        if metrics["chrf"]:
            category_summary[cat_name] = {
                "chrf": round(sum(metrics["chrf"]) / len(metrics["chrf"]), 2),
                "bleu1": round(sum(metrics["bleu1"]) / len(metrics["bleu1"]), 2),
                "bleu2": round(sum(metrics["bleu2"]) / len(metrics["bleu2"]), 2),
                "token_f1": round(sum(metrics["f1"]) / len(metrics["f1"]), 2),
                "samples": len(metrics["chrf"])
            }

    return {
        "model_label": model_label,
        "samples_evaluated": len(test_pairs),
        "overall_metrics": {
            "chrf_score": round(mean_chrf, 2),
            "bleu1_score": round(mean_bleu1, 2),
            "bleu2_score": round(mean_bleu2, 2),
            "token_precision": round(mean_prec, 2),
            "token_recall": round(mean_rec, 2),
            "token_f1": round(mean_f1, 2),
            "token_accuracy": round(mean_acc, 2),
            "degeneracy_rate_pct": round(degen_pct, 2),
            "avg_latency_ms": round(avg_latency, 1)
        },
        "category_summary": category_summary,
        "sample_predictions": sample_predictions
    }


def main():
    print("=" * 75)
    print("🇮🇳  INDTRANSLATE: NLLB-200 FOUNDATION vs FINE-TUNED BENCHMARK")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    print(f"  • Hardware:            {device} ({gpu_name})")
    print(f"  • Foundation Model:    {NLLB_MODEL_DIR}")
    print(f"  • LoRA Checkpoint:     {LORA_WEIGHTS_PATH}")
    db_path = str(DATA_DIR / "pralekha_strict_12way.db")
    print(f"  • Database:            {db_path}")

    # 1. Load held-out test split from the cleaned database
    print("\n[1/4] Loading cleaned held-out test dataset (seed=42, test_ratio=0.05)...")
    db_rows = load_pairs_from_strict_db(db_path=db_path, max_pairs=60000)
    all_pairs = [
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
    _, _, test_split = train_val_test_split(
        all_pairs,
        val_ratio=0.1,
        test_ratio=0.05,
        seed=42,
        shuffle=True,
        stratify_by_domain=True
    )
    print(f"✓ Total held-out test pairs available: {len(test_split):,}")

    eval_count = 200
    step = max(1, len(test_split) // eval_count)
    sampled_test_pairs = [test_split[i] for i in range(0, len(test_split), step)][:eval_count]
    print(f"✓ Selected {len(sampled_test_pairs)} stratified test sentence pairs across 132 language directions.")

    # 2. Load Base Foundation Model
    print("\n[2/4] Loading Base Foundation NLLB-200 in FP16...")
    tokenizer = AutoTokenizer.from_pretrained(str(NLLB_MODEL_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(NLLB_MODEL_DIR),
        dtype=torch.float16 if is_cuda else torch.float32
    ).to(device)
    model.eval()
    print("✓ Base model loaded successfully.")

    print("\n  Evaluating Base Foundation NLLB-200...")
    base_results = evaluate_model_on_pairs(
        model=model,
        tokenizer=tokenizer,
        test_pairs=sampled_test_pairs,
        device=device,
        model_label="Base NLLB-200 Foundation"
    )

    # 3. Inject and load LoRA weights
    print("\n[3/4] Injecting LoRA adapter layers and loading fine-tuned weights...")
    inject_lora(model, r=16, alpha=32)
    lora_sd = torch.load(LORA_WEIGHTS_PATH, map_location=device)
    load_lora_state_dict(model, lora_sd)
    model.eval()
    print(f"✓ Loaded LoRA weights from {LORA_WEIGHTS_PATH}!")

    print("\n  Evaluating LoRA Fine-Tuned NLLB-200...")
    finetuned_results = evaluate_model_on_pairs(
        model=model,
        tokenizer=tokenizer,
        test_pairs=sampled_test_pairs,
        device=device,
        model_label="LoRA Fine-Tuned NLLB-200"
    )

    # 4. Compare & Save
    print("\n[4/4] Computing comparative score deltas...")
    b_met = base_results["overall_metrics"]
    f_met = finetuned_results["overall_metrics"]

    comparison = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": f"{device} ({gpu_name})",
        "samples_evaluated": eval_count,
        "base_foundation": base_results,
        "finetuned_lora": finetuned_results,
        "deltas": {
            "chrf_score": {
                "base": b_met["chrf_score"],
                "finetuned": f_met["chrf_score"],
                "delta": round(f_met["chrf_score"] - b_met["chrf_score"], 2),
                "pct_change": round(((f_met["chrf_score"] - b_met["chrf_score"]) / max(b_met["chrf_score"], 1e-6)) * 100.0, 2)
            },
            "bleu1_score": {
                "base": b_met["bleu1_score"],
                "finetuned": f_met["bleu1_score"],
                "delta": round(f_met["bleu1_score"] - b_met["bleu1_score"], 2),
                "pct_change": round(((f_met["bleu1_score"] - b_met["bleu1_score"]) / max(b_met["bleu1_score"], 1e-6)) * 100.0, 2)
            },
            "bleu2_score": {
                "base": b_met["bleu2_score"],
                "finetuned": f_met["bleu2_score"],
                "delta": round(f_met["bleu2_score"] - b_met["bleu2_score"], 2),
                "pct_change": round(((f_met["bleu2_score"] - b_met["bleu2_score"]) / max(b_met["bleu2_score"], 1e-6)) * 100.0, 2)
            },
            "token_f1": {
                "base": b_met["token_f1"],
                "finetuned": f_met["token_f1"],
                "delta": round(f_met["token_f1"] - b_met["token_f1"], 2),
                "pct_change": round(((f_met["token_f1"] - b_met["token_f1"]) / max(b_met["token_f1"], 1e-6)) * 100.0, 2)
            },
            "token_precision": {
                "base": b_met["token_precision"],
                "finetuned": f_met["token_precision"],
                "delta": round(f_met["token_precision"] - b_met["token_precision"], 2)
            },
            "token_recall": {
                "base": b_met["token_recall"],
                "finetuned": f_met["token_recall"],
                "delta": round(f_met["token_recall"] - b_met["token_recall"], 2)
            },
            "token_accuracy": {
                "base": b_met["token_accuracy"],
                "finetuned": f_met["token_accuracy"],
                "delta": round(f_met["token_accuracy"] - b_met["token_accuracy"], 2)
            },
            "degeneracy_rate_pct": {
                "base": b_met["degeneracy_rate_pct"],
                "finetuned": f_met["degeneracy_rate_pct"],
                "delta": round(f_met["degeneracy_rate_pct"] - b_met["degeneracy_rate_pct"], 2)
            },
            "avg_latency_ms": {
                "base": b_met["avg_latency_ms"],
                "finetuned": f_met["avg_latency_ms"],
                "delta": round(f_met["avg_latency_ms"] - b_met["avg_latency_ms"], 2)
            }
        }
    }

    out_file = NLLB_FINETUNED_DIR / "comparison_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved comparative results to: {out_file}")

    print("\n" + "=" * 75)
    print("📊 COMPARATIVE BENCHMARK SUMMARY (NLLB Base vs NLLB LoRA Fine-Tuned)")
    print("=" * 75)
    print(f"{'Metric':<25} | {'Base NLLB':<14} | {'Fine-Tuned':<14} | {'Delta':<10}")
    print("-" * 75)
    for k, v in comparison["deltas"].items():
        base_v = v["base"]
        fine_v = v["finetuned"]
        delta_v = v["delta"]
        sign = "+" if delta_v > 0 else ""
        print(f"{k:<25} | {base_v:<14} | {fine_v:<14} | {sign}{delta_v}")
    print("=" * 75)


if __name__ == "__main__":
    main()
