#!/usr/bin/env python3
"""
IndTranslate: AI4Bharat Pralekha Seq2Seq Transformer Test Set Evaluation
========================================================================
Evaluates the trained Pralekha Seq2Seq model checkpoint on the held-out
test dataset (6,000 sentence pairs across all 132 language directions).

Metrics Computed:
1. Test Cross-Entropy Loss & Perplexity (PPL = exp(loss)) over all 6,000 pairs
2. Directional Breakdown:
   - English -> Indic
   - Indic -> Indic
   - Indic -> English
3. Generation Metrics on Held-out Test Samples:
   - chrF++ (Character n-gram F-score with beta=2)
   - BLEU-1 & BLEU-2 scores with brevity penalty
   - Token Precision, Recall, and F1 Score
   - Token Overlap Accuracy
   - Script Validity & Degeneracy Rate (broken combining marks check)
4. Saves full evaluation results to backend/saved_models/pralekha/test_evaluation_results.json
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
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import DATA_DIR, MODELS_DIR
from backend.app.services.pralekha_dataset import (
    ParallelSentencePair,
    PralekhaTokenizer,
    get_pralekha_dataloaders,
    ALL_LANGUAGES
)
from backend.app.services.pralekha_trainer import (
    PralekhaSeq2SeqConfig,
    PralekhaSeq2SeqTransformer,
    load_pairs_from_strict_db,
    load_pralekha_model,
    resolve_target_lang_id,
    PRALEKHA_MODEL_DIR
)
from backend.app.services.pralekha_service import _is_degenerate_output


# ---------------------------------------------------------------------------
# Metric Helper Functions
# ---------------------------------------------------------------------------

def tokenize_indic(text: str) -> List[str]:
    """Tokenize Indic and English text into normalized words."""
    cleaned = re.sub(r'[^\w\s\u0900-\u0D7F]', ' ', text)
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
    return chrf_score * 100.0  # Return percentage 0-100


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

def evaluate_test_set(
    db_path: str = "backend/data/pralekha_strict_12way.db",
    max_pairs: Optional[int] = 500000,
    batch_size: int = 32,
    gen_samples: int = 300
) -> Dict[str, Any]:
    print("=" * 75)
    print("🇮🇳  INDTRANSLATE: PRALEKHA SEQ2SEQ HELD-OUT TEST EVALUATION")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    print(f"  • Hardware Device:       {device} ({gpu_name})")
    print(f"  • Model Directory:       {PRALEKHA_MODEL_DIR}")
    print(f"  • Target Database:       {db_path}")
    print(f"  • Batch Size:            {batch_size}")
    print("=" * 75)

    # 1. Load trained model & tokenizer
    print("\n[1/4] Loading trained Pralekha model checkpoint...")
    model, tokenizer, _ = load_pralekha_model(PRALEKHA_MODEL_DIR, device=device)
    model.eval()
    print(f"✓ Model loaded successfully: {model.config.vocab_size} vocab | {model.config.d_model} d_model | {model.config.num_encoder_layers} enc / {model.config.num_decoder_layers} dec layers.")

    # 2. Re-create the deterministic dataset splits
    print("\n[2/4] Loading sentence pairs and isolating held-out test split (seed=42)...")
    pairs = load_pairs_from_strict_db(db_path=db_path, max_pairs=max_pairs)
    print(f"✓ Loaded {len(pairs):,} parallel sentence pairs across all 132 language directions.")

    _, _, test_loader, _ = get_pralekha_dataloaders(
        pairs=pairs,
        tokenizer=tokenizer,
        batch_size=batch_size,
        val_ratio=0.1,
        test_ratio=0.05,
        max_source_length=128,
        max_target_length=128,
        pin_memory=is_cuda,
        seed=42
    )

    if test_loader is None or len(test_loader) == 0:
        raise RuntimeError("Test DataLoader is empty! Check split ratios.")

    total_test_samples = len(test_loader.dataset)
    print(f"✓ Isolated held-out test split: {len(test_loader)} batches ({total_test_samples:,} sentence pairs).")

    # 3. Full Test Cross-Entropy Loss & Perplexity Computation
    print("\n[3/4] Evaluating Test Cross-Entropy Loss and Perplexity across all 6,000 pairs...")
    test_loss_accum = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    # Grouped stats by direction category
    cat_loss: Dict[str, float] = defaultdict(float)
    cat_tokens: Dict[str, int] = defaultdict(int)

    # Direction-specific stats
    dir_loss: Dict[str, float] = defaultdict(float)
    dir_tokens: Dict[str, int] = defaultdict(int)

    start_eval_time = time.time()

    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            src_langs = batch["source_langs"]
            tgt_langs = batch["target_langs"]

            if is_cuda:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels
                )

            logits = outputs["logits"]
            b_loss = outputs["loss"].item()
            test_loss_accum += b_loss

            # Per-token / per-category analysis
            for i in range(len(src_langs)):
                sl = src_langs[i]
                tl = tgt_langs[i]
                direction = f"{sl}->{tl}"

                # Category categorization
                if sl == "en":
                    category = "English -> Indic"
                elif tl == "en":
                    category = "Indic -> English"
                else:
                    category = "Indic -> Indic"

                sample_labels = labels[i]
                sample_logits = logits[i]
                mask = (sample_labels != -100)
                n_tok = mask.sum().item()

                if n_tok > 0:
                    s_loss = criterion(sample_logits[mask], sample_labels[mask]).item()
                    cat_loss[category] += s_loss
                    cat_tokens[category] += n_tok
                    dir_loss[direction] += s_loss
                    dir_tokens[direction] += n_tok
                    total_tokens += n_tok

            if b_idx % 40 == 0 or b_idx == len(test_loader):
                cur_avg = test_loss_accum / b_idx
                cur_ppl = math.exp(min(cur_avg, 20.0))
                print(f"  Batch [{b_idx}/{len(test_loader)}] | Avg Test Loss: {cur_avg:.4f} | Perplexity: {cur_ppl:.2f}")

    eval_time = time.time() - start_eval_time
    avg_test_loss = test_loss_accum / len(test_loader)
    test_perplexity = math.exp(min(avg_test_loss, 20.0))

    # Category summaries
    cat_summary = {}
    for cat in ["English -> Indic", "Indic -> Indic", "Indic -> English"]:
        if cat_tokens[cat] > 0:
            c_loss = cat_loss[cat] / cat_tokens[cat]
            c_ppl = math.exp(min(c_loss, 20.0))
            cat_summary[cat] = {
                "token_loss": round(c_loss, 4),
                "perplexity": round(c_ppl, 2),
                "total_tokens": cat_tokens[cat]
            }

    print("\n" + "-" * 75)
    print("📊 OVERALL TEST SET PERFORMANCE (6,000 sentence pairs across 132 directions):")
    print(f"  • Test Cross-Entropy Loss: {avg_test_loss:.4f}")
    print(f"  • Test Perplexity (PPL):   {test_perplexity:.2f}")
    print(f"  • Total Evaluated Tokens:  {total_tokens:,}")
    print(f"  • Evaluation Duration:     {eval_time:.2f}s ({eval_time / len(test_loader) * 1000:.1f} ms/batch)")
    print("-" * 75)
    print("🌐 CATEGORY BREAKDOWN:")
    for cat, stats in cat_summary.items():
        print(f"  • {cat:20s} | Loss: {stats['token_loss']:.4f} | PPL: {stats['perplexity']:.2f} | Tokens: {stats['total_tokens']:,}")
    print("-" * 75)

    # 4. Generation Evaluation on Stratified Test Samples
    print(f"\n[4/4] Running Autoregressive Greedy Decoding on {gen_samples} stratified test sentences...")
    
    # Stratified sampling of indices from test dataset
    step_skip = max(1, total_test_samples // gen_samples)
    sampled_indices = list(range(0, total_test_samples, step_skip))[:gen_samples]

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
        item = test_loader.dataset[idx]
        src_text = item["source_text"]
        ref_text = item["target_text"]
        src_lang = item["source_lang"]
        tgt_lang = item["target_lang"]

        if src_lang == "en":
            cat = "English -> Indic"
        elif tgt_lang == "en":
            cat = "Indic -> English"
        else:
            cat = "Indic -> Indic"

        # Model Autoregressive Generation with Target Script Logit Masking
        src_encoded = tokenizer.encode(src_text, target_lang=tgt_lang, add_special_tokens=True)
        src_tensor = torch.tensor([src_encoded], dtype=torch.long, device=device)
        src_mask = (src_tensor != tokenizer.pad_token_id).long()
        tgt_lang_id = resolve_target_lang_id(tgt_lang)

        # Precompute target script logit mask & combining marks mask
        script_mask = tokenizer.get_script_mask(tgt_lang, device=device) if hasattr(tokenizer, "get_script_mask") else None
        comb_mask = tokenizer.get_combining_marks_mask(device=device) if hasattr(tokenizer, "get_combining_marks_mask") else None

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=src_tensor,
                attention_mask=src_mask,
                target_lang=tgt_lang,
                target_lang_id=tgt_lang_id,
                logit_mask=script_mask,
                combining_marks_mask=comb_mask,
                max_length=64,
                num_beams=4,
                length_penalty=0.6,
                repetition_penalty=1.35,
                no_repeat_ngram_size=3
            )

        pred_tokens = out_ids[0].tolist()
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

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
    print("📈 GENERATION BENCHMARK SCORES (Sampled Held-out Test Set):")
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

    print("\n🌐 CATEGORY GENERATION SCORES:")
    category_gen_summary = {}
    for cat, m in cat_gen_metrics.items():
        if m["chrf"]:
            c_chrf = sum(m["chrf"]) / len(m["chrf"])
            c_b1 = sum(m["bleu1"]) / len(m["bleu1"])
            c_b2 = sum(m["bleu2"]) / len(m["bleu2"])
            c_f1 = sum(m["f1"]) / len(m["f1"])
            category_gen_summary[cat] = {
                "chrf": round(c_chrf, 2),
                "bleu1": round(c_b1, 2),
                "bleu2": round(c_b2, 2),
                "token_f1": round(c_f1, 2),
                "samples": len(m["chrf"])
            }
            print(f"  • {cat:20s} | chrF++: {c_chrf:5.2f} | BLEU-1: {c_b1:5.2f} | BLEU-2: {c_b2:5.2f} | F1: {c_f1:5.2f}% (N={len(m['chrf'])})")

    # Sample predictions display
    print("\n📝 REPRESENTATIVE SAMPLE PREDICTIONS ON HELD-OUT TEST DATA:")
    print("-" * 75)
    sample_display = [gen_results[i] for i in [0, 20, 50, 100, 150, 200, 250] if i < len(gen_results)]
    for s in sample_display:
        print(f"[{s['source_lang'].upper()} -> {s['target_lang'].upper()}] ({s['category']})")
        print(f"  Source:    {s['source_text']}")
        print(f"  Reference: {s['reference_text']}")
        print(f"  Generated: {s['predicted_text']}")
        print(f"  chrF++:    {s['chrf']:.1f} | BLEU-1: {s['bleu1']:.1f} | F1: {s['token_f1']:.1f}%")
        print()

    # Save to JSON
    output_json_path = PRALEKHA_MODEL_DIR / "test_evaluation_results.json"
    evaluation_record = {
        "model_name": "AI4Bharat Pralekha Direct Indic Seq2Seq Transformer",
        "checkpoint": "backend/saved_models/pralekha/pralekha_checkpoint.pt",
        "dataset": "backend/data/pralekha_strict_12way.db",
        "total_test_pairs": total_test_samples,
        "test_batches": len(test_loader),
        "batch_size": batch_size,
        "test_loss": round(avg_test_loss, 4),
        "test_perplexity": round(test_perplexity, 2),
        "total_tokens_evaluated": total_tokens,
        "category_loss_summary": cat_summary,
        "generation_metrics": {
            "strategy": "script_logit_masking_with_bos_conditioning",
            "samples_evaluated": len(sampled_indices),
            "chrf_score": round(mean_chrf, 2),
            "bleu1_score": round(mean_bleu1, 2),
            "bleu2_score": round(mean_bleu2, 2),
            "token_precision": round(mean_prec, 2),
            "token_recall": round(mean_rec, 2),
            "token_f1": round(mean_f1, 2),
            "token_accuracy": round(mean_acc, 2),
            "degeneracy_rate_pct": round(degen_rate, 2),
            "avg_latency_ms": round(gen_duration / max(len(sampled_indices), 1) * 1000, 1),
            "category_summary": category_gen_summary
        },
        "sample_predictions": gen_results[:25],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_record, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved full test evaluation results to: {output_json_path}")
    return evaluation_record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Pralekha model on held-out test set")
    parser.add_argument("--db-path", type=str, default="backend/data/pralekha_strict_12way.db")
    parser.add_argument("--max-pairs", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gen-samples", type=int, default=300)
    args = parser.parse_args()

    evaluate_test_set(
        db_path=args.db_path,
        max_pairs=args.max_pairs,
        batch_size=args.batch_size,
        gen_samples=args.gen_samples
    )
