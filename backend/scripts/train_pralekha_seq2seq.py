#!/usr/bin/env python3
"""
AI4Bharat Pralekha Seq2Seq Transformer Training Script.
Trains a direct multilingual translation model across 132 language directions
(English + 11 Indic languages: hi, bn, gu, kn, ml, mr, or, pa, ta, te, ur)
using the strict 12-way parallel corpus in backend/data/pralekha_strict_12way.db.

Features:
- PyTorch AMP (FP16) mixed precision acceleration for NVIDIA RTX 4050 GPU.
- Forced target language conditioning (<2lang>) to prevent mode collapse.
- Dynamic learning rate schedule with warmup & Cosine Annealing.
- Live sample translation generation after each epoch.
- Dual serialization: PyTorch .pt checkpoint and zero-copy .safetensors.
- Live cache reload for the IndTranslate backend server.
"""

import os
import sys
import time
import math
import json
import sqlite3
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

# Add project root to sys.path
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
    PralekhaSentencePieceTokenizer,
    load_pairs_from_strict_db,
    save_pralekha_checkpoint,
    resolve_target_lang_id,
    PRALEKHA_MODEL_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_pralekha")

# Benchmark test prompts to evaluate after each epoch
BENCHMARK_PROMPTS = [
    ("en", "hi", "The weather is nice today."),
    ("en", "ta", "The weather is nice today."),
    ("en", "ur", "The weather is nice today."),
    ("en", "mr", "I am going to school."),
    ("hi", "bn", "आज मौसम बहुत अच्छा है।"),
    ("en", "gu", "Welcome to India.")
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AI4Bharat Pralekha Seq2Seq Multilingual Model"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="backend/data/pralekha_strict_12way.db",
        help="Path to strict 12-way parallel SQLite database"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backend/saved_models/pralekha",
        help="Directory to save trained model weights and configs"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="edge-42m",
        choices=["edge-42m", "tiny-demo", "custom"],
        help="Model architecture preset: 'edge-42m' (512d, 6e/6d, 2048ff) or 'tiny-demo' (256d, 3e/3d, 1024ff)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate for AdamW optimizer (default: 3e-4)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1500,
        help="Linear warmup steps before cosine decay (default: 1500)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Mixed precision type: 'bf16' (bfloat16), 'fp16', 'fp32', or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=50000,
        help="Max parallel sentence pairs to train on (0 = all available)"
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Transformer model embedding dimension (default: 512)"
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of multi-head attention heads (default: 8)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of encoder and decoder layers (default: 6)"
    )
    parser.add_argument(
        "--dim-ff",
        type=int,
        default=2048,
        help="Feedforward network dimension (default: 2048)"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for Beam Search decoding evaluation (default: 4)"
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.6,
        help="Beam Search length penalty alpha (default: 0.6)"
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="auto",
        choices=["auto", "spm", "char"],
        help="Tokenizer type: 'spm' (SentencePiece BPE), 'char' (PralekhaTokenizer), or 'auto'"
    )
    parser.add_argument(
        "--spm-model-path",
        type=str,
        default="backend/saved_models/pralekha/sentencepiece.bpe.model",
        help="Path to SentencePiece BPE model file (.model)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability (default: 0.1)"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=128,
        help="Maximum token sequence length (default: 128)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: 'cuda' or 'cpu' (default: autodetect)"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable FP16 Automatic Mixed Precision (AMP)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume training from existing checkpoint in output-dir if present (default: True)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not resume; start training from random initialization"
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Enable early stopping when validation loss stops decreasing (default: True)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs without validation loss improvement before stopping (default: 3)"
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.005,
        help="Minimum improvement in validation loss to qualify as a decrease (default: 0.005)"
    )
    return parser.parse_args()


def generate_sample_translations(
    model: PralekhaSeq2SeqTransformer,
    tokenizer: Any,
    device: torch.device,
    num_beams: int = 4,
    length_penalty: float = 0.6
) -> None:
    """Runs beam search decoded sample translations to visualize generation quality."""
    model.eval()
    print(f"\n  [Live Sample Generation (Beam Search k={num_beams}, alpha={length_penalty})]:")
    with torch.no_grad():
        for src_lang, tgt_lang, prompt in BENCHMARK_PROMPTS:
            try:
                # Encode source with target lang routing tag (e.g., <2hi>)
                src_encoded = tokenizer.encode(prompt, target_lang=tgt_lang, add_special_tokens=True)
                src_tensor = torch.tensor([src_encoded], dtype=torch.long, device=device)
                src_mask = (src_tensor != tokenizer.pad_token_id).long()

                # Generate autoregressively with Beam Search
                tgt_lang_id = resolve_target_lang_id(tgt_lang)
                out_ids = model.generate(
                    input_ids=src_tensor,
                    attention_mask=src_mask,
                    target_lang_id=tgt_lang_id,
                    max_length=64,
                    num_beams=num_beams,
                    length_penalty=length_penalty
                )
                pred_tokens = out_ids[0].tolist()
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                print(f"    • [{src_lang} -> {tgt_lang}] '{prompt}' => '{pred_text}'")
            except Exception as e:
                print(f"    • [{src_lang} -> {tgt_lang}] Generation error: {e}")
    print()


def run_training():
    args = parse_arguments()
    start_total_time = time.time()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device Setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_cuda = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"

    # Precision configuration resolution
    if args.no_amp or not is_cuda:
        precision_type = "fp32"
        amp_dtype = None
        use_amp = False
    elif args.precision == "auto":
        if torch.cuda.is_bf16_supported():
            precision_type = "bf16"
            amp_dtype = torch.bfloat16
            use_amp = True
        else:
            precision_type = "fp16"
            amp_dtype = torch.float16
            use_amp = True
    elif args.precision == "bf16":
        precision_type = "bf16"
        amp_dtype = torch.bfloat16
        use_amp = True
    elif args.precision == "fp16":
        precision_type = "fp16"
        amp_dtype = torch.float16
        use_amp = True
    else:
        precision_type = "fp32"
        amp_dtype = None
        use_amp = False

    use_scaler = use_amp and (amp_dtype == torch.float16)

    # Architecture configuration resolution
    if args.arch == "edge-42m":
        d_model = 512 if args.d_model == 512 else args.d_model
        nhead = 8 if args.nhead == 8 else args.nhead
        num_layers = 6 if args.num_layers == 6 else args.num_layers
        dim_ff = 2048 if args.dim_ff == 2048 else args.dim_ff
    elif args.arch == "tiny-demo":
        d_model = 256
        nhead = 8
        num_layers = 3
        dim_ff = 1024
    else:
        d_model = args.d_model
        nhead = args.nhead
        num_layers = args.num_layers
        dim_ff = args.dim_ff

    print("=" * 70)
    print("🇮🇳  INDTRANSLATE: AI4BHARAT PRALEKHA SEQ2SEQ MODEL TRAINER")
    print("=" * 70)
    print(f"  • Hardware Device:       {device} ({gpu_name})")
    print(f"  • Mixed Precision (AMP): {use_amp} ({precision_type.upper()})")
    print(f"  • Target Database:       {args.db_path}")
    print(f"  • Checkpoint Directory:  {out_dir}")
    print(f"  • Architecture Preset:   {args.arch.upper()} (layers={num_layers}e/{num_layers}d, d_model={d_model}, dim_ff={dim_ff}, Pre-LN=True)")
    print(f"  • Hyperparameters:       epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, warmup_steps={args.warmup_steps}")
    print(f"  • Evaluation Decoding:   Beam Search (k={args.num_beams}, alpha={args.length_penalty})")
    print("=" * 70)

    # 2. Dataset Loading
    max_p = None if args.max_pairs <= 0 else args.max_pairs
    print(f"\n[1/4] Loading sentence pairs from strict 12-way database...")
    raw_pairs = load_pairs_from_strict_db(db_path=args.db_path, max_pairs=max_p)
    if not raw_pairs:
        print(f"❌ Error: No training pairs loaded from {args.db_path}. Verify database existence.")
        sys.exit(1)

    if raw_pairs and isinstance(raw_pairs[0], dict):
        pairs = [
            ParallelSentencePair(
                source_text=p["source_text"],
                target_text=p["target_text"],
                source_lang=p["source_lang"],
                target_lang=p["target_lang"],
                domain=p.get("split", "strict_12way"),
                id=f"{p.get('doc_id', 'doc')}_{p['source_lang']}_{p['target_lang']}"
            )
            for p in raw_pairs
        ]
    else:
        pairs = raw_pairs

    print(f"✓ Loaded {len(pairs):,} parallel sentence pairs across all 132 language directions.")

    # 3. DataLoaders and Tokenizer
    print("\n[2/4] Initializing Tokenizer and PyTorch DataLoaders...")
    custom_tokenizer = None
    spm_model_file = Path(args.spm_model_path)
    if args.tokenizer_type == "spm" or (args.tokenizer_type == "auto" and spm_model_file.exists()):
        if spm_model_file.exists():
            print(f"✓ Loading SentencePiece BPE Tokenizer from {spm_model_file}...")
            custom_tokenizer = PralekhaSentencePieceTokenizer(spm_model_file)
        else:
            logger.warning(f"SentencePiece model not found at {spm_model_file}, falling back to PralekhaTokenizer")

    train_loader, val_loader, test_loader, tokenizer = get_pralekha_dataloaders(
        pairs=pairs,
        tokenizer=custom_tokenizer,
        batch_size=args.batch_size,
        max_source_length=args.max_len,
        max_target_length=args.max_len,
        pin_memory=is_cuda
    )

    print(f"✓ Multilingual Vocabulary: {tokenizer.vocab_size} tokens across 22 Scheduled Indic scripts + English.")
    print(f"✓ Split: {len(train_loader)} train batches | {len(val_loader)} validation batches.")

    # 4. Model Architecture Instantiation
    print(f"\n[3/4] Building Pralekha Seq2Seq Transformer Model ({args.arch})...")
    config = PralekhaSeq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_languages=len(ALL_LANGUAGES)
    )

    model = PralekhaSeq2SeqTransformer(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total Trainable Parameters: {num_params:,} (~{num_params / 1e6:.2f}M params)")

    # Checkpoint Resume & Prior State Loading
    start_epoch = 1
    best_val_loss = float("inf")
    checkpoint_path = out_dir / "pralekha_checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt.get("model_state_dict") or ckpt.get("model")
            if state_dict:
                model.load_state_dict(state_dict)
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_loss = float(ckpt.get("val_loss", float("inf")))
                print(f"✓ Resumed from existing checkpoint (Prior Epoch: {start_epoch - 1}, Prior Best Val Loss: {best_val_loss:.4f})")
        except Exception as e:
            logger.warning(f"Could not load existing checkpoint to resume: {e}")

    # 5. Optimizer, Scaler, and Scheduler with Linear Warmup & Cosine Annealing
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = min(args.warmup_steps, max(total_steps // 10, 100))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 6. Training Loop with Dynamic Early Stopping
    print("\n[4/4] Starting Optimization Epochs on GPU...")
    print(f"  • Strategy: Continue training until validation loss stops decreasing")
    print(f"  • Early Stopping Patience: {args.patience} epochs | Min Delta: {args.min_delta}")
    history: List[Dict[str, Any]] = []
    patience = args.patience
    min_delta = args.min_delta
    no_improvement_epochs = 0
    end_epoch = start_epoch + args.epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        t_start = time.time()
        model.train()
        train_loss_accum = 0.0
        train_tokens = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels
                    )
                    loss = outputs["loss"]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels
                )
                loss = outputs["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠ Warning: Non-finite loss ({loss.item()}) at step {step}, skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            train_loss_accum += loss.item()
            train_tokens += (labels != tokenizer.pad_token_id).sum().item()

            if step % 200 == 0 or step == len(train_loader):
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch [{epoch}/{end_epoch}] | Step [{step}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} | LR: {cur_lr:.2e}"
                )

        avg_train_loss = train_loss_accum / max(len(train_loader), 1)

        # Validation Step
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                decoder_input_ids = batch["decoder_input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            labels=labels
                        )
                        v_loss = outputs["loss"]
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels
                    )
                    v_loss = outputs["loss"]

                val_loss_accum += v_loss.item()

        avg_val_loss = val_loss_accum / max(len(val_loader), 1)
        epoch_sec = time.time() - t_start

        # Check for validation decrease
        is_decrease = (avg_val_loss < (best_val_loss - min_delta))
        marker = "★ [IMPROVED]" if is_decrease else f"[PATIENCE {no_improvement_epochs + 1}/{patience}]"

        print(
            f"  >>> Epoch {epoch} Complete | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} {marker} | Time: {epoch_sec:.2f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "time_seconds": round(epoch_sec, 2)
        })

        # Run sample evaluation generations with Beam Search
        generate_sample_translations(
            model,
            tokenizer,
            device,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty
        )

        if is_decrease:
            improvement = best_val_loss - avg_val_loss
            print(f"  ★ Validation loss decreased by {improvement:.4f} ({best_val_loss:.4f} -> {avg_val_loss:.4f})! Saving optimal checkpoint...")
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0

            # Save optimal checkpoint immediately
            saved_paths = save_pralekha_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=out_dir,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=round(avg_train_loss, 4),
                val_loss=round(best_val_loss, 4),
                dataset_desc=f"{args.db_path} ({len(pairs):,} pairs across 132 directions)"
            )
            for k, p in saved_paths.items():
                print(f"  ✓ Saved {k.capitalize()}: {p}")

            try:
                from backend.app.services.pralekha_service import reset_pralekha_inference_cache
                reset_pralekha_inference_cache()
            except Exception:
                pass
        else:
            no_improvement_epochs += 1
            print(f"  ⚠ Validation loss did not decrease (Current: {avg_val_loss:.4f} vs Best: {best_val_loss:.4f}). "
                  f"Patience counter: [{no_improvement_epochs}/{patience}]")

            if args.early_stopping and no_improvement_epochs >= patience:
                print("\n" + "=" * 70)
                print("🛑 EARLY STOPPING TRIGGERED:")
                print(f"   Validation loss has stopped decreasing for {patience} consecutive epochs.")
                print(f"   Optimal Validation Loss achieved: {best_val_loss:.4f}")
                print("=" * 70)
                break

    # 7. Final Summary
    total_duration = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f"🎉 Training Successfully Completed in {total_duration:.2f}s ({total_duration / 60:.1f} mins)!")
    print(f"Optimal Validation Loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_training()
