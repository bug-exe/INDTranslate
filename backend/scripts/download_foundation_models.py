#!/usr/bin/env python3
"""
Downloader for Foundation NMT Models:
1. facebook/nllb-200-distilled-600M
2. ai4bharat/indictrans2-indic-indic-dist-320M
3. ai4bharat/indictrans2-en-indic-dist-200M
4. ai4bharat/indictrans2-indic-en-dist-200M
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("download_foundation_models")

FOUNDATION_DIR = Path("backend/saved_models/foundation")
FOUNDATION_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE = FOUNDATION_DIR / "download_status.json"

MODELS_TO_DOWNLOAD = [
    {
        "repo_id": "facebook/nllb-200-distilled-600M",
        "target_dir": FOUNDATION_DIR / "nllb-200-distilled-600M",
        "name": "NLLB-200 (600M Distilled)",
        "description": "Meta NLLB-200 distilled 600M multilingual model supporting 200+ languages including all 22 Indian languages."
    },
    {
        "repo_id": "ai4bharat/indictrans2-indic-indic-dist-320M",
        "target_dir": FOUNDATION_DIR / "indictrans2-indic-indic-dist-320M",
        "name": "IndicTrans2 (320M Indic-to-Indic Distilled)",
        "description": "AI4Bharat IndicTrans2 direct Indic-to-Indic 320M distilled model covering 22 scheduled languages."
    },
    {
        "repo_id": "ai4bharat/indictrans2-en-indic-dist-200M",
        "target_dir": FOUNDATION_DIR / "indictrans2-en-indic-dist-200M",
        "name": "IndicTrans2 (200M English-to-Indic Distilled)",
        "description": "AI4Bharat IndicTrans2 English-to-Indic 200M distilled model."
    },
    {
        "repo_id": "ai4bharat/indictrans2-indic-en-dist-200M",
        "target_dir": FOUNDATION_DIR / "indictrans2-indic-en-dist-200M",
        "name": "IndicTrans2 (200M Indic-to-English Distilled)",
        "description": "AI4Bharat IndicTrans2 Indic-to-English 200M distilled model."
    }
]


def update_status(status_dict: dict):
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write status file: {e}")


def main():
    logger.info("Starting foundation models download pipeline...")
    status = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {}
    }
    for m in MODELS_TO_DOWNLOAD:
        status["models"][m["repo_id"]] = {
            "name": m["name"],
            "status": "pending",
            "path": str(m["target_dir"])
        }
    update_status(status)

    for m in MODELS_TO_DOWNLOAD:
        repo_id = m["repo_id"]
        target_dir = m["target_dir"]
        logger.info(f"Downloading {m['name']} from {repo_id}...")
        status["models"][repo_id]["status"] = "downloading"
        status["models"][repo_id]["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        update_status(status)

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            local_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            logger.info(f"Successfully downloaded {m['name']} to {local_path}")
            status["models"][repo_id]["status"] = "completed"
            status["models"][repo_id]["completion_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            update_status(status)
        except Exception as e:
            logger.error(f"Failed to download {repo_id}: {e}", exc_info=True)
            status["models"][repo_id]["status"] = "failed"
            status["models"][repo_id]["error"] = str(e)
            update_status(status)

    status["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    status["overall_status"] = "all_downloads_finished"
    update_status(status)
    logger.info("Foundation models download pipeline complete.")


if __name__ == "__main__":
    main()
