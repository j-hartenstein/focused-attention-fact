"""Modal app definition for FACT experiments.

Defines the GPU stub, container image, and shared volume for model caching.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Image: install all deps
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.45",
        "peft>=0.13",
        "datasets>=2.20",
        "accelerate>=0.33",
        "openai>=1.40",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pyyaml",
        "rich",
        "huggingface_hub",
        "lm_eval>=0.4",
        # lm_eval[ifeval] extra deps (langdetect needed for language detection instructions)
        "langdetect",
        "immutabledict",
        "nltk>=3.9.1",
    )
    .run_commands("python -m nltk.downloader punkt_tab")  # IFEval uses punkt tokenizer
    .pip_install("hf-transfer")  # faster HF downloads
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---------------------------------------------------------------------------
# App + volume
# ---------------------------------------------------------------------------
app = modal.App("fact-alignment", image=image)

# Persistent volume for HuggingFace model cache (avoids re-downloading)
model_volume = modal.Volume.from_name("fact-model-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("fact-data", create_if_missing=True)

MODEL_CACHE_DIR = "/root/.cache/huggingface"
DATA_DIR = "/data"
RESULTS_DIR = "/results"
