"""Central configuration for the drum-loop inverse-modelling deployment.

Unlike the research repo this was extracted from, this package is self-contained: the ADT + tempo
checkpoints and the one-shot LoRA adapter are bundled under ``models/`` and referenced by relative
paths, and the inference model definitions are vendored (no training framework required).

The ONLY external pieces you must provide yourself are the Stable-Audio-3 base model weights and the
SA3 Python environment used to run the one-shot extractor (see README). Point the ``SA3_*`` paths
below (or the ``SA3_PYTHON`` environment variable) at your install.
"""
import os
from pathlib import Path

# --- repo layout -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]               # .../drum-loops-inverse-modelling
MODELS = ROOT / "models"
CACHE_ROOT = ROOT / "cache"                               # one-shot / analysis cache

# --- audio constants ---------------------------------------------------------
SR = 44100
FPS = 100
LOOP_SECONDS = 4.0
LOOP_LEN = int(SR * LOOP_SECONDS)                        # 176400 samples analysed per loop
ONESHOT_LEN = SR                                         # K = 1 s one-shot length
INSTRUMENTS = ["kick", "snare", "hh"]                    # GT track order
INSTRU_DIR = {"kick": "kick", "snare": "snare", "hh": "hihats"}
ONSET_CHANNELS = {"kick": 0, "snare": 1, "hh": 2}        # 3-ch ADT output order

# --- ADT + tempo checkpoints (bundled, shared trunk) -------------------------
# ADT: the baseline_all ADTOF-style CRNN (3-ch kick/snare/hihats onset transcription).
ADT_CKPT = str(MODELS / "adt" / "adt_baseline_all.ckpt")
PEAK_PICKING_JSON = str(MODELS / "adt" / "peak_picking.json")
# tempo: madmom-style TCN head at the 'shallow' (post-CNN) tap on the FROZEN ADT trunk.
# shallow chosen over mid/deep: highest FSL-1147 Acc1 among TCN insert positions
TEMPO_CKPT = str(MODELS / "tempo" / "tempo_shallow_tcn_faithful.ckpt")
TEMPO_POSITION = "shallow"
TEMPO_MODULE = "tcn-faithful"
TEMPO_PROJ_DIM = 256          # tcn-faithful internally fixes eff_d=16; kept for the hparam signature
TEMPO_MID_GRU = 1
TEMPO_LOW = 60                # BPM = argmax(logits) + TEMPO_LOW

# --- Stable-Audio-3 one-shot extraction (Stable-one-shot) --------------------
# Two extractor configurations are released, matching the two options reported in the thesis:
#   small-r4    SA3-Small-Music backbone + rank-4  LoRA (2.6M adapter params) -- efficient (default)
#   medium-r16  SA3-Medium      backbone + rank-16 LoRA (20.7M adapter params) -- higher fidelity
# Both adapters were trained for 4000 steps on the clean (mode03-excluded) DITS configuration.
# Select with the SA3_VARIANT environment variable; both LoRA adapters are bundled, the two base
# models are downloaded separately into models/sa3-base/ and models/sa3-base-medium/ (see README).
SA3_VARIANTS = {
    "small-r4":   {"lora": "lora_small_r4_step04000.safetensors",   "rank": 4,
                   "base_dir": "sa3-base"},
    "medium-r16": {"lora": "lora_medium_r16_step04000.safetensors", "rank": 16,
                   "base_dir": "sa3-base-medium"},
}
SA3_VARIANT = os.environ.get("SA3_VARIANT", "small-r4")
if SA3_VARIANT not in SA3_VARIANTS:
    raise ValueError(f"unknown SA3_VARIANT {SA3_VARIANT!r} (expected one of {list(SA3_VARIANTS)})")
_SA3 = SA3_VARIANTS[SA3_VARIANT]

SA3_LORA = str(MODELS / "lora" / _SA3["lora"])
SA3_RANK = _SA3["rank"]
SA3_BASE_DIR = MODELS / _SA3["base_dir"]
SA3_DIR = str(ROOT / "sa3")                              # bundled SA3 glue scripts
SA3_SCRIPT = "infer_loop.py"
# --- point these at your SA3 install ----------------------------------------
# Python interpreter of the stable-audio-tools environment (the one with stable_audio_tools
# installed). There is no portable default -- set the SA3_PYTHON environment variable, e.g.
#   export SA3_PYTHON=/path/to/stable-audio-tools/.venv/bin/python          (Linux/macOS)
#   set     SA3_PYTHON=C:\path\to\stable-audio-tools\.venv\Scripts\python.exe   (Windows)
# or edit the fallback below. Only the one-shot / full / params modes need it.
SA3_PYTHON = os.environ.get("SA3_PYTHON", "")
# Base model config + weights (download into the variant's base dir — see README).
SA3_REPO_CFG = os.environ.get("SA3_REPO_CFG", str(SA3_BASE_DIR / "model_config.json"))
SA3_BASE_CKPT = os.environ.get("SA3_BASE_CKPT", str(SA3_BASE_DIR / "model.safetensors"))
# ----------------------------------------------------------------------------
SA3_PY = SA3_PYTHON                                      # alias used by pipeline.oneshots
SA3_SEED = 0                  # which generated seed to keep as the one-shot
SA3_STEPS = 50                # rectified-flow Euler sampling steps

# --- velocity estimation (locked optimal configuration from the pilot) -------
VEL_LOSS = "both"             # L1 + multi-resolution STFT
VEL_LR = 0.1
VEL_ITERS = 500
VEL_TRANSFORM = "sigmoid"
VEL_INIT_LOGIT = 10.0
VEL_SMOOTHING = True
VEL_MODE = "poly"             # polyphonic default for real-world loops (no voicing annotation)


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
