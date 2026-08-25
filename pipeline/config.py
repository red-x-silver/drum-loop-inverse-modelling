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
# shallow chosen over mid/deep: best Acc1 (0.864) and fewest octave errors on the prior set.
TEMPO_CKPT = str(MODELS / "tempo" / "tempo_shallow_tcn_faithful.ckpt")
TEMPO_POSITION = "shallow"
TEMPO_MODULE = "tcn-faithful"
TEMPO_PROJ_DIM = 256          # tcn-faithful internally fixes eff_d=16; kept for the hparam signature
TEMPO_MID_GRU = 1
TEMPO_LOW = 60                # BPM = argmax(logits) + TEMPO_LOW

# --- Stable-Audio-3 one-shot extraction (small-rank4, step 4000) -------------
# The LoRA adapter is bundled; the base model + its own venv are provided by you (see README).
SA3_LORA = str(MODELS / "lora" / "lora_small_r4_step04000.safetensors")
SA3_RANK = 4
SA3_DIR = str(ROOT / "sa3")                              # bundled SA3 glue scripts
SA3_SCRIPT = "infer_loop.py"
# --- edit these to point at your SA3 install --------------------------------
# Python interpreter of the stable-audio-tools environment (has stable_audio_tools installed).
SA3_PYTHON = os.environ.get(
    "SA3_PYTHON", r"D:/stable-audio-3/stable-audio-tools/.venv/Scripts/python.exe")
# Base model config + weights (download into models/sa3-base/ — see README).
SA3_REPO_CFG = os.environ.get("SA3_REPO_CFG", str(MODELS / "sa3-base" / "model_config.json"))
SA3_BASE_CKPT = os.environ.get("SA3_BASE_CKPT", str(MODELS / "sa3-base" / "model.safetensors"))
# ----------------------------------------------------------------------------
SA3_PY = SA3_PYTHON                                      # alias used by pipeline.oneshots
SA3_SEED = 0                  # which generated seed to keep as the one-shot
SA3_STEPS = 50

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
