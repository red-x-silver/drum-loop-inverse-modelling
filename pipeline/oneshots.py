"""One-shot extraction via the Stable-one-shot LoRA-adapted Stable-Audio-3 model.

The configured variant (``C.SA3_VARIANT``: ``small-r4`` or ``medium-r16``) selects the LoRA adapter
and its matching base model. The SA3 model lives in a separate venv, so it is invoked as a
subprocess. Results are cached by the SHA1 of the analysed 4 s loop audio *and* the variant, so
re-analysing the same loop is instant while the two variants never share cache entries.
"""
import hashlib
import shutil
import subprocess

import torch
import torch.nn.functional as F
import torchaudio

from . import config as C


def _fit(wav, wsr, k=C.ONESHOT_LEN, sr=C.SR):
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if wsr != sr:
        wav = torchaudio.functional.resample(wav, wsr, sr)
    L = wav.shape[-1]
    return wav[:, :k] if L >= k else F.pad(wav, (0, k - L))         # (1, K)


def loop_hash(wave):
    """SHA1 of the (mono, 4 s) loop tensor for cache keying."""
    x = wave.detach().cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha1(x).hexdigest()[:16]


def cache_dir(key, variant=None):
    """Cache directory for one loop under one extractor variant.

    Namespacing by variant keeps ``small-r4`` and ``medium-r16`` results apart, so switching
    variants re-extracts instead of reusing the other model's one-shots."""
    return C.CACHE_ROOT / "oneshots" / (variant or C.SA3_VARIANT) / key


def _check_sa3_install():
    """Fail with an actionable message rather than an opaque subprocess error."""
    import os
    if not C.SA3_PY:
        raise RuntimeError(
            "SA3_PYTHON is not set. The one-shot extractor runs in a separate stable-audio-tools "
            "environment; point SA3_PYTHON at that environment's python executable (or edit "
            "SA3_PYTHON in pipeline/config.py). See the README, 'One-shot extraction setup'. "
            "The 'transcribe' mode does not need this.")
    if not os.path.exists(C.SA3_PY):
        raise RuntimeError(f"SA3_PYTHON does not exist: {C.SA3_PY}")
    missing = [p for p in (C.SA3_REPO_CFG, C.SA3_BASE_CKPT) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Stable-Audio-3 base model for variant '{C.SA3_VARIANT}' not found: "
            f"{', '.join(missing)}. Download it into {C.SA3_BASE_DIR} (see that folder's README), "
            f"or set SA3_REPO_CFG / SA3_BASE_CKPT.")
    if not os.path.exists(C.SA3_LORA):
        raise RuntimeError(f"LoRA adapter missing: {C.SA3_LORA}")


def extract_oneshots(loop_wav_path, wave_for_hash, seed=C.SA3_SEED, steps=C.SA3_STEPS):
    """Return one-shots [3,1,K] (kick,snare,hh) + the cache dir. Cache-hit -> no SA3 call."""
    cache = cache_dir(loop_hash(wave_for_hash))
    paths = {inst: cache / f"one_shot_{C.INSTRU_DIR[inst]}.wav" for inst in C.INSTRUMENTS}

    if not all(p.exists() for p in paths.values()):
        _check_sa3_install()
        tmp = cache / "_sa3_raw"
        tmp.mkdir(parents=True, exist_ok=True)
        cmd = [C.SA3_PY, C.SA3_SCRIPT, "--loop", str(loop_wav_path), "--out", str(tmp),
               "--ckpt", C.SA3_LORA, "--rank", str(C.SA3_RANK),
               "--repo-cfg", C.SA3_REPO_CFG, "--base-ckpt", C.SA3_BASE_CKPT,
               "--seeds", str(seed + 1), "--steps", str(steps), "--start-sec", "0.0"]
        subprocess.run(cmd, cwd=C.SA3_DIR, check=True)
        for inst in C.INSTRUMENTS:
            src = tmp / f"{C.INSTRU_DIR[inst]}_seed{seed}_gen.wav"
            paths[inst].parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, paths[inst])

    one = torch.stack([_fit(*torchaudio.load(str(paths[inst]))) for inst in C.INSTRUMENTS], dim=0)
    return one, cache                                                # (3,1,K), Path
