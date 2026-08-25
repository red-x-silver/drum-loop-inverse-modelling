"""One-shot extraction via the Stable-Audio-3 small-rank4 (step 4000) LoRA model.

The SA3 model lives in a separate venv, so it is invoked as a subprocess. Results are cached by
the SHA1 of the analysed 4 s loop audio, so re-analysing the same loop is instant.
"""
import hashlib
import os
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


def extract_oneshots(loop_wav_path, wave_for_hash, seed=C.SA3_SEED, steps=C.SA3_STEPS):
    """Return one-shots [3,1,K] (kick,snare,hh) + the cache dir. Cache-hit -> no SA3 call."""
    key = loop_hash(wave_for_hash)
    cache = C.CACHE_ROOT / "oneshots" / key
    paths = {inst: cache / f"one_shot_{C.INSTRU_DIR[inst]}.wav" for inst in C.INSTRUMENTS}

    if not all(p.exists() for p in paths.values()):
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
