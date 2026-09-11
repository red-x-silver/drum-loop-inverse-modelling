"""End-to-end inverse-modelling pipeline: a drum loop wav -> absolute-time parameter estimates.

    wav -> [pad/truncate to 4 s] -> ADT+tempo (shared trunk) -> peak-pick onsets
        -> SA3 one-shot extraction -> analysis-by-synthesis velocity -> {tempo, onsets, velocities}
"""
import torch
import torchaudio

from . import config as C
from .adt_tempo_model import ADTTempoModel
from .onsets import build_pickers, pick_onsets
from .oneshots import cache_dir, extract_oneshots, loop_hash
from .velocity import make_renderer, estimate_velocities
from .quantize import quantize_params


def load_loop_4s(path, sr=C.SR, n=C.LOOP_LEN):
    """Load, mono-mix, resample to 44.1 kHz, and pad/truncate to exactly 4 s -> [1, n]."""
    wav, wsr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if wsr != sr:
        wav = torchaudio.functional.resample(wav, wsr, sr)
    L = wav.shape[-1]
    wav = wav[:, :n] if L >= n else torch.nn.functional.pad(wav, (0, n - L))
    return wav                                                        # (1, n)


class Pipeline:
    """Holds the warm singletons (ADT+tempo model, pickers, renderer)."""

    def __init__(self, device=None):
        self.device = device or C.device()
        self.model = ADTTempoModel(device=self.device)
        self.pickers = build_pickers()
        self.renderer = make_renderer()

    def analyze(self, wav_path, do_velocity=True, do_oneshots=True):
        target = load_loop_4s(wav_path)                              # [1, T]
        key = loop_hash(target)
        C.CACHE_ROOT.mkdir(parents=True, exist_ok=True)

        adt = self.model.analyze(target.squeeze(0))                  # onset_env + tempo
        onsets = pick_onsets(adt["onset_env"], self.pickers)         # per-instrument secs+samples

        result = {
            "cache_key": key,
            "sample_rate": C.SR,
            "duration_analyzed_s": C.LOOP_SECONDS,
            "tempo_bpm": adt["tempo_bpm"],
            "instruments": {inst: {"onsets_s": onsets[inst]["seconds"],
                                   "onsets_samples": onsets[inst]["samples"],
                                   "velocities": None} for inst in C.INSTRUMENTS},
            "oneshot_dir": None,
            "reconstruction": None,
        }
        mix = None
        if do_oneshots:
            # persist the analysed 4 s loop for SA3 (and as the reconstruction target reference)
            loop_wav = cache_dir(key) / "loop_4s.wav"
            loop_wav.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(loop_wav), target, C.SR)
            one_shots, oneshot_dir = extract_oneshots(loop_wav, target)
            result["oneshot_dir"] = str(oneshot_dir)
            result["oneshot_model"] = C.SA3_VARIANT               # 'small-r4' | 'medium-r16'
            if do_velocity:
                onsets_samples = [onsets[inst]["samples"] for inst in C.INSTRUMENTS]
                vels, mix, info = estimate_velocities(self.renderer, one_shots, onsets_samples,
                                                      target, device=self.device)
                for i, inst in enumerate(C.INSTRUMENTS):
                    result["instruments"][inst]["velocities"] = [float(v) for v in vels[i]]
                result["reconstruction_loss"] = float(info["final_loss"])

        # Phase 2: quantise absolute-time onsets/velocities into step-based parameters
        onsets_s = [onsets[inst]["seconds"] for inst in C.INSTRUMENTS]
        vel = [result["instruments"][inst]["velocities"] for inst in C.INSTRUMENTS]
        vel = vel if all(v is not None for v in vel) else None
        result["quantized"] = quantize_params(onsets_s, vel, adt["tempo_bpm"])
        return result, target, mix
