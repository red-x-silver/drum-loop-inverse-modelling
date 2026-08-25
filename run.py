"""Drum-loop inverse-modelling — single driver script (CLI + importable API).

Given a single drum-loop audio file, estimate the drum-machine parameters that reproduce it. Four
modes, matching the four levels of the inverse model:

  1. transcribe : ADT + tempo estimation          -> tempo (BPM) + per-instrument onset times
  2. oneshots   : one-shot extraction (SA3 + LoRA) -> kick / snare / hi-hat one-shot wavs
  3. full       : 1 + 2 + per-onset velocities     -> absolute-time parameters + a reconstruction
  4. params     : 3 + quantisation                 -> step vectors, beat type, swing, step velocities

CLI:
    python run.py transcribe loop.wav
    python run.py oneshots   loop.wav --out out/
    python run.py full       loop.wav --out out/
    python run.py params     loop.wav --out out/ --json params.json

Programmatic:
    from run import transcribe, oneshots, full, params
    result = params("loop.wav", out_dir="out/")

The heavy models (ADT + tempo trunk, differentiable renderer) are loaded lazily and reused across
calls within a process via a module-level singleton.
"""
import argparse
import json
import os
import sys

# make the repo root importable (adtof_pytorch, differentiable_renderer, velocity_estimation, pipeline)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import config as C


# --------------------------------------------------------------------------- shared model singleton
_PIPELINE = None


def get_pipeline(device=None):
    """Lazily build and cache the warm inference pipeline (ADT+tempo model, pickers, renderer)."""
    global _PIPELINE
    if _PIPELINE is None:
        from pipeline.infer import Pipeline
        _PIPELINE = Pipeline(device=device)
    return _PIPELINE


def _save_audio(path, wav, sr=C.SR):
    import torchaudio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, wav, sr)


def _emit(result, json_path=None):
    txt = json.dumps(result, indent=2)
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as f:
            f.write(txt)
    return result


# --------------------------------------------------------------------------------------- mode 1
def transcribe(wav_path, device=None, json_path=None):
    """ADT + tempo estimation only. Returns {tempo_bpm, instruments:{onsets_s, onsets_samples}}."""
    pipe = get_pipeline(device)
    result, _target, _mix = pipe.analyze(wav_path, do_oneshots=False, do_velocity=False)
    out = {
        "mode": "transcribe",
        "sample_rate": result["sample_rate"],
        "duration_analyzed_s": result["duration_analyzed_s"],
        "tempo_bpm": result["tempo_bpm"],
        "instruments": {inst: {"onsets_s": result["instruments"][inst]["onsets_s"],
                               "onsets_samples": result["instruments"][inst]["onsets_samples"]}
                        for inst in C.INSTRUMENTS},
    }
    return _emit(out, json_path)


# --------------------------------------------------------------------------------------- mode 2
def oneshots(wav_path, out_dir=None, device=None, json_path=None):
    """One-shot extraction only (Stable-Audio-3 small-r4 LoRA). Writes 3 one-shot wavs.
    Does not run the ADT/tempo model."""
    import torchaudio
    from pipeline.infer import load_loop_4s
    from pipeline.oneshots import extract_oneshots, loop_hash

    target = load_loop_4s(wav_path)                              # [1, T]
    key = loop_hash(target)
    loop_wav = C.CACHE_ROOT / "oneshots" / key / "loop_4s.wav"
    loop_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(loop_wav), target, C.SR)
    _one, oneshot_dir = extract_oneshots(loop_wav, target)

    out = {"mode": "oneshots", "cache_key": key, "sample_rate": C.SR,
           "oneshot_dir": str(oneshot_dir), "oneshots": {}}
    for inst in C.INSTRUMENTS:
        src = os.path.join(str(oneshot_dir), f"one_shot_{C.INSTRU_DIR[inst]}.wav")
        dst = src
        if out_dir:
            dst = os.path.join(out_dir, f"{inst}.wav")
            os.makedirs(out_dir, exist_ok=True)
            import shutil
            shutil.copyfile(src, dst)
        out["oneshots"][inst] = dst
    return _emit(out, json_path)


# ------------------------------------------------------------------------------- modes 3 & 4
def _analyze_full(wav_path, out_dir, device, with_quantize):
    pipe = get_pipeline(device)
    result, target, mix = pipe.analyze(wav_path, do_oneshots=True, do_velocity=True)

    out = {
        "sample_rate": result["sample_rate"],
        "duration_analyzed_s": result["duration_analyzed_s"],
        "cache_key": result["cache_key"],
        "tempo_bpm": result["tempo_bpm"],
        "instruments": {inst: {
            "onsets_s": result["instruments"][inst]["onsets_s"],
            "onsets_samples": result["instruments"][inst]["onsets_samples"],
            "velocities": result["instruments"][inst]["velocities"],
        } for inst in C.INSTRUMENTS},
        "reconstruction_loss": result.get("reconstruction_loss"),
    }
    if with_quantize:
        out["quantized"] = result["quantized"]

    # write audio artifacts
    if out_dir:
        _save_audio(os.path.join(out_dir, "loop.wav"), target)
        if mix is not None:
            _save_audio(os.path.join(out_dir, "recon.wav"), mix)
        import shutil
        for inst in C.INSTRUMENTS:
            src = os.path.join(result["oneshot_dir"], f"one_shot_{C.INSTRU_DIR[inst]}.wav")
            if os.path.exists(src):
                os.makedirs(out_dir, exist_ok=True)
                shutil.copyfile(src, os.path.join(out_dir, f"{inst}.wav"))
        out["out_dir"] = out_dir
    return out


def full(wav_path, out_dir=None, device=None, json_path=None):
    """ADT + tempo + one-shots + per-onset velocities. Absolute-time parameters + reconstruction."""
    out = _analyze_full(wav_path, out_dir, device, with_quantize=False)
    out["mode"] = "full"
    return _emit(out, json_path)


def params(wav_path, out_dir=None, device=None, json_path=None):
    """Everything in `full` plus quantisation into drum-machine parameters (step vectors, beat type,
    swing, step velocities)."""
    out = _analyze_full(wav_path, out_dir, device, with_quantize=True)
    out["mode"] = "params"
    return _emit(out, json_path)


# ------------------------------------------------------------------------------------------- CLI
_MODES = {"transcribe": transcribe, "oneshots": oneshots, "full": full, "params": params}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=list(_MODES), help="which level of the inverse model to run")
    ap.add_argument("loop", help="path to the input drum-loop audio file")
    ap.add_argument("--out", dest="out_dir", default=None,
                    help="directory to write audio artifacts (loop/recon/one-shots) into")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="also write the result dict to this JSON file")
    ap.add_argument("--device", default=None, help="force 'cuda' or 'cpu' (default: auto)")
    args = ap.parse_args(argv)

    fn = _MODES[args.mode]
    if args.mode == "transcribe":
        result = fn(args.loop, device=args.device, json_path=args.json_path)
    else:
        result = fn(args.loop, out_dir=args.out_dir, device=args.device, json_path=args.json_path)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
