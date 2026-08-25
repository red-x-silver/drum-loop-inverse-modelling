# Drum loop inverse modelling

Given a single **drum-loop audio file**, estimate the drum-machine parameters that reproduce it:
tempo, per-instrument onsets, one-shot samples, per-onset velocities, and a quantised 16-step
pattern. This is the **deployment** build of the thesis system — inference only, no training code.

The pipeline is:

```
loop.wav ─▶ [pad/truncate to 4 s] ─▶ ADT + tempo (shared trunk) ─▶ peak-pick onsets
         ─▶ one-shot extraction (Stable-Audio-3 + LoRA) ─▶ analysis-by-synthesis velocities
         ─▶ quantisation ─▶ {tempo, onsets, one-shots, velocities, step vectors}
```

## Models

- **ADT + tempo (shared trunk).** The **ADT** network is an ADTOF-style CRNN (`ADTOFFrameRNN`), the
  `baseline_all` checkpoint, producing 3-channel (kick / snare / hi-hats) onset activations. The
  **tempo** head is a madmom-style TCN (`tcn-faithful`, 7 dilated blocks, receptive field ~5 s)
  tapped at the **`shallow`** position (post-CNN) on the *frozen* ADT trunk, giving a 141-way BPM
  classifier (60–200 BPM). One forward pass yields onsets **and** tempo. Both checkpoints are
  bundled under `models/` (~7 MB).
- **One-shot extractor.** Stable-Audio-3 **small** base model + a **rank-4 LoRA** adapter
  (`lora_small_r4_step04000.safetensors`) that inpaints an isolated kick / snare / hi-hat one-shot
  from the loop's latent. The LoRA is bundled (~5 MB); the base model is downloaded separately
  (see below).
- **Velocity.** Per-onset velocities are recovered by analysis-by-synthesis: a differentiable drum
  renderer is optimised (500 iters, L1 + multi-resolution STFT) to match the input loop.

## Layout

```
run.py                 # the driver — 4 modes, CLI + importable API
pipeline/              # inference wrappers
  config.py            #   paths + constants (edit the SA3_* paths for your machine)
  adt_tempo_model.py   #   shared-trunk ADT + tempo (loads the bundled checkpoints)
  tempo_head.py        #   tempo head nn.Modules (extracted for inference)
  onsets.py            #   peak-picking of ADT activations
  oneshots.py          #   Stable-Audio-3 subprocess wrapper (cached)
  velocity.py          #   analysis-by-synthesis velocity estimation
  quantize.py          #   absolute-time -> 16-step drum-machine parameters
  infer.py             #   Pipeline (ties it together)
adtof_pytorch/         # vendored ADT model + audio front-end + peak picker
differentiable_renderer.py, velocity_estimation.py   # vendored (torch-only)
sa3/                   # Stable-Audio-3 glue scripts (run in the SA3 environment)
models/
  adt/                 # bundled ADT checkpoint + peak_picking.json
  tempo/               # bundled tempo checkpoint
  lora/                # bundled rank-4 LoRA adapter
  sa3-base/            # <- download the SA3 base model here (see models/sa3-base/README.md)
webapp/                # demo local web app (Flask)
cache/                 # one-shot / analysis cache (auto-populated)
```

## Setup

The system uses **two Python environments**:

### 1. Main inference environment

Runs the ADT + tempo model, velocity estimation, quantisation, and the web app.

```bash
pip install -r requirements.txt
```

Install a `torch` / `torchaudio` build matching your CUDA/CPU setup. A GPU is used automatically when
available.

### 2. One-shot extraction setup (Stable-Audio-3)

The one-shot extractor runs in a **separate** environment because it depends on the
`stable-audio-tools` library. The glue scripts in `sa3/` are bundled; you provide the environment and
the base model:

1. Create a venv and install [`stable-audio-tools`](https://github.com/Stability-AI/stable-audio-tools)
   into it.
2. Download the **Stable-Audio-3 small** base model into `models/sa3-base/` — see
   [`models/sa3-base/README.md`](models/sa3-base/README.md).
3. Point the pipeline at that venv's Python, either by editing `SA3_PYTHON` in
   [`pipeline/config.py`](pipeline/config.py) or via an environment variable:

   ```bash
   export SA3_PYTHON=/path/to/stable-audio-tools/.venv/bin/python
   # optional, if the base model lives outside models/sa3-base/:
   export SA3_BASE_CKPT=/path/to/model.safetensors
   export SA3_REPO_CFG=/path/to/model_config.json
   ```

Mode 1 (`transcribe`) does **not** need the SA3 environment — only the one-shot / full / params modes do.

## Usage

One driver, four modes:

| Mode         | What it does                                              | Needs SA3? |
|--------------|----------------------------------------------------------|:----------:|
| `transcribe` | ADT + tempo → tempo (BPM) + per-instrument onset times    | no         |
| `oneshots`   | one-shot extraction → kick / snare / hi-hat one-shot wavs  | yes        |
| `full`       | transcribe + oneshots + per-onset velocities (+ recon)    | yes        |
| `params`     | full + quantisation → 16-step drum-machine parameters     | yes        |

### Command line

```bash
python run.py transcribe loop.wav
python run.py oneshots   loop.wav --out out/
python run.py full       loop.wav --out out/
python run.py params     loop.wav --out out/ --json out/params.json
```

`--out DIR` writes audio artifacts (`loop.wav`, `recon.wav`, `kick/snare/hh.wav`); `--json FILE`
also writes the result dict; the result is printed to stdout as JSON either way. Add `--device cpu`
to force CPU.

### As a library

```python
from run import transcribe, oneshots, full, params

result = params("loop.wav", out_dir="out/")
print(result["tempo_bpm"], result["quantized"]["step_vectors"])
```

The heavy models load lazily and are reused across calls within a process.

## Web app

A local demo: upload a loop, get back the tempo, one-shots, reconstruction, and parameters with
audio playback.

```bash
python webapp/app.py          # serves http://127.0.0.1:5001
```

(The web app runs `full`/`params`, so it needs the SA3 setup above.)

## Output

`params` (the most complete mode) returns:

```jsonc
{
  "tempo_bpm": 130,
  "instruments": {
    "kick":  { "onsets_s": [...], "onsets_samples": [...], "velocities": [...] },
    "snare": { ... }, "hh": { ... }
  },
  "reconstruction_loss": 1.94,
  "quantized": {
    "num_steps": 16, "steps_per_beat": 4,
    "beat_type": "16th",
    "swing": [0.5, 0.5, 0.5],
    "step_vectors":    [[1,0,0,0, 1,0,0,0, ...], ...],   // kick, snare, hh
    "step_velocities": [[...], ...],
    "instruments": ["kick", "snare", "hh"]
  }
}
```
