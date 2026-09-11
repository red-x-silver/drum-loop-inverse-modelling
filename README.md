# Drum loop inverse modelling

Given a single **drum-loop audio file**, estimate the drum-machine parameters that reconstruct it:
tempo, per-instrument onsets, one-shot samples, per-onset velocities, and a quantised 16-step
pattern. This is the **deployment** build of the thesis system

The pipeline is:

```
loop.wav ─▶ [pad/truncate to 4 s] ─▶ ADT + tempo (shared trunk) ─▶ peak-pick onsets
         ─▶ one-shot extraction (Stable-Audio-3 + LoRA) ─▶ analysis-by-synthesis velocities
         ─▶ quantisation ─▶ {tempo, onsets, one-shots, velocities, step vectors}
```

## Models

- **DITS-ADT: ADT model with tempo estimation (shared trunk).** The **ADT** network is an ADTOF-style CRNN, the
  `baseline_all` (from the thesis experiment) checkpoint, producing 3-channel (kick / snare / hi-hats) onset activations. The
  **tempo** head is a madmom-style TCN (`tcn-faithful`, 7 dilated blocks, receptive field ~5 s)
  tapped at the **`shallow`** position (post-CNN) on the *frozen* ADT trunk, giving a 141-way BPM
  classifier (60–200 BPM). One forward pass yields per-instrument onset positions **and** a global tempo. Both checkpoints are
  bundled under `models/` (~7 MB).
- **Stable-one-shot: drum one-shot extractor.** A LoRA-adapted Stable-Audio-3 model that inpaints an
  isolated kick / snare / hi-hat one-shot from the loop's latent. **Two configurations are released**,
  selected with the `SA3_VARIANT` environment variable:

  | `SA3_VARIANT` | Backbone | LoRA | Adapter params | Adapter file | Base model |
  |---|---|---|:--:|---|:--:|
  | `small-r4` *(default)* | SA3-Small-Music | rank 4 | 2.6 M (~5 MB) | `lora_small_r4_step04000.safetensors` | ~2.2 GB |
  | `medium-r16` | SA3-Medium | rank 16 | 20.7 M (~40 MB) | `lora_medium_r16_step04000.safetensors` | ~8.6 GB |

  `small-r4` is the efficient option; `medium-r16` gives higher extraction fidelity. Both adapters
  were trained for 4000 steps on the clean (audio-effects-excluded) training configuration, and
  **both are bundled**; the base models are downloaded separately (see below). The one-shot cache is
  namespaced per variant, so switching variants re-extracts rather than reusing the other model's
  output, and every result dict records which extractor produced it under `oneshot_model`.
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
  lora/                # bundled LoRA adapters (small-r4 and medium-r16)
  sa3-base/            # <- download the SA3 small base model here  (see its README)
  sa3-base-medium/     # <- download the SA3 medium base model here (only for SA3_VARIANT=medium-r16)
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
2. Download the base model for the variant you want:
   - `small-r4` (default) → **Stable-Audio-3 small** into `models/sa3-base/` — see
     [`models/sa3-base/README.md`](models/sa3-base/README.md).
   - `medium-r16` → **Stable-Audio-3 medium** into `models/sa3-base-medium/` — see
     [`models/sa3-base-medium/README.md`](models/sa3-base-medium/README.md).

   You only need the one you intend to run.
3. Point the pipeline at that venv's Python, either by editing `SA3_PYTHON` in
   [`pipeline/config.py`](pipeline/config.py) or via an environment variable:

   ```bash
   export SA3_PYTHON=/path/to/stable-audio-tools/.venv/bin/python

   # pick the extractor (default: small-r4); this selects the LoRA adapter AND its base model
   export SA3_VARIANT=medium-r16

   # optional, if the base model lives outside the variant's models/ folder:
   export SA3_BASE_CKPT=/path/to/model.safetensors
   export SA3_REPO_CFG=/path/to/model_config.json
   ```

Mode 1 (`transcribe`) does **not** need the SA3 environment — only the one-shot / full / params modes do.

<details>
<summary><b>Re-training the one-shot LoRA</b> (optional — both trained adapters are bundled)</summary>

[`sa3/train.py`](sa3/train.py) reproduces the LoRA fine-tuning. It reads a **pre-encoded SAME latent
cache** built from the synthesised training dataset, which is not distributed with this repo, so its
paths have no portable defaults — set them for your machine:

| Variable | Meaning |
|---|---|
| `SA3_REPO_CFG`, `SA3_BASE_CKPT` | base model to adapt (small or medium) |
| `SA3_TRAIN_ROOT`, `SA3_VAL_ROOT` | pre-encoded SAME latent cache (train / validation) |
| `SA3_GT_LOOKUP` | optional `.npz` of ground-truth one-shot latents for held-out validation |
| `SA3_RUN_DIR` | where checkpoints and demos are written (default `runs/lora`) |

```bash
python sa3/train.py --full --steps 4000 --transient-weight 0.1 --drop-modes mode03 \
  --run-dir runs/lora
```

`--drop-modes mode03` excludes the audio-effects renderings, matching the released adapters: with
effects applied to the mixture but not to the ground-truth one-shot, the pair would be noisy
supervision that forces the model to learn effect-inversion on top of extraction.

</details>

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
  "oneshot_model": "small-r4",        // which extractor variant produced the one-shots
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

### Notes on the quantised output

Two behaviours of [`pipeline/quantize.py`](pipeline/quantize.py) are **implementation-only guards**
for degenerate inputs — they are not part of the estimation method and never fire on a normal
`params` run, where every active step is backed by an onset that carries an estimated velocity:

- If an active step somehow has no folded onset velocity, its `step_velocities` entry falls back to
  that track's **mean** estimated velocity.
- If velocities are unavailable altogether (e.g. `quantize_params` called with `velocities=None`,
  as when quantising onsets that were never passed through the velocity stage), every active step is
  emitted at a flat **0.8** so the step vectors still render.

Otherwise each active step carries the velocity of the onset that quantised onto it, and where
several onsets fold onto one step the **maximum** is retained, since a drum-machine step holds a
single velocity and the loudest hit is the perceptually dominant one.
