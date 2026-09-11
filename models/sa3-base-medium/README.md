# Stable-Audio-3 **medium** base model goes here

This folder holds the base model for the **`medium-r16`** one-shot extractor variant — the
higher-fidelity of the two released configurations. It is only needed if you run with
`SA3_VARIANT=medium-r16`; the default `small-r4` variant uses [`../sa3-base/`](../sa3-base/README.md)
instead.

The base weights are large (~8.6 GB) and are **not** bundled with this repo — download them and
place the two files in this folder:

```
models/sa3-base-medium/
  model.safetensors     # ~8.6 GB base weights
  model_config.json     # base model config (medium)
```

Download from the Stability AI model page:

- https://huggingface.co/stabilityai/stable-audio-3-medium

(Log in / accept the model licence on Hugging Face, then download `model.safetensors` and
`model_config.json`.)

If you keep the base model elsewhere, point the pipeline at it with environment variables instead of
copying it here:

```bash
export SA3_VARIANT=medium-r16
export SA3_BASE_CKPT=/path/to/medium/model.safetensors
export SA3_REPO_CFG=/path/to/medium/model_config.json
```

The matching **LoRA adapter** (`models/lora/lora_medium_r16_step04000.safetensors`, rank 16,
20.7 M adapter parameters, ~40 MB) **is** bundled — only the base model needs downloading.

> **Pair the adapter with the right base.** A rank-16 medium adapter will not load onto the small
> backbone (and vice versa). `SA3_VARIANT` selects both together, so prefer setting that single
> variable over overriding `SA3_LORA` / `SA3_BASE_CKPT` individually.
