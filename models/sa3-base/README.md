# Stable-Audio-3 base model goes here

The one-shot extractor uses the **Stable-Audio-3 small (music)** base model. The base weights are
large (~2.2 GB) and are **not** bundled with this repo — download them and place the two files in
this folder:

```
models/sa3-base/
  model.safetensors     # ~2.2 GB base weights
  model_config.json     # base model config (small vs. medium)
```

Download from the Stability AI model page:

- https://huggingface.co/stabilityai/stable-audio-3-small-music

(Log in / accept the model licence on Hugging Face, then download `model.safetensors` and
`model_config.json`.)

If you keep the base model elsewhere, point the pipeline at it with environment variables instead of
copying it here:

```bash
export SA3_BASE_CKPT=/path/to/model.safetensors
export SA3_REPO_CFG=/path/to/model_config.json
```

The **LoRA adapter** (`models/lora/lora_small_r4_step04000.safetensors`, rank 4) is bundled — only the
base model needs downloading.
