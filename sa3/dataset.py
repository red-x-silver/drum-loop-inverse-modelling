"""Option B data path — build the inpaint "continuation canvas" from cached SAME latents.

Per example: pick a (loop,mode) npz + an instrument present in it, then assemble a fixed
256-frame canvas that plugs straight into SA3's existing local-additive inpaint seam:

  frames                role            padding_mask  inpaint_mask  masked_input
  0 .. 43   (44)        loop context    1 (real)      1 (keep)      = loop latent
  44 .. 54  (11)        one-shot TARGET 1 (real)      0 (generate)  = 0   <-- loss here
  55 .. 255 (201)       silence pad     0 (padding)   1             = 0

The model denoises the whole canvas but loss is taken only where (padding_mask & ~inpaint_mask),
i.e. the one-shot frames. Loop enters as clean context via inpaint_masked_input (256-dim) + the
1-ch mask -> 257 -> the per-block zero-init local-additive seam. No new parameters.

Latent geometry (channels / frame counts) is read from the npz meta at runtime, not hardcoded.
"""
import os, glob, json, random
import numpy as np, torch
from torch.utils.data import Dataset

INSTRUMENTS = {"os_kick": "kick", "os_snare": "snare", "os_hihats": "hi-hat"}
LOOP_KEYS   = ["loop_pad", "loop_tiled"]
# seq_params sample-name key -> (DOSE instrument dir, prompt label), for GT-lookup (VAL) mode
SEQ_INSTS   = [("kick_sample_name","kick","kick"), ("snare_sample_name","snare","snare"),
               ("hh_sample_name","hihats","hi-hat")]
# Canonical SAME "silence" latent frame (256,) — encoded digital silence, NOT zeros.
# Zeros are out-of-distribution and decode to audible noise; this decodes to true silence.
_SIL_PATH = os.path.join(os.path.dirname(__file__), "same_silence_latent.npy")

class DrumLatentCanvasDataset(Dataset):
    def __init__(self, roots, canvas_frames=256, prompt_tmpl="isolated {inst} drum one-shot",
                 loop_variant="both", seconds_mode="effective", gap_frames=3, limit=0, loop_range=None,
                 gt_lookup=None, drop_modes=None):
        """roots: list of dirs to glob for *.npz (e.g. the TRAIN G: tree).
        loop_variant: 'pad' | 'tiled' | 'both' (both => each loop variant is its own example).
        seconds_mode: 'effective' (real frames/fps) or 'loop' (loop seconds).
        loop_range: (lo,hi) -> keep only loop_N with lo<=N<hi (held-out split by loop index)."""
        import re as _re
        self.files = []
        for r in roots:
            self.files += glob.glob(os.path.join(r, "**", "*.npz"), recursive=True)
        if loop_range is not None:
            lo, hi = loop_range
            def _idx(p):
                m = _re.search(r"loop_(\d+)", p.replace("\\", "/"))
                return int(m.group(1)) if m else -1
            self.files = [f for f in self.files if lo <= _idx(f) < hi]
        if drop_modes:                                        # e.g. ["mode03"] -> exclude FX-augmented loops
            self.files = [f for f in self.files if not any(dm in os.path.basename(f) for dm in drop_modes)]
        self.files.sort()
        if limit: self.files = self.files[:limit]
        assert self.files, f"no npz under {roots} (loop_range={loop_range})"
        self.N = canvas_frames
        self.prompt_tmpl = prompt_tmpl
        self.loop_variant = loop_variant
        self.seconds_mode = seconds_mode
        self.gap = gap_frames                                 # silent buffer frames between loop and one-shot
        self.gt = gt_lookup                                   # {"inst/name.wav": latent}: VAL GT-lookup mode
        # (256,1) real silence latent to fill the gap (in-distribution, decodes to true silence)
        self.sil = torch.from_numpy(np.load(_SIL_PATH).astype(np.float32)).unsqueeze(1) if os.path.exists(_SIL_PATH) else None
        # expand (file, loop_variant, instrument) index so each item is one supervised pair
        self.index = []
        for f in self.files:
            variants = LOOP_KEYS if loop_variant == "both" else [f"loop_{loop_variant}"]
            self.index.append((f, variants))  # instruments resolved at load (depends on presence)

    def __len__(self):
        return len(self.files)

    def _build(self, zL, zO, inst, meta):
        C = zL.shape[0]; N = self.N; G = self.gap
        nL, nO = zL.shape[1], zO.shape[1]
        os0 = nL + G                                          # one-shot start (after loop + gap)
        os1 = os0 + nO
        assert os1 <= N, f"canvas {N} too small for {nL}+{G}+{nO}"
        x1 = torch.zeros(C, N, dtype=torch.float32)
        x1[:, :nL] = zL                                      # frames 0..nL-1        loop context
        x1[:, os0:os1] = zO                                  # frames os0..os1-1     one-shot target
        if G > 0 and self.sil is not None:
            x1[:, nL:os0] = self.sil.expand(-1, G)           # gap = real silence latent (not zeros)
        inpaint_mask = torch.ones(1, N, dtype=torch.float32)
        inpaint_mask[:, os0:os1] = 0.0                       # 0 = generate (one-shot region only)
        masked_input = x1 * inpaint_mask                     # loop kept, gap=0, one-shot zeroed
        padding_mask = torch.zeros(N, dtype=torch.bool)
        padding_mask[:os1] = True                            # real = loop+gap+one-shot; rest padding
        fps = meta.get("latent_fps", 10.7666)
        seconds = (nL / fps) if self.seconds_mode == "loop" else (os1 / fps)
        prompt = self.prompt_tmpl.format(inst=inst)
        return {
            "x1": x1, "inpaint_mask": inpaint_mask, "inpaint_masked_input": masked_input,
            "padding_mask": padding_mask, "prompt": prompt, "seconds_total": float(seconds),
            "gen_slice": (os0, os1), "loop_slice": (0, nL), "uid": meta.get("uid",""), "instrument": inst,
        }

    def __getitem__(self, i):
        # Iterative (not recursive) scan so a run of unreadable/os-less files can't blow the stack.
        n = len(self.index)
        for step in range(n):
            j = (i + step) % n
            f, variants = self.index[j]
            try:
                d = np.load(f, allow_pickle=True)
                meta = json.loads(str(d["meta"]))
                lvar = random.choice(variants) if len(variants) > 1 else variants[0]
                if lvar not in d.files: lvar = LOOP_KEYS[0]
                zL = torch.from_numpy(d[lvar].astype(np.float32))
                if self.gt is not None:
                    # VAL GT-lookup mode: one-shot comes from DOSE via seq_params sample names
                    sp = (meta.get("seq_params") or {})
                    present = []
                    for seqkey, dose_inst, label in SEQ_INSTS:
                        names = sp.get(seqkey) or []
                        if names and f"{dose_inst}/{names[0]}" in self.gt:
                            present.append((f"{dose_inst}/{names[0]}", label))
                    if not present:
                        continue
                    key, inst = random.choice(present)
                    zO = torch.from_numpy(self.gt[key].astype(np.float32))
                else:
                    present = [(k, INSTRUMENTS[k]) for k in INSTRUMENTS if k in d.files]
                    if not present:
                        continue
                    okey, inst = random.choice(present)
                    zO = torch.from_numpy(d[okey].astype(np.float32))
                return self._build(zL, zO, inst, meta)
            except Exception as e:                           # locked/corrupt/transient IO -> next file
                if step < 3: print(f"[dataset] skip {os.path.basename(f)}: {type(e).__name__}: {e}")
                continue
        raise RuntimeError(f"no usable npz found scanning from index {i}")


def collate(batch):
    out = {}
    for k in ("x1", "inpaint_mask", "inpaint_masked_input"):
        out[k] = torch.stack([b[k] for b in batch])
    out["padding_mask"] = torch.stack([b["padding_mask"] for b in batch])
    out["seconds_total"] = [b["seconds_total"] for b in batch]
    out["prompt"] = [b["prompt"] for b in batch]
    out["gen_slice"] = [b["gen_slice"] for b in batch]
    out["loop_slice"] = [b["loop_slice"] for b in batch]
    out["uid"] = [b["uid"] for b in batch]
    out["instrument"] = [b["instrument"] for b in batch]
    return out


if __name__ == "__main__":
    import sys
    roots = sys.argv[1:] or [os.environ.get("SA3_TRAIN_ROOT", "")]
    if not roots[0]:
        raise SystemExit("usage: python dataset.py <latent-root> [...]  (or set SA3_TRAIN_ROOT)")
    ds = DrumLatentCanvasDataset(roots, limit=200)
    print(f"dataset files: {len(ds)}")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, collate_fn=collate, shuffle=True)
    b = next(iter(dl))
    print("x1", b["x1"].shape, b["x1"].dtype)
    print("inpaint_mask", b["inpaint_mask"].shape, "ones/zeros per item:",
          [(int(m.sum()), int((m==0).sum())) for m in b["inpaint_mask"]])
    print("inpaint_masked_input", b["inpaint_masked_input"].shape,
          "loop-region nonzero, one-shot-region zero check:",
          [(float(b["inpaint_masked_input"][i,:, :44].abs().sum()>0),
            float(b["inpaint_masked_input"][i,:,44:55].abs().sum()==0)) for i in range(4)])
    print("padding_mask real frames:", [int(p.sum()) for p in b["padding_mask"]])
    print("seconds_total", [round(s,2) for s in b["seconds_total"]])
    print("prompts", b["prompt"])
    print("gen_slice", b["gen_slice"])
    print("instruments", b["instrument"])
