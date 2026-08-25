"""Generate held-out one-shots for a SPECIFIC instrument from a trained LoRA checkpoint.
Forces the target instrument (bypassing the dataset's random pick) so we can get a balanced set.
"""
import os, sys, json, argparse, glob
import numpy as np, torch, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
import train as T
from dataset import DrumLatentCanvasDataset, collate

SEQKEY={"kick":"kick_sample_name","snare":"snare_sample_name","hihats":"hh_sample_name"}
LABEL ={"kick":"kick","snare":"snare","hihats":"hi-hat"}

def load_lora_ckpt(model, path):
    from safetensors.torch import load_file
    sd=load_file(path)
    r1=model.model.load_state_dict(sd, strict=False)
    r2=model.conditioner.load_state_dict(sd, strict=False)
    loaded=len(sd) - (len([k for k in sd if k in r1.unexpected_keys and k in r2.unexpected_keys]))
    print(f"[lora] loaded {len(sd)} lora tensors from {os.path.basename(path)}")

def build_for(ds, npz_path, dose_inst):
    d=np.load(npz_path, allow_pickle=True); meta=json.loads(str(d["meta"]))
    sp=meta.get("seq_params") or {}
    names=sp.get(SEQKEY[dose_inst]) or []
    if not names: return None
    key=f"{dose_inst}/{names[0]}"
    if ds.gt is None or key not in ds.gt: return None
    if "loop_pad" not in d.files: return None
    zL=torch.from_numpy(d["loop_pad"].astype(np.float32))
    zO=torch.from_numpy(ds.gt[key].astype(np.float32))
    return ds._build(zL, zO, LABEL[dose_inst], meta)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:/stage3/optionB_run_nomode3/lora_last.safetensors")
    ap.add_argument("--insts", nargs="+", default=["kick","snare"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"D:/stage3/optionB_nomode3_extra")
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    model,cfg=T.load_model()
    T.add_lora_(model, cfg)
    load_lora_ckpt(model, args.ckpt)
    model.eval()

    gt=dict(np.load(r"G:/exp-datasets-same-latents/dose_val_oneshots.npz"))
    ds=DrumLatentCanvasDataset([T.VAL_ROOT], gap_frames=3, gt_lookup=gt, drop_modes=["mode03"])
    files=ds.files

    for inst in args.insts:
        picked=[]
        for f in files:
            ex=build_for(ds, f, inst)
            if ex is not None:
                picked.append(ex)
            if len(picked)>=args.n: break
        if not picked:
            print(f"[gen] no {inst} loops found"); continue
        batch=collate(picked)
        gen=T.sample_oneshot(model, batch, steps=args.steps)
        for i in range(gen.shape[0]):
            au=T.decode_region(model, gen[i:i+1], batch["gen_slice"][i])
            sf.write(os.path.join(args.out, f"{inst}_{i:02d}_gen.wav"), au, model.sample_rate, subtype="PCM_24")
            gt_au=T.decode_region(model, batch["x1"][i:i+1].to(T.DEV), batch["gen_slice"][i])
            sf.write(os.path.join(args.out, f"{inst}_{i:02d}_gt.wav"), gt_au, model.sample_rate, subtype="PCM_24")
            loop_au=T.decode_region(model, batch["x1"][i:i+1].to(T.DEV), batch["loop_slice"][i], trim=0)
            sf.write(os.path.join(args.out, f"{inst}_{i:02d}_loop.wav"), loop_au, model.sample_rate, subtype="PCM_24")
        print(f"[gen] {inst}: wrote {gen.shape[0]} (gen/gt/loop triplets)", flush=True)
    print(f"[done] -> {args.out}")

if __name__=="__main__":
    main()
