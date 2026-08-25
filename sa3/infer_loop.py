"""Extract kick/snare/hihat one-shots from an ARBITRARY external loop wav (no GT needed).
Encodes a 4s window of the loop with the frozen SAME AE, feeds it through the trained inpaint
seam, and samples the one-shot region for each instrument (multiple seeds for variety)."""
import os, sys, argparse
import numpy as np, torch, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
import train as T
from generate import load_lora_ckpt

INSTS=[("kick","kick"),("snare","snare"),("hihats","hi-hat")]  # (id, prompt label)
SIL=os.path.join(os.path.dirname(__file__),"same_silence_latent.npy")

def load_loop(path, sr, n):
    x,fsr=sf.read(path, dtype="float32", always_2d=True)     # (T,ch)
    x=torch.from_numpy(x.T)                                   # (ch,T)
    if x.shape[0]==1: x=x.repeat(2,1)
    elif x.shape[0]>2: x=x[:2]
    if fsr!=sr:
        import torchaudio
        x=torchaudio.functional.resample(x, fsr, sr)
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--loop", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=r"D:/stage3/optionB_run_nomode3/lora_last.safetensors")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank -- MUST match the checkpoint (r4->4)")
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--repo-cfg", default=None, help="base model_config.json (small vs medium)")
    ap.add_argument("--base-ckpt", default=None, help="base model.safetensors (small vs medium)")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--no-lora", action="store_true", help="raw SA3 base model (no LoRA) baseline")
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prefix = "baseline_" if args.no_lora else ""

    if args.repo_cfg:  T.REPO_CFG = args.repo_cfg     # select small vs medium base
    if args.base_ckpt: T.CKPT     = args.base_ckpt
    model,cfg=T.load_model()
    if not args.no_lora:
        T.add_lora_(model, cfg, rank=args.rank, alpha=args.alpha); load_lora_ckpt(model, args.ckpt)
    else:
        print("[baseline] raw SA3 base model, NO LoRA", flush=True)
    model.eval()
    sr=model.sample_rate; dsr=model.pretransform.downsampling_ratio; fps=sr/dsr
    loop_n=int(round(4.0*sr))

    # load + crop 4s window, encode -> loop latent
    x=load_loop(args.loop, sr, loop_n*4)
    s0=int(round(args.start_sec*sr))
    seg=x[:, s0:s0+loop_n]
    if seg.shape[-1] < loop_n: seg=torch.nn.functional.pad(seg,(0,loop_n-seg.shape[-1]))
    model.pretransform.eval()
    with torch.no_grad():
        zL=model.pretransform.encode(seg.unsqueeze(0).to(T.DEV)).squeeze(0)   # (256, ~44)
    nL=zL.shape[1]
    sf.write(os.path.join(args.out,"loop_input.wav"), seg.mean(0).clamp(-1,1).numpy(), sr, subtype="PCM_24")
    print(f"[loop] {os.path.basename(args.loop)} | 4s window @ {args.start_sec}s -> {nL} latent frames", flush=True)

    # build canvas: loop | silence gap | (one-shot region generated) | silence pad
    sil=torch.from_numpy(np.load(SIL).astype(np.float32)).unsqueeze(1).to(T.DEV)  # (256,1)
    N=256; gap=3; os0=nL+gap; nO=11; os1=os0+nO
    x1=torch.zeros(256,N,device=T.DEV); x1[:,:nL]=zL; x1[:,nL:os0]=sil.expand(-1,gap)
    imask=torch.ones(1,N,device=T.DEV); imask[:,os0:os1]=0.0
    minp=x1*imask
    pad=torch.zeros(N,dtype=torch.bool,device=T.DEV); pad[:os1]=True
    seconds=os1/fps
    B=len(INSTS)
    batch={
        "x1": x1.unsqueeze(0).repeat(B,1,1), "inpaint_mask": imask.unsqueeze(0).repeat(B,1,1),
        "inpaint_masked_input": minp.unsqueeze(0).repeat(B,1,1),
        "padding_mask": pad.unsqueeze(0).repeat(B,1),
        "seconds_total":[seconds]*B, "prompt":[f"isolated {lbl} drum one-shot" for _,lbl in INSTS],
        "gen_slice":[(os0,os1)]*B, "loop_slice":[(0,nL)]*B,
        "instrument":[lbl for _,lbl in INSTS], "uid":["loop"]*B,
    }
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        gen=T.sample_oneshot(model, batch, steps=args.steps)
        for i,(iid,lbl) in enumerate(INSTS):
            au=T.decode_region(model, gen[i:i+1], batch["gen_slice"][i])
            sf.write(os.path.join(args.out, f"{prefix}{iid}_seed{seed}_gen.wav"), au, sr, subtype="PCM_24")
        print(f"[gen] seed {seed}: wrote {B} one-shots", flush=True)
    print(f"[done] -> {args.out}")

if __name__=="__main__":
    main()
