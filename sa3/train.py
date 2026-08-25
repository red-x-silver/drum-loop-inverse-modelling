"""Option B trainer — LoRA on SA3's existing inpaint seam, continuation-canvas fixed mask.

Faithfully mirrors SA3's DiffusionCondInpaintTrainingWrapper rf_denoiser step, but substitutes
our FIXED continuation-canvas mask (loop=context, one-shot=generate) for the random inpaint mask,
and reads pre-encoded SAME latents from the cache. Base frozen, only LoRA (rank-16) trains.

Modes:
  --test-forward     load model, run ONE training step, print loss (wiring check)
  --smoke N          overfit K cached pairs for N steps; log loss; sample+decode a one-shot

Run:  .venv/Scripts/python.exe scripts_drumext/optionB_train.py --smoke 400 --pairs 8
"""
import os, sys, json, struct, argparse, time, math
import numpy as np, torch, torch.nn.functional as F, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__))
from dataset import DrumLatentCanvasDataset, collate

REPO_CFG = r"D:\stable-audio-3\stabilityaistable-audio-3-medium\model_config.json"
CKPT     = r"D:\stable-audio-3\stabilityaistable-audio-3-medium\model.safetensors"
CACHE    = r"G:/exp-datasets-same-latents/TRAIN-allconfigs/none-mono"
TRAIN_ROOT = r"G:/exp-datasets-same-latents/TRAIN-allconfigs"
VAL_ROOT   = r"G:/exp-datasets-same-latents/VAL-allconfigs"
OUT      = r"D:\stage3\optionB_smoke"
RUN_DIR  = r"D:\stage3\optionB_run"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def read_safetensors_all(path):
    """Read every tensor via plain file reads (avoids mmap/commit issues on the 9 GB file)."""
    npd={'F32':np.float32,'F16':np.float16,'F64':np.float64,'I64':np.int64,'I32':np.int32,'U8':np.uint8,'BOOL':np.bool_}
    tdt={'F32':torch.float32,'F16':torch.float16,'F64':torch.float64,'BF16':torch.bfloat16,'I64':torch.int64,'I32':torch.int32,'U8':torch.uint8,'BOOL':torch.bool}
    out={}
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); start=8+n
        for k,m in h.items():
            if k=='__metadata__': continue
            a,b=m['data_offsets']
            if b-a==0: out[k]=torch.empty(m['shape'],dtype=tdt[m['dtype']]); continue
            if m['dtype']=='BF16':
                arr=np.fromfile(path,dtype=np.uint16,count=(b-a)//2,offset=start+a)
                out[k]=torch.from_numpy(arr).view(torch.bfloat16).reshape(m['shape'])
            else:
                out[k]=torch.from_numpy(np.fromfile(path,dtype=npd[m['dtype']],count=(b-a)//np.dtype(npd[m['dtype']]).itemsize,offset=start+a)).reshape(m['shape'])
    return out

def load_model():
    from stable_audio_tools.models.factory import create_model_from_config
    cfg=json.load(open(REPO_CFG))
    model=create_model_from_config(cfg)
    sd=read_safetensors_all(CKPT)
    r=model.load_state_dict(sd, strict=False)
    # DiT ('model.*') and inpaint/seconds projections must all load; t5gemma comes from HF cache.
    dit_missing=[k for k in r.missing_keys if k.startswith("model.")]
    assert len(dit_missing)==0, f"DiT weights incomplete: {len(dit_missing)} missing e.g. {dit_missing[:3]}"
    print(f"[load] state_dict: {len(sd)} tensors | missing {len(r.missing_keys)} (t5gemma from HF), unexpected {len(r.unexpected_keys)}, DiT fully loaded")
    del sd
    model=model.to(DEV).eval()
    return model, cfg

def add_lora_(model, cfg, rank=16, alpha=None):
    from functools import partial
    from stable_audio_tools.models.lora import add_lora, get_lora_params, LoRAParametrization
    # matches model_config_lora.json; build the layer-type-keyed config the wrapper uses
    alpha = rank if alpha is None else alpha
    adapter, exc, inc = "lora", ["seconds_total"], None
    lora_cfg = {
        torch.nn.Linear: {"weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=alpha, adapter_type=adapter)},
        torch.nn.Conv1d: {"weight": partial(LoRAParametrization.from_conv1d, rank=rank, lora_alpha=alpha, adapter_type=adapter)},
    }
    model.model.requires_grad_(False); model.conditioner.requires_grad_(False)
    if model.pretransform is not None: model.pretransform.requires_grad_(False)
    add_lora(model.model, lora_cfg, include=inc, exclude=exc)
    add_lora(model.conditioner, lora_cfg, include=inc, exclude=exc)
    params=[*get_lora_params(model.model), *get_lora_params(model.conditioner)]
    ntrain=sum(p.numel() for p in params)
    print(f"[lora] rank={rank} alpha={alpha} exclude={exc} | trainable LoRA params: {ntrain:,}")
    return params

def sample_t(B):
    try:
        from stable_audio_tools.training.utils import truncated_logistic_normal_rescaled
        t=1-truncated_logistic_normal_rescaled(B).to(DEV)
    except Exception:
        t=torch.sigmoid(torch.randn(B,device=DEV))   # logit-normal fallback
    return t

def build_cond(model, batch):
    md=[{"prompt":p, "seconds_total":s} for p,s in zip(batch["prompt"], batch["seconds_total"])]
    cond=model.conditioner(md, DEV)
    cond["inpaint_mask"]=[batch["inpaint_mask"].to(DEV)]
    cond["inpaint_masked_input"]=[batch["inpaint_masked_input"].to(DEV)]
    return cond

def _mrstft(pred, gt, ffts, eps=1e-5):
    """Multi-resolution STFT: log-magnitude L1 + magnitude L1 over fft sizes.
    pred/gt: (B,S) mono -> PER-SAMPLE loss (B,)."""
    loss=0.0; k=0
    for n in ffts:
        if pred.shape[-1] < n: continue
        win=torch.hann_window(n, device=pred.device)
        P=torch.stft(pred, n, n//4, n, win, return_complex=True).abs()
        G=torch.stft(gt,   n, n//4, n, win, return_complex=True).abs()
        loss=loss + (torch.log(P+eps)-torch.log(G+eps)).abs().mean(dim=(1,2)) + (P-G).abs().mean(dim=(1,2))
        k+=1
    return loss/max(k,1)

def transient_loss(model, x1_hat, x1, gen_slice, atk_ms=30.0, sub=None):
    """Decode the one-shot region of x1_hat (grad) vs x1 (target), attack-weighted MR-STFT.
    Frozen SAME AE kept in eval so decode is deterministic (no softnorm train-noise)."""
    g0,g1=gen_slice; sr=model.sample_rate
    B=x1_hat.shape[0]; sub=B if sub is None else min(sub,B)
    model.pretransform.eval()
    pred_a=model.pretransform.decode(x1_hat[:sub,:,g0:g1])   # (sub,2,S) grad flows to x1_hat->pred->LoRA
    with torch.no_grad():
        gt_a=model.pretransform.decode(x1[:sub,:,g0:g1])
    p=pred_a.mean(1); g=gt_a.mean(1)                         # mono (sub,S)
    full=_mrstft(p, g, (2048,1024,512,256,128,64))
    n=int(sr*atk_ms/1000)
    atk=_mrstft(p[:,:n], g[:,:n], (512,256,128,64))          # first ~30 ms, upweighted by adding
    return full + atk

def train_step(model, batch, cfg_dropout=0.0, transient_weight=0.0, transient_sub=2):
    x1=batch["x1"].to(DEV)                                   # (B,256,256)
    pad=batch["padding_mask"].to(DEV)                        # (B,256) bool
    imask=batch["inpaint_mask"].to(DEV)                      # (B,1,256)
    cond=build_cond(model, batch)
    B=x1.shape[0]
    t=sample_t(B)
    if getattr(model, "dist_shift", None) is not None:
        ds=model.pretransform.downsampling_ratio
        eff=torch.tensor([int(math.ceil(int(s*model.sample_rate)/ds)) for s in batch["seconds_total"]],device=DEV)
        t=model.dist_shift.shift(t, eff)
    a=(1-t)[:,None,None]; sg=t[:,None,None]
    noise=torch.randn_like(x1)
    noised=x1*a + noise*sg
    target=noise - x1                                        # rf_denoiser velocity
    pred=model(noised, t, cond=cond, cfg_dropout_prob=cfg_dropout, padding_mask=pad)
    # latent loss only on generate region: padding=True AND inpaint_mask==0
    lm=(pad & (~imask.squeeze(1).bool())).unsqueeze(1).expand_as(pred)
    loss_lat=F.mse_loss(pred[lm], target[lm])
    loss_tr=torch.tensor(0.0, device=DEV)
    if transient_weight > 0:
        x1_hat=noised - sg*pred                             # rf: recover clean-data estimate
        tr_ps=transient_loss(model, x1_hat, x1, batch["gen_slice"][0], sub=transient_sub)  # (sub,)
        w=(1.0 - t[:tr_ps.shape[0]]).clamp(min=0)           # trust low-noise estimates; ~0 at t->1
        loss_tr=(w*tr_ps).sum()/(w.sum()+1e-6)
        loss=loss_lat + transient_weight*loss_tr
    else:
        loss=loss_lat
    return loss, loss_lat.detach(), loss_tr.detach()

@torch.no_grad()
def sample_oneshot(model, batch, steps=50):
    """rf euler inpaint sampling: denoise full canvas, extract the generate (one-shot) region."""
    x1=batch["x1"].to(DEV); pad=batch["padding_mask"].to(DEV); imask=batch["inpaint_mask"].to(DEV)
    cond=build_cond(model, batch)
    B=x1.shape[0]
    x=torch.randn_like(x1)
    ts=torch.linspace(1,0,steps+1,device=DEV)
    for i in range(steps):
        t=ts[i].expand(B)
        v=model(x, t, cond=cond, cfg_dropout_prob=0.0, padding_mask=pad)
        x=x + v*(ts[i+1]-ts[i])                             # euler toward t=0
    return x                                                 # (B,256,256) generated canvas

def decode_region(model, z_canvas, sl, trim=3e-3):
    g0,g1=sl
    model.pretransform.eval()
    with torch.no_grad():
        y=model.pretransform.decode(z_canvas[:, :, g0:g1]).squeeze(0).mean(0).clamp(-1,1).cpu().numpy()
    if trim and trim>0:
        e=np.abs(y); idx=np.where(e>trim)[0]
        return y[:idx[-1]+1] if len(idx) else y
    return y

def save_lora(model, path, rank=16, alpha=None):
    from stable_audio_tools.models.lora import get_lora_state_dict, save_lora_safetensors
    alpha = rank if alpha is None else alpha
    sd={**get_lora_state_dict(model.model), **get_lora_state_dict(model.conditioner)}
    save_lora_safetensors(sd, {"rank":rank,"alpha":alpha,"adapter_type":"lora","exclude":["seconds_total"]}, path)

def run_full(args, model, params):
    RUN_DIR=args.run_dir
    os.makedirs(RUN_DIR, exist_ok=True); os.makedirs(os.path.join(RUN_DIR,"demos"), exist_ok=True)
    from torch.utils.data import DataLoader
    t0=time.time()
    # Train on all TRAIN loops (baked-in one-shots). Held-out eval = the real VAL loops paired
    # with their TRUE GT one-shots, fetched from DOSE validation via seq_params sample names.
    GT_LOOKUP=r"G:/exp-datasets-same-latents/dose_val_oneshots.npz"
    gt=dict(np.load(GT_LOOKUP)) if os.path.exists(GT_LOOKUP) else None
    dm=args.drop_modes or None
    train_ds=DrumLatentCanvasDataset([TRAIN_ROOT], gap_frames=3, drop_modes=dm)
    val_ds  =DrumLatentCanvasDataset([VAL_ROOT],   gap_frames=3, gt_lookup=gt, drop_modes=dm)
    print(f"[data] train {len(train_ds)} npz (TRAIN), held-out {len(val_ds)} npz (VAL loops, GT one-shots from {len(gt) if gt else 0} DOSE-val) | drop_modes={dm} ({time.time()-t0:.0f}s glob)", flush=True)
    dl=DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2,
                  collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_batch=collate([val_ds[i] for i in range(args.demos)])   # fixed held-out demo set
    opt=torch.optim.AdamW(params, lr=args.lr)
    model.train(); model.pretransform.eval()
    step=0; micro=0; run=rl=rt=0.0; accum=args.grad_accum
    opt.zero_grad()
    while step < args.steps:
        for batch in dl:
            loss,ll,lt=train_step(model, batch, cfg_dropout=0.1,
                                  transient_weight=args.transient_weight, transient_sub=args.transient_sub)
            (loss/accum).backward(); run+=loss.item(); rl+=ll.item(); rt+=lt.item(); micro+=1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step+=1
                if step % 20 == 0:
                    d=20*accum
                    print(f"[step {step:5d}/{args.steps}] loss={run/d:.4f} (lat={rl/d:.4f} tr={rt/d:.4f}) | {step*accum*args.bs/(time.time()-t0):.0f} samp/s", flush=True)
                    run=rl=rt=0.0
                if step % args.demo_every == 0 or step==args.steps:
                    model.eval()
                    gen=sample_oneshot(model, val_batch, steps=50)
                    for i in range(gen.shape[0]):
                        inst=val_batch['instrument'][i]
                        au=decode_region(model, gen[i:i+1], val_batch["gen_slice"][i])
                        sf.write(os.path.join(RUN_DIR,"demos",f"s{step:05d}_{inst}_{i}_gen.wav"),
                                 au, model.sample_rate, subtype="PCM_24")
                        if step==args.demo_every:            # fixed inputs -> save once
                            gt_au=decode_region(model, val_batch["x1"][i:i+1].to(DEV), val_batch["gen_slice"][i])
                            sf.write(os.path.join(RUN_DIR,"demos",f"gt_{inst}_{i}.wav"),
                                     gt_au, model.sample_rate, subtype="PCM_24")
                            loop_au=decode_region(model, val_batch["x1"][i:i+1].to(DEV), val_batch["loop_slice"][i], trim=0)
                            sf.write(os.path.join(RUN_DIR,"demos",f"loop_{inst}_{i}.wav"),
                                     loop_au, model.sample_rate, subtype="PCM_24")
                    save_lora(model, os.path.join(RUN_DIR, f"lora_step{step:05d}.safetensors"), args.rank, args.alpha)
                    save_lora(model, os.path.join(RUN_DIR, "lora_last.safetensors"), args.rank, args.alpha)
                    print(f"[demo+ckpt @ step {step}] wrote {gen.shape[0]} held-out one-shots + lora", flush=True)
                    model.train()
                if step>=args.steps: break
    print(f"[done] full run {step} steps in {(time.time()-t0)/60:.1f}min -> {RUN_DIR}", flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test-forward", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--demo-every", type=int, default=500)
    ap.add_argument("--demos", type=int, default=12)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--transient-weight", type=float, default=0.0, help="weight of the decoded attack-weighted MR-STFT loss")
    ap.add_argument("--transient-sub", type=int, default=2, help="how many batch items to decode for the transient loss (VRAM/speed)")
    ap.add_argument("--run-dir", default=RUN_DIR, help="output dir for demos + LoRA checkpoints")
    ap.add_argument("--drop-modes", nargs="*", default=None, help="mode substrings to exclude, e.g. mode03")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank")
    ap.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default = rank)")
    ap.add_argument("--repo-cfg", default=None, help="override model_config.json path (e.g. the small model)")
    ap.add_argument("--ckpt", default=None, help="override model.safetensors path")
    args=ap.parse_args()
    if args.repo_cfg or args.ckpt:
        global REPO_CFG, CKPT
        if args.repo_cfg: REPO_CFG=args.repo_cfg
        if args.ckpt: CKPT=args.ckpt
        print(f"[cfg] using model: {REPO_CFG}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    t0=time.time()
    model,cfg=load_model()
    params=add_lora_(model, cfg, rank=args.rank, alpha=args.alpha)
    print(f"[ready] model+lora in {time.time()-t0:.0f}s | sr={model.sample_rate}")

    if args.full:
        run_full(args, model, params)
        return

    ds=DrumLatentCanvasDataset([CACHE], gap_frames=3, limit=max(args.pairs, args.bs))
    from torch.utils.data import DataLoader, Subset
    fixed=Subset(ds, list(range(args.pairs)))                # small fixed set to overfit
    dl=DataLoader(fixed, batch_size=args.bs, collate_fn=collate, shuffle=True)

    if args.test_forward:
        batch=collate([ds[i] for i in range(args.bs)])
        loss,ll,lt=train_step(model, batch, transient_weight=args.transient_weight, transient_sub=args.transient_sub)
        print(f"[test-forward] loss={loss.item():.4f} (lat={ll.item():.4f} tr={lt.item():.4f}) finite={torch.isfinite(loss).item()}")
        return

    opt=torch.optim.AdamW(params, lr=args.lr)
    model.train(); model.pretransform.eval(); step=0
    while step < args.smoke:
        for batch in dl:
            opt.zero_grad()
            loss,ll,lt=train_step(model, batch, cfg_dropout=0.0,
                                  transient_weight=args.transient_weight, transient_sub=args.transient_sub)
            loss.backward(); opt.step(); step+=1
            if step % 25 == 0 or step==1:
                print(f"[step {step:4d}] loss={loss.item():.4f} (lat={ll.item():.4f} tr={lt.item():.4f})", flush=True)
            if step>=args.smoke: break
    # sample + decode the overfit pairs
    model.eval()
    batch=collate([ds[i] for i in range(min(args.pairs,4))])
    gen=sample_oneshot(model, batch, steps=50)
    for i in range(gen.shape[0]):
        au=decode_region(model, gen[i:i+1], batch["gen_slice"][i])
        fn=os.path.join(OUT, f"gen_{batch['instrument'][i]}_{i}.wav")
        sf.write(fn, au, model.sample_rate, subtype="PCM_24")
        # also the ground-truth target for A/B
        gt=decode_region(model, batch["x1"][i:i+1].to(DEV), batch["gen_slice"][i])
        sf.write(fn.replace("gen_","gt_"), gt, model.sample_rate, subtype="PCM_24")
        print(f"  wrote gen/gt {batch['instrument'][i]} ({len(au)/model.sample_rate:.2f}s)")
    print(f"[done] {OUT}")

if __name__=="__main__":
    main()
