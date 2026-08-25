"""
velocity_estimation.py -- analysis-by-synthesis per-onset velocity estimation.

Freeze onsets + one-shots, render the loop differentiably (differentiable_renderer.py,
absolute sample domain, summed mix, no source separation), and optimise the per-onset
velocities by back-prop on the reconstruction loss.

This module provides the reusable pieces:
  * make_loss(kind)            -- L1 / MR-STFT / both reconstruction loss
  * optimize_velocities(...)   -- the Adam optimisation over per-onset velocities
  * pearson / scale_norm_mae   -- scale-robust per-track velocity metrics
  * velocity_metrics(...)      -- aggregate metrics for one loop (per-track -> mean)

The __main__ block is a self-contained ORACLE self-test: it builds random one-shots,
onsets and ground-truth velocities, renders a target with the same renderer, then checks
the optimiser recovers the velocities (Pearson ~ 1, scale-normalised MAE ~ 0). It needs no
dataset, no one-shot library, and no ADT/one-shot estimation systems.
"""
import math
import torch
import torch.nn.functional as F

from differentiable_renderer import DifferentiableDrumRenderer

try:
    import auraloss
    _HAVE_AURALOSS = True
except Exception:
    _HAVE_AURALOSS = False


# ---------------------------------------------------------------- losses
def make_loss(kind="both", alpha_time=1.0, alpha_freq=1.0,
              fft_sizes=(256, 512, 1024), hop_sizes=(64, 128, 256), win_lengths=(256, 512, 1024)):
    """kind in {'l1','mrstft','both'}. Operates on summed mixes shaped [1, T]."""
    mr = None
    if kind in ("mrstft", "both"):
        if not _HAVE_AURALOSS:
            raise ImportError("auraloss required for MR-STFT loss (pip install auraloss)")
        mr = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=list(fft_sizes),
                                                   hop_sizes=list(hop_sizes),
                                                   win_lengths=list(win_lengths))

    def loss_fn(pred, target):                                 # pred,target: [1, T]
        total = pred.new_zeros(())
        if kind in ("l1", "both"):
            total = total + alpha_time * F.l1_loss(pred, target)
        if kind in ("mrstft", "both"):
            total = total + alpha_freq * mr(pred.unsqueeze(0), target.unsqueeze(0))  # [1,1,T]
        return total
    return loss_fn


# ---------------------------------------------------------------- optimiser
def optimize_velocities(renderer, one_shots, onsets, target, *, mode="poly", smoothing=True,
                        loss_kind="both", alpha_time=1.0, alpha_freq=1.0, lr=0.05, iters=500,
                        transform="sigmoid", init_logit=4.0, loop_length=None, device=None,
                        patience=60, min_delta=1e-6, verbose=False):
    """Estimate per-onset velocities for one loop.

    renderer  : DifferentiableDrumRenderer
    one_shots : [B, 1, K]
    onsets    : list of B LongTensors (sample positions per track)
    target    : [1, T] summed loop to reconstruct
    transform : 'sigmoid' (velocity in (0,1)) or 'softplus' (>=0)
    returns   : (velocities: list of B tensors, info dict)
    """
    device = device or (target.device if target.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu"))
    one_shots = one_shots.to(device)
    target = target.to(device)
    T = int(loop_length) if loop_length is not None else target.shape[-1]
    onsets = [torch.as_tensor(o, device=device).long().reshape(-1) for o in onsets]

    logits = [torch.full((o.numel(),), float(init_logit), device=device, requires_grad=True)
              for o in onsets]
    act = torch.sigmoid if transform == "sigmoid" else F.softplus
    opt = torch.optim.Adam([l for l in logits if l.numel() > 0], lr=lr)
    loss_fn = make_loss(loss_kind, alpha_time, alpha_freq)

    best, best_it, best_vel = float("inf"), 0, None
    for it in range(iters):
        velocities = [act(l) for l in logits]
        mix = renderer.render_from_onsets(one_shots, onsets, velocities, loop_length=T,
                                          mode=mode, smoothing_envelopes=smoothing)
        loss = loss_fn(mix, target)
        opt.zero_grad(); loss.backward(); opt.step()
        lv = float(loss.detach())
        if lv < best - min_delta:
            best, best_it = lv, it
            best_vel = [act(l).detach().clone() for l in logits]
        if verbose and it % max(1, iters // 10) == 0:
            print(f"    it {it:4d}  loss {lv:.6f}")
        if it - best_it >= patience:
            break
    return best_vel, {"final_loss": best, "iters_run": it + 1, "best_it": best_it}


# ---------------------------------------------------------------- metrics
def pearson(est, gt):
    """Scale/offset-invariant. Returns nan if <2 points or zero variance."""
    if est.numel() < 2:
        return float("nan")
    a = est - est.mean(); b = gt - gt.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if float(denom) > 0 else float("nan")


def scale_norm_mae(est, gt):
    """MAE after the optimal per-track gain a* = <est,gt>/<est,est> (handles scale ambiguity)."""
    if est.numel() == 0:
        return float("nan")
    ee = float(est @ est)
    a = float(est @ gt) / ee if ee > 0 else 1.0
    return float((a * est - gt).abs().mean())


def velocity_metrics(est_list, gt_list):
    """est_list, gt_list: per-track lists of 1-D tensors (one loop). Per-track -> mean over tracks."""
    prs, maes, cov = [], [], 0
    for est, gt in zip(est_list, gt_list):
        est, gt = est.detach().float().cpu(), gt.detach().float().cpu()
        r = pearson(est, gt)
        maes.append(scale_norm_mae(est, gt))
        if not math.isnan(r):
            prs.append(r); cov += 1
    return {
        "pearson": (sum(prs) / len(prs)) if prs else float("nan"),
        "mae_scalenorm": (sum(maes) / len(maes)) if maes else float("nan"),
        "n_tracks": len(gt_list),
        "n_tracks_pearson": cov,       # tracks with a defined Pearson (>=2 onsets, GT variance>0)
    }


# ---------------------------------------------------------------- oracle self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    sr = 44100
    T = 2 * sr
    r = DifferentiableDrumRenderer(sample_rate=sr, loop_length=T)
    B, K = 3, int(0.25 * sr)
    one_shots = torch.randn(B, 1, K)

    # random onsets per track (sorted, spaced) + ground-truth velocities in (0.2, 1.0)
    onsets, gt_vel = [], []
    for _ in range(B):
        n = torch.randint(6, 14, (1,)).item()
        pos = torch.sort(torch.randint(0, T - K, (n,))).values
        onsets.append(pos)
        gt_vel.append(torch.empty(n).uniform_(0.2, 1.0))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    print(f"{'mode':<6}{'smooth':<8}{'loss':<8}{'pearson':>9}{'mae*':>8}{'recon':>10}{'iters':>7}")
    for mode in ("poly", "mono"):
        for sm in (True, False):
            for lk in ("both", "l1", "mrstft"):
                target = r.render_from_onsets(one_shots, onsets, gt_vel, loop_length=T,
                                              mode=mode, smoothing_envelopes=sm).detach()
                est, info = optimize_velocities(r, one_shots, onsets, target, mode=mode, smoothing=sm,
                                                loss_kind=lk, lr=0.1, iters=400, device=dev)
                m = velocity_metrics(est, gt_vel)
                print(f"{mode:<6}{str(sm):<8}{lk:<8}{m['pearson']:>9.4f}{m['mae_scalenorm']:>8.4f}"
                      f"{info['final_loss']:>10.5f}{info['iters_run']:>7d}")
    print("OK (oracle: pearson should be ~1.0, mae* ~0)")
