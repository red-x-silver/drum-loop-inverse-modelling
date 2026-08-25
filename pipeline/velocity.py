"""Per-onset velocity estimation via analysis-by-synthesis, with the locked optimal configuration
(loss=both, lr=0.1, 500 iters, sigmoid init 10, smoothing on, polyphonic)."""
import torch

from . import config as C
from differentiable_renderer import DifferentiableDrumRenderer
from velocity_estimation import optimize_velocities


def make_renderer():
    return DifferentiableDrumRenderer(sample_rate=C.SR, loop_length=C.LOOP_LEN)


def estimate_velocities(renderer, one_shots, onsets_samples, target, device=None):
    """one_shots [3,1,K]; onsets_samples: list of 3 python lists (sample positions); target [1,T].
    Returns (velocities: list of 3 tensors, reconstruction mix [1,T], info dict)."""
    device = device or C.device()
    onsets = [torch.tensor(o, dtype=torch.long) for o in onsets_samples]
    T = target.shape[-1]
    vels, info = optimize_velocities(
        renderer, one_shots, onsets, target, mode=C.VEL_MODE, smoothing=C.VEL_SMOOTHING,
        loss_kind=C.VEL_LOSS, lr=C.VEL_LR, iters=C.VEL_ITERS, transform=C.VEL_TRANSFORM,
        init_logit=C.VEL_INIT_LOGIT, loop_length=T, device=device)
    with torch.no_grad():
        mix = renderer.render_from_onsets(one_shots.to(device), onsets, [v.to(device) for v in vels],
                                          loop_length=T, mode=C.VEL_MODE,
                                          smoothing_envelopes=C.VEL_SMOOTHING).cpu()
    return [v.detach().cpu() for v in vels], mix, info
