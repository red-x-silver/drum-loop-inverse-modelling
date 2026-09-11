"""Phase 2 quantisation: absolute-time onsets/velocities -> step-based parameters.

Self-contained implementation matching the thesis description exactly (one bar, first N*tau period):
  * onset -> step: floor against the grid with a swing-tolerant forward rounding at rho, then mod N
  * beat-type: 16th if any occupied odd step, else 8th
  * swing per track: mean grid residual on swing-affected steps, inverting delta = (2*sigma-1)*tau,
    clipped to [0.50, 0.71] and snapped to the established swing presets
  * step velocity: SAME floor+rho assignment as q_j; MAX of onsets folded on the step
    (track-mean fallback for an active step with no folded onset)

The swing classes are those used by the dataset generator (kon_sequencer/data_modules.py). rho (0.75)
exceeds the largest swing offset ratio (2*0.71 - 1 = 0.42) so even a maximally-swung onset is not
rounded forward off its step.
"""
import numpy as np

from . import config as C

RHO = 0.75
SWING_CLASSES = [0.50, 0.54, 0.58, 0.62, 0.66, 0.71]


def _assign_step(t, tau, num_steps, rho=RHO):
    k = int(np.floor(t / tau))
    if (t - k * tau) / tau > rho:
        k += 1
    return k % num_steps


def quantize_params(onsets_s, velocities, tempo, num_steps=16, steps_per_beat=4):
    """onsets_s: list[3] of onset-second lists (kick,snare,hh); velocities: aligned list[3] or None.
    Only the first bar (t < N*tau) is quantised, as the loop repeats it."""
    tau = 60.0 / (tempo * steps_per_beat)
    bar = num_steps * tau
    n = len(onsets_s)

    step_vectors = [[0] * num_steps for _ in range(n)]
    vel_acc = [{s: [] for s in range(num_steps)} for _ in range(n)]      # onset velocities per step
    assigned = [[] for _ in range(n)]                                    # (step, onset_time) first bar

    for j in range(n):
        vj = velocities[j] if (velocities is not None and velocities[j] is not None) else None
        for i, t in enumerate(onsets_s[j]):
            if t < 0 or t >= bar:                                        # first bar only
                continue
            k = _assign_step(t, tau, num_steps)
            step_vectors[j][k] = 1
            assigned[j].append((k, float(t)))
            if vj is not None and i < len(vj):
                vel_acc[j][k].append(float(vj[i]))

    # beat type -> swing-affected steps
    odd = any(step_vectors[j][s] for j in range(n) for s in range(1, num_steps, 2))
    beat_type = "16th" if odd else "8th"
    W = {2, 6, 10, 14} if beat_type == "8th" else set(range(1, num_steps, 2))

    # swing per track from grid residuals on swing steps
    swing = []
    for j in range(n):
        deltas = [t - k * tau for (k, t) in assigned[j] if k in W]
        if deltas:
            s = 0.5 * (float(np.mean(deltas)) / tau + 1.0)
            s = float(np.clip(s, 0.50, 0.71))
            s = min(SWING_CLASSES, key=lambda c: abs(c - s))
        else:
            s = 0.50
        swing.append(round(s, 3))

    # per-step velocities aligned to q_j (same assignment); track-mean / 0.8 fallback.
    # When several onsets fold onto one step, the MAX (loudest hit) represents the step, since a
    # drum-machine step carries a single velocity and the loudest onset is the perceptually dominant one.
    step_velos = []
    for j in range(n):
        vj = velocities[j] if (velocities is not None and velocities[j] is not None) else None
        if vj is None:
            step_velos.append([0.8 if step_vectors[j][s] else 0.0 for s in range(num_steps)])
            continue
        tmean = float(np.mean(vj)) if len(vj) else 1.0
        row = [0.0] * num_steps
        for s in range(num_steps):
            if step_vectors[j][s]:
                acc = vel_acc[j][s]
                row[s] = round(float(max(acc)) if acc else tmean, 4)
        step_velos.append(row)

    return {"num_steps": num_steps, "steps_per_beat": steps_per_beat,
            "beat_type": beat_type, "swing": swing,
            "step_vectors": step_vectors, "step_velocities": step_velos,
            "instruments": C.INSTRUMENTS}
