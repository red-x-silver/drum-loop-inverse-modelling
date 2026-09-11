"""
differentiable_renderer.py -- a fully differentiable drum-loop renderer.

This reproduces the ESSENTIAL audio rendering of render_dataset.py
(TaoDrumMachine.sequencer.render_multi_tracks_{monophonic,polyphonic}: velocity-scaled one-shots
placed at onset positions, overlap-added, with small smoothing envelopes), but
DELIBERATELY omits every non-differentiable stage of that pipeline:

    * NO detect_and_normalize_clipping
    * NO lufs_normalize_loop  (and no +-0.999 clamp)
    * NO build_drum_fx / build_master_fx  (Pedalboard)
    * NO encode_loop_to_dac  (DAC codec)
    * never wrapped in torch.no_grad()

Everything is plain torch, so gradients flow to the per-onset velocities (and to the
one-shots). Onset positions are frozen integer sample indices; only the velocities
(gains) are optimised -- the analysis-by-synthesis setup.

Primary interface for the velocity experiment: `render_from_onsets(...)`, which works
entirely in the ABSOLUTE TEMPORAL DOMAIN (onset positions in audio samples). No step
vectors, tempo, or swing are involved. `render(...)` (grid/step-vector based, matching
the dataset renderer exactly) is kept only as a fidelity reference.

Options
-------
mode : "mono" | "poly"
    mono -> per-track monophonic: a retrigger chokes the previous hit on that track
            (FluidSynth-style release); one voice per track at a time.
    poly -> overlapping retriggers on a track simply sum (no choke).
smoothing_envelopes : bool
    True  -> apply the small smoothing envelopes exactly as the dataset renderer:
             linear attack de-click, sample-end anti-click fade, whole-loop end fade,
             and (mono) the smooth release choke.
    False -> raw velocity-scaled overlap-add (no attack/end/loop-end fades); mono still
             enforces one voice per track but as a hard truncation at the next onset.
"""
import math
import torch


class DifferentiableDrumRenderer:
    def __init__(self, sample_rate=44100, num_steps=16, steps_per_beat=4,
                 loop_length=176400, min_attack_ms=0.2, end_release_ms=2.0,
                 loop_end_fade_out_ms=4.0):
        self.sample_rate = sample_rate
        self.num_steps = num_steps
        self.steps_per_beat = steps_per_beat
        self.loop_length = loop_length
        self.min_attack_len = max(int(sample_rate * min_attack_ms / 1000.0), 1)
        self.end_release_len = max(int(sample_rate * end_release_ms / 1000.0), 1)
        self.loop_end_fade_out_len = max(int(sample_rate * loop_end_fade_out_ms / 1000.0), 1)

    # ================================================================
    # Shared differentiable core: place velocity-scaled one-shots at
    # (rows, poss) sample positions and overlap-add into [B, 1, T].
    # `vels` carries the gradient; `poss`/`rows` are frozen integer indices.
    # ================================================================
    def _render_events(self, one_shot_samples, rows, poss, vels, T, mode, smoothing):
        B, _, K = one_shot_samples.shape
        device, dtype = one_shot_samples.device, torch.float32

        one_shot = one_shot_samples.to(device=device, dtype=dtype)
        one_shot = one_shot - one_shot.mean(dim=-1, keepdim=True)          # per-one-shot DC removal
        one_shot = one_shot[:, 0, :]                                       # [B, K]

        if rows.numel() == 0:
            return torch.zeros(B, 1, T, device=device, dtype=dtype)

        rows = rows.to(device).long()
        poss = poss.to(device).long()
        vels = vels.to(device=device, dtype=dtype)

        # sort events row-major (row, then position) -> needed for the mono choke,
        # harmless for poly. Gather keeps velocities aligned and differentiable.
        order = torch.argsort(rows * (T + 1) + poss)
        rows, poss, vels = rows[order], poss[order], vels[order]

        o = torch.arange(K, device=device)
        of = o.to(dtype).unsqueeze(0)                                      # [1, K]
        seg = one_shot[rows]                                              # [E, K]

        CB_PER_AMP = 960.0 / 200.0                                         # -96 dB floor
        if smoothing:
            attack_env = torch.clamp(of / self.min_attack_len, 0.0, 1.0)  # [1, K]
            end_prog = torch.clamp((of - (K - self.end_release_len)) / self.end_release_len, 0.0, 1.0)
            end_env = torch.where(end_prog >= 1.0, torch.zeros_like(end_prog),
                                  torch.pow(10.0, -CB_PER_AMP * end_prog))  # [1, K]

        if mode == "mono":
            FLUID_BUFSIZE, MIN_NOTE_MS, KILL_TC = 64, 10.0, -7200.0
            sr = float(self.sample_rate)
            min_note_len = max(int(MIN_NOTE_MS * sr / 1000.0), 1)
            release_sec = 2.0 ** (min(max(KILL_TC, -7200.0), 8000.0) / 1200.0)
            count_buffers = 1 + int(release_sec * sr / FLUID_BUFSIZE)
            LARGE = K + count_buffers * FLUID_BUFSIZE + 1
            nxt_pos = torch.full_like(poss, LARGE); nxt_pos[:-1] = poss[1:]
            nxt_row = torch.full_like(rows, -1);    nxt_row[:-1] = rows[1:]
            killed = nxt_row == rows
            kill_d = torch.where(killed, nxt_pos - poss, torch.full_like(poss, LARGE)).to(dtype)
            rs = torch.clamp_min(kill_d, float(min_note_len)).unsqueeze(1)  # [E, 1]
            if smoothing:
                in_rel = of >= rs
                relblock = torch.floor((of - rs) / FLUID_BUFSIZE)
                volenv = torch.where(in_rel, torch.clamp(1.0 - relblock / count_buffers, 0.0, 1.0),
                                     torch.ones_like(relblock))
                rel_env = torch.pow(10.0, -CB_PER_AMP * (1.0 - volenv))
                rel_env = torch.where(in_rel & (relblock >= count_buffers),
                                      torch.zeros_like(rel_env), rel_env)
                env = attack_env * rel_env * end_env                       # [E, K]
            else:
                env = (of < rs).to(dtype)                                  # hard choke at next onset
        else:  # poly
            env = attack_env * end_env if smoothing else torch.ones(1, K, device=device, dtype=dtype)

        seg = seg * vels.unsqueeze(1) * env                                # [E, K]

        out_flat = torch.zeros(B * T, device=device, dtype=dtype)
        tpos = poss.unsqueeze(1) + o.unsqueeze(0)                          # [E, K] absolute time
        inb = (tpos >= 0) & (tpos < T)
        gidx = rows.unsqueeze(1) * T + tpos
        out_flat = out_flat.index_add(0, gidx[inb], seg[inb])
        tracks = out_flat.view(B, T).unsqueeze(1)                          # [B, 1, T]

        if smoothing:
            L = min(self.loop_end_fade_out_len, T)
            if L > 0:
                end_fade = torch.ones(T, device=device, dtype=dtype)
                end_fade[-L:] = torch.linspace(1.0, 0.0, L, device=device, dtype=dtype)
                tracks = tracks * end_fade.view(1, 1, -1)
        return tracks

    # ================================================================
    # PRIMARY interface: absolute onset positions (in samples) + per-onset velocities.
    # ================================================================
    def render_from_onsets(self, one_shot_samples, onsets, velocities, loop_length=None,
                           mode="poly", smoothing_envelopes=True,
                           return_tracks=False, return_mix=True):
        """Render entirely in the absolute sample domain -- no step vectors / tempo / swing.

        one_shot_samples : [B, 1, K]           one one-shot per track.
        onsets           : list of length B; onsets[t] is a 1-D tensor/array of INTEGER
                           sample positions for track t.
        velocities       : list of length B; velocities[t] is a 1-D float tensor of the
                           SAME length as onsets[t] (the trainable per-onset gains).
        loop_length      : T in samples (defaults to self.loop_length).

        Returns the summed mono mix [1, T] (default) and/or per-track tracks [B, 1, T].
        Gradients flow to `velocities`.
        """
        assert mode in ("mono", "poly")
        T = int(loop_length) if loop_length is not None else self.loop_length
        B = one_shot_samples.shape[0]
        device = one_shot_samples.device
        assert len(onsets) == B and len(velocities) == B, "onsets/velocities must be per-track lists of length B"

        rows, poss, vels = [], [], []
        for t in range(B):
            p = torch.as_tensor(onsets[t], device=device).reshape(-1)
            v = velocities[t].reshape(-1) if torch.is_tensor(velocities[t]) \
                else torch.as_tensor(velocities[t], device=device).reshape(-1)
            assert p.numel() == v.numel(), f"track {t}: {p.numel()} onsets vs {v.numel()} velocities"
            rows.append(torch.full((p.numel(),), t, device=device, dtype=torch.long))
            poss.append(p); vels.append(v)
        rows = torch.cat(rows) if rows else torch.empty(0, dtype=torch.long, device=device)
        poss = torch.cat(poss) if poss else torch.empty(0, dtype=torch.long, device=device)
        vels = torch.cat(vels) if vels else torch.empty(0, device=device)

        tracks = self._render_events(one_shot_samples, rows, poss, vels, T, mode, smoothing_envelopes)
        mix = tracks.sum(dim=0)                                            # [1, T]
        if return_tracks and return_mix:
            return mix, tracks
        return tracks if return_tracks else mix

    # ================================================================
    # Grid / step-vector interface (fidelity reference to the dataset renderer).
    # ================================================================
    def calculate_samples_per_step(self, tempo):
        samples_per_beat = math.floor(self.sample_rate * 60.0 / float(tempo))
        return int(math.floor(samples_per_beat / self.steps_per_beat))

    def _grid_positions(self, velo_vectors, tempo, swing_indices_multi, swing_amounts,
                        nudge_offsets):
        device = velo_vectors.device
        num_tracks, num_steps = velo_vectors.shape
        sps = self.calculate_samples_per_step(tempo)
        this_loop = sps * num_steps
        step_idx = torch.arange(num_steps, device=device).unsqueeze(0).expand(num_tracks, -1)
        positions = step_idx * sps
        if nudge_offsets is not None:
            positions = positions + (nudge_offsets.to(device) * (self.sample_rate / 1000.0)).round().long()
        elif swing_indices_multi is not None:
            if swing_amounts is None:
                swing_amounts = torch.full((num_tracks,), 0.5, device=device)
            positions = positions.clone()
            for t, idx in enumerate(swing_indices_multi):
                idx = torch.as_tensor(idx, device=device).long()
                if idx.numel() == 0:
                    continue
                off = int(round((2.0 * float(swing_amounts[t]) - 1.0) * sps))
                positions[t, idx[idx >= 0]] += off
        positions = torch.clamp(positions, 0, this_loop - 1).round().long()
        return positions, this_loop

    def render(self, one_shot_samples, velo_vectors, tempo, swing_indices_multi=None,
               swing_amounts=None, mode="poly", smoothing_envelopes=True, tile_mode="tile",
               nudge_offsets=None, return_tracks=False, return_mix=True):
        """Grid renderer (step vectors + tempo). Kept as a fidelity reference; the velocity
        experiment should use render_from_onsets instead."""
        assert mode in ("mono", "poly")
        device = velo_vectors.device
        B, num_steps = velo_vectors.shape
        positions, this_loop = self._grid_positions(velo_vectors, tempo, swing_indices_multi,
                                                    swing_amounts, nudge_offsets)
        # scatter velocities into one bar (differentiable), then tile/pad to loop_length
        act_one = torch.zeros(B, this_loop, device=device)
        flat_trk = torch.arange(B, device=device).repeat_interleave(num_steps)
        act_one = act_one.index_put((flat_trk, positions.reshape(-1)),
                                    velo_vectors.reshape(-1).float(), accumulate=True)
        if tile_mode == "pad":
            act = torch.zeros(B, self.loop_length, device=device)
            cl = min(this_loop, self.loop_length)
            act = act.index_copy(1, torch.arange(cl, device=device), act_one[:, :cl])
        else:
            act = act_one.repeat(1, math.ceil(self.loop_length / this_loop))[:, :self.loop_length]
        nz = (act > 0).nonzero(as_tuple=False)
        rows, poss = nz[:, 0], nz[:, 1]
        vels = act[rows, poss]
        tracks = self._render_events(one_shot_samples, rows, poss, vels, self.loop_length,
                                     mode, smoothing_envelopes)
        mix = tracks.sum(dim=0)
        if return_tracks and return_mix:
            return mix, tracks
        return tracks if return_tracks else mix


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    sr = 44100
    r = DifferentiableDrumRenderer(sample_rate=sr, loop_length=4 * sr)
    B, K, T = 3, int(0.3 * sr), 4 * sr
    one_shots = torch.randn(B, 1, K)

    # --- absolute-onset interface (the experiment's interface) ---
    onsets = [torch.tensor([0, 22050, 44100, 132300]),           # kick
              torch.tensor([22050, 66150, 110250, 154350]),      # snare
              torch.arange(0, T, 11025)]                         # hats (16th-ish)
    velocities = [torch.rand(len(o)).clamp(0.1, 1.0).requires_grad_(True) for o in onsets]
    print(f"{'mode':<6}{'smooth':<8}{'mix':<16}{'grad->velo?':<12}{'nonzero'}")
    for mode in ("poly", "mono"):
        for sm in (True, False):
            mix = r.render_from_onsets(one_shots, onsets, velocities, loop_length=T,
                                       mode=mode, smoothing_envelopes=sm)
            loss = mix.pow(2).mean(); loss.backward()
            gs = [v.grad for v in velocities]
            ok = all(g is not None and torch.isfinite(g).all() for g in gs)
            nz = all(g.abs().sum() > 0 for g in gs)
            print(f"{mode:<6}{str(sm):<8}{str(tuple(mix.shape)):<16}{str(ok):<12}{nz}")
            for v in velocities:
                v.grad = None

    # --- fidelity: render_from_onsets == grid render, single bar (tile_mode="pad") ---
    velo = torch.rand(B, 16); velo[velo < 0.4] = 0.0
    pos, _ = r._grid_positions(velo, 120, None, None, None)
    on = [pos[t][velo[t] > 0] for t in range(B)]
    ve = [velo[t][velo[t] > 0] for t in range(B)]
    a = r.render_from_onsets(one_shots, on, ve, loop_length=r.loop_length, mode="poly", smoothing_envelopes=True)
    b = r.render(one_shots, velo, 120, mode="poly", smoothing_envelopes=True, tile_mode="pad")
    print("from_onsets vs grid render (pad)  max|delta| =", (a - b).abs().max().item())
    print("OK")