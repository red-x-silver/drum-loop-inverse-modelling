"""Assembled shared-trunk ADT + tempo model (deployment build, no training framework).

The trained TCN tempo head was trained with the ADT trunk FROZEN, so the tempo checkpoint's trunk is
identical to the baseline_all ADT trunk. This wrapper rebuilds the two nn.Modules directly and loads
the bundled checkpoints:
  * the ADT trunk (ADTOFFrameRNN, 3-ch) <- the baseline_all ADT checkpoint  (authoritative onset model)
  * the tempo head (TempoHead, tcn_faithful) <- the tempo checkpoint's ``head.*`` weights
It then runs the trunk ONCE per loop and reads:
  * onsets  -> full trunk (all 3 BiGRUs) -> output_layer -> sigmoid  (3-ch activations)
  * tempo   -> shallow tap (post-CNN) -> TCN tempo head             (141-way -> BPM)
so onset transcription and tempo estimation are produced concurrently from one forward pass.
"""
from . import config as C
from .tempo_head import TempoHead, feat_dim, trunk_feature, TEMPO_LOW, NUM_TEMPO

import numpy as np
import torch

from adtof_pytorch import create_frame_rnn_model
from adtof_pytorch.audio import create_adtof_processor


def _strip(state_dict, prefix):
    sd = state_dict.get("state_dict", state_dict)
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class ADTTempoModel:
    def __init__(self, adt_ckpt=C.ADT_CKPT, tempo_ckpt=C.TEMPO_CKPT, device=None):
        self.device = device or C.device()
        self.proc = create_adtof_processor()
        n_bins = self.proc.get_n_bins()

        # ADT trunk (3-ch kick/snare/hh). AblationModule stored it under the "model." prefix.
        crnn = create_frame_rnn_model(n_bins=n_bins, output_classes=3)
        adt_sd = torch.load(adt_ckpt, map_location="cpu")
        missing, unexpected = crnn.load_state_dict(_strip(adt_sd, "model."), strict=False)
        if any("output_layer" in m for m in missing):
            raise RuntimeError(f"ADT trunk failed to load (missing {missing[:4]}...)")

        # Tempo head (tcn_faithful) at the shallow tap. TempoLit stored it under "head.".
        eff_d = 16 if C.TEMPO_MODULE.replace("-", "_") == "tcn_faithful" else C.TEMPO_PROJ_DIM
        head = TempoHead(feat_dim(crnn, C.TEMPO_POSITION), eff_d, NUM_TEMPO,
                         C.TEMPO_MODULE.replace("-", "_"))
        tempo_sd = torch.load(tempo_ckpt, map_location="cpu")
        head.load_state_dict(_strip(tempo_sd, "head."), strict=True)

        self.crnn = crnn.eval().to(self.device)
        self.head = head.eval().to(self.device)
        self.position = C.TEMPO_POSITION           # 'shallow' | 'mid' | 'deep'
        self.mid_gru = C.TEMPO_MID_GRU

    # ------------------------------------------------------------------ input
    def waveform_to_spec(self, wave, sr=C.SR):
        """wave: 1-D or [1,T] float tensor/np -> model input spec tensor [1, T, n_bins, 1]."""
        if torch.is_tensor(wave):
            wave = wave.detach().cpu().numpy()
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        spec = self.proc.process_waveform(wave, sr)                 # (T, n_bins, 1)
        return torch.from_numpy(spec).float().unsqueeze(0).to(self.device)

    # ---------------------------------------------------- shared-trunk forward
    @torch.no_grad()
    def forward_both(self, spec):
        """Single shared-trunk pass -> (onset_env [T,3] in [0,1], tempo_logits [141]).
        The tempo tap depends on the head's training position: 'shallow' = post-CNN (pre-GRU),
        'mid' = after `mid_gru` BiGRUs, 'deep' = after all BiGRUs. Onsets always use the full stack."""
        B, T = spec.shape[0], spec.shape[1]
        x = spec.permute(0, 3, 1, 2)                                # [B, C, T, F]
        for blk in self.crnn.cnn_blocks:
            x = blk(x)
        x = x.permute(0, 2, 3, 1).reshape(B, T, -1)                 # [B, T, feat]
        if getattr(self.crnn, "context_layer", None) is not None:
            x = self.crnn.context_layer(x)
        tap = x if self.position == "shallow" else None            # post-CNN tap
        for i, gru in enumerate(self.crnn.gru_layers):
            x, _ = gru(x)
            if self.position == "mid" and i == self.mid_gru - 1:
                tap = x
        if self.position == "deep":
            tap = x
        onset_env = torch.sigmoid(self.crnn.output_layer(x))[0]     # [T, 3]
        tempo_logits = self.head(tap)[0]                            # [141]
        return onset_env.cpu(), tempo_logits.cpu()

    @torch.no_grad()
    def analyze(self, wave, sr=C.SR):
        spec = self.waveform_to_spec(wave, sr)
        onset_env, tempo_logits = self.forward_both(spec)
        tempo_bpm = int(torch.argmax(tempo_logits).item() + TEMPO_LOW)
        return {"onset_env": onset_env.numpy(),                     # (T, 3)
                "tempo_logits": tempo_logits.numpy(),
                "tempo_bpm": tempo_bpm}
