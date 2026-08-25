"""Tempo head + trunk-feature taps, extracted for inference from the research repo's train_tempo.py.

This is the deployment subset: the nn.Module definitions needed to rebuild and run the trained
tempo head (the training LightningModule, datasets, wandb logging, and CLI have been dropped). The
class bodies are copied verbatim so the trained checkpoint's ``head.*`` weights load exactly.

Only the ``tcn_faithful`` variant at the ``shallow`` tap is used by the shipped model; the other
variants/classes are kept intact so the head-construction signature matches the checkpoint hparams.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPO_LOW, TEMPO_HIGH = 60, 200
NUM_TEMPO = TEMPO_HIGH - TEMPO_LOW + 1   # 141
# TCN depth: 7 dilated blocks (dilations 1..64, kernel 5) -> receptive field 1+4*(2^7-1)=509 frames
# = 5.09 s @100fps, the minimal power-of-2 stack that still covers the full 4 s / 401-frame loop.
TCN_BLOCKS = 7


def trunk_feature(crnn, x, position, mid_gru=1):
    """Run the trunk up to `position`; return per-frame features [B, T, D]. x: [B, T, F, C=1].
    position='spec' bypasses the ADT net entirely -> the raw pre-processed spectrogram frames."""
    B, T = x.shape[0], x.shape[1]
    if position == "spec":
        return x.reshape(B, T, -1)                     # [B, T, F*C] raw spectrogram, no ADT processing
    h = x.permute(0, 3, 1, 2)                          # [B, C, T, F]
    for blk in crnn.cnn_blocks:
        h = blk(h)
    h = h.permute(0, 2, 3, 1).reshape(B, T, -1)        # [B, T, cnn_output_features]
    if getattr(crnn, "context_layer", None) is not None:
        h = crnn.context_layer(h)
    if position == "shallow":
        return h
    n = len(crnn.gru_layers)
    k = n if position == "deep" else min(mid_gru, n)
    for i in range(k):
        h, _ = crnn.gru_layers[i](h)                   # [B, T, final_gru_size]
    return h


def feat_dim(crnn, position):
    if position == "spec":
        return crnn.n_bins * getattr(crnn, "n_channels", 1)     # raw spectrogram F*C
    if position == "shallow":
        return crnn.cnn_output_features * getattr(crnn, "context_multiplier", 1)
    return crnn.final_gru_size


class MultiFilter(nn.Module):
    """tempo-cnn/FCN-style periodicity block: parallel dense time-convs of fixed LENGTHS (32..256 frames
    @100fps, spanning the beat-period range) on a channel bottleneck, concatenated -> 1x1 back to width d."""
    def __init__(self, d, lengths=(32, 64, 96, 128, 192, 256), bottleneck=64, per_branch=24):
        super().__init__()
        self.bottleneck = nn.Conv1d(d, bottleneck, 1)
        self.branches = nn.ModuleList([nn.Conv1d(bottleneck, per_branch, k, padding=k // 2) for k in lengths])
        oc = per_branch * len(lengths)
        self.bn = nn.BatchNorm1d(oc)
        self.reduce = nn.Conv1d(oc, d, 1)

    def forward(self, h):                               # h: [B, d, T]
        T = h.shape[-1]
        b = torch.relu(self.bottleneck(h))
        outs = [br(b)[..., :T] for br in self.branches]  # crop even-kernel +1 -> T
        c = torch.relu(self.bn(torch.cat(outs, dim=1)))
        return self.reduce(c)                            # [B, d, T]


class SpatialDropout1d(nn.Module):
    """Channel-wise ('spatial') dropout for [B, C, T] — drops whole feature maps (Tompson et al.),
    which is what Bock et al. (2019) use inside the TCN. Version-safe (no nn.Dropout1d dependency)."""
    def __init__(self, p):
        super().__init__(); self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        return F.dropout2d(x.unsqueeze(-1), self.p, self.training).squeeze(-1)


class TCNResBlock(nn.Module):
    """madmom-style TCN block (Bock et al. 2019, Fig 2): spatial-dropout -> dilated conv -> ELU ->
    spatial-dropout -> {1x1 residual (added to input), 1x1 skip}. kernel=5 odd -> padding preserves T."""
    def __init__(self, c, k=5, dilation=1, dropout=0.1):
        super().__init__()
        self.drop1 = SpatialDropout1d(dropout)
        self.dilated = nn.Conv1d(c, c, k, padding=(k // 2) * dilation, dilation=dilation)
        self.act = nn.ELU()
        self.drop2 = SpatialDropout1d(dropout)
        self.skip = nn.Conv1d(c, c, 1)
        self.res = nn.Conv1d(c, c, 1)

    def forward(self, x):                               # x: [B, c, T]
        y = self.act(self.dilated(self.drop1(x)))
        y = self.drop2(y)
        return x + self.res(y), self.skip(y)            # (residual out, skip out)


class TCNStack(nn.Module):
    """madmom TCN aggregator: n_blocks dilated-residual blocks with dilation 1,2,4,...,2^(n-1); skip
    outputs summed (WaveNet-style) then ELU. c=16 matches madmom's TCN width. n_blocks=TCN_BLOCKS(7) ->
    dilations 1..64 -> receptive field 509 frames (5.09 s), covering the 401-frame / 4 s loop."""
    def __init__(self, c, n_blocks=TCN_BLOCKS, k=5, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TCNResBlock(c, k, dilation=2 ** i, dropout=dropout)
                                     for i in range(n_blocks)])
        self.act = nn.ELU()

    def forward(self, x):                               # x: [B, c, T]
        skips = 0
        for b in self.blocks:
            x, skip = b(x)
            skips = skips + skip
        return self.act(skips)                          # [B, c, T]


def _pool_out(n, k=3, s=3):
    return (n - k) // s + 1


class TCNFrontEnd(nn.Module):
    """Bock et al. (2019) spectrogram front-end: conv(1->16,3x3)->ELU->maxpool(freq/3)->conv(16->16,3x3)
    ->ELU->maxpool(freq/3)->conv(16->16, 1xF') collapsing the remaining freq bins -> [B, T, 16]. dropout 0.1."""
    def __init__(self, n_bins, c=16, dropout=0.1):
        super().__init__()
        self.c1 = nn.Conv2d(1, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.pool = nn.MaxPool2d((1, 3))
        f_rem = _pool_out(_pool_out(n_bins))            # freq bins left after the two (1,3) pools
        self.c3 = nn.Conv2d(c, c, (1, f_rem))           # collapse freq -> 1
        self.act = nn.ELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):                               # x: [B, T, F, 1]
        x = x.permute(0, 3, 1, 2)                       # [B, 1, T, F]
        x = self.drop(self.pool(self.act(self.c1(x))))
        x = self.drop(self.pool(self.act(self.c2(x))))
        x = self.act(self.c3(x))                        # [B, c, T, 1]
        return x.squeeze(-1).transpose(1, 2)            # [B, T, c]


class ScratchTCN(nn.Module):
    """Full madmom-style TCN trained FROM SCRATCH: spectrogram conv front-end -> TCN stack -> summed
    skips -> global avg-pool -> Dropout(0.5) -> linear softmax. No ADT backbone."""
    def __init__(self, n_bins, n_cls, n_blocks=TCN_BLOCKS, c=16):
        super().__init__()
        self.frontend = TCNFrontEnd(n_bins, c)
        self.tcn = TCNStack(c, n_blocks=n_blocks)
        self.mlp = nn.Sequential(nn.Dropout(0.5), nn.Linear(c, n_cls))

    def forward(self, x):                               # x: [B, T, F, 1]
        h = self.frontend(x)                            # [B, T, c]
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2) # [B, T, c]
        return self.mlp(h.mean(dim=1))                  # [B, n_cls]


class TempoHead(nn.Module):
    """Shared: 1x1 proj -> [aggregator] -> global avg-pool over time -> classifier -> 141.
    variant in {small, large, tcn_faithful}:
      small        : no aggregator (plain pool).
      large        : tempo-cnn/FCN multi-filter block (parallel dense kernels) at width d.
      tcn_faithful : Bock et al. (2019) TCN block as-published EXCEPT the front-end (ADT tap, not their
                     spectrogram conv), the tempo range/target/loss/training (ours), and the depth
                     (TCN_BLOCKS=7 / dilations 1..64 sized to cover the 4 s loop, not their 11 / 1..1024)."""
    def __init__(self, d_in, d, n_cls, variant):
        super().__init__()
        if variant not in ("small", "large", "tcn_faithful"):
            raise ValueError(f"unknown module variant: {variant!r} (expected small/large/tcn_faithful)")
        self.proj = nn.Linear(d_in, d)
        if variant == "large":
            self.agg = MultiFilter(d)
        elif variant == "tcn_faithful":
            self.agg = TCNStack(d, n_blocks=TCN_BLOCKS)  # dilations 1..64, RF 509 frames (covers 4 s)
        else:
            self.agg = None
        if variant == "tcn_faithful":
            self.mlp = nn.Sequential(nn.Dropout(0.5), nn.Linear(d, n_cls))   # single softmax layer (paper head)
        else:
            self.mlp = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_cls))

    def forward(self, h):                               # h: [B, T, d_in]
        h = self.proj(h)                                # [B, T, d]
        if self.agg is not None:
            h = self.agg(h.transpose(1, 2)).transpose(1, 2)   # [B, T, d]
        z = h.mean(dim=1)                               # [B, d]  global average pool over time
        return self.mlp(z)                              # [B, n_cls]
