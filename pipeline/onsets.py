"""Peak-picking of the ADT onset activations into per-instrument onset positions.

Uses the checkpoint's sibling peak_picking.json params (order kick/snare/hihats), the same
NotePeakPickingProcessor and no-quantization frame->sample conversion as extract_onsets_adt.py.
"""
import json

from . import config as C
from adtof_pytorch.post_processing import NotePeakPickingProcessor


def build_pickers(pp_json=C.PEAK_PICKING_JSON):
    pp = json.load(open(pp_json))["pp_params"]              # [kick, snare, hihats]
    return {"kick": NotePeakPickingProcessor(**pp[0]),
            "snare": NotePeakPickingProcessor(**pp[1]),
            "hh": NotePeakPickingProcessor(**pp[2])}


def pick_onsets(onset_env, pickers, sr=C.SR):
    """onset_env: (T, 3) activations. -> dict per instrument with seconds + sample positions."""
    out = {}
    for inst, ch in C.ONSET_CHANNELS.items():
        secs = sorted(t for t, _ in pickers[inst].process(onset_env[:, ch]))
        out[inst] = {"seconds": [float(s) for s in secs],
                     "samples": [int(round(s * sr)) for s in secs]}
    return out
