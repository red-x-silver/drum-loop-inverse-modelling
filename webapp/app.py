"""
FINAL — Drum-loop inverse-modelling web app (Flask backend).

Wraps the full estimation system into one endpoint: upload a drum loop, the server pads/truncates
it to 4 s, runs the shared-trunk ADT + TCN-tempo model, peak-picks onsets, extracts one-shots with
the Stable-Audio-3 small-r4 LoRA model (cached), estimates per-onset velocities by analysis-by-
synthesis, and returns absolute-time parameter estimates plus audio for playback.
"""
import os
import sys
import shutil
import uuid
import traceback

import torchaudio
from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename

FINAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FINAL_DIR not in sys.path:
    sys.path.insert(0, FINAL_DIR)
from pipeline import config as C

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

UPLOADS = os.path.join(app.root_path, "static", "uploads")
RESULTS = os.path.join(app.root_path, "static", "results")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        f = request.files.get("file")
        if f is None or f.filename == "":
            return jsonify({"error": "no file uploaded"}), 400
        uid = uuid.uuid4().hex[:12]
        up = os.path.join(UPLOADS, f"{uid}_{secure_filename(f.filename)}")
        f.save(up)

        from model_manager import get_pipeline
        result, target, mix = get_pipeline().analyze(up)

        key = result["cache_key"]
        outdir = os.path.join(RESULTS, key)
        os.makedirs(outdir, exist_ok=True)
        torchaudio.save(os.path.join(outdir, "loop.wav"), target, C.SR)
        if mix is not None:
            torchaudio.save(os.path.join(outdir, "recon.wav"), mix, C.SR)
        for inst in C.INSTRUMENTS:
            src = os.path.join(result["oneshot_dir"], f"one_shot_{C.INSTRU_DIR[inst]}.wav")
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(outdir, f"{inst}.wav"))

        base = f"/static/results/{key}"
        result["urls"] = {
            "loop": f"{base}/loop.wav",
            "recon": f"{base}/recon.wav" if mix is not None else None,
            "oneshots": {inst: f"{base}/{inst}.wav" for inst in C.INSTRUMENTS},
        }
        result.pop("reconstruction", None)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # threaded so the browser can fetch static audio while a long analysis is in flight
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
