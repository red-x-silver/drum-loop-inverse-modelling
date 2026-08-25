"""Lazy singleton for the warm inference pipeline (ADT+tempo model, pickers, renderer)."""
import os
import sys

FINAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FINAL_DIR not in sys.path:
    sys.path.insert(0, FINAL_DIR)

from pipeline.infer import Pipeline

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline
