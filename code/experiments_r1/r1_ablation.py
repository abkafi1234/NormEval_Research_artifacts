"""Revision R1: BTSD ablation and Bangla sentiment robustness for recent normalizers.

Runs runs/exp_ablation.py unchanged (intrinsic metrics through the normeval package, 10-fold
Traditional DSP with four classifiers and three seeds, McNemar, package cross-check) on new units.
The only change is where the normalized text comes from: the cache written by r1_norm.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
import exp_common as C  # noqa: E402
import exp_ablation as A  # noqa: E402
import r1_norm as N  # noqa: E402

A.UNITS = [
    ("btsd_en_stanza", "btsd", "en", "stanza"),
    ("btsd_en_llm", "btsd", "en", "llm"),
    ("btsd_bn_bkit", "btsd", "bn", "bkit"),
    ("btsd_bn_llm", "btsd", "bn", "llm"),
    ("sentiment_bkit", "sentiment", "bn", "bkit"),
    ("sentiment_llm", "sentiment", "bn", "llm"),
]
_state = {"corpus": None}
_orig_phase1 = A.phase1


def english(texts, method):
    m = N.lookup(method, "btsd_en")
    return [m[str(t).lower()] for t in texts]            # lowercased input, as the other English units


def bangla(texts, method):
    btsd = N.lookup(method, "btsd_bn") if os.path.exists(N.cache_path(method, "btsd_bn")) else {}
    sent = N.lookup(method, "sentiment") if os.path.exists(N.cache_path(method, "sentiment")) else {}
    out = []
    for t in texts:
        t = str(t)
        out.append(btsd[t] if t in btsd else sent[t])
    return out


C.normalize_english_batch = english
C.normalize_bangla_batch = bangla

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--methods", default="stanza,bkit,llm")
    a = ap.parse_args()
    need = {"btsd": {"en": ["btsd_en"], "bn": ["btsd_bn"]}, "sentiment": {"bn": ["sentiment"]}}
    A.UNITS = [u for u in A.UNITS if u[3] in a.methods.split(",")
               and all(os.path.exists(N.cache_path(u[3], n)) for n in need[u[1]][u[2]])]
    C.log("units: " + ", ".join(u[0] for u in A.UNITS))
    t0 = time.time()
    C.log(f"R1 ablation start | seeds={C.SEEDS} | out={C.OUT}")
    A.phase1(None)
    A.phase2(a.jobs)
    A.phase3()
    C.log(f"R1 ablation done in {(time.time() - t0) / 60:.1f} min")
