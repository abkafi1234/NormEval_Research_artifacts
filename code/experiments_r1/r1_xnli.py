"""Revision R1: the XNLI experiment (runs/exp_xnli.py, unchanged) with the Stanza lemmatizer as the
morphological step. Only the normalization function is replaced: full = XNLI orthographic stage applied
to the lemmatizer's output. The classifier for unnormalized MultiNLI is the one already fitted for the
paper (runs/cache/xnli/clf_orig.joblib); the classifier for normalized MultiNLI is fitted here.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
import exp_common as C  # noqa: E402
import exp_xnli as X  # noqa: E402
import r1_norm as N  # noqa: E402

METHOD = "stanza"
_maps = {}


def lemmas(lang):
    if lang not in _maps:
        _maps[lang] = N.lookup(METHOD, f"xnli:{lang}")
        if lang == "en":
            _maps[lang].update(N.lookup(METHOD, "mnli"))
    return _maps[lang]


def xnli_full(text, lang):
    return C.xnli_ortho(lemmas(lang)[text], lang)


def classifiers():
    import joblib
    from datasets import load_dataset
    cdir = os.path.join(N.R1, "cache", "xnli_" + METHOD)
    os.makedirs(cdir, exist_ok=True)
    orig = joblib.load(os.path.join(C.CACHE, "xnli", "clf_orig.joblib"))
    info = json.load(open(os.path.join(C.CACHE, "xnli", "clf_info.json")))
    path = os.path.join(cdir, "clf_norm.joblib")
    if os.path.exists(path):
        return {"orig": orig, "norm": joblib.load(path)}, json.load(open(os.path.join(cdir, "clf_info.json")))
    mnli = load_dataset("nyu-mll/multi_nli", split="train")
    prem, hyp, y = list(mnli["premise"]), list(mnli["hypothesis"]), np.array(mnli["label"])
    enc = C.encoder(C.MINILM)
    t0 = time.time()
    u = C.encode(enc, [xnli_full(t, "en") for t in prem], batch_size=256)
    v = C.encode(enc, [xnli_full(t, "en") for t in hyp], batch_size=256)
    Xn = X.infersent(u, v)
    del u, v
    gc.collect()
    C.log(f"norm features {Xn.shape} in {time.time() - t0:.0f}s; fitting LR")
    clf = X.make_lr().fit(Xn, y)
    info = dict(info)
    info["norm"] = {"n_iter": int(np.max(clf.n_iter_)), "converged": bool(np.max(clf.n_iter_) < 1000),
                    "classes": clf.classes_.tolist(), "method": METHOD}
    joblib.dump(clf, path)
    json.dump(info, open(os.path.join(cdir, "clf_info.json"), "w"), indent=1)
    return {"orig": orig, "norm": clf}, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(X.LANGS))
    a = ap.parse_args()
    C.xnli_full = xnli_full
    C.OUT = os.path.join(C.OUT, "xnli_" + METHOD)        # outputs/xnli_stanza/xnli/<lang>.json
    clfs, info = classifiers()
    for lang in a.langs.split(","):
        if C.done("xnli", f"{lang}.json"):
            C.log(f"{lang}: cached"); continue
        X.run_language(lang, clfs, info, None)
