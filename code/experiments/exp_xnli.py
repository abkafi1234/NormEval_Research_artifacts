"""Experiment A: XNLI in six languages. See EXPERIMENTAL_PLAN.md §2.

A1 intrinsic metrics via normeval (concatenated premise+hypothesis)
A2 vocabulary decomposition V_raw / V_ortho / V_full
A3 zero-shot MPD (LR trained once on raw and once on normalized MultiNLI InferSent features)
A4 in-language Traditional DSP (MiniLM embeddings + LR), 10-fold, seeds 42/7/2024, McNemar,
   plus a normeval-package cross-check at seed 42.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time

import numpy as np

import exp_common as C

LANGS = ["ar", "de", "en", "es", "fr", "ru"]


def make_lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)


def l2(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def infersent(u, v):
    u, v = l2(u), l2(v)
    return np.concatenate([u, v, np.abs(u - v), u * v], axis=1).astype(np.float32)


def zero_shot_classifiers(limit):
    import joblib
    from datasets import load_dataset
    cdir = os.path.join(C.CACHE, "xnli" if not limit else f"xnli_limit{limit}")
    os.makedirs(cdir, exist_ok=True)
    paths = {c: os.path.join(cdir, f"clf_{c}.joblib") for c in ("orig", "norm")}
    info_path = os.path.join(cdir, "clf_info.json")
    if all(os.path.exists(p) for p in paths.values()):
        C.log("zero-shot classifiers cached")
        return {c: joblib.load(p) for c, p in paths.items()}, json.load(open(info_path))
    mnli = load_dataset("nyu-mll/multi_nli", split="train")
    prem, hyp, y = list(mnli["premise"]), list(mnli["hypothesis"]), np.array(mnli["label"])
    if limit:
        prem, hyp, y = prem[:limit], hyp[:limit], y[:limit]
    C.log(f"MultiNLI n={len(y)}; labels {dict(zip(*np.unique(y, return_counts=True)))}")
    enc = C.encoder(C.MINILM)
    info = {"n_train": len(y), "encoder": C.MINILM}
    clfs = {}
    for cond in ("orig", "norm"):
        t0 = time.time()
        if cond == "norm":
            prem_c = [C.xnli_full(t, "en") for t in prem]
            hyp_c = [C.xnli_full(t, "en") for t in hyp]
        else:
            prem_c, hyp_c = prem, hyp
        u = C.encode(enc, prem_c, batch_size=256)
        v = C.encode(enc, hyp_c, batch_size=256)
        X = infersent(u, v)
        del u, v, prem_c, hyp_c
        gc.collect()
        C.log(f"{cond}: features {X.shape} in {time.time() - t0:.0f}s; fitting LR")
        t1 = time.time()
        clf = make_lr().fit(X, y)
        info[cond] = {"n_iter": int(np.max(clf.n_iter_)), "converged": bool(np.max(clf.n_iter_) < 1000),
                      "classes": clf.classes_.tolist(), "fit_seconds": round(time.time() - t1, 1)}
        joblib.dump(clf, paths[cond])
        clfs[cond] = clf
        del X
        gc.collect()
        C.log(f"{cond}: LR fitted in {time.time() - t1:.0f}s (n_iter={info[cond]['n_iter']})")
    json.dump(info, open(info_path, "w"), indent=1)
    return clfs, info


def vocab_size(texts):
    return len({tok for t in texts for tok in t.split()})


def run_language(lang, clfs, clf_info, limit):
    from datasets import load_dataset
    from normeval import NormalizationEvaluator
    from sklearn.metrics import f1_score
    t0 = time.time()
    ds = load_dataset("facebook/xnli", lang, split="validation")
    p_raw, h_raw, y = list(ds["premise"]), list(ds["hypothesis"]), np.array(ds["label"])
    if limit:
        idx = np.linspace(0, len(y) - 1, limit).astype(int)
        p_raw, h_raw, y = [p_raw[i] for i in idx], [h_raw[i] for i in idx], y[idx]
    p_ort, h_ort = [C.xnli_ortho(t, lang) for t in p_raw], [C.xnli_ortho(t, lang) for t in h_raw]
    p_ful, h_ful = [C.xnli_full(t, lang) for t in p_raw], [C.xnli_full(t, lang) for t in h_raw]
    cat_raw = [f"{p} {h}" for p, h in zip(p_raw, h_raw)]
    cat_ort = [f"{p} {h}" for p, h in zip(p_ort, h_ort)]
    cat_ful = [f"{p} {h}" for p, h in zip(p_ful, h_ful)]
    enc = C.encoder(C.MINILM)
    res = {"lang": lang, "n": int(len(y)), "label_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}}

    # A1 intrinsic (normeval package, default whitespace tokenizer)
    ev = NormalizationEvaluator(cat_raw, cat_ful, embedding_model=enc)
    cr = ev.calculate_cr()
    irs = ev.calculate_irs(batch_size=256)
    res["intrinsic"] = {"CR": cr, "VRG": 0.0 if cr < 1 else 1 - 1 / cr, "IRS": irs, "AES_beta1": ev.calculate_aes(cr, irs),
                        "ANLD": ev.calculate_anld(), "KL": ev.calculate_kl_divergence()}

    # A2 vocabulary decomposition
    vr, vo, vf = vocab_size(cat_raw), vocab_size(cat_ort), vocab_size(cat_ful)
    cr_t, cr_o, cr_m = vr / vf, vr / vo, vo / vf
    res["cr_decomposition"] = {"V_raw": vr, "V_ortho": vo, "V_full": vf, "CR_total": cr_t, "CR_ortho": cr_o, "CR_morph": cr_m,
                               "morph_share": math.log(cr_m) / math.log(cr_t) if cr_t > 1 else float("nan")}

    # A3 zero-shot MPD
    X_o = infersent(C.encode(enc, p_raw, 256), C.encode(enc, h_raw, 256))
    X_n = infersent(C.encode(enc, p_ful, 256), C.encode(enc, h_ful, 256))
    pred_o, pred_n = clfs["orig"].predict(X_o), clfs["norm"].predict(X_n)
    f1o, f1n = float(f1_score(y, pred_o, average="macro")), float(f1_score(y, pred_n, average="macro"))
    res["zero_shot"] = {"F1_orig": f1o, "F1_norm": f1n, "MPD": f1n - f1o, "clf_info": clf_info,
                        "mcnemar": C.mcnemar_paired(pred_o == y, pred_n == y),
                        "pred_orig": pred_o.tolist(), "pred_norm": pred_n.tolist()}
    del X_o, X_n

    # A4 in-language Traditional DSP on MiniLM embeddings of the concatenated texts
    E_o, E_n = C.encode(enc, cat_raw, 32), C.encode(enc, cat_ful, 32)   # batch 32 = package default
    res["in_language_dsp"] = {}
    for seed in C.SEEDS:
        det = C.traditional_dsp_detailed(cat_raw, cat_ful, y, make_lr(), C.N_SPLITS, seed, X_orig=E_o, X_norm=E_n)
        res["in_language_dsp"][str(seed)] = C.dsp_unit_result(det, y)
        st = res["in_language_dsp"][str(seed)]["stats"]
        C.log(f"{lang} seed {seed}: delta={st['delta_mean']:+.4f} p={st['wilcoxon_p']:.4f} "
              f"McNemar p={res['in_language_dsp'][str(seed)]['mcnemar']['p_value']:.3g}")
    # normeval cross-check (the package re-encodes internally with the same model)
    ev.labels, ev.classifiers = y.tolist(), [make_lr()]
    pkg = ev.calculate_traditional_dsp(n_splits=C.N_SPLITS, random_state=42, average_method="macro")
    res["package_check_seed42"] = {"package": pkg["LogisticRegression"],
                                   "replica_delta": res["in_language_dsp"].get("42", {}).get("stats", {}).get("delta_mean"),
                                   "replica_p": res["in_language_dsp"].get("42", {}).get("stats", {}).get("wilcoxon_p")}
    res["texts_sample"] = {"raw": cat_raw[:3], "ortho": cat_ort[:3], "full": cat_ful[:3]}
    res["seconds"] = round(time.time() - t0, 1)
    ic, zs = res["intrinsic"], res["zero_shot"]
    C.log(f"{lang}: CR={ic['CR']:.4f} IRS={ic['IRS']:.4f} ANLD={ic['ANLD']:.4f} KL={ic['KL']:.3f} | "
          f"F1 {zs['F1_orig']:.4f}->{zs['F1_norm']:.4f} MPD={zs['MPD']:+.4f} | {res['seconds']}s")
    C.save(res, "xnli", f"{lang}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-train", type=int, default=None, help="MultiNLI subsample (smoke test)")
    ap.add_argument("--limit", type=int, default=None, help="XNLI subsample per language (smoke test)")
    ap.add_argument("--langs", default=",".join(LANGS))
    a = ap.parse_args()
    t0 = time.time()
    C.log(f"xnli start | seeds={C.SEEDS} | out={C.OUT} | device={C.device()}")
    clfs, info = zero_shot_classifiers(a.limit_train)
    for lang in a.langs.split(","):
        if C.done("xnli", f"{lang}.json"):
            C.log(f"{lang}: cached"); continue
        run_language(lang, clfs, info, a.limit)
    C.log(f"xnli done in {(time.time() - t0) / 60:.1f} min")
