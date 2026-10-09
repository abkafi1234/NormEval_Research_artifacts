"""Revision R1: FLORES-200 intrinsic profile and dense retrieval for recent normalizers.

For each method (snowball = the paper's pipeline, re-measured here on the same machine and
encoder settings; stanza; llm) and each of the 15 languages:
  raw -> ortho = FLORES orthographic stage -> full
  snowball: full = Snowball stems of ortho            (as flores_benchmark/common.full_pipeline)
  lemmatizers: full = orthographic stage applied to the lemmatizer's output on the raw sentence
The measurements copy flores_benchmark/tiers.py (tier B and tier C): compression decomposition,
ANLD and KL on ortho -> full, IRS = cosine(ortho, full) with the multilingual MiniLM encoder (fp32),
AES at beta = 1, and recall@1 for retrieving each sentence's English translation, with the same
condition applied to queries and pool. Spearman correlations across the 14 query languages follow
flores_benchmark/predictiveness.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
import exp_common as C  # noqa: E402
import r1_norm as N  # noqa: E402

PIVOT = "eng_Latn"


def vocab(texts):
    v = set()
    for t in texts:
        v.update(t.split())
    return v


def decomposition(raw, ortho, full):
    v_raw, v_ortho, v_full = len(vocab(raw)), len(vocab(ortho)), len(vocab(full))
    cr_total, cr_ortho, cr_morph = v_raw / v_full, v_raw / v_ortho, v_ortho / v_full
    share = (np.log(cr_morph) / np.log(cr_total)) if cr_total > 1 and cr_morph > 0 else float("nan")
    return {"V_raw": v_raw, "V_ortho": v_ortho, "V_full": v_full, "CR_total": cr_total, "CR_ortho": cr_ortho,
            "CR_morph": cr_morph, "morph_share": float(share)}


def anld(raw_texts, norm_texts):
    from difflib import SequenceMatcher
    import Levenshtein
    mapping = {}
    for r, n in zip(raw_texts, norm_texts):
        rt, nt = r.split(), n.split()
        sm = SequenceMatcher(a=rt, b=nt, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("equal", "replace"):
                for k in range(min(i2 - i1, j2 - j1)):
                    mapping.setdefault(rt[i1 + k], nt[j1 + k])
            elif tag == "delete":
                for k in range(i1, i2):
                    mapping.setdefault(rt[k], "")
    per_word = [Levenshtein.distance(w, s) / len(w) for w, s in mapping.items() if len(w) > 0]
    return float(np.mean(per_word)), per_word


def kl(raw_texts, norm_texts, eps=1e-12):
    from scipy.stats import entropy
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(tokenizer=lambda s: s.split(), token_pattern=None, lowercase=False)
    vec.fit(list(raw_texts) + list(norm_texts))
    p = np.asarray(vec.transform(raw_texts).sum(axis=0)).ravel().astype(float) + eps
    q = np.asarray(vec.transform(norm_texts).sum(axis=0)).ravel().astype(float) + eps
    return float(entropy(p / p.sum(), q / q.sum()))


def embed(model, texts):
    e = C.encode(model, texts, batch_size=128)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def full_text(method, code, raw, limit):
    if method == "snowball":
        return [C.flores_full(t, code) for t in raw]
    m = N.lookup(method, f"flores:{code}", limit)
    return [C.flores_ortho(m[t], code) for t in raw]


def run(method, limit):
    enc = C.encoder(C.MINILM)
    langs = N.LANGS15
    raw = {c: N.load_flores(c, limit) for c in langs}
    orth = {c: [C.flores_ortho(t, c) for t in raw[c]] for c in langs}
    full = {c: full_text(method, c, raw[c], limit) for c in langs}
    n = len(raw[PIVOT]); gold = np.arange(n)
    piv = {k: embed(enc, v) for k, v in (("raw", raw[PIVOT]), ("orth", orth[PIVOT]), ("full", full[PIVOT]))}
    res = {}
    for c in langs:
        d = decomposition(raw[c], orth[c], full[c])
        d["n_sentences"] = len(raw[c])
        a, per_word = anld(orth[c], full[c])
        d["ANLD_morph"] = a
        d["ANLD_morph_ci95"] = list(C.bootstrap_ci(per_word)[1:])
        d["KL_morph"] = kl(orth[c], full[c])
        e_r, e_o, e_f = embed(enc, raw[c]), embed(enc, orth[c]), embed(enc, full[c])
        sims = np.sum(e_o * e_f, axis=1)
        d["IRS_morph"] = float(sims.mean())
        d["IRS_morph_ci95"] = list(C.bootstrap_ci(sims)[1:])
        vrg = 1.0 - 1.0 / d["CR_morph"] if d["CR_morph"] > 0 else 0.0
        d["AES_morph"] = float(2 * d["IRS_morph"] * vrg / (d["IRS_morph"] + vrg)) if (d["IRS_morph"] + vrg) > 0 else 0.0
        d["token_ratio_full_over_ortho"] = sum(len(t.split()) for t in full[c]) / max(1, sum(len(t.split()) for t in orth[c]))
        if c != PIVOT:
            cor = {k: (np.argmax(e @ piv[k].T, axis=1) == gold) for k, e in (("raw", e_r), ("orth", e_o), ("full", e_f))}
            r = {}
            for k in cor:
                pt, lo, hi = C.bootstrap_ci(cor[k].astype(float))
                r[f"recall1_{k}"] = pt; r[f"recall1_{k}_ci95"] = [lo, hi]
            for k in ("orth", "full"):
                pt, lo, hi = C.bootstrap_ci(cor[k].astype(float) - cor["raw"].astype(float))
                r[f"delta_{k}"] = pt; r[f"delta_{k}_ci95"] = [lo, hi]
                r[f"mcnemar_{k}"] = C.mcnemar_paired(cor["raw"], cor[k])
            pt, lo, hi = C.bootstrap_ci(cor["full"].astype(float) - cor["orth"].astype(float))
            r["delta_morph"] = pt; r["delta_morph_ci95"] = [lo, hi]
            r["mcnemar_morph"] = C.mcnemar_paired(cor["orth"], cor["full"])
            r["correct_full"] = cor["full"].astype(int).tolist()
            r["correct_raw"] = cor["raw"].astype(int).tolist()
            d["retrieval"] = r
        d["sample"] = {"raw": raw[c][0], "ortho": orth[c][0], "full": full[c][0]}
        res[c] = d
        rr = d.get("retrieval", {})
        C.log(f"{method} {c}: CR_morph={d['CR_morph']:.4f} ANLD={a:.4f} KL={d['KL_morph']:.3f} IRS={d['IRS_morph']:.4f}"
              + (f" R@1 raw={rr['recall1_raw']:.4f} full={rr['recall1_full']:.4f}" if rr else ""))
    from scipy.stats import spearmanr
    q = [c for c in langs if c != PIVOT]
    pred = {}
    for tgt in ("delta_full", "delta_orth", "delta_morph"):
        y = [res[c]["retrieval"][tgt] for c in q]
        pred[tgt] = {}
        for name, key in (("CR_morph", "CR_morph"), ("ANLD", "ANLD_morph"), ("KL", "KL_morph"), ("IRS", "IRS_morph"), ("AES_beta1", "AES_morph")):
            rho, p = spearmanr([res[c][key] for c in q], y)
            pred[tgt][name] = {"rho": float(rho), "p": float(p)}
    out = {"method": method, "encoder": C.MINILM, "dtype": "float32", "pivot": PIVOT, "limit": limit,
           "per_language": res, "predictiveness": {"n_languages": len(q), "languages": q, "spearman": pred}}
    C.save(out, "flores", f"{method}.json" if not limit else f"{method}__limit{limit}.json")
    C.log(f"{method} predictiveness (delta_full): " + " ".join(f"{k}={v['rho']:+.3f}(p={v['p']:.3f})" for k, v in pred["delta_full"].items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="snowball,stanza,llm")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for m in a.methods.split(","):
        run(m, a.limit)
