"""Experiments B (BTSD ablation: English x4, Bangla x2), C (Bangla sentiment x2) and D2
(same-corpus IRS with BanglaBERT and multilingual MiniLM). See EXPERIMENTAL_PLAN.md §3–5.

Phase 1 (sequential, GPU): normalize, cache normalized texts, intrinsic metrics via normeval.
Phase 2 (parallel, CPU):   10-fold Traditional DSP per (unit, seed, classifier), plus a
                           normeval-package cross-check at seed 42 for LR and MNB.
Phase 3:                   summary.json with everything the paper's tables need.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import exp_common as C

UNITS = [
    ("btsd_en_porter", "btsd", "en", "porter"),
    ("btsd_en_snowball", "btsd", "en", "snowball"),
    ("btsd_en_wordnet", "btsd", "en", "wordnet"),
    ("btsd_en_spacy", "btsd", "en", "spacy"),
    ("btsd_bn_bnltk", "btsd", "bn", "bnltk"),
    ("btsd_bn_banlemma", "btsd", "bn", "banlemma"),
    ("sentiment_bnltk", "sentiment", "bn", "bnltk"),
    ("sentiment_banlemma", "sentiment", "bn", "banlemma"),
]
CLASSIFIERS = ["LogisticRegression", "MultinomialNB", "SVC", "RandomForestClassifier"]
BETAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]


def make_clf(name):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    return {"LogisticRegression": LogisticRegression(), "MultinomialNB": MultinomialNB(),
            "SVC": SVC(random_state=42), "RandomForestClassifier": RandomForestClassifier(random_state=42)}[name]


def make_tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(tokenizer=C.whitespace, token_pattern=None, lowercase=False)


def _subsample(n, limit):
    return list(range(n)) if not limit or limit >= n else sorted(set(np.linspace(0, n - 1, limit).astype(int).tolist()))


def load_btsd(limit=None):
    import pandas as pd
    bn = pd.read_excel(os.path.join(C.DATA, "Bangla Transformation of Sentence Dataset(BTSD).xlsx"))
    en = pd.read_excel(os.path.join(C.DATA, "Bangla Transformation of Sentence Dataset(BTSD)_english translation.xlsx"))
    assert len(bn) == len(en), (len(bn), len(en))
    idx = _subsample(len(bn), limit)
    bn_texts = [str(x) for x in bn[bn.columns[0]].astype(str).tolist()]
    en_texts = [str(x) for x in en[en.columns[0]].astype(str).tolist()]
    labels = [str(x) for x in bn[bn.columns[1]].astype(str).tolist()]
    return [bn_texts[i] for i in idx], [en_texts[i] for i in idx], [labels[i] for i in idx]


def load_sentiment(limit=None):
    import pandas as pd
    df = pd.read_excel(os.path.join(C.DATA, "Bangla Sentiment Dataset", "Sentiment dataset.xlsx"))
    n_all = len(df)
    df = df.dropna(subset=["Label"])
    texts = df["Tense"].astype(str).tolist()
    labels = [{0: "Negative", 1: "Positive", 2: "Neutral"}[int(x)] for x in df["Label"].tolist()]
    idx = _subsample(len(texts), limit)
    return [texts[i] for i in idx], [labels[i] for i in idx], n_all


def texts_path(unit):
    return C.out_path("ablation", "texts", f"{unit}.json")


def phase1(limit):
    from normeval import NormalizationEvaluator
    bn_texts, en_texts, btsd_labels = load_btsd(limit)
    sent_texts, sent_labels, sent_rows = load_sentiment(limit)
    C.log(f"BTSD n={len(bn_texts)} | sentiment n={len(sent_texts)} (rows before dropping unlabeled: {sent_rows})")
    for unit, corpus, lang, method in UNITS:
        if C.done("ablation", "intrinsic", f"{unit}.json"):
            C.log(f"{unit}: intrinsic cached"); continue
        t0 = time.time()
        if corpus == "btsd":
            orig, labels = (en_texts if lang == "en" else bn_texts), btsd_labels
        else:
            orig, labels = sent_texts, sent_labels
        norm = (C.normalize_english_batch(orig, method) if lang == "en" else C.normalize_bangla_batch(orig, method))
        os.makedirs(os.path.dirname(texts_path(unit)), exist_ok=True)
        with open(texts_path(unit), "w", encoding="utf-8") as f:
            json.dump({"orig": orig, "norm": norm, "labels": labels}, f, ensure_ascii=False)
        enc = C.encoder(C.DISTILBERT if lang == "en" else C.BANGLABERT, mean_pooled=True)
        ev = NormalizationEvaluator(orig, norm, embedding_model=enc, tokenizer=C.whitespace)
        cr = ev.calculate_cr()
        irs = ev.calculate_irs(batch_size=64)
        res = {"unit": unit, "corpus": corpus, "lang": lang, "method": method, "n": len(orig),
               "label_counts": {k: int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
               "CR": cr, "VRG": 0.0 if cr < 1 else 1 - 1 / cr, "IRS": irs, "AES_beta1": ev.calculate_aes(cr, irs),
               "ANLD": ev.calculate_anld(), "KL": ev.calculate_kl_divergence(),
               "encoder": C.DISTILBERT if lang == "en" else C.BANGLABERT}
        if lang == "bn":   # D2: same corpus, second encoder
            ev2 = NormalizationEvaluator(orig, norm, embedding_model=C.encoder(C.MINILM), tokenizer=C.whitespace)
            res["IRS_minilm"] = ev2.calculate_irs(batch_size=64)
        if lang == "en":
            vrg = res["VRG"]
            res["AES_beta_sweep"] = {str(b): ((1 + b * b) * irs * vrg / (b * b * irs + vrg)) if (b * b * irs + vrg) > 0 else 0.0
                                     for b in BETAS}
        res["seconds"] = round(time.time() - t0, 1)
        C.log(f"{unit}: CR={cr:.4f} IRS={irs:.4f} ANLD={res['ANLD']:.4f} KL={res['KL']:.2f}"
              + (f" IRS_minilm={res['IRS_minilm']:.4f}" if "IRS_minilm" in res else ""))
        C.save(res, "ablation", "intrinsic", f"{unit}.json")


def dsp_job(unit, seed, clf_name):
    fname = f"{unit}__{clf_name}__seed{seed}.json"
    if C.done("ablation", "dsp", fname):
        return fname, "cached"
    t0 = time.time()
    d = json.load(open(texts_path(unit), encoding="utf-8"))
    det = C.traditional_dsp_detailed(d["orig"], d["norm"], d["labels"], make_clf(clf_name), C.N_SPLITS, seed,
                                     vectorizer=make_tfidf())
    res = {"unit": unit, "classifier": clf_name, "seed": seed, "n_splits": C.N_SPLITS, **C.dsp_unit_result(det, d["labels"]),
           "seconds": round(time.time() - t0, 1)}
    C.save(res, "ablation", "dsp", fname)
    return fname, f"delta={res['stats']['delta_mean']:+.4f} p={res['stats']['wilcoxon_p']:.4f}"


def pkg_check_job(unit):
    """Run normeval's own calculate_traditional_dsp at seed 42 (LR, MNB) to confirm the replica."""
    fname = f"{unit}__pkgcheck.json"
    if C.done("ablation", "pkgcheck", fname):
        return fname, "cached"
    from normeval import NormalizationEvaluator
    d = json.load(open(texts_path(unit), encoding="utf-8"))
    ev = NormalizationEvaluator(d["orig"], d["norm"], labels=d["labels"],
                                classifiers=[make_clf("LogisticRegression"), make_clf("MultinomialNB")],
                                tokenizer=C.whitespace)
    out = ev.calculate_traditional_dsp(n_splits=C.N_SPLITS, random_state=42, average_method="macro", vectorizer=make_tfidf())
    C.save({"unit": unit, "package_result": out}, "ablation", "pkgcheck", fname)
    return fname, json.dumps({k: round(v["DSP (Delta)"], 4) for k, v in out.items()})


def phase2(jobs):
    from joblib import Parallel, delayed
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    tasks = [delayed(dsp_job)(u[0], s, c) for u in UNITS for s in C.SEEDS for c in CLASSIFIERS]
    tasks += [delayed(pkg_check_job)(u[0]) for u in UNITS]
    C.log(f"phase 2: {len(tasks)} jobs on {jobs} workers")
    for name, msg in Parallel(n_jobs=jobs, backend="loky", verbose=0, return_as="generator_unordered")(tasks):
        C.log(f"  {name}: {msg}")


def phase3():
    summary = {"units": {}}
    for unit, corpus, lang, method in UNITS:
        u = {"intrinsic": json.load(open(C.out_path("ablation", "intrinsic", f"{unit}.json")))}
        u["intrinsic"].pop("_meta", None)
        u["dsp"] = {}
        for clf in CLASSIFIERS:
            for s in C.SEEDS:
                p = C.out_path("ablation", "dsp", f"{unit}__{clf}__seed{s}.json")
                if os.path.exists(p):
                    r = json.load(open(p))
                    u["dsp"].setdefault(clf, {})[str(s)] = {"stats": r["stats"], "mcnemar": r["mcnemar"]}
        pc = C.out_path("ablation", "pkgcheck", f"{unit}__pkgcheck.json")
        if os.path.exists(pc):
            pkg = json.load(open(pc))["package_result"]
            u["package_check_seed42"] = {
                clf: {"package_delta": v["DSP (Delta)"], "package_p": v["p-value"],
                      "replica_delta": u["dsp"].get(clf, {}).get("42", {}).get("stats", {}).get("delta_mean"),
                      "replica_p": u["dsp"].get(clf, {}).get("42", {}).get("stats", {}).get("wilcoxon_p")}
                for clf, v in pkg.items()}
        summary["units"][unit] = u
    C.save(summary, "ablation", "summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="subsample size per corpus (smoke test)")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    t0 = time.time()
    C.log(f"ablation start | seeds={C.SEEDS} | out={C.OUT} | limit={a.limit}")
    phase1(a.limit)
    phase2(a.jobs)
    phase3()
    C.log(f"ablation done in {(time.time() - t0) / 60:.1f} min")
