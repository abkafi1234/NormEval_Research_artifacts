"""The four benchmark tiers.

A: orthographic vs morphological compression across all 204 FLORES variants,
   plus a controlled multi-script contrast (same language, same sentences,
   two writing systems) that isolates orthography from morphology.
B: full intrinsic profile for the 15 Snowball-supported languages.
C: cross-lingual retrieval, restricted to languages where the encoder is
   actually competent, so a normalization delta is interpretable.
D: FLORES Bangla as a third corpus for the BanLemma/BNLTK comparison.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict

import numpy as np

from common import (
    ARTIFACTS, BENCHMARK_LANGS, BENCHMARK_QUERY_LANGS,
    FLORES_TO_SNOWBALL, RETRIEVAL_PIVOT, bootstrap_ci, build_encoder,
    cohens_dz, compression_decomposition, encode, flores_languages, full_pipeline,
    has_stemmer, load_flores, mcnemar_paired, orthographic, save_json, vocab,
    vocabulary_ratio_replicates,
)

MULTISCRIPT = ["ace", "arb", "bjn", "kas", "knc", "min", "taq", "zho"]


# ---------------------------------------------------------------------------
# Tier A
# ---------------------------------------------------------------------------
def tier_a(limit=None, n_boot=10000, langs=None):
    print("\n=== Tier A: orthographic vs morphological compression ===")
    langs = langs or flores_languages()
    results = {}
    t0 = time.time()
    for i, lc in enumerate(langs, 1):
        raw = load_flores(lc, limit=limit)
        orth = [orthographic(t, lc) for t in raw]
        full = [full_pipeline(t, lc) for t in raw]
        d = compression_decomposition(raw, orth, full)
        d["n_sentences"] = len(raw)
        d["script"] = lc.rsplit("_", 1)[1]
        d["has_stemmer"] = has_stemmer(lc)

        # Independent-split replication rather than a bootstrap: see
        # vocabulary_ratio_replicates() for why a bootstrap is invalid for
        # vocabulary ratios.
        dev_raw = load_flores(lc, splits=("dev",), limit=limit)
        dvt_raw = load_flores(lc, splits=("devtest",), limit=limit)
        d["CR_ortho_replicates"] = vocabulary_ratio_replicates(
            {"dev": dev_raw, "devtest": dvt_raw},
            {"dev": [orthographic(t, lc) for t in dev_raw],
             "devtest": [orthographic(t, lc) for t in dvt_raw]},
        )
        results[lc] = d
        if i % 25 == 0 or i == len(langs):
            print(f"  [{i}/{len(langs)}] {time.time()-t0:.0f}s")

    # Controlled contrast: same language, same sentences, different script.
    controlled = {}
    for base in MULTISCRIPT:
        variants = {lc: results[lc] for lc in results if lc.rsplit("_", 1)[0] == base}
        if len(variants) < 2:
            continue
        controlled[base] = {
            "variants": {lc: {"script": v["script"], "CR_ortho": v["CR_ortho"],
                              "CR_ortho_replicates": v["CR_ortho_replicates"],
                              "CR_total": v["CR_total"], "V_raw": v["V_raw"],
                              "V_ortho": v["V_ortho"]}
                         for lc, v in variants.items()},
            "CR_ortho_spread": max(v["CR_ortho"] for v in variants.values())
                               - min(v["CR_ortho"] for v in variants.values()),
        }

    by_script = defaultdict(list)
    for lc, d in results.items():
        by_script[d["script"]].append(d["CR_ortho"])
    script_summary = {
        s: {"n_languages": len(v), "mean_CR_ortho": float(np.mean(v)),
            "min": float(np.min(v)), "max": float(np.max(v))}
        for s, v in sorted(by_script.items(), key=lambda kv: -len(kv[1]))
    }

    out = {"per_language": results, "controlled_multiscript": controlled,
           "by_script": script_summary}
    save_json(out, "tier_a_orthographic.json")
    return out


# ---------------------------------------------------------------------------
# Tier B
# ---------------------------------------------------------------------------
def _anld(raw_texts, norm_texts):
    """Alignment-based normalized Levenshtein distance, matching the paper's
    definition (sequence alignment, not positional pairing)."""
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
    if not mapping:
        return float("nan"), []
    per_word = [Levenshtein.distance(w, s) / len(w) for w, s in mapping.items() if len(w) > 0]
    return float(np.mean(per_word)), per_word


def _kl(raw_texts, norm_texts, eps=1e-12):
    from sklearn.feature_extraction.text import CountVectorizer
    from scipy.stats import entropy
    vec = CountVectorizer(tokenizer=lambda s: s.split(), token_pattern=None, lowercase=False)
    vec.fit(list(raw_texts) + list(norm_texts))
    p = np.asarray(vec.transform(raw_texts).sum(axis=0)).ravel().astype(float) + eps
    q = np.asarray(vec.transform(norm_texts).sum(axis=0)).ravel().astype(float) + eps
    return float(entropy(p / p.sum(), q / q.sum()))


def tier_b(limit=None, n_boot=10000, encoder=None, model_name=None, compile_model=True):
    print("\n=== Tier B: full intrinsic profile, Snowball languages ===")
    langs = [lc for lc in flores_languages() if has_stemmer(lc)]
    print(f"  {len(langs)} languages with a stemmer: {langs}")
    if encoder is None and model_name:
        encoder = build_encoder(model_name, compile_model=compile_model)

    results = {}
    for lc in langs:
        raw = load_flores(lc, limit=limit)
        orth = [orthographic(t, lc) for t in raw]
        full = [full_pipeline(t, lc) for t in raw]
        d = compression_decomposition(raw, orth, full)
        d["n_sentences"] = len(raw)

        # ANLD measured on the morphological stage only (ortho -> full), so it
        # reflects stemming rather than punctuation/casing edits.
        anld_mean, anld_per_word = _anld(orth, full)
        pt, lo, hi = bootstrap_ci(anld_per_word, n_boot=n_boot)
        d["ANLD_morph"] = anld_mean
        d["ANLD_morph_ci95"] = [lo, hi]
        d["ANLD_n_types"] = len(anld_per_word)
        d["KL_morph"] = _kl(orth, full)

        if encoder is not None:
            e_o = encode(encoder, orth)
            e_f = encode(encoder, full)
            sims = np.sum(e_o * e_f, axis=1)  # already L2-normalized
            pt2, lo2, hi2 = bootstrap_ci(sims, n_boot=n_boot)
            d["IRS_morph"] = float(np.mean(sims))
            d["IRS_morph_ci95"] = [lo2, hi2]
            vrg = 1.0 - 1.0 / d["CR_morph"] if d["CR_morph"] > 0 else 0.0
            irs = d["IRS_morph"]
            d["AES_morph"] = float(2 * irs * vrg / (irs + vrg)) if (irs + vrg) > 0 else 0.0
        results[lc] = d
        print(f"  {lc}: CR_morph={d['CR_morph']:.4f} ANLD={d['ANLD_morph']:.4f} "
              f"KL={d['KL_morph']:.3f}" + (f" IRS={d.get('IRS_morph', float('nan')):.4f}" if encoder else ""))

    save_json(results, "tier_b_intrinsic.json")
    return results


# ---------------------------------------------------------------------------
# Tier C
# ---------------------------------------------------------------------------
def tier_c(limit=None, n_boot=10000, model_name=None, compile_model=True,
           pivot=None, competence_floor=0.50, langs=None, encoder=None,
           dtype="float32"):
    """Cross-lingual retrieval: query in L, retrieve its translation from the
    pivot-language pool. Reported only where the *baseline* (unnormalized)
    retrieval is competent, since a delta between two failing systems is
    uninterpretable."""
    print("\n=== Tier C: cross-lingual retrieval ===")
    if encoder is None:
        encoder = build_encoder(model_name, compile_model=compile_model, dtype=dtype)

    pivot = pivot or RETRIEVAL_PIVOT
    langs = langs if langs is not None else BENCHMARK_QUERY_LANGS
    print(f"  pivot={pivot} | {len(langs)} query languages: {langs}")
    piv_raw = load_flores(pivot, limit=limit)
    piv_orth = [orthographic(t, pivot) for t in piv_raw]

    idx_raw = encode(encoder, piv_raw)
    idx_orth = encode(encoder, piv_orth)
    n = len(piv_raw)
    gold = np.arange(n)

    results = {}
    t0 = time.time()
    for i, lc in enumerate(langs, 1):
        try:
            q_raw = load_flores(lc, limit=limit)
        except FileNotFoundError:
            continue
        q_orth = [orthographic(t, lc) for t in q_raw]

        e_qr = encode(encoder, q_raw)
        e_qo = encode(encoder, q_orth)

        pred_raw = np.argmax(e_qr @ idx_raw.T, axis=1)
        pred_orth = np.argmax(e_qo @ idx_orth.T, axis=1)
        cor_raw = (pred_raw == gold)
        cor_orth = (pred_orth == gold)

        r_raw, lo_r, hi_r = bootstrap_ci(cor_raw.astype(float), n_boot=n_boot)
        r_orth, lo_o, hi_o = bootstrap_ci(cor_orth.astype(float), n_boot=n_boot)
        delta_items = cor_orth.astype(float) - cor_raw.astype(float)
        d_pt, d_lo, d_hi = bootstrap_ci(delta_items, n_boot=n_boot)

        entry = {
            "n_queries": n,
            "recall1_raw": r_raw, "recall1_raw_ci95": [lo_r, hi_r],
            "recall1_orth": r_orth, "recall1_orth_ci95": [lo_o, hi_o],
            "delta_orth": d_pt, "delta_orth_ci95": [d_lo, d_hi],
            "mcnemar_orth": mcnemar_paired(cor_raw, cor_orth),
            "competent": bool(r_raw >= competence_floor),
        }

        if has_stemmer(lc):
            q_full = [full_pipeline(t, lc) for t in q_raw]
            piv_full = [full_pipeline(t, pivot) for t in piv_raw]
            e_qf = encode(encoder, q_full)
            idx_full = encode(encoder, piv_full)
            cor_full = (np.argmax(e_qf @ idx_full.T, axis=1) == gold)
            r_f, lo_f, hi_f = bootstrap_ci(cor_full.astype(float), n_boot=n_boot)
            df_pt, df_lo, df_hi = bootstrap_ci(
                cor_full.astype(float) - cor_raw.astype(float), n_boot=n_boot)
            entry.update({
                "recall1_full": r_f, "recall1_full_ci95": [lo_f, hi_f],
                "delta_full": df_pt, "delta_full_ci95": [df_lo, df_hi],
                "mcnemar_full": mcnemar_paired(cor_raw, cor_full),
            })

        results[lc] = entry
        if i % 25 == 0 or i == len(langs):
            print(f"  [{i}/{len(langs)}] {time.time()-t0:.0f}s")

    comp = [v for v in results.values() if v["competent"]]
    summary = {
        "n_languages": len(results),
        "n_competent": len(comp),
        "competence_floor": competence_floor,
        "mean_delta_orth_competent": float(np.mean([v["delta_orth"] for v in comp])) if comp else float("nan"),
    }
    print(f"  competent languages: {len(comp)}/{len(results)} at floor {competence_floor}")
    save_json({"per_language": results, "summary": summary}, "tier_c_retrieval.json")
    return {"per_language": results, "summary": summary}


# ---------------------------------------------------------------------------
# Tier D
# ---------------------------------------------------------------------------
def tier_d(limit=None, n_boot=10000, encoder=None, model_name=None, compile_model=True):
    """FLORES Bangla as an independent third corpus for BanLemma vs BNLTK."""
    print("\n=== Tier D: Bangla third corpus (FLORES ben_Beng) ===")
    sys.path.insert(0, r"C:\Users\Kafi\AppData\Local\Temp\claude\d--Research-NormEvaluator-pypi-normeval-main\8d800d1a-59f6-48d8-a4b7-3a878e0e4a45\scratchpad\stubs")
    sys.path.insert(0, r"D:\Research\NormEvaluator_pypi\normeval-main\Reproducible Experiment\bangla\BanLemma-main")
    from bnltk.stemmer import BanglaStemmer
    import banlemma

    bn = load_flores("ben_Beng", limit=limit)
    bn_dev = load_flores("ben_Beng", splits=("dev",), limit=limit)
    bn_dvt = load_flores("ben_Beng", splits=("devtest",), limit=limit)
    stemmer = BanglaStemmer()

    def norm_with(tool, texts):
        if tool == "bnltk":
            return [" ".join(stemmer.stem(tok) for tok in t.split()) for t in texts]
        return [banlemma.lemmatize(t) for t in texts]

    tools = {name: norm_with(name, bn) for name in ("bnltk", "banlemma")}
    if encoder is None and model_name:
        encoder = build_encoder(model_name, compile_model=compile_model)

    results = {}
    for name, norm in tools.items():
        v_raw, v_norm = len(vocab(bn)), len(vocab(norm))
        cr = v_raw / v_norm if v_norm else float("nan")
        anld_mean, anld_pw = _anld(bn, norm)
        a_pt, a_lo, a_hi = bootstrap_ci(anld_pw, n_boot=n_boot)
        d = {
            "n_sentences": len(bn), "V_raw": v_raw, "V_norm": v_norm,
            "CR": cr,
            "CR_replicates": vocabulary_ratio_replicates(
                {"dev": bn_dev, "devtest": bn_dvt},
                {"dev": norm_with(name, bn_dev), "devtest": norm_with(name, bn_dvt)},
            ),
            "ANLD": anld_mean, "ANLD_ci95": [a_lo, a_hi],
            "KL": _kl(bn, norm),
        }
        if encoder is not None:
            e_r, e_n = encode(encoder, bn), encode(encoder, norm)
            sims = np.sum(e_r * e_n, axis=1)
            i_pt, i_lo, i_hi = bootstrap_ci(sims, n_boot=n_boot)
            d["IRS"] = float(np.mean(sims)); d["IRS_ci95"] = [i_lo, i_hi]
            vrg = 1.0 - 1.0 / cr if cr > 0 else 0.0
            d["AES"] = float(2 * d["IRS"] * vrg / (d["IRS"] + vrg)) if (d["IRS"] + vrg) > 0 else 0.0
        results[name] = d
        print(f"  {name}: CR={cr:.4f} ANLD={anld_mean:.4f} KL={d['KL']:.3f}"
              + (f" IRS={d['IRS']:.4f} AES={d['AES']:.4f}" if encoder else ""))

    # Does the intrinsic ordering match the two corpora already in the paper?
    if len(results) == 2:
        a, b = results["banlemma"], results["bnltk"]
        results["_ordering"] = {
            "banlemma_CR_gt_bnltk": bool(a["CR"] > b["CR"]),
            "banlemma_ANLD_gt_bnltk": bool(a["ANLD"] > b["ANLD"]),
            "paper_pattern_BTSD_and_sentiment": "BanLemma > BNLTK on CR, ANLD, AES; IRS ~equal",
        }
    save_json(results, "tier_d_bangla.json")
    return results
