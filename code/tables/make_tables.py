"""Flatten every result in outputs/ into CSV tables under tables/.

    python code/tables/make_tables.py          (run from anywhere; standard library only)

Values are written at full stored precision. Rounding happens only where a number is
typeset in the paper, so every printed value can be traced to one cell here and one field
in a JSON file. tables/INDEX.md lists each table with the JSON fields it reads.
"""
from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs")
TAB = os.path.join(ROOT, "tables")

XNLI_LANGS = ["ar", "de", "en", "es", "fr", "ru"]
SEEDS = ["42", "7", "2024"]
UNITS = ["btsd_en_porter", "btsd_en_snowball", "btsd_en_wordnet", "btsd_en_spacy",
         "btsd_bn_bnltk", "btsd_bn_banlemma", "sentiment_bnltk", "sentiment_banlemma"]
CLFS = ["LogisticRegression", "MultinomialNB", "SVC", "RandomForestClassifier"]
BM25_PRIMARY, BM25_SENS = "k1=0.9,b=0.4", "k1=1.2,b=0.75"

INDEX: list[tuple[str, int, str, str]] = []


def load(*parts):
    with open(os.path.join(OUT, *parts), encoding="utf-8") as f:
        return json.load(f)


def write(name, header, rows, source, note=""):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            assert len(r) == len(header), (name, r)
            w.writerow(["" if v is None else v for v in r])
    INDEX.append((name, len(rows), source, note))


def mc(m):
    """McNemar block -> (only first correct, only second correct, statistic, p, exact)."""
    b = m.get("orig_only_correct", m.get("n_a_only"))
    c = m.get("norm_only_correct", m.get("n_b_only"))
    return [b, c, m["statistic"], m["p_value"], m["exact"]]


def dsp_row(block):
    s = block["stats"]
    return [s["n_folds"], s["mean_orig"], s["mean_norm"], s["delta_mean"], s["delta_sd"],
            s["ci95"][0], s["ci95"][1], s["d_z"], s["wilcoxon_p"]] + mc(block["mcnemar"])


DSP_HEAD = ["n_folds", "F1_orig_mean", "F1_norm_mean", "delta_mean", "delta_sd", "delta_ci95_lo",
            "delta_ci95_hi", "d_z", "wilcoxon_p", "mcnemar_orig_only_correct",
            "mcnemar_norm_only_correct", "mcnemar_statistic", "mcnemar_p", "mcnemar_exact"]


# --------------------------------------------------------------------------- XNLI
def xnli():
    X = {l: load("xnli", f"{l}.json") for l in XNLI_LANGS}
    write("xnli_intrinsic_zero_shot.csv",
          ["lang", "n", "CR", "VRG", "IRS", "AES_beta1", "ANLD", "KL", "F1_orig", "F1_norm", "MPD",
           "zs_mcnemar_orig_only_correct", "zs_mcnemar_norm_only_correct", "zs_mcnemar_statistic",
           "zs_mcnemar_p", "zs_mcnemar_exact"],
          [[l, X[l]["n"]] + [X[l]["intrinsic"][k] for k in ("CR", "VRG", "IRS", "AES_beta1", "ANLD", "KL")]
           + [X[l]["zero_shot"][k] for k in ("F1_orig", "F1_norm", "MPD")] + mc(X[l]["zero_shot"]["mcnemar"])
           for l in XNLI_LANGS],
          "outputs/xnli/<lang>.json: intrinsic.*, zero_shot.*",
          "Intrinsic metrics on concatenated premise+hypothesis (validation, n=2490); zero-shot F1 is macro-F1.")
    write("xnli_cr_decomposition.csv",
          ["lang", "V_raw", "V_ortho", "V_full", "CR_total", "CR_ortho", "CR_morph", "morph_share"],
          [[l] + [X[l]["cr_decomposition"][k] for k in ("V_raw", "V_ortho", "V_full", "CR_total",
                                                          "CR_ortho", "CR_morph", "morph_share")]
           for l in XNLI_LANGS],
          "outputs/xnli/<lang>.json: cr_decomposition.*",
          "morph_share = ln(CR_morph) / ln(CR_total).")
    write("xnli_in_language_dsp.csv", ["lang", "seed"] + DSP_HEAD,
          [[l, s] + dsp_row(X[l]["in_language_dsp"][s]) for l in XNLI_LANGS for s in SEEDS],
          "outputs/xnli/<lang>.json: in_language_dsp.<seed>.*",
          "10-fold stratified CV, LogisticRegression on MiniLM embeddings; Wilcoxon exact on 10 paired fold scores.")
    rows = []
    for l in XNLI_LANGS:
        pc = X[l]["package_check_seed42"]
        pd, pp = pc["package"]["DSP (Delta)"], pc["package"]["p-value"]
        rows.append([l, pd, pp, pc["replica_delta"], pc["replica_p"], abs(pd - pc["replica_delta"])])
    write("xnli_package_check.csv",
          ["lang", "package_delta", "package_p", "replica_delta", "replica_p", "abs_diff_delta"], rows,
          "outputs/xnli/<lang>.json: package_check_seed42.*",
          "normeval.calculate_traditional_dsp (seed 42) vs the fold-level replica used for the tables.")
    info = load("zero_shot_classifiers", "clf_info.json")
    write("xnli_zero_shot_classifiers.csv",
          ["condition", "n_train", "encoder", "n_iter", "converged", "fit_seconds"],
          [[c, info["n_train"], info["encoder"], info[c]["n_iter"], info[c]["converged"], info[c]["fit_seconds"]]
           for c in ("orig", "norm")],
          "outputs/zero_shot_classifiers/clf_info.json",
          "LogisticRegression(C=1, lbfgs, max_iter=1000) on InferSent features of MultiNLI train.")


# --------------------------------------------------------------------------- BTSD / Bangla sentiment
def ablation():
    S = load("ablation", "summary.json")["units"]
    sweep_keys = sorted({k for u in UNITS for k in S[u]["intrinsic"].get("AES_beta_sweep", {})}, key=float)
    write("ablation_intrinsic.csv",
          ["unit", "corpus", "lang", "method", "n", "encoder", "CR", "VRG", "IRS", "AES_beta1", "ANLD", "KL",
           "IRS_minilm"] + [f"AES_beta_{k}" for k in sweep_keys],
          [[u] + [S[u]["intrinsic"].get(k) for k in ("corpus", "lang", "method", "n", "encoder", "CR", "VRG",
                                                      "IRS", "AES_beta1", "ANLD", "KL", "IRS_minilm")]
           + [S[u]["intrinsic"].get("AES_beta_sweep", {}).get(k) for k in sweep_keys] for u in UNITS],
          "outputs/ablation/summary.json: units.<unit>.intrinsic (= outputs/ablation/intrinsic/<unit>.json)",
          "IRS uses distilbert-base-uncased (English) or csebuetnlp/banglabert (Bangla), mean-pooled; "
          "IRS_minilm is the multilingual MiniLM control for Bangla.")
    write("ablation_dsp.csv", ["unit", "classifier", "seed"] + DSP_HEAD,
          [[u, c, s] + dsp_row(S[u]["dsp"][c][s]) for u in UNITS for c in CLFS for s in SEEDS],
          "outputs/ablation/summary.json: units.<unit>.dsp.<clf>.<seed> (= outputs/ablation/dsp/*.json)",
          "10-fold stratified CV on TF-IDF features (whitespace tokenizer, no lowercasing).")
    rows = []
    for u in UNITS:
        for c, v in S[u]["package_check_seed42"].items():
            rows.append([u, c, v["package_delta"], v["package_p"], v["replica_delta"], v["replica_p"],
                         abs(v["package_delta"] - v["replica_delta"])])
    write("ablation_package_check.csv",
          ["unit", "classifier", "package_delta", "package_p", "replica_delta", "replica_p", "abs_diff_delta"],
          rows, "outputs/ablation/summary.json: units.<unit>.package_check_seed42",
          "normeval.calculate_traditional_dsp vs the replica, seed 42.")


# --------------------------------------------------------------------------- BM25 on Belebele
def bm25():
    B = load("bm25", "summary.json")
    langs = list(B["table"])
    rows = []
    for l in langs:
        per = load("bm25", f"{l}.json")
        for cond, byp in B["table"][l].items():
            e = per["conditions"][cond]
            for p, m in byp.items():
                rows.append([l, cond, p, per["n_questions"], per["n_passages"], m["R@1"], m["R@10"], m["MRR@10"],
                             m["mean_rank"], e["passage_types"], e["passage_tokens"],
                             e["questions_with_no_indexed_term"] if p == BM25_PRIMARY else None])
    write("bm25_retrieval.csv",
          ["lang", "condition", "params", "n_questions", "n_passages", "R@1", "R@10", "MRR@10", "mean_rank",
           "passage_types", "passage_tokens", "questions_with_no_indexed_term"], rows,
          "outputs/bm25/summary.json: table.*; outputs/bm25/<lang>.json: conditions.*",
          "Okapi BM25, question -> own-language passage (488 passages); ties ranked pessimistically.")
    rows = []
    for l in langs:
        for comp, v in B["comparisons"][l].items():
            rows.append([l, comp, v["delta_R@1"], *v["delta_R@1_ci95"], v["delta_MRR@10"], *v["delta_MRR@10_ci95"]]
                        + mc(v["mcnemar_R@1"]))
    write("bm25_comparisons.csv",
          ["lang", "comparison", "delta_R@1", "delta_R@1_ci95_lo", "delta_R@1_ci95_hi", "delta_MRR@10",
           "delta_MRR@10_ci95_lo", "delta_MRR@10_ci95_hi", "mcnemar_R@1_first_only_correct",
           "mcnemar_R@1_second_only_correct", "mcnemar_statistic", "mcnemar_p", "mcnemar_exact"], rows,
          "outputs/bm25/summary.json: comparisons.* (primary parameters k1=0.9, b=0.4)",
          "delta = second condition minus first; bootstrap 95% CIs over questions.")
    rows = []
    for l in langs:
        t = B["table"][l]
        pairs = [("tok", "ortho"), ("ortho", "full"), ("tok", "full")] if "full" in t else \
                [("tok", "ortho"), ("tok", "bnltk"), ("tok", "banlemma"), ("bnltk", "banlemma")]
        for a, b in pairs:
            dp = t[b][BM25_PRIMARY]["MRR@10"] - t[a][BM25_PRIMARY]["MRR@10"]
            ds = t[b][BM25_SENS]["MRR@10"] - t[a][BM25_SENS]["MRR@10"]
            rows.append([l, f"{a}->{b}", dp, ds, (dp > 0) == (ds > 0)])
    write("bm25_sensitivity.csv", ["lang", "comparison", "delta_MRR@10_k1=0.9_b=0.4", "delta_MRR@10_k1=1.2_b=0.75",
                                   "same_sign"], rows,
          "derived from outputs/bm25/summary.json: table.*",
          "Parameter-sensitivity check: does the sign of each MRR@10 change survive k1=1.2, b=0.75?")
    P = B["predictiveness"]
    rows = [[t, d, v["rho"], v["p"], v["p_holm"], P["n_languages"]]
            for t, dd in P["spearman"].items() for d, v in dd.items()]
    write("bm25_predictiveness.csv", ["target", "intrinsic_dimension", "spearman_rho", "p", "p_holm", "n_languages"],
          rows, "outputs/bm25/summary.json: predictiveness.spearman",
          "Intrinsic dimensions from FLORES tier B (morphological stage); Holm correction within each target.")
    tb = load("flores_benchmark", "tier_b_intrinsic.json")
    dims = {"CR_morph": "CR_morph", "ANLD_morph": "ANLD_morph", "KL_morph": "KL_morph", "IRS_morph": "IRS_morph",
            "AES_morph": "AES_morph"}
    rows = [[l] + [P["targets"][t][i] for t in P["targets"]] + [tb[l][k] for k in dims.values()]
            for i, l in enumerate(P["languages"])]
    write("bm25_predictiveness_inputs.csv", ["lang"] + list(P["targets"]) + list(dims), rows,
          "outputs/bm25/summary.json: predictiveness.targets + outputs/flores_benchmark/tier_b_intrinsic.json",
          "The per-language vectors the Spearman correlations are computed from.")


# --------------------------------------------------------------------------- encoder dependence of IRS
def encoder_check():
    F = load("encoder_check", "flores_ben.json")
    S = load("ablation", "summary.json")["units"]
    rows = [["FLORES-200 ben_Beng (dev+devtest)", F["n"], t, F["tools"][t]["CR"], F["tools"][t]["IRS_minilm"],
             F["tools"][t]["IRS_banglabert"], F["tools"][t]["AES_beta1_minilm"], F["tools"][t]["AES_beta1_banglabert"]]
            for t in ("bnltk", "banlemma")]
    for corpus, label in (("btsd", "BTSD (Bangla)"), ("sentiment", "Bangla sentiment")):
        for t in ("bnltk", "banlemma"):
            i = S[f"{corpus}_bn_{t}" if corpus == "btsd" else f"{corpus}_{t}"]["intrinsic"]
            rows.append([label, i["n"], t, i["CR"], i["IRS_minilm"], i["IRS"], None, i["AES_beta1"]])
    write("encoder_dependence_irs.csv",
          ["corpus", "n", "tool", "CR", "IRS_minilm", "IRS_banglabert", "AES_beta1_minilm", "AES_beta1_banglabert"],
          rows, "outputs/encoder_check/flores_ben.json; outputs/ablation/summary.json (bn units)",
          "Same normalized texts scored by two encoders; the BNLTK-BanLemma IRS gap is the quantity of interest.")


# --------------------------------------------------------------------------- FLORES-200 benchmark (tiers A-C)
def flores():
    A = load("flores_benchmark", "tier_a_orthographic.json")
    keys = sorted({k for v in A["per_language"].values() for k in v if not isinstance(v[k], (dict, list))})
    write("flores_tier_a_orthographic.csv", ["lang"] + keys,
          [[l] + [v.get(k) for k in keys] for l, v in sorted(A["per_language"].items())],
          "outputs/flores_benchmark/tier_a_orthographic.json: per_language", "All 204 FLORES-200 variants.")
    rows = [[lang, var, d["script"], d["CR_ortho"], g["CR_ortho_spread"]]
            for lang, g in A["controlled_multiscript"].items() for var, d in g["variants"].items()]
    write("flores_multiscript.csv", ["language", "variant", "script", "CR_ortho", "CR_ortho_spread"], rows,
          "outputs/flores_benchmark/tier_a_orthographic.json: controlled_multiscript",
          "Same language and sentences in two scripts.")
    write("flores_by_script.csv", ["script", "n_languages", "mean_CR_ortho", "min", "max"],
          [[s, v["n_languages"], v["mean_CR_ortho"], v["min"], v["max"]] for s, v in A["by_script"].items()],
          "outputs/flores_benchmark/tier_a_orthographic.json: by_script")
    B = load("flores_benchmark", "tier_b_intrinsic.json")
    write("flores_tier_b_intrinsic.csv",
          ["lang", "V_raw", "V_ortho", "V_full", "CR_total", "CR_ortho", "CR_morph", "morph_share", "ANLD_morph",
           "ANLD_morph_ci95_lo", "ANLD_morph_ci95_hi", "KL_morph", "IRS_morph", "IRS_morph_ci95_lo",
           "IRS_morph_ci95_hi", "AES_morph", "n_sentences"],
          [[l] + [B[l][k] for k in ("V_raw", "V_ortho", "V_full", "CR_total", "CR_ortho", "CR_morph", "morph_share",
                                    "ANLD_morph")] + B[l]["ANLD_morph_ci95"] + [B[l]["KL_morph"], B[l]["IRS_morph"]]
           + B[l]["IRS_morph_ci95"] + [B[l]["AES_morph"], B[l]["n_sentences"]] for l in B],
          "outputs/flores_benchmark/tier_b_intrinsic.json", "15 Snowball languages, dev+devtest (2,009 sentences).")
    C = load("flores_benchmark", "tier_c_retrieval.json")
    rows = []
    for l, v in C["per_language"].items():
        rows.append([l, v["n_queries"], v["competent"], v["recall1_raw"], *v["recall1_raw_ci95"], v["recall1_orth"],
                     *v["recall1_orth_ci95"], v["delta_orth"], *v["delta_orth_ci95"], v["mcnemar_orth"]["p_value"],
                     v["recall1_full"], *v["recall1_full_ci95"], v["delta_full"], *v["delta_full_ci95"],
                     v["mcnemar_full"]["p_value"]])
    write("flores_tier_c_dense_retrieval.csv",
          ["lang", "n_queries", "competent", "R@1_raw", "R@1_raw_ci95_lo", "R@1_raw_ci95_hi", "R@1_ortho",
           "R@1_ortho_ci95_lo", "R@1_ortho_ci95_hi", "delta_ortho", "delta_ortho_ci95_lo", "delta_ortho_ci95_hi",
           "mcnemar_ortho_p", "R@1_full", "R@1_full_ci95_lo", "R@1_full_ci95_hi", "delta_full",
           "delta_full_ci95_lo", "delta_full_ci95_hi", "mcnemar_full_p"], rows,
          "outputs/flores_benchmark/tier_c_retrieval.json: per_language",
          "Dense cross-lingual retrieval with multilingual MiniLM (fp32).")
    P = load("flores_benchmark", "precision_check.json")
    write("flores_precision_check.csv", ["lang", "IRS_fp32", "IRS_bf16", "abs_diff", "max_per_sentence_abs_diff"],
          [[l] + [v[k] for k in ("IRS_fp32", "IRS_bf16", "abs_diff", "max_per_sentence_abs_diff")]
           for l, v in P.items() if not l.startswith("_")],
          "outputs/flores_benchmark/precision_check.json",
          "Why the benchmark used fp32: bf16 moved Bangla IRS by 1.5e-3.")


# --------------------------------------------------------------------------- dense vs sparse contrast
def dense_vs_sparse():
    B = load("bm25", "summary.json")
    C = load("flores_benchmark", "tier_c_retrieval.json")["per_language"]
    rows = []
    for l in sorted(B["table"]):
        if "full" not in B["table"][l]:
            continue
        cm = B["comparisons"][l]
        rows.append([l, cm["tok->full"]["delta_R@1"], cm["tok->full"]["mcnemar_R@1"]["p_value"],
                     cm["ortho->full"]["delta_MRR@10"], C[l]["delta_full"] if l in C else None,
                     C[l]["mcnemar_full"]["p_value"] if l in C else None])
    write("dense_vs_sparse.csv",
          ["lang", "bm25_delta_R@1_tok_to_full", "bm25_mcnemar_p", "bm25_delta_MRR@10_ortho_to_full",
           "dense_delta_R@1_raw_to_full", "dense_mcnemar_p"], rows,
          "outputs/bm25/summary.json + outputs/flores_benchmark/tier_c_retrieval.json",
          "Same normalization pipeline, two retrieval consumers. English has no dense row "
          "(it is the retrieval pivot in tier C).")


def main():
    xnli()
    ablation()
    bm25()
    encoder_check()
    flores()
    dense_vs_sparse()
    lines = ["# Tables generated from outputs/", "",
             "Regenerate with `python code/tables/make_tables.py`. Full stored precision; no rounding.", "",
             "| file | rows | read from | note |", "|---|---:|---|---|"]
    lines += [f"| `{n}` | {r} | {s} | {t} |" for n, r, s, t in INDEX]
    with open(os.path.join(TAB, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    for n, r, _, _ in INDEX:
        print(f"{r:5d}  {n}")


if __name__ == "__main__":
    main()
