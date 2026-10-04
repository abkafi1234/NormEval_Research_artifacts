"""Check every number in the manuscript's result tables against the reproducibility tables.

    python code/verification/check_paper_numbers.py [path/to/paper.tex]

Default manuscript path: ../paper/paper.tex relative to the reproducibility folder.
Standard library only. Each printed number is compared with the stored value rounded to the
printed number of decimals (scientific notation: to the printed mantissa digits). Writes
reports/paper_vs_outputs.md and exits non-zero if anything disagrees.
"""
from __future__ import annotations

import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXPATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "paper", "paper.tex")
# Since 2026-10-01 some tables live in the supplementary material; check both documents.
SUPPPATH = os.path.join(os.path.dirname(TEXPATH), "..", "supplementary", "supplementary.tex")
TAB = os.path.join(ROOT, "tables")
TEX = open(TEXPATH, encoding="utf-8").read()
if os.path.exists(SUPPPATH):
    TEX += "\n" + open(SUPPPATH, encoding="utf-8").read()
APPX = next((p for p in (os.path.join(os.path.dirname(TEXPATH), "appendix_cr_table.tex"),
                         os.path.join(os.path.dirname(SUPPPATH), "appendix_cr_table.tex"))
             if os.path.exists(p)), "")

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:\\times10\^\{[-+]?\d+\})?")
RESULTS: list[tuple[str, int, list[str]]] = []


def rows_csv(name):
    with open(os.path.join(TAB, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table_rows(label, text=None):
    text = text or TEX
    i = text.index(r"\label{" + label + "}")
    body = text[i:text.index(r"\end{table}", i)]
    body = body[body.index(r"\midrule"):body.index(r"\bottomrule")]
    out, section, group = [], None, None
    for raw in body.split("\n"):
        line = raw.strip()
        m = re.match(r"\\multicolumn\{\d+\}\{l\}\{\\textit\{([^}]*)\}\}", line)
        if m:
            section = m.group(1)
            continue
        m = re.match(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", line)
        if m:
            group, line = m.group(1), line[m.end():].strip()
        if not line.endswith("\\\\"):
            continue
        out.append({"section": section, "group": group, "cells": [c.strip() for c in line[:-2].split("&")]})
    return out


def clean(cell):
    for a, b in (("$", ""), (r"\pm", " "), ("{,}", ""), (r"\%", ""), (r"\textbf{", ""), (r"\mathbf{", "")):
        cell = cell.replace(a, b)
    return cell


def nums(cells):
    return [n for c in cells for n in NUM.findall(clean(c))]


def same(printed, expected):
    m = re.match(r"([-+]?\d+(?:\.\d+)?)\\times10\^\{([-+]?\d+)\}", printed)
    if m:
        mant, exp = m.groups()
        d = len(mant.split(".")[1]) if "." in mant else 0
        em, ee = f"{expected:.{d}e}".split("e")
        return float(em) == float(mant) and int(ee) == int(exp)
    d = len(printed.split(".")[1]) if "." in printed else 0
    return float(f"{expected:.{d}f}") == float(printed)


def check(table, key, printed, expected):
    """printed: list of number strings; expected: list of floats (None = skip)."""
    bad = []
    if len(printed) != len(expected):
        bad.append(f"{key}: {len(printed)} numbers printed, {len(expected)} expected ({printed})")
    else:
        for i, (p, e) in enumerate(zip(printed, expected)):
            if e is not None and not same(p, e):
                bad.append(f"{key} col {i + 1}: printed {p}, stored {e!r}")
    n = sum(e is not None for e in expected)
    for t, (tn, cnt, errs) in enumerate(RESULTS):
        if tn == table:
            RESULTS[t] = (tn, cnt + n, errs + bad)
            return
    RESULTS.append((table, n, bad))


f = float
# ----------------------------------------------------------------------------- XNLI
LANG = {"Arabic": "ar", "German": "de", "English": "en", "Spanish": "es", "French": "fr", "Russian": "ru"}
X = {r["lang"]: r for r in rows_csv("xnli_intrinsic_zero_shot.csv")}
D = {r["lang"]: r for r in rows_csv("xnli_cr_decomposition.csv")}
S = {(r["lang"], r["seed"]): r for r in rows_csv("xnli_in_language_dsp.csv")}


def lang_of(cell):
    return LANG[cell.split()[0]]


for r in table_rows("tab:multilingual_nli_results"):
    x = X[lang_of(r["cells"][0])]
    check("tab:multilingual_nli_results", r["cells"][0], nums(r["cells"][1:]),
          [f(x[k]) for k in ("CR", "VRG", "IRS", "AES_beta1", "ANLD", "KL", "F1_orig", "F1_norm", "MPD")])
for r in table_rows("tab:cr_decomposition"):
    d = D[lang_of(r["cells"][0])]
    check("tab:cr_decomposition", r["cells"][0], nums(r["cells"][1:]),
          [f(d[k]) for k in ("V_raw", "V_ortho", "V_full", "CR_total", "CR_ortho", "CR_morph")] + [100 * f(d["morph_share"])])
for r in table_rows("tab:multilingual_nli_results_dsp"):
    s = S[(lang_of(r["cells"][0]), "42")]
    check("tab:multilingual_nli_results_dsp", r["cells"][0], nums(r["cells"][1:]), [f(s["delta_mean"]), f(s["wilcoxon_p"])])
for r in table_rows("tab:ext-xnli"):
    s = S[(lang_of(r["cells"][0]), "42")]
    check("tab:ext-xnli", r["cells"][0], nums(r["cells"][1:]),
          [f(s[k]) for k in ("delta_mean", "delta_sd", "delta_ci95_lo", "delta_ci95_hi", "d_z", "wilcoxon_p", "mcnemar_p")])
for r in table_rows("tab:ext-seed-xnli"):
    l = lang_of(r["cells"][0])
    check("tab:ext-seed-xnli", r["cells"][0], nums(r["cells"][1:]),
          [v for s in ("42", "7", "2024") for v in (f(S[(l, s)]["delta_mean"]), f(S[(l, s)]["wilcoxon_p"]))])

# ----------------------------------------------------------------------------- BTSD / sentiment
UNIT = {("English", "Porter"): "btsd_en_porter", ("English", "Snowball"): "btsd_en_snowball",
        ("English", "WordNet"): "btsd_en_wordnet", ("English", "SpaCy"): "btsd_en_spacy",
        ("Bangla", "BNLTK"): "btsd_bn_bnltk", ("Bangla", "BanLemma"): "btsd_bn_banlemma"}
CLF = {"LR": "LogisticRegression", "MNB": "MultinomialNB", "SVM": "SVC", "RF": "RandomForestClassifier"}
AI = {r["unit"]: r for r in rows_csv("ablation_intrinsic.csv")}
AD = {(r["unit"], r["classifier"], r["seed"]): r for r in rows_csv("ablation_dsp.csv")}
INTR = ("CR", "VRG", "IRS", "AES_beta1", "ANLD", "KL")
EXT = ("delta_mean", "delta_sd", "delta_ci95_lo", "delta_ci95_hi", "d_z", "wilcoxon_p", "mcnemar_p")

for r in table_rows("tab:comparison_multirow_final"):
    u = UNIT[(r["group"], r["cells"][1])]
    check("tab:comparison_multirow_final", u, nums(r["cells"][2:]), [f(AI[u][k]) for k in INTR])
for r in table_rows("tab:aes_beta_sensitivity"):
    u = UNIT[("English", r["cells"][0])]
    check("tab:aes_beta_sensitivity", u, nums(r["cells"][1:]),
          [f(AI[u][f"AES_beta_{b}"]) for b in ("0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "4.0")])
for r in table_rows("tab:btsd_dsp"):
    u = UNIT[(r["group"], r["cells"][1])]
    ds = [f(AD[(u, c, "42")]["delta_mean"]) for c in CLF.values()]
    nsig = sum(f(AD[(u, c, "42")]["wilcoxon_p"]) <= 0.05 for c in CLF.values())
    check("tab:btsd_dsp", u, nums(r["cells"][2:5]), [sum(ds) / 4, min(ds), max(ds), nsig, 4])
for r in table_rows("tab:final_comparison"):
    u = "sentiment_" + r["cells"][0].split()[0].lower()
    check("tab:final_comparison", u, nums(r["cells"][1:]), [f(AI[u][k]) for k in INTR])
for r in table_rows("tab:final_comparison_dsp"):
    u = "sentiment_" + r["cells"][0].lower()
    check("tab:final_comparison_dsp", u, nums(r["cells"][1:]),
          [v for c in CLF.values() for v in (f(AD[(u, c, "42")]["delta_mean"]), f(AD[(u, c, "42")]["wilcoxon_p"]))])
for r in table_rows("tab:ext-btsd-en"):
    u = UNIT[("English", r["group"])]
    s = AD[(u, CLF[r["cells"][1]], "42")]
    check("tab:ext-btsd-en", f"{u} {r['cells'][1]}", nums(r["cells"][2:]), [f(s[k]) for k in EXT])
for r in table_rows("tab:ext-bangla"):
    corpus = "btsd_bn_" if r["section"].startswith("BTSD") else "sentiment_"
    u = corpus + r["group"].lower()
    s = AD[(u, CLF[r["cells"][1]], "42")]
    check("tab:ext-bangla", f"{u} {r['cells'][1]}", nums(r["cells"][2:]), [f(s[k]) for k in EXT])
for r in table_rows("tab:ext-seed-reversal"):
    u = ("btsd_bn_" if r["group"] == "BTSD" else "sentiment_") + r["cells"][1].lower()
    check("tab:ext-seed-reversal", u, nums(r["cells"][2:]),
          [v for s in ("42", "7", "2024") for v in (f(AD[(u, "MultinomialNB", s)]["delta_mean"]),
                                                     f(AD[(u, "MultinomialNB", s)]["wilcoxon_p"]))])

# ----------------------------------------------------------------------------- FLORES
TB = {r["lang"]: r for r in rows_csv("flores_tier_b_intrinsic.csv")}
TC = {r["lang"]: r for r in rows_csv("flores_tier_c_dense_retrieval.csv")}
TA = {r["lang"]: r for r in rows_csv("flores_tier_a_orthographic.csv")}
MS = {(r["language"], r["script"]): r for r in rows_csv("flores_multiscript.csv")}
BYS = {r["script"]: r for r in rows_csv("flores_by_script.csv")}


def code(cell):
    return cell.replace("\\_", "_").strip()


for r in table_rows("tab:flores_intrinsic"):
    b = TB[code(r["cells"][0])]
    check("tab:flores_intrinsic", code(r["cells"][0]), nums(r["cells"][1:]),
          [f(b["CR_total"]), f(b["CR_ortho"]), f(b["CR_morph"]), 100 * f(b["morph_share"]), f(b["ANLD_morph"]),
           f(b["IRS_morph"]), f(b["AES_morph"])])
for r in table_rows("tab:flores_retrieval"):
    c = TC[code(r["cells"][0])]
    check("tab:flores_retrieval", code(r["cells"][0]), nums(r["cells"][1:]),
          [f(c[k]) for k in ("R@1_raw", "R@1_ortho", "delta_ortho", "R@1_full", "delta_full")])
NAMES = {"Central Kanuri": "knc", "Acehnese": "ace", "Minangkabau": "min", "Kashmiri": "kas", "Tamasheq": "taq",
         "Banjar": "bjn", "Standard Arabic": "arb", "Chinese": "zho"}
for r in table_rows("tab:crossscript"):
    lg = NAMES[r["cells"][0]]
    a, b = MS[(lg, r["cells"][1])], MS[(lg, r["cells"][4])]
    check("tab:crossscript", lg, nums([r["cells"][2], r["cells"][5], r["cells"][6]]),
          [f(a["CR_ortho"]), f(b["CR_ortho"]), f(a["CR_ortho_spread"])])
for r in table_rows("tab:appendix_script_summary"):
    s = BYS[re.search(r"\((\w+)\)", r["cells"][0]).group(1)]
    check("tab:appendix_script_summary", r["cells"][0], nums(r["cells"][1:]),
          [f(s["n_languages"]), f(s["mean_CR_ortho"]), f(s["min"]), f(s["max"])])
if os.path.exists(APPX):
    appx = open(APPX, encoding="utf-8").read()
    for lg, v in re.findall(r"([a-z]{3}\\_[A-Za-z]{4})\s*&\s*([0-9.]+)", appx):
        check("tab:appendix_cr_ortho", code(lg), [v], [f(TA[code(lg)]["CR_ortho"])])


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5


PRED = open(os.path.join(ROOT, "reports", "flores_predictiveness_output.txt"), encoding="utf-8").read()
PV = {m[0]: (f(m[2]), f(m[4])) for m in re.findall(r"^(\w+)\s+rho=([-+]?[\d.]+) p=([\d.]+)\s+rho=([-+]?[\d.]+) p=([\d.]+)", PRED, re.M)}
langs = [l for l in TC if l in TB]
DIM = {"IRS": ("IRS_morph", "IRS"), "ANLD": ("ANLD_morph", "ANLD"), "KL": ("KL_morph", "KL"),
       "CR": ("CR_morph", "CR_morph"), "AES": ("AES_morph", "AES")}
for r in table_rows("tab:flores_predict"):
    key = next(k for k in DIM if k in r["cells"][0])
    col, pk = DIM[key]
    x = [f(TB[l][col]) for l in langs]
    check("tab:flores_predict", key, nums(r["cells"][1:]),
          [spearman(x, [f(TC[l]["delta_ortho"]) for l in langs]), PV[pk][0],
           spearman(x, [f(TC[l]["delta_full"]) for l in langs]), PV[pk][1]])

# ----------------------------------------------------------------------------- BM25 and encoder dependence (new)
BR = {(r["lang"], r["condition"]): r for r in rows_csv("bm25_retrieval.csv") if r["params"] == "k1=0.9,b=0.4"}
BC = {(r["lang"], r["comparison"]): r for r in rows_csv("bm25_comparisons.csv")}
DV = {r["lang"]: r for r in rows_csv("dense_vs_sparse.csv")}
for r in table_rows("tab:bm25"):
    l = code(r["cells"][0])
    st = BC[(l, "ortho->full")]
    dense = DV[l]["dense_delta_R@1_raw_to_full"]
    check("tab:bm25", l, nums(r["cells"][1:]),
          [f(BR[(l, c)]["MRR@10"]) for c in ("tok", "ortho", "full")]
          + [f(BC[(l, "tok->ortho")]["delta_MRR@10"]), f(st["delta_MRR@10"]), f(st["delta_MRR@10_ci95_lo"]),
             f(st["delta_MRR@10_ci95_hi"]), f(BC[(l, "tok->full")]["delta_R@1"])] + ([f(dense)] if dense else []))
E = {(r["corpus"].split(" ")[0], r["tool"]): r for r in rows_csv("encoder_dependence_irs.csv")}
for r in table_rows("tab:encoder_dependence"):
    c = r["cells"][0].split(" ")[0].replace("Sentiment", "Bangla")
    vals = [f(E[(c, "bnltk")]["n"])]
    for col in ("IRS_minilm", "IRS_banglabert"):
        a, b = f(E[(c, "bnltk")][col]), f(E[(c, "banlemma")][col])
        vals += [a, b, a - b]
    check("tab:encoder_dependence", r["cells"][0], nums(r["cells"][1:]), vals)

# ----------------------------------------------------------------------------- report
total = sum(n for _, n, _ in RESULTS)
errors = [e for _, _, errs in RESULTS for e in errs]
lines = ["# Manuscript numbers vs. stored outputs", "",
         f"Manuscript: `{os.path.relpath(TEXPATH, ROOT)}` + `{os.path.relpath(SUPPPATH, ROOT)}` "
         f"(+ `appendix_cr_table.tex`). "
         "Generated by `code/verification/check_paper_numbers.py`.", "",
         "Each printed number is compared with the stored value (tables/*.csv, generated from outputs/) rounded "
         "to the printed precision. `tab:flores_predict` p-values are checked against "
         "`reports/flores_predictiveness_output.txt`; its rho values are recomputed here from tiers B and C.", "",
         "| table | numbers checked | mismatches |", "|---|---:|---:|"]
lines += [f"| `{t}` | {n} | {len(e)} |" for t, n, e in RESULTS]
lines += [f"| **total** | **{total}** | **{len(errors)}** |", ""]
if errors:
    lines += ["## Mismatches", ""] + [f"- {e}" for e in errors]
else:
    lines += ["No mismatches."]
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
with open(os.path.join(ROOT, "reports", "paper_vs_outputs.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if errors else 0)
