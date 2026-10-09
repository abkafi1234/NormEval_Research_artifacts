"""Revision R1: tables and values for the new-methods results, generated from the stored outputs.

    python3 records/19_ipm_editor_revision/make_r1_results.py

Reads   experiments/outputs/summary_r1.json, experiments/outputs/ablation/intrinsic/*.json,
        reproducibility/outputs/ablation/summary.json (the existing tools, for the reference rows)
Writes  fragments/gen_tables_newmethods.tex   three tables
        fragments/gen_values.json             every number used in the new prose, as <<name>> -> text
build_revision.py substitutes the <<name>> tokens in the fragments; nothing is typed by hand.
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXP = os.path.join(HERE, "experiments", "outputs")
S = json.load(open(os.path.join(EXP, "summary_r1.json")))
OLD = json.load(open(os.path.join(ROOT, "reproducibility", "outputs", "ablation", "summary.json")))["units"]
NEW = json.load(open(os.path.join(EXP, "ablation", "summary.json")))["units"]
HAS_LLM = "llm" in S["flores"]

NAME = {"arb_Arab": "Arabic", "dan_Latn": "Danish", "nld_Latn": "Dutch", "eng_Latn": "English", "fin_Latn": "Finnish",
        "fra_Latn": "French", "deu_Latn": "German", "hun_Latn": "Hungarian", "ita_Latn": "Italian",
        "nob_Latn": "Norwegian", "por_Latn": "Portuguese", "ron_Latn": "Romanian", "rus_Cyrl": "Russian",
        "spa_Latn": "Spanish", "swe_Latn": "Swedish"}
V = {}


def sgn(x, nd=4):
    return "$%+.*f$" % (nd, x)


def pval(p):
    if p < 0.001:
        return "$<0.001$"
    return "%.3f" % p


def rho_cell(d):
    star = "\\textbf{%+.3f}" % d["rho"] if d["p"] < 0.05 else "%+.3f" % d["rho"]
    return "%s (%s)" % (star, pval(d["p"]))


# ------------------------------------------------------------------ Table A: per language
sn, st = S["flores"]["snowball"]["per_language"], S["flores"]["stanza"]["per_language"]
ll = S["flores"]["llm"]["per_language"] if HAS_LLM else None
langs = sorted(NAME, key=lambda c: st[c].get("delta_full", 1.0))
rows = []
for c in langs:
    b = S["bm25"][c]
    dense = lambda P: "--" if "delta_full" not in P[c] else "$%+.1f$" % (100 * P[c]["delta_full"])
    cells = [NAME[c], "%.3f" % sn[c]["CR_morph"], "%.3f" % st[c]["CR_morph"]] + (["%.3f" % ll[c]["CR_morph"]] if HAS_LLM else [])
    cells += ["%.3f" % sn[c]["IRS_morph"], "%.3f" % st[c]["IRS_morph"]] + (["%.3f" % ll[c]["IRS_morph"]] if HAS_LLM else [])
    cells += [dense(sn), dense(st)] + ([dense(ll)] if HAS_LLM else [])
    cells += [sgn(b["ortho->full"], 3), sgn(b["ortho->stanza"]["d"], 3)] + ([sgn(b["ortho->llm"]["d"], 3)] if HAS_LLM else [])
    rows.append(" & ".join(cells) + " \\\\")
k = 3 if HAS_LLM else 2
heads = ["Snowball", "Stanza"] + (["LLM"] if HAS_LLM else [])
tabA = (
    "\\begin{table}[!htbp]\n\\centering\n\\caption{A stemmer and " + ("two recent lemmatizers, the neural lemmatizer of Stanza and a small open-weight LLM prompted with in-context examples," if HAS_LLM else "a neural lemmatizer")
    + " on the same FLORES-200 content: morphological compression, semantic retention, the change in dense recall@1 from unnormalized text to the full pipeline (percentage points; English is the retrieval pivot), and the change in BM25 MRR@10 on Belebele from the orthographic stage to the full pipeline. Rows are sorted by the dense-retrieval change under Stanza.}\n"
    "\\label{tab:r1_flores}\n" + ("\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n" if HAS_LLM else "\\small\n\\setlength{\\tabcolsep}{4pt}\n") +
    "\\begin{tabular}{l" + "c" * (4 * k) + "}\n\\toprule\n"
    " & \\multicolumn{%d}{c}{$CR_{\\text{morph}}$} & \\multicolumn{%d}{c}{IRS} & \\multicolumn{%d}{c}{$\\Delta$ dense R@1} & \\multicolumn{%d}{c}{$\\Delta$ BM25 MRR@10} \\\\\n" % (k, k, k, k)
    + "".join("\\cmidrule(lr){%d-%d}" % (2 + i * k, 1 + (i + 1) * k) for i in range(4)) + "\n"
    "\\textbf{Language} & " + " & ".join(heads * 4) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------------ Table B: predictiveness
dims = [("$CR_{\\text{morph}}$", "CR_morph", "CR_morph"), ("ANLD", "ANLD", "ANLD_morph"), ("KL", "KL", "KL_morph"),
        ("IRS", "IRS", "IRS_morph"), ("$AES_{\\beta=1}$", "AES_beta1", "AES_morph")]
rowsB = []
for label, k1, k2 in dims:
    cells = [label, rho_cell(S["flores"]["snowball"]["spearman"]["delta_full"][k1]), rho_cell(S["flores"]["stanza"]["spearman"]["delta_full"][k1])]
    if HAS_LLM:
        cells.append(rho_cell(S["flores"]["llm"]["spearman"]["delta_full"][k1]))
    cells += [rho_cell(S["flores"]["pooled"][k2]), rho_cell(S["bm25"]["_snowball"][k2]), rho_cell(S["bm25"]["_stanza"][k2])]
    if HAS_LLM:
        cells.append(rho_cell(S["bm25"]["_llm"][k2]))
    rowsB.append(" & ".join(cells) + " \\\\")
npool = S["flores"]["pooled"]["CR_morph"]["n"]
tabB = (
    "\\begin{table}[!htbp]\n\\centering\n\\caption{Which intrinsic dimensions anticipate the downstream change depends on the normalizer and on the consumer. Spearman $\\rho$ ($p$) across languages between each dimension of the morphological stage and (left) the change in dense recall@1 from unnormalized text to the full pipeline, 14 languages per method and %d points pooled, and (right) the change in BM25 MRR@10 from the orthographic stage to the full pipeline, 15 languages. Coefficients with unadjusted $p < 0.05$ are in bold. The Snowball columns reproduce Table~\\ref{tab:flores_predict} and Section~\\ref{sec:bm25}.}\n" % npool
    + "\\label{tab:r1_predict}\n" + ("\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n" if HAS_LLM else "\\small\n\\setlength{\\tabcolsep}{4pt}\n") + ("\\resizebox{\\textwidth}{!}{" if HAS_LLM else "") + "\\begin{tabular}{l" + "c" * (2 * k + 1) + "}\n\\toprule\n"
    " & \\multicolumn{%d}{c}{Dense retrieval change} & \\multicolumn{%d}{c}{BM25 change} \\\\\n" % (k + 1, k)
    + "\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}\n" % (k + 2, k + 3, 2 * k + 2)
    + "\\textbf{Dimension} & " + " & ".join(heads + ["Pooled"] + heads) + " \\\\\n\\midrule\n" + "\n".join(rowsB) + "\n\\bottomrule\n\\end{tabular}" + ("}" if HAS_LLM else "") + "\n\\end{table}\n")

# ------------------------------------------------------------------ Table C: ablation and robustness
def unit_row(name, u, new):
    i = u["intrinsic"]
    out = [name, "%.3f" % i["CR"], "%.3f" % i["IRS"], "%.3f" % i["AES_beta1"], "%.3f" % i["ANLD"], "%.2f" % i["KL"]]
    for clf in ("MultinomialNB", "LogisticRegression"):
        s = u["dsp"][clf]["42"]["stats"]
        out.append("%s (%s)" % (sgn(s["delta_mean"]), "%.4f" % s["wilcoxon_p"]))
    return " & ".join(out) + " \\\\"


blocks = [("BTSD, English", [("Snowball", OLD["btsd_en_snowball"], 0), ("spaCy", OLD["btsd_en_spacy"], 0), ("Stanza", NEW["btsd_en_stanza"], 1)]
           + ([("LLM", NEW["btsd_en_llm"], 1)] if "btsd_en_llm" in NEW else [])),
          ("BTSD, Bangla", [("BNLTK", OLD["btsd_bn_bnltk"], 0), ("BanLemma", OLD["btsd_bn_banlemma"], 0), ("Bkit", NEW["btsd_bn_bkit"], 1)]
           + ([("LLM", NEW["btsd_bn_llm"], 1)] if "btsd_bn_llm" in NEW else [])),
          ("Sentiment, Bangla", [("BNLTK", OLD["sentiment_bnltk"], 0), ("BanLemma", OLD["sentiment_banlemma"], 0), ("Bkit", NEW["sentiment_bkit"], 1)]
           + ([("LLM", NEW["sentiment_llm"], 1)] if "sentiment_llm" in NEW else []))]
rowsC = []
for title, units in blocks:
    rowsC.append("\\multicolumn{8}{l}{\\emph{%s}} \\\\" % title)
    rowsC += [unit_row(n + ("$^{\\dagger}$" if new else ""), u, new) for n, u, new in units]
    rowsC.append("\\midrule")
tabC = (
    "\\begin{table}[!htbp]\n\\centering\n\\caption{Recent normalizers ($^{\\dagger}$) beside the tools of Sections~\\ref{sec:ablation} and~\\ref{sec:robustness} on the classification corpora: intrinsic profile, and the Traditional DSP macro-F1 change with its Wilcoxon $p$-value for Multinomial Naive Bayes (MNB) and Logistic Regression (LR), 10 folds, seed 42. IRS uses DistilBERT for English and BanglaBERT for Bangla.}\n"
    "\\label{tab:r1_ablation}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccccccc}\n\\toprule\n"
    "\\textbf{Method} & \\textbf{CR} & \\textbf{IRS} & \\textbf{$AES_{\\beta=1}$} & \\textbf{ANLD} & \\textbf{KL} & \\textbf{MNB $\\Delta$ ($p$)} & \\textbf{LR $\\Delta$ ($p$)} \\\\\n\\midrule\n"
    + "\n".join(rowsC[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ------------------------------------------------------------------ Table D: XNLI with Stanza
XL = {"ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French", "ru": "Russian"}
XS = {l: json.load(open(os.path.join(ROOT, "reproducibility", "outputs", "xnli", l + ".json"))) for l in XL}
XN = {l: json.load(open(os.path.join(EXP, "xnli_stanza", "xnli", l + ".json"))) for l in XL}
rowsD = []
for l in sorted(XL, key=lambda l: XN[l]["zero_shot"]["MPD"]):
    cells = [XL[l]]
    for X in (XS, XN):
        cells += ["%.3f" % X[l]["cr_decomposition"]["CR_morph"], "%.3f" % X[l]["intrinsic"]["IRS"]]
    for X in (XS, XN):
        cells.append(sgn(X[l]["zero_shot"]["MPD"]))
    for X in (XS, XN):
        st_ = X[l]["in_language_dsp"]["42"]["stats"]
        cells.append("%s (%.4f)" % (sgn(st_["delta_mean"]), st_["wilcoxon_p"]))
    rowsD.append(" & ".join(cells) + " \\\\")
tabD = (
    "\\begin{table}[!htbp]\n\\centering\n\\caption{XNLI with the Snowball stemmer and with the Stanza lemmatizer as the morphological step: morphological compression, semantic retention, the zero-shot Model Performance Delta (classifier trained on English MultiNLI), and the in-language Traditional DSP change with its Wilcoxon $p$-value (10 folds, seed 42). Rows are sorted by the zero-shot delta under Stanza. The Snowball values are those of Section~\\ref{sec:xnli}.}\n"
    "\\label{tab:r1_xnli}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccccccc}\n\\toprule\n"
    " & \\multicolumn{2}{c}{Snowball} & \\multicolumn{2}{c}{Stanza} & \\multicolumn{2}{c}{Zero-shot MPD} & \\multicolumn{2}{c}{In-language $\\Delta$ ($p$)} \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
    "\\textbf{Language} & $CR_{\\text{morph}}$ & IRS & $CR_{\\text{morph}}$ & IRS & Snowball & Stanza & Snowball & Stanza \\\\\n\\midrule\n"
    + "\n".join(rowsD) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
mp = [XN[l]["zero_shot"]["MPD"] for l in XL]
V["xstz_mpd_min"], V["xstz_mpd_max"] = "%.4f" % max(mp), "%.4f" % min(mp)          # least and most negative
V["xstz_mpd_neg"] = str(sum(x < 0 for x in mp))
V["xstz_mpd_pmax"] = "%.4f" % max(XN[l]["zero_shot"]["mcnemar"]["p_value"] for l in XL)
V["xstz_smaller_n"] = str(sum(abs(XN[l]["zero_shot"]["MPD"]) < abs(XS[l]["zero_shot"]["MPD"]) for l in XL))
V["xsnow_mpd_min"], V["xsnow_mpd_max"] = "%.4f" % max(XS[l]["zero_shot"]["MPD"] for l in XL), "%.4f" % min(XS[l]["zero_shot"]["MPD"] for l in XL)
V["xstz_inlang_sig42"] = str(sum(XN[l]["in_language_dsp"]["42"]["stats"]["wilcoxon_p"] <= 0.05 for l in XL))
V["xsnow_inlang_sig42"] = str(sum(XS[l]["in_language_dsp"]["42"]["stats"]["wilcoxon_p"] <= 0.05 for l in XL))
allp = [(l, s_, XN[l]["in_language_dsp"][s_]["stats"]["wilcoxon_p"]) for l in XL for s_ in ("42", "7", "2024")]
V["xstz_inlang_sig_all"] = str(sum(p_ <= 0.05 for _, _, p_ in allp))
V["xstz_inlang_n_all"] = str(len(allp))
V["xstz_inlang_sig_which"] = ", ".join("%s at seed %s ($p = %.4f$)" % (XL[l], s_, p_) for l, s_, p_ in allp if p_ <= 0.05)
V["xstz_inlang_absmax"] = "%.4f" % max(abs(XN[l]["in_language_dsp"][s_]["stats"]["delta_mean"]) for l in XL for s_ in ("42", "7", "2024"))
V["xstz_de_mpd"] = "%+.4f" % XN["de"]["zero_shot"]["MPD"]
V["xstz_de_dsp"], V["xstz_de_dsp_p"] = "%+.4f" % XN["de"]["in_language_dsp"]["42"]["stats"]["delta_mean"], "%.4f" % XN["de"]["in_language_dsp"]["42"]["stats"]["wilcoxon_p"]
V["xstz_f1_min"], V["xstz_f1_max"] = "%.3f" % min(XN[l]["zero_shot"]["F1_orig"] for l in XL), "%.3f" % max(XN[l]["zero_shot"]["F1_orig"] for l in XL)

open(os.path.join(HERE, "fragments", "gen_tables_newmethods.tex"), "w", encoding="utf-8").write(tabA + "\n" + tabB + "\n" + tabC + "\n" + tabD)

# ------------------------------------------------------------------ values for the prose
q = [c for c in NAME if c != "eng_Latn"]
for m, P in (("snow", sn), ("stz", st)) + ((("llm", ll),) if HAS_LLM else ()):
    d = sorted(-100 * P[c]["delta_full"] for c in q)
    V[m + "_loss_min"], V[m + "_loss_max"], V[m + "_loss_med"] = "%.1f" % d[0], "%.1f" % d[-1], "%.1f" % statistics.median(d)
    V[m + "_dense_neg"] = str(sum(P[c]["delta_full"] < 0 for c in q))
    V[m + "_dense_sig"] = str(sum(P[c]["p_full"] < 0.05 for c in q))
for m in ["stanza"] + (["llm"] if HAS_LLM else []):
    t = "stz" if m == "stanza" else m
    b = S["bm25"]["_" + m]
    V[t + "_bm25_min"], V[t + "_bm25_max"] = "%.3f" % b["min"], "%.3f" % b["max"]
    V[t + "_bm25_pos"], V[t + "_bm25_cigain"] = str(b["n_positive"]), str(b["n_ci_gain"])
    V[t + "_bm25_pmax"] = "%.4f" % max(S["bm25"][c]["ortho->" + m]["p"] for c in NAME)
    better = [c for c in NAME if S["bm25"][c]["full->" + m]["ci"][0] > 0]
    worse = [c for c in NAME if S["bm25"][c]["full->" + m]["ci"][1] < 0]
    V[t + "_bm25_better_n"], V[t + "_bm25_worse_n"] = str(len(better)), str(len(worse))
    V[t + "_bm25_better"] = ", ".join(sorted(NAME[c] for c in better))
    V[t + "_bm25_worse"] = ", ".join(sorted(NAME[c] for c in worse))
    V[t + "_bm25_same_n"] = str(15 - len(better) - len(worse))
V["stz_cr_higher_n"] = str(sum(st[c]["CR_morph"] > sn[c]["CR_morph"] for c in NAME))
V["stz_irs_higher_n"] = str(sum(st[c]["IRS_morph"] > sn[c]["IRS_morph"] for c in NAME))
for c, tag in (("fin_Latn", "fin"), ("hun_Latn", "hun"), ("arb_Arab", "arb")):
    for m, P in (("snow", sn), ("stz", st)):
        V[f"{m}_{tag}_cr"] = "%.3f" % P[c]["CR_morph"]
        V[f"{m}_{tag}_irs"] = "%.3f" % P[c]["IRS_morph"]
        V[f"{m}_{tag}_loss"] = "%.1f" % (-100 * P[c]["delta_full"])
for m, t in (("snowball", "snow"), ("stanza", "stz")) + ((("llm", "llm"),) if HAS_LLM else ()):
    for k1, tag in (("CR_morph", "cr"), ("ANLD", "anld"), ("KL", "kl"), ("IRS", "irs")):
        d = S["flores"][m]["spearman"]["delta_full"][k1]
        V[f"{t}_rho_{tag}"], V[f"{t}_p_{tag}"] = "%+.3f" % d["rho"], ("< 0.001" if d["p"] < 0.001 else "= %.3f" % d["p"])
for k2, tag in (("CR_morph", "cr"), ("ANLD_morph", "anld"), ("KL_morph", "kl"), ("IRS_morph", "irs")):
    d = S["flores"]["pooled"][k2]
    V[f"pool_rho_{tag}"], V[f"pool_p_{tag}"] = "%+.3f" % d["rho"], ("< 0.001" if d["p"] < 0.001 else "= %.3f" % d["p"] if d["p"] >= 0.01 else "= %.4f" % d["p"])
    d = S["bm25"]["_stanza"][k2]
    V[f"stzbm_rho_{tag}"], V[f"stzbm_p_{tag}"] = "%+.3f" % d["rho"], "= %.3f" % d["p"]
    d = S["flores"]["pooled"][k2]["change_stanza"]
    V[f"chg_rho_{tag}"], V[f"chg_p_{tag}"] = "%+.3f" % d["rho"], ("= %.3f" % d["p"] if d["p"] >= 0.01 else "= %.4f" % d["p"])
V["pool_n"] = str(npool)
# English ablation and the beta sweep
sw = {m: OLD[f"btsd_en_{m}"]["intrinsic"]["AES_beta_sweep"] for m in ("porter", "snowball", "wordnet", "spacy")}
sw["stanza"] = NEW["btsd_en_stanza"]["intrinsic"]["AES_beta_sweep"]
if "btsd_en_llm" in NEW:
    sw["llm"] = NEW["btsd_en_llm"]["intrinsic"]["AES_beta_sweep"]
best = {b: max(sw, key=lambda m: sw[m][b]) for b in sw["stanza"]}
V["sweep_best"] = "; ".join("$\\beta = %s$: %s" % (b, {"spacy": "spaCy", "llm": "the LLM"}.get(m, m.capitalize())) for b, m in best.items())
for m in ("stanza", "spacy", "snowball"):
    V[f"aes1_{m}"] = "%.3f" % sw[m]["1.0"]
    V[f"aes15_{m}"] = "%.3f" % sw[m]["1.5"]
i = NEW["btsd_en_stanza"]["intrinsic"]
V["stz_en_cr"], V["stz_en_irs"], V["stz_en_anld"] = "%.3f" % i["CR"], "%.3f" % i["IRS"], "%.3f" % i["ANLD"]
s = NEW["btsd_en_stanza"]["dsp"]["MultinomialNB"]["42"]["stats"]
V["stz_en_mnb"], V["stz_en_mnb_p"] = "%+.4f" % s["delta_mean"], "%.4f" % s["wilcoxon_p"]
V["spacy_en_mnb"] = "%+.4f" % OLD["btsd_en_spacy"]["dsp"]["MultinomialNB"]["42"]["stats"]["delta_mean"]
# Bangla
for u, tag in (("btsd_bn_bkit", "bkit_btsd"), ("sentiment_bkit", "bkit_sent")):
    i = NEW[u]["intrinsic"]
    V[tag + "_cr"], V[tag + "_irs"], V[tag + "_anld"], V[tag + "_kl"] = "%.3f" % i["CR"], "%.3f" % i["IRS"], "%.3f" % i["ANLD"], "%.2f" % i["KL"]
    d = NEW[u]["dsp"]["MultinomialNB"]
    s = d["42"]["stats"]
    V[tag + "_mnb"], V[tag + "_mnb_p"] = "%+.4f" % s["delta_mean"], "%.4f" % s["wilcoxon_p"]
    V[tag + "_mnb_ci"] = "[%+.4f, %+.4f]" % tuple(s["ci95"])
    V[tag + "_mnb_dz"] = "%.2f" % s["d_z"]
    V[tag + "_mnb_pmax"] = "%.4f" % max(d[x]["stats"]["wilcoxon_p"] for x in d)
    mant, ex = ("%.1e" % max(d[x]["mcnemar"]["p_value"] for x in d)).split("e")
    V[tag + "_mnb_mcn"] = "%s\\times10^{%d}" % (mant, int(ex))
    V[tag + "_mnb_seeds"] = ", ".join("%+.4f" % d[x]["stats"]["delta_mean"] for x in ("42", "7", "2024"))
for u, tag in (("btsd_bn_banlemma", "banlemma_btsd"), ("btsd_bn_bnltk", "bnltk_btsd")):
    V[tag + "_cr"] = "%.3f" % OLD[u]["intrinsic"]["CR"]
    V[tag + "_anld"] = "%.3f" % OLD[u]["intrinsic"]["ANLD"]
b = S["bm25"]["ben_Beng"]["comparisons"]
V["bkit_bm25"], V["banlemma_bm25"], V["bnltk_bm25"] = "%+.3f" % b["tok->bkit"]["d"], "%+.3f" % b["tok->banlemma"]["d"], "%+.3f" % b["tok->bnltk"]["d"]
V["bkit_vs_banlemma_bm25"] = "%+.3f" % b["banlemma->bkit"]["d"]
V["bkit_vs_banlemma_ci"] = "[%.3f, %.3f]" % tuple(b["banlemma->bkit"]["ci"])
V["bkit_vs_banlemma_p"] = "%.4f" % b["banlemma->bkit"]["p"]
# ------------------------------------------------------------------ the LLM lemmatizer
if HAS_LLM:
    import glob
    cr = {c: ll[c]["CR_morph"] for c in NAME}
    V["llm_cr_min"], V["llm_cr_max"] = "%.3f" % min(cr.values()), "%.3f" % max(cr.values())
    V["llm_cr_lowest_n"] = str(sum(cr[c] < min(sn[c]["CR_morph"], st[c]["CR_morph"]) for c in NAME))
    V["snow_cr_min"], V["snow_cr_max"] = "%.3f" % min(sn[c]["CR_morph"] for c in NAME), "%.3f" % max(sn[c]["CR_morph"] for c in NAME)
    V["stz_cr_min"], V["stz_cr_max"] = "%.3f" % min(st[c]["CR_morph"] for c in NAME), "%.3f" % max(st[c]["CR_morph"] for c in NAME)
    V["llm_irs_min"], V["llm_irs_max"] = "%.3f" % min(ll[c]["IRS_morph"] for c in NAME), "%.3f" % max(ll[c]["IRS_morph"] for c in NAME)
    V["llm_irs_highest_n"] = str(sum(ll[c]["IRS_morph"] > max(sn[c]["IRS_morph"], st[c]["IRS_morph"]) for c in NAME))
    V["llm_loss_smaller_n"] = str(sum(ll[c]["delta_full"] >= st[c]["delta_full"] for c in q))
    V["llm_hun_loss"] = "%.1f" % (-100 * ll["hun_Latn"]["delta_full"])
    V["llm_loss_max_exhun"] = "%.1f" % max(-100 * ll[c]["delta_full"] for c in q if c != "hun_Latn")
    V["llm_morph_loss_max"] = "%.1f" % max(-100 * ll[c]["delta_morph"] for c in q)
    gain = [c for c in NAME if S["bm25"][c]["ortho->llm"]["ci"][0] > 0]
    loss = [c for c in NAME if S["bm25"][c]["ortho->llm"]["ci"][1] < 0]
    V["llm_bm25_gain"], V["llm_bm25_loss"] = ", ".join(sorted(NAME[c] for c in gain)), ", ".join(sorted(NAME[c] for c in loss))
    V["llm_bm25_loss_n"], V["llm_bm25_null_n"] = str(len(loss)), str(15 - len(gain) - len(loss))
    V["llm_bm25_below_stz_n"] = str(sum(S["bm25"][c]["llm"] < S["bm25"][c]["stanza"] for c in NAME))
    V["llm_bm25_below_snow_n"] = str(sum(S["bm25"][c]["llm"] < S["bm25"][c]["snowball"] for c in NAME))
    V["llm_dense_pmin"] = "%.3f" % min(d["p"] for d in S["flores"]["llm"]["spearman"]["delta_full"].values())
    V["llm_dense_rhomax"] = "%.2f" % max(abs(d["rho"]) for d in S["flores"]["llm"]["spearman"]["delta_full"].values())
    for k2, tag in (("CR_morph", "cr"), ("ANLD_morph", "anld"), ("KL_morph", "kl"), ("IRS_morph", "irs")):
        d = S["bm25"]["_llm"][k2]
        V[f"llmbm_rho_{tag}"], V[f"llmbm_p_{tag}"] = "%+.3f" % d["rho"], "= %.3f" % d["p"]
        d = S["flores"]["pooled"][k2]["change_llm"]
        V[f"chgllm_rho_{tag}"], V[f"chgllm_p_{tag}"] = "%+.3f" % d["rho"], ("< 0.001" if d["p"] < 0.001 else "= %.3f" % d["p"])
    H = {}
    for f in glob.glob(os.path.join(HERE, "experiments", "cache", "ud", "heldout_accuracy__*.json")):
        H.update(json.load(open(f)))
    HN = dict(NAME, ben_Beng="Bengali")
    lo, hi = min(H, key=lambda c: H[c]["lemma_accuracy"]), max(H, key=lambda c: H[c]["lemma_accuracy"])
    V["llm_acc_min"], V["llm_acc_max"], V["llm_acc_min_lang"], V["llm_acc_max_lang"] = "%.2f" % H[lo]["lemma_accuracy"], "%.2f" % H[hi]["lemma_accuracy"], HN[lo], HN[hi]
    lo, hi = min(H, key=lambda c: H[c]["accuracy_on_inflected"]), max(H, key=lambda c: H[c]["accuracy_on_inflected"])
    V["llm_accinf_min"], V["llm_accinf_max"], V["llm_accinf_min_lang"], V["llm_accinf_max_lang"] = "%.2f" % H[lo]["accuracy_on_inflected"], "%.2f" % H[hi]["accuracy_on_inflected"], HN[lo], HN[hi]
    V["llm_accinf_median"] = "%.2f" % statistics.median(v["accuracy_on_inflected"] for v in H.values())
    V["llm_acc_langs"] = str(len(H))
    V["llm_acc_matched_min"], V["llm_acc_matched_max"] = str(min(v["length_matched"] for v in H.values())), str(max(v["length_matched"] for v in H.values()))
    V["llm_acc_ben"] = "%.2f" % H["ben_Beng"]["lemma_accuracy"]
    metas = [json.load(open(f))["meta"] for f in glob.glob(os.path.join(HERE, "experiments", "cache", "norm", "llm", "*.json"))]
    fb, un = sum(m["fallback_units"] for m in metas), sum(m["n_units"] for m in metas)
    V["llm_fallback_n"], V["llm_units_n"], V["llm_fallback_pct"] = str(fb), "{:,}".format(un).replace(",", "{,}"), "%.2f" % (100.0 * fb / un)
    i = NEW["btsd_en_llm"]["intrinsic"]
    V["llm_en_cr"], V["llm_en_irs"], V["llm_en_anld"] = "%.3f" % i["CR"], "%.3f" % i["IRS"], "%.3f" % i["ANLD"]
    V["snow_en_lr"] = "%+.4f" % OLD["btsd_en_snowball"]["dsp"]["LogisticRegression"]["42"]["stats"]["delta_mean"]
    V["aes1_llm"] = "%.3f" % sw["llm"]["1.0"]
    V["llm_sweep_below_stz_n"] = str(sum(sw["llm"][b] < sw["stanza"][b] for b in sw["llm"]))
    V["sweep_n_beta"] = str(len(sw["llm"]))
    for u, tag in (("btsd_en_llm", "llm_en"), ("btsd_bn_llm", "llm_bnbtsd"), ("sentiment_llm", "llm_sent"), ("btsd_en_stanza", "stz_en")):
        for clf, ct in (("MultinomialNB", "mnb"), ("LogisticRegression", "lr")):
            d = NEW[u]["dsp"][clf]
            V[f"{tag}_{ct}"], V[f"{tag}_{ct}_p"] = "%+.4f" % d["42"]["stats"]["delta_mean"], "%.4f" % d["42"]["stats"]["wilcoxon_p"]
            V[f"{tag}_{ct}_pmax"] = "%.4f" % max(d[x]["stats"]["wilcoxon_p"] for x in d)
            V[f"{tag}_{ct}_pmin"] = "%.4f" % min(d[x]["stats"]["wilcoxon_p"] for x in d)
    for u, tag in (("btsd_bn_llm", "llm_bnbtsd"), ("sentiment_llm", "llm_sent")):
        V[tag + "_cr"], V[tag + "_irs"] = "%.3f" % NEW[u]["intrinsic"]["CR"], "%.3f" % NEW[u]["intrinsic"]["IRS"]
        ps = [(clf, x, d["stats"]["wilcoxon_p"]) for clf, sd in NEW[u]["dsp"].items() for x, d in sd.items()]
        V[tag + "_sig_n"], V[tag + "_combo_n"] = str(sum(p_ <= 0.05 for _, _, p_ in ps)), str(len(ps))
    V["llm_bn_bm25"] = "%+.3f" % b["tok->llm"]["d"]
    V["llm_bn_bm25_ci"] = "$[%+.3f, %+.3f]$" % tuple(b["tok->llm"]["ci"])
json.dump(V, open(os.path.join(HERE, "fragments", "gen_values.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k_, v_ in V.items():
    print("%-24s %s" % (k_, v_))
