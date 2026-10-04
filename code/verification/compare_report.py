"""Compare every re-run number with the value printed in Paper/template3.tex.
Writes 08_experiments/comparison_report.md. Paper values are transcribed from template3.tex tables."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
REPORT = os.path.join(HERE, "..", "comparison_report.md")
L = ["ar", "de", "en", "es", "fr", "ru"]
NAME = {"ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French", "ru": "Russian"}

# ---------------- paper values (template3.tex)
P_XNLI = {  # CR, VRG, IRS, AES, ANLD, KL, F1o, F1n, MPD
 "ar": (1.9997, .4999, .7710, .6066, .3373, 19.1801, .6353, .5659, -.0695),
 "de": (1.6708, .4014, .7852, .5313, .2924, 18.3566, .6491, .5500, -.0991),
 "en": (1.9374, .4839, .7934, .6011, .2467, 13.1078, .6746, .6243, -.0504),
 "es": (2.1379, .5322, .7050, .6066, .3223, 17.7564, .6594, .5536, -.1058),
 "fr": (1.8158, .4493, .7188, .5530, .3134, 17.2263, .6593, .5573, -.1020),
 "ru": (2.0761, .5183, .8080, .6315, .3047, 21.4779, .6434, .5644, -.0790)}
P_CRD = {  # V_raw, V_ortho, V_full, CR_total, CR_ortho, CR_morph, share%
 "ar": (12174, 10370, 6088, 1.9997, 1.1740, 1.7034, 76.9), "ru": (13991, 11182, 6739, 2.0761, 1.2512, 1.6593, 69.3),
 "es": (9971, 7406, 4664, 2.1379, 1.3463, 1.5879, 60.9), "fr": (10143, 7647, 5586, 1.8158, 1.3264, 1.3690, 52.6),
 "de": (10469, 7998, 6266, 1.6708, 1.3090, 1.2764, 47.5), "en": (8546, 6002, 4411, 1.9374, 1.4239, 1.3607, 46.6)}
P_XDSP = {  # seed 42: delta, SD, CI lo, CI hi, dz, Wilcoxon p, McNemar p
 "ar": (-.0297, .0386, -.0573, -.0021, -.77, .0645, .0024), "de": (-.0086, .0394, -.0368, .0196, -.22, .4922, .3687),
 "en": (-.0461, .0321, -.0691, -.0232, -1.44, .0039, 3.16e-6), "es": (-.0289, .0243, -.0462, -.0115, -1.19, .0195, .0065),
 "fr": (-.0201, .0182, -.0331, -.0071, -1.11, .0137, .0532), "ru": (-.0244, .0355, -.0498, .0010, -.69, .1055, .0136)}
P_XSEED = {  # seeds 42, 7, 2024: (delta, p)
 "ar": [(-.0297, .0645), (-.0330, .0039), (-.0286, .0488)], "de": [(-.0086, .4922), (-.0067, .4922), (.0025, .6953)],
 "en": [(-.0461, .0039), (-.0434, .0020), (-.0269, .0840)], "es": [(-.0289, .0195), (-.0428, .0039), (-.0260, .0645)],
 "fr": [(-.0201, .0137), (-.0189, .0273), (-.0132, .3223)], "ru": [(-.0244, .1055), (-.0283, .0020), (-.0208, .0195)]}
P_INT = {  # CR, VRG, IRS, AES, ANLD, KL
 "btsd_en_porter": (2.0115, .5029, .8532, .6328, .2311, 14.02), "btsd_en_snowball": (2.0274, .5068, .8721, .6410, .2304, 13.12),
 "btsd_en_wordnet": (1.6820, .4055, .9430, .5671, .1229, 10.31), "btsd_en_spacy": (1.9539, .4882, .9372, .6420, .1796, 12.25),
 "btsd_bn_bnltk": (1.2482, .1988, .9380, .3281, .1442, 10.60), "btsd_bn_banlemma": (1.5171, .3408, .9342, .4994, .2206, 12.18),
 "sentiment_bnltk": (1.2816, .2197, .9379, .3560, .1593, 8.26), "sentiment_banlemma": (1.3740, .2722, .9439, .4226, .1860, 10.71)}
CLF = ["LogisticRegression", "MultinomialNB", "SVC", "RandomForestClassifier"]
P_DSP = {  # per unit, per classifier (LR, MNB, SVM, RF): delta, SD, CI lo, CI hi, dz, Wilcoxon p, McNemar p (seed 42)
 "btsd_en_porter": [(-.0198, .0171, -.0320, -.0076, -1.16, .0059, 9.21e-5), (-.0746, .0229, -.0910, -.0581, -3.25, .0020, 9.77e-24),
                    (-.0150, .0175, -.0275, -.0025, -.86, .0371, .0169), (-.0076, .0149, -.0182, .0030, -.51, .2754, .1777)],
 "btsd_en_snowball": [(-.0204, .0161, -.0319, -.0089, -1.26, .0039, 6.14e-5), (-.0729, .0250, -.0908, -.0551, -2.92, .0020, 1.04e-22),
                      (-.0146, .0179, -.0274, -.0017, -.81, .0371, .0236), (-.0124, .0152, -.0233, -.0016, -.82, .0137, .0230)],
 "btsd_en_wordnet": [(-.0253, .0156, -.0365, -.0142, -1.63, .0039, 1.51e-6), (-.0780, .0191, -.0917, -.0643, -4.08, .0020, 1.46e-26),
                     (-.0195, .0169, -.0316, -.0074, -1.16, .0020, .0008), (-.0142, .0124, -.0231, -.0054, -1.15, .0039, .0098)],
 "btsd_en_spacy": [(.0039, .0156, -.0073, .0151, .25, .4922, .4514), (-.0679, .0177, -.0805, -.0553, -3.84, .0020, 7.87e-21),
                   (-.0004, .0132, -.0098, .0090, -.03, .7695, .9533), (.0042, .0168, -.0079, .0162, .25, .3750, .4808)],
 "btsd_bn_bnltk": [(-.0124, .0112, -.0204, -.0044, -1.11, .0137, .0002), (-.0130, .0161, -.0246, -.0015, -.81, .0273, .0211),
                   (-.0130, .0087, -.0192, -.0067, -1.49, .0020, .0002), (-.0155, .0091, -.0220, -.0090, -1.71, .0020, 6.84e-6)],
 "btsd_bn_banlemma": [(.0036, .0103, -.0037, .0110, .35, .2324, .2639), (.0322, .0107, .0245, .0399, 3.00, .0020, 1.06e-6),
                      (.0026, .0127, -.0065, .0116, .20, .6250, .4199), (-.0051, .0118, -.0135, .0033, -.43, .1934, .2609)],
 "sentiment_bnltk": [(-.0094, .0137, -.0191, .0004, -.68, .0645, .0032), (-.0110, .0093, -.0177, -.0044, -1.19, .0195, .0001),
                     (.0019, .0073, -.0033, .0071, .26, .5566, .6243), (-.0003, .0063, -.0048, .0043, -.04, 1.0000, .9654)],
 "sentiment_banlemma": [(-.0114, .0154, -.0224, -.0004, -.74, .0371, .0040), (-.0151, .0103, -.0224, -.0078, -1.47, .0039, 4.04e-6),
                        (-.0012, .0100, -.0083, .0060, -.12, .6250, .4962), (.0016, .0090, -.0049, .0080, .18, .2754, .6528)]}
P_SEEDREV = {"btsd_bn_bnltk": [(-.0130, .0273), (-.0129, .0273), (-.0203, .0098)],
             "btsd_bn_banlemma": [(.0322, .0020), (.0270, .0098), (.0277, .0020)],
             "sentiment_bnltk": [(-.0110, .0195), (-.0123, .0039), (-.0099, .0039)],
             "sentiment_banlemma": [(-.0151, .0039), (-.0174, .0020), (-.0134, .0039)]}
P_BETA = {"btsd_en_porter": [.8196, .7489, .6821, .6328, .5756, .5479, .5153], "btsd_en_snowball": [.8366, .7622, .6924, .6410, .5817, .5531, .5196],
          "btsd_en_wordnet": [.8748, .7454, .6383, .5671, .4917, .4576, .4195], "btsd_en_spacy": [.8891, .7916, .7041, .6420, .5726, .5399, .5024]}
BETAS = ["0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "4.0"]

rows, mism = [], []


def cmp(table, cell, paper, rerun, tol):
    ok = rerun is not None and abs(paper - rerun) <= tol
    rows.append((table, cell, paper, rerun, ok))
    if not ok:
        mism.append((table, cell, paper, rerun))
    return ok


def fmt(v):
    if v is None:
        return "—"
    return f"{v:.3g}" if (v != 0 and abs(v) < 1e-3) else f"{v:.4f}"


X = {l: json.load(open(os.path.join(OUT, "xnli", f"{l}.json"))) for l in L}
A = json.load(open(os.path.join(OUT, "ablation", "summary.json")))["units"]

for l in L:
    x, p = X[l], P_XNLI[l]
    i, z = x["intrinsic"], x["zero_shot"]
    for k, (name, val, tol) in enumerate([("CR", i["CR"], 6e-5), ("VRG", i["VRG"], 1.1e-4), ("IRS", i["IRS"], 6e-5), ("AES", i["AES_beta1"], 2e-4),
                                           ("ANLD", i["ANLD"], 6e-5), ("KL", i["KL"], 6e-5), ("F1_orig", z["F1_orig"], 6e-5),
                                           ("F1_norm", z["F1_norm"], 6e-5), ("MPD", z["MPD"], 1.1e-4)]):
        cmp("XNLI main", f"{NAME[l]} {name}", p[k], val, tol)
    d, pc = x["cr_decomposition"], P_CRD[l]
    for k, (name, val, tol) in enumerate([("V_raw", d["V_raw"], 0), ("V_ortho", d["V_ortho"], 0), ("V_full", d["V_full"], 0),
                                           ("CR_total", d["CR_total"], 6e-5), ("CR_ortho", d["CR_ortho"], 6e-5), ("CR_morph", d["CR_morph"], 6e-5),
                                           ("morph share %", 100 * d["morph_share"], 0.051)]):
        cmp("CR decomposition", f"{NAME[l]} {name}", pc[k], val, tol)
    s42 = x["in_language_dsp"]["42"]
    st, mc, px = s42["stats"], s42["mcnemar"], P_XDSP[l]
    for k, (name, val, tol) in enumerate([("delta", st["delta_mean"], 6e-5), ("SD", st["delta_sd"], 6e-5), ("CI lo", st["ci95"][0], 6e-5),
                                           ("CI hi", st["ci95"][1], 6e-5), ("d_z", st["d_z"], 0.0051), ("Wilcoxon p", st["wilcoxon_p"], 6e-5)]):
        cmp("XNLI in-language DSP (seed 42)", f"{NAME[l]} {name}", px[k], val, tol)
    cmp("XNLI in-language DSP (seed 42)", f"{NAME[l]} McNemar p", px[6], mc["p_value"], max(6e-5, 0.02 * px[6]))
    for si, seed in enumerate(["42", "7", "2024"]):
        stt = x["in_language_dsp"][seed]["stats"]
        cmp("XNLI seeds", f"{NAME[l]} seed {seed} delta", P_XSEED[l][si][0], stt["delta_mean"], 6e-5)
        cmp("XNLI seeds", f"{NAME[l]} seed {seed} p", P_XSEED[l][si][1], stt["wilcoxon_p"], 6e-5)

for u, p in P_INT.items():
    i = A[u]["intrinsic"]
    for k, (name, val, tol) in enumerate([("CR", i["CR"], 6e-5), ("VRG", i["VRG"], 6e-5), ("IRS", i["IRS"], 6e-5),
                                           ("AES", i["AES_beta1"], 6e-5), ("ANLD", i["ANLD"], 6e-5), ("KL", i["KL"], 0.0051)]):
        cmp("Intrinsic (BTSD/sentiment)", f"{u} {name}", p[k], val, tol)
for u, per in P_DSP.items():
    for ci, clf in enumerate(CLF):
        r = A[u]["dsp"][clf]["42"]
        st, mc, p = r["stats"], r["mcnemar"], per[ci]
        for k, (name, val, tol) in enumerate([("delta", st["delta_mean"], 6e-5), ("SD", st["delta_sd"], 6e-5), ("CI lo", st["ci95"][0], 6e-5),
                                               ("CI hi", st["ci95"][1], 6e-5), ("d_z", st["d_z"], 0.0051), ("Wilcoxon p", st["wilcoxon_p"], 6e-5)]):
            cmp("DSP detail (BTSD/sentiment)", f"{u} {clf} {name}", p[k], val, tol)
        cmp("DSP detail (BTSD/sentiment)", f"{u} {clf} McNemar p", p[6], mc["p_value"], max(6e-5, 0.02 * p[6]))
for u, per in P_SEEDREV.items():
    for si, seed in enumerate(["42", "7", "2024"]):
        st = A[u]["dsp"]["MultinomialNB"][seed]["stats"]
        cmp("Seed reversal (MNB)", f"{u} seed {seed} delta", per[si][0], st["delta_mean"], 6e-5)
        cmp("Seed reversal (MNB)", f"{u} seed {seed} p", per[si][1], st["wilcoxon_p"], 6e-5)
for u, vals in P_BETA.items():
    sweep = A[u]["intrinsic"]["AES_beta_sweep"]
    for b, pv in zip(BETAS, vals):
        cmp("AES_beta sweep", f"{u} beta={b}", pv, sweep[b], 6e-5)

# ---------------- headline checks
H = []
rev = all(A["btsd_bn_banlemma"]["dsp"]["MultinomialNB"][s]["stats"]["delta_mean"] > 0 and A["btsd_bn_banlemma"]["dsp"]["MultinomialNB"][s]["stats"]["wilcoxon_p"] <= .05
          and A["sentiment_banlemma"]["dsp"]["MultinomialNB"][s]["stats"]["delta_mean"] < 0 and A["sentiment_banlemma"]["dsp"]["MultinomialNB"][s]["stats"]["wilcoxon_p"] <= .05
          for s in ("42", "7", "2024"))
H.append(("Bangla reversal: BanLemma MNB significantly + on BTSD and − on sentiment, all 3 seeds", rev))
mpd_neg = all(X[l]["zero_shot"]["MPD"] < 0 for l in L)
H.append(("XNLI zero-shot MPD negative in all six languages", mpd_neg))
de = X["de"]
de_ok = (de["zero_shot"]["MPD"] < -0.05 and all(de["in_language_dsp"][s]["stats"]["wilcoxon_p"] > .05 for s in ("42", "7", "2024"))
         and de["in_language_dsp"]["42"]["mcnemar"]["p_value"] > .05)
H.append(("German: large zero-shot cost, but in-language effect non-significant in all 3 seeds and McNemar n.s.", de_ok))
both_ns = [l for l in L if X[l]["in_language_dsp"]["42"]["stats"]["wilcoxon_p"] > .05 and X[l]["in_language_dsp"]["42"]["mcnemar"]["p_value"] > .05]
H.append((f"German is the only language where both tests are n.s. at seed 42 (re-run: {both_ns})", both_ns == ["de"]))
sw = {u: A[u]["intrinsic"]["AES_beta_sweep"] for u in P_BETA}
top = {b: max(sw, key=lambda u: sw[u][b]) for b in BETAS}
beta_ok = all(top[b] == "btsd_en_spacy" for b in ("0.25", "0.5", "0.75", "1.0")) and all(top[b] == "btsd_en_snowball" for b in ("1.5", "2.0", "4.0"))
H.append(("English AES_beta: SpaCy first for beta<=1, Snowball first for beta>=1.5", beta_ok))
sig42 = [l for l in L if X[l]["in_language_dsp"]["42"]["stats"]["wilcoxon_p"] <= .05]
H.append((f"'Three of six languages significant at seed 42 (en, es, fr)' (re-run: {sig42})", sorted(sig42) == ["en", "es", "fr"]))
mc_sig = [l for l in L if X[l]["in_language_dsp"]["42"]["mcnemar"]["p_value"] <= .05]
H.append((f"McNemar significant at seed 42 for ar, en, es, ru; fr borderline (re-run significant: {mc_sig})", sorted(mc_sig) == ["ar", "en", "es", "ru"]))

# ---------------- write report
by_table = {}
for t, c, p, r, ok in rows:
    by_table.setdefault(t, [0, 0])
    by_table[t][0] += 1
    by_table[t][1] += ok
lines = ["# Re-run vs. template3: comparison report", "",
         "Generated by `code/compare_report.py` from `outputs/`. Tolerances: rounding of the printed value "
         "(±0.5 in the last printed digit); McNemar p within 2% relative.", "",
         "## Headline findings", "", "| Finding | Replicates? |", "|---|---|"]
lines += [f"| {h} | {'**yes**' if ok else '**NO**'} |" for h, ok in H]
lines += ["", "## Cell-level agreement by table", "", "| Table | Cells | Match | Mismatch |", "|---|---|---|---|"]
lines += [f"| {t} | {n} | {m} | {n - m} |" for t, (n, m) in by_table.items()]
tot, ok_tot = len(rows), sum(r[4] for r in rows)
lines += [f"| **Total** | **{tot}** | **{ok_tot}** | **{tot - ok_tot}** |", "", "## Every mismatch", "",
          "| Table | Cell | Paper | Re-run | Difference |", "|---|---|---|---|---|"]
lines += [f"| {t} | {c} | {fmt(p)} | {fmt(r)} | {fmt(None if r is None else r - p)} |" for t, c, p, r in mism]
open(REPORT, "w").write("\n".join(lines) + "\n")
print(f"cells compared: {tot}, matching: {ok_tot}, mismatching: {tot - ok_tot}")
for h, ok in H:
    print(("REPLICATES  " if ok else "DOES NOT    ") + h)
print("\nmismatches by table:", {t: n - m for t, (n, m) in by_table.items() if n - m})
