"""Revision R1: summary of the new-method results (reads outputs/, prints tables, writes outputs/summary_r1.json)."""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
methods = [m for m in ("snowball", "stanza", "llm") if os.path.exists(os.path.join(OUT, "flores", m + ".json"))]
F = {m: json.load(open(os.path.join(OUT, "flores", m + ".json"))) for m in methods}
S = {"flores": {}, "bm25": {}}
q = F["snowball"]["predictiveness"]["languages"]
print("== FLORES-200: dense recall@1 change, full pipeline vs raw (points), and intrinsic dimensions")
for m in methods:
    P = F[m]["per_language"]
    d = [100 * P[c]["retrieval"]["delta_full"] for c in q]
    sig = sum(P[c]["retrieval"]["mcnemar_full"]["p_value"] < 0.05 for c in q)
    neg = sum(x < 0 for x in d)
    S["flores"][m] = {"delta_full_min": min(d), "delta_full_max": max(d), "delta_full_median": float(np.median(d)),
                      "n_negative": neg, "n_significant": sig, "spearman": F[m]["predictiveness"]["spearman"],
                      "per_language": {c: {k: P[c][k] for k in ("CR_total", "CR_ortho", "CR_morph", "ANLD_morph", "KL_morph", "IRS_morph", "AES_morph")}
                                       | ({"delta_full": P[c]["retrieval"]["delta_full"], "delta_morph": P[c]["retrieval"]["delta_morph"],
                                           "recall1_raw": P[c]["retrieval"]["recall1_raw"], "recall1_full": P[c]["retrieval"]["recall1_full"],
                                           "p_full": P[c]["retrieval"]["mcnemar_full"]["p_value"]} if "retrieval" in P[c] else {}) for c in P}}
    print(f"-- {m}: loss range {min(d):.1f} to {max(d):.1f} points, median {np.median(d):.1f}; negative in {neg}/14, McNemar p<0.05 in {sig}/14")
    for t in ("delta_full", "delta_morph"):
        print("   Spearman vs", t, {k: (round(v["rho"], 3), round(v["p"], 4)) for k, v in F[m]["predictiveness"]["spearman"][t].items()})
print("\nlang       " + "  ".join(f"{m[:5]:>5s}:dF   CRm   ANLD    KL   IRS" for m in methods))
for c in sorted(F["snowball"]["per_language"]):
    row = c + "  "
    for m in methods:
        v = F[m]["per_language"][c]
        dF = 100 * v["retrieval"]["delta_full"] if "retrieval" in v else float("nan")
        row += f"  {dF:8.1f} {v['CR_morph']:.3f} {v['ANLD_morph']:.3f} {v['KL_morph']:5.1f} {v['IRS_morph']:.3f}"
    print(row)
if len(methods) > 1:
    print("\n== pooled over methods and languages; and within-language change from Snowball")
    S["flores"]["pooled"] = {}
    for key in ("CR_morph", "ANLD_morph", "KL_morph", "IRS_morph", "AES_morph"):
        x = [F[m]["per_language"][c][key] for m in methods for c in q]
        y = [F[m]["per_language"][c]["retrieval"]["delta_full"] for m in methods for c in q]
        r, p = spearmanr(x, y)
        line = f"{key}: pooled n={len(x)} rho={r:+.3f} p={p:.4f}"
        S["flores"]["pooled"][key] = {"n": len(x), "rho": float(r), "p": float(p)}
        for m in methods[1:]:
            dx = [F[m]["per_language"][c][key] - F["snowball"]["per_language"][c][key] for c in q]
            dy = [F[m]["per_language"][c]["retrieval"]["delta_full"] - F["snowball"]["per_language"][c]["retrieval"]["delta_full"] for c in q]
            r2, p2 = spearmanr(dx, dy)
            line += f" | change {m}-snowball rho={r2:+.3f} p={p2:.4f}"
            S["flores"]["pooled"][key][f"change_{m}"] = {"rho": float(r2), "p": float(p2)}
        print(line)
bp = os.path.join(OUT, "bm25", "summary.json")
if os.path.exists(bp):
    B = json.load(open(bp))
    print("\n== Belebele BM25, MRR@10")
    conds = [c for c in ("stanza", "llm") if c in next(iter(B["table"].values()))]
    for l, t in B["table"].items():
        c = B["comparisons"][l]
        if l == "ben_Beng":
            print(l, {k: round(v["MRR@10"], 4) for k, v in t.items()})
            for k, v in c.items():
                print("    ", k, f"{v['delta_MRR@10']:+.4f}", [round(x, 3) for x in v["delta_MRR@10_ci95"]], "McNemar p=%.2g" % v["mcnemar_R@1"]["p_value"])
            S["bm25"][l] = {"table": t, "comparisons": {k: {"d": v["delta_MRR@10"], "ci": v["delta_MRR@10_ci95"], "p": v["mcnemar_R@1"]["p_value"]} for k, v in c.items()}}
            continue
        row = f"{l} ortho={t['ortho']['MRR@10']:.4f} snowball={t['full']['MRR@10']:.4f}"
        S["bm25"][l] = {"ortho": t["ortho"]["MRR@10"], "snowball": t["full"]["MRR@10"], "ortho->full": c["ortho->full"]["delta_MRR@10"]}
        for m in conds:
            a, s = c[f"ortho->{m}"], c.get(f"full->{m}")
            row += f" | {m}={t[m]['MRR@10']:.4f} ortho->{m} {a['delta_MRR@10']:+.4f} [{a['delta_MRR@10_ci95'][0]:+.3f},{a['delta_MRR@10_ci95'][1]:+.3f}] p={a['mcnemar_R@1']['p_value']:.3g}"
            S["bm25"][l][m] = t[m]["MRR@10"]
            S["bm25"][l][f"ortho->{m}"] = {"d": a["delta_MRR@10"], "ci": a["delta_MRR@10_ci95"], "p": a["mcnemar_R@1"]["p_value"],
                                           "dR1": a["delta_R@1"]}
            if s:
                row += f"; vs snowball {s['delta_MRR@10']:+.4f} [{s['delta_MRR@10_ci95'][0]:+.3f},{s['delta_MRR@10_ci95'][1]:+.3f}]"
                S["bm25"][l][f"full->{m}"] = {"d": s["delta_MRR@10"], "ci": s["delta_MRR@10_ci95"], "p": s["mcnemar_R@1"]["p_value"]}
        print(row)
    L = [l for l in B["table"] if l != "ben_Beng"]
    ds = [B["comparisons"][l]["ortho->full"]["delta_MRR@10"] for l in L]
    S["bm25"]["_snowball"] = {"min": min(ds), "max": max(ds)}
    for key in ("CR_morph", "ANLD_morph", "KL_morph", "IRS_morph", "AES_morph"):
        r, p = spearmanr([F["snowball"]["per_language"][l][key] for l in L], ds)
        S["bm25"]["_snowball"][key] = {"rho": float(r), "p": float(p)}
    print("-- snowball BM25 gain vs dimensions:", {k: (round(v["rho"], 3), round(v["p"], 3)) for k, v in S["bm25"]["_snowball"].items() if isinstance(v, dict)})
    for m in conds:
        g = sum(S["bm25"][l][f"ortho->{m}"]["ci"][0] > 0 for l in L)
        pos = sum(S["bm25"][l][f"ortho->{m}"]["d"] > 0 for l in L)
        ds = [S["bm25"][l][f"ortho->{m}"]["d"] for l in L]
        print(f"-- ortho->{m}: positive in {pos}/15, CI excludes zero (gain) in {g}/15, range {min(ds):+.4f} to {max(ds):+.4f}")
        S["bm25"][f"_{m}"] = {"n_positive": pos, "n_ci_gain": g, "min": min(ds), "max": max(ds)}
        # does any intrinsic dimension anticipate the BM25 gain for this method?
        if m in F:
            for key in ("CR_morph", "ANLD_morph", "KL_morph", "IRS_morph", "AES_morph"):
                r, p = spearmanr([F[m]["per_language"][l][key] for l in L], ds)
                print(f"     {key} vs BM25 gain: rho={r:+.3f} p={p:.3f}")
                S["bm25"][f"_{m}"][key] = {"rho": float(r), "p": float(p)}
ab = os.path.join(OUT, "ablation", "summary.json")
if os.path.exists(ab):
    A = json.load(open(ab))["units"]
    print("\n== Ablation and robustness (seed 42; MNB and LR shown)")
    S["ablation"] = {}
    for u, v in A.items():
        i = v["intrinsic"]
        line = f"{u}: CR={i['CR']:.4f} IRS={i['IRS']:.4f} AES={i['AES_beta1']:.4f} ANLD={i['ANLD']:.4f} KL={i['KL']:.2f}"
        S["ablation"][u] = {"intrinsic": {k: i[k] for k in ("CR", "IRS", "AES_beta1", "ANLD", "KL")} | ({"IRS_minilm": i["IRS_minilm"]} if "IRS_minilm" in i else {}), "dsp": {}}
        for clf, seeds in v["dsp"].items():
            S["ablation"][u]["dsp"][clf] = {s: {"delta": r["stats"]["delta_mean"], "sd": r["stats"]["delta_sd"], "ci": r["stats"]["ci95"],
                                                 "dz": r["stats"]["d_z"], "p": r["stats"]["wilcoxon_p"], "mcnemar_p": r["mcnemar"]["p_value"],
                                                 "f1_orig": r["stats"]["mean_orig"]} for s, r in seeds.items()}
        print(line)
        for clf in ("MultinomialNB", "LogisticRegression", "SVC", "RandomForestClassifier"):
            if clf in v["dsp"]:
                print("    " + clf + ": " + "  ".join(f"seed{s}: {r['stats']['delta_mean']:+.4f} (p={r['stats']['wilcoxon_p']:.4f}, McN p={r['mcnemar']['p_value']:.2g})" for s, r in v["dsp"][clf].items()))
json.dump(S, open(os.path.join(OUT, "summary_r1.json"), "w"), indent=1)
