"""Does any intrinsic metric predict the downstream retrieval cost?

Rank correlation between each intrinsic measurement and the retrieval delta,
across the 14 competent languages. If intrinsic severity predicted downstream
damage we would expect strong negative correlations (more compression /
more distortion -> larger loss).
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, "artifacts/tier_b_intrinsic.json"), encoding="utf-8"))
C = json.load(open(os.path.join(HERE, "artifacts/tier_c_retrieval.json"), encoding="utf-8"))["per_language"]

langs = [l for l in C if l in B and "delta_full" in C[l]]
print(f"languages: {len(langs)}\n")

rows = []
for l in langs:
    rows.append((l, B[l]["CR_morph"], B[l]["ANLD_morph"], B[l]["KL_morph"],
                 B[l]["IRS_morph"], B[l].get("AES_morph", np.nan),
                 C[l]["delta_orth"], C[l]["delta_full"]))

names = ["CR_morph", "ANLD", "KL", "IRS", "AES"]
print(f"{'intrinsic':10s} {'vs delta_orth (Spearman)':>28s} {'vs delta_full (Spearman)':>28s}")
for i, nm in enumerate(names, start=1):
    x = np.array([r[i] for r in rows], dtype=float)
    for j, tgt in ((6, "orth"), (7, "full")):
        pass
    y_o = np.array([r[6] for r in rows], dtype=float)
    y_f = np.array([r[7] for r in rows], dtype=float)
    ro, po = spearmanr(x, y_o)
    rf, pf = spearmanr(x, y_f)
    print(f"{nm:10s} {f'rho={ro:+.3f} p={po:.3f}':>28s} {f'rho={rf:+.3f} p={pf:.3f}':>28s}")

print("\nExtremes (full pipeline):")
by_cr = sorted(rows, key=lambda r: -r[1])
by_dl = sorted(rows, key=lambda r: r[7])
print(f"  highest CR_morph : {by_cr[0][0]} CR={by_cr[0][1]:.4f} ANLD={by_cr[0][2]:.4f} delta={by_cr[0][7]:+.4f}")
print(f"  worst retrieval  : {by_dl[0][0]} CR={by_dl[0][1]:.4f} ANLD={by_dl[0][2]:.4f} delta={by_dl[0][7]:+.4f}")
print(f"  best retrieval   : {by_dl[-1][0]} CR={by_dl[-1][1]:.4f} ANLD={by_dl[-1][2]:.4f} delta={by_dl[-1][7]:+.4f}")

print("\n--- LaTeX rows: FLORES intrinsic profile (Tier B) ---")
for l, cr_m, anld, kl, irs, aes, _, _ in sorted(rows, key=lambda r: -r[1]):
    b = B[l]
    print(f"{l.replace('_','\\_')} & {b['CR_total']:.4f} & {b['CR_ortho']:.4f} & {cr_m:.4f} & "
          f"{b['morph_share']*100:.1f}\\% & {anld:.4f} & {irs:.4f} & {aes:.4f} \\\\")

print("\n--- LaTeX rows: retrieval (Tier C) ---")
for l in sorted(langs, key=lambda l: C[l]["delta_full"]):
    v = C[l]
    print(f"{l.replace('_','\\_')} & {v['recall1_raw']:.4f} & {v['recall1_orth']:.4f} & "
          f"${v['delta_orth']:+.4f}$ & {v['recall1_full']:.4f} & ${v['delta_full']:+.4f}$ \\\\")
