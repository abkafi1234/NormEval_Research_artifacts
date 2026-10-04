"""Print the headline numbers from a completed run."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")


def load(name):
    with open(os.path.join(ART, name), encoding="utf-8") as f:
        return json.load(f)


A = load("tier_a_orthographic.json")
B = load("tier_b_intrinsic.json")
C = load("tier_c_retrieval.json")
D = load("tier_d_bangla.json")

print("=" * 78)
print("CONTROLLED CROSS-SCRIPT CONTRAST  (same language, same sentences, 2 scripts)")
print("=" * 78)
for base, d in A["controlled_multiscript"].items():
    parts = [f"{v['script']}={v['CR_ortho']:.4f}" for _, v in sorted(d["variants"].items())]
    print(f"  {base:5s}  " + "  ".join(parts) + f"   spread={d['CR_ortho_spread']:.4f}")

print()
print("=" * 78)
print("CR_ortho BY SCRIPT  (204 variants; appendix material)")
print("=" * 78)
for s, v in list(A["by_script"].items())[:14]:
    print(f"  {s:8s} n={v['n_languages']:3d}  mean={v['mean_CR_ortho']:.4f}"
          f"  range=[{v['min']:.4f}, {v['max']:.4f}]")

print()
print("=" * 78)
print("TIER B: morphological compression, 15 Snowball languages")
print("=" * 78)
rows = [(lc, d["CR_total"], d["CR_ortho"], d["CR_morph"], d["morph_share"],
         d["ANLD_morph"], d["IRS_morph"], d.get("AES_morph", float("nan")))
        for lc, d in B.items()]
rows.sort(key=lambda r: -r[3])
print(f"  {'lang':10s} {'CR_tot':>7s} {'CR_ort':>7s} {'CR_mor':>7s} {'morph%':>7s} "
      f"{'ANLD':>7s} {'IRS':>7s} {'AES':>7s}")
for lc, t, o, m, sh, a, i, ae in rows:
    print(f"  {lc:10s} {t:7.4f} {o:7.4f} {m:7.4f} {sh*100:6.1f}% {a:7.4f} {i:7.4f} {ae:7.4f}")

print()
print("=" * 78)
print("TIER C: cross-lingual retrieval vs English  (recall@1)")
print("=" * 78)
per = C["per_language"]
print(f"  {'lang':10s} {'R@1 raw':>9s} {'R@1 orth':>9s} {'delta':>9s} {'95% CI':>20s} "
      f"{'McNemar p':>11s} {'ok?':>4s}")
for lc, v in sorted(per.items(), key=lambda kv: kv[1]["delta_orth"]):
    ci = f"[{v['delta_orth_ci95'][0]:+.4f},{v['delta_orth_ci95'][1]:+.4f}]"
    print(f"  {lc:10s} {v['recall1_raw']:9.4f} {v['recall1_orth']:9.4f} "
          f"{v['delta_orth']:+9.4f} {ci:>20s} {v['mcnemar_orth']['p_value']:11.2e} "
          f"{'Y' if v['competent'] else 'N':>4s}")
print(f"\n  summary: {C['summary']}")

print()
print("  --- full pipeline (orthographic + stemming), where a stemmer exists ---")
print(f"  {'lang':10s} {'R@1 raw':>9s} {'R@1 full':>9s} {'delta':>9s} {'95% CI':>20s} {'McNemar p':>11s}")
for lc, v in sorted(per.items(), key=lambda kv: kv[1].get("delta_full", 0)):
    if "delta_full" not in v:
        continue
    ci = f"[{v['delta_full_ci95'][0]:+.4f},{v['delta_full_ci95'][1]:+.4f}]"
    print(f"  {lc:10s} {v['recall1_raw']:9.4f} {v['recall1_full']:9.4f} "
          f"{v['delta_full']:+9.4f} {ci:>20s} {v['mcnemar_full']['p_value']:11.2e}")

print()
print("=" * 78)
print("TIER D: Bangla third corpus (FLORES) vs the paper's two corpora")
print("=" * 78)
for name in ("bnltk", "banlemma"):
    d = D[name]
    print(f"  {name:9s} CR={d['CR']:.4f}  ANLD={d['ANLD']:.4f}  KL={d['KL']:.3f}  "
          f"IRS={d.get('IRS', float('nan')):.4f}  AES={d.get('AES', float('nan')):.4f}")
    r = d["CR_replicates"]
    print(f"            CR by split: dev={r['by_split']['dev']:.4f} "
          f"devtest={r['by_split']['devtest']:.4f} spread={r['split_spread']:.4f}")
print(f"  ordering: {D['_ordering']}")
print()
print("  paper's corpora for comparison:")
print("    BTSD      BanLemma CR=1.5171 ANLD=0.2206 IRS=0.9342 | BNLTK CR=1.2482 ANLD=0.1442 IRS=0.9380")
print("    Sentiment BanLemma CR=1.3740 ANLD=0.1860 IRS=0.9439 | BNLTK CR=1.2816 ANLD=0.1593 IRS=0.9379")
