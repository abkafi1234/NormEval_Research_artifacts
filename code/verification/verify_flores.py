"""Cross-check every FLORES number in Paper/template3.tex against benchmark artifacts."""
import json, re
import numpy as np
from scipy.stats import spearmanr

ROOT = "/Users/kafi/Research/NormEvaluator_pypi"
ART = f"{ROOT}/normeval-main/benchmark/artifacts"
A = json.load(open(f"{ART}/tier_a_orthographic.json"))
B = json.load(open(f"{ART}/tier_b_intrinsic.json"))
C = json.load(open(f"{ART}/tier_c_retrieval.json"))["per_language"]
tex = open(f"{ROOT}/Paper/template3.tex").read()
problems = []

def chk(label, paper, actual, tol):
    if abs(paper - actual) > tol:
        problems.append(f"{label}: paper={paper} artifact={actual:.6f}")

# ---- Table flores_intrinsic
blk = tex.split(r"\label{tab:flores_intrinsic}")[1].split(r"\end{tabular}")[0]
n = 0
for line in blk.splitlines():
    m = re.match(r"(\w+)\\_(\w+) & (.+)\\\\", line.strip())
    if not m:
        continue
    lc = f"{m.group(1)}_{m.group(2)}"
    vals = [v.strip().replace(r"\%", "") for v in m.group(3).split("&")]
    b = B[lc]
    keys = [("CR_total", 1), ("CR_ortho", 1), ("CR_morph", 1), ("morph_share", 100),
            ("ANLD_morph", 1), ("IRS_morph", 1), ("AES_morph", 1)]
    for (k, scale), v in zip(keys, vals):
        tol = 0.051 if scale == 100 else 0.00006
        chk(f"flores_intrinsic {lc} {k}", float(v), b[k] * scale, tol)
    n += 1
print(f"flores_intrinsic rows checked: {n}")

# sort order claim: sorted by CR_morph
order_paper = [l for l in re.findall(r"(\w+\\_\w+) &", blk)]
order_paper = [o.replace("\\_", "_") for o in order_paper]
order_true = sorted(B, key=lambda l: -B[l]["CR_morph"])
if order_paper != order_true:
    problems.append(f"flores_intrinsic sort order differs: {order_paper} vs {order_true}")

# ---- Table flores_retrieval
blk = tex.split(r"\label{tab:flores_retrieval}")[1].split(r"\end{tabular}")[0]
n = 0
for line in blk.splitlines():
    m = re.match(r"(\w+)\\_(\w+) & (.+)\\\\", line.strip())
    if not m:
        continue
    lc = f"{m.group(1)}_{m.group(2)}"
    vals = [float(v.strip().strip("$")) for v in m.group(3).split("&")]
    c = C[lc]
    for k, v in zip(["recall1_raw", "recall1_orth", "delta_orth", "recall1_full", "delta_full"], vals):
        chk(f"retrieval {lc} {k}", v, c[k], 0.00006)
    n += 1
print(f"retrieval rows checked: {n}")

pv = [C[l][f"mcnemar_{k}"]["p_value"] for l in C for k in ("orth", "full")]
print(f"McNemar: {len(pv)} tests, all p<0.05: {all(p < 0.05 for p in pv)}, "
      f"max p={max(pv):.3g}, count p<1e-5: {sum(p < 1e-5 for p in pv)}")
print("baseline R@1 range:", min(C[l]['recall1_raw'] for l in C), max(C[l]['recall1_raw'] for l in C))
do = sorted((C[l]["delta_orth"], l) for l in C)
df = sorted((C[l]["delta_full"], l) for l in C)
print("delta_orth range:", do[0], do[-1], "| excl hun:", do[1], do[-1])
print("delta_full range:", df[0], df[-1])

# ---- Table flores_predict
langs = [l for l in C if l in B and "delta_full" in C[l]]
print(f"predictiveness languages: {len(langs)}")
feats = {"IRS": "IRS_morph", "ANLD": "ANLD_morph", "KL": "KL_morph", "CR_morph": "CR_morph", "AES": "AES_morph"}
for nm, k in feats.items():
    x = [B[l][k] for l in langs]
    ro, po = spearmanr(x, [C[l]["delta_orth"] for l in langs])
    rf, pf = spearmanr(x, [C[l]["delta_full"] for l in langs])
    print(f"  {nm:9s} orth rho={ro:+.3f} p={po:.3f} | full rho={rf:+.3f} p={pf:.3f}")

# ---- Cross-script table
blk = tex.split(r"\label{tab:crossscript}")[1].split(r"\end{tabular}")[0]
ctrl = A["controlled_multiscript"]
for line in blk.splitlines():
    m = re.match(r"([A-Za-z ]+?)\s*& (\w+) & ([\d.]+) & & (\w+) & ([\d.]+) & ([\d.]+)", line.strip())
    if not m:
        continue
    name, s1, v1, s2, v2, sp = m.groups()
    found = False
    for base, d in ctrl.items():
        scripts = {v["script"]: v["CR_ortho"] for v in d["variants"].values()}
        if s1 in scripts and s2 in scripts and abs(scripts[s1] - float(v1)) < 1e-4:
            found = True
            chk(f"crossscript {name} {s2}", float(v2), scripts[s2], 6e-5)
            chk(f"crossscript {name} spread", float(sp), d["CR_ortho_spread"], 6e-5)
    if not found:
        problems.append(f"crossscript row {name} not matched")
spreads = sorted(((d["CR_ortho_spread"], b) for b, d in ctrl.items()), reverse=True)
print("spreads:", [(b, round(s, 4)) for s, b in spreads])
print("count spread > 0.15 excluding the max:", sum(s > 0.15 for s, _ in spreads[1:]))

# ---- Script summary
blk = tex.split(r"\label{tab:appendix_script_summary}")[1].split(r"\end{tabular}")[0]
for line in blk.splitlines():
    m = re.search(r"\((\w+)\)\s*&\s*(\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+)", line)
    if not m:
        continue
    s, nl, mean, mn, mx = m.groups()
    d = A["by_script"][s]
    if int(nl) != d["n_languages"]:
        problems.append(f"script {s} n={nl} vs {d['n_languages']}")
    chk(f"script {s} mean", float(mean), d["mean_CR_ortho"], 6e-5)
    chk(f"script {s} min", float(mn), d["min"], 6e-5)
    chk(f"script {s} max", float(mx), d["max"], 6e-5)
multi = {s: d for s, d in A["by_script"].items() if d["n_languages"] >= 2}
print("scripts with >=2 variants in artifact:", sorted(multi), "| total variants:", len(A["per_language"]))

# ---- Appendix per-variant table
app = open(f"{ROOT}/Paper/appendix_cr_table.tex").read()
pairs = re.findall(r"(\w+)\\_(\w+) & ([\d.]+)", app)
print(f"appendix per-variant entries: {len(pairs)}")
for a_, b_, v in pairs:
    chk(f"appendix {a_}_{b_}", float(v), A["per_language"][f"{a_}_{b_}"]["CR_ortho"], 6e-5)
missing = set(A["per_language"]) - {f"{a}_{b}" for a, b, _ in pairs}
if missing:
    problems.append(f"appendix missing variants: {sorted(missing)}")

print("\n=== PROBLEMS ===" if problems else "\nAll FLORES numbers match artifacts.")
for p in problems:
    print(" -", p)
