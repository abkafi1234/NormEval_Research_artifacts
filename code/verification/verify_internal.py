"""Internal-consistency checks for tables that have no saved artifacts."""
import math, re
import numpy as np
from scipy.stats import t as tdist

tex = open("/Users/kafi/Research/NormEvaluator_pypi/Paper/template3.tex").read()
issues = []

def aes(irs, vrg, b=1.0):
    d = b * b * irs + vrg
    return (1 + b * b) * irs * vrg / d if d > 0 else 0.0

def num(s):
    return float(s.replace("$", "").replace("-", "-").replace("−", "-").replace("+", "").strip())

# ---- XNLI main table: VRG, AES, MPD
blk = tex.split(r"\label{tab:multilingual_nli_results}")[1].split(r"\end{tabular}")[0]
for line in blk.splitlines():
    m = re.match(r"(\w+) \((\w+)\)\s*& (.+)\\\\", line.strip())
    if not m:
        continue
    v = [num(x) for x in m.group(3).split("&")]
    cr, vrg, irs, a, anld, kl, f1o, f1n, mpd = v
    if abs((1 - 1 / cr) - vrg) > 6e-5: issues.append(f"XNLI {m.group(2)} VRG {vrg} vs {1-1/cr:.5f}")
    if abs(aes(irs, vrg) - a) > 2e-4: issues.append(f"XNLI {m.group(2)} AES {a} vs {aes(irs, vrg):.5f}")
    if abs((f1n - f1o) - mpd) > 1.1e-4: issues.append(f"XNLI {m.group(2)} MPD {mpd} vs {f1n-f1o:.5f}")

# ---- CR decomposition
blk = tex.split(r"\label{tab:cr_decomposition}")[1].split(r"\end{tabular}")[0]
xnli_rows = {}
for line in blk.splitlines():
    m = re.match(r"(\w+) \((\w+)\)\s*& (.+)\\\\", line.strip())
    if not m:
        continue
    parts = [p.strip() for p in m.group(3).split("&")]
    vr, vo, vf = (int(p.replace("{,}", "").replace(" ", "")) for p in parts[:3])
    tot, ort, mor = (float(p) for p in parts[3:6])
    share = float(parts[6].replace(r"\%", ""))
    xnli_rows[m.group(2)] = (tot, ort, mor, share)
    for lab, paper, true in [("CR_total", tot, vr / vf), ("CR_ortho", ort, vr / vo), ("CR_morph", mor, vo / vf)]:
        if abs(paper - true) > 6e-5: issues.append(f"CRdecomp {m.group(2)} {lab} {paper} vs {true:.5f}")
    true_share = 100 * math.log(vo / vf) / math.log(vr / vf)
    if abs(share - true_share) > 0.051: issues.append(f"CRdecomp {m.group(2)} share {share} vs {true_share:.2f}")
print("XNLI CR_morph ranking:", sorted(xnli_rows, key=lambda k: -xnli_rows[k][2]))
print("XNLI CR_total ranking:", sorted(xnli_rows, key=lambda k: -xnli_rows[k][0]))

# ---- BTSD / sentiment intrinsic tables
def intrinsic(label, pat):
    blk = tex.split(label)[1].split(r"\end{tabular}")[0]
    out = {}
    for line in blk.splitlines():
        m = re.search(pat, line)
        if not m:
            continue
        name = m.group(1)
        cr, vrg, irs, a, anld, kl = (float(x) for x in m.groups()[1:])
        out[name] = (cr, vrg, irs, a, anld, kl)
        if abs((1 - 1 / cr) - vrg) > 6e-5: issues.append(f"{label} {name} VRG {vrg} vs {1-1/cr:.5f}")
        if abs(aes(irs, vrg) - a) > 2e-4: issues.append(f"{label} {name} AES {a} vs {aes(irs, vrg):.5f}")
    return out
num6 = r"([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+)"
btsd = intrinsic(r"\label{tab:comparison_multirow_final}", r"& (\w+)\s*& " + num6)
sent = intrinsic(r"\label{tab:final_comparison}", r"^(\w+) \\cite\{\w+\}\s*& " + num6)
print("BTSD rows:", list(btsd), "| sentiment rows:", list(sent))
for i, k in enumerate(["CR", "VRG", "IRS", "AES", "ANLD", "KL"]):
    o1 = btsd["BanLemma"][i] > btsd["BNLTK"][i]
    o2 = sent["BanLemma"][i] > sent["BNLTK"][i]
    print(f"  BanLemma>BNLTK on {k}: BTSD={o1} sentiment={o2}" + ("   <-- ORDER FLIPS" if o1 != o2 else ""))
en = {k: v for k, v in btsd.items() if k in ("Porter", "Snowball", "WordNet", "SpaCy")}
print("English CR order:", sorted(en, key=lambda k: -en[k][0]), "| KL order:", sorted(en, key=lambda k: -en[k][5]))

# ---- AES beta sweep (paper computes from CR and IRS in the BTSD table)
blk = tex.split(r"\label{tab:aes_beta_sensitivity}")[1].split(r"\end{tabular}")[0]
betas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
for line in blk.splitlines():
    m = re.match(r"(\w+)\s*& (.+)\\\\", line.strip())
    if not m or m.group(1) not in en:
        continue
    vals = [float(re.sub(r"\\textbf\{([\d.]+)\}", r"\1", x).strip()) for x in m.group(2).split("&")]
    cr, _, irs, *_ = en[m.group(1)]
    for b, v in zip(betas, vals):
        tv = aes(irs, 1 - 1 / cr, b)
        if abs(tv - v) > 1.5e-4: issues.append(f"AES_beta {m.group(1)} b={b}: {v} vs {tv:.5f}")
for b in betas:
    sc = {k: aes(en[k][2], 1 - 1 / en[k][0], b) for k in en}
    print(f"  beta={b}: ranking {sorted(sc, key=lambda k: -sc[k])}  SpaCy-Snowball={sc['SpaCy']-sc['Snowball']:+.5f}")

# ---- Extended stats: CI and d_z consistent with mean/SD (n=10, df=9)
tcrit = tdist.ppf(0.975, 9)
for lab in [r"\label{tab:ext-xnli}", r"\label{tab:ext-btsd-en}", r"\label{tab:ext-bangla}"]:
    blk = tex.split(lab)[1].split(r"\end{tabular}")[0]
    for line in blk.splitlines():
        m = re.search(r"\$([+-][\d.]+) \\pm ([\d.]+)\$ & \$\[([+-]?[\d.]+), ([+-]?[\d.]+)\]\$ & \$([+-][\d.]+)\$ & ([\d.]+)", line)
        if not m:
            continue
        mu, sd, lo, hi, dz, pw = (float(x) for x in m.groups())
        half = tcrit * sd / math.sqrt(10)
        if abs((mu - half) - lo) > 2.5e-4 or abs((mu + half) - hi) > 2.5e-4:
            issues.append(f"{lab} CI {line.strip()[:40]}... expected [{mu-half:.4f},{mu+half:.4f}]")
        if abs(mu / sd - dz) > 0.03:
            issues.append(f"{lab} d_z {dz} vs {mu/sd:.2f} ({line.strip()[:30]})")
        # exact 10-fold Wilcoxon p-values are multiples of 1/512
        k = pw * 512
        if pw < 1 and abs(k - round(k)) > 0.06:
            issues.append(f"{lab} Wilcoxon p={pw} not attainable at n=10 exact")

# ---- all Wilcoxon p-values quoted anywhere in DSP / seed tables
for lab in [r"\label{tab:multilingual_nli_results_dsp}", r"\label{tab:final_comparison_dsp}",
            r"\label{tab:ext-seed-reversal}", r"\label{tab:ext-seed-xnli}"]:
    blk = tex.split(lab)[1].split(r"\end{tabular}")[0]
    for p in re.findall(r"\((0\.\d{4})\)|& (0\.\d{4}) \\\\", blk):
        pv = float(p[0] or p[1])
        if abs(pv * 512 - round(pv * 512)) > 0.06:
            issues.append(f"{lab} p={pv} not attainable at n=10 exact")

print("\n=== ISSUES ===" if issues else "\nNo arithmetic issues.")
for i in issues:
    print(" -", i)
