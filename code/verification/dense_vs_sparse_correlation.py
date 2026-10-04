"""Rank correlation between the dense-retrieval loss and the BM25 gain across languages.

    python code/verification/dense_vs_sparse_correlation.py

Reads tables/dense_vs_sparse.csv (the 14 languages measured in both settings) and writes
reports/dense_vs_sparse_correlation.txt. Standard library only: Spearman's rho with average
ranks, two-sided p from Student's t with n-2 degrees of freedom (the scipy.stats.spearmanr
default). Cross-check on the re-run machine with scipy 1.18.1: rho = -0.1956043956, p = 0.5027500913.
"""
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r, i = [0.0] * len(v), 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return num / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def betacf(a, b, x, itmax=300, eps=1e-15):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < eps:
            break
    return h


def betai(a, b, x):
    if x <= 0 or x >= 1:
        return float(x >= 1)
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * betacf(b, a, 1 - x) / b


def spearman(x, y):
    rho = pearson(ranks(x), ranks(y))
    df = len(x) - 2
    t = rho * math.sqrt(df / (1 - rho * rho))
    return rho, betai(df / 2, 0.5, df / (df + t * t))


rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "tables", "dense_vs_sparse.csv"), encoding="utf-8"))
        if r["dense_delta_R@1_raw_to_full"]]
dense = [float(r["dense_delta_R@1_raw_to_full"]) for r in rows]
lines = [f"n = {len(rows)} languages: {', '.join(r['lang'] for r in rows)}"]
for col, label in (("bm25_delta_R@1_tok_to_full", "BM25 delta R@1 (tok -> full)"),
                   ("bm25_delta_MRR@10_ortho_to_full", "BM25 delta MRR@10 (stemming step)")):
    rho, p = spearman(dense, [float(r[col]) for r in rows])
    lines.append(f"dense delta R@1 (raw -> full) vs {label}: Spearman rho = {rho:+.10f}, two-sided p = {p:.10f}")
lines.append("Manuscript (Section 'Lexical Retrieval'): rho = -0.20, p = 0.50 (first line).")
out = os.path.join(ROOT, "reports", "dense_vs_sparse_correlation.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
