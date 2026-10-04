"""Find every decimal number in the manuscript's prose and captions and try to trace it to the outputs.

    python code/verification/check_prose_numbers.py [path/to/paper.tex ...]

Default: the manuscript and, since 2026-10-01, the supplementary material.

Tables are excluded: check_paper_numbers.py verifies them. A prose number is "traced" if some
stored value in tables/*.csv (or 100x it, for percentages and percentage points) rounds to the
printed number at the printed precision. Untraced numbers are listed with their sentence for
manual review. Some are legitimately not in the outputs (literature values, thresholds,
arithmetic facts), so this is a screening tool. Writes reports/prose_numbers.md.
"""
import csv
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATHS = sys.argv[1:] or [os.path.join(ROOT, "..", "paper", "paper.tex"),
                         os.path.join(ROOT, "..", "supplementary", "supplementary.tex")]
PATHS = [p for p in PATHS if os.path.exists(p)]
TEXPATH = PATHS[0]
tex = "\n".join(open(p, encoding="utf-8").read() for p in PATHS)
body = "\n".join(t[t.index(r"\begin{document}"):] for t in
                 (open(p, encoding="utf-8").read() for p in PATHS))
# drop tabular bodies (checked elsewhere) but keep captions; drop code listings and comments
body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", body, flags=re.S)
body = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", " ", body, flags=re.S)
body = re.sub(r"(?m)(?<!\\)%.*$", "", body)
body = re.sub(r"\\(label|ref|cite[pt]?|url|href|input|includegraphics)\*?(\[[^\]]*\])?\{[^}]*\}", " ", body)

pool = []
for path in glob.glob(os.path.join(ROOT, "tables", "*.csv")):
    for row in csv.DictReader(open(path, encoding="utf-8")):
        for v in row.values():
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            pool.append(x)
            if -1 <= x <= 1:
                pool.append(100 * x)
pool = sorted(set(pool))


def traced(s):
    d = len(s.split(".")[1])
    val = float(s)
    return any(f"{abs(x):.{d}f}" == f"{abs(val):.{d}f}" for x in pool)


rows = []
for m in re.finditer(r"(?<![\w.{])[-+]?\d+\.\d+(?![\w.])", body):
    s = m.group(0).lstrip("+-")
    a = max(body.rfind(". ", 0, m.start()), body.rfind("\n", 0, m.start()))
    b = body.find(". ", m.end())
    ctx = re.sub(r"\s+", " ", body[a + 1: (b if b > 0 else m.end() + 120) + 1])[:260]
    line = tex.count("\n", 0, tex.index(r"\begin{document}")) + body.count("\n", 0, m.start()) + 1
    rows.append((s, traced(s), ctx))

untraced = [r for r in rows if not r[1]]
lines = ["# Decimal numbers in prose and captions", "",
         f"{' + '.join('`%s`' % os.path.relpath(p, ROOT) for p in PATHS)}: {len(rows)} decimal numbers outside tables; "
         f"{len(rows) - len(untraced)} traced to a stored value; {len(untraced)} not traced (listed for review).", "",
         "| number | sentence |", "|---:|---|"]
seen = set()
for s, _, ctx in untraced:
    if (s, ctx) in seen:
        continue
    seen.add((s, ctx))
    lines.append(f"| {s} | {ctx.replace('|', '/')} |")
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
open(os.path.join(ROOT, "reports", "prose_numbers.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
