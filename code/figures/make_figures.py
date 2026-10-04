"""Draw the manuscript's seven figures from tables/*.csv.

    python code/figures/make_figures.py [output_dir]

No new analysis. Every plotted value is read from tables/*.csv, the same numbers the
manuscript's tables print. Two derived quantities are recomputed and checked:
  - the dense-retrieval Spearman coefficients in Figure 5b are recomputed from tiers B and C, as
    check_paper_numbers.py does, and asserted to equal reports/flores_predictiveness_output.txt;
  - the AES_beta=1 iso-curves in Figure 6 follow from the definition of AES (Section 3).
Figure 1 is conceptual: its text condenses Table tab:matrix and the data section.

Requires matplotlib (3.9.4 used) and a LaTeX installation: text is set with text.usetex, so
the figure fonts match the manuscript. Writes <output_dir>/fig{1..7}_*.pdf (default
../paper/figures) and reports/figure_values.md (every plotted value, with its source).
"""
import csv
import os
import re
import shutil
import sys

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
if shutil.which("latex") is None:
    sys.exit("LaTeX not found: the figures are typeset with text.usetex to match the manuscript.")

# ---------------------------------------------------------------- style
TW = 6.30  # manuscript \textwidth (16 cm) in inches
INK, MUTED, HAIR, PANEL = "#222222", "#6B6B6B", "#D0D0D0", "#F4F5F7"
# Okabe-Ito palette; one fixed colour per role across all figures.
ORTHO, MORPH = "#56B4E9", "#009E73"
DENSE, LEX = "#D55E00", "#0072B2"
BNLTK, BANLEMMA = "#E69F00", "#CC79A7"
EN_ALGO = {"porter": "#7F7F7F", "snowball": LEX, "wordnet": MORPH, "spacy": DENSE}

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}",
    "font.family": "serif", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.6, "axes.edgecolor": "#555555", "axes.labelcolor": INK,
    "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

LANG = {"arb": "Arabic", "dan": "Danish", "deu": "German", "eng": "English", "fin": "Finnish",
        "fra": "French", "hun": "Hungarian", "ita": "Italian", "nld": "Dutch",
        "nob": "Norwegian", "por": "Portuguese", "ron": "Romanian", "rus": "Russian",
        "spa": "Spanish", "swe": "Swedish",
        "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French",
        "ru": "Russian"}
SCRIPT = {"Latn": "Latin", "Arab": "Arabic", "Cyrl": "Cyrillic", "Deva": "Devanagari",
          "Beng": "Bengali", "Ethi": "Ethiopic", "Tibt": "Tibetan", "Hebr": "Hebrew",
          "Mymr": "Myanmar", "Tfng": "Tifinagh", "Hant": "Han (Trad.)", "Grek": "Greek",
          "Gujr": "Gujarati", "Armn": "Armenian", "Jpan": "Japanese", "Knda": "Kannada",
          "Geor": "Georgian", "Khmr": "Khmer", "Hang": "Hangul", "Laoo": "Lao",
          "Mlym": "Malayalam", "Orya": "Odia", "Guru": "Gurmukhi", "Olck": "Ol Chiki",
          "Sinh": "Sinhala", "Taml": "Tamil", "Telu": "Telugu", "Thai": "Thai",
          "Hans": "Han (Simpl.)"}
MULTISCRIPT = {"ace": "Acehnese", "arb": "Standard Arabic", "bjn": "Banjar", "kas": "Kashmiri",
               "knc": "Central Kanuri", "min": "Minangkabau", "taq": "Tamasheq",
               "zho": "Chinese"}

VALUES = []  # (figure, item, value, source) for reports/figure_values.md


def rows(name):
    with open(os.path.join(ROOT, "tables", name + ".csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def log(fig, item, value, source):
    VALUES.append((fig, item, value, source))
    return value


def tex(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def panel_label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, r"\textbf{%s}" % letter, transform=ax.transAxes, fontsize=10,
            ha="right", va="bottom", color=INK)


def hairline(ax, value, axis="x", **kw):
    kw = {"color": "#8A8A8A", "lw": 0.6, "zorder": 0, **kw}
    (ax.axvline if axis == "x" else ax.axhline)(value, **kw)


def spread(ys, gap):
    """Nudge label positions apart so no two are closer than gap (order preserved)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        if out[i] - out[j] < gap:
            out[i] = out[j] + gap
    return out


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.join(ROOT, "..")))


def f(x):
    return float(x)


# ================================================================ Figure 1 (conceptual)
def box(ax, x, y, w, h, fc=PANEL, ec="#C3C7CC", lw=0.6, r=0.05, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r,
                                fc=fc, ec=ec, lw=lw, **kw))


def arrow(ax, x0, y0, x1, y1, color="#555555"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=color, shrinkA=0, shrinkB=0))


def figure1():
    W, H = TW, 5.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    x0, gap = 0.10, 0.08

    # (a) framework ------------------------------------------------------
    ax.text(0.08, H - 0.06, r"\textbf{a}", fontsize=10, va="top")
    ax.text(0.30, H - 0.075, "The framework: six dimensions, each paired with how it can "
            "mislead and what the protocol requires", fontsize=8.3, va="top")

    fy, fh = 4.56, 0.44
    flow = [(0.25, 1.10, "Original text", None),
            (1.80, 2.70, "Normalizer", "cleanup, stemming or lemmatization"),
            (4.95, 1.10, "Normalized text", None)]
    for x, w, t, sub in flow:
        box(ax, x, fy, w, fh, fc="white", ec="#8A9099", lw=0.7)
        if sub:
            ax.text(x + w / 2, fy + fh * 0.66, r"\textbf{%s}" % t, ha="center", va="center",
                    fontsize=7.8)
            ax.text(x + w / 2, fy + fh * 0.28, sub, ha="center", va="center", fontsize=6.3,
                    color=MUTED)
        else:
            ax.text(x + w / 2, fy + fh / 2, r"\textbf{%s}" % t, ha="center", va="center",
                    fontsize=7.8)
    arrow(ax, 1.35, fy + fh / 2, 1.80, fy + fh / 2)
    arrow(ax, 4.50, fy + fh / 2, 4.95, fy + fh / 2)
    cy = fy - 0.13
    ax.plot([0.80, 0.80, 5.50, 5.50], [fy, cy, cy, fy], color="#8A9099", lw=0.7)
    arrow(ax, W / 2, cy, W / 2, cy - 0.22, color="#8A9099")
    ax.text(W / 2 + 0.07, cy - 0.11, "both texts, the same data and folds", va="center",
            fontsize=6.3, color=MUTED)

    tiles = [
        ("CR", "compression",
         "vocabulary\nreduction,\n$|V_o|\\,/\\,|V_n|$",
         "credits case and\ndiacritic cleanup\nto the normalizer",
         "decompose into\northo $\\times$ morph"),
        ("KL", "distributional shift",
         "shift in unigram\nfrequencies",
         "size depends on\nthe corpus\nvocabulary",
         "compare only by\nrank, within one\ncorpus"),
        ("ANLD", "form distortion",
         "character-level\nedits per aligned\nword",
         "misaligns unless\nwords are aligned;\nno safe threshold",
         "align first; a\ndiagnostic, not\na pass/fail gate"),
        ("IRS", "semantic retention",
         "embedding\nsimilarity of\nsentences",
         "depends on the\nencoder",
         "name the encoder;\nnever compare\nacross encoders"),
        (r"AES$_\beta$", "trade-off",
         "$F_\\beta$-style score\nof VRG and IRS",
         "a fixed weighting\nhides a user\npreference",
         "report a sweep\nover $\\beta$"),
        ("MPD / DSP", "downstream effect",
         "change in task\nperformance",
         "near-chance\nbaseline; too few\nfolds for a test",
         "competent\nbaseline, 10\nfolds, effect\nsizes, seeds"),
    ]
    tw = (W - 2 * x0 - 5 * gap) / 6
    ty, th = 1.95, 2.0
    groups = [(0, 1, "corpus level"), (2, 2, "word level"), (3, 3, "sentence level"),
              (4, 4, "trade-off"), (5, 5, "downstream")]
    gy = ty + th + 0.10
    for a, b, label in groups:
        xa, xb = x0 + a * (tw + gap) + 0.03, x0 + b * (tw + gap) + tw - 0.03
        ax.plot([xa, xb], [gy, gy], color=LEX, lw=1.0, solid_capstyle="butt")
        ax.text((xa + xb) / 2, gy + 0.04, r"\textit{%s}" % label, ha="center", va="bottom",
                fontsize=6.8, color=LEX)
    lh = 0.094
    for i, (name, sub, meas, mis, req) in enumerate(tiles):
        x = x0 + i * (tw + gap)
        box(ax, x, ty, tw, th)
        box(ax, x, ty + th - 0.40, tw, 0.40, fc="#E4ECF4", ec="#C3C7CC")
        ax.text(x + tw / 2, ty + th - 0.14, r"\textbf{%s}" % name, ha="center", va="center",
                fontsize=8.5)
        ax.text(x + tw / 2, ty + th - 0.31, sub, ha="center", va="center", fontsize=6.2,
                color=MUTED)
        y = ty + th - 0.49
        for head, body in (("measures", meas), ("can mislead", mis), ("requires", req)):
            ax.text(x + 0.06, y, r"\textsc{%s}" % head, fontsize=5.7, color=MUTED, va="top")
            ax.text(x + 0.06, y - 0.115, body, fontsize=5.8, va="top", linespacing=1.12)
            y -= 0.115 + lh * (body.count("\n") + 1) + 0.075

    oy, oh = 1.38, 0.42
    for i in range(6):
        cx = x0 + i * (tw + gap) + tw / 2
        arrow(ax, cx, ty, cx, oy + oh, color="#8A9099")
    box(ax, x0, oy, W - 2 * x0, oh, fc="#E4ECF4", ec=LEX, lw=0.8, r=0.08)
    ax.text(W / 2, oy + oh * 0.68, r"\textbf{Output: a profile, not a ranking}",
            ha="center", va="center", fontsize=7.6)
    ax.text(W / 2, oy + oh * 0.30, r"The six are reported together; the user weighs the "
            r"trade-offs ($\beta$) for their own language, data and task.",
            ha="center", va="center", fontsize=6.7)

    # (b) study design ---------------------------------------------------
    by = 1.22
    ax.text(0.08, by, r"\textbf{b}", fontsize=10, va="top")
    ax.text(0.30, by - 0.015, "Study design: three stages, five data sets", fontsize=8.3,
            va="top")
    blocks = [
        ("XNLI", "6 languages; 2{,}490 pairs each\nzero-shot transfer\n"
         "in-language DSP, 10 folds", ["RQ1", "RQ2", "RQ3", "RQ4"]),
        ("FLORES-200", "204 language--script variants\n15 with a stemmer\n"
         "dense retrieval, 14 languages", ["RQ1", "RQ2", "RQ3", "RQ4"]),
        ("Belebele", "16 languages; 900 questions\nover 488 passages each\n"
         "BM25 lexical retrieval", ["RQ1", "RQ4"]),
        ("Bangla corpora", "BTSD: 3{,}793 sentences\nsentiment: 9{,}154 texts\n"
         "6 normalizers, 4 classifiers\n3 seeds", ["RQ1", "RQ2", "RQ4"]),
    ]
    bw = (W - 2 * x0 - 3 * gap) / 4
    bh, byy = 0.80, 0.02
    stages = [(0, 0, "stage 1: breadth"), (1, 2, "stage 2: content held fixed"),
              (3, 3, "stage 3: depth")]
    sy = byy + bh + 0.08
    for a, b, label in stages:
        xa, xb = x0 + a * (bw + gap) + 0.03, x0 + b * (bw + gap) + bw - 0.03
        ax.plot([xa, xb], [sy, sy], color=MORPH, lw=1.0, solid_capstyle="butt")
        ax.text((xa + xb) / 2, sy + 0.04, r"\textit{%s}" % label, ha="center", va="bottom",
                fontsize=6.8, color=MORPH)
    for i, (name, body, rqs) in enumerate(blocks):
        x = x0 + i * (bw + gap)
        box(ax, x, byy, bw, bh)
        ax.text(x + 0.08, byy + bh - 0.07, r"\textbf{%s}" % name, fontsize=7.6, va="top")
        ax.text(x + 0.08, byy + bh - 0.23, body, fontsize=6.1, va="top", linespacing=1.12)
        cx = x + bw - 0.07
        for rq in reversed(rqs):
            cw = 0.26
            box(ax, cx - cw, byy + 0.05, cw, 0.12, fc="white", ec=LEX, lw=0.6, r=0.04)
            ax.text(cx - cw / 2, byy + 0.11, r"\textsf{%s}" % rq, fontsize=5.3,
                    ha="center", va="center", color=LEX)
            cx -= cw + 0.03
    save(fig, "fig1_framework.pdf")


# ================================================================ Figure 2
def figure2():
    tier_a = rows("flores_tier_a_orthographic")
    multi = rows("flores_multiscript")
    fig = plt.figure(figsize=(TW, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1], wspace=0.30)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) every variant, grouped by script
    by = {}
    for r in tier_a:
        by.setdefault(r["script"], []).append(r)
    order = sorted(by, key=lambda s: -np.mean([f(r["CR_ortho"]) for r in by[s]]))
    rng = np.random.default_rng(0)
    for i, s in enumerate(order):
        vals = np.array([f(r["CR_ortho"]) for r in by[s]])
        stem = np.array([r["has_stemmer"] == "True" for r in by[s]])
        jit = rng.uniform(-1, 1, len(vals)) * min(0.36, 0.06 * len(vals))
        ax.scatter(i + jit[~stem], vals[~stem], s=5, color="#A8A8A8", lw=0, zorder=2)
        ax.scatter(i + jit[stem], vals[stem], s=9, color=INK, lw=0, zorder=3)
        m = log("2a", "mean CR_ortho, %s (n=%d)" % (s, len(vals)), vals.mean(),
                "flores_by_script / flores_tier_a_orthographic")
        ax.plot([i - 0.38, i + 0.38], [m, m], color=ORTHO, lw=1.8, zorder=4,
                solid_capstyle="butt")
    for r in tier_a:
        if r["lang"] in ("vie_Latn", "yor_Latn"):
            name = {"vie_Latn": "Vietnamese", "yor_Latn": "Yoruba"}[r["lang"]]
            v = log("2a", "CR_ortho " + r["lang"], f(r["CR_ortho"]), "flores_tier_a_orthographic")
            ax.annotate(r"%s %.3f" % (name, v), (0.1, v), xytext=(1.0, v), fontsize=6.3,
                        va="center", color=MUTED,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="#9A9A9A"))
    hairline(ax, 1.0, axis="y")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SCRIPT[s] for s in order], rotation=62,
                       ha="right", rotation_mode="anchor", fontsize=6.2)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0.97, 2.47)
    ax.set_ylabel(r"$CR_{\text{ortho}}$")
    ax.tick_params(axis="x", length=0)
    ax.scatter([], [], s=5, color="#A8A8A8", label="variant")
    ax.scatter([], [], s=9, color=INK, label="variant with a stemmer")
    ax.plot([], [], color=ORTHO, lw=1.8, label="script mean")
    ax.legend(loc="center", bbox_to_anchor=(0.6, 0.62), handletextpad=0.3,
              borderaxespad=0.1)
    panel_label(ax, "a", x=-0.06)

    # (b) same language, two scripts
    pairs = {}
    for r in multi:
        pairs.setdefault(r["language"], []).append(r)
    langs = sorted(pairs, key=lambda k: f(pairs[k][0]["CR_ortho_spread"]))
    for y, k in enumerate(langs):
        a, b = pairs[k]
        va, vb = f(a["CR_ortho"]), f(b["CR_ortho"])
        bx.plot([va, vb], [y, y], color=HAIR, lw=2.2, solid_capstyle="round", zorder=1)
        for v, r in ((va, a), (vb, b)):
            log("2b", "CR_ortho " + r["variant"], v, "flores_multiscript")
            bx.scatter(v, y, s=22, color=ORTHO, edgecolor="white", lw=0.8, zorder=3)
        lo, hi = (a, b) if va < vb else (b, a)
        bx.text(min(va, vb) - 0.016, y, lo["script"], ha="right", va="center", fontsize=5.6,
                color=MUTED)
        bx.text(max(va, vb) + 0.016, y, hi["script"], ha="left", va="center", fontsize=5.6,
                color=MUTED)
        s = log("2b", "spread " + k, f(a["CR_ortho_spread"]), "flores_multiscript")
        bx.text(1.13, y, "%.3f" % s, ha="right", va="center", fontsize=6.3, color=INK,
                transform=bx.get_yaxis_transform())
    bx.text(1.13, len(langs) - 0.35, r"\textit{spread}", ha="right", va="bottom", fontsize=6.3,
            transform=bx.get_yaxis_transform(),
            color=MUTED)
    bx.set_yticks(range(len(langs)))
    bx.set_yticklabels([MULTISCRIPT[k] for k in langs], fontsize=6.8)
    bx.set_xlim(0.83, 1.60)
    bx.set_ylim(-0.6, len(langs) - 0.2)
    bx.set_xticks([1.0, 1.2, 1.4])
    bx.set_xlabel(r"$CR_{\text{ortho}}$")
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    panel_label(bx, "b", x=-0.02)
    save(fig, "fig2_script_confound.pdf")


# ================================================================ Figure 3
def figure3():
    tb = rows("flores_tier_b_intrinsic")
    xd = rows("xnli_cr_decomposition")
    fig = plt.figure(figsize=(TW, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.35)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) stacked log-compression, sorted by morphological share
    tb = sorted(tb, key=lambda r: f(r["morph_share"]))
    for y, r in enumerate(tb):
        o, t = f(r["CR_ortho"]), f(r["CR_total"])
        ax.barh(y, np.log(o), left=0, height=0.68, color=ORTHO, edgecolor="white", lw=0.8)
        ax.barh(y, np.log(t) - np.log(o), left=np.log(o), height=0.68, color=MORPH,
                edgecolor="white", lw=0.8)
        share = log("3a", "morph share " + r["lang"], f(r["morph_share"]), "flores_tier_b_intrinsic")
        log("3a", "CR_total " + r["lang"], t, "flores_tier_b_intrinsic")
        log("3a", "CR_ortho " + r["lang"], o, "flores_tier_b_intrinsic")
        ax.text(np.log(t) + 0.015, y, r"%.1f\%%" % (100 * share), va="center", fontsize=6.3,
                color=INK)
    ticks = [1, 1.25, 1.5, 1.75, 2, 2.25]
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels(["%g" % t for t in ticks])
    ax.set_xlim(0, np.log(2.55))
    ax.set_yticks(range(len(tb)))
    ax.set_yticklabels([LANG[r["lang"][:3]] for r in tb], fontsize=6.8)
    ax.set_ylim(-0.6, len(tb) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"$CR_{\text{total}} = CR_{\text{ortho}} \times CR_{\text{morph}}$ "
                  r"(log scale)")
    ax.legend(handles=[Patch(color=ORTHO, label="orthographic cleanup"),
                       Patch(color=MORPH, label=r"stemming (\%: its share)")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.0,
              handleheight=0.8, borderaxespad=0.2, columnspacing=1.2)
    panel_label(ax, "a", x=-0.03, y=1.08)

    # (b) XNLI ranks by CR_total and by CR_morph
    tot = sorted(xd, key=lambda r: -f(r["CR_total"]))
    mor = sorted(xd, key=lambda r: -f(r["CR_morph"]))
    rank_t = {r["lang"]: i + 1 for i, r in enumerate(tot)}
    rank_m = {r["lang"]: i + 1 for i, r in enumerate(mor)}
    for r in xd:
        l = r["lang"]
        moved = rank_t[l] != rank_m[l]
        c, lw, z = (INK, 1.3, 3) if moved else ("#B5B5B5", 1.0, 2)
        bx.plot([0, 1], [rank_t[l], rank_m[l]], color=c, lw=lw, zorder=z,
                marker="o", ms=3.5, mfc=c, mec="white", mew=0.6)
        vt = log("3b", "CR_total " + l, f(r["CR_total"]), "xnli_cr_decomposition")
        vm = log("3b", "CR_morph " + l, f(r["CR_morph"]), "xnli_cr_decomposition")
        bx.text(-0.06, rank_t[l], r"%s \,{%.3f}" % (LANG[l], vt),
                ha="right", va="center", fontsize=6.6, color=c if moved else MUTED)
        bx.text(1.06, rank_m[l], r"{%.3f}\, %s" % (vm, LANG[l]),
                ha="left", va="center", fontsize=6.6, color=c if moved else MUTED)
    bx.set_xlim(-0.95, 1.95)
    bx.set_ylim(6.6, 0.3)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["rank by\n" + r"$CR_{\text{total}}$", "rank by\n" +
                        r"$CR_{\text{morph}}$"], linespacing=1.1)
    bx.xaxis.tick_top()
    bx.tick_params(axis="x", length=0, pad=2)
    bx.set_yticks([])
    for s in ("left", "bottom", "top", "right"):
        bx.spines[s].set_visible(False)
    bx.text(0.5, 6.95, "XNLI, six languages", ha="center", fontsize=6.6, color=MUTED)
    panel_label(bx, "b", x=0.0, y=1.08)
    save(fig, "fig3_cr_decomposition.pdf")


# ================================================================ Figure 4
def figure4():
    tb = {r["lang"]: r for r in rows("flores_tier_b_intrinsic")}
    tc = {r["lang"]: r for r in rows("flores_tier_c_dense_retrieval")}
    langs = sorted(tb, key=lambda l: -f(tb[l]["CR_morph"]))
    cols = [  # key, header, direction for rank 1, format
        ("CR_morph", r"$CR_{\text{morph}}$", "high", "%.3f"),
        ("KL_morph", r"KL", "high", "%.2f"),
        ("ANLD_morph", r"ANLD", "high", "%.3f"),
        ("IRS_morph", r"IRS", "low", "%.3f"),
        ("AES_morph", r"AES$_{\beta=1}$", "high", "%.3f"),
        ("delta_full", r"dense $\Delta$R@1", "low", "%.3f"),
    ]
    n = len(langs)
    ranks = np.full((n, len(cols)), np.nan)
    vals = np.full((n, len(cols)), np.nan)
    for j, (key, _, d, _) in enumerate(cols):
        src = tc if key == "delta_full" else tb
        have = [l for l in langs if l in src]
        v = {l: f(src[l][key]) for l in have}
        srt = sorted(have, key=lambda l: -v[l] if d == "high" else v[l])
        for i, l in enumerate(langs):
            if l in v:
                vals[i, j] = log("4", "%s %s" % (key, l), v[l],
                                 "flores_tier_c_dense_retrieval" if src is tc
                                 else "flores_tier_b_intrinsic")
                ranks[i, j] = srt.index(l) + 1
    cmap = LinearSegmentedColormap.from_list("seq", ["#08306B", "#2F6DB0", "#8DB9DD",
                                                     "#E7F0F8"])
    fig, ax = plt.subplots(figsize=(4.35, 3.55))
    xs = [0, 1, 2, 3, 4, 5.35]
    for i in range(n):
        for j, x in enumerate(xs):
            if np.isnan(ranks[i, j]):
                ax.add_patch(plt.Rectangle((x - 0.47, i - 0.46), 0.94, 0.92, fc="#F0F0F0",
                                           ec="none"))
                ax.text(x, i, r"\textit{pivot}", ha="center", va="center", fontsize=5.8,
                        color=MUTED)
                continue
            t = (ranks[i, j] - 1) / (n - 1)
            c = cmap(t)
            ax.add_patch(plt.Rectangle((x - 0.47, i - 0.46), 0.94, 0.92, fc=c, ec="none"))
            txt = cols[j][3] % vals[i, j]
            txt = txt.replace("-", r"$-$")
            ax.text(x, i, txt, ha="center", va="center", fontsize=6.1,
                    color="white" if t < 0.45 else INK)
    ax.set_xlim(-0.55, 5.85)
    ax.set_ylim(n - 0.5, -1.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([LANG[l[:3]] for l in langs], fontsize=6.8)
    ax.set_xticks([])
    for j, x in enumerate(xs):
        ax.text(x, -0.98, cols[j][1], ha="center", va="bottom", fontsize=7.2)
        ax.text(x, -0.60, r"\textit{%s}" % ("high = 1" if cols[j][2] == "high" else
                                           ("loss = 1" if j == 5 else "low = 1")),
                ha="center", va="bottom", fontsize=5.6, color=MUTED)
    ax.plot([4.47 + 0.2, 4.47 + 0.2], [-1.45, n - 0.5], color="#9A9A9A", lw=0.6)
    ax.text(2.0, -1.62, r"\textit{intrinsic dimensions (morphological stage)}", ha="center",
            va="bottom", fontsize=6.4, color=MUTED)
    ax.text(5.35, -1.62, r"\textit{downstream}", ha="center", va="bottom", fontsize=6.4,
            color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, ticks=[1, 5, 10, 15])
    cb.ax.invert_yaxis()
    cb.set_label("rank among the 15 languages (14 for retrieval)", fontsize=6.4)
    cb.ax.tick_params(labelsize=6.2, length=2)
    cb.outline.set_visible(False)
    save(fig, "fig4_dimension_ranks.pdf")


# ================================================================ Figure 5
def figure5():
    tc = {r["lang"]: r for r in rows("flores_tier_c_dense_retrieval")}
    bm = {(r["lang"], r["comparison"]): r for r in rows("bm25_comparisons")}
    langs = [r["lang"] for r in rows("dense_vs_sparse")]
    langs = sorted(langs, key=lambda l: f(tc[l]["delta_full"]) if l in tc else 1)
    fig = plt.figure(figsize=(TW, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.40)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) opposite signs
    for y, l in enumerate(langs):
        b = bm[(l, "tok->full")]
        v = log("5a", "BM25 dR@1 tok->full " + l, f(b["delta_R@1"]), "bm25_comparisons")
        lo, hi = f(b["delta_R@1_ci95_lo"]), f(b["delta_R@1_ci95_hi"])
        ax.plot([100 * lo, 100 * hi], [y + 0.13, y + 0.13], color=LEX, lw=1.0)
        ax.scatter(100 * v, y + 0.13, s=14, color=LEX, edgecolor="white", lw=0.6, zorder=3)
        if l in tc:
            d = tc[l]
            v = log("5a", "dense dR@1 raw->full " + l, f(d["delta_full"]),
                    "flores_tier_c_dense_retrieval")
            lo, hi = f(d["delta_full_ci95_lo"]), f(d["delta_full_ci95_hi"])
            ax.plot([100 * lo, 100 * hi], [y - 0.13, y - 0.13], color=DENSE, lw=1.0)
            ax.scatter(100 * v, y - 0.13, s=14, color=DENSE, edgecolor="white", lw=0.6,
                       zorder=3)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([LANG[l[:3]] for l in langs], fontsize=6.8)
    ax.set_ylim(len(langs) - 0.4, -1.9)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-60, 27)
    ax.set_xticks([-60, -50, -40, -30, -20, -10, 0, 10, 20])
    ax.set_xticklabels([r"$%d$" % t if t <= 0 else r"$+%d$" % t for t in
                        [-60, -50, -40, -30, -20, -10, 0, 10, 20]])
    ax.set_xlabel(r"change in recall@1 after normalization (percentage points)")
    ax.text(-1.5, -1.25, r"\textbf{dense retrieval loses}" + "\n" +
            r"{\scriptsize FLORES-200; unnormalized $\rightarrow$ full}",
            ha="right", va="center", fontsize=6.8, color=DENSE, linespacing=1.2)
    ax.text(1.5, -1.25, r"\textbf{BM25 gains}" + "\n" +
            r"{\scriptsize Belebele; tokenized $\rightarrow$ full}",
            ha="left", va="center", fontsize=6.8, color=LEX, linespacing=1.2)
    ax.text(-58, len(langs) - 1, r"\textit{no dense row:}" + "\n" +
            r"\textit{English is the pivot}", fontsize=5.8, color=MUTED, va="center",
            linespacing=1.1)
    panel_label(ax, "a", x=-0.03)

    # (b) which dimensions predict which outcome
    tb = {r["lang"]: r for r in rows("flores_tier_b_intrinsic")}
    dims = [("IRS_morph", "IRS", "IRS"), ("ANLD_morph", "ANLD", "ANLD"),
            ("KL_morph", "KL", "KL"), ("CR_morph", "CR_morph", r"$CR_{\text{morph}}$"),
            ("AES_morph", "AES_beta1", r"AES$_{\beta=1}$")]
    report = open(os.path.join(ROOT, "reports", "flores_predictiveness_output.txt"),
                  encoding="utf-8").read()
    dense_langs = sorted(tc)
    pred = {}
    for r in rows("bm25_predictiveness"):
        pred[(r["target"], r["intrinsic_dimension"])] = (f(r["spearman_rho"]), f(r["p"]))
    targets = [("dense", "delta_ortho", "ortho"),
               ("dense", "delta_full", "full"),
               ("bm25", "ortho_dMRR@10", "ortho"),
               ("bm25", "morph_dMRR@10", "stem"),
               ("bm25", "morph_dR@1", r"stem$^{\dagger}$"),
               ("bm25", "total_dMRR@10", "full")]
    rep_key = {"IRS_morph": "IRS", "ANLD_morph": "ANLD", "KL_morph": "KL",
               "CR_morph": "CR_morph", "AES_morph": "AES"}
    rho = np.zeros((len(dims), len(targets)))
    pv = np.zeros_like(rho)
    for i, (tbkey, bmkey, _) in enumerate(dims):
        for j, (kind, tkey, _) in enumerate(targets):
            if kind == "dense":
                x = [f(tb[l][tbkey]) for l in dense_langs]
                y = [f(tc[l][tkey]) for l in dense_langs]
                res = spearmanr(x, y)
                rr, pp = float(res.statistic), float(res.pvalue)
                m = re.search(r"^%s\s+rho=([+-][\d.]+) p=([\d.]+)\s+rho=([+-][\d.]+) p=([\d.]+)"
                              % rep_key[tbkey], report, re.M)
                want = (f(m.group(1)), f(m.group(2))) if tkey == "delta_ortho" else \
                    (f(m.group(3)), f(m.group(4)))
                assert round(rr, 3) == want[0] and round(pp, 3) == want[1], (tbkey, tkey)
                src = "recomputed from tiers B+C (= reports/flores_predictiveness_output.txt)"
            else:
                rr, pp = pred[(tkey, bmkey)]
                src = "bm25_predictiveness"
            rho[i, j] = log("5b", "rho %s vs %s %s" % (tbkey, kind, tkey), rr, src)
            pv[i, j] = log("5b", "p %s vs %s %s" % (tbkey, kind, tkey), pp, src)
    # shade encodes |rho| (single hue); the sign is printed. Role colours stay on the headers.
    cmap = LinearSegmentedColormap.from_list("mag", ["#F4F4F4", "#9FB0C4", "#34495E"])
    xs = [0, 1, 2.35, 3.35, 4.35, 5.35]
    for i in range(len(dims)):
        for j, x in enumerate(xs):
            c = cmap(abs(rho[i, j]))
            sig = pv[i, j] < 0.05
            bx.add_patch(plt.Rectangle((x - 0.47, i - 0.46), 0.94, 0.92, fc=c,
                                       ec=INK if sig else "none", lw=1.1 if sig else 0))
            s = ("%.2f" % rho[i, j]).replace("-", r"\textminus ")
            if sig:
                s = r"\textbf{%s}" % s
            bx.text(x, i, s, ha="center", va="center", fontsize=5.7,
                    color="white" if abs(rho[i, j]) > 0.55 else INK)
    bx.set_xlim(-0.55, 5.85)
    bx.set_ylim(len(dims) - 0.5, -1.55)
    bx.set_yticks(range(len(dims)))
    bx.set_yticklabels([d[2] for d in dims], fontsize=7)
    bx.set_xticks([])
    for j, x in enumerate(xs):
        bx.text(x, -0.56, targets[j][2], ha="center", va="bottom", fontsize=6.3)
    for xa, xb, lab, c in ((-0.45, 1.45, r"\textbf{dense loss}", DENSE),
                           (1.9, 5.8, r"\textbf{BM25 gain}", LEX)):
        bx.plot([xa, xb], [-0.98, -0.98], color=c, lw=1.0)
        bx.text((xa + xb) / 2, -1.04, lab, ha="center", va="bottom", fontsize=6.6, color=c)
    bx.tick_params(length=0)
    for s in bx.spines.values():
        s.set_visible(False)
    bx.text(-0.5, len(dims) - 0.1, r"Spearman $\rho$; shade $= |\rho|$; outlined: $p < 0.05$"
            "\n" + r"BM25: MRR@10, except $^{\dagger}$recall@1", fontsize=5.8, color=MUTED,
            va="top", linespacing=1.2)
    panel_label(bx, "b", x=-0.02, y=1.02)
    save(fig, "fig5_dense_vs_lexical.pdf")


# ================================================================ Figure 6
def aes(irs, vrg, beta=1.0):
    return (1 + beta ** 2) * irs * vrg / (beta ** 2 * irs + vrg)


def iso_irs(c, vrg):
    """IRS on the AES_beta=1 = c curve, from 2*I*V/(I+V) = c."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(2 * vrg - c > 0, c * vrg / (2 * vrg - c), np.nan)


def figure6():
    ab = {r["unit"]: r for r in rows("ablation_intrinsic")}
    fig = plt.figure(figsize=(TW, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.82, 1.05], wspace=0.42)
    ax, bx, cx = (fig.add_subplot(gs[i]) for i in range(3))
    names = {"porter": "Porter", "snowball": "Snowball", "wordnet": "WordNet",
             "spacy": "SpaCy", "bnltk": "BNLTK", "banlemma": "BanLemma"}

    # (a) English: CR vs IRS with AES_beta=1 iso-curves
    cr = np.linspace(1.55, 2.15, 300)
    for c in (0.56, 0.60, 0.64):
        irs = iso_irs(c, 1 - 1 / cr)
        ax.plot(cr, irs, color="#C8C8C8", lw=0.7, zorder=1)
        k = np.nanargmin(np.abs(irs - 0.975))
        ax.text(cr[k] + 0.008, 0.975, "%.2f" % c, fontsize=5.8, color="#8A8A8A", va="center")
    ax.text(1.565, 0.842, "grey lines:\n" + r"AES$_{\beta=1}$ contours", fontsize=5.8,
            color="#8A8A8A", va="center", linespacing=1.1)
    off = {"porter": (-0.02, 0.0, "right"), "snowball": (-0.02, 0.0, "right"),
           "wordnet": (0.0, -0.011, "center"), "spacy": (0.0, 0.011, "center")}
    for m in ("porter", "snowball", "wordnet", "spacy"):
        r = ab["btsd_en_" + m]
        x = log("6a", "CR btsd_en_" + m, f(r["CR"]), "ablation_intrinsic")
        y = log("6a", "IRS btsd_en_" + m, f(r["IRS"]), "ablation_intrinsic")
        a1 = log("6a", "AES_beta1 btsd_en_" + m, f(r["AES_beta1"]), "ablation_intrinsic")
        assert abs(aes(y, 1 - 1 / x) - a1) < 1e-9
        ax.scatter(x, y, s=26, color=EN_ALGO[m], edgecolor="white", lw=0.8, zorder=3)
        dx, dy, ha = off[m]
        ax.text(x + dx, y + dy, r"%s \,{%.3f}" % (names[m], a1),
                fontsize=6.4, ha=ha, va="center")
    ax.set_xlim(1.55, 2.15)
    ax.set_ylim(0.83, 0.985)
    ax.set_xlabel("CR")
    ax.set_ylabel("IRS (DistilBERT)")
    ax.set_title(r"English, BTSD", fontsize=7.5, color=MUTED, pad=3)
    panel_label(ax, "a", x=-0.14)

    # (b) Bangla: both corpora, BanglaBERT
    cr = np.linspace(1.15, 1.6, 300)
    for c in (0.30, 0.40, 0.50):
        irs = iso_irs(c, 1 - 1 / cr)
        bx.plot(cr, irs, color="#C8C8C8", lw=0.7, zorder=1)
        k = np.nanargmin(np.abs(irs - 0.9495))
        bx.text(cr[k] + 0.008, 0.9495, "%.2f" % c, fontsize=5.8, color="#8A8A8A", va="center")
    for corpus, mk, lab in (("btsd_bn_", "o", "BTSD"), ("sentiment_", "s", "sentiment")):
        pts = []
        for tool, col in (("bnltk", BNLTK), ("banlemma", BANLEMMA)):
            r = ab[corpus + tool]
            x = log("6b", "CR " + corpus + tool, f(r["CR"]), "ablation_intrinsic")
            y = log("6b", "IRS " + corpus + tool, f(r["IRS"]), "ablation_intrinsic")
            log("6b", "AES_beta1 " + corpus + tool, f(r["AES_beta1"]), "ablation_intrinsic")
            bx.scatter(x, y, s=24, marker=mk, color=col, edgecolor="white", lw=0.8, zorder=3)
            pts.append((x, y))
        bx.plot(*zip(*pts), color="#BDBDBD", lw=0.6, zorder=2)
    bx.scatter([], [], s=18, color=BNLTK, label="BNLTK")
    bx.scatter([], [], s=18, color=BANLEMMA, label="BanLemma")
    bx.scatter([], [], s=18, marker="o", color="#8A8A8A", label="BTSD")
    bx.scatter([], [], s=18, marker="s", color="#8A8A8A", label="sentiment")
    bx.legend(loc="lower left", ncol=2, handletextpad=0.1, borderaxespad=0.2,
              columnspacing=0.6, fontsize=6.2)
    bx.set_xlim(1.15, 1.6)
    bx.set_ylim(0.925, 0.952)
    bx.set_xlabel("CR")
    bx.set_ylabel("IRS (BanglaBERT)")
    bx.set_title(r"Bangla, two corpora", fontsize=7.5, color=MUTED, pad=3)
    panel_label(bx, "b", x=-0.2)

    # (c) the beta sweep, English
    betas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
    ends = {}
    for m in ("porter", "snowball", "wordnet", "spacy"):
        r = ab["btsd_en_" + m]
        ys = [log("6c", "AES_beta%g btsd_en_%s" % (b, m), f(r["AES_beta_%s" % b]),
                  "ablation_intrinsic") for b in betas]
        cx.plot(betas, ys, color=EN_ALGO[m], lw=1.2, marker="o", ms=2.6, mec="white",
                mew=0.4, zorder=3)
        ends[m] = ys
    labs = ["porter", "snowball", "wordnet", "spacy"]
    ly = spread([ends[m][-1] for m in labs], 0.028)
    for m, y in zip(labs, ly):
        cx.text(4.35, y, names[m], fontsize=6.4, va="center", color=EN_ALGO[m])
    cx.axvspan(1.0, 1.5, color="#EDEDED", lw=0, zorder=0)
    cx.text(1.22, 0.905, "SpaCy and\nSnowball\nswap", fontsize=5.8, ha="center", va="top",
            color=MUTED, linespacing=1.05)
    cx.set_xscale("log")
    cx.set_xticks(betas)
    cx.set_xticklabels(["0.25", "0.5", "0.75", "1", "1.5", "2", "4"])
    cx.minorticks_off()
    cx.set_xlim(0.22, 4.4)
    cx.set_ylim(0.4, 0.91)
    cx.set_xlabel(r"$\beta$ (log scale; larger favours compression)")
    cx.set_ylabel(r"AES$_\beta$")
    cx.set_title(r"English, BTSD: weighting sweep", fontsize=7.5, color=MUTED, pad=3)
    panel_label(cx, "c", x=-0.15)
    save(fig, "fig6_tradeoff.pdf")


# ================================================================ Figure 7
def figure7():
    dsp = rows("ablation_dsp")
    fig = plt.figure(figsize=(TW, 2.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.95], wspace=0.55)
    ax, bx, cx = (fig.add_subplot(gs[i]) for i in range(3))
    seeds = ["42", "7", "2024"]

    # (a) same tool, two corpora
    y, yt, yl = 0, [], []
    for corpus, clab in (("btsd_bn_", "BTSD"), ("sentiment_", "Sentiment")):
        ax.text(-0.0445, y - 0.55, r"\textbf{%s}" % clab, fontsize=6.8, va="center", ha="left")
        for tool, col in (("bnltk", BNLTK), ("banlemma", BANLEMMA)):
            for s in seeds:
                r = next(r for r in dsp if r["unit"] == corpus + tool and
                         r["classifier"] == "MultinomialNB" and r["seed"] == s)
                v = log("7a", "MNB delta %s seed %s" % (corpus + tool, s), f(r["delta_mean"]),
                        "ablation_dsp")
                lo, hi = f(r["delta_ci95_lo"]), f(r["delta_ci95_hi"])
                log("7a", "MNB wilcoxon p %s seed %s" % (corpus + tool, s), f(r["wilcoxon_p"]),
                    "ablation_dsp")
                ax.plot([lo, hi], [y, y], color=col, lw=1.0)
                ax.scatter(v, y, s=15, color=col, edgecolor="white", lw=0.6, zorder=3)
                yt.append(y)
                yl.append("seed %s" % s)
                y += 0.8
            y += 0.35
        y += 0.9
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=5.8)
    ax.set_ylim(y - 1.5, -1.0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.045, 0.047)
    ax.set_xticks([-0.03, 0, 0.03])
    ax.set_xticklabels([r"$-0.03$", "0", r"$+0.03$"])
    ax.set_xlabel(r"$\Delta$ macro-F1, Multinomial NB")
    ax.scatter([], [], s=15, color=BNLTK, label="BNLTK")
    ax.scatter([], [], s=15, color=BANLEMMA, label="BanLemma")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.0), ncol=1, handletextpad=0.1,
              borderaxespad=0.1, labelspacing=0.25)
    ax.set_title("across corpora", fontsize=7.5, color=MUTED, pad=24)
    panel_label(ax, "a", x=-0.2, y=1.2)

    # (b) same language, two protocols
    zs = {r["lang"]: r for r in rows("xnli_intrinsic_zero_shot")}
    il = rows("xnli_in_language_dsp")
    order = sorted(zs, key=lambda l: f(zs[l]["MPD"]))
    for i, l in enumerate(order):
        if l == "de":
            bx.axhspan(i - 0.45, i + 0.45, color="#EDEDED", lw=0, zorder=0)
        m = log("7b", "zero-shot MPD " + l, f(zs[l]["MPD"]), "xnli_intrinsic_zero_shot")
        bx.scatter(m, i - 0.27, marker="D", s=14, color=INK, edgecolor="white", lw=0.5,
                   zorder=3)
        for k, s in enumerate(seeds):
            r = next(r for r in il if r["lang"] == l and r["seed"] == s)
            v = log("7b", "in-language delta %s seed %s" % (l, s), f(r["delta_mean"]),
                    "xnli_in_language_dsp")
            lo, hi = f(r["delta_ci95_lo"]), f(r["delta_ci95_hi"])
            yy = i - 0.04 + 0.16 * k
            bx.plot([lo, hi], [yy, yy], color=LEX, lw=0.9)
            bx.scatter(v, yy, s=9, color=LEX, edgecolor="white", lw=0.4, zorder=3)
    bx.axvline(0, color=INK, lw=0.8, zorder=1)
    bx.set_yticks(range(len(order)))
    bx.set_yticklabels([LANG[l] for l in order], fontsize=6.8)
    bx.set_ylim(len(order) - 0.45, -0.6)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlim(-0.115, 0.032)
    bx.set_xticks([-0.10, -0.05, 0])
    bx.set_xticklabels([r"$-0.10$", r"$-0.05$", "0"])
    bx.set_xlabel(r"$\Delta$ macro-F1, XNLI")
    bx.scatter([], [], marker="D", s=12, color=INK, label="zero-shot MPD")
    bx.scatter([], [], s=10, color=LEX, label="in-language, 3 seeds")
    bx.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.0), ncol=1, handletextpad=0.1,
              borderaxespad=0.1, labelspacing=0.25)
    bx.set_title("across protocols", fontsize=7.5, color=MUTED, pad=24)
    panel_label(bx, "b", x=-0.2, y=1.2)

    # (c) same texts, two encoders: BNLTK - BanLemma IRS gap
    enc = rows("encoder_dependence_irs")
    corp = [("FLORES-200 ben_Beng (dev+devtest)", "FLORES-200"), ("BTSD (Bangla)", "BTSD"),
            ("Bangla sentiment", "Sentiment")]
    for i, (key, lab) in enumerate(corp):
        rb = next(r for r in enc if r["corpus"] == key and r["tool"] == "bnltk")
        rl = next(r for r in enc if r["corpus"] == key and r["tool"] == "banlemma")
        g1 = log("7c", "IRS gap MiniLM " + lab, f(rb["IRS_minilm"]) - f(rl["IRS_minilm"]),
                 "encoder_dependence_irs")
        g2 = log("7c", "IRS gap BanglaBERT " + lab,
                 f(rb["IRS_banglabert"]) - f(rl["IRS_banglabert"]), "encoder_dependence_irs")
        cx.plot([g1, g2], [i, i], color=HAIR, lw=2.0, zorder=1, solid_capstyle="round")
        cx.scatter(g1, i, s=22, facecolor="white", edgecolor=INK, lw=1.0, zorder=3)
        cx.scatter(g2, i, s=22, color=INK, edgecolor="white", lw=0.6, zorder=3)
        cx.text(0.006, i - 0.3, r"$%+.3f \rightarrow %+.3f$" % (g1, g2), ha="left",
                va="center", fontsize=6.0, color=MUTED)
    cx.axvline(0, color=INK, lw=0.8, zorder=0)
    cx.set_yticks(range(len(corp)))
    cx.set_yticklabels([c[1] for c in corp], fontsize=6.8)
    cx.set_ylim(len(corp) - 0.45, -0.7)
    cx.tick_params(axis="y", length=0)
    cx.spines["left"].set_visible(False)
    cx.set_xlim(-0.02, 0.145)
    cx.set_xticks([0, 0.05, 0.10])
    cx.set_xticklabels(["0", "0.05", "0.10"])
    cx.set_xlabel("IRS gap\n" + r"(BNLTK $-$ BanLemma)", linespacing=1.1)
    cx.scatter([], [], s=18, facecolor="white", edgecolor=INK, lw=1.0, label="MiniLM")
    cx.scatter([], [], s=18, color=INK, label="BanglaBERT")
    cx.legend(loc="lower left", bbox_to_anchor=(-0.03, 1.0), ncol=1, handletextpad=0.1,
              borderaxespad=0.1, labelspacing=0.25)
    cx.set_title("across encoders", fontsize=7.5, color=MUTED, pad=24)
    panel_label(cx, "c", x=-0.2, y=1.2)
    save(fig, "fig7_nontransfer.pdf")


def write_values():
    lines = ["# Values plotted in the figures", "",
             "Written by `code/figures/make_figures.py`. Every value is read from `tables/*.csv` "
             "(source column); none is typed by hand.", "",
             "| figure | item | value | source |", "|---|---|---:|---|"]
    for fig, item, v, src in VALUES:
        lines.append("| %s | %s | %.6g | %s |" % (fig, item, v, src))
    path = os.path.join(ROOT, "reports", "figure_values.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(path, ROOT), "(%d values)" % len(VALUES))


# ================================================================ Graphical abstract
def graphical_abstract():
    """Elsevier graphical abstract: 2.5:1 (w:h), at least 1328 x 531 px, legible at 500 x 200 px.

    Drawn at 10 x 4 in and written as PDF and as a 300 dpi TIFF (3000 x 1200 px). Times, as
    Elsevier recommends for graphical abstracts; minimal text, no title inside the image. Every
    number is read from tables/*.csv.
    """
    tc = rows("flores_tier_c_dense_retrieval")
    bm = rows("bm25_comparisons")
    multi = rows("flores_multiscript")
    dsp = rows("ablation_dsp")
    dense = [log("GA", "dense dR@1 raw->full " + r["lang"], f(r["delta_full"]),
                 "flores_tier_c_dense_retrieval") for r in tc]
    lex = [log("GA", "BM25 dR@1 tok->full " + r["lang"], f(r["delta_R@1"]), "bm25_comparisons")
           for r in bm if r["comparison"] == "tok->full"]
    knc = {r["script"]: log("GA", "CR_ortho " + r["variant"], f(r["CR_ortho"]),
                            "flores_multiscript") for r in multi if r["language"] == "knc"}
    mnb = {}
    for unit, lab in (("btsd_bn_banlemma", "BTSD"), ("sentiment_banlemma", "Sentiment")):
        r = next(r for r in dsp if r["unit"] == unit and r["classifier"] == "MultinomialNB"
                 and r["seed"] == "42")
        mnb[lab] = log("GA", "MNB delta %s seed 42" % unit, f(r["delta_mean"]), "ablation_dsp")

    W, H = 10.0, 4.0
    with plt.rc_context({"text.latex.preamble":
                         r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{newtxtext}"
                         r"\usepackage{newtxmath}",
                         "savefig.bbox": None, "savefig.pad_inches": 0}):
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis("off")

        def heading(x, text):
            ax.text(x, 3.72, r"\textbf{%s}" % text, fontsize=15, va="center", color=INK)

        # left: the step being evaluated
        heading(0.25, "Text normalization")
        for k, (t, fc) in enumerate((("Original text", "white"), ("Normalizer", "#E4ECF4"),
                                     ("Normalized text", "white"))):
            y = 2.85 - 0.78 * k
            box(ax, 0.3, y, 2.3, 0.52, fc=fc, ec="#8A9099", lw=1.0, r=0.08)
            ax.text(1.45, y + 0.26, t, ha="center", va="center", fontsize=13.5)
            if k < 2:
                arrow(ax, 1.45, y, 1.45, y - 0.26)
        ax.text(1.45, 0.72, r"\textit{connected, connecting,}" + "\n" +
                r"\textit{connection} $\rightarrow$ \textit{connect}", ha="center",
                va="center", fontsize=12, color=MUTED, linespacing=1.2)
        ax.text(1.45, 0.22, "lossy and one-way", ha="center", va="center", fontsize=12,
                color=MUTED)

        arrow(ax, 2.78, 2.1, 3.18, 2.1, color="#555555")

        # middle: the six dimensions and the output
        heading(3.35, "Six dimensions")
        tiles = [("CR", "compression"), ("KL", "distribution"), ("ANLD", "word form"),
                 ("IRS", "meaning"), (r"AES$_\beta$", r"trade-off ($\beta$)"),
                 ("MPD", "downstream")]
        tw, th, gx, gy = 0.98, 0.84, 0.08, 0.1
        for k, (name, sub_) in enumerate(tiles):
            cx, cy = 3.35 + (k % 3) * (tw + gx), 2.35 - (k // 3) * (th + gy)
            box(ax, cx, cy, tw, th, fc=PANEL, ec="#C3C7CC", lw=0.8, r=0.06)
            ax.text(cx + tw / 2, cy + th * 0.62, r"\textbf{%s}" % name, ha="center",
                    va="center", fontsize=15)
            ax.text(cx + tw / 2, cy + th * 0.26, sub_, ha="center", va="center", fontsize=11,
                    color=MUTED)
        box(ax, 3.35, 0.2, 3 * tw + 2 * gx, 0.46, fc="#E4ECF4", ec=LEX, lw=1.2, r=0.1)
        ax.text(3.35 + (3 * tw + 2 * gx) / 2, 0.43, r"\textbf{a profile, not a ranking}",
                ha="center", va="center", fontsize=14, color=INK)

        arrow(ax, 6.55, 2.1, 6.95, 2.1, color="#555555")

        # right: three things a single score hides, each drawn from the data
        heading(7.1, "What one score hides")
        rows_y = [2.72, 1.62, 0.52]
        texts = [r"same pipeline: dense $\downarrow$, BM25 $\uparrow$",
                 "same language, two scripts",
                 r"same Bangla tool, two corpora ($\Delta$F1)"]
        for y, t in zip(rows_y, texts):
            ax.text(7.1, y + 0.6, t, fontsize=12.5, va="center", color=INK)

        # (1) dense vs BM25 change in recall@1, every language
        a1 = fig.add_axes([7.1 / W, (rows_y[0] - 0.05) / H, 2.65 / W, 0.42 / H])
        a1.scatter([100 * v for v in dense], [0.35] * len(dense), s=22, color=DENSE, lw=0,
                   alpha=0.85)
        a1.scatter([100 * v for v in lex], [-0.35] * len(lex), s=22, color=LEX, lw=0,
                   alpha=0.85)
        a1.axvline(0, color=INK, lw=1.0)
        a1.set_xlim(-60, 25)
        a1.set_ylim(-0.8, 0.8)
        a1.set_yticks([])
        a1.set_xticks([-50, -25, 0, 20])
        a1.set_xticklabels([r"$-50$", r"$-25$", "0", r"$+20$ pts"], fontsize=10)
        a1.text(-40, -0.35, "dense retrieval", fontsize=10.5, color=DENSE, va="center",
                ha="center")
        a1.text(13, 0.35, "BM25", fontsize=10.5, color=LEX, va="center", ha="center")
        for sp in ("left", "top", "right"):
            a1.spines[sp].set_visible(False)
        a1.tick_params(axis="x", length=2, pad=1)

        # (2) Central Kanuri in Arabic and Latin script
        a2 = fig.add_axes([7.1 / W, (rows_y[1] - 0.05) / H, 2.65 / W, 0.42 / H])
        lo, hi = knc["Latn"], knc["Arab"]
        a2.plot([lo, hi], [0, 0], color=HAIR, lw=5, solid_capstyle="round", zorder=1)
        a2.scatter([lo, hi], [0, 0], s=60, color=ORTHO, edgecolor="white", lw=1, zorder=3)
        a2.text(lo - 0.02, 0, "Latin", ha="right", va="center", fontsize=10.5, color=MUTED)
        a2.text(hi + 0.02, 0, "Arabic", ha="left", va="center", fontsize=10.5, color=MUTED)
        a2.text((lo + hi) / 2, 0.45, r"$\Delta CR_{\text{ortho}} = %.3f$" % (hi - lo),
                ha="center", va="bottom", fontsize=11.5, color=INK)
        a2.set_xlim(1.0, 1.58)
        a2.set_ylim(-0.8, 1.2)
        a2.axis("off")

        # (3) BanLemma with Multinomial NB on the two Bangla corpora
        a3 = fig.add_axes([7.1 / W, (rows_y[2] - 0.12) / H, 2.65 / W, 0.5 / H])
        labs = ["BTSD", "Sentiment"]
        vals = [mnb[l] for l in labs]
        a3.barh([0.5, -0.5], vals, height=0.6, color=BANLEMMA)
        a3.axvline(0, color=INK, lw=1.0)
        for yy, l, v in zip([0.5, -0.5], labs, vals):
            a3.text(v + (0.002 if v > 0 else -0.002), yy, "%s %s" % (l, ("%+.3f" % v).replace(
                "-", r"$-$")), va="center", ha="left" if v > 0 else "right", fontsize=10.5)
        a3.set_xlim(-0.05, 0.075)
        a3.set_ylim(-1.1, 1.1)
        a3.axis("off")

        path = os.path.join(OUT, "graphical_abstract.pdf")
        fig.savefig(path)
        fig.savefig(os.path.join(OUT, "graphical_abstract.tif"), dpi=300,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.join(ROOT, "..")), "and graphical_abstract.tif")


if __name__ == "__main__":
    for fn in (figure1, figure2, figure3, figure4, figure5, figure6, figure7,
               graphical_abstract):
        fn()
    write_values()
