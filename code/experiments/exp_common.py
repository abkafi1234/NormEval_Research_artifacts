"""Shared utilities for the re-run experiments. See EXPERIMENTAL_PLAN.md.

Every setting mirrors Paper/template3.tex and the reproducibility notebook, except the fold
count (10, as the paper reports) and the extra outputs saved for verification (per-fold scores,
out-of-fold predictions, McNemar, CIs, effect sizes).
"""
from __future__ import annotations

import json
import os
import platform
import re
import string
import sys
import time
import unicodedata
from functools import lru_cache

import numpy as np

# --------------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ~/Research/NormEvaluator_pypi
PROJ = os.path.join(ROOT, "normeval-main")
REPRO = os.path.join(PROJ, "Reproducible Experiment")
DATA = os.path.join(REPRO, "Dataset")
BANLEMMA_DIR = os.path.join(REPRO, "bangla", "BanLemma-main")
FLORES = os.path.join(PROJ, "dataset", "flores200_dataset")
TIER_B = os.path.join(PROJ, "benchmark", "artifacts", "tier_b_intrinsic.json")
OUT = os.environ.get("RUN_OUT", os.path.join(ROOT, "runs", "outputs"))
CACHE = os.path.join(ROOT, "runs", "cache")
STUBS = os.path.join(ROOT, "runs", "stubs")
for _p in (STUBS, BANLEMMA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEEDS = tuple(int(s) for s in os.environ.get("RUN_SEEDS", "42,7,2024").split(","))
N_SPLITS = 10
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DISTILBERT = "distilbert-base-uncased"
BANGLABERT = "csebuetnlp/banglabert"


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- saving
def versions():
    import importlib
    out = {"python": platform.python_version(), "host": platform.node()}
    for m in ("numpy", "scipy", "sklearn", "torch", "transformers", "sentence_transformers",
              "datasets", "spacy", "nltk", "statsmodels", "normeval"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:  # noqa: BLE001
            out[m] = None
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        out["gpu"] = None
    return out


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def out_path(*parts):
    return os.path.join(OUT, *parts)


def done(*parts):
    return os.path.exists(out_path(*parts))


def save(obj, *parts):
    path = out_path(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = dict(obj)
    obj["_meta"] = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "script": os.path.basename(sys.argv[0]),
                    "argv": sys.argv[1:], "versions": versions()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_default)
    os.replace(tmp, path)
    log(f"saved {path}")
    return path


# --------------------------------------------------------------------------- XNLI / FLORES pipeline
ARABIC_PUNCT = "،؛؟٫٬٪﴾﴿ـ«»"
_PUNCT_XNLI = str.maketrans("", "", string.punctuation + ARABIC_PUNCT)          # notebook definition
CJK_PUNCT = "、。，．：；？！「」『』（）"
DEVANAGARI_PUNCT = "।॥"
_PUNCT_FLORES = str.maketrans("", "", string.punctuation + ARABIC_PUNCT + CJK_PUNCT + DEVANAGARI_PUNCT)  # benchmark/common.py
_TURKISH = str.maketrans({"İ": "i", "I": "ı"})

XNLI_SNOWBALL = {"ar": "arabic", "de": "german", "en": "english", "es": "spanish", "fr": "french", "ru": "russian"}
FLORES_SNOWBALL = {
    "arb_Arab": "arabic", "dan_Latn": "danish", "nld_Latn": "dutch", "eng_Latn": "english",
    "fin_Latn": "finnish", "fra_Latn": "french", "deu_Latn": "german", "hun_Latn": "hungarian",
    "ita_Latn": "italian", "nob_Latn": "norwegian", "por_Latn": "portuguese", "ron_Latn": "romanian",
    "rus_Cyrl": "russian", "spa_Latn": "spanish", "swe_Latn": "swedish",
}


@lru_cache(maxsize=None)
def snowball(name):
    from nltk.stem.snowball import SnowballStemmer
    return SnowballStemmer(name)


def _ortho(text, table, turkish=False):
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", text)
    if turkish:
        t = t.translate(_TURKISH)
    t = t.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.translate(table)
    return re.sub(r"\s+", " ", t).strip()


def xnli_ortho(text, lang):
    """Notebook pipeline without stemming (orthographic stage for XNLI)."""
    return _ortho(text, _PUNCT_XNLI, turkish=(lang == "tr"))


def xnli_full(text, lang):
    """Notebook `normalize_text`: orthographic stage + language-specific Snowball."""
    st = snowball(XNLI_SNOWBALL[lang])
    return " ".join(st.stem(w) for w in xnli_ortho(text, lang).split() if w)


def flores_ortho(text, lang_code=""):
    """benchmark/common.py `orthographic`."""
    return _ortho(text, _PUNCT_FLORES, turkish=lang_code.startswith("tur"))


def flores_full(text, lang_code):
    t = flores_ortho(text, lang_code)
    name = FLORES_SNOWBALL.get(lang_code)
    if name is None:
        return t
    st = snowball(name)
    return " ".join(st.stem(w) for w in t.split() if w)


def whitespace(text):
    return text.split()


# --------------------------------------------------------------------------- BTSD / Bangla normalizers
WORD_RE = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=None)
def _spacy():
    import spacy
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def normalize_english_batch(texts, method):
    """Notebook `normalize_english_batch`."""
    texts = [str(t).lower() for t in texts]
    if method == "spacy":
        return [" ".join(tok.lemma_ for tok in doc if not tok.is_space) for doc in _spacy().pipe(texts, batch_size=64)]
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.stem.snowball import SnowballStemmer
    fn = {"porter": PorterStemmer().stem, "snowball": SnowballStemmer("english").stem,
          "wordnet": WordNetLemmatizer().lemmatize}[method]
    return [" ".join(fn(t) for t in WORD_RE.findall(text)) for text in texts]


@lru_cache(maxsize=None)
def _bnltk():
    from bnltk.stemmer import BanglaStemmer
    return BanglaStemmer()


def normalize_bangla_batch(texts, method):
    """Notebook `normalize_bangla_batch`."""
    texts = [str(t) for t in texts]
    if method == "banlemma":
        import banlemma
        return [banlemma.lemmatize(t) for t in texts]
    if method == "bnltk":
        st = _bnltk()
        return [" ".join(st.stem(tok) for tok in t.split()) for t in texts]
    raise ValueError(method)


# --------------------------------------------------------------------------- encoders
@lru_cache(maxsize=None)
def encoder(name, mean_pooled=False):
    from sentence_transformers import SentenceTransformer, models as stm
    if mean_pooled:
        w = stm.Transformer(name)
        p = stm.Pooling(w.get_word_embedding_dimension(), pooling_mode="mean")
        return SentenceTransformer(modules=[w, p], device=device())
    return SentenceTransformer(name, device=device())


def encode(model, texts, batch_size=32):
    return np.asarray(model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                                   convert_to_numpy=True), dtype=np.float32)


# --------------------------------------------------------------------------- statistics
def paired_stats(f_orig, f_norm):
    """Same Wilcoxon call as normeval.calculate_traditional_dsp, plus SD, t-CI and Cohen's d_z."""
    from scipy.stats import t as tdist, wilcoxon
    fo, fn = np.asarray(f_orig, float), np.asarray(f_norm, float)
    d = fn - fo
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    half = float(tdist.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    p = 1.0 if np.all(d == 0) else float(wilcoxon(fo, fn).pvalue)
    return {"n_folds": n, "mean_orig": float(fo.mean()), "mean_norm": float(fn.mean()),
            "delta_mean": mean, "delta_sd": sd, "ci95": [mean - half, mean + half],
            "d_z": (mean / sd) if sd and sd > 0 else float("nan"), "wilcoxon_p": p}


def mcnemar_paired(correct_a, correct_b):
    """benchmark/common.py `mcnemar_paired` (exact below 25 discordant pairs, else corrected chi2)."""
    from statsmodels.stats.contingency_tables import mcnemar
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n11, n10 = int(np.sum(a & b)), int(np.sum(a & ~b))
    n01, n00 = int(np.sum(~a & b)), int(np.sum(~a & ~b))
    exact = (n10 + n01) < 25
    res = mcnemar([[n11, n10], [n01, n00]], exact=exact, correction=not exact)
    return {"orig_only_correct": n10, "norm_only_correct": n01, "both": n11, "neither": n00,
            "statistic": float(res.statistic), "p_value": float(res.pvalue), "exact": bool(exact)}


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=12345):
    """benchmark/common.py `bootstrap_ci` (percentile bootstrap of the mean)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    stats = v[idx].mean(axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------- Traditional DSP (detailed)
def traditional_dsp_detailed(texts_orig, texts_norm, labels, clf, n_splits, seed,
                             vectorizer=None, X_orig=None, X_norm=None):
    """Replicates normeval.NormalizationEvaluator.calculate_traditional_dsp fold by fold,
    and additionally returns per-fold scores and out-of-fold predictions."""
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    to, tn = np.array(texts_orig), np.array(texts_norm)
    f_o, f_n = [], []
    pred_o, pred_n = np.empty(len(y), dtype=y.dtype), np.empty(len(y), dtype=y.dtype)
    fold = np.empty(len(y), dtype=int)
    for k, (tr, te) in enumerate(skf.split(list(texts_orig), y)):
        if X_orig is not None:
            xo_tr, xo_te, xn_tr, xn_te = X_orig[tr], X_orig[te], X_norm[tr], X_norm[te]
        else:
            vo = clone(vectorizer)
            xo_tr, xo_te = vo.fit_transform(to[tr]), None
            xo_te = vo.transform(to[te])
            vn = clone(vectorizer)
            xn_tr = vn.fit_transform(tn[tr])
            xn_te = vn.transform(tn[te])
        po = clone(clf).fit(xo_tr, y[tr]).predict(xo_te)
        pn = clone(clf).fit(xn_tr, y[tr]).predict(xn_te)
        f_o.append(float(f1_score(y[te], po, average="macro")))
        f_n.append(float(f1_score(y[te], pn, average="macro")))
        pred_o[te], pred_n[te], fold[te] = po, pn, k
    return {"f1_orig": f_o, "f1_norm": f_n, "pred_orig": pred_o.tolist(), "pred_norm": pred_n.tolist(),
            "fold": fold.tolist()}


def dsp_unit_result(det, labels):
    y = np.asarray(labels)
    st = paired_stats(det["f1_orig"], det["f1_norm"])
    mc = mcnemar_paired(np.asarray(det["pred_orig"]) == y, np.asarray(det["pred_norm"]) == y)
    return {"stats": st, "mcnemar": mc, "per_fold": {"f1_orig": det["f1_orig"], "f1_norm": det["f1_norm"]},
            "oof": {"pred_orig": det["pred_orig"], "pred_norm": det["pred_norm"], "fold": det["fold"]}}
