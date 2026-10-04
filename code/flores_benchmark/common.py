"""Shared infrastructure for the FLORES-200 benchmark.

Covers: the normalization pipelines (split into orthographic and
morphological stages so their contributions can be measured separately),
a bf16 + torch.compile encoder wrapper tuned for the RTX 4070, and the
statistical helpers (bootstrap CIs, paired tests) used by every tier.
"""
from __future__ import annotations

import os
import re
import json
import string
import unicodedata
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = r"D:\Research\NormEvaluator_pypi\normeval-main"
FLORES = os.path.join(REPO, "dataset", "flores200_dataset")
ARTIFACTS = os.path.join(REPO, "benchmark", "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Normalization pipeline, staged
#
# Stage 1 (orthographic): NFKC, lowercase, diacritic strip, punctuation removal
# Stage 2 (morphological): language-specific Snowball stemming
#
# Keeping these separate is the whole point: it lets us attribute vocabulary
# compression to trivial orthographic cleanup vs genuine morphological
# reduction, instead of crediting both to one number.
# ---------------------------------------------------------------------------
ARABIC_PUNCT = "\u060c\u061b\u061f\u066b\u066c\u066a\ufd3e\ufd3f\u0640\u00ab\u00bb"
CJK_PUNCT = "\u3001\u3002\uff0c\uff0e\uff1a\uff1b\uff1f\uff01\u300c\u300d\u300e\u300f\uff08\uff09"
DEVANAGARI_PUNCT = "\u0964\u0965"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + ARABIC_PUNCT + CJK_PUNCT + DEVANAGARI_PUNCT)
_TURKISH_TABLE = str.maketrans({"\u0130": "i", "I": "\u0131"})

# FLORES language code (ISO-639-3_Script) -> NLTK Snowball language name.
# Only these 15 have a stemmer; every other FLORES language can still be
# measured for orthographic compression.
FLORES_TO_SNOWBALL = {
    "arb_Arab": "arabic",   "dan_Latn": "danish",   "nld_Latn": "dutch",
    "eng_Latn": "english",  "fin_Latn": "finnish",  "fra_Latn": "french",
    "deu_Latn": "german",   "hun_Latn": "hungarian","ita_Latn": "italian",
    "nob_Latn": "norwegian","por_Latn": "portuguese","ron_Latn": "romanian",
    "rus_Cyrl": "russian",  "spa_Latn": "spanish",  "swe_Latn": "swedish",
}


# ---------------------------------------------------------------------------
# The benchmark suite.
#
# A language belongs in the suite only if the *complete* metric set can be
# computed for it, which requires a morphological normalizer: the 15 Snowball
# languages. For these, every metric is defined -- CR_total / CR_ortho /
# CR_morph, ANLD, KL, IRS, AES, and retrieval.
#
# Languages without a normalizer are excluded, because CR_morph would be
# identically 1.0 and ANLD identically 0 -- not measurements, just absences.
# They are still measured for the orthographic-only analysis (Tier A), where no
# stemmer is required and the data is therefore complete for that claim.
#
# Bangla is deliberately NOT in the multilingual suite. It is the paper's depth
# probe and is studied on its own two corpora (BTSD, sentiment) with a
# Bangla-specific encoder. Mixing it in here would have introduced two
# problems: the multilingual encoder cannot do Bangla retrieval (baseline
# recall@1 ~0.21, below any competence floor), and IRS computed with a
# multilingual encoder is not comparable to IRS computed with BanglaBERT in the
# depth analysis.
# ---------------------------------------------------------------------------
BENCHMARK_LANGS = sorted(FLORES_TO_SNOWBALL.keys())
RETRIEVAL_PIVOT = "eng_Latn"
BENCHMARK_QUERY_LANGS = [lc for lc in BENCHMARK_LANGS if lc != RETRIEVAL_PIVOT]


@lru_cache(maxsize=None)
def _stemmer(snowball_name: str):
    from nltk.stem.snowball import SnowballStemmer
    return SnowballStemmer(snowball_name)


def orthographic(text: str, lang_code: str = "") -> str:
    """Stage 1 only: everything except stemming."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    if lang_code.startswith("tur"):
        t = t.translate(_TURKISH_TABLE)
    t = t.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", t).strip()


def full_pipeline(text: str, lang_code: str) -> str:
    """Stage 1 + stage 2. Falls back to stage 1 where no stemmer exists."""
    t = orthographic(text, lang_code)
    name = FLORES_TO_SNOWBALL.get(lang_code)
    if name is None:
        return t
    st = _stemmer(name)
    return " ".join(st.stem(w) for w in t.split() if w)


def has_stemmer(lang_code: str) -> bool:
    return lang_code in FLORES_TO_SNOWBALL


# ---------------------------------------------------------------------------
# FLORES IO
# ---------------------------------------------------------------------------
def flores_languages() -> list[str]:
    d = os.path.join(FLORES, "dev")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".dev"))


def load_flores(lang_code: str, splits=("dev", "devtest"), limit: int | None = None) -> list[str]:
    """Load a language. Default uses dev+devtest (~2,009 sentences) so every
    measurement has as much data as FLORES provides."""
    out: list[str] = []
    for sp in splits:
        path = os.path.join(FLORES, sp, f"{lang_code}.{sp}")
        with open(path, encoding="utf-8") as f:
            out.extend(line.rstrip("\n") for line in f)
    return out[:limit] if limit else out


# ---------------------------------------------------------------------------
# Vocabulary / compression
# ---------------------------------------------------------------------------
def vocab(texts) -> set[str]:
    """Case-sensitive whitespace vocabulary, matching normeval's calculate_cr
    (CountVectorizer with lowercase=False)."""
    v: set[str] = set()
    for t in texts:
        v.update(t.split())
    return v


def compression_decomposition(raw, ortho, full) -> dict:
    v_raw, v_ortho, v_full = len(vocab(raw)), len(vocab(ortho)), len(vocab(full))
    cr_total = v_raw / v_full if v_full else float("nan")
    cr_ortho = v_raw / v_ortho if v_ortho else float("nan")
    cr_morph = v_ortho / v_full if v_full else float("nan")
    share = (np.log(cr_morph) / np.log(cr_total)) if cr_total > 1 and cr_morph > 0 else float("nan")
    return {
        "V_raw": v_raw, "V_ortho": v_ortho, "V_full": v_full,
        "CR_total": cr_total, "CR_ortho": cr_ortho, "CR_morph": cr_morph,
        "morph_share": share,
    }


# ---------------------------------------------------------------------------
# Statistics
#
# Every reported quantity gets a bootstrap CI. Where two conditions are
# compared on the same items, we also run a paired test, so no number in the
# benchmark rests on a point estimate alone.
# ---------------------------------------------------------------------------
def bootstrap_ci(values, statistic=np.mean, n_boot=10000, alpha=0.05, seed=12345):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = statistic(values[idx], axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(values)), float(lo), float(hi)


def _binary_doc_term(items):
    """Sentence x token binary incidence matrix (CSR)."""
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(tokenizer=lambda s: s.split(), token_pattern=None,
                          lowercase=False, binary=True)
    return vec.fit_transform(list(items))


def vocabulary_ratio_replicates(numer_by_split: dict, denom_by_split: dict) -> dict:
    """Uncertainty for a vocabulary ratio, done correctly.

    A bootstrap is NOT valid here. Vocabulary size is a saturating function of
    sample size (Heaps' law), so resampling with replacement (~63% unique
    sentences) shrinks both vocabularies, and because raw and normalized text
    have different Heaps' exponents the *ratio* shifts systematically. The
    bootstrap distribution ends up biased low enough that it need not even
    contain the point estimate.

    A compression ratio is also a descriptive statistic of a fixed corpus, not
    an estimate of a population parameter: for a given corpus it is exact. The
    question that actually matters is whether it generalizes to other text of
    the same language, which independent splits answer directly. FLORES
    provides two (dev, devtest), so we report the ratio on each plus the
    pooled value and the observed spread.
    """
    out = {}
    for split in numer_by_split:
        vn, vd = len(vocab(numer_by_split[split])), len(vocab(denom_by_split[split]))
        out[split] = vn / vd if vd else float("nan")
    pooled_n = [s for split in numer_by_split for s in numer_by_split[split]]
    pooled_d = [s for split in denom_by_split for s in denom_by_split[split]]
    vn, vd = len(vocab(pooled_n)), len(vocab(pooled_d))
    vals = [v for v in out.values() if not np.isnan(v)]
    return {
        "pooled": vn / vd if vd else float("nan"),
        "by_split": out,
        "split_spread": float(max(vals) - min(vals)) if len(vals) > 1 else float("nan"),
    }


def cohens_dz(diffs) -> float:
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else float("nan")


def mcnemar_paired(correct_a, correct_b) -> dict:
    """Exact/corrected McNemar on paired per-item correctness."""
    from statsmodels.stats.contingency_tables import mcnemar
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n11 = int(np.sum(a & b)); n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b)); n00 = int(np.sum(~a & ~b))
    exact = (n10 + n01) < 25
    res = mcnemar([[n11, n10], [n01, n00]], exact=exact, correction=not exact)
    return {"n_a_only": n10, "n_b_only": n01, "statistic": float(res.statistic),
            "p_value": float(res.pvalue), "exact": bool(exact)}


# ---------------------------------------------------------------------------
# Encoder: bf16 + torch.compile, tuned for Ada (RTX 4070)
# ---------------------------------------------------------------------------
def build_encoder(model_name: str, compile_model: bool = True, dtype: str = "bfloat16"):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]

    model = SentenceTransformer(model_name, device="cuda")
    if torch_dtype != torch.float32:
        model = model.to(torch_dtype)
    model.eval()

    if compile_model:
        try:
            # dynamic=True avoids a recompile per unique sequence length, which
            # would otherwise erase the speedup on variable-length input.
            inner = model[0].auto_model
            model[0].auto_model = torch.compile(inner, dynamic=True)
            _ = model.encode(["warmup one", "warmup two"], batch_size=2,
                             show_progress_bar=False, convert_to_numpy=True)
            print(f"    encoder: {model_name} | {dtype} | torch.compile ON")
        except Exception as e:  # noqa: BLE001
            print(f"    encoder: {model_name} | {dtype} | compile FAILED ({type(e).__name__}), eager")
    else:
        print(f"    encoder: {model_name} | {dtype} | compile off")
    return model


def encode(model, texts, batch_size=256, normalize=True):
    import torch
    with torch.inference_mode():
        emb = model.encode(
            list(texts), batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=normalize,
        )
    return np.asarray(emb, dtype=np.float32)


def save_json(obj, name):
    path = os.path.join(ARTIFACTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  -> saved {path}")
    return path
