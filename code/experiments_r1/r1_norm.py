"""Revision R1: normalize the study's texts with recent methods and cache the output.

Methods
  stanza : Stanza 1.15 neural pipeline (tokenize, mwt, pos, lemma), one model per language
  llm    : in-context lemmatization with an open-weight instruction model run locally in 4-bit
  bkit   : the lemmatizer of the Bkit toolkit (BanSuite), Bangla only
Text sets (the same texts the existing experiments use)
  flores:<code>    2,009 FLORES-200 sentences (dev + devtest)
  belebele:<code>  488 passages + 900 questions
  btsd_en          BTSD English translations, lowercased (as normalize_english_batch does)
  btsd_bn, sentiment   Bangla corpora, raw
Cache: runs_r1/cache/norm/<method>/<set>.json = {"raw": [...], "norm": [...], "meta": {...}}
The lemmatizer output is NOT post-processed here; the orthographic stage is applied by the
measurement scripts, exactly as for the Snowball pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
import exp_common as C  # noqa: E402

R1 = os.path.dirname(os.path.abspath(__file__))
NORM = os.path.join(R1, "cache", "norm")
LLM_MODEL = os.environ.get("R1_LLM", "Qwen/Qwen3-4B-Instruct-2507")

STANZA_LANG = {"arb_Arab": "ar", "dan_Latn": "da", "nld_Latn": "nl", "eng_Latn": "en", "fin_Latn": "fi",
               "fra_Latn": "fr", "deu_Latn": "de", "hun_Latn": "hu", "ita_Latn": "it", "nob_Latn": "nb",
               "por_Latn": "pt", "ron_Latn": "ro", "rus_Cyrl": "ru", "spa_Latn": "es", "swe_Latn": "sv"}
LANG_NAME = {"arb_Arab": "Arabic", "dan_Latn": "Danish", "nld_Latn": "Dutch", "eng_Latn": "English",
             "fin_Latn": "Finnish", "fra_Latn": "French", "deu_Latn": "German", "hun_Latn": "Hungarian",
             "ita_Latn": "Italian", "nob_Latn": "Norwegian", "por_Latn": "Portuguese", "ron_Latn": "Romanian",
             "rus_Cyrl": "Russian", "spa_Latn": "Spanish", "swe_Latn": "Swedish", "ben_Beng": "Bengali"}
LANGS15 = sorted(STANZA_LANG)
XNLI_CODE = {"ar": "arb_Arab", "de": "deu_Latn", "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "ru": "rus_Cyrl"}
BANGLA = "ben_Beng"


# --------------------------------------------------------------------------- text sets
def load_flores(code, limit=None):
    out = []
    for sp in ("dev", "devtest"):
        with open(os.path.join(C.FLORES, sp, f"{code}.{sp}"), encoding="utf-8") as f:
            out.extend(line.rstrip("\n") for line in f)
    return out[:limit] if limit else out


def load_belebele(code, limit=None):
    from datasets import load_dataset
    ds = load_dataset("facebook/belebele", code, split="test")
    passages = list(dict.fromkeys(ds["flores_passage"]))
    questions = list(ds["question"])
    if limit:
        passages, questions = passages[:limit], questions[:limit]
    return passages, questions


def text_set(name, limit=None):
    """Returns (language code, list of raw texts)."""
    if name.startswith("flores:"):
        code = name.split(":")[1]
        return code, load_flores(code, limit)
    if name.startswith("belebele:"):
        code = name.split(":")[1]
        p, q = load_belebele(code, limit)
        return code, p + q
    if name.startswith("xnli:"):
        from datasets import load_dataset
        lang = name.split(":")[1]
        ds = load_dataset("facebook/xnli", lang, split="validation")
        return XNLI_CODE[lang], list(ds["premise"]) + list(ds["hypothesis"])
    if name == "mnli":
        from datasets import load_dataset
        ds = load_dataset("nyu-mll/multi_nli", split="train")
        texts = list(ds["premise"]) + list(ds["hypothesis"])
        return "eng_Latn", texts[:limit] if limit else texts
    import exp_ablation as A
    if name in ("btsd_en", "btsd_bn"):
        bn, en, _ = A.load_btsd(limit)
        return ("eng_Latn", [str(t).lower() for t in en]) if name == "btsd_en" else (BANGLA, bn)
    if name == "sentiment":
        texts, _, _ = A.load_sentiment(limit)
        return BANGLA, texts
    raise ValueError(name)


def cache_path(method, name, limit=None):
    tag = name.replace(":", "__") + (f"__limit{limit}" if limit else "")
    return os.path.join(NORM, method, tag + ".json")


def sha(texts):
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def cached(method, name, limit=None):
    """Normalized texts for a set; raises if the cache is missing or was made from other texts."""
    d = json.load(open(cache_path(method, name, limit), encoding="utf-8"))
    return d


def lookup(method, name, limit=None):
    """dict raw text -> normalized text."""
    d = cached(method, name, limit)
    return dict(zip(d["raw"], d["norm"]))


# --------------------------------------------------------------------------- Stanza
def run_stanza(code, texts):
    import stanza
    lang = STANZA_LANG[code]
    procs = "tokenize,mwt,pos,lemma"
    try:
        nlp = stanza.Pipeline(lang, processors=procs, tokenize_no_ssplit=True, use_gpu=True, verbose=False)
    except Exception:  # languages without an MWT model
        nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma", tokenize_no_ssplit=True, use_gpu=True, verbose=False)
    out = []
    docs = [stanza.Document([], text=t if t.strip() else ".") for t in texts]
    B = 200
    for i in range(0, len(docs), B):
        if i and i % 50000 == 0:
            C.log(f"  stanza {lang}: {i}/{len(docs)}")
        for d in nlp(docs[i:i + B]):
            out.append(" ".join((w.lemma if w.lemma else w.text) for s in d.sentences for w in s.words))
    meta = {"stanza": stanza.__version__, "lang": lang, "processors": list(nlp.processors),
            "packages": {k: str(getattr(v, "config", {}).get("model_path", "")).split("/")[-1] for k, v in nlp.processors.items()}}
    del nlp
    return out, meta


# --------------------------------------------------------------------------- LLM
SYSTEM = ("You are a lemmatizer. You replace every word of a text with its lemma (dictionary form) "
          "and change nothing else.")
INSTR = ("Lemmatize the following {lang} text. Replace each word with its lemma (dictionary form) in {lang}. "
         "Keep the word order. Write lemmas in lowercase and leave out punctuation, as in the examples. "
         "Keep numbers and names. Do not translate, do not explain, do not add or remove words. "
         "Output only the lemmatized text on one line.\n\n"
         "{examples}"
         "Text: {text}\nLemmas:")
ENGLISH_EXAMPLE = "Example (English)\nText: The children were running to the oldest houses.\nLemmas: the child be run to the old house\n\n"


def examples_block(code):
    """In-context examples: K gold-lemmatized sentences of the language from a UD treebank
    (cache/ud/fewshot.json, written by r1_ud_fewshot.py); an English example if there are none."""
    f = os.path.join(R1, "cache", "ud", "fewshot.json")
    ex = json.load(open(f, encoding="utf-8")).get(code, {}).get("examples", []) if os.path.exists(f) else []
    if not ex:
        return ENGLISH_EXAMPLE
    return "".join("Example %d\nText: %s\nLemmas: %s\n\n" % (i + 1, e["text"], e["lemmas"]) for i, e in enumerate(ex))
SENT_SPLIT = re.compile(r"(?<=[.!?।؟])\s+")
_llm = {}


def llm_load():
    if _llm:
        return _llm["tok"], _llm["model"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, quantization_config=q, device_map={"": 0})
    model.eval()
    _llm.update(tok=tok, model=model)
    return tok, model


def llm_generate(prompts, n_words, batch_size):
    import torch
    tok, model = llm_load()
    chats = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
    order = sorted(range(len(chats)), key=lambda i: -len(chats[i]))
    out = [None] * len(chats)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([chats[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        max_new = int(min(400, 24 + 4.0 * max(n_words[i] for i in idx)))
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
        for i, g in zip(idx, gen):
            text = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out[i] = text.strip().split("\n")[0].strip()
    return out


def valid(src, hyp):
    """An output is accepted if it is non-empty and has between 0.6 and 1.4 times the input's words
    (punctuation is dropped and multiword tokens may be split, so the counts need not be equal)."""
    a, b = len(src.split()), len(hyp.split())
    return b > 0 and 0.6 * a - 1 <= b <= 1.4 * a + 1


def run_llm(code, texts, batch_size=int(os.environ.get("R1_LLM_BATCH", "12"))):
    lang = LANG_NAME[code]
    # sentence units, so long passages are generated in short pieces and rejoined
    units, owner = [], []
    for i, t in enumerate(texts):
        parts = [p for p in SENT_SPLIT.split(t.strip()) if p.strip()] or [t]
        for p in parts:
            units.append(p); owner.append(i)
    uniq = list(dict.fromkeys(units))
    block = examples_block(code)
    prompts = [INSTR.format(lang=lang, text=u, examples=block) for u in uniq]
    t0 = time.time()
    hyp = llm_generate(prompts, [len(u.split()) for u in uniq], batch_size)
    res = dict(zip(uniq, hyp))
    bad = [u for u in uniq if not valid(u, res[u])]
    retried = len(bad)
    if bad:   # one retry with the instruction restated; then fall back to the input, unchanged
        again = llm_generate([INSTR.format(lang=lang, text=u, examples=block) + " " for u in bad], [len(u.split()) for u in bad], max(1, batch_size // 2))
        for u, h in zip(bad, again):
            if valid(u, h):
                res[u] = h
    fallback = [u for u in uniq if not valid(u, res[u])]
    for u in fallback:
        res[u] = u
    out = [[] for _ in texts]
    for u, i in zip(units, owner):
        out[i].append(res[u])
    meta = {"model": LLM_MODEL, "quantization": "bitsandbytes 4-bit nf4, bf16 compute", "decoding": "greedy",
            "system": SYSTEM, "instruction": INSTR, "examples": block, "n_texts": len(texts), "n_units": len(uniq),
            "retried_units": retried, "fallback_units": len(fallback),
            "fallback_rate": len(fallback) / max(1, len(uniq)), "seconds": round(time.time() - t0, 1),
            "validity_rule": "non-empty and 0.6*n-1 <= output words <= 1.4*n+1"}
    return [" ".join(o) for o in out], meta


# --------------------------------------------------------------------------- Bkit
def run_bkit(code, texts):
    from bkit import lemmatizer
    out, failed = [], 0
    for t in texts:
        try:
            out.append(lemmatizer.lemmatize(t))
        except Exception:  # noqa: BLE001
            out.append(t); failed += 1
    return out, {"bkit": "0.0.9", "function": "bkit.lemmatizer.lemmatize", "failed_texts": failed}


RUN = {"stanza": run_stanza, "llm": run_llm, "bkit": run_bkit}


def plan(method):
    if method == "bkit":
        return [f"flores:{BANGLA}", f"belebele:{BANGLA}", "btsd_bn", "sentiment"]
    sets = [f"flores:{c}" for c in LANGS15] + [f"belebele:{c}" for c in LANGS15] + ["btsd_en"]
    if method == "llm":
        sets += [f"flores:{BANGLA}", f"belebele:{BANGLA}", "btsd_bn", "sentiment"]
    return sets


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(RUN))
    ap.add_argument("--sets", default=None, help="comma-separated; default: the full plan for the method")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for name in (a.sets.split(",") if a.sets else plan(a.method)):
        path = cache_path(a.method, name, a.limit)
        if os.path.exists(path):
            C.log(f"{a.method} {name}: cached"); continue
        code, raw = text_set(name, a.limit)
        t0 = time.time()
        uniq = list(dict.fromkeys(raw))
        norm, meta = RUN[a.method](code, uniq)
        assert len(norm) == len(uniq)
        meta.update(set=name, lang=code, n=len(uniq), sha256_raw=sha(uniq), seconds_total=round(time.time() - t0, 1),
                    versions=C.versions(), saved=time.strftime("%Y-%m-%d %H:%M:%S"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"raw": uniq, "norm": norm, "meta": meta}, f, ensure_ascii=False, indent=0)
        os.replace(path + ".tmp", path)
        extra = f" fallback={meta['fallback_units']}/{meta['n_units']}" if a.method == "llm" else ""
        C.log(f"{a.method} {name}: n={len(uniq)} {meta['seconds_total']}s{extra} | {uniq[0][:60]!r} -> {norm[0][:60]!r}")
