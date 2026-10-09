"""Revision R1: BM25 on Belebele with recent normalizers as the morphological step.

Uses the functions of runs/exp_bm25.py unchanged (tokenizer, BM25, pessimistic tie ranking, metrics,
comparisons). Conditions: tok, ortho, full (Snowball) as in the paper, plus
  stanza, llm : tok(orthographic stage(lemmatizer output))          (15 languages)
  bkit, llm   : tok(lemmatizer output), like bnltk and banlemma     (Bangla)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
import exp_common as C  # noqa: E402
import exp_bm25 as E  # noqa: E402
import r1_norm as N  # noqa: E402


METHODS = ["stanza", "llm", "bkit"]


def run_language(lang, limit):
    t0 = time.time()
    passages, questions_u = N.load_belebele(lang)
    from datasets import load_dataset
    ds = load_dataset("facebook/belebele", lang, split="test")
    pid = {p: i for i, p in enumerate(passages)}
    questions = list(ds["question"])
    gold = np.array([pid[p] for p in ds["flores_passage"]])
    if limit:
        questions, gold = questions[:limit], gold[:limit]
    bangla = lang == N.BANGLA
    new = [m for m in (["bkit", "llm"] if bangla else ["stanza", "llm"])
           if m in METHODS and os.path.exists(N.cache_path(m, f"belebele:{lang}"))]
    maps = {m: N.lookup(m, f"belebele:{lang}") for m in new}

    def tokens(text, cond):
        if cond in maps:
            t = maps[cond][text]
            return E.tok(t) if bangla else E.tok(C.flores_ortho(t, lang))
        return E.tokens(text, lang, cond)

    conds = (["tok", "ortho", "bnltk", "banlemma"] if bangla else ["tok", "ortho", "full"]) + new
    res = {"lang": lang, "n_questions": len(questions), "n_passages": len(passages), "conditions": {}}
    ranks = {}
    for cond in conds:
        d_toks = [tokens(p, cond) for p in passages]
        q_toks = [tokens(q, cond) for q in questions]
        entry = {"passage_types": len({w for d in d_toks for w in d}), "passage_tokens": int(sum(len(d) for d in d_toks)), "metrics": {}}
        for pname, (k1, b) in E.PARAMS.items():
            S, _ = E.bm25_scores(d_toks, q_toks, k1, b)
            r = E.ranks_of_gold(S, gold)
            entry["metrics"][pname] = E.metrics(r)
            if pname == E.PRIMARY:
                ranks[cond] = r
                entry["ranks"] = r.tolist()
        res["conditions"][cond] = entry
        m = entry["metrics"][E.PRIMARY]
        C.log(f"{lang} {cond:8s}: R@1={m['R@1']:.4f} MRR@10={m['MRR@10']:.4f} types={entry['passage_types']}")
    base = "tok" if bangla else "ortho"
    morph = [c for c in conds if c not in ("tok", "ortho")]
    pairs = [("tok", "ortho")] + [(base, c) for c in morph] + ([("tok", c) for c in morph] if not bangla else [])
    pairs += [(a, b) for i, a in enumerate(morph) for b in morph[i + 1:]]
    res["comparisons"] = {f"{a}->{b}": E.compare(ranks[a], ranks[b]) for a, b in pairs}
    res["seconds"] = round(time.time() - t0, 1)
    C.save(res, "bm25", f"{lang}.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--methods", default="stanza,llm,bkit")
    ap.add_argument("--force", action="store_true", help="recompute languages that already have an output")
    ap.add_argument("--langs", default=",".join(N.LANGS15 + [N.BANGLA]))
    a = ap.parse_args()
    METHODS[:] = a.methods.split(",")
    results = {}
    for lang in a.langs.split(","):
        results[lang] = json.load(open(C.out_path("bm25", f"{lang}.json"))) if (C.done("bm25", f"{lang}.json") and not a.force) else run_language(lang, a.limit)
    C.save({"primary_params": E.PRIMARY,
            "table": {l: {c: r["conditions"][c]["metrics"][E.PRIMARY] for c in r["conditions"]} for l, r in results.items()},
            "comparisons": {l: r["comparisons"] for l, r in results.items()}}, "bm25", "summary.json")
