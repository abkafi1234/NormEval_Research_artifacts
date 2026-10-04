"""Experiment E (new): BM25 question->passage retrieval on Belebele. See EXPERIMENTAL_PLAN.md §6.

Belebele questions are written on FLORES-200 passages and are parallel across languages, so
content is held fixed. Each question (language L) ranks the 488 passages in L with Okapi BM25.
The relevant passage is its own. Conditions are applied identically to questions and passages:
  tok   : whitespace split + Unicode punctuation/symbols stripped from token edges (no case folding)
  ortho : FLORES orthographic stage (NFKC, lowercase, diacritic strip, punctuation removal) + tok
  full  : ortho + Snowball stemming                  (15 Snowball languages)
  bnltk / banlemma : Bangla tools on the raw text, then tok (as in the Bangla experiments)
Ties are ranked pessimistically (a tied gold passage is placed after the tied documents).
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata

import numpy as np
from scipy import sparse

import exp_common as C

LANGS = sorted(C.FLORES_SNOWBALL)
BANGLA = "ben_Beng"
PARAMS = {"k1=0.9,b=0.4": (0.9, 0.4), "k1=1.2,b=0.75": (1.2, 0.75)}
PRIMARY = "k1=0.9,b=0.4"


def _edge(ch):
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


def tok(text):
    out = []
    for w in text.split():
        i, j = 0, len(w)
        while i < j and _edge(w[i]):
            i += 1
        while j > i and _edge(w[j - 1]):
            j -= 1
        if i < j:
            out.append(w[i:j])
    return out


def tokens(text, lang, cond):
    if cond == "tok":
        return tok(text)
    if cond == "ortho":
        return tok(C.flores_ortho(text, lang))
    if cond == "full":
        st = C.snowball(C.FLORES_SNOWBALL[lang])
        return [st.stem(w) for w in tok(C.flores_ortho(text, lang))]
    if cond == "bnltk":
        st = C._bnltk()
        return [st.stem(w) for w in tok(text)]
    if cond == "banlemma":
        import banlemma
        return tok(banlemma.lemmatize(text))
    raise ValueError(cond)


def bm25_scores(doc_toks, q_toks, k1, b):
    vocab, rows, cols, vals = {}, [], [], []
    for i, d in enumerate(doc_toks):
        counts = {}
        for w in d:
            counts[w] = counts.get(w, 0) + 1
        for w, c in counts.items():
            rows.append(i); cols.append(vocab.setdefault(w, len(vocab))); vals.append(c)
    n_docs, n_voc = len(doc_toks), len(vocab)
    tf = sparse.coo_matrix((np.asarray(vals, float), (rows, cols)), shape=(n_docs, n_voc))
    dl = np.bincount(tf.row, weights=tf.data, minlength=n_docs)
    avgdl = dl.mean()
    df = np.bincount(tf.col, minlength=n_voc)
    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
    w = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)) * idf[tf.col]
    W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=(n_docs, n_voc))
    qr, qc, qv = [], [], []
    for i, q in enumerate(q_toks):
        counts = {}
        for t in q:
            j = vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            qr.append(i); qc.append(j); qv.append(c)
    Q = sparse.csr_matrix((np.asarray(qv, float), (qr, qc)), shape=(len(q_toks), n_voc))
    return (Q @ W.T).toarray(), n_voc


def ranks_of_gold(S, gold):
    g = S[np.arange(len(gold)), gold][:, None]
    return 1 + (S > g).sum(axis=1) + ((S == g).sum(axis=1) - 1)


def metrics(r):
    r = np.asarray(r)
    return {"R@1": float((r == 1).mean()), "R@10": float((r <= 10).mean()),
            "MRR@10": float(np.where(r <= 10, 1.0 / r, 0.0).mean()), "mean_rank": float(r.mean())}


def compare(ra, rb):
    ra, rb = np.asarray(ra), np.asarray(rb)
    r1a, r1b = ra == 1, rb == 1
    rra, rrb = np.where(ra <= 10, 1.0 / ra, 0.0), np.where(rb <= 10, 1.0 / rb, 0.0)
    d1 = C.bootstrap_ci(r1b.astype(float) - r1a.astype(float))
    dm = C.bootstrap_ci(rrb - rra)
    return {"delta_R@1": d1[0], "delta_R@1_ci95": [d1[1], d1[2]],
            "delta_MRR@10": dm[0], "delta_MRR@10_ci95": [dm[1], dm[2]],
            "mcnemar_R@1": C.mcnemar_paired(r1a, r1b)}


def run_language(lang, limit):
    from datasets import load_dataset
    t0 = time.time()
    ds = load_dataset("facebook/belebele", lang, split="test")
    passages = list(dict.fromkeys(ds["flores_passage"]))
    pid = {p: i for i, p in enumerate(passages)}
    questions = list(ds["question"])
    gold = np.array([pid[p] for p in ds["flores_passage"]])
    if limit:
        questions, gold = questions[:limit], gold[:limit]
    conds = ["tok", "ortho", "full"] if lang != BANGLA else ["tok", "ortho", "bnltk", "banlemma"]
    res = {"lang": lang, "n_questions": len(questions), "n_passages": len(passages), "conditions": {}}
    ranks = {}
    for cond in conds:
        d_toks = [tokens(p, lang, cond) for p in passages]
        q_toks = [tokens(q, lang, cond) for q in questions]
        entry = {"passage_types": len({w for d in d_toks for w in d}),
                 "passage_tokens": int(sum(len(d) for d in d_toks)),
                 "questions_with_no_indexed_term": None, "metrics": {}}
        for pname, (k1, b) in PARAMS.items():
            S, n_voc = bm25_scores(d_toks, q_toks, k1, b)
            r = ranks_of_gold(S, gold)
            entry["metrics"][pname] = metrics(r)
            if pname == PRIMARY:
                ranks[cond] = r
                entry["ranks"] = r.tolist()
                entry["questions_with_no_indexed_term"] = int((S.max(axis=1) == 0).sum())
        entry["sample"] = {"question": q_toks[0][:25], "passage": d_toks[gold[0]][:25]}
        res["conditions"][cond] = entry
        m = entry["metrics"][PRIMARY]
        C.log(f"{lang} {cond:8s}: R@1={m['R@1']:.4f} R@10={m['R@10']:.4f} MRR@10={m['MRR@10']:.4f} types={entry['passage_types']}")
    pairs = [("tok", "ortho"), ("ortho", "full"), ("tok", "full")] if lang != BANGLA else \
            [("tok", "ortho"), ("tok", "bnltk"), ("tok", "banlemma"), ("bnltk", "banlemma")]
    res["comparisons"] = {f"{a}->{b}": compare(ranks[a], ranks[b]) for a, b in pairs}
    res["seconds"] = round(time.time() - t0, 1)
    C.save(res, "bm25", f"{lang}.json")
    return res


def predictiveness(results):
    from scipy.stats import spearmanr
    tier_b = json.load(open(C.TIER_B))
    dims = {"CR_morph": "CR_morph", "ANLD": "ANLD_morph", "KL": "KL_morph", "IRS": "IRS_morph", "AES_beta1": "AES_morph"}
    langs = [l for l in LANGS if l in results and l in tier_b]
    targets = {}
    for name, (a, b, metric) in {"morph_dMRR@10": ("ortho", "full", "MRR@10"), "morph_dR@1": ("ortho", "full", "R@1"),
                                 "total_dMRR@10": ("tok", "full", "MRR@10"), "ortho_dMRR@10": ("tok", "ortho", "MRR@10")}.items():
        targets[name] = [results[l]["conditions"][b]["metrics"][PRIMARY][metric] - results[l]["conditions"][a]["metrics"][PRIMARY][metric]
                         for l in langs]
    out = {"n_languages": len(langs), "languages": langs, "targets": targets, "spearman": {}}
    for tname, y in targets.items():
        rows = []
        for dname, key in dims.items():
            rho, p = spearmanr([tier_b[l][key] for l in langs], y)
            rows.append([dname, float(rho), float(p)])
        order = np.argsort([r[2] for r in rows])          # Holm correction within each target
        m, adj, running = len(rows), [0.0] * len(rows), 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * rows[i][2]))
            adj[i] = running
        out["spearman"][tname] = {r[0]: {"rho": r[1], "p": r[2], "p_holm": adj[k]} for k, r in enumerate(rows)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="questions per language (smoke test)")
    ap.add_argument("--langs", default=",".join(LANGS + [BANGLA]))
    a = ap.parse_args()
    t0 = time.time()
    C.log(f"bm25 start | out={C.OUT}")
    results = {}
    for lang in a.langs.split(","):
        path = C.out_path("bm25", f"{lang}.json")
        results[lang] = json.load(open(path)) if C.done("bm25", f"{lang}.json") else run_language(lang, a.limit)
    summary = {"primary_params": PRIMARY,
               "table": {l: {c: r["conditions"][c]["metrics"] for c in r["conditions"]} for l, r in results.items()},
               "comparisons": {l: r["comparisons"] for l, r in results.items()}}
    if len([l for l in LANGS if l in results]) >= 5:
        summary["predictiveness"] = predictiveness(results)
    C.save(summary, "bm25", "summary.json")
    C.log(f"bm25 done in {(time.time() - t0) / 60:.1f} min")
