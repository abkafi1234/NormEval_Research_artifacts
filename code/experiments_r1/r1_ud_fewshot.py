"""Revision R1: in-context examples and a held-out accuracy check for the LLM lemmatizer.

Downloads one Universal Dependencies treebank per language (dev split, or test where no dev split
exists), takes the first K sentences of 6-25 words as in-context examples (text and gold lemmas),
and the next N as a held-out check. Writes cache/ud/fewshot.json. With --evaluate it runs the LLM
lemmatizer of r1_norm.py on the held-out sentences and reports lemma accuracy against gold.
Gold lemmas are lowercased; Arabic diacritics and the Finnish compound marker '#' are removed, so
that examples show the unvocalized, unmarked forms the orthographic stage would produce anyway.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
import exp_common as C  # noqa: E402

R1 = os.path.dirname(os.path.abspath(__file__))
UD = os.path.join(R1, "cache", "ud")
TREEBANK = {"arb_Arab": ("UD_Arabic-PADT", "ar_padt"), "dan_Latn": ("UD_Danish-DDT", "da_ddt"),
            "nld_Latn": ("UD_Dutch-Alpino", "nl_alpino"), "eng_Latn": ("UD_English-EWT", "en_ewt"),
            "fin_Latn": ("UD_Finnish-TDT", "fi_tdt"), "fra_Latn": ("UD_French-GSD", "fr_gsd"),
            "deu_Latn": ("UD_German-GSD", "de_gsd"), "hun_Latn": ("UD_Hungarian-Szeged", "hu_szeged"),
            "ita_Latn": ("UD_Italian-ISDT", "it_isdt"), "nob_Latn": ("UD_Norwegian-Bokmaal", "no_bokmaal"),
            "por_Latn": ("UD_Portuguese-Bosque", "pt_bosque"), "ron_Latn": ("UD_Romanian-RRT", "ro_rrt"),
            "rus_Cyrl": ("UD_Russian-SynTagRus", "ru_syntagrus"), "spa_Latn": ("UD_Spanish-AnCora", "es_ancora"),
            "swe_Latn": ("UD_Swedish-Talbanken", "sv_talbanken"), "ben_Beng": ("UD_Bengali-BRU", "bn_bru")}
K, N_EVAL = 5, 60


def clean_lemma(lemma, code):
    lemma = lemma.lower().replace("#", "")
    if code == "arb_Arab":
        lemma = "".join(c for c in unicodedata.normalize("NFKD", lemma) if not unicodedata.combining(c))
    return lemma


def fetch(code):
    repo, stem = TREEBANK[code]
    os.makedirs(UD, exist_ok=True)
    for branch in ("master", "dev"):
        for split in ("dev", "test"):
            path = os.path.join(UD, f"{stem}-ud-{split}.conllu")
            if os.path.exists(path):
                return path, split
            url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/{branch}/{stem}-ud-{split}.conllu"
            try:
                urllib.request.urlretrieve(url, path)
                return path, split
            except Exception:  # noqa: BLE001
                if os.path.exists(path):
                    os.remove(path)
    raise SystemExit(f"no treebank file for {code}")


def sentences(path, code):
    out, text, toks = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# text ="):
            text = line[len("# text ="):].strip()
        elif line and not line.startswith("#"):
            f = line.split("\t")
            if "-" in f[0] or "." in f[0]:
                continue
            toks.append((f[1], clean_lemma(f[2], code), f[3]))
        elif not line:
            if text and toks:
                out.append((text, toks))
            text, toks = None, []
    return out


def build():
    data = {}
    for code in TREEBANK:
        path, split = fetch(code)
        sents = [s for s in sentences(path, code) if 6 <= len(s[1]) <= 25 and all(t[1] not in ("_", "") for t in s[1])]
        ex = [{"text": t, "lemmas": " ".join(l for _, l, u in toks if u != "PUNCT")} for t, toks in sents[:K]]
        ev = [{"text": t, "forms": [w for w, _, u in toks if u != "PUNCT"], "lemmas": [l for _, l, u in toks if u != "PUNCT"]}
              for t, toks in sents[K:K + N_EVAL]]
        data[code] = {"treebank": TREEBANK[code][0], "split": split, "examples": ex, "eval": ev}
        C.log(f"{code}: {TREEBANK[code][0]} {split}: {len(sents)} usable sentences, {len(ex)} examples, {len(ev)} held out")
    json.dump(data, open(os.path.join(UD, "fewshot.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return data


def evaluate(codes):
    import r1_norm as N
    data = json.load(open(os.path.join(UD, "fewshot.json"), encoding="utf-8"))
    res = {}
    for code in codes:
        ev = data[code]["eval"]
        hyp, meta = N.run_llm(code, [e["text"] for e in ev])
        tot = ok = same = changed_gold = changed_ok = 0
        for e, h in zip(ev, hyp):
            h = [w for w in C.flores_ortho(h, code).split()]
            g = [C.flores_ortho(l, code) for l in e["lemmas"]]
            f = [C.flores_ortho(w, code) for w in e["forms"]]
            keep = [i for i, x in enumerate(g) if x]
            g, f = [g[i] for i in keep], [f[i] for i in keep]
            if len(h) != len(g):
                continue
            same += 1
            for a, b, w in zip(h, g, f):
                tot += 1; ok += a == b
                if b != w:
                    changed_gold += 1; changed_ok += a == b
        res[code] = {"model": N.LLM_MODEL, "n_sentences": len(ev), "length_matched": same, "tokens": tot,
                     "lemma_accuracy": ok / max(1, tot), "accuracy_on_inflected": changed_ok / max(1, changed_gold),
                     "share_inflected": changed_gold / max(1, tot), "fallback_units": meta["fallback_units"], "seconds": meta["seconds"]}
        C.log(f"{code}: acc={res[code]['lemma_accuracy']:.3f} acc_inflected={res[code]['accuracy_on_inflected']:.3f} "
              f"matched={same}/{len(ev)} sec={meta['seconds']}")
    tag = N.LLM_MODEL.split("/")[-1]
    json.dump(res, open(os.path.join(UD, f"heldout_accuracy__{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", default=None, help="comma-separated language codes, or 'all'")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(UD, "fewshot.json")):
        build()
    if a.evaluate:
        evaluate(sorted(TREEBANK) if a.evaluate == "all" else a.evaluate.split(","))
