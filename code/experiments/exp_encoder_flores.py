"""Experiment D1: FLORES-200 Bangla (ben_Beng, dev+devtest), BNLTK vs BanLemma.
Reproduces the dropped tier D (multilingual MiniLM IRS) and adds BanglaBERT IRS on the same
sentences, so encoder and corpus are not confounded. See EXPERIMENTAL_PLAN.md §5.
"""
from __future__ import annotations

import argparse
import os
import time

import exp_common as C


def load_flores(code, splits=("dev", "devtest"), limit=None):
    out = []
    for sp in splits:
        with open(os.path.join(C.FLORES, sp, f"{code}.{sp}"), encoding="utf-8") as f:
            out.extend(line.rstrip("\n") for line in f)
    return out[:limit] if limit else out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    from normeval import NormalizationEvaluator
    t0 = time.time()
    C.log(f"encoder check (FLORES ben_Beng) start | out={C.OUT}")
    bn = load_flores("ben_Beng", limit=a.limit)
    res = {"corpus": "FLORES-200 ben_Beng dev+devtest", "n": len(bn), "tools": {}}
    for tool in ("bnltk", "banlemma"):
        norm = C.normalize_bangla_batch(bn, tool)
        ev = NormalizationEvaluator(bn, norm, embedding_model=C.encoder(C.MINILM), tokenizer=C.whitespace)
        cr = ev.calculate_cr()
        irs_m = ev.calculate_irs(batch_size=256)
        ev_b = NormalizationEvaluator(bn, norm, embedding_model=C.encoder(C.BANGLABERT, mean_pooled=True), tokenizer=C.whitespace)
        irs_b = ev_b.calculate_irs(batch_size=64)
        r = {"CR": cr, "ANLD": ev.calculate_anld(), "KL": ev.calculate_kl_divergence(),
             "IRS_minilm": irs_m, "IRS_banglabert": irs_b,
             "AES_beta1_minilm": ev.calculate_aes(cr, irs_m), "AES_beta1_banglabert": ev.calculate_aes(cr, irs_b),
             "sample": {"orig": bn[:2], "norm": norm[:2]}}
        res["tools"][tool] = r
        C.log(f"{tool}: CR={cr:.4f} ANLD={r['ANLD']:.4f} KL={r['KL']:.3f} IRS(MiniLM)={irs_m:.4f} IRS(BanglaBERT)={irs_b:.4f}")
    t = res["tools"]
    res["irs_gap"] = {"minilm": t["bnltk"]["IRS_minilm"] - t["banlemma"]["IRS_minilm"],
                      "banglabert": t["bnltk"]["IRS_banglabert"] - t["banlemma"]["IRS_banglabert"]}
    C.save(res, "encoder_check", "flores_ben.json")
    C.log(f"encoder check done in {time.time() - t0:.0f}s")
