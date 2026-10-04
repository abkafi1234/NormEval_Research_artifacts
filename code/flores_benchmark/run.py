"""Benchmark driver.

  python run.py --smoke     fast sanity pass: 15 sentences/language, few
                            languages, tiny bootstrap. Verifies every code
                            path executes and the GPU stack is healthy.
  python run.py --full      the real run: all 204 languages, dev+devtest
                            (~2,009 sentences each), 10,000 bootstrap
                            resamples, paired significance tests.
  python run.py --precision compare bf16 vs fp32 embeddings before trusting
                            any bf16 number that reaches a table.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import build_encoder, encode, flores_languages, load_flores, orthographic, save_json
import tiers

# Same encoder the paper already uses, for methodological consistency.
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Precision policy. run.py --precision compares bf16 against fp32 on real
# FLORES text: the worst case (Bangla) shifts mean IRS by 1.5e-3. The paper
# argues from IRS differences as small as 0.006, so a bf16 artifact would be a
# quarter of the effect being claimed. We therefore keep fp32 for every number
# that reaches a table and rely on torch.compile plus large batches for speed;
# the cost is roughly five minutes on a full run.
DTYPE = "float32"


def precision_check(model_name=MODEL, n=400, compile_model=False):
    """bf16 speeds things up but perturbs embeddings slightly. IRS is reported
    to 4 decimals and the paper makes 'indistinguishable within 0.006' claims,
    so quantify the perturbation before relying on it."""
    print("\n=== Precision check: bfloat16 vs float32 ===")
    langs = ["eng_Latn", "ben_Beng", "arb_Arab", "rus_Cyrl"]
    enc32 = build_encoder(model_name, compile_model=False, dtype="float32")
    enc16 = build_encoder(model_name, compile_model=compile_model, dtype="bfloat16")

    out = {}
    for lc in langs:
        raw = load_flores(lc, limit=n)
        orth = [orthographic(t, lc) for t in raw]
        # IRS as the paper computes it: mean paired cosine similarity
        s32 = np.sum(encode(enc32, raw) * encode(enc32, orth), axis=1)
        s16 = np.sum(encode(enc16, raw) * encode(enc16, orth), axis=1)
        irs32, irs16 = float(s32.mean()), float(s16.mean())
        out[lc] = {
            "IRS_fp32": irs32, "IRS_bf16": irs16,
            "abs_diff": abs(irs32 - irs16),
            "max_per_sentence_abs_diff": float(np.max(np.abs(s32 - s16))),
        }
        print(f"  {lc}: fp32={irs32:.6f} bf16={irs16:.6f} |diff|={abs(irs32-irs16):.2e} "
              f"max/sent={out[lc]['max_per_sentence_abs_diff']:.2e}")

    worst = max(v["abs_diff"] for v in out.values())
    verdict = ("bf16 SAFE for reported IRS (worst |diff| < 1e-4)" if worst < 1e-4
               else "bf16 NOT safe at 4dp -- use fp32 for tabled IRS")
    print(f"  verdict: {verdict}")
    out["_verdict"] = {"worst_abs_diff": worst, "safe_at_4dp": bool(worst < 1e-4)}
    save_json(out, "precision_check.json")
    return out


def smoke():
    print("=" * 70)
    print("SMOKE TEST  (15 sentences/language, reduced scope, tiny bootstrap)")
    print("=" * 70)
    t0 = time.time()
    n, nb = 15, 200

    some = ["eng_Latn", "ben_Beng", "arb_Arab", "arb_Latn", "rus_Cyrl",
            "deu_Latn", "spa_Latn", "zho_Hans", "zho_Hant", "kas_Arab", "kas_Deva"]

    tiers.tier_a(limit=n, n_boot=nb, langs=some)

    enc = build_encoder(MODEL, compile_model=True, dtype=DTYPE)
    tiers.tier_b(limit=n, n_boot=nb, encoder=enc)
    tiers.tier_c(limit=n, n_boot=nb, encoder=enc,
                 langs=["ben_Beng", "arb_Arab", "deu_Latn", "spa_Latn"],
                 competence_floor=0.0)
    tiers.tier_d(limit=n, n_boot=nb, encoder=enc)

    print("\n" + "=" * 70)
    print(f"SMOKE TEST PASSED in {time.time()-t0:.1f}s -- all four tiers executed")
    print("=" * 70)


def full(n_boot=10000, competence_floor=0.50):
    from common import BENCHMARK_LANGS, BENCHMARK_QUERY_LANGS
    print("=" * 70)
    print("FULL RUN  (fp32, dev+devtest ~2,009 sentences/language, "
          f"{n_boot} bootstrap resamples)")
    print(f"  benchmark suite: {len(BENCHMARK_LANGS)} languages with complete "
          f"metric coverage\n  {BENCHMARK_LANGS}")
    print("=" * 70)
    t0 = time.time()

    # Tier A sweeps all 204 variants: the orthographic claim needs no
    # stemmer, so every language has complete data for it. Suite languages are
    # reported in the main tables; the rest support the cross-script analysis.
    tiers.tier_a(limit=None, n_boot=n_boot)
    print(f"[tier A done {time.time()-t0:.0f}s]")

    enc = build_encoder(MODEL, compile_model=True, dtype=DTYPE)
    tiers.tier_b(limit=None, n_boot=n_boot, encoder=enc)
    print(f"[tier B done {time.time()-t0:.0f}s]")

    # Tier D (FLORES Bangla) is intentionally not run. Bangla is the paper's
    # depth probe and is studied on its own corpora with a Bangla-specific
    # encoder; see the note in common.py.

    # Retrieval restricted to the suite: a normalization delta is only
    # meaningful where the full pipeline (including stemming) can be applied.
    tiers.tier_c(limit=None, n_boot=n_boot, encoder=enc,
                 langs=BENCHMARK_QUERY_LANGS,
                 competence_floor=competence_floor)
    print(f"[tier C done {time.time()-t0:.0f}s]")

    print("\n" + "=" * 70)
    print(f"FULL RUN COMPLETE in {(time.time()-t0)/60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--precision", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--competence-floor", type=float, default=0.50)
    a = ap.parse_args()
    if a.precision:
        precision_check()
    elif a.smoke:
        smoke()
    elif a.full:
        full(n_boot=a.n_boot, competence_floor=a.competence_floor)
    else:
        ap.print_help()
