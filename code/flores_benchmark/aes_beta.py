"""Generalized Adjusted Efficiency Score, AES_beta.

    AES_beta = (1 + b^2) * IRS * VRG / (b^2 * IRS + VRG),   VRG = 1 - 1/CR

Follows the Van Rijsbergen E-measure / F_beta construction. The term NOT
carrying b^2 in the denominator dominates as b grows, so:

    b -> 0    AES -> IRS   (semantic fidelity prioritized)
    b  = 1    AES  = harmonic mean (the balanced baseline)
    b -> inf  AES -> VRG   (compression prioritized; edge / memory-bound)

Both degenerate cases survive any beta: a do-nothing normalizer (VRG = 0) and a
meaning-destroying one (IRS = 0) each score 0.
"""
import numpy as np


def vrg_from_cr(cr):
    return 0.0 if cr is None or cr <= 0 else 1.0 - 1.0 / cr


def aes_beta(irs, vrg, beta=1.0):
    denom = (beta ** 2) * irs + vrg
    if denom <= 0:          # IRS = VRG = 0; package convention
        return 0.0
    return (1 + beta ** 2) * irs * vrg / denom


def _check():
    print("=" * 74)
    print("BOUNDARY BEHAVIOUR")
    print("=" * 74)
    irs, vrg = 0.90, 0.40
    print(f"  IRS={irs}  VRG={vrg}")
    for b in (0.01, 0.5, 1.0, 2.0, 100.0):
        print(f"    beta={b:<7} AES={aes_beta(irs, vrg, b):.6f}")
    print(f"    beta->0   should approach IRS ={irs}")
    print(f"    beta->inf should approach VRG ={vrg}")
    print(f"    beta=1    harmonic mean       ={2*irs*vrg/(irs+vrg):.6f}")

    print("\n  degenerate cases (must be 0 at every beta):")
    for b in (0.5, 1.0, 2.0):
        print(f"    beta={b}: identity(VRG=0)={aes_beta(0.95, 0.0, b):.6f}  "
              f"destructive(IRS=0)={aes_beta(0.0, 0.95, b):.6f}  "
              f"both-zero={aes_beta(0.0, 0.0, b):.6f}")

    print()
    print("=" * 74)
    print("DOES beta CHANGE THE RANKING?  (BTSD English ablation, paper values)")
    print("=" * 74)
    algos = {                       # CR,      IRS
        "Porter":   (2.0115, 0.8532),
        "Snowball": (2.0274, 0.8721),
        "WordNet":  (1.6820, 0.9430),
        "SpaCy":    (1.9539, 0.9372),
    }
    betas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
    print(f"  {'algorithm':10s} " + " ".join(f"b={b:<6}" for b in betas))
    scores = {}
    for name, (cr, irs) in algos.items():
        v = vrg_from_cr(cr)
        row = [aes_beta(irs, v, b) for b in betas]
        scores[name] = row
        print(f"  {name:10s} " + " ".join(f"{s:.4f} " for s in row))

    print(f"\n  {'winner':10s} " + " ".join(
        f"{max(scores, key=lambda k: scores[k][i])[:7]:<7}" for i in range(len(betas))))

    print("\n  Snowball vs SpaCy margin (SpaCy - Snowball):")
    for i, b in enumerate(betas):
        d = scores["SpaCy"][i] - scores["Snowball"][i]
        print(f"    beta={b:<5} {d:+.5f}  -> {'SpaCy' if d > 0 else 'Snowball'}")


if __name__ == "__main__":
    _check()
