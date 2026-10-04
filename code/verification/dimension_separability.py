"""Are the six dimensions separable in our own data? Spearman rank correlation between every
pair of intrinsic dimensions across the 15 FLORES-200 languages (artifacts), plus the ordering
of the four English BTSD algorithms per dimension (template3 Table comparison_multirow_final)."""
import itertools, json
from scipy.stats import spearmanr

ART = "/Users/kafi/Research/NormEvaluator_pypi/normeval-main/benchmark/artifacts/tier_b_intrinsic.json"
B = json.load(open(ART))
langs = sorted(B)
dims = {"CR_morph": "CR_morph", "KL": "KL_morph", "ANLD": "ANLD_morph", "IRS": "IRS_morph", "AES_b1": "AES_morph"}
print(f"FLORES-200, {len(langs)} languages: pairwise Spearman rho (p)")
for a, b in itertools.combinations(dims, 2):
    r, p = spearmanr([B[l][dims[a]] for l in langs], [B[l][dims[b]] for l in langs])
    print(f"  {a:8s} vs {b:8s}: rho={r:+.3f}  p={p:.3f}")
top = {d: max(langs, key=lambda l: B[l][dims[d]]) for d in dims}
print("  language ranked most severe/highest per dimension:", top)

btsd = {  # CR, IRS, AES, ANLD, KL  (template3, English BTSD)
    "Porter": (2.0115, 0.8532, 0.6328, 0.2311, 14.02), "Snowball": (2.0274, 0.8721, 0.6410, 0.2304, 13.12),
    "WordNet": (1.6820, 0.9430, 0.5671, 0.1229, 10.31), "SpaCy": (1.9539, 0.9372, 0.6420, 0.1796, 12.25)}
print("\nEnglish BTSD, order per dimension (highest first):")
for i, d in enumerate(["CR", "IRS", "AES", "ANLD", "KL"]):
    print(f"  {d:5s}: {' > '.join(sorted(btsd, key=lambda k: -btsd[k][i]))}")

# Does the beta parameter make AES carry information beyond CR_morph and IRS?
C = json.load(open(ART.replace("tier_b_intrinsic", "tier_c_retrieval")))["per_language"]
def aes(irs, vrg, b):
    d = b * b * irs + vrg
    return (1 + b * b) * irs * vrg / d if d > 0 else 0.0
print("\nAES_beta across FLORES: rank agreement with CR_morph and IRS, and prediction of retrieval loss")
q = [l for l in langs if l in C]
for beta in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
    a = {l: aes(B[l]["IRS_morph"], 1 - 1 / B[l]["CR_morph"], beta) for l in langs}
    r_cr, _ = spearmanr([a[l] for l in langs], [B[l]["CR_morph"] for l in langs])
    r_irs, _ = spearmanr([a[l] for l in langs], [B[l]["IRS_morph"] for l in langs])
    r_ret, p_ret = spearmanr([a[l] for l in q], [C[l]["delta_full"] for l in q])
    print(f"  beta={beta:<4}: rho(AES,CR_morph)={r_cr:+.3f}  rho(AES,IRS)={r_irs:+.3f}  "
          f"rho(AES,retrieval delta)={r_ret:+.3f} (p={p_ret:.3f})")
