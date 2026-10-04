"""Smoke test: parallel MultiNLI normalization must equal the sequential result exactly."""
import time
import exp_common as C
import exp_xnli_parallel_candidate as X
from datasets import load_dataset

if __name__ == "__main__":
    prem = list(load_dataset("nyu-mll/multi_nli", split="train")["premise"][:20000])
    t0 = time.time(); seq = [C.xnli_full(t, "en") for t in prem]; ts = time.time() - t0
    t0 = time.time(); par = X.parallel_normalize(prem, "en", procs=4); tp = time.time() - t0
    print(f"identical={seq == par} n={len(prem)} sequential={ts:.1f}s parallel(4 procs)={tp:.1f}s example={par[0][:70]!r}", flush=True)
