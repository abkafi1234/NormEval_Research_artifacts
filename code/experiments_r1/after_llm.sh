#!/usr/bin/env bash
# Waits for the LLM normalization to finish, then runs the measurements that use it.
source ~/Research/NormEvaluator_pypi/runs_r1/env.sh
while pgrep -f "[r]1_norm.py --method llm" > /dev/null; do sleep 60; done
if grep -q Traceback runs_r1/logs/norm_llm.log; then echo "LLM_NORM_FAILED $(date)" > runs_r1/logs/after_llm.status; exit 1; fi
$PY runs_r1/r1_flores.py --methods llm > runs_r1/logs/flores_llm.log 2>&1
$PY runs_r1/r1_bm25.py --force > runs_r1/logs/bm25_all.log 2>&1
$PY runs_r1/r1_ablation.py --methods stanza,bkit,llm --jobs 20 > runs_r1/logs/ablation_llm.log 2>&1
$PY runs_r1/r1_ud_fewshot.py --evaluate all > runs_r1/logs/heldout_all.log 2>&1
$PY runs_r1/r1_summary.py > runs_r1/logs/summary.log 2>&1
echo "ALL_DONE $(date)" > runs_r1/logs/after_llm.status
