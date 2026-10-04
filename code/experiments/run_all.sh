#!/usr/bin/env bash
# Runs all experiments sequentially (see EXPERIMENTAL_PLAN.md). Logs go to runs/logs/.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD/runs:${PYTHONPATH:-}"
mkdir -p runs/logs
for step in exp_bm25 exp_encoder_flores exp_ablation exp_xnli; do
  echo "=== $step start $(date '+%F %T')" | tee -a runs/logs/run_all.log
  python "runs/$step.py" > "runs/logs/$step.log" 2>&1
  echo "=== $step exit=$? $(date '+%F %T')" | tee -a runs/logs/run_all.log
done
echo "ALL_DONE $(date '+%F %T')" | tee -a runs/logs/run_all.log
