cd ~/Research/NormEvaluator_pypi
export PYTHONPATH=$PWD/runs_r1:$PWD/runs/stubs:$PWD/runs TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 RUN_OUT=$PWD/runs_r1/outputs
PY="$PWD/.venv_r1/bin/python -W ignore"
