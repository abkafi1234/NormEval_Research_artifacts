#!/usr/bin/env bash
source ~/Research/NormEvaluator_pypi/runs_r1/env.sh
export CUDA_VISIBLE_DEVICES=
$PY runs_r1/r1_norm.py --method stanza --sets xnli:ar,xnli:de,xnli:en,xnli:es,xnli:fr,xnli:ru,mnli > runs_r1/logs/norm_stanza_xnli.log 2>&1 && $PY runs_r1/r1_xnli.py > runs_r1/logs/xnli_stanza.log 2>&1
echo XNLI_CHAIN_END >> runs_r1/logs/xnli_stanza.log
