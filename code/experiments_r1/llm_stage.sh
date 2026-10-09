#!/usr/bin/env bash
# Stage 1: LLM on the remaining FLORES-200 languages, then every measurement that needs only FLORES and the
# classification corpora. Stage 2: LLM on Belebele, then BM25 with the LLM and the summary again.
source ~/Research/NormEvaluator_pypi/runs_r1/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True R1_LLM_BATCH=12
F=""; for c in arb_Arab deu_Latn rus_Cyrl fra_Latn spa_Latn ita_Latn por_Latn nld_Latn swe_Latn dan_Latn nob_Latn ron_Latn; do F="$F,flores:$c"; done
$PY runs_r1/r1_norm.py --method llm --sets ${F#,} >> runs_r1/logs/norm_llm.log 2>&1 || { echo "STAGE1_NORM_FAILED $(date)" > runs_r1/logs/llm_stage1.status; exit 1; }
$PY runs_r1/r1_flores.py --methods llm > runs_r1/logs/flores_llm.log 2>&1
$PY runs_r1/r1_ablation.py --methods stanza,bkit,llm --jobs 20 > runs_r1/logs/ablation_llm.log 2>&1
$PY runs_r1/r1_ud_fewshot.py --evaluate all > runs_r1/logs/heldout_all.log 2>&1
$PY runs_r1/r1_summary.py > runs_r1/logs/summary.log 2>&1
echo "STAGE1_DONE $(date)" > runs_r1/logs/llm_stage1.status
B=""; for c in ben_Beng eng_Latn fin_Latn hun_Latn arb_Arab deu_Latn rus_Cyrl fra_Latn spa_Latn ita_Latn por_Latn nld_Latn swe_Latn dan_Latn nob_Latn ron_Latn; do B="$B,belebele:$c"; done
$PY runs_r1/r1_norm.py --method llm --sets ${B#,} >> runs_r1/logs/norm_llm.log 2>&1 || { echo "STAGE2_NORM_FAILED $(date)" > runs_r1/logs/llm_stage2.status; exit 1; }
$PY runs_r1/r1_bm25.py --force > runs_r1/logs/bm25_all.log 2>&1
$PY runs_r1/r1_summary.py > runs_r1/logs/summary.log 2>&1
echo "STAGE2_DONE $(date)" > runs_r1/logs/llm_stage2.status
