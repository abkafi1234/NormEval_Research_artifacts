# Tables generated from outputs/

Regenerate with `python code/tables/make_tables.py`. Full stored precision; no rounding.

| file | rows | read from | note |
|---|---:|---|---|
| `xnli_intrinsic_zero_shot.csv` | 6 | outputs/xnli/<lang>.json: intrinsic.*, zero_shot.* | Intrinsic metrics on concatenated premise+hypothesis (validation, n=2490); zero-shot F1 is macro-F1. |
| `xnli_cr_decomposition.csv` | 6 | outputs/xnli/<lang>.json: cr_decomposition.* | morph_share = ln(CR_morph) / ln(CR_total). |
| `xnli_in_language_dsp.csv` | 18 | outputs/xnli/<lang>.json: in_language_dsp.<seed>.* | 10-fold stratified CV, LogisticRegression on MiniLM embeddings; Wilcoxon exact on 10 paired fold scores. |
| `xnli_package_check.csv` | 6 | outputs/xnli/<lang>.json: package_check_seed42.* | normeval.calculate_traditional_dsp (seed 42) vs the fold-level replica used for the tables. |
| `xnli_zero_shot_classifiers.csv` | 2 | outputs/zero_shot_classifiers/clf_info.json | LogisticRegression(C=1, lbfgs, max_iter=1000) on InferSent features of MultiNLI train. |
| `ablation_intrinsic.csv` | 8 | outputs/ablation/summary.json: units.<unit>.intrinsic (= outputs/ablation/intrinsic/<unit>.json) | IRS uses distilbert-base-uncased (English) or csebuetnlp/banglabert (Bangla), mean-pooled; IRS_minilm is the multilingual MiniLM control for Bangla. |
| `ablation_dsp.csv` | 96 | outputs/ablation/summary.json: units.<unit>.dsp.<clf>.<seed> (= outputs/ablation/dsp/*.json) | 10-fold stratified CV on TF-IDF features (whitespace tokenizer, no lowercasing). |
| `ablation_package_check.csv` | 16 | outputs/ablation/summary.json: units.<unit>.package_check_seed42 | normeval.calculate_traditional_dsp vs the replica, seed 42. |
| `bm25_retrieval.csv` | 98 | outputs/bm25/summary.json: table.*; outputs/bm25/<lang>.json: conditions.* | Okapi BM25, question -> own-language passage (488 passages); ties ranked pessimistically. |
| `bm25_comparisons.csv` | 49 | outputs/bm25/summary.json: comparisons.* (primary parameters k1=0.9, b=0.4) | delta = second condition minus first; bootstrap 95% CIs over questions. |
| `bm25_sensitivity.csv` | 49 | derived from outputs/bm25/summary.json: table.* | Parameter-sensitivity check: does the sign of each MRR@10 change survive k1=1.2, b=0.75? |
| `bm25_predictiveness.csv` | 20 | outputs/bm25/summary.json: predictiveness.spearman | Intrinsic dimensions from FLORES tier B (morphological stage); Holm correction within each target. |
| `bm25_predictiveness_inputs.csv` | 15 | outputs/bm25/summary.json: predictiveness.targets + outputs/flores_benchmark/tier_b_intrinsic.json | The per-language vectors the Spearman correlations are computed from. |
| `encoder_dependence_irs.csv` | 6 | outputs/encoder_check/flores_ben.json; outputs/ablation/summary.json (bn units) | Same normalized texts scored by two encoders; the BNLTK-BanLemma IRS gap is the quantity of interest. |
| `flores_tier_a_orthographic.csv` | 204 | outputs/flores_benchmark/tier_a_orthographic.json: per_language | All 204 FLORES-200 variants. |
| `flores_multiscript.csv` | 16 | outputs/flores_benchmark/tier_a_orthographic.json: controlled_multiscript | Same language and sentences in two scripts. |
| `flores_by_script.csv` | 29 | outputs/flores_benchmark/tier_a_orthographic.json: by_script |  |
| `flores_tier_b_intrinsic.csv` | 15 | outputs/flores_benchmark/tier_b_intrinsic.json | 15 Snowball languages, dev+devtest (2,009 sentences). |
| `flores_tier_c_dense_retrieval.csv` | 14 | outputs/flores_benchmark/tier_c_retrieval.json: per_language | Dense cross-lingual retrieval with multilingual MiniLM (fp32). |
| `flores_precision_check.csv` | 4 | outputs/flores_benchmark/precision_check.json | Why the benchmark used fp32: bf16 moved Bangla IRS by 1.5e-3. |
| `dense_vs_sparse.csv` | 15 | outputs/bm25/summary.json + outputs/flores_benchmark/tier_c_retrieval.json | Same normalization pipeline, two retrieval consumers. English has no dense row (it is the retrieval pivot in tier C). |
