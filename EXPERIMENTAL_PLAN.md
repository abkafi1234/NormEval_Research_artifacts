# Experimental plan: re-running the missing experiments on `fedora-diu`

**Date:** 2026-09-29. **Author decisions:** re-run the experiments whose outputs were not
retained, add a BM25 experiment, and use the GPU.

**Copies of this plan:**
- local: `NormEvaluator_pypi/paper update/records/08_experiments/EXPERIMENTAL_PLAN.md`
- remote: `fedora-diu:~/Research/NormEvaluator_pypi/EXPERIMENTAL_PLAN.md`

## 0. Rules
1. **Every setting below reproduces what `Paper/template3.tex` reports.** Deviations are listed
   in §8.
2. The re-run numbers become the paper's backed numbers. **If a headline finding does not
   replicate** (the Bangla corpus reversal, the German protocol disagreement, negative XNLI MPD,
   the β-dependent English ranking), **stop and ask the author before changing any claim.**
3. Every result is saved with its raw per-fold scores and per-item predictions, so that every
   statistic can be recomputed from files without re-running anything.
4. Nothing outside `~/Research/NormEvaluator_pypi/` is touched on the remote machine.

## 1. Environment
- **Machine:** `fedora-diu` (Fedora 43, 28 CPU cores, 15 GB RAM, NVIDIA RTX 3050 8 GB,
  driver 580, CUDA 13.0).
- **Python:** 3.12 in an isolated `uv` venv: `~/Research/NormEvaluator_pypi/.venv`.
- **Key packages:** torch 2.14.0+cu130, scikit-learn 1.9.1, transformers 5.17.0,
  sentence-transformers 6.1.0, spaCy 3.8.16 (en_core_web_sm 3.8.0), scipy 1.18.1, normeval
  0.1.4 (editable, from the project copy), bnltk, bnlp_toolkit (a gensim stub; word2vec is
  unused), BanLemma (the repository's 62-word sample dictionary, as in the paper). Full list:
  `env_freeze.txt`.
- **Precision:** all embeddings in fp32 on the GPU (checklist C9).

## 2. Experiment A: XNLI, six languages (Tables `multilingual_nli_results`, `cr_decomposition`, `multilingual_nli_results_dsp`, `ext-xnli`, `ext-seed-xnli`)
- **Data:** XNLI validation, 2,490 pairs each, for ar, de, en, es, fr, ru; English MultiNLI
  train (392,702 pairs) for the zero-shot classifier.
- **Normalization:** NFKC → lowercase → diacritic strip → punctuation removal
  (string.punctuation + extended Arabic) → language-specific Snowball stemming. Applied to
  premise and hypothesis separately.
- **A1 Intrinsic, via the `normeval` package:** CR, IRS, AES_{β=1}, ANLD, KL on the
  concatenated "premise hypothesis" texts; whitespace tokenizer; IRS encoder
  `paraphrase-multilingual-MiniLM-L12-v2`.
- **A2 CR decomposition:** V_raw, V_ortho (all steps except stemming), V_full; case-sensitive
  whitespace vocabularies of the concatenated texts.
- **A3 Zero-shot MPD:**
  - InferSent features [u, v, |u−v|, u⊙v] from L2-normalized MiniLM embeddings (1,536-d).
  - Logistic regression (C = 1.0, lbfgs, max_iter = 1000), trained once on raw MultiNLI and
    once on normalized MultiNLI.
  - Macro-F1 on each language's raw vs normalized validation set; MPD = F1_norm − F1_orig.
- **A4 In-language Traditional DSP:**
  - Features: MiniLM embeddings of the concatenated texts (as `calculate_traditional_dsp` does
    when an embedding model is set).
  - Classifier: the same logistic regression.
  - Folds: stratified 10-fold, seeds 42, 7 and 2024.
- **Saved per language and seed:** per-fold macro-F1 (orig, norm), out-of-fold predictions for
  every item, Wilcoxon p (scipy default, exact at n = 10), McNemar p, mean ± SD, 95% CI (t,
  df = 9), Cohen's d_z.

## 3. Experiment B: BTSD ablation, English and Bangla (Tables `comparison_multirow_final`, `btsd_dsp`, `ext-btsd-en`, `ext-bangla` BTSD part, `aes_beta_sensitivity`, `ext-seed-reversal`)
- **Data:** BTSD, 3,793 sentences (Simple 1,554 / Compound 1,196 / Complex 1,043), with its
  English translation.
- **English normalizers:** Porter, Snowball, WordNet (on lowercased `\w+` tokens) and spaCy
  lemmas (en_core_web_sm; parser and ner off). The original text is the raw English sentence.
- **Bangla normalizers:** BNLTK (stemmer per whitespace token) and BanLemma
  (`banlemma.lemmatize`).
- **Intrinsic:** via the `normeval` package with the whitespace tokenizer. IRS encoders:
  mean-pooled `distilbert-base-uncased` (English) and mean-pooled `csebuetnlp/banglabert`
  (Bangla).
- **Traditional DSP:**
  - Features: TF-IDF (whitespace tokenizer, lowercase = False), fit inside each fold.
  - Classifiers: LogisticRegression(), MultinomialNB(), SVC(random_state=42),
    RandomForestClassifier(random_state=42). n_jobs may be raised for speed; results are
    unchanged.
  - Folds: stratified 10-fold, seeds 42, 7 and 2024. The paper reports seeds 7 and 2024 only
    for the Bangla MNB comparison; all are saved.
- **Saved:** as in A4, per classifier.
- **B-AES_β:** the sweep β ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 4} from the re-run English CR and IRS.

## 4. Experiment C: Bangla sentiment robustness (Tables `final_comparison`, `final_comparison_dsp`, `ext-bangla` sentiment part, `ext-seed-reversal`)
- **Data:** the sentiment corpus; 9,162 rows, 8 unlabeled rows dropped → 9,154. Text column
  `Tense`; labels {0: Negative, 1: Positive, 2: Neutral} per `label.txt`.
- **Normalizers, encoder, DSP and saved outputs:** as in Experiment B (Bangla).

## 5. Experiment D: IRS encoder dependence (Table `matrix` failure mode; §5 text)
- **D1:** FLORES-200 ben_Beng (dev + devtest, 2,009 sentences), BNLTK vs BanLemma. CR, ANLD
  and KL, plus IRS with multilingual MiniLM. This reproduces the dropped tier D of
  `benchmark/tiers.py`.
- **D2 (same corpus, two encoders):** IRS for BNLTK and BanLemma on BTSD-bn and sentiment,
  with BanglaBERT (as in B/C) and with multilingual MiniLM. **This is a new check:** the
  original claim compared encoders on different corpora, and D2 separates encoder from
  corpus.

## 6. Experiment E (NEW): BM25 lexical retrieval on Belebele
- **Why:** the Introduction motivates normalization through lexical retrieval, but the earlier
  retrieval test used a dense encoder (plan objection R4).
- **Data:** Belebele test split; 900 questions per language, each written on a FLORES-200
  passage (488 unique passages). The data is parallel across languages, so content is held
  fixed.
- **Task:** monolingual retrieval. Each question (in language L) ranks all 488 passages in L;
  the relevant passage is its own.
- **Languages:** the 15 Snowball languages (arb_Arab, dan_Latn, deu_Latn, eng_Latn, fin_Latn,
  fra_Latn, hun_Latn, ita_Latn, nld_Latn, nob_Latn, por_Latn, ron_Latn, rus_Cyrl, spa_Latn,
  swe_Latn) plus Bangla (ben_Beng).
- **Conditions,** applied identically to questions and passages:
  1. `tok`: whitespace split, Unicode punctuation stripped from token edges, no case folding
     (the unnormalized baseline);
  2. `ortho`: the FLORES orthographic stage (NFKC, lowercase, diacritic strip, punctuation
     removal);
  3. `full`: `ortho` + Snowball stemming.
  - **Bangla:** `tok`, BNLTK and BanLemma, each applied to the tokenized raw text, as in the
    Bangla experiments. `ortho` is reported separately because it strips the hasanta.
- **Model:** Okapi BM25 with k1 = 0.9, b = 0.4 (the Anserini/Pyserini defaults used by Mr. TyDi
  and MIRACL); a sensitivity run with k1 = 1.2, b = 0.75.
- **Metrics:** R@1, R@10, MRR@10.
- **Statistics:** McNemar on per-query R@1 correctness; paired bootstrap 95% CIs (10,000
  resamples) on the ΔR@1 and ΔMRR@10.
- **Predictiveness:** Spearman ρ between the BM25 deltas (full − ortho; full − tok) and the
  FLORES tier-B intrinsic dimensions (CR_morph, ANLD, KL, IRS, AES) across the 15 languages,
  with Holm correction. **n = 15, so exploratory.**

## 7. Outputs and verification
- **Outputs:** `runs/outputs/{xnli,ablation,encoder_check,bm25}/*.json`, each file with
  metadata (script, seed, package versions, timestamp). Logs are in `runs/logs/`.
- **Comparison report:** `runs/outputs/comparison_report.md` puts every template3 table cell
  next to its re-run value, gives the difference, and flags whether each headline finding
  replicates.
- **Local copy:** all outputs are copied back to
  `paper update/records/08_experiments/outputs/`. There they are checked against the
  manuscript, the same way the FLORES artifacts were.

## 8. Known deviations from the original runs
- **Newer library versions** (see §1). The originals are unknown: a Windows machine with an
  RTX 4070, per `benchmark/common.py`. Small numeric differences are expected. Differences in
  the sign or significance of a headline finding are handled by Rule 2.
- **The GPU differs** (RTX 3050 vs RTX 4070). Embedding values may differ in the last digits.
- **New data sources:** Experiment D2 and Experiment E are new. Their results are additions, not
  replications.
- **Not run:** Classification and Generative DSP (the paper states they are not exercised);
  FLORES tiers A–C (already backed by saved artifacts).

## 9. Expected runtime
About 2–3 hours in total:
- XNLI: the MultiNLI encoding and two large logistic-regression fits dominate, ≈ 1 h;
- BTSD and sentiment DSP: ≈ 1 h, with units run in parallel;
- BM25 and encoder check: minutes.

Runs go in the background (nohup) with checkpointing, so an interruption loses at most one unit.

---
## Run log (appended as the run proceeds)
- **2026-09-29 23:52:** code copied to `fedora-diu:~/Research/NormEvaluator_pypi/runs/` (local
  copy in `08_experiments/code/`).
- **Smoke tests passed** for all five scripts (`runs/outputs_smoke/`).
  - The Traditional DSP replica matches `normeval.calculate_traditional_dsp` to ~1e-16 with
    identical p-values, for every unit and for XNLI.
  - Porter and Snowball outputs differ on 64/300 subsampled sentences; the identical smoke
    deltas are a small-sample effect, to be re-checked on the full data.
- **2026-09-29 23:56:** full run launched (`nohup setsid bash runs/run_all.sh`).
  Order: bm25 → encoder_flores → ablation → xnli.
- **23:57:** BM25 done (exit 0). Outputs copied to `08_experiments/outputs/bm25/`.
  - **Headline:** stemming improves BM25 MRR@10 in all 15 languages (+0.025 to +0.185; McNemar
    p ≤ 0.021 — corrected 2026-09-30: the largest p is 0.0202, Portuguese; first written as "≤ 0.02"). No intrinsic dimension significantly predicts the size of the gain (all
    p > 0.15; Holm ≈ 1).
  - **Bangla:** BanLemma +0.117 and BNLTK +0.075 over `tok`; BanLemma vs BNLTK p = 0.0004.
  - **Contrast with dense retrieval** (tier C), where normalization hurts in all 14 languages.
  - The k1 = 1.2, b = 0.75 sensitivity run is to be checked.
- **2026-09-30 00:07:** ablation done (9.4 min, 12 parallel workers, 96 DSP jobs + 8 package
  checks, exit 0). Encoder check done (exit 0).
- **Replication check (ablation):**
  - All 8 units' intrinsic metrics (CR, IRS, ANLD, KL) equal template3 to 4 decimals.
  - All 12 Bangla-reversal MNB cells (BNLTK/BanLemma × BTSD/sentiment × seeds 42/7/2024) equal
    template3 exactly (delta and Wilcoxon p). **The headline replicates.**
- **Encoder check:**
  - FLORES ben_Beng IRS gap (BNLTK − BanLemma): MiniLM 0.1267 vs BanglaBERT 0.0017.
  - Same corpus (D2): BTSD 0.0320 (MiniLM) vs 0.0038 (BanglaBERT); sentiment ≈ 0.0008 vs
    −0.0060.
  - So encoder dependence is also corpus-dependent; state this in the paper.
- **Parallelism (author request "do parallel processing and smoke testing first"):**
  - XNLI could not run concurrently with the ablation: 12 workers used ~11.4 GB of 15.7 GB RAM,
    and XNLI needs 6–8 GB.
  - A parallel MultiNLI normalizer was written and smoke-tested. From stdin (`python -`) it
    hung, because spawn workers cannot re-import a stdin script (a test-harness artifact). From
    a script file it passed: identical output on 20k texts.
  - It was **not** hot-swapped into the running XNLI (it saves only 2–3 min; a restart would
    cost ~10 min). It is kept as `runs/exp_xnli_parallel_candidate.py`.
- **Incident:** a `pkill -f` pattern matched its own SSH shell and closed the session. No
  experiment process was affected (verified by PID). Stuck test processes were then killed by
  exact PID.
- **00:13:** XNLI zero-shot features for raw MultiNLI encoded in 349 s; LR fitting (~7 cores).
- **2026-09-30 00:29:** XNLI done (22.2 min). **ALL_DONE, every stage exit 0.** All outputs,
  logs and `env_freeze.txt` copied to `08_experiments/outputs/`.
- **Comparison with template3** (`code/compare_report.py` → `comparison_report.md`):
  - 498 cells compared, 408 exact, 90 different. **All 90 differences are in the XNLI tables.**
    BTSD/sentiment/intrinsic/seed-reversal/AES_β cells: 0 differences.
  - The XNLI differences come from embedding numerics (newer library versions, different GPU).
    - Intrinsic metrics: exact for 5 languages; Spanish V_full 4664→4663 (one vocabulary word).
    - Zero-shot MPD shifts ≤ 0.006; all six remain negative. The classifiers converged (178/187
      iterations).
    - In-language deltas shift ≤ 0.004. Four significance flags near 0.05 change: ar seed 42
      becomes significant; ar seed 2024 and fr seed 7 become n.s.; ru seed 42 stays n.s.
  - **Headlines:** 6 of 7 replicate (Bangla reversal; negative MPD; German; German the only
    both-n.s. language; β flip; McNemar significance set).
  - Changed: the seed-42 Wilcoxon count is 4 (+ar) rather than 3.
- **BM25 sensitivity (k1 = 1.2, b = 0.75):** same sign in 15/15 languages; Bangla order
  unchanged.
- **Text error found** (not a re-run issue): template3 says German's zero-shot MPD is "the
  second-most negative"; it is third in both the paper's table and the re-run.
- **2026-09-30 — Results into the manuscript** (author: "go"). XNLI tables and text now use the
  re-run values. BM25 (`sec:bm25`, `tab:bm25`) and encoder dependence (`tab:encoder_dependence`)
  were added. Every number in the manuscript's 21 numeric tables (1,178 values) was checked
  against these outputs: 0 mismatches after one pre-existing rounding fix
  (`reproducibility/reports/paper_vs_outputs.md`).
- **Reproducibility package:** `paper update/reproducibility/` (code as run, all outputs, CSV
  tables, provenance, checksums).
