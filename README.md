# Reproducibility package

For: *Profiles, not rankings: Multidimensional multilingual evaluation of stemming,
lemmatization, and text normalization* (manuscript: `../paper/paper.tex`).

This folder holds the exact code that produced every reported number, every raw output, flat
CSV versions of those outputs, the software and hardware environment, and data provenance with
checksums.
- **Unmodified copies:** `code/experiments/`, `code/flores_benchmark/`, `code/normeval_0.1.4/`
  and everything under `outputs/`.
- **Added to check and document them:** this README, `environment/`, `tables/`, `reports/`,
  `code/tables/`, `code/verification/` (audit scripts) and `MANIFEST.sha256`.

## Layout

```
README.md                  this file
EXPERIMENTAL_PLAN.md       the pre-registered plan for the 2026-09-30 re-run, with its run log
MANIFEST.sha256            sha256 of every file in this folder (verify with the command below)
code/
  experiments/             exact scripts run on 2026-09-30 (byte-identical to the run directory)
    exp_common.py          shared paths, normalizers, encoders, statistics, DSP replica
    exp_xnli.py            XNLI: intrinsic metrics, CR decomposition, zero-shot MPD, in-language DSP
    exp_ablation.py        BTSD (EN, BN) and Bangla sentiment: intrinsic metrics, AES_beta sweep, DSP
    exp_bm25.py            BM25 question->passage retrieval on Belebele (new)
    exp_encoder_flores.py  IRS encoder-dependence check on FLORES-200 Bangla (new)
    run_all.sh             runs the four scripts in order: bm25, encoder_flores, ablation, xnli
    stubs/gensim/          import shim for bnlp-toolkit (see environment/provenance.md)
    exp_xnli_parallel_candidate.py   NOT USED for any reported number (see "Unused code")
    smoke_parallel_norm.py           smoke test for that candidate; NOT USED for any number
  flores_benchmark/        FLORES-200 benchmark (tiers A-C), run 2026-07-28/29
    artifacts -> ../../outputs/flores_benchmark   (symlink so the scripts find their inputs)
  normeval_0.1.4/          the normeval package source used (byte-identical to PyPI 0.1.4)
  tables/make_tables.py    outputs/*.json -> tables/*.csv (standard library only)
  figures/make_figures.py  tables/*.csv -> the manuscript's seven figures and the graphical
                           abstract (../paper/figures/*.pdf, graphical_abstract.tif)
  verification/            audit scripts (see "Verification")
outputs/
  xnli/                    6 JSON, one per language (all seeds, per-fold scores, predictions)
  ablation/                summary.json + per-unit intrinsic, per-(unit, classifier, seed) DSP,
                           package cross-checks, and the normalized texts themselves
  bm25/                    16 per-language JSON (ranks of every question) + summary.json
  encoder_check/           flores_ben.json
  zero_shot_classifiers/   the two fitted MultiNLI classifiers (.joblib) + clf_info.json
  flores_benchmark/        tier_a_orthographic, tier_b_intrinsic, tier_c_retrieval, precision_check
  logs/                    stdout/stderr of every script + run_all.log (start/exit times, exit codes)
  smoke_tests/             reduced-size smoke runs made before the full run
tables/                    flat CSVs generated from outputs/ (see tables/INDEX.md)
reports/
  paper_vs_outputs.md      every number in the manuscript's 21 numeric tables vs the stored outputs
  prose_numbers.md         decimal numbers in the manuscript's prose that need a manual trace
  figure_values.md         every value plotted in the figures, with the CSV it was read from
  comparison_report.md     every re-run number vs the value printed in the pre-revision manuscript
  dense_vs_sparse_correlation.txt   the rank correlation quoted in the BM25 section
  flores_predictiveness_output.txt, dimension_separability_output.txt, flores_summary_output.txt
environment/
  provenance.md            machine, packages, model and dataset revisions, data checksums
  env_freeze.txt           pip freeze of the run environment
  requirements.txt         the same, installable (normeval pinned to 0.1.4 from PyPI)
  data_checksums.sha256    sha256 of every local input file (BTSD, sentiment, BanLemma, FLORES)
```

## Where each table's numbers come from

`tables/INDEX.md` lists every CSV with the JSON fields it reads. By manuscript table:

| manuscript table (label) | CSV in `tables/` | raw output | produced by |
|---|---|---|---|
| XNLI intrinsic profile + zero-shot MPD (`tab:multilingual_nli_results`) | `xnli_intrinsic_zero_shot.csv` | `outputs/xnli/<lang>.json` | `exp_xnli.py` |
| XNLI compression decomposition (`tab:cr_decomposition`) | `xnli_cr_decomposition.csv` | same | `exp_xnli.py` |
| XNLI in-language Traditional DSP (`tab:multilingual_nli_results_dsp`) | `xnli_in_language_dsp.csv` (seed 42), `xnli_package_check.csv` | same | `exp_xnli.py` |
| XNLI extended statistics (`tab:ext-xnli`) | `xnli_in_language_dsp.csv` (seed 42) | same | `exp_xnli.py` |
| XNLI seed sensitivity (`tab:ext-seed-xnli`) | `xnli_in_language_dsp.csv` (seeds 42, 7, 2024) | same | `exp_xnli.py` |
| zero-shot classifier convergence (text) | `xnli_zero_shot_classifiers.csv` | `outputs/zero_shot_classifiers/clf_info.json` | `exp_xnli.py` |
| BTSD intrinsic profile (`tab:comparison_multirow_final`) | `ablation_intrinsic.csv` | `outputs/ablation/summary.json` | `exp_ablation.py` |
| AES_beta sweep (`tab:aes_beta_sensitivity`) | `ablation_intrinsic.csv` (`AES_beta_*` columns) | same | `exp_ablation.py` |
| BTSD Traditional DSP summary (`tab:btsd_dsp`) | `ablation_dsp.csv` (seed 42) | same | `exp_ablation.py` |
| Bangla sentiment intrinsic / DSP (`tab:final_comparison`, `tab:final_comparison_dsp`) | `ablation_intrinsic.csv`, `ablation_dsp.csv` | same | `exp_ablation.py` |
| extended BTSD / Bangla statistics (`tab:ext-btsd-en`, `tab:ext-bangla`) | `ablation_dsp.csv` (seed 42) | same | `exp_ablation.py` |
| Bangla reversal across seeds (`tab:ext-seed-reversal`) | `ablation_dsp.csv` (MultinomialNB, 3 seeds) | same | `exp_ablation.py` |
| BM25 retrieval on Belebele (new table) | `bm25_retrieval.csv`, `bm25_comparisons.csv`, `bm25_sensitivity.csv`, `bm25_predictiveness.csv` | `outputs/bm25/` | `exp_bm25.py` |
| dense vs sparse contrast (new, text/table) | `dense_vs_sparse.csv` | `outputs/bm25/summary.json`, `outputs/flores_benchmark/tier_c_retrieval.json` | derived |
| IRS encoder dependence (new table) | `encoder_dependence_irs.csv` | `outputs/encoder_check/flores_ben.json`, `outputs/ablation/summary.json` | `exp_encoder_flores.py`, `exp_ablation.py` |
| cross-script contrast (`tab:crossscript`) | `flores_multiscript.csv` | `outputs/flores_benchmark/tier_a_orthographic.json` | `flores_benchmark/run.py --full` |
| FLORES intrinsic profile (`tab:flores_intrinsic`) | `flores_tier_b_intrinsic.csv` | `tier_b_intrinsic.json` | `run.py --full` |
| FLORES dense retrieval (`tab:flores_retrieval`) | `flores_tier_c_dense_retrieval.csv` | `tier_c_retrieval.json` | `run.py --full` |
| FLORES predictiveness (`tab:flores_predict`) | `reports/flores_predictiveness_output.txt` | tiers B and C | `flores_benchmark/predictiveness.py` |
| script summary / all 204 variants (`tab:appendix_script_summary`, `tab:appendix_cr_ortho`) | `flores_by_script.csv`, `flores_tier_a_orthographic.csv` | `tier_a_orthographic.json` | `run.py --full`, `make_appendix_table.py` |

`tab:literature` (prior work), `tab:exp_design`, `tab:matrix` and `tab:ablation_summary` are
descriptive and contain no measured values.

## Status of each result

| result | status |
|---|---|
| BTSD (English, Bangla) and Bangla sentiment: all intrinsic values, all DSP cells, all seeds | Re-run 2026-09-30; **identical** to the pre-revision manuscript in every compared cell (`reports/comparison_report.md`). |
| XNLI | Re-run 2026-09-30. The manuscript now reports the re-run values. They differ from the pre-revision values by embedding numerics: zero-shot MPD by at most 0.006, in-language deltas by at most 0.004, and Spanish `V_full` by one word type. Four fold-level significance flags near p = 0.05 changed; all are reported as they now stand. |
| BM25 on Belebele; IRS encoder dependence | New in this revision (2026-09-30). |
| FLORES-200 tiers A-C | Original artifacts (2026-07-28/29), not re-run; every FLORES number in the manuscript was checked against them. |

## Reproduce

### 1. Environment

Python 3.12, then `pip install -r environment/requirements.txt`.
- torch 2.14.0 was the CUDA 13.0 build (`+cu130`); install it from the matching PyTorch index
  for your CUDA version.
- NLTK data: `python -m nltk.downloader wordnet omw-1.4`.
- The spaCy model `en_core_web_sm` 3.8.0 is pinned in `requirements.txt`.

A GPU is optional (fp32 is used throughout). The reported run used an RTX 3050 (8 GB).

### 2. Directory layout the scripts expect

`exp_common.py` resolves paths relative to its parent directory `ROOT`, laid out as on the run
machine:

```
ROOT/
  normeval-main/                                   (project repository)
    Reproducible Experiment/Dataset/...xlsx        (BTSD EN/BN, Bangla sentiment)
    Reproducible Experiment/bangla/BanLemma-main/  (BanLemma, with its sample dictionary)
    dataset/flores200_dataset/{dev,devtest}/       (FLORES-200)
    benchmark/artifacts/tier_b_intrinsic.json      (read by exp_bm25.py for predictiveness)
  runs/                                            <- copy code/experiments/* here
```

XNLI, MultiNLI and Belebele are downloaded from the Hugging Face Hub on first use. The
revisions used are in `environment/provenance.md`. Input-file checksums are in
`environment/data_checksums.sha256`.

### 3. Run

```
cd ROOT
bash runs/run_all.sh          # writes runs/outputs/, logs to runs/logs/; ~34 min on the reference machine
```

Individual scripts, as used for the smoke tests (`RUN_OUT` redirects the output folder;
`RUN_SEEDS` overrides seeds 42, 7, 2024):

```
RUN_OUT=runs/outputs_smoke python runs/exp_bm25.py --limit 60 --langs eng_Latn,deu_Latn,ben_Beng
RUN_OUT=runs/outputs_smoke python runs/exp_encoder_flores.py --limit 60
RUN_OUT=runs/outputs_smoke python runs/exp_ablation.py --limit 300 --jobs 8
RUN_OUT=runs/outputs_smoke python runs/exp_xnli.py --limit-train 3000 --limit 300 --langs de,ar
```

Scripts skip any output file that already exists, so an interrupted run resumes. Delete
`runs/outputs/` (and `runs/cache/xnli/` for the zero-shot classifiers) to force a full re-run.

FLORES-200 benchmark, from `normeval-main/benchmark/`: `python run.py --precision`, then
`python run.py --full` (fp32, 10,000 bootstrap resamples; artifacts go to `benchmark/artifacts/`).
`python predictiveness.py` and `python make_appendix_table.py` post-process them.

### 4. Tables

```
python code/tables/make_tables.py      # outputs/ -> tables/*.csv and tables/INDEX.md
```

### 5. Figures

```
python code/figures/make_figures.py    # tables/*.csv -> ../paper/figures/fig1..fig7 (PDF) and
                                       # graphical_abstract.pdf / .tif (Elsevier: 3000 x 1200 px, 300 dpi)
```

- **Needs:** matplotlib, numpy and scipy (used: 3.9.4, 2.0.2 and 1.13.1 under Python 3.9.6), and a
  LaTeX installation (TeX Live 2026 used), because the text is typeset with LaTeX to match the
  manuscript's fonts.
- **No new analysis:** every plotted value is read from `tables/*.csv`, and
  `reports/figure_values.md` lists each one with its source file.
- **Checks built in:**
  - The Spearman coefficients for dense retrieval in Figure 5b are recomputed from tiers B and
    C. The script stops unless they equal `reports/flores_predictiveness_output.txt`.
  - Each AES_β=1 in Figure 6a is checked against its definition from CR and IRS.
- **Graphical abstract:** set in Times (newtx), as Elsevier recommends. Its numbers (dense and
  BM25 changes in recall@1, Central Kanuri's two scripts, BanLemma's Multinomial NB deltas)
  are read from the same CSVs.
- **Figure 1** is conceptual. Its text condenses the manuscript's measurement matrix
  (`tab:matrix`) and data section.

### 6. Check the manuscript

```
python code/verification/check_paper_numbers.py      # tables in ../paper/paper.tex and
                                                     # ../supplementary/supplementary.tex vs tables/*.csv
python code/verification/dense_vs_sparse_correlation.py
```

### 7. Integrity

```
shasum -a 256 -c MANIFEST.sha256       # or: sha256sum -c MANIFEST.sha256
```

## Verification already performed

- **Manuscript against outputs:** `code/verification/check_paper_numbers.py` →
  `reports/paper_vs_outputs.md`.
  - It parses all 21 numeric tables of `../paper/paper.tex` (and `appendix_cr_table.tex`).
  - It compares 1,178 printed values with the stored outputs at the printed precision.
  - Result: 0 mismatches, after one pre-existing rounding slip was corrected (AES_β=1.5,
    Snowball: 0.5817 → 0.5818).
  - Re-run it after any manuscript edit; it exits non-zero on any disagreement.
- **Prose numbers:** `code/verification/check_prose_numbers.py` → `reports/prose_numbers.md`.
  - It lists every decimal number outside tables that cannot be traced to a stored value, for
    review by hand.
  - 2026-10-01, after the conversion to elsarticle (the highlights repeat one traced value): 253
    numbers; 244 traced, and the other 9 were checked by hand (literature values, the environment description, and
    margins derived from traced values).
- **Statistics quoted in the text:** `code/verification/dense_vs_sparse_correlation.py` →
  `reports/dense_vs_sparse_correlation.txt` (standard library; matches scipy to 10 digits).
- **Outputs:** the 151 files in `outputs/{xnli,ablation,encoder_check,bm25,logs}` were checksum-
  compared with the run directory on the re-run machine on 2026-09-30: all identical.
- **Code:** every script in `code/experiments/` is byte-identical to the run directory.
  `exp_xnli.py` is the file that was on disk when `run_all.sh` started it (modified 23:52;
  XNLI started 00:07:35). Its log shows sequential normalization.
- **Package:** `normeval` is byte-identical to PyPI 0.1.4.
- **DSP replica:** the fold-level replica used for the tables reproduces
  `normeval.calculate_traditional_dsp`. Deltas agree to about 1e-16 and Wilcoxon p-values are
  identical, for every BTSD/sentiment unit and every XNLI language
  (`tables/*package_check.csv`, column `abs_diff_delta`).
- **Convergence:** both zero-shot classifiers converged (178 and 187 lbfgs iterations of 1,000).
- **Against the pre-revision manuscript:** `code/verification/compare_report.py` →
  `reports/comparison_report.md`.
  - 498 cells were compared; 408 are exact.
  - All 90 differences are XNLI cells.
- **FLORES:** `code/verification/verify_flores.py` cross-checked every FLORES number in the
  manuscript against the artifacts. `predictiveness.py`, re-run on 2026-09-30, reproduces
  `tab:flores_predict` exactly (`reports/flores_predictiveness_output.txt`).
- `verify_flores.py`, `verify_internal.py` and `dimension_separability.py` use absolute paths
  from the author's machine. They are kept as records of the audit.

## Unused code, kept for transparency

- `exp_xnli_parallel_candidate.py`: a version of `exp_xnli.py` that normalizes MultiNLI in
  parallel.
  - It was written and smoke-tested during the run.
  - It was not swapped into the running job and produced no reported number.
  - Its smoke output is `outputs/smoke_tests/xnli/de.json` (`_meta.script` says so). It
    overwrote the earlier sequential smoke file for German.
- `summarize.py` in `code/flores_benchmark/` loads a tier-D artifact that `run.py --full` does
  not produce (tier D was dropped). It cannot run on these artifacts, and no reported number
  depends on it.

## Not included

- **Third-party datasets and model weights:** XNLI, MultiNLI, Belebele, FLORES-200 and the
  encoders. Revisions and checksums are given so the exact inputs can be obtained.
- **The BTSD and Bangla sentiment spreadsheets:** these are in the project repository
  (`normeval-main/Reproducible Experiment/Dataset/`), with checksums in
  `environment/data_checksums.sha256`. The normalized texts derived from them are in
  `outputs/ablation/texts/`.

## Revision R1: recent normalizers (added 2026-10)

Added for the first revision. Nothing above this section was changed, and no earlier file was
modified except this README, `outputs.zip` (rebuilt) and `MANIFEST.sha256` (regenerated).

**What was run.** Three recent normalizers under the protocol, controls, folds and seeds of the
original experiments. Results are in Section 5.7 and Tables 12-15 of the revised manuscript.

| Normalizer | What it is | Applied to |
|---|---|---|
| Stanza 1.15.0 lemmatizer | neural pipeline (tokenize, mwt, pos, lemma), one model per language | FLORES-200 (15 languages), Belebele (15), XNLI (6, with MultiNLI training text), English BTSD |
| Bkit 0.0.9 lemmatizer | rule-based Bangla lemmatizer of the Bkit toolkit (BanSuite) | Bangla BTSD, Bangla sentiment, Belebele Bangla |
| LLM lemmatizer | `Qwen/Qwen3-4B-Instruct-2507`, bitsandbytes 4-bit (nf4), greedy decoding, five in-context examples per language from Universal Dependencies | FLORES-200 (15 languages + Bangla), Belebele (16), English and Bangla BTSD, Bangla sentiment |

A lemmatizer is applied to the raw sentence, and the orthographic stage is applied to its
output, so the "full" condition is comparable with orthographic stage + Snowball. The Snowball
pipeline was measured again in the same run (`outputs/r1/flores/snowball.json`) and reproduces
`tab:flores_predict`.

**Where things are.**

- `code/experiments_r1/`: the scripts as run.
  - `r1_norm.py` produces the normalized text for one method and one text set and caches it.
    It holds the LLM instruction and the validity rule.
  - `r1_ud_fewshot.py` downloads the treebanks, picks the in-context examples and scores the
    LLM against held-out gold lemmas.
  - `r1_flores.py` (profile and dense retrieval), `r1_bm25.py` (Belebele), `r1_ablation.py`
    (classification corpora), `r1_xnli.py` (XNLI with Stanza), `r1_summary.py` (collects
    everything into `summary_r1.json`).
  - `make_r1_results.py` turns the stored outputs into the four tables and every number
    quoted in Section 5.7.
  - `env.sh`, `llm_stage.sh`, `xnli_stanza_cpu.sh`, `after_llm.sh`: the launchers. These use
    paths on the author's machine and are kept as records of the run.
- `outputs/r1/`: every raw output. `summary_r1.json` is the file the tables are built from.
- `outputs/r1/normalized_texts/<method>/<text set>.json`: input and output text of each new
  normalizer, with the settings used (`meta`). For the LLM, `meta` holds the system message,
  the instruction, the in-context examples, and the number of retried and fallback units.
- `outputs/r1/llm_in_context/`: the in-context examples (`fewshot.json`) and the held-out
  lemma accuracy of the LLM per language.
- `outputs/r1/logs/`: run logs.
- `environment/env_r1_overlay.txt`: packages added on top of `env_freeze.txt`.
- `tables/r1_*.csv`: flat tables behind Tables 12-15 and the quoted numbers.

**What a reader should know.**

- The LLM is one small model under one prompt. On held-out Universal Dependencies sentences it
  reproduced the gold lemma for 46-94% of all tokens but for 7-71% (median 26%) of the tokens
  whose lemma differs from the form (`tables/r1_llm_heldout_accuracy.csv`). The held-out score
  uses only sentences whose output has as many words as the gold sentence has tokens
  (`length_matched`, 12 to 58 of 60 sentences; 19 of 20 for Bengali).
- Of 99,709 sentence units sent to the LLM, 42 failed the length check twice and were left
  unlemmatized. The counts per text set are in each `meta`.
- The LLM was not run on XNLI: the 519,214 MultiNLI training sentences were beyond the
  available computation (one 8 GB GPU; about 0.75 s per sentence).
- The LLM run was stopped once for a timing check and restarted in two stages. Text sets
  finished before the stop were kept; the Arabic FLORES set, half done at the stop, was
  redone. `after_llm.sh` is the launcher of the first attempt and `llm_stage.sh` of the second.
  `r1_llm_bench.py` is the timing check; it produced no reported number.
- An 8-billion-parameter model of the same family was tried and did not fit in memory with the
  in-context examples. It produced no reported number.
- The Stanza output for the MultiNLI training sentences (74 MB) is not included; the XNLI
  results computed from it are in `outputs/r1/xnli_stanza/`.
- BanglaLem (a fine-tuned BanglaT5 lemmatizer) was considered and not run, because we could not
  find released trained weights.
