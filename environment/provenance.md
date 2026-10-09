# Provenance: software, hardware, models, data

Recorded 2026-09-30 from the machine that produced `outputs/` (except the FLORES benchmark;
see the last section).

## Re-run machine (XNLI, BTSD, Bangla sentiment, BM25, encoder check)

| item | value |
|---|---|
| host | `fedora` (reached as `fedora-diu`), run directory `~/Research/NormEvaluator_pypi/` |
| OS | Fedora Linux 43, kernel 6.18.4-200.fc43.x86_64, glibc 2.42 |
| CPU / RAM | Intel Core i7-14700K (28 logical CPUs), 15.7 GB RAM |
| GPU | NVIDIA GeForce RTX 3050, 8 GB; driver 580.119.02; CUDA 13.0 |
| Python | 3.12.12 (uv virtual environment) |
| key packages | torch 2.14.0+cu130, transformers 5.17.0, sentence-transformers 6.1.0, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.3, datasets 5.0.1, statsmodels 0.15.0, nltk 3.10.3, spaCy 3.8.16, pandas 3.0.6 |
| full pin list | `env_freeze.txt` (as frozen); `requirements.txt` (same, with the editable install replaced by `normeval==0.1.4`) |
| run | 2026-09-29 23:56:18 to 2026-09-30 00:29:50 (+06:00), `runs/run_all.sh`, every stage exit 0 (`outputs/logs/run_all.log`) |
| embeddings | fp32 throughout (no bf16, no torch.compile) |

Every output JSON carries a `_meta` block (save time, script, argv, package versions, GPU).

## normeval package

- Installed editable from `~/Research/NormEvaluator_pypi/normeval-main` (version 0.1.4).
- The source is byte-identical to the PyPI release `normeval==0.1.4`: the wheel
  `normeval-0.1.4-py3-none-any.whl` (sha256 `b62aa08a54167d2eb7188eec5a3f9c29f32b43dcdf5a3040aaf4fb1b13f6c6ab`,
  uploaded 2026-06-09) unpacks to the same two files. They are copied to
  `code/normeval_0.1.4/`:

| file | sha256 |
|---|---|
| `normeval/__init__.py` | `6290a6860ac08e0ac1634fe6b0dcb033c152d33b25444e594be2183b5cd0187c` |
| `normeval/evaluator.py` | `53e8195e439a6db001098cab452c83aa99dbd96f83d9d9fe300f0dc48a101a5d` |

## Pretrained models (Hugging Face revision resolved at run time)

| model | used for | revision |
|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | IRS (XNLI, FLORES, Bangla control), XNLI zero-shot and in-language features | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` |
| `distilbert-base-uncased` (mean-pooled) | IRS, English BTSD | `12040accade4e8a0f71eabdb258fecc2e7e948be` |
| `csebuetnlp/banglabert` (mean-pooled) | IRS, Bangla | `9ce791f330578f50da6bc52b54205166fb5d1c8c` |
| spaCy `en_core_web_sm` 3.8.0 | spaCy lemmatizer (BTSD English) | release wheel, see `env_freeze.txt` |
| NLTK data `wordnet`, `omw-1.4` | WordNet lemmatizer | `~/nltk_data/corpora` on the re-run machine |

## Datasets

Hugging Face datasets (revision of `main` resolved at run time):

| dataset | used for | revision |
|---|---|---|
| `facebook/xnli` (ar, de, en, es, fr, ru; validation, 2,490 pairs each) | XNLI experiments | `b8dd5d7af51114dbda02c0e3f6133f332186418e` |
| `nyu-mll/multi_nli` (train, 392,702 pairs) | zero-shot NLI classifiers | `da70db2af9d09693783c3320c4249840212ee221` |
| `facebook/belebele` (16 languages, test: 900 questions, 488 passages) | BM25 retrieval | `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` |

Local files (identical on the author's machine and on the re-run machine):

| file (under `normeval-main/`) | sha256 |
|---|---|
| `Reproducible Experiment/Dataset/Bangla Transformation of Sentence Dataset(BTSD).xlsx` | `3c563fec17174d4a41efd6a53a3db75ed7cc545523966b7c90b78c5d6a943b3e` |
| `Reproducible Experiment/Dataset/Bangla Transformation of Sentence Dataset(BTSD)_english translation.xlsx` | `a0b0555aa1614016943d74bb0fae7ae5558a90dae2263ac14ac6f0295fb5832c` |
| `Reproducible Experiment/Dataset/Bangla Sentiment Dataset/Sentiment dataset.xlsx` | `eb391cd77c038edad681a429d2f2d99bf8f4d8b30c696296c2bbd0cf10047c48` |
| `Reproducible Experiment/bangla/BanLemma-main/data/sample/dictionary.json` (BanLemma sample dictionary) | `e08abb312cf948280950ef7ec0aaaeca69a11a8e8024cc9c2a7bb6dd5645a067` |
| `Reproducible Experiment/bangla/BanLemma-main/banlemma/_lemmatize.py` | `6dfaa0ae401e3fac353217fbef9442b1523fb8418bfddf4a86593b624d04dda6` |
| `dataset/flores200_dataset/dev/ben_Beng.dev` | `aee7fe16ec08ee5997a1aeb85e6bba27072cbb108b193fa20ca9bfc084872f65` |
| `dataset/flores200_dataset/devtest/ben_Beng.devtest` | `6699aa77b4c93d520971868cd5ff06a1e3b5ddfde852da13337f5194bc23086f` |

`data_checksums.sha256` lists the full BanLemma tree used.

## Compatibility shim

`bnlp-toolkit` 4.4.1 imports `gensim` at import time for word2vec/doc2vec features this project
never calls. `code/experiments/stubs/gensim/` is a stub package that satisfies the import and
raises `NotImplementedError` if anything tries to use it. No reported number touches it.
`exp_common.py` puts `runs/stubs` on `sys.path`.

## FLORES-200 benchmark (tiers A to C): a different, earlier run

- Artifacts `outputs/flores_benchmark/*.json` were produced on 2026-07-28/29 by
  `code/flores_benchmark/run.py --full` (and `--precision`) on the author's workstation
  (RTX 4070, per the comments in `code/flores_benchmark/common.py`).
- Settings: fp32 embeddings, torch.compile, dev+devtest (2,009 sentences per variant), and
  10,000 bootstrap resamples.
- No package freeze was saved for that run. The artifacts were cross-checked against every
  FLORES number in the manuscript (`code/verification/verify_flores.py`), and were not
  regenerated.
- The Bangla FLORES encoder check (`outputs/encoder_check/`) belongs to the 2026-09-30 re-run,
  not to this benchmark.
