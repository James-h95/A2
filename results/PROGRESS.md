# A2 progress log

## Stage 0 — Preconditions (2026-10-07, first attempt) — BLOCKED, since resolved

- `A2/data/` exists but is empty: no TweetQA files.
- `A2/template.ipynb` is missing. `../A1_dataset/CITS4012_Assignment_Template.ipynb` is the A1 template, so it was not used.
- Spec read (`../CITS4012_A2_2026.pdf`). It agrees with the plan: main model from scratch with attention, no external data, test set only for final evaluation (validation may be split for tuning), one notebook built on the provided template with saved logs, ACL report of up to 6 pages.
- Environment: Apple M4 Pro with 24 GB. There is no conda env for this project. Only `regtr38` has torch (2.4.1, MPS works). A dedicated env (`cits4012-a2`) is still to be created.
- Waiting on the team to download the data and template. Stage 0 resumes after that.

## Stage 0 — Data report (2026-10-07) — CHECKPOINT: awaiting split approval

**Built**
- Copied `../TweetQA/{train,test}.csv` into `data/` (checksums match) and `../CITS4012_YourGroupID.ipynb` into `template.ipynb`.
- Created conda env `cits4012-a2` (Python 3.11, torch 2.14.1 with MPS, transformers 5.19, nltk 3.10, gensim 4.4). Install log: `results/logs/env_create.log`.
- `scripts/data_report.py` writes `results/DATA_REPORT.md`. Statistics come from train only.

**Key numbers**
- train.csv has 10,692 rows over 5,119 tweets, 1 gold answer each. test.csv has 1,086 rows over 583 tweets, 2 gold answers each. test.csv is the original TweetQA dev set.
- **No validation split is provided.** Test does have gold answers.
- Answers are short: p50 is 2 tokens, p95 is 7, 99.8% are ≤ 10. The default span cap L=15 is generous, so L=10 is worth including in the sweep.
- Tweets: p50 32 tokens, max 77. Questions: p50 8. Question first-words: what 42%, who 28%, how 8%, where 7%.
- Extractiveness: 57.5% of answers are raw substrings of the tweet and 64.8% are after Unicode-aware normalisation. In 13.7% no answer token appears at all (number words vs digits, split handles like "nws houston" vs @NWSHouston, metadata questions, non-English tweets).
- 97% of tweets end with a `Name (@handle) Month D, YYYY` byline, and 24.9% of answers lie inside it. This supports A3 (byline-aware self-attention) and the A2 normaliser.
- Every tweet has an @mention (the byline), 38% have hashtags, 6.7% have emoji, 2% have elongation. There are no URLs or HTML entities because the dataset already strips them. No answer contains an emoji, so the normaliser maps emoji to `<emoji>`. 124 answers contain `#` and 152 contain `@`.
- Noise: 46 exact-duplicate rows and 4 "tweet does not exist" rows in train.
- Template sections: `# Readme`, `# 1.Dataset Processing`, `# 2. Model Implementation`, `# 3.Testing and Evaluation`. Headings must not be modified.

**Surprises**
- Normalisation must strip Unicode punctuation. With ASCII-only stripping, matches against `—`, `’` and `“` were silently lost (58.7% → 64.8% extractive). Stage 1's evaluator will use the same rule.
- Validation carved from train has 1 gold answer per example, but test has 2. Validation scores will therefore run systematically below test scores because the metrics take the max over golds. The report must say this.

**Proposed split (needs team approval)**
- Carve validation from train: about 10% of tweets (about 512 tweets, about 1,070 questions). The split is **grouped by tweet** so no tweet appears in both, stratified by the question first-word of each tweet's first question, seed 4012. The rest (about 9,600) is train.
- Leave `test.csv` untouched as the locked test set.
- Drop exact-duplicate rows from train/val before splitting. Keep the 4 noise rows (too few to matter) and record the choice.
- **Approved by the team (2026-10-07):** 10% validation carved from train, grouped by tweet and stratified by question word, seed 4012. Exact-duplicate rows are dropped. test.csv stays locked.

## Stage 1 — Evaluation, preprocessing, silver spans (2026-10-07) — DONE

**Built**
- `src/evaluate.py`: Unicode-aware normaliser; BLEU-1 (nltk, smoothing method1), METEOR (nltk + WordNet), ROUGE-L (own LCS, F with β=1.2, the coco/TweetQA convention), EM and token-F1. Each is the max over golds; the corpus score is the mean ×100.
- `src/data.py`:
  - Loading with numpy-repr answer parsing; grouped and stratified split (`get_splits`); `TEST_LOCKED` guard.
  - Regex tokeniser that keeps char offsets.
  - Switchable normaliser (`@user→<user>`, url→`<url>`, emoji→`<emoji>`, `#tag→# tag`, elongation squash). The char-CNN always sees the raw token.
  - 8 token flags (mention, hashtag, url, emoji, number, capitalised, **in_byline**, exact match with the other sequence).
  - Silver spans: best span plus a soft set with F1≥τ and weights ∝ F1. Spans must start and end on content tokens.
  - Vocabs from train only, min_freq 2. Padded DataLoader.
- `src/runlog.py` appends to `results/runs.csv`. `tests/test_pipeline.py` has 16 tests, all passing.
- `scripts/stage1_oracle.py` writes `results/logs/stage1_oracle.log`, `results/splits.json` and `results/stage1_summary.json`.

**Key numbers**
- Split: train 9,570 questions / 4,608 tweets; val 1,076 / 511. Disjoint by tweet. 46 duplicates dropped.
- **Oracle (extractive ceiling), L=15, τ=0.5:**
  - val: BLEU-1 71.76, METEOR 55.10, ROUGE-L 75.97, EM 60.78, F1 77.00
  - train: BLEU-1 74.48, METEOR 58.54, ROUGE-L 78.26, EM 64.19, F1 79.17
- L ∈ {10, 15, 20} and normaliser on/off leave the oracle unchanged (≤0.1 difference), because answers are short.
- 84.9% of val examples have a span with F1>0. 82–84% are "trainable" (best F1 ≥ τ); only these contribute to the training loss.
- Vocab: 15,590 words and 213 chars. Val tweet-token OOV 7.2%. **22% of val examples have an OOV token inside the best span**, which motivates the char-CNN (A1).

**Surprises**
- METEOR caps an identical answer at 1 − 0.5/m³ (fragmentation penalty), so even the oracle scores only 55 METEOR. With 2-token answers, perfect predictions get 93.75.
- The soft set is wide (about 16 spans per example at τ=0.5). For a 1-token gold, only about 57% of the soft start mass sits on the gold token, so soft targets favour slightly longer spans. A5 tests whether that helps.
- FastText (optional) is deferred to Stage 4, where it is one of the sweep options for word init.

## Stage 2 — B1 lexical baseline (2026-10-07) — DONE

**Built**
- `src/baselines/b1_lexical.py` scores sliding windows by proximity to question content words, with an answer-type bonus (who/where → capitalised or @mention, when → number or month, how many → number) and a length penalty. Windows may not contain question words.
- `scripts/stage2_b1.py` picks the 3 parameters on **train** (no learning, so val stays clean). Log: `results/logs/stage2_b1.log`.

**Key numbers** (chosen config: max_len 5, length_penalty 0, type_bonus 1)
- B1 val: BLEU-1 17.12, METEOR 14.31, ROUGE-L 21.07, EM 10.32, F1 21.04. Train is similar: ROUGE-L 21.69.
- Without the type rules, val ROUGE-L drops to 17.06. Answer-type cues are worth about 4 points.
- The pipeline works end to end. Everything is logged in runs.csv.

## Stage 3 — Main model v1 (2026-10-07) — DONE

**Built**
- `src/model.py` (TweetBiDAF):
  - Embeddings: word emb 100 ⊕ char-CNN over raw tokens (32-d chars, 50 filters × widths 3/4/5) ⊕ 8 flags, then a 2-layer highway.
  - Shared BiLSTM encoder (H=128).
  - Trilinear attention with C2Q and Q2C, G=[h; c2q; h∘c2q; h∘q2c].
  - Gated 4-head self-attention with residual and LayerNorm.
  - 2-layer modelling BiLSTM and start head, then one more BiLSTM and the end head. Banded decoding with j < i+L.
  - All five ablation flags exist. `forward(return_attn=True)` returns S, C2Q, Q2C and self-attention.
- **Speed fix:** packed LSTMs on MPS ran at 0.33 s/batch. Replacing them with a padding-exact BiLSTM (reverse each sequence within its length via one gather, run two forward LSTMs, un-reverse) gives 0.084 s/batch. A unit test proves it matches PyTorch's packed BiLSTM to 1e-5.
- `src/train.py`: Adam with lr 1e-3, batch 32, dropout 0.2, grad-clip 5, early stopping on val ROUGE-L (patience 4), seeds set everywhere. Saves checkpoints to `checkpoints/`, per-epoch history to `results/history/`, and every run to runs.csv. The `--smoke` flag runs 200 examples × 2 epochs and asserts that loss decreases and the attention shapes are right.
- Also ready: `scripts/run_queue.py` (smoke-tested job queue), `src/fasttext_init.py` (in-domain FastText from the train split only), and `scripts/diagnose.py`.

**Key numbers**
- Smoke test passed: loss 6.57 → 6.14; S (4,34,10), self-attention (4,34,34).
- **main_v1, seed 1, val:** BLEU-1 33.71, METEOR 28.91, ROUGE-L 40.16, EM 22.12, F1 40.36. Best epoch 7, early stop at 11. About 15 s/epoch on MPS, 3.2 min total, 4.09M params.
- Diagnostics:
  - Mean predicted length 2.21 tokens (gold 2.33).
  - ROUGE-L is 48.4 on trainable val examples and 1.8 on the 191 abstractive ones, which no extractive model can get.
  - Gold span in the tweet body: 49.1. In the byline: 27.5.
  - By question type: who 48.7, where 46.7, what 34.8, how 34.3, why 26.7.

**Surprises**
- The model overfits after about epoch 7 (train loss keeps falling while val plateaus), so regularisation choices matter in Stage 4.
- B4 smoke (t5-small, 1 epoch, 200 examples) already reaches ROUGE-L 51. The original T5 checkpoints were pre-trained on a multitask mixture that includes SQuAD, so B4 is a very strong reference point. The report must say so.

## Stage 4 — Tune and freeze (2026-10-07) — DONE

**Built**
- One-factor sweep around the v1 config: lr {1e-3, 5e-4}, hidden {96, 128, 192}, dropout {0.2, 0.3}, L {10, 15, 20}, word init {random, FastText-in-domain}.
- The base config has 3 seeds; each variant was run on seed 1. `scripts/stage4_select.py` applies the adoption rule (variant val ROUGE-L > max of the base seed range). Log: `results/logs/stage4_select.log`.
- The two adopted variants were confirmed with 3 seeds each.
- **Why one factor at a time instead of the full 72-cell grid:** the full grid is about 5 h of MPS time. The adoption rule only needs each factor compared against the seed range.

**Key numbers (val ROUGE-L)**
- Base (lr 1e-3, H128, dropout 0.2, L15, random init), 3 seeds: 40.19 ± 0.52, range [39.68, 40.72].
- One-factor results, seed 1:
  - lr 5e-4: **41.22 (adopted)**
  - FastText init: **40.96 (adopted)**
  - within the base range: H96 40.32, H192 40.32, dropout 0.3 40.40, L10 40.49, L20 40.12
- 3-seed confirmation:
  - lr 5e-4 alone: 41.00 ± 0.45
  - **lr 5e-4 + FastText: 42.57 ± 0.84** (seeds 43.12 / 41.60 / 42.99), with F1 42.59, EM 24.38, BLEU-1 36.71, METEOR 31.77. Every seed beats the best base seed.
- **FROZEN** (`results/FROZEN_CONFIG.json`): lr 5e-4, FastText (gensim skip-gram 100-d, trained only on train-split tweets and questions), H128, dropout 0.2, L15, τ 0.5, soft spans, all components on. No architecture changes after this point.

**Surprises**
- Capacity (hidden size) and span length don't matter within noise. The gains come from optimisation (a lower lr) and from in-domain subword-aware word vectors.
- The FastText vectors are trained from scratch on TweetQA train only, so the main model is still "from scratch" with no external data. The report must make this explicit.

## Stage 5 — Ablations (2026-10-07) — CANCELLED by the team partway through

- The local MPS queue was stopped on request. From now on, training runs on Kaggle/Colab GPUs instead of the MacBook.
- Finished before the cancel (logged in runs.csv, with predictions saved): A1 s1–s3, A2 s1–s3, A3 s1–s3, A4 s1–s2. A4 s3, A5 and B2 had not finished, and partial runs were not logged.
- Next: package the ablation queue (and B3–B5) as a Kaggle/Colab GPU script.
- **GPU runner built:** `gpu_runner.ipynb`, generated by `scripts/make_gpu_notebook.py`. It writes the src files into the runtime, checks the data checksums, runs a smoke test, and runs a queue spread across the available GPUs:
  - frozen model ×3, A1–A5 ×3 and B2 ×3 (21 runs)
  - B3, B4 and B5 on val
  - optionally, the Stage 3/4 sweep
  
  Everything is packed into `a2_gpu_results.zip`. `runs.csv` notes now record the device. The embedded code passed a smoke test in isolation.

## GPU run 1 on Kaggle (2026-10-07/08) — finished, but outputs could not be retrieved

- All 21 main-model runs and B3–B5 completed in about 76 min on 2×T4. The summary printed in the Kaggle log matches the Mac results (full model val ROUGE-L 42.95 ± 0.47).
- Kaggle then hung for more than 17 h on "saving output", with Output still 0 B. The cause was about 600 MB in /kaggle/working (the zip plus the unzipped work folder with checkpoints). The run had to be cancelled, so none of its files can be retrieved, and **its numbers can't be used** (they aren't in our runs.csv).
- Fix in `scripts/make_gpu_notebook.py`: the Kaggle work folder moves to /tmp, so only the zip lands in /kaggle/working. The zip keeps every log, history and prediction file, but only the checkpoints we need (full model, A3, A5, B2): about 12 × 16 MB instead of 21+.
- Observation from the log, to be confirmed by the rerun: A5 (single span) beat the soft-span full model (val ROUGE-L 44.5 vs 43.0).
- **Correction (2026-10-08):** Kaggle did eventually save GPU run 1. The user downloaded it, so no rerun is needed.
  - Merged into A2 without overwriting anything:
    - 21 checkpoints → `checkpoints/`
    - histories, predictions and logs → `results/`
    - the raw file → `results/runs_gpu_kaggle.csv`
    - 24 rows appended to `results/runs.csv`, tagged `device=cuda:Tesla T4 platform=kaggle-gpu-run1`
  - Checks:
    - same validation split as local
    - re-scored predictions match the logged numbers exactly
    - no empty predictions
    - frozen config identical
  - **Stage 5/8 tables use the GPU rows only.** The Mac ablation rows stay in the log as partial or cancelled runs.

## Main model v2 and GPU run 2 prepared (2026-10-08)

- **Team decision:** the main model switches to single silver-span training (`soft_spans=False`). `results/FROZEN_CONFIG.json` is now v2 (the old config is kept as `FROZEN_CONFIG_v1.json`). Evidence: run-1 A5 scored val ROUGE-L 44.51 ± 0.11 vs 42.95 ± 0.47 for the soft-target model, on all 3 seeds. The v2 main model is run 1's `A5_single_span` (same config), so it is not retrained.
- **New `soft_tau` setting:** `tau` still decides which examples are trainable, and `soft_tau` sets the soft-set width separately, so the width can be varied while keeping the same 8,057 training examples. Soft spans per example (train): 17.1 at τ=0.5, 3.6 at 0.7, 1.3 at 0.85. Share of the soft start target on the best start token: 0.48 → 0.67 → 0.73.
- **`gpu_runner_run2.ipynb`** (`scripts/make_gpu_notebook.py --run 2`), 18 runs:
  - A1–A4 re-run on the v2 base, named `S_A*`
  - the soft-target width experiment at soft_tau 0.7 and 0.85
  - 3 seeds each; no baselines
  
  B2 is unchanged (its soft spans were already off). Dry-run passed locally.
- **Bug caught:** `--set` silently accepted mistyped overrides (found when zsh passed two settings as one argument). `_parse_set` now rejects unknown keys and mismatched types, with a new test (20 tests pass). All 51 job configs validate. Run 1 wasn't affected: the queue passes each setting separately, and the per-ablation parameter counts confirm it.

## GPU run 2 on Kaggle (2026-10-08) — DONE, merged

- 18 runs, all rc=0, ~45 min. Checks: every run's config matches the intended settings, same validation split, re-scored predictions match the logged numbers exactly, frozen v2 config identical. Merged without overwriting; runs.csv rows tagged `platform=kaggle-gpu-run2`, raw file kept as `results/runs_gpu_kaggle_run2.csv`. `results/RESULTS.md` regenerated for v2.
- **Ablations on main model v2 (val ROUGE-L 44.51 ± 0.11):** A1 −char-CNN −0.94, A2 −normaliser&flags −11.68, A3 −self-attention −1.99, A4 −Q2C −1.01, A5 wide soft targets −1.56, B2 all off −14.37. **Every ablation's seed range lies entirely below the main model's** (min main 44.39 > max of each ablation), so every component now shows a measurable contribution. On v1, A4 (Q2C) was within noise.
- **Soft-target width:** single 44.51 > medium (τ .7) 43.70 > wide (τ .5) 42.95 ≈ narrow (τ .85) 42.90 ± 1.13 (one weak seed, 41.60). ROUGE-L is not monotone in width, but **EM rises steadily as the target narrows**: 23.95 → 26.95 → 27.26 → 27.94. That fits the explanation that wide soft targets blur span boundaries. No soft width beats the single span, which confirms the v2 decision.
- Note: `results_summary.py` and `RESULTS.md` were missing from the working tree because the team committed them on `main` and then branched `optimisation` from the earlier commit. Restored from `main` (unstaged); branches untouched.
