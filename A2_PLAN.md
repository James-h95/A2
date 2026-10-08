# CITS4012 A2 — TweetQA — Execution plan for Claude Code

Spec: `../CITS4012_A2_2026.pdf` (wins on any conflict). Team plan doc (Claude Docs): "CITS4012 Group Project Plan — TweetQA".
Deadline: 18 Oct 2026 23:59 AWST. Internal submit target: 17 Oct. Design freeze: 12 Oct.

Work through the stages in order **without stopping**, except at two checkpoints: (1) end of Stage 0, if the test split has no gold answers or the data differs from this plan; (2) before Stage 8, the one-time test-set evaluation. At the end of every stage, append a short summary (what was built, key numbers, anything surprising) to `results/PROGRESS.md`, then continue.

---

## Hard rules (never break these)

1. Main model is designed and trained **from scratch** (RNN/LSTM/GRU/Transformer + attention). No pretrained weights or embeddings in the main model unless the team confirms GloVe-Twitter is allowed (default: NOT allowed).
2. **Never copy code** from GitHub, paper repos or tutorials. Write every module here. PyTorch built-in layers (`nn.LSTM`, `nn.MultiheadAttention`, `nn.Conv1d`) are fine. HF `transformers` is fine for baselines only.
3. **No external datasets.** Only the provided TweetQA files.
4. **Test set is used once**, at the very end, for the final table. All selection/tuning on validation. Keep a `TEST_LOCKED = True` flag; evaluation on test raises unless it is set False.
5. Every number in the report must come from a logged run. All runs append to `results/runs.csv` (run_id, model, config JSON, seed, split, BLEU-1, METEOR, ROUGE-L, EM, F1, train time).
6. Final deliverable = ONE notebook `CITS4012_<GroupID>.ipynb` built on the provided Colab template, runnable from a clean Colab runtime, with logs saved as outputs.
7. ~~Train locally on this Apple Silicon Mac.~~ **Changed 2026-10-08 by the team:** full training runs on a Kaggle/Colab GPU via `gpu_runner.ipynb`, which the user runs. Device selection stays `mps → cuda → cpu`. Run a ≤200-example smoke test before every full run (locally, seconds). Stages 3–4 ran on the Mac (MPS) before this change. Stage 5 and the B3–B5 baselines ran on Kaggle 2×T4.

## Layout

```
A2/
  A2_PLAN.md            (this file)
  data/                 TweetQA files from the unit's shared Drive folder (team downloads manually)
  template.ipynb        the A2 Colab template (team downloads manually)
  src/
    data.py             loading, splits, tokenisation with char offsets, normaliser, silver spans, vocab, DataLoader
    evaluate.py         BLEU-1, METEOR, ROUGE-L, EM, token-F1 (max over gold answers)
    model.py            TweetBiDAF with ablation flags; forward(..., return_attn=False)
    train.py            config dict -> train / early stop / checkpoint / log to runs.csv
    baselines/          b1_lexical.py, b3_extractive_ft.py, b4_t5_ft.py, b5_llm.py
    analysis.py         attention heatmaps, slice tables, error-taxonomy sampling
  tests/                pytest for evaluate + span labelling + offsets
  results/              runs.csv, figures/, tables/ (LaTeX)
  build_notebook.py     pastes src/ modules + run cells into the template -> final .ipynb
```

Develop as `.py` modules (easier to test and diff); `build_notebook.py` assembles the single required notebook at the end. The final notebook must not import from `src/` — code is inlined.

---

## Stage 0 — Preconditions and data report  (Oct 7–8)

- Check `data/` and `template.ipynb` exist. If missing, STOP and tell the user to download:
  - Datasets folder: https://drive.google.com/drive/folders/1j5lP1gBrewzpFSH5Ya1SQ6echsiJaI3j?usp=drive_link (TweetQA only)
  - Colab template: https://colab.research.google.com/drive/1LF8KINrQHKXZBOhDyVe7y3EA5dm5_iYX (File > Download .ipynb, save as `A2/template.ipynb`)
- Inspect the files: format, field names, splits provided, number of examples per split.
- **Key check: does the test split contain gold answers?** If not, propose a split (e.g. carve a held-out test from train, stratified by question word) and STOP for team approval.
- Write `results/DATA_REPORT.md`: split sizes; tweet / question / answer length (tokens) percentiles; number of gold answers per example; % answers that are an exact substring of the tweet (case-insensitive); % after normalisation; top 15 question first-words; share of answers that are @mentions, hashtags, numbers/dates; 10 random examples printed raw.
- Read the template notebook and list its required sections.

## Stage 1 — Evaluation, preprocessing, silver spans  (Oct 7–9)

- `evaluate.py`: normalise (lowercase, strip punctuation/articles, collapse whitespace); BLEU-1 (nltk `sentence_bleu`, weights (1,0,0,0), smoothing), METEOR (nltk), ROUGE-L F (own LCS implementation), EM, token-F1; each = max over gold answers; corpus score = mean ×100. Tests: gold vs gold = 100; empty prediction = 0.
- Tokeniser that keeps character offsets into the raw tweet.
- Tweet normaliser (`@user`→`<user>`, URLs→`<url>`, elongation squash to max 2 repeats, `#tag`→`#` + `tag`, emoji→`<emoji>` or kept as token — decide from DATA_REPORT), returning a mapping back to raw offsets. Must be switchable off (ablation A2).
- Silver span labeller: for each example enumerate spans up to length L (default 15 tokens); score by token-F1 (tie-break ROUGE-L) against each gold answer; store best span AND the soft-target set (all spans with F1 ≥ τ, default τ=0.5, weights ∝ F1). Ablation A5 toggles soft vs single.
- Report the **oracle score** on train and validation (best span vs gold, all metrics). This is the ceiling for every extractive model.
- Vocab from training split only (min_freq 2), char vocab, token-type flags (hashtag, mention, url, emoji, number), exact-match-with-question flag.
- Optional: train FastText (gensim) on training tweets + questions only, dim 100–200, as an alternative word-vector init.

## Stage 2 — B1 lexical baseline  (Oct 8–9)

- Slide windows of length 1..L over the tweet; score by overlap with question non-stopwords plus proximity; return the best window, excluding words already in the question. Evaluate on validation. Log to runs.csv. This proves the pipeline end to end.

## Stage 3 — Main model v1  (Oct 8–11)

`model.py` — TweetBiDAF. Config flags for every ablation:

| flag | default | ablation |
|---|---|---|
| `use_char_cnn` | True | A1 |
| `use_normaliser`, `use_token_flags` | True | A2 |
| `use_self_attn` | True | A3 |
| `biattn_mode` | `"both"` (`"c2q"` for A4) | A4 |
| `soft_spans` | True | A5 |

Layers: (1) embedding = word emb ⊕ char-CNN (filters 100, widths 3–5, max-pool) ⊕ flags → 2-layer highway; (2) shared BiLSTM encoder (hidden 128); (3) trilinear similarity S_ij = wᵀ[h_i; u_j; h_i∘u_j], C2Q = softmax_j(S)·U, Q2C = softmax_i(max_j S)·H, G = [h; c2q; h∘c2q; h∘q2c]; (4) gated self-attention over G (or 1 Transformer encoder block, 4 heads) with residual + LayerNorm; (5) 2-layer modelling BiLSTM, start/end linear heads with padding masks; decode best (i, j) with i ≤ j ≤ i+L.
Loss: cross-entropy on start/end; with `soft_spans`, KL/soft-CE against the soft start/end distributions.
`forward(..., return_attn=True)` returns the S matrix, C2Q weights, Q2C weights, self-attention weights.

`train.py`: Adam lr 1e-3, batch 32, dropout 0.2, grad-clip 5, up to 30 epochs, early stopping on val ROUGE-L (patience 4), seed everything (python/numpy/torch/cuda, deterministic flags). Save best checkpoint + config. Log every epoch's train loss and val metrics (these logs must appear in the final notebook).

Deliver: smoke test (200 examples, 2 epochs, loss decreases, attention shapes correct), then the full run (done on MPS before the GPU-only rule; later runs on Kaggle). Report val metrics and time per epoch.

## Stage 4 — Tune and freeze  (Oct 11–12)

Small grid on validation only: lr {1e-3, 5e-4}, hidden {96, 128, 192}, dropout {0.2, 0.3}, L {10, 15, 20}, word init {random, FastText-in-domain}. One seed for the sweep, then 3 seeds for the chosen config. Adoption rule: a change counts only if it beats the seed range. Write `results/FROZEN_CONFIG.json`. **No architecture changes after this.**

## Stage 5 — Ablations  (Oct 12–14)

A1–A5 from the frozen config, 3 seeds each, all else identical. Output `results/tables/ablations.tex` (mean ± sd for all metrics) plus slice breakdowns tied to each hypothesis:
- A1: examples whose gold answer contains a token OOV in the training vocab
- A2: answers that are @mentions / hashtags / usernames
- A3: answers located > N tokens from the nearest question-word match, or in the username/date line
- A4: overall only
- A5: examples whose gold answer is not an exact tweet substring

## Stage 6 — Baselines B2–B5  (Oct 9–13, in parallel with Stage 3–5)

- B2: main model with all flags off (vanilla BiDAF-lite) — just a config.
- B3: fine-tune `vinai/bertweet-base` (fallback `roberta-base`) with a span head on the same silver spans; 3 epochs, lr 3e-5, max_len 128. Map predicted subword span back to raw text.
- B4: fine-tune `t5-small` (or `google/flan-t5-base` if time) on "question: … context: …" → answer; beam 4, max 16 tokens.
- B5 (optional): zero-/few-shot open instruct model (1–3B, e.g. Qwen2.5-1.5B/3B-Instruct) with a fixed prompt; record the prompt verbatim; greedy decoding; strip to a short answer.
- Same `evaluate()` for all; validation for development, test only at the end. Record checkpoint names and hyperparameters in runs.csv.

## Stage 7 — Analysis  (Oct 13–15)

- `analysis.py`: heatmap of the tweet×question attention (and C2Q/Q2C), with start/end probability strips; for 6 chosen validation examples: 3 successes, 3 failures by type (username answer, abstractive answer, slang/misspelt key word). Print question, tweet, gold, prediction on each figure.
- Self-attention map for one example, with vs without A3.
- Export 100 random validation errors to `results/error_sample.csv` with columns for manual tagging (question type, error cause). Summarise counts once tagged.
- Save all figures as PDF + PNG in `results/figures/`.

## Stage 8 — Final run, notebook, tables  (Oct 15–17)

- Set `TEST_LOCKED = False` once; evaluate the frozen main model (3 seeds), B1–B5 on test. Generate `results/tables/main.tex`.
- `build_notebook.py`: fill the template — setup/installs, data download instructions, all src code inlined in order, training cells (with a `TRAIN = False` switch that loads checkpoints from a public Hugging Face repo instead), evaluation, ablations, baselines, analysis. Keep saved outputs/logs.
- Clean-runtime Colab check (team does this): run top to bottom with `TRAIN = False`; confirm every reported number reproduces.
- Cross-check: every number in the LaTeX report appears in runs.csv / notebook output.
