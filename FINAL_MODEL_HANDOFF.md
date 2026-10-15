# Final-model inference handoff — TweetQA

Prepared: 15 October 2026 (Australia/Perth).

This document is written for the next team member and any AI assistant implementing the final inference notebook. It describes the **current best validation configuration**, its exact artifacts, and how to reproduce it before predicting the locked test data.

**This handoff does not authorize or perform test prediction/evaluation.** The current user explicitly requested stopping before test use. Obtain a new instruction from the responsible team member before unlocking test. This is also consistent with the one-time final-test checkpoint in `A2_PLAN.md`. Preparation and validation reproduction can proceed without unlocking it.

## 1. The selected system

Use this exact pipeline:

```text
raw tweet + question
        |
        +--> saved TweetBiDAF v2 seed 1 --> start/end probabilities
        +--> saved TweetBiDAF v2 seed 2 --> start/end probabilities
        +--> saved TweetBiDAF v2 seed 3 --> start/end probabilities
                         |
              equal-weight probability average
                         |
              Top-10 legal answer spans
                         |
          original 30-feature OOF LambdaMART
                         |
              one span per input question
                         |
           substring of the original raw tweet
```

The final predictor is extractive. It does not generate new text. The three QA models and the reranker are already fitted; final inference requires **no training**.

The original OOF reranker was trained using candidates from three tweet-grouped fold models, each predicting a held-out fold. The **three full-data-split QA seeds used at inference are different from those OOF fold models**. Never substitute a fold model for a final QA checkpoint.

The reranker's OOF source was `CurrentGated-s1-20261014-175436`; the final ensemble uses the three `A5_single_span` checkpoints listed below. This combination was deliberately evaluated and is the selected validation configuration. Do not replace its ensemble with CurrentGated just because the OOF summary names that source.

## 2. Repository and version

- Repository: <https://github.com/James-h95/A2>
- Branch: `kaz/v2-submission-fix`
- Code and experiment notebooks through Step 15: `0939b840086bf4adacb88a41851256b31f807f74`.
- Code plus all 13 locally available artifact ZIPs: `84b85d90a997cc98632dc06a822cd42a5c19ec17`.
- This Markdown was created after `84b85d9`; the handoff file itself is not part of that earlier commit.

For a pinned reproduction checkout:

```bash
git clone --branch kaz/v2-submission-fix https://github.com/James-h95/A2.git
cd A2
git checkout 84b85d90a997cc98632dc06a822cd42a5c19ec17
```

Use the `src/` modules in this repository. Do not work in the separate local `A2_review` folder. Do not assume the legacy `build_notebook.py` already implements the ensemble-plus-reranker pipeline: final inference still needs to be wired explicitly.

## 3. Exact artifacts to load

Paths in this document are relative to the repository root, unless shown as paths inside ZIPs.

### 3.1 Full QA ensemble

Archive: `a2_v2_colab_artifacts.zip`.

Load exactly these three members, in seed order:

| Seed | ZIP member |
|---|---|
| 1 | `checkpoints/A5_single_span-s1-20261014-071243.pt` |
| 2 | `checkpoints/A5_single_span-s2-20261014-071828.pt` |
| 3 | `checkpoints/A5_single_span-s3-20261014-072428.pt` |

Despite their historical `A5` names, these are the selected single-span v2 full models. **Do not use** the `S_A3_no_selfattn` checkpoints in the same archive.

Use `src.train.load_checkpoint(path, device)` to load each checkpoint. It returns `(model, cfg, word_vocab, char_vocab)` and puts the model in evaluation mode. Each model must use **its own saved word and character vocabulary**. Do not reconstruct a shared vocabulary and do not retrain FastText.

### 3.2 Selected reranker

Archive: `oof_span_reranker_artifacts.zip`.

Load exactly:

```text
oof_reranker_results/oof_lambdamart_reranker.joblib
```

Use `joblib.load`. Check `n_features_in_ == len(src.rerank.FEATURE_NAMES) == 30`.

`reranker_next_artifacts.zip/feature_study_original.joblib` is a later refit of the same original setup and was validated on the ensemble. It is **not byte-identical** to the original saved model. Prefer the original artifact above and verify its exact predictions against the saved validation reference; do not swap models based solely on similar filenames.

### 3.3 Artifacts NOT selected

Do not use any of these as the final reranker:

- `oof_linear_reranker.joblib` — the linear alternative.
- `feature_study_enhanced.joblib` — the eight extra boundary/confidence features.
- `RepresentationOOF.joblib` — the six extra BiLSTM representation features.
- `ensemble_oof_lambdamart.joblib` in `ensemble_oof_artifacts.zip` — Step 15 matched-ensemble OOF experiment, which did not beat the selected original reranker.

No Step 10 fold checkpoints, Step 13 representation caches, or Step 15 fold ensembles are required for final inference.

### 3.4 SHA-256 identity checks

Verify the files before loading. A mismatch means investigate provenance; do not silently choose a different model.

| File/archive | SHA-256 |
|---|---|
| `a2_v2_colab_artifacts.zip` | `6fc46dc5ef256426d8c4b95da2898483edba34687a62013bc2889cd59c9badd4` |
| `oof_span_reranker_artifacts.zip` | `83436c4eff8604c8ebed3f898fab5874e5e67972097743b13389e4fcd05e4c00` |
| `reranker_next_artifacts.zip` | `dad07b029ec8b8d657438cc379f6ac6dd9c172ce41a9370f830e8f36facde5db` |
| `ensemble_feature_artifacts.zip` | `a257c825adab25438dd98899ca3178c83eb141e5c6a22d5dd3297312f86fccdd` |
| Extracted seed-1 `.pt` | `1761eba471347e22b2f5313a5d4e9f021d1c571baae2a1ff6909ead65ff80e36` |
| Extracted seed-2 `.pt` | `7a75861188cf954daa8a8ca587a59c77636f71e316ea72092f84ff23e5ed63ec` |
| Extracted seed-3 `.pt` | `c81dde15edf4679d2d3dda90ef2d9108ab837d449e855435336e163f07b128d9` |
| Extracted original OOF LambdaMART `.joblib` | `93e3689c180815e972d5efd52fb5c7e0a626aa6261f0739d406c5fcea8ee9d32` |

Use `hashlib.sha256(file_bytes).hexdigest()`; for large files, update the digest in chunks. `ZipFile.testzip()` should also return `None`.

## 4. Validation results and the selection decision

All these scores are validation percentages on the fixed **1,076 questions from 511 tweets**, not test scores.

| System | BLEU-1 | METEOR | ROUGE-L | EM | F1 |
|---|---:|---:|---:|---:|---:|
| Three-seed ensemble, Top-1 only | 40.13489 | 34.07875 | 45.71929 | 28.53160 | 45.88715 |
| **Ensemble + original OOF LambdaMART (selected)** | **41.61186** | **34.85726** | **47.11184** | 30.11152 | **47.31877** |
| Ensemble + extra boundary/confidence features | 41.47597 | 34.92650 | 46.69676 | **30.66914** | 46.88825 |
| Ensemble + matched ensemble-OOF training (Step 15) | 41.29808 | 34.43474 | 46.50311 | 30.11152 | 46.74965 |

The six BiLSTM representation features were tested on a **single-model** candidate set: ROUGE-L rose from 45.56 to 45.87, with a paired 95% interval spanning zero. This is not an ensemble result and is not directly comparable to 47.11.

The selected reranker improved ensemble ROUGE-L by +1.39255, with a tweet-cluster bootstrap 95% interval of [+0.26843, +2.52038]. Step 15 changed ROUGE-L by −0.60874 versus the selected system, with interval [−1.46053, +0.22014]. The latter does not establish a clear difference, but it provides no reason to replace the selected model.

These intervals do not correct for repeated validation-based model selection. Do not claim the test score will equal 47.11.

Step 14's Oracle scores (68.43 Top10, 73.22 Top30, 71.07 seed-candidate union) use gold answers to select a candidate. They are diagnostic ceilings, **not actual model performance**. Top30 and candidate union have not been adopted. Five-fold OOF has not been run.

## 5. Mandatory validation reproduction BEFORE test access

1. Work from a clean runtime and load the exact three QA checkpoints and original OOF model.
2. Use `D.get_splits()` with unchanged defaults (`seed=4012`, `val_frac=0.10`). This reads `data/train.csv`, removes exact duplicate training rows, and groups by tweet. Do not invent a new split.
3. Use the returned validation rows without sorting, filtering, deduplicating, or changing their `row_id` values.
4. Predict using the algorithm below and compare by `row_id`, not list position alone.
5. Reference predictions are in either:
   - `reranker_next_artifacts.zip/Ensemble_fixed_oof_exploratory_predictions.json`;
   - `ensemble_feature_artifacts.zip/Ensemble_original_predictions.json`;
   - `ensemble_oof_artifacts.zip/OriginalOOF_predictions.json`.
6. Check identical row-ID sets and **exact equality of all 1,076 answer strings**. These references agree in the validated experiments.
7. Evaluate the unchanged gold answers with `src.evaluate.evaluate`; expect the selected row in section 4. When comparing to rounded numbers above use a tolerance of `1e-4`; preferably read the full-precision values from the reference ZIP's `summary.json`.
8. If reproduction fails, stop and diagnose before test access. Do not tune thresholds to make the score match. Check checkpoint identities, vocabulary ownership, probabilities versus logits, token alignment, feature order, dependency versions and Top-K tie ordering.

For a line-ending-independent input identity check, the original `train.csv` SHA-1 is:

```python
hashlib.sha1(Path("data/train.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
# 8a52bb7dff56b11fdad1ab6fd3b848fbadc22849
```

## 6. Exact prediction contract

### 6.1 Input identity and gold isolation

Each input row must contain `row_id`, `qid`, `question`, and `tweet`. `row_id` is the original row index in that input file. Preserve it even if rows are identical. Do not assume `qid` is unique or convert it to an integer.

The repository's `D.preprocess` also requires `answers` and computes silver spans. Silver spans are for training/analysis, **not predictions**. To ensure inference never consults test answers, keep real golds in a separate evaluation structure and pass rows with `answers=[""]` to preprocessing. Optionally blank `answer_field` as well. This gives valid batch metadata while leaving tokenization, flags, vocabulary encoding and prediction features unchanged.

Do not filter on `trainable`, `best`, `best_f1`, or answer substring availability. Always use `D.make_loader(..., train=False)`. Even questions with no extractive gold answer must receive a prediction.

Reject empty tweet/question token sequences explicitly instead of silently dropping those rows. No fallback policy has been validated for empty inputs; the team must decide one if such inputs occur.

### 6.2 Preprocessing and model execution

For each seed separately:

```python
prep = dict(
    use_normaliser=cfg["use_normaliser"],
    max_span_len=cfg["max_span_len"],
    tau=cfg["tau"],
    soft_tau=cfg.get("soft_tau"),
)
examples = D.preprocess(label_free_rows, prep)
D.encode(examples, own_word_vocab, own_char_vocab)
```

Use the existing tokenizer/normalizer unchanged. In particular, preserve raw character offsets, hashtag splitting, mention/URL normalization, character inputs and body/byline flags. No extra text cleaning, truncation or stopword removal.

Run each model with `model.eval()` and `torch.no_grad()` (or `inference_mode`). Batch size 64 matches the experiments. GPU is recommended; CPU is possible but should still pass validation reproduction. Model training/FastText initialization is unnecessary.

Verify that all seeds produce the same row order and identical raw token spans. Their integer vocabulary IDs may differ and should not be compared as if shared.

### 6.3 Probability average and legal spans

For an example with `n` real tweet tokens:

1. Trim each model's start/end logits to `[:n]` before softmax, excluding batch padding.
2. Compute each model's softmax start probabilities and end probabilities.
3. Move probabilities to CPU as float32 and average the three starts and the three ends separately, each with weight `1/3`.
4. Clamp each average to at least `1e-30`, then take its logarithm.
5. Span score: `log(mean_start[i]) + log(mean_end[j])`.
6. Legal spans satisfy `0 <= i <= j < n` and `j - i < 15` (inclusive span length at most 15 tokens).
7. Mask all other scores with negative infinity. Select `min(10, number_of_legal_spans)` using PyTorch `topk` on the flattened score matrix, as in the experiment.
8. Decode flattened indices with `i, j = divmod(index, n)`. Both token endpoints are inclusive.

Do not average logits, log-probabilities, span ranks, predicted strings, or the reranker's output scores. Do not use a majority vote. Do not replace probability averaging with the Step 14 candidate union.

For every candidate store exactly:

```text
start, end, text, start_logp, end_logp, base_score, region
```

`text = D.span_text(example, start, end)` and `region = R.span_region(example, start, end)`. Store candidates in descending ensemble base-score order; this order determines rank-based features. Keep different span positions even when their text strings are identical. No answer-string deduplication was used in the selected configuration.

### 6.4 Reranking

Use the existing `R.rerank(reranker, examples, candidate_records)` unchanged. It computes the 30 features in `R.FEATURE_NAMES` order, calls LambdaMART's `predict`, and uses `np.argmax` to select one candidate. Do not add feature scaling or extra features. Do not use the linear model's `decision_function` for LambdaMART.

Output answer text is copied from the original tweet offsets. Do not normalize or trim it for export; normalization happens inside the evaluator. Save the selected candidate rank and span offsets for auditability.

## 7. Implementation reference (prediction function only)

This code is a reference for the next implementer; it has not been executed against test. Run validation reproduction first. It assumes the trusted checkpoints have already been extracted and `D`, `T`, `R`, `torch`, `numpy as np` are imported from the pinned repository/runtime. `checkpoint_paths` must be the exact seed-ordered paths in section 3 and `reranker` the selected original model.

```python
def predict_selected_system(raw_rows, checkpoint_paths, reranker, device="cuda"):
    import json
    assert len(checkpoint_paths) == 3
    assert len({r["row_id"] for r in raw_rows}) == len(raw_rows)
    assert reranker.n_features_in_ == len(R.FEATURE_NAMES) == 30
    rows = [dict(r, answers=[""], answer_field="") for r in raw_rows]
    reference_examples = None
    all_probabilities = []
    reference_cfg = None

    def effective(cfg):
        return json.loads(json.dumps(
            {**T.DEFAULT_TRAIN, **T._default_model(), **cfg}, sort_keys=True
        ))

    for path in checkpoint_paths:
        model, cfg, wv, cv = T.load_checkpoint(path, device)
        if reference_cfg is None:
            reference_cfg = effective(cfg)
        assert effective(cfg) == reference_cfg
        assert cfg["max_span_len"] == 15
        assert not cfg.get("use_joint_span_scorer", False)
        examples = D.preprocess(rows, dict(
            use_normaliser=cfg["use_normaliser"],
            max_span_len=cfg["max_span_len"],
            tau=cfg["tau"], soft_tau=cfg.get("soft_tau"),
        ))
        assert all(e["t_tokens"] and e["q_tokens"] for e in examples)
        D.encode(examples, wv, cv)
        if reference_examples is None:
            reference_examples = examples
        else:
            assert [e["row_id"] for e in examples] == [
                e["row_id"] for e in reference_examples
            ]
            assert [e["t_tokens"] for e in examples] == [
                e["t_tokens"] for e in reference_examples
            ]
            assert [e["q_tokens"] for e in examples] == [
                e["q_tokens"] for e in reference_examples
            ]
        values = []
        model.eval()
        with torch.no_grad():
            for batch in D.make_loader(examples, batch_size=64, train=False):
                out = model(T.to_device(batch, device))
                for b, ex in enumerate(batch["meta"]):
                    n = len(ex["t_tokens"])
                    values.append((
                        ex["row_id"],
                        torch.softmax(out["start_logits"][b, :n], -1).cpu(),
                        torch.softmax(out["end_logits"][b, :n], -1).cpu(),
                    ))
        assert [v[0] for v in values] == [r["row_id"] for r in rows]
        all_probabilities.append(values)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = []
    for k, ex in enumerate(reference_examples):
        items = [values[k] for values in all_probabilities]
        assert all(item[0] == ex["row_id"] for item in items)
        start = torch.stack([x[1] for x in items]).mean(0).clamp_min(1e-30).log()
        end = torch.stack([x[2] for x in items]).mean(0).clamp_min(1e-30).log()
        n = len(start)
        positions = torch.arange(n)
        width = positions[None, :] - positions[:, None]
        legal = (width >= 0) & (width < 15)
        scores = (start[:, None] + end[None, :]).masked_fill(~legal, -float("inf"))
        best_scores, indices = scores.flatten().topk(min(10, int(legal.sum())))
        candidates = []
        for score, flat in zip(best_scores.tolist(), indices.tolist()):
            i, j = divmod(flat, n)
            candidates.append(dict(
                start=i, end=j, text=D.span_text(ex, i, j),
                start_logp=float(start[i]), end_logp=float(end[j]),
                base_score=score, region=R.span_region(ex, i, j),
            ))
        records.append(dict(row_id=ex["row_id"], qid=ex["qid"], candidates=candidates))

    predictions, selections = R.rerank(reranker, reference_examples, records)
    assert len(predictions) == len(raw_rows)
    assert all(np.isfinite(s["rerank_score"]) for s in selections)
    outputs = []
    for ex, pred, selection in zip(reference_examples, predictions, selections):
        i, j = selection["start"], selection["end"]
        char_start = ex["t_tokens"][i]["start"]
        char_end = ex["t_tokens"][j]["end"]
        assert pred == ex["tweet"][char_start:char_end]
        outputs.append(dict(
            row_id=ex["row_id"], qid=ex["qid"], prediction=pred,
            selected_rank=selection["selected_rank"],
            start_token=i, end_token=j,
            start_char=char_start, end_char_exclusive=char_end,
            base_score=selection["base_score"],
            rerank_score=selection["rerank_score"],
        ))
    return outputs, records
```

Checkpoint compatibility detail: older v2 `.pt` files omit later default options (e.g. `self_attn_mode`, joint-span options). Compare **effective** configs after merging `T.DEFAULT_TRAIN` and `T._default_model()`, with tuple/list normalization. Do not repeat Step 15's original raw-dict equality bug. Do not disable genuine configuration mismatch checks.

## 8. Runtime preparation

The experimental Colab notebooks install `gensim`, `emoji`, `nltk`, `scikit-learn`, `joblib`, `lightgbm`, and use Colab's PyTorch/NumPy. No exact package lockfile was saved for these runs, so successful validation reproduction is the compatibility check. Record installed versions in the final run manifest.

Load only the trusted team-created `.pt` and `.joblib` artifacts. PyTorch loading here uses `weights_only=False` because the checkpoint also contains configuration and vocabularies. Use the repository loader rather than changing that format. Joblib files also deserialize model objects.

`src.evaluate` downloads NLTK WordNet/OMW resources when METEOR is first requested. Allow that setup before final evaluation. Warnings about `X does not have valid feature names` can occur for the NumPy input to LightGBM; they occurred in successful validated runs. Do not suppress an actual feature-count/order error as though it were that warning.

If reusing a live Colab runtime, avoid importing stale `src` modules from a different work directory. Prefer a clean runtime. Load paths and set `TWEETQA_DIR`/results variables before importing the modules if overriding them.

## 9. Test handling AFTER a new team instruction

The current repository sets `D.TEST_LOCKED=True`. `D.load_test()` raises while locked. Keep this default in source control. When the responsible team member explicitly approves the final run, set `D.TEST_LOCKED=False` **in that approved runtime**, then load the original `test.csv`. Do not run a sweep or choose a model from test scores.

The existing loader expects CSV columns `qid`, `Question`, `Tweet`, and `Answer`; it parses `Answer` with `D.parse_answers`. The project documentation identifies this file as the provided TweetQA development set reserved as our test. Do not substitute an upstream dataset split. Its current contents were not read during this handoff.

If the delivered test file has no `Answer` column, the existing loader is unsuitable: implement a label-free reader for the actual schema, preserving raw `qid` and row order. Give the prediction function placeholder answers, and **skip scoring without real golds**. Do not infer or fabricate gold answers. Confirm any required submission schema with the team/assignment spec; no final submission CSV format is established by this document.

For a labelled test:

1. Retain original gold lists keyed by `row_id` solely for evaluation.
2. Run the selected, frozen pipeline with label-free preprocessing as above.
3. Save predictions before consulting scores.
4. Evaluate once with the same `src.evaluate.evaluate` used for validation.
5. Do not call `oracle_predictions`, `fit_lambdamart_reranker`, `fit_pairwise_reranker`, `T.train`, vocabulary building, or FastText fitting on test.

No retraining on train+validation is part of the selected configuration. Such retraining would create a new, unvalidated model/reranker combination and must be treated as a separate experiment, not an implementation detail.

## 10. Required output and provenance

Recommended internal files (adapt the external submission format only after checking the spec):

- `final_predictions.json`: one record per original row, in original order, with the fields returned by the reference function.
- `final_predictions.csv`: at least `row_id`, `qid`, `prediction`; use a real CSV writer to quote tweet substrings correctly.
- `top10_candidates.jsonl`: per-row candidate positions, text, scores and region for auditing.
- `run_manifest.json`: Git revision, SHA-256 identities of weights/reranker/input CSV, seed checkpoint order, feature names/order, Top-K=10, max span length=15, equal averaging weights, package versions, device, counts, timestamp with timezone and whether test was approved/unlocked.
- `test_metrics.json` only if real gold labels were available and the final run was authorized.
- Append reported evaluation numbers to `results/runs.csv` using `src.runlog.log_run`; represent the ensemble as one named system, not as another independent training seed. Include checkpoint/reranker identities in config/notes.

Validate output count, unique `row_id`, identical input/output row-ID sets, preserved `qid`, legal spans, finite scores and exact raw substring extraction. Repeated `qid` values or repeated input rows must not be dropped. Store outputs in a new run directory instead of overwriting existing validation artifacts.

## 11. Known limits and what is not uploaded

- The compact result ZIPs save metrics, predictions and fitted rerankers; they do **not** include all fold checkpoints or probability caches.
- Step 10 fold checkpoints remain in Drive; newly trained Step 15 fold checkpoints/caches are in `MyDrive/CITS4012_A2_ensemble_oof`. They are unnecessary for selected-system inference.
- At preparation time, `representation_reranker_artifacts.zip` was not present in the local repository, although Step 13's executed Notebook contains its results. It is not required for the selected system.
- Checkpoints for the selected three full models **are** inside the uploaded `a2_v2_colab_artifacts.zip`.
- The final clean-runtime submission notebook has not been verified as implementing this selected pipeline. Do not equate an experimental Notebook completing successfully with a submission artifact being ready.

## 12. Suggested prompt for the next AI assistant

> Read FINAL_MODEL_HANDOFF.md and the referenced src modules. Implement a clean-runtime inference notebook for the pinned selected system: the three exact A5_single_span v2 checkpoints, equal-weight start/end probability averaging, legal Top-10 spans, and the original 30-feature OOF LambdaMART. First reproduce all saved validation answer strings and metrics. Do not train models, refit the reranker, change features, use candidate union, increase folds, or access test without a new explicit instruction. Keep gold answers separate from inference preprocessing. If required artifacts, data schema, configuration compatibility or validation reproduction differ, report the concrete mismatch before proceeding. Preserve provenance and export per-row predictions with raw tweet offsets.
