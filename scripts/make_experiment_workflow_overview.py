"""Create an English, notebook-based overview of the complete experiment flow."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_workflow_overview.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


cells = [md("""# TweetQA Project — Experiment Workflow and Decisions

This notebook is the English navigation document for the project. It explains what each experiment tested, which artifacts it consumes and produces, and why each result led to the next step. All reported model-selection results use the fixed validation split derived from `train.csv`. The locked `test.csv` has not been used in these experiments.
"""),
md("""## Step 1 — Establish the task and leakage-safe split

TweetQA contains answers that are sometimes abstractive, but the principal model is an extractive span predictor. Exact duplicate rows are removed, and tweets—not individual questions—are assigned to either training or validation so that questions about the same tweet cannot cross the split.

The project uses character features, normalised word lookup, token flags, a shared BiLSTM encoder, BiDAF context-to-question and question-to-context attention, gated self-attention, and independent start/end span heads.

Primary implementation: `src/data.py`, `src/model.py`, and `src/train.py`.
"""),
md("""## Step 2 — Select the stable single-span baseline

The original target study compared wide, medium, and narrow soft sets against one deterministic silver span. The single-span target won across three seeds.

| Target | Validation ROUGE-L | EM |
|---|---:|---:|
| Wide soft | 42.95 | 23.95 |
| Medium soft | 43.70 | 26.95 |
| Narrow soft | 42.90 | 27.26 |
| **Single span** | **44.51** | **27.94** |

Notebook: `colab_v2_retrain.ipynb`.
"""),
md("""## Step 3 — Verify that attention is useful

Removing self-attention reduced validation ROUGE-L by roughly 1.5–1.8 points. This established that attention contributes useful information. Later one-seed screening found that ungated residual attention was approximately tied with the current gated block, while question-conditioned gated attention improved ROUGE-L by only 0.22. The current gated attention was therefore retained.

Run notebooks: `colab_attention_analysis.ipynb`, `colab_attention_variants.ipynb`.

Report notebooks: `attention_results_report.ipynb`, `attention_variants_report.ipynb`.
"""),
md("""## Step 4 — Diagnose boundary and attribution errors

Error analysis found three important groups: partial-overlap or boundary errors, wrong spans with no overlap, and answers that are not well represented by any tweet span. Attribution/byline confusion existed but was smaller than the general boundary problem.

A start-conditioned end head did not improve the result. This rejected the simplest boundary-aware output modification.

Run notebook: `colab_boundary_attention_experiment.ipynb`.

Report notebook: `boundary_attention_results_report.ipynb`.
"""),
md("""## Step 5 — Test generation for non-extractive answers

A pointer-generator model was evaluated because some TweetQA answers cannot be copied exactly from the tweet. It reached only 35.57 ROUGE-L overall, generated frequent unknown tokens, and occasionally hallucinated entities. The small gain on examples with no good extractive span did not offset the large loss on extractive examples.

Run notebook: `colab_pointer_generator_experiment.ipynb`.

Report notebook: `pointer_generator_results_report.ipynb`.
"""),
md("""## Step 6 — Test output objectives and noisy-label weighting

Joint span scoring, hard-negative ranking, and silver-span quality weighting were isolated in a controlled component study. Joint scoring and hard negatives did not help. Quality weighting looked promising in one seed, but a matched three-seed confirmation produced only +0.44 mean ROUGE-L with one negative seed and substantially higher variance. It was not adopted.

Run notebooks: `colab_joint_span_experiment.ipynb`, `colab_quality_weight_3seed.ipynb`.

Report notebooks: `joint_span_results_report.ipynb`, `quality_weight_3seed_report.ipynb`.
"""),
md("""## Step 7 — Measure whether reranking is theoretically worthwhile

The current gated model's Top-K span candidates were evaluated with a gold-aware oracle. Top-10 raised the ceiling from 44.14 to 67.16 ROUGE-L and increased exact-answer coverage from 27.32% to 49.16%. For 531 of 1,076 validation questions, a better span existed within the Top-10 list.

This result showed that candidate availability was not the main bottleneck; candidate ordering was.

Run notebook: `colab_topk_span_oracle.ipynb`.

Report notebook: `topk_span_oracle_report.ipynb`.
"""),
md("""## Step 8 — Train lightweight rerankers on in-sample candidates

A fixed pairwise logistic model and a shallow LambdaMART model used 30 interpretable features: base scores, answer length and position, lexical overlap, question type, and body/byline region. When their training candidates came from a base model that had already seen those questions, gains were weak: LambdaMART improved ROUGE-L by only 0.37 and the confidence interval included zero.

This exposed a stacking distribution mismatch rather than disproving reranking.

Run notebook: `colab_linear_span_reranker.ipynb`.

Report notebook: `linear_span_reranker_report.ipynb`.
"""),
md("""## Step 9 — Generate out-of-fold candidates

The training split was divided into three tweet-grouped folds. For every fold, a new base model was trained without that fold, with early stopping performed on a separate inner development subset. Each reranker-training candidate was therefore produced for a question and tweet unseen by its generating model.

The unchanged shallow LambdaMART reranker then improved the seed-1 final model by:

| Metric | Base Top-1 | OOF LambdaMART | Difference |
|---|---:|---:|---:|
| ROUGE-L | 44.14 | **45.56** | **+1.42** |
| F1 | 44.22 | **45.73** | **+1.50** |
| EM | 27.32 | **28.90** | **+1.58** |

The paired 95% bootstrap intervals were positive for all three metrics. This is the strongest successful extension so far.

Run notebook: `colab_oof_span_reranker.ipynb`.

Report notebook: `oof_span_reranker_report.ipynb`.
"""),
md("""## Step 10 — Interpret the current best system

The current best candidate is the original single-span TweetBiDAF v2 model followed by the OOF-trained shallow LambdaMART reranker. Compared with the earlier three-seed single-span mean of 44.51, the current seed-1 system reaches 45.56 ROUGE-L, an absolute difference of about +1.05. The controlled same-checkpoint gain is +1.42.

The reranker helps most on `what` and `who` questions and on body spans. It still harms several smaller slices, including `why`, `when`, and byline answers. These slice findings should guide future work but should not be used to tune and re-evaluate on the same validation set.
"""),
md("""## Step 11 — Recommended confirmation before final test use

Apply the already fitted OOF LambdaMART reranker to Top-10 candidates from the available full-model seeds. This measures sensitivity to base-model initialisation without retraining or tuning the reranker. If the gain remains positive across seeds, freeze the complete pipeline before accessing the locked test set.

Do not select new features, question-type routing rules, or thresholds from the same validation results without introducing a new development protocol.
"""),
md("""## Step 12 — Notebook execution map

| Order | Run notebook | Result notebook | Purpose |
|---:|---|---|---|
| 1 | `colab_v2_retrain.ipynb` | — | Establish the three-seed single-span baseline |
| 2 | `colab_attention_analysis.ipynb` | `attention_results_report.ipynb` | Confirm the value of self-attention |
| 3 | `colab_boundary_attention_experiment.ipynb` | `boundary_attention_results_report.ipynb` | Test boundary-conditioned output |
| 4 | `colab_pointer_generator_experiment.ipynb` | `pointer_generator_results_report.ipynb` | Test generation for non-extractive answers |
| 5 | `colab_joint_span_experiment.ipynb` | `joint_span_results_report.ipynb` | Isolate output-objective components |
| 6 | `colab_quality_weight_3seed.ipynb` | `quality_weight_3seed_report.ipynb` | Confirm quality weighting over three seeds |
| 7 | `colab_attention_variants.ipynb` | `attention_variants_report.ipynb` | Compare attention variants |
| 8 | `colab_topk_span_oracle.ipynb` | `topk_span_oracle_report.ipynb` | Measure reranking headroom |
| 9 | `colab_linear_span_reranker.ipynb` | `linear_span_reranker_report.ipynb` | Test in-sample linear and tree rerankers |
| 10 | `colab_oof_span_reranker.ipynb` | `oof_span_reranker_report.ipynb` | Train and evaluate leakage-controlled rerankers |
""")]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT)
