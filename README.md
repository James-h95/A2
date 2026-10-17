# TweetQA — CITS4012 group project

Working branch: `branchx`. Start with [the selected-model handoff](docs/handoff/FINAL_MODEL_HANDOFF.md).

| Folder / file | Purpose |
|---|---|
| `docs/planning/` | Original project plan and intermediate reranker instructions |
| `docs/handoff/` | Exact selected-system and reproduction instructions |
| `docs/reports/` | Report inputs and short experiment interpretations |
| `notebooks/experiments/` | Original Step 00–15 experiment notebooks, in step-number order |
| `notebooks/evaluation/` | Validation/evaluation notebooks already present in this checkout |
| `submission/` | Assignment template, notebook content cells, and future generated submission notebooks |
| `build_notebook.py` | Root entry point that assembles the submission scaffold |
| `src/` | Model, preprocessing, evaluation, baseline, and reranking implementation |
| `scripts/` | Experiment runners, analysis utilities, and notebook generators |
| `data/` | Original dataset CSVs; unchanged |
| `checkpoints/` | Existing saved main-model checkpoints; `boundary/` contains the separate boundary experiment |
| `artifacts/experiments/` | Original experiment ZIP archives, with unchanged contents and filenames |
| `results/` | Existing logs, predictions, histories, configurations and summaries |
| `results/figures/attention_preview/` | Original saved attention/training previews |
| `tests/` | Pipeline tests |
| `output/` | Existing generated presentation/report assets |

## Using the organised repository

Run terminal commands from the repository root. For example:

```sh
python scripts/make_colab_v2_notebook.py
python build_notebook.py --group YOUR_GROUP --hf-repo OWNER/REPO
```

Notebook generators now write to `notebooks/experiments/`. The submission builder reads `submission/template.ipynb` and `submission/notebook_parts/cells.py`, and writes its generated notebook to `submission/`. **The existing builder is still a scaffold: it has not yet been upgraded to the complete selected ensemble/reranker and final evaluation submission.**

Saved experiment and evaluation notebooks were moved byte-for-byte, preserving their code, outputs, metadata and historical paths. Colab workflows still use the same upload filenames and Drive paths. For a local report notebook that expects an archive beside it, provide its archive from `artifacts/experiments/` or select that archive using its existing upload mechanism; no legacy root aliases were added.

The source code, dataset, standard main-model checkpoint paths and recorded experiment outputs have not been rewritten. Internal archive member paths are unchanged. Older documents can contain historical root-level filenames; use [the move manifest](docs/REORGANISATION.json) to find their new repository paths.

This reorganisation neither retrains nor evaluates models, and does not commit or push changes. Nothing was deleted. Evaluation downloads stored elsewhere in Downloads were left there; this task only reorganises files already in this checkout.
