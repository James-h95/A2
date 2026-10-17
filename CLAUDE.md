# CITS4012 A2 — TweetQA group project

- The full plan is `docs/planning/A2_PLAN.md`. The assignment spec is `../CITS4012_A2_2026.pdf`, and the spec wins on any conflict.
- Run the stages in order without stopping. Stop and ask only at the two checkpoints in docs/planning/A2_PLAN.md: the Stage 0 data check and the test-set evaluation before Stage 8. Log each stage's summary in results/PROGRESS.md.
- Follow the "Hard rules" in docs/planning/A2_PLAN.md: main model from scratch, no copied code from GitHub or paper repos, no external data, test set locked until Stage 8, every reported number logged in results/runs.csv.
- **Do not train on the Mac.** Full training runs go on a Kaggle/Colab GPU via `step01a_gpu_runner.ipynb` (built by `scripts/make_gpu_notebook.py`), which the user runs. Locally, only unit tests and ≤200-example smoke tests (seconds). Ask before starting any background job longer than a minute or two.
- The git database lives outside OneDrive at `~/git-dirs/CITS4012-A2.git` (`A2/.git` is a pointer file). Model checkpoints are git-ignored. Share them via OneDrive or the Hugging Face repo.
- Keep the code in src/ as .py modules with tests in tests/. The single submission notebook is assembled at the end by build_notebook.py.

- Repository layout: experiment notebooks are in `notebooks/experiments/`; experiment ZIPs are in `artifacts/experiments/`; the submission template and content are in `submission/`. Use `README.md` and `docs/REORGANISATION.json` to resolve older paths.
