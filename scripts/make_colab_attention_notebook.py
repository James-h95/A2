"""Build the Colab notebook for the self-attention ablation and figures."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_GENERATOR = Path(__file__).with_name("make_colab_v2_notebook.py")
OUT = ROOT / "colab_attention_analysis.ipynb"


def load_base():
    spec = importlib.util.spec_from_file_location("make_colab_v2_notebook", BASE_GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build():
    base = load_base()
    intro = """# CITS4012 A2 — Self-attention ablation and attention figures

This notebook continues from the TweetBiDAF v2 checkpoints already saved in
`MyDrive/CITS4012_A2_v2`.

- It restores the existing v2 checkpoints and results from Google Drive.
- It does **not** retrain the three v2 main-model runs.
- It trains `S_A3_no_selfattn` for seeds 1, 2 and 3.
- It then produces the attention figures and qualitative error sample.
- The test set remains locked and is not used.

Select a GPU runtime, then run every cell from top to bottom. When prompted,
upload the assignment copy of `train.csv`.
"""
    settings = r'''# This notebook is preconfigured for the A3 self-attention ablation.
RUN_SMOKE_TEST = False
RUN_V2_MAIN = False
RUN_ABLATIONS = True
ABLATIONS_TO_RUN = ["S_A3_no_selfattn"]

SAVE_TO_DRIVE = True
DRIVE_DIR = "/content/drive/MyDrive/CITS4012_A2_v2"
'''

    cells = [base.md(intro), base.md("## 0. Fixed experiment settings"), base.code(settings),
             base.md("## 1. Runtime and restore from Google Drive"), base.code(base.SETUP),
             base.md("## 2. TweetQA training data"), base.code(base.DATA),
             base.md("## 3. Project source code")]
    for rel in base.FILES:
        body = (ROOT / rel).read_text(encoding="utf-8")
        cells.append(base.code(f"%%writefile {rel}\n{body}"))
    frozen = (ROOT / "results" / "FROZEN_CONFIG.json").read_text(encoding="utf-8")
    cells.append(base.code("Path('results/FROZEN_CONFIG.json').write_text('''" + frozen + "''')"))
    cells += [base.md("## 4. A3 jobs and resume logic"), base.code(base.RUNNER),
              base.md("## 5. Confirm that the smoke test is disabled"), base.code(base.SMOKE),
              base.md("## 6. Confirm that v2 main-model retraining is disabled"), base.code(base.RUN_MAIN),
              base.md("## 7. Train S_A3_no_selfattn — three seeds"), base.code(base.RUN_ABLATIONS_CELL),
              base.md("## 8. Validation summary"), base.code(base.SUMMARY),
              base.md("## 9. Generate attention figures and error analysis"), base.code(base.ANALYSIS),
              base.md("## 10. Download updated artifacts"), base.code(base.PACKAGE)]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"gpuType": "T4", "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, "with", len(cells), "cells")


if __name__ == "__main__":
    build()
