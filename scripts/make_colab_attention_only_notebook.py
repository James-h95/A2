"""Build a Colab notebook that only reruns qualitative attention analysis."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "experiments" / "step02b_colab_attention_only.ipynb"


def load_base():
    path = Path(__file__).with_name("make_colab_v2_notebook.py")
    spec = importlib.util.spec_from_file_location("make_colab_v2_notebook", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


INTRO = """# CITS4012 A2 — Attention analysis only

This notebook uses the already trained TweetBiDAF v2 and
`S_A3_no_selfattn` checkpoints. **It performs no training.**

It restores `a2_v2_colab_artifacts.zip` from Google Drive, regenerates the
attention heatmaps and qualitative analysis on the validation split, previews
the PNG figures, and saves a small analysis-only ZIP to Google Drive.

The test set remains locked. Only the assignment copy of `train.csv` is needed.
Run every cell from top to bottom.
"""


SETUP = r'''import os, sys, subprocess, shutil, zipfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
assert IN_COLAB, "This notebook is designed for Google Colab."
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "gensim", "emoji", "nltk", "matplotlib"], check=True)

import torch
print("Device available:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

from google.colab import drive
drive.mount("/content/drive")

WORK = Path("/content/A2_attention_only")
DRIVE_DIR = Path("/content/drive/MyDrive/CITS4012_A2_v2")
SOURCE_ARCHIVE = DRIVE_DIR / "a2_v2_colab_artifacts.zip"
assert SOURCE_ARCHIVE.exists(), f"Archive not found: {SOURCE_ARCHIVE}"

if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
with zipfile.ZipFile(SOURCE_ARCHIVE) as z:
    z.extractall(WORK)
for p in (WORK / "src", WORK / "scripts", WORK / "data"):
    p.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
print("Restored:", SOURCE_ARCHIVE)
'''


DATA = r'''# Upload only the assignment-provided TweetQA train.csv.
# Validation is deterministically carved from this file; test.csv is not used.
import hashlib

name = "train.csv"
target = WORK / "data" / name
print("Expected training-data path:", target)
if target.exists():
    print("DATA BRANCH A: train.csv already exists at the expected path; no upload or copy needed.")
else:
    # VS Code's "Upload to Colab" normally puts the file in one of these locations.
    known_uploads = [
        Path("/content/train.csv"),
        Path("/content/data/train.csv"),
        Path("/content/A2/data/train.csv"),
    ]
    source = next((p for p in known_uploads if p.exists()), None)
    if source is not None:
        print("DATA BRANCH B: found a VS Code upload; copying", source, "->", target)
        shutil.copy2(source, target)
    else:
        print("DATA BRANCH C: train.csv was not found; opening the browser upload widget.")
        from google.colab import files
        uploaded = files.upload()
        assert name in uploaded, f"Expected {name}; uploaded: {list(uploaded)}"
        target.write_bytes(uploaded[name])

content = target.read_bytes().replace(b"\r\n", b"\n")
actual = hashlib.sha1(content).hexdigest()
expected = "8a52bb7dff56b11fdad1ab6fd3b848fbadc22849"
print("SHA-1 after newline normalisation:", actual)
assert actual == expected, "Use the assignment-provided TweetQA train.csv"
print("DATA READY")
'''


VERIFY = r'''import csv

rows = list(csv.DictReader(open(WORK / "results" / "runs.csv", encoding="utf-8")))

def checkpoint_for(model, seed=1):
    matches = [r for r in rows if r["model"] == model and r["seed"] == str(seed) and r["split"] == "val"]
    assert matches, f"Missing run record: {model}, seed {seed}"
    path = WORK / "checkpoints" / f"{matches[-1]['run_id']}.pt"
    assert path.exists(), f"Missing checkpoint: {path}"
    return path

main_checkpoint = checkpoint_for("A5_single_span", 1)
a3_checkpoint = checkpoint_for("S_A3_no_selfattn", 1)
print("Main checkpoint:", main_checkpoint.name)
print("A3 checkpoint:  ", a3_checkpoint.name)
print("CHECKPOINTS READY — no training will be run")
'''


ANALYSE = r'''# Run the analysis directly so any error is shown at its original line.
import runpy
runpy.run_path(str(WORK / "scripts" / "stage7_analysis.py"), run_name="__main__")
print("ATTENTION ANALYSIS COMPLETE")
'''


PREVIEW = r'''from IPython.display import display, Image

pngs = sorted((WORK / "results" / "figures").glob("*.png"))
print(f"Generated {len(pngs)} PNG figures:")
for p in pngs:
    print(" -", p.name)
for p in pngs:
    display(Image(filename=str(p), width=1000))
'''


PACKAGE = r'''# Package only the attention-analysis deliverables.
analysis_zip = Path("/content/a2_attention_analysis.zip")
with zipfile.ZipFile(analysis_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in ("results/figures", "results/tables"):
        folder = WORK / rel
        if folder.exists():
            for p in folder.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(WORK))
    for rel in ("results/error_sample.csv", "results/stage7_summary.json"):
        p = WORK / rel
        if p.exists():
            z.write(p, p.relative_to(WORK))

drive_zip = DRIVE_DIR / analysis_zip.name
shutil.copy2(analysis_zip, drive_zip)
print("Saved analysis ZIP to Google Drive:", drive_zip)
print("Size:", round(analysis_zip.stat().st_size / 1e6, 1), "MB")

from google.colab import files
files.download(str(analysis_zip))
'''


def build():
    base = load_base()
    cells = [base.md(INTRO),
             base.md("## 1. Restore the trained artifacts from Google Drive"), base.code(SETUP),
             base.md("## 2. Upload and validate train.csv"), base.code(DATA),
             base.md("## 3. Analysis source code")]
    for rel in base.FILES:
        body = (ROOT / rel).read_text(encoding="utf-8")
        cells.append(base.code(f"%%writefile {rel}\n{body}"))
    cells += [base.md("## 4. Verify the required checkpoints"), base.code(VERIFY),
              base.md("## 5. Generate the attention analysis"), base.code(ANALYSE),
              base.md("## 6. Preview the generated figures"), base.code(PREVIEW),
              base.md("## 7. Save and download the analysis package"), base.code(PACKAGE)]
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
