"""Build a self-contained Colab notebook for retraining TweetBiDAF v2.

The notebook embeds the project source code, keeps the test set locked, trains
the v2 main model for three seeds, and optionally runs selected ablations.
Artifacts are backed up to Google Drive after every completed run.
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "experiments" / "step01_colab_v2_retrain.ipynb"
FILES = [
    "src/__init__.py", "src/evaluate.py", "src/data.py", "src/runlog.py",
    "src/model.py", "src/fasttext_init.py", "src/train.py",
    "src/analysis.py", "scripts/stage5_tables.py", "scripts/stage7_analysis.py",
]


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


INTRO = """# CITS4012 A2 — TweetBiDAF v2 retraining on Google Colab

This notebook retrains the final **TweetBiDAF v2** model from scratch on TweetQA.

- Default run: v2 main model, seeds 1, 2 and 3.
- Optional run: selected ablations, disabled by default.
- Evaluation during development: validation only.
- The test set stays locked. This notebook does **not** perform final test evaluation.
- Checkpoints, histories, validation predictions and logs are copied to Google Drive after every run.

Before running, select **Runtime → Change runtime type → T4 GPU**. Then run cells from top to bottom.
When prompted, upload the assignment copy of `train.csv`. The test file is deliberately not used or uploaded here.
"""

SETTINGS = r'''# Settings: edit only this cell when choosing extra experiments.
RUN_SMOKE_TEST = True
RUN_V2_MAIN = True
RUN_ABLATIONS = False

# Used only when RUN_ABLATIONS=True. Start with S_A3_no_selfattn because it is
# required for the qualitative comparison of self-attention on vs off.
ABLATIONS_TO_RUN = ["S_A3_no_selfattn"]

# Set False only if you intentionally do not want persistent Google Drive backups.
SAVE_TO_DRIVE = True
DRIVE_DIR = "/content/drive/MyDrive/CITS4012_A2_v2"
'''

SETUP = r'''import os, sys, subprocess, shutil, json, glob, time
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
assert IN_COLAB, "This notebook is designed for Google Colab."
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "gensim", "emoji", "nltk", "matplotlib"], check=True)

import torch
assert torch.cuda.is_available(), "GPU is not enabled. Select Runtime > Change runtime type > T4 GPU."
print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)

WORK = Path("/content/A2_v2")
for p in (WORK / "src", WORK / "scripts", WORK / "data", WORK / "results" / "logs"):
    p.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BACKUP = Path(DRIVE_DIR)
    BACKUP.mkdir(parents=True, exist_ok=True)
    # Resume from artifacts saved by an earlier interrupted session.
    for name in ("checkpoints", "results"):
        src = BACKUP / name
        dst = WORK / name
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
else:
    BACKUP = None

def backup_artifacts():
    if BACKUP is None:
        return
    for name in ("checkpoints", "results"):
        src = WORK / name
        if src.exists():
            shutil.copytree(src, BACKUP / name, dirs_exist_ok=True)
    print("Backed up artifacts to", BACKUP)
'''

DATA = r'''# Upload only the TweetQA training file. Validation is carved from this file.
# The test file is deliberately excluded from this development notebook.
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
        shutil.copy(source, target)
    else:
        print("DATA BRANCH C: train.csv was not found; opening the browser upload widget.")
        from google.colab import files
        uploaded = files.upload()
        assert name in uploaded, f"Expected {name}; uploaded: {list(uploaded)}"
        target.write_bytes(uploaded[name])

import hashlib
expected = {"train.csv": "8a52bb7dff56b11fdad1ab6fd3b848fbadc22849"}
for name, sha1 in expected.items():
    # Git may check the CSV out with Windows CRLF endings. Normalise to LF so
    # byte-level line-ending changes do not reject otherwise identical data.
    content = (WORK / "data" / name).read_bytes().replace(b"\r\n", b"\n")
    actual = hashlib.sha1(content).hexdigest()
    print("SHA-1 check after newline normalisation:", name, actual)
    assert actual == sha1, f"Unexpected {name}; use the assignment-provided TweetQA file"
print("DATA READY: the training file is valid; continue to the next cell.")
'''

RUNNER = r'''# Frozen v2 configuration and experiment definitions.
FROZEN_V2 = json.loads(Path("results/FROZEN_CONFIG.json").read_text())
FROZEN_V2 = {k: v for k, v in FROZEN_V2.items() if not k.startswith("_")}
assert FROZEN_V2["soft_spans"] is False

MAIN_JOBS = [dict(name="A5_single_span", seed=s, set=FROZEN_V2) for s in (1, 2, 3)]
ABLATION_OVERRIDES = {
    "S_A1_no_char": {"use_char_cnn": False},
    "S_A2_no_norm_flags": {"use_normaliser": False, "use_token_flags": False},
    "S_A3_no_selfattn": {"use_self_attn": False},
    "S_A4_c2q_only": {"biattn_mode": "c2q"},
    "cand_lr5e-4_ft": {"soft_spans": True, "soft_tau": 0.5},  # A5: former v1
    "B2_vanilla": {"use_char_cnn": False, "use_normaliser": False,
                    "use_token_flags": False, "use_self_attn": False,
                    "biattn_mode": "c2q", "soft_spans": False},
}
unknown = set(ABLATIONS_TO_RUN) - set(ABLATION_OVERRIDES)
assert not unknown, f"Unknown ablations: {sorted(unknown)}"
ABLATION_JOBS = [
    dict(name=name, seed=s, set={**FROZEN_V2, **ABLATION_OVERRIDES[name]})
    for name in ABLATIONS_TO_RUN for s in (1, 2, 3)
]

def completed(name, seed):
    if not Path("results/runs.csv").exists():
        return False
    import csv
    return any(r["model"] == name and r["seed"] == str(seed) and r["split"] == "val"
               for r in csv.DictReader(open("results/runs.csv", encoding="utf-8")))

def run_job(job):
    if completed(job["name"], job["seed"]):
        print("SKIP completed:", job["name"], "seed", job["seed"])
        return
    args = [sys.executable, "-m", "src.train", "--name", job["name"],
            "--seed", str(job["seed"]), "--set"]
    args += [f"{k}={json.dumps(v)}" for k, v in job["set"].items()]
    print("RUN:", job["name"], "seed", job["seed"], flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    rc = subprocess.run(args, env=env).returncode
    assert rc == 0, f"Training failed: {job['name']} seed {job['seed']}"
    backup_artifacts()
'''

SMOKE = r'''if RUN_SMOKE_TEST:
    subprocess.run([sys.executable, "-m", "src.train", "--name", "smoke",
                    "--smoke", "--set", "lr=0.0005", "word_init=\"fasttext\"",
                    "soft_spans=false"], check=True)
    print("Smoke test passed")
'''

RUN_MAIN = r'''if RUN_V2_MAIN:
    for job in MAIN_JOBS:
        run_job(job)
else:
    print("RUN_V2_MAIN=False: main-model training skipped")
'''

RUN_ABLATIONS_CELL = r'''if RUN_ABLATIONS:
    for job in ABLATION_JOBS:
        run_job(job)
else:
    print("RUN_ABLATIONS=False: optional ablations skipped")
'''

SUMMARY = r'''import csv, statistics
rows = list(csv.DictReader(open("results/runs.csv", encoding="utf-8"))) if Path("results/runs.csv").exists() else []
groups = {}
for r in rows:
    if r["split"] == "val":
        groups.setdefault(r["model"], []).append(r)
for name, rs in groups.items():
    vals = [float(r["rougeL"]) for r in rs]
    tail = f" ± {statistics.stdev(vals):.2f}" if len(vals) > 1 else ""
    print(f"{name:24s} n={len(rs)} ROUGE-L={statistics.mean(vals):.2f}{tail}")
backup_artifacts()
'''

ANALYSIS = r'''# Run qualitative attention analysis only when the required checkpoints exist.
main_ready = completed("A5_single_span", 1)
a3_ready = completed("S_A3_no_selfattn", 1)
if main_ready and a3_ready:
    subprocess.run([sys.executable, "scripts/stage7_analysis.py"], check=True)
    backup_artifacts()
    print("Attention figures and error sample were generated.")
else:
    print("Stage 7 skipped. It requires A5_single_span seed 1 and S_A3_no_selfattn seed 1.")
'''

PACKAGE = r'''# Create a portable archive. Test evaluation is deliberately excluded.
import zipfile
archive = Path("/content/a2_v2_colab_artifacts.zip")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in (WORK / "checkpoints", WORK / "results"):
        if folder.exists():
            for p in folder.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(WORK))
print(archive, round(archive.stat().st_size / 1e6, 1), "MB")
if BACKUP is not None:
    drive_archive = BACKUP / archive.name
    shutil.copy2(archive, drive_archive)
    print("Saved updated archive to Google Drive:", drive_archive)
from google.colab import files
files.download(str(archive))
'''


def build():
    cells = [md(INTRO), md("## 0. Experiment switches"), code(SETTINGS),
             md("## 1. Runtime and persistent backup"), code(SETUP),
             md("## 2. TweetQA data"), code(DATA),
             md("## 3. Project source code")]
    for rel in FILES:
        body = (ROOT / rel).read_text(encoding="utf-8")
        cells.append(code(f"%%writefile {rel}\n{body}"))
    frozen = (ROOT / "results" / "FROZEN_CONFIG.json").read_text(encoding="utf-8")
    cells.append(code("Path('results/FROZEN_CONFIG.json').write_text('''" + frozen + "''')"))
    cells += [md("## 4. Jobs and resume logic"), code(RUNNER),
              md("## 5. Smoke test"), code(SMOKE),
              md("## 6. Retrain TweetBiDAF v2 — three seeds"), code(RUN_MAIN),
              md("## 7. Optional ablations"), code(RUN_ABLATIONS_CELL),
              md("## 8. Validation summary"), code(SUMMARY),
              md("## 9. Optional attention analysis"), code(ANALYSIS),
              md("## 10. Download artifacts"), code(PACKAGE)]
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
