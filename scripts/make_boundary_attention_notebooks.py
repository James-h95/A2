"""Build the training and lightweight report notebooks for the boundary-aware head."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_OUT = ROOT / "colab_boundary_attention_experiment.ipynb"
REPORT_OUT = ROOT / "boundary_attention_results_report.ipynb"
FILES = [
    "src/__init__.py", "src/evaluate.py", "src/data.py", "src/runlog.py",
    "src/model.py", "src/fasttext_init.py", "src/train.py",
]


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


SETTINGS = r'''# The experiment is isolated from all earlier A2 runs.
RUN_SMOKE_TEST = True
RUN_THREE_SEEDS = True
SAVE_TO_DRIVE = True
AUTO_DOWNLOAD_ZIP = False
DRIVE_DIR = "/content/drive/MyDrive/CITS4012_A2_boundary_attention"
ARCHIVE_NAME = "boundary_attention_artifacts.zip"
'''

SETUP = r'''import os, sys, subprocess, shutil, json, csv, statistics, zipfile, hashlib
from pathlib import Path

assert "google.colab" in sys.modules, "Run this notebook in Google Colab."
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gensim", "emoji", "nltk"], check=True)
import torch
assert torch.cuda.is_available(), "Enable a GPU runtime before continuing."
print("GPU:", torch.cuda.get_device_name(0), "| torch:", torch.__version__)

WORK = Path("/content/A2_boundary_attention")
RESULTS = WORK / "boundary_results"
CHECKPOINTS = WORK / "boundary_checkpoints"
for p in (WORK / "src", WORK / "data", RESULTS / "logs", CHECKPOINTS):
    p.mkdir(parents=True, exist_ok=True)
os.environ["A2_RESULTS"] = str(RESULTS)
os.environ["A2_CKPT"] = str(CHECKPOINTS)
os.chdir(WORK)

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BACKUP = Path(DRIVE_DIR)
    BACKUP.mkdir(parents=True, exist_ok=True)
    for src_name, dst in (("boundary_results", RESULTS), ("boundary_checkpoints", CHECKPOINTS)):
        src = BACKUP / src_name
        if src.exists(): shutil.copytree(src, dst, dirs_exist_ok=True)
else:
    BACKUP = None

def backup_artifacts():
    if BACKUP is None: return
    for src, name in ((RESULTS, "boundary_results"), (CHECKPOINTS, "boundary_checkpoints")):
        if src.exists(): shutil.copytree(src, BACKUP / name, dirs_exist_ok=True)
    print("Backed up to", BACKUP)
'''

DATA = r'''# Only train.csv is used; validation is carved from it exactly as in v2.
target = WORK / "data" / "train.csv"
if not target.exists():
    candidates = [Path("/content/train.csv"), Path("/content/data/train.csv"), Path("/content/A2/data/train.csv")]
    source = next((p for p in candidates if p.exists()), None)
    if source:
        shutil.copy2(source, target); print("Copied", source, "->", target)
    else:
        from google.colab import files
        uploaded = files.upload()
        assert "train.csv" in uploaded
        target.write_bytes(uploaded["train.csv"])
content = target.read_bytes().replace(b"\r\n", b"\n")
actual = hashlib.sha1(content).hexdigest()
assert actual == "8a52bb7dff56b11fdad1ab6fd3b848fbadc22849", "Use the assignment train.csv"
print("DATA READY:", target, actual)
'''

JOBS = r'''from src import data as D
from src import train as T

frozen = json.loads(Path("results/FROZEN_CONFIG.json").read_text())
frozen = {k: v for k, v in frozen.items() if not k.startswith("_")}
CONTROL = {**frozen, "condition_end_on_start": False}
BOUNDARY = {**frozen, "condition_end_on_start": True}
EXPERIMENTS = {
    "BoundaryControl_v2": CONTROL,
    "BoundaryAware_start2end": BOUNDARY,
}

def completed(name, seed):
    p = RESULTS / "runs.csv"
    return p.exists() and any(r["model"] == name and r["seed"] == str(seed) and r["split"] == "val"
                              for r in csv.DictReader(p.open(encoding="utf-8")))

def run_job(name, seed, cfg):
    if completed(name, seed):
        print("SKIP completed:", name, seed); return
    args = [sys.executable, "-m", "src.train", "--name", name, "--seed", str(seed), "--set"]
    args += [f"{k}={json.dumps(v)}" for k, v in cfg.items()]
    print("RUN:", name, "seed", seed, flush=True)
    subprocess.run(args, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    backup_artifacts()
'''

SMOKE = r'''if RUN_SMOKE_TEST:
    args = [sys.executable, "-m", "src.train", "--name", "boundary_smoke", "--smoke", "--set",
            "lr=0.0005", 'word_init="fasttext"', "soft_spans=false", "condition_end_on_start=true"]
    subprocess.run(args, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    print("BOUNDARY-AWARE SMOKE TEST PASSED")
'''

TRAIN = r'''if RUN_THREE_SEEDS:
    for name, cfg in EXPERIMENTS.items():
        for seed in (1, 2, 3): run_job(name, seed, cfg)
else:
    print("Training disabled")
'''

ANALYSE = r'''# Create all report inputs while train.csv and preprocessing code are available.
from src.evaluate import token_f1
runs = list(csv.DictReader((RESULTS / "runs.csv").open(encoding="utf-8")))
selected = [r for r in runs if r["model"] in EXPERIMENTS and r["split"] == "val"]
for name in EXPERIMENTS:
    assert len([r for r in selected if r["model"] == name]) == 3, f"Need 3 completed seeds for {name}"

_, val_rows = D.get_splits()
val_ex = D.preprocess(val_rows, dict(use_normaliser=True, max_span_len=frozen["max_span_len"], tau=frozen["tau"]))
by_id = {e["row_id"]: e for e in val_ex}
detail = []
for r in selected:
    preds = json.loads((RESULTS / "preds" / f"{r['run_id']}.json").read_text())
    for p in preds:
        e = by_id[p["row_id"]]
        pi, pj = p["span"]
        gi, gj = e["best"] if e["best"] is not None else (-1, -1)
        span_iou = 0.0
        if gi >= 0:
            inter = max(0, min(pj, gj) - max(pi, gi) + 1)
            union = max(pj, gj) - min(pi, gi) + 1
            span_iou = inter / union
        detail.append(dict(
            model=r["model"], seed=int(r["seed"]), run_id=r["run_id"], row_id=e["row_id"],
            question=e["question"], gold=" | ".join(e["answers"]), pred=p["pred"],
            pred_start=pi, pred_end=pj, gold_start=gi, gold_end=gj,
            start_exact=int(pi == gi and gi >= 0), end_exact=int(pj == gj and gj >= 0),
            span_exact=int(pi == gi and pj == gj and gi >= 0), span_iou=round(span_iou, 6),
            answer_f1=round(100 * max(token_f1(p["pred"], a) for a in e["answers"]), 4),
            trainable=int(e["trainable"])))

cols = list(detail[0])
with (RESULTS / "boundary_details.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(detail)

def mean_sd(xs):
    return [statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0]
summary = {"models": {}, "paired_seed_deltas": {}}
for name in EXPERIMENTS:
    rs = [r for r in selected if r["model"] == name]
    ds = [d for d in detail if d["model"] == name and d["trainable"]]
    summary["models"][name] = {
        "run_ids": [r["run_id"] for r in rs],
        "metrics": {m: mean_sd([float(r[m]) for r in rs]) for m in ("bleu1", "meteor", "rougeL", "em", "f1")},
        "boundary": {k: mean_sd([100 * sum(d[k] for d in ds if d["seed"] == s) /
                                  sum(1 for d in ds if d["seed"] == s) for s in (1, 2, 3)])
                     for k in ("start_exact", "end_exact", "span_exact")},
        "span_iou": mean_sd([100 * sum(d["span_iou"] for d in ds if d["seed"] == s) /
                              sum(1 for d in ds if d["seed"] == s) for s in (1, 2, 3)]),
    }
for s in (1, 2, 3):
    c = next(r for r in selected if r["model"] == "BoundaryControl_v2" and r["seed"] == str(s))
    b = next(r for r in selected if r["model"] == "BoundaryAware_start2end" and r["seed"] == str(s))
    summary["paired_seed_deltas"][str(s)] = {m: float(b[m]) - float(c[m]) for m in ("bleu1", "meteor", "rougeL", "em", "f1")}
(RESULTS / "boundary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2)); backup_artifacts()
'''

PACKAGE = r'''archive = Path("/content") / ARCHIVE_NAME
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in (RESULTS, CHECKPOINTS):
        for p in folder.rglob("*"):
            if p.is_file(): z.write(p, p.relative_to(WORK))
if BACKUP is not None:
    shutil.copy2(archive, BACKUP / ARCHIVE_NAME)
    print("Saved archive to", BACKUP / ARCHIVE_NAME)
print("Archive:", archive, f"{archive.stat().st_size/1e6:.1f} MB")
if AUTO_DOWNLOAD_ZIP:
    from google.colab import files
    files.download(str(archive))
else:
    print("Download it from MyDrive when ready; automatic browser download is disabled.")
'''

LOAD_REPORT = r'''import csv, io, json, statistics, sys, zipfile
from pathlib import Path
from IPython.display import display, Markdown

NAME = "boundary_attention_artifacts.zip"
candidates = [Path(NAME), Path("/content") / NAME,
              Path("/content/drive/MyDrive/CITS4012_A2_boundary_attention") / NAME]
archive = next((p for p in candidates if p.exists()), None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive")
    archive = next((p for p in candidates if p.exists()), None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import files
    uploaded = files.upload(); assert NAME in uploaded
    archive = Path("/content") / NAME; archive.write_bytes(uploaded[NAME])
assert archive is not None, f"Place {NAME} beside this notebook or upload it."
zf = zipfile.ZipFile(archive)
required = {"boundary_results/runs.csv", "boundary_results/boundary_summary.json",
            "boundary_results/boundary_details.csv"}
assert required <= set(zf.namelist()), f"Missing: {sorted(required-set(zf.namelist()))}"
runs = list(csv.DictReader(io.StringIO(zf.read("boundary_results/runs.csv").decode())))
summary = json.loads(zf.read("boundary_results/boundary_summary.json"))
details = list(csv.DictReader(io.StringIO(zf.read("boundary_results/boundary_details.csv").decode())))
print("Loaded", archive, "| rows", len(details), "| no training or inference")
'''

REPORT_METRICS = r'''models = [("BoundaryControl_v2", "Current v2"),
          ("BoundaryAware_start2end", "Start-conditioned end head")]
metrics = ("bleu1", "meteor", "rougeL", "em", "f1")
lines = ["| Model | BLEU-1 | METEOR | ROUGE-L | EM | F1 |", "|---|---:|---:|---:|---:|---:|"]
for key, label in models:
    cells = [f"{summary['models'][key]['metrics'][m][0]:.2f} ± {summary['models'][key]['metrics'][m][1]:.2f}" for m in metrics]
    lines.append("| " + label + " | " + " | ".join(cells) + " |")
display(Markdown("## 1. Validation results over three seeds\n\n" + "\n".join(lines)))
deltas = {m: statistics.mean([v[m] for v in summary["paired_seed_deltas"].values()]) for m in metrics}
display(Markdown(f"Paired mean change (new − v2): **ROUGE-L {deltas['rougeL']:+.2f}**, "
                 f"**F1 {deltas['f1']:+.2f}**, **EM {deltas['em']:+.2f}** points."))
'''

REPORT_BOUNDARY = r'''lines = ["| Model | Start exact | End exact | Full span exact | Span IoU |",
         "|---|---:|---:|---:|---:|"]
for key, label in models:
    b = summary["models"][key]["boundary"]
    vals = [b[x] for x in ("start_exact", "end_exact", "span_exact")]
    vals.append(summary["models"][key]["span_iou"])
    lines.append("| " + label + " | " + " | ".join(f"{m:.2f} ± {s:.2f}" for m, s in vals) + " |")
display(Markdown("## 2. Boundary diagnostics on trainable validation examples\n\n" + "\n".join(lines)))
display(Markdown("Start/end exactness is measured against the deterministic best silver span. "
                 "It diagnoses boundary selection; it is not a substitute for answer-text metrics."))
'''

REPORT_CASES = r'''# Compare seed-matched predictions and show the clearest improvements/regressions.
by = {(r["model"], int(r["seed"]), int(r["row_id"])): r for r in details}
pairs = []
ids = sorted({int(r["row_id"]) for r in details})
for seed in (1, 2, 3):
    for rid in ids:
        c = by.get(("BoundaryControl_v2", seed, rid)); n = by.get(("BoundaryAware_start2end", seed, rid))
        if c and n:
            pairs.append((float(n["answer_f1"])-float(c["answer_f1"]), seed, rid, c, n))

def case_table(items, title):
    lines = ["| ΔF1 | Seed | Question | Gold | v2 prediction | New prediction |", "|---:|---:|---|---|---|---|"]
    for d, seed, _, c, n in items:
        clean = lambda x: x.replace("|", "/").replace("\n", " ")
        lines.append(f"| {d:+.1f} | {seed} | {clean(c['question'])} | {clean(c['gold'])} | {clean(c['pred'])} | {clean(n['pred'])} |")
    display(Markdown(f"### {title}\n\n" + "\n".join(lines)))
case_table(sorted(pairs, reverse=True)[:10], "Largest improvements")
case_table(sorted(pairs)[:10], "Largest regressions")
'''

REPORT_CONCLUSION = r'''report = f"""# Boundary-aware attention experiment

The start-conditioned end head changes mean validation ROUGE-L by {deltas['rougeL']:+.2f},
F1 by {deltas['f1']:+.2f}, and EM by {deltas['em']:+.2f} points relative to the
same TweetBiDAF v2 configuration trained in the same notebook over three seeds.

The boundary table determines whether any change is specifically associated with
start/end selection rather than only an aggregate metric. The example tables show
both improvements and regressions. Results are validation-only; the locked test set
was not accessed. Silver-span exactness is diagnostic because some TweetQA gold
answers are not extractive.
"""
Path("boundary_attention_report.md").write_text(report, encoding="utf-8")
display(Markdown("## 4. Report-ready conclusion\n\n" + report))
print("Saved boundary_attention_report.md")
zf.close()
'''


def notebook(cells, gpu=False):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"; meta["colab"]["gpuType"] = "T4"
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def build():
    train_cells = [md("""# Boundary-aware TweetBiDAF experiment

This notebook performs one controlled validation experiment. It trains the current
TweetBiDAF v2 and a variant whose end head attends to the predicted start distribution,
using identical data, hyperparameters, and seeds. The test set is never loaded.
Artifacts are kept separate from all earlier runs."""),
        md("## 0. Settings"), code(SETTINGS), md("## 1. Runtime and Drive backup"), code(SETUP),
        md("## 2. Upload train.csv"), code(DATA), md("## 3. Embedded project source")]
    for rel in FILES:
        train_cells.append(code(f"%%writefile {rel}\n" + (ROOT / rel).read_text(encoding="utf-8")))
    frozen = (ROOT / "results" / "FROZEN_CONFIG.json").read_text(encoding="utf-8")
    train_cells += [code("Path('results').mkdir(exist_ok=True)\nPath('results/FROZEN_CONFIG.json').write_text('''" + frozen + "''')"),
                    md("## 4. Controlled jobs"), code(JOBS), md("## 5. Smoke test"), code(SMOKE),
                    md("## 6. Train both models, three seeds each"), code(TRAIN),
                    md("## 7. Build boundary diagnostics"), code(ANALYSE),
                    md("## 8. Package to the separate MyDrive folder"), code(PACKAGE)]
    report_cells = [md("""# Boundary-aware attention results

This lightweight notebook reads `boundary_attention_artifacts.zip`, compares the
controlled runs, and shows boundary diagnostics and paired examples. It performs no
training, checkpoint loading, inference, or test-set access."""),
        md("## 0. Load the downloaded artifact"), code(LOAD_REPORT), code(REPORT_METRICS),
        code(REPORT_BOUNDARY), md("## 3. Paired qualitative cases"), code(REPORT_CASES),
        code(REPORT_CONCLUSION)]
    TRAIN_OUT.write_text(json.dumps(notebook(train_cells, True), ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(notebook(report_cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", TRAIN_OUT.name, len(train_cells), "cells")
    print("wrote", REPORT_OUT.name, len(report_cells), "cells")


if __name__ == "__main__":
    build()
