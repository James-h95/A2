"""Build gpu_runner.ipynb: a self-contained Kaggle/Colab notebook that runs the GPU jobs
(frozen main model, ablations A1-A5, B2, and baselines B3-B5) and zips the results.

  python scripts/make_gpu_notebook.py
"""
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
FILES = ["src/__init__.py", "src/baselines/__init__.py", "src/evaluate.py", "src/data.py", "src/runlog.py",
         "src/model.py", "src/fasttext_init.py", "src/train.py", "src/baselines/b3_extractive_ft.py",
         "src/baselines/b4_t5_ft.py", "src/baselines/b5_llm.py", "scripts/run_queue.py"]

FROZEN = {k: v for k, v in json.loads((ROOT / "results" / "FROZEN_CONFIG.json").read_text()).items()
          if not k.startswith("_")}
OVR = {"lr": FROZEN["lr"], "word_init": FROZEN["word_init"]}
ABL = {
    "cand_lr5e-4_ft": {},
    "A1_no_char": {"use_char_cnn": False},
    "A2_no_norm_flags": {"use_normaliser": False, "use_token_flags": False},
    "A3_no_selfattn": {"use_self_attn": False},
    "A4_c2q_only": {"biattn_mode": "c2q"},
    "A5_single_span": {"soft_spans": False},
    "B2_vanilla": {"use_char_cnn": False, "use_normaliser": False, "use_token_flags": False,
                   "use_self_attn": False, "biattn_mode": "c2q", "soft_spans": False},
}
MAIN_JOBS = [dict(name=n, seed=s, set={**OVR, **o}) for n, o in ABL.items() for s in (1, 2, 3)]
STAGE34_JOBS = ([dict(name="main_v1", seed=1, set={})] + [dict(name="sw_base", seed=s, set={}) for s in (2, 3)] +
                [dict(name=n, seed=1, set=o) for n, o in [("sw_lr5e-4", {"lr": 5e-4}), ("sw_h96", {"hidden": 96}),
                 ("sw_h192", {"hidden": 192}), ("sw_do0.3", {"dropout": 0.3}), ("sw_L10", {"max_span_len": 10}),
                 ("sw_L20", {"max_span_len": 20}), ("sw_fasttext", {"word_init": "fasttext"})]] +
                [dict(name="sw_lr5e-4", seed=s, set={"lr": 5e-4}) for s in (2, 3)])

INTRO = """# CITS4012 A2: GPU runner (Kaggle / Colab)

This notebook runs every GPU job for the project and packs the results into **one zip** to download.
It is only a runner. The submission notebook is built separately from the same code.

**What it runs**
- the frozen main model, 3 seeds
- ablations A1–A5 and B2, 3 seeds each
- baselines B3 (BERTweet), B4 (t5-small) and B5 (Qwen2.5-1.5B)

Everything is evaluated on validation only; the test set stays locked. Expected time is about 1–1.5 h on Kaggle's 2×T4.

### Kaggle (recommended)
1. kaggle.com → **Create → New Notebook → File → Import Notebook**, then pick `gpu_runner.ipynb`.
2. Right panel → **Add Input → Upload** a dataset containing `train.csv` and `test.csv` (from the TweetQA folder). Any dataset name is fine.
3. Right panel → **Settings**: *Accelerator* = **GPU T4 x2**, *Internet* = **On** (needs a phone-verified account).
4. Click **Save Version → Save & Run All (Commit)**. It runs in the background, so you can close the tab.
5. When it finishes, open the version → **Output** → download **`a2_gpu_results.zip`**, and send it back (put it in the `A2/` folder).

### Colab
1. colab.research.google.com → **File → Upload notebook** → `gpu_runner.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**. When asked, upload `train.csv` and `test.csv`.
4. Keep the tab open (free Colab can disconnect when idle). The zip downloads automatically at the end.
"""

SETUP = '''# ---------------- settings ----------------
RUN_MAIN = True        # frozen model + ablations A1-A5 + B2 (21 runs)
RUN_BASELINES = True   # B3 BERTweet, B4 t5-small, B5 Qwen2.5-1.5B (val)
RUN_STAGE34 = False    # also redo the Stage 3/4 sweep on GPU (12 more runs, ~30 min)

import os, sys, subprocess, glob, shutil, json, time
from pathlib import Path
ON_KAGGLE = Path("/kaggle/working").exists()
IN_COLAB = "google.colab" in sys.modules
# On Kaggle the work happens in /tmp: everything in /kaggle/working gets uploaded as output when the
# version is saved, and a large output there made Kaggle hang. Only the final zip goes to /kaggle/working.
WORK = Path("/tmp/A2") if ON_KAGGLE else Path("/content/A2") if IN_COLAB else Path("A2_gpu").resolve()
(WORK / "src" / "baselines").mkdir(parents=True, exist_ok=True)
(WORK / "scripts").mkdir(exist_ok=True)
(WORK / "data").mkdir(exist_ok=True)
(WORK / "results" / "logs").mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gensim", "emoji", "nltk", "sentencepiece"], check=True)
import torch, gensim, transformers
N_GPU = torch.cuda.device_count()
assert N_GPU > 0, "No GPU: switch the accelerator/runtime to a GPU (see instructions above)"
print("platform:", "kaggle" if ON_KAGGLE else "colab" if IN_COLAB else "other", "| GPUs:",
      [torch.cuda.get_device_name(i) for i in range(N_GPU)], "| torch", torch.__version__, "| transformers", transformers.__version__, "| gensim", gensim.__version__)
'''

DATA = '''# ---------------- data: find train.csv / test.csv ----------------
def find(name):
    pats = [f"/kaggle/input/**/{name}", f"/content/**/{name}", str(WORK / "data" / name)]
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
for name in ("train.csv", "test.csv"):
    src = find(name)
    if src is None and IN_COLAB:
        from google.colab import files
        print(f"Please upload {name}")
        up = files.upload()
        src = next(iter(up))
    assert src, f"{name} not found: add it as a Kaggle input (see instructions)"
    if Path(src).resolve() != (WORK / "data" / name).resolve():
        shutil.copy(src, WORK / "data" / name)
import hashlib
for name in ("train.csv", "test.csv"):
    h = hashlib.sha1((WORK / "data" / name).read_bytes()).hexdigest()
    print(name, h)
assert hashlib.sha1((WORK / "data" / "train.csv").read_bytes()).hexdigest() == "8a52bb7dff56b11fdad1ab6fd3b848fbadc22849", "unexpected train.csv"
assert hashlib.sha1((WORK / "data" / "test.csv").read_bytes()).hexdigest() == "e33faa7614839d10a06b3607e74e51b70a0ee329", "unexpected test.csv"
'''

RUN_HELPER = '''# ---------------- helper: run a command, stream its output, keep a log ----------------
def sh(cmd, log=None):
    print("$", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    f = open(log, "w") if log else None
    for line in p.stdout:
        if f: f.write(line); f.flush()
        if "Warning" not in line and "it/s]" not in line:
            print(line, end="", flush=True)
    rc = p.wait()
    if f: f.close()
    print("exit code", rc)
    return rc

# smoke test (200 examples, 2 epochs, checks loss decreases and attention shapes)
assert sh([sys.executable, "-m", "src.train", "--name", "smoke", "--smoke"]) == 0
'''


def main_cell(jobs_var, tag, flag):
    return f'''if {flag}:
    Path("results/{tag}_jobs.json").write_text(json.dumps({jobs_var}))
    t0 = time.time()
    sh([sys.executable, "scripts/run_queue.py", "results/{tag}_jobs.json", "--parallel", str(2 * N_GPU),
        "--gpus", str(N_GPU), "--tag", "{tag}"], log="results/logs/{tag}_queue.log")
    print(f"{tag} done in {{(time.time() - t0) / 60:.1f}} min")
'''


BASELINES = '''if RUN_BASELINES:
    sh([sys.executable, "src/baselines/b3_extractive_ft.py", "--seed", "1"], log="results/logs/stage6_b3_bertweet_s1.log")
    sh([sys.executable, "src/baselines/b4_t5_ft.py", "--seed", "1"], log="results/logs/stage6_b4_t5small_s1.log")
    sh([sys.executable, "src/baselines/b5_llm.py"], log="results/logs/stage6_b5_qwen1.5b.log")
'''

SUMMARY = '''# ---------------- summary of everything logged in this session ----------------
import csv, statistics
rows = list(csv.DictReader(open("results/runs.csv"))) if Path("results/runs.csv").exists() else []
print(len(rows), "runs logged in this session")
groups = {}
for r in rows:
    groups.setdefault((r["model"], r["split"]), []).append(r)
for (m, split), rs in groups.items():
    vals = {k: [float(r[k]) for r in rs] for k in ("bleu1", "meteor", "rougeL", "em", "f1")}
    print(f"{m:26s} {split} n={len(rs)} " + " ".join(
        f"{k}={statistics.mean(v):.2f}" + (f"±{statistics.stdev(v):.2f}" if len(v) > 1 else "") for k, v in vals.items()))
'''

ZIP = '''# ---------------- pack results + checkpoints into one zip ----------------
out = Path("/kaggle/working") if ON_KAGGLE else Path("/content") if IN_COLAB else WORK.parent
zip_base = out / "a2_gpu_results"
import zipfile
# keep only the checkpoints the analysis and the final test evaluation need (full model, A3, A5, B2);
# every run's logs, per-epoch history and validation predictions are kept
KEEP_CKPT = ("cand_lr5e-4_ft-", "A3_no_selfattn-", "A5_single_span-", "B2_vanilla-")
with zipfile.ZipFile(f"{zip_base}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in (WORK / "results").rglob("*"):
        if p.is_file():
            z.write(p, Path("gpu") / p.relative_to(WORK))
    for p in (WORK / "checkpoints").glob("*.pt"):
        if p.name.startswith(KEEP_CKPT):
            z.write(p, Path("gpu") / p.relative_to(WORK), compress_type=zipfile.ZIP_STORED)
print(f"{zip_base}.zip", round(Path(f"{zip_base}.zip").stat().st_size / 1e6, 1), "MB")
if ON_KAGGLE:
    print("files in /kaggle/working (this is what Kaggle saves):",
          [(q.name, round(q.stat().st_size / 1e6, 1)) for q in Path("/kaggle/working").iterdir()])
if IN_COLAB:
    from google.colab import files
    files.download(f"{zip_base}.zip")
'''


def build():
    cells = [nbformat.v4.new_markdown_cell(INTRO), nbformat.v4.new_code_cell(SETUP), nbformat.v4.new_code_cell(DATA),
             nbformat.v4.new_markdown_cell("## Project code (written to files; identical to `A2/src`)")]
    for f in FILES:
        body = (ROOT / f).read_text(encoding="utf-8")
        assert body.strip(), f"{f} is empty: %%writefile needs a body"
        cells.append(nbformat.v4.new_code_cell(f"%%writefile {f}\n{body}"))
    frozen_json = json.dumps(dict(FROZEN), indent=1, sort_keys=True)
    cells.append(nbformat.v4.new_code_cell(
        f"Path('results/FROZEN_CONFIG.json').write_text('''{frozen_json}''')\n"
        f"MAIN_JOBS = {MAIN_JOBS!r}\nSTAGE34_JOBS = {STAGE34_JOBS!r}\n"
        "print(len(MAIN_JOBS), 'main jobs;', len(STAGE34_JOBS), 'optional stage-3/4 jobs')"))
    cells += [nbformat.v4.new_markdown_cell("## Run"), nbformat.v4.new_code_cell(RUN_HELPER),
              nbformat.v4.new_code_cell(main_cell("MAIN_JOBS", "gpu_main", "RUN_MAIN")),
              nbformat.v4.new_code_cell(main_cell("STAGE34_JOBS", "gpu_stage34", "RUN_STAGE34")),
              nbformat.v4.new_code_cell(BASELINES), nbformat.v4.new_code_cell(SUMMARY), nbformat.v4.new_code_cell(ZIP)]
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
    out = ROOT / "gpu_runner.ipynb"
    nbformat.write(nb, out)
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    build()
