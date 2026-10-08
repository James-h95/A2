"""Markdown and glue code cells for the submission notebook (build_notebook.py).
Module cells ({"type": "module"}) are inlined from src/ at build time."""

README = r"""
**Task.** TweetQA (Xiong et al., 2019): answer a question about a tweet. The gold answers are short and mostly extractive (about 65% occur in the tweet after normalisation). Metrics: BLEU-1, METEOR, ROUGE-L (the official TweetQA metrics), plus EM and token-F1.

**Main model (trained from scratch): TweetBiDAF.** Word emb ⊕ char-CNN over raw tokens ⊕ tweet token flags (incl. a byline flag) → highway → shared BiLSTM → trilinear bi-attention (C2Q + Q2C) → gated multi-head self-attention → modelling BiLSTMs → span start/end. It is trained with *soft silver-span* targets, because TweetQA answers are free text, not spans. The word vectors are initialised from a FastText model **trained from scratch on the TweetQA training split only**. No pretrained weights and no external data are used in the main model.

**Baselines** (clearly separate from the main model):
- B1: lexical sliding window (no learning)
- B2: the main model with every task-specific component switched off
- B3: fine-tuned `vinai/bertweet-base` span extractor
- B4: fine-tuned `t5-small` generator
- B5: few-shot `Qwen2.5-1.5B-Instruct` (no fine-tuning)

**Data and splits.** `train.csv` (10,692) and `test.csv` (1,086, which is the original TweetQA dev set and has gold answers) from the unit's shared folder. Validation is 10% of `train.csv` (1,076 questions), **grouped by tweet**, stratified by question word, seed 4012. The 46 exact-duplicate rows are dropped. **The test set is locked (`TEST_LOCKED`) and was used once, for the final table (Section 3.3).** All tuning, ablations and analysis use validation only.

**How to run (clean Colab runtime, GPU recommended).**
1. Run all cells top to bottom. The Setup cell installs `gensim nltk transformers sentencepiece emoji huggingface_hub gdown`.
2. Data: the notebook tries to download `TweetQA/train.csv` and `TweetQA/test.csv` from the unit's shared Drive folder with `gdown`. If that fails, upload both files to `./data/` (or set the `TWEETQA_DIR` environment variable).
3. `TRAIN = False` (default) downloads our checkpoints, per-epoch histories, logs and `runs.csv` from the public Hugging Face repo **`{HF_REPO}`**, then re-evaluates the saved models. `TRAIN = True` retrains everything: about 2 h on a T4 for the main model, sweep and ablations.
4. `RUN_BASELINES = False` shows the logged B3–B5 runs. `True` re-runs them (GPU, about 1 h).

**Reproducibility.** Every reported number is a row in `results/runs.csv`. Seeds are fixed (python, numpy, torch, cuda). The saved outputs below come from the run used for the report.

**Note on code layout.** The template asks for OOP code at the bottom. Our model classes have to be defined before the cells that train and evaluate them, so a top-to-bottom run needs them where they are. They are grouped in Section 2 and clearly labelled.
"""

SETUP = r'''
# ---------------- configuration ----------------
TRAIN = False          # True: retrain every model from scratch. False: load our checkpoints + logs from Hugging Face
RUN_BASELINES = False  # True: re-run the pretrained baselines B3-B5 (GPU, ~1 h). False: show their logged results
HF_REPO = "{HF_REPO}"  # public repo with checkpoints/, results/ (runs.csv, histories, logs, predictions)

import os, sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gensim", "nltk", "transformers>=4.40",
                    "sentencepiece", "emoji", "huggingface_hub", "gdown"], check=True)

import ast, csv, json, math, random, re, shutil, statistics, string, time, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
print("torch", torch.__version__, "| device:",
      "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
'''

DATA_DL = r'''
# ---------------- data: TweetQA train.csv / test.csv into ./data ----------------
DATA_FOLDER_URL = "https://drive.google.com/drive/folders/1j5lP1gBrewzpFSH5Ya1SQ6echsiJaI3j"
Path("data").mkdir(exist_ok=True)
if not (Path("data/train.csv").exists() and Path("data/test.csv").exists()):
    try:
        import gdown
        gdown.download_folder(DATA_FOLDER_URL, output="_shared_datasets", quiet=True, remaining_ok=True)
        for p in Path("_shared_datasets").rglob("*.csv"):
            if "tweetqa" in str(p.parent).lower() and p.name in ("train.csv", "test.csv"):
                shutil.copy(p, Path("data") / p.name)
    except Exception as e:  # noqa: BLE001
        print("automatic download failed:", e)
assert Path("data/train.csv").exists() and Path("data/test.csv").exists(), \
    "Upload TweetQA train.csv and test.csv to ./data (see Readme)"
print({p.name: sum(1 for _ in open(p, encoding="utf-8")) - 1 for p in sorted(Path("data").glob("*.csv"))}, "rows (incl. multi-line)")
'''

ARTIFACTS = r'''
# ---------------- trained artifacts (only when not retraining) ----------------
if not TRAIN and not Path("results/runs.csv").exists():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_REPO, local_dir=".", allow_patterns=["checkpoints/*", "results/**"])
print("runs logged:", sum(1 for _ in open("results/runs.csv")) - 1 if Path("results/runs.csv").exists() else 0)
'''

ALIASES = r'''
# The code above was developed as separate modules (data, analysis, ...). Inlined in one notebook,
# the module-qualified names (D.get_splits, A.slices) resolve to this notebook's namespace.
D = A = sys.modules[__name__]
'''

STATS = r'''
# ---------------- dataset statistics (training split only; test stays locked) ----------------
train_rows, val_rows = get_splits()
print(f"train: {len(train_rows)} questions / {len({r['tweet'] for r in train_rows})} tweets | "
      f"val: {len(val_rows)} / {len({r['tweet'] for r in val_rows})} | tweets shared: "
      f"{len({r['tweet'] for r in train_rows} & {r['tweet'] for r in val_rows})}")
def pct(xs, ps=(50, 90, 95, 99, 100)):
    xs = sorted(xs); return {p: xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))] for p in ps}
print("tweet tokens", pct([len(tokenize(r["tweet"])) for r in {r["tweet"]: r for r in train_rows}.values()]))
print("question tokens", pct([len(tokenize(r["question"])) for r in train_rows]))
print("answer tokens", pct([len(tokenize(a)) for r in train_rows for a in r["answers"]]))
sub = sum(any(normalize_answer(a) and f" {normalize_answer(a)} " in f" {normalize_answer(r['tweet'])} "
              for a in r["answers"]) for r in train_rows)
print(f"answer occurs in tweet after normalisation: {100 * sub / len(train_rows):.1f}%")
print("question words:", Counter(_question_word(r["question"]) for r in train_rows).most_common())
for r in train_rows[:3]:
    print(f"\nQ: {r['question']}\nA: {r['answers']}\nT: {r['tweet']}")
'''

ORACLE = r'''
# ---------------- silver spans and the extractive oracle (ceiling for every span model) ----------------
tr_ex, va_ex = preprocess(train_rows), preprocess(val_rows)
for name, exs in (("train", tr_ex), ("val", va_ex)):
    s = evaluate(oracle_predictions(exs), [e["answers"] for e in exs])
    print(f"oracle {name}: {fmt(s)} | span with F1>=tau: {100 * sum(e['trainable'] for e in exs) / len(exs):.1f}%")
ex = next(e for e in va_ex if e["trainable"] and len(e["soft"]) > 3)
print("\nexample:", ex["question"], "| gold:", ex["answers"])
print("best span:", span_text(ex, *ex["best"]), "| soft set (top 5):",
      [(span_text(ex, i, j), round(w, 3)) for i, j, w in sorted(ex["soft"], key=lambda x: -x[2])[:5]])
wv_demo, cv_demo = build_vocabs(tr_ex)
print(f"vocab (train only, min_freq 2): {len(wv_demo)} words, {len(cv_demo)} chars")
'''

MODEL_SUMMARY = r'''
FROZEN = json.loads(Path("results/FROZEN_CONFIG.json").read_text())
FROZEN_CFG = {k: v for k, v in FROZEN.items() if not k.startswith("_")}
print("frozen config:", json.dumps(FROZEN_CFG, sort_keys=True))
_m = TweetBiDAF(len(wv_demo), len(cv_demo), {k: FROZEN_CFG[k] for k in DEFAULT_MODEL})
print(_m)
print(f"trainable parameters: {sum(p.numel() for p in _m.parameters() if p.requires_grad):,}")
'''

TRAIN_ALL = r'''
# ---------------- training (TRAIN=True retrains; logs are written to results/) ----------------
BASE = {**DEFAULT_TRAIN, **DEFAULT_MODEL}
SWEEP = {"sw_lr5e-4": {"lr": 5e-4}, "sw_h96": {"hidden": 96}, "sw_h192": {"hidden": 192}, "sw_do0.3": {"dropout": 0.3},
         "sw_L10": {"max_span_len": 10}, "sw_L20": {"max_span_len": 20}, "sw_fasttext": {"word_init": "fasttext"}}
ABLATIONS = {
    "A1_no_char": {"use_char_cnn": False},
    "A2_no_norm_flags": {"use_normaliser": False, "use_token_flags": False},
    "A3_no_selfattn": {"use_self_attn": False},
    "A4_c2q_only": {"biattn_mode": "c2q"},
    "A5_single_span": {"soft_spans": False},
    "B2_vanilla": {"use_char_cnn": False, "use_normaliser": False, "use_token_flags": False,
                   "use_self_attn": False, "biattn_mode": "c2q", "soft_spans": False},
}
if TRAIN:
    train(BASE, seed=1, name="main_v1")                                   # Stage 3
    for s in (2, 3):
        train(BASE, seed=s, name="sw_base")                               # Stage 4: base seed range
    for name, over in SWEEP.items():
        train({**BASE, **over}, seed=1, name=name)                        # Stage 4: one-factor sweep
    for s in (2, 3):
        train({**BASE, "lr": 5e-4}, seed=s, name="sw_lr5e-4")
    for s in (1, 2, 3):
        train(FROZEN_CFG, seed=s, name="cand_lr5e-4_ft")                  # frozen config, 3 seeds
    for name, over in ABLATIONS.items():                                  # Stage 5
        for s in (1, 2, 3):
            train({**FROZEN_CFG, **over}, seed=s, name=name)
else:
    print("TRAIN=False: using the logged runs and checkpoints downloaded from", HF_REPO)
'''

TRAIN_LOGS = r'''
# ---------------- saved training logs: every epoch of every reported main-model run ----------------
MAIN_RUNS = ["main_v1", "sw_base", *SWEEP, "cand_lr5e-4_ft", *ABLATIONS]
for r in read_runs():
    if r["model"] in MAIN_RUNS and r["split"] == "val":
        h = json.loads(Path(f"results/history/{r['run_id']}.json").read_text())
        print(f"== {r['run_id']} (best epoch {h['best_epoch']}, {float(r['train_time_s'])/60:.1f} min)")
        for x in h["history"]:
            print(f"   epoch {x['epoch']:2d} loss={x['train_loss']:.4f} val " +
                  " ".join(f"{m}={x[m]:.2f}" for m in METRICS))
'''

B1_RUN = r'''
# ---------------- B1 lexical baseline (no learning; 3 parameters chosen on the training split) ----------------
import itertools
grid = []
for ml, lp, tb in itertools.product([1, 2, 3, 4, 5], [0.0, 0.1, 0.3], [0.0, 1.0]):
    s = evaluate(b1_predict(tr_ex, max_len=ml, length_penalty=lp, type_bonus=tb), [e["answers"] for e in tr_ex], metrics=("rougeL",))
    grid.append((s["rougeL"], dict(max_len=ml, length_penalty=lp, type_bonus=tb)))
B1_CFG = max(grid, key=lambda x: x[0])[1]
print("B1 parameters chosen on train:", B1_CFG)
print("B1 val:", fmt(evaluate(b1_predict(va_ex, **B1_CFG), [e["answers"] for e in va_ex])))
'''

BASELINE_LOGS = r'''
# ---------------- B3-B5 (pretrained baselines) ----------------
if RUN_BASELINES:
    s, *_ = b3_run(dict(B3_CFG), seed=1, split="val"); print("B3 val", fmt(s))
    s, *_ = b4_run(dict(B4_CFG), seed=1, split="val"); print("B4 val", fmt(s))
    s, *_ = b5_run(dict(B5_CFG), split="val"); print("B5 val", fmt(s))
else:
    for f in sorted(Path("results/logs").glob("stage6_*.log")):
        print(f"==== {f.name}")
        print("\n".join(l for l in f.read_text().splitlines() if not l.startswith(("Loading", " ", "{", "}"))
                        or "epoch" in l)[-3000:])
'''

SELECTION = r'''
# ---------------- 3.1 model selection on validation (Stage 4 adoption rule) ----------------
runs = [r for r in read_runs() if r["split"] == "val"]
def grp(name): return [r for r in runs if r["model"] == name]
base = grp("main_v1") + grp("sw_base")
lo, hi = min(float(r["rougeL"]) for r in base), max(float(r["rougeL"]) for r in base)
print(f"base (3 seeds) ROUGE-L range [{lo:.2f}, {hi:.2f}]")
for name in SWEEP:
    r = grp(name)[0]
    v = float(r["rougeL"])
    print(f"  {name:12s} rougeL={v:.2f} -> {'ADOPT' if v > hi else 'within range' if v >= lo else 'worse'}")
for name in ("main_v1+sw_base", "sw_lr5e-4", "cand_lr5e-4_ft"):
    rs = base if name == "main_v1+sw_base" else grp(name)
    print(f"{name:16s} n={len(rs)} " + " ".join(f"{m}={statistics.mean(float(r[m]) for r in rs):.2f}"
          f"±{statistics.stdev(float(r[m]) for r in rs):.2f}" for m in METRICS))
'''

ABLATION_TABLE = r'''
# ---------------- 3.2 ablations (validation, 3 seeds each) + hypothesis slices ----------------
ABL_ROWS = [("cand_lr5e-4_ft", "Full model"), *[(k, k) for k in ABLATIONS]]
sl = slices(va_ex, wv_demo)
print("slice sizes:", {k: len(v) for k, v in sl.items()})
for name, label in ABL_ROWS:
    rs = grp(name)
    ms = {m: mean_sd([float(r[m]) for r in rs]) for m in METRICS}
    per_slice = defaultdict(list)
    for r in rs:
        for k, v in slice_scores(f"results/preds/{r['run_id']}.json", va_ex, sl).items():
            per_slice[k].append(v)
    print(f"{label:18s} " + " ".join(f"{m}={ms[m][0]:.2f}±{ms[m][1]:.2f}" for m in METRICS))
    print(" " * 19 + "slice ROUGE-L: " + " ".join(f"{k}={statistics.mean(v):.1f}" for k, v in per_slice.items()))
'''

TEST_EVAL = r'''
# ---------------- 3.3 final test-set evaluation (run once; numbers match results/runs.csv) ----------------
TEST_LOCKED = False                      # unlocked only here, for the final table
test_rows = load_test()
device = get_device()
golds = [r["answers"] for r in test_rows]
final = {}
for name in ("cand_lr5e-4_ft", "B2_vanilla"):
    per_seed = []
    for r in sorted(grp(name), key=lambda r: r["seed"]):
        model, cfg, wv, cv = load_checkpoint(f"checkpoints/{r['run_id']}.pt", device)
        te = encode(preprocess(test_rows, dict(use_normaliser=cfg["use_normaliser"], max_span_len=cfg["max_span_len"],
                                               tau=cfg["tau"])), wv, cv)
        preds, _ = predict(model, te, device, max_span_len=cfg["max_span_len"])
        per_seed.append(evaluate(preds, golds))
        print(f"{name} seed {r['seed']} test: {fmt(per_seed[-1])}")
    final[name] = {m: mean_sd([s[m] for s in per_seed]) for m in METRICS}
te_b1 = preprocess(test_rows)
final["B1_lexical"] = evaluate(b1_predict(te_b1, **B1_CFG), golds)
final["oracle"] = evaluate(oracle_predictions(te_b1), golds)
for r in read_runs():                                        # pretrained baselines: logged test runs
    if r["split"] == "test" and r["model"].startswith(("B3", "B4", "B5")):
        final[r["model"]] = {m: float(r[m]) for m in METRICS}
print("\nFINAL TEST TABLE")
for k, v in final.items():
    print(f"{k:28s} " + " ".join(f"{m}={v[m][0]:.2f}±{v[m][1]:.2f}" if isinstance(v[m], tuple) else f"{m}={v[m]:.2f}"
                                  for m in METRICS))
TEST_LOCKED = True
'''

FIGURES = r'''
# ---------------- 3.4 attention analysis (validation examples, full model seed 1) ----------------
from IPython.display import Image, display
summary = json.loads(Path("results/stage7_summary.json").read_text())
full_run = summary["full_run"]
model, cfg, wv, cv = load_checkpoint(f"checkpoints/{full_run}.pt", get_device())
va_enc = encode(preprocess(val_rows, dict(use_normaliser=cfg["use_normaliser"], max_span_len=cfg["max_span_len"],
                                          tau=cfg["tau"])), wv, cv)
by_row = {e["row_id"]: e for e in va_enc}
for name, rid in summary["picked"].items():
    attention_figure(model, by_row[rid], get_device(), FIG_DIR / f"attn_{name}", max_span_len=cfg["max_span_len"])
    print(name); display(Image(filename=str(FIG_DIR / f"attn_{name}.png"), width=900))
for f in ("self_attn_A3", "training_curves"):
    if (FIG_DIR / f"{f}.png").exists():
        print(f); display(Image(filename=str(FIG_DIR / f"{f}.png"), width=900))
'''

ERRORS = r'''
# ---------------- 3.5 error analysis: 100 random validation errors of the full model ----------------
err = list(csv.DictReader(open("results/error_sample.csv", encoding="utf-8")))
print("automatic pre-tags:", Counter(r["auto_tag"] for r in err).most_common())
if any(r.get("error_cause") for r in err):
    print("manual error causes:", Counter(r["error_cause"] for r in err if r["error_cause"]).most_common())
print("by question type:", Counter(r["question_type"] for r in err).most_common())
for r in err[:10]:
    print(f"- Q: {r['question']} | gold: {r['gold']} | pred: {r['pred']} | {r['auto_tag']}")
'''


def M(path, note=""):
    return {"type": "module", "path": path, "note": note}


def C(text):
    return {"type": "code", "text": text}


def T(text):
    return {"type": "md", "text": text}


PARTS = {
    "s1": [
        T("## 1.0 Setup, data and trained artifacts"), C(SETUP), C(DATA_DL), C(ARTIFACTS),
        T("## 1.1 Evaluation metrics\nBLEU-1, METEOR and ROUGE-L are the official TweetQA metrics. We add EM and token-F1. Each metric is the max over gold answers, and the corpus score is the mean ×100."),
        M("evaluate.py"),
        T("## 1.2 Loading, splits, tokenisation, tweet normaliser, token flags, silver spans, vocabulary, batching"),
        M("data.py"), M("runlog.py", "experiment log: every reported number is a row of results/runs.csv"), C(ALIASES),
        T("## 1.3 Dataset statistics (training split)"), C(STATS),
        T("## 1.4 Silver spans and the extractive oracle"), C(ORACLE),
    ],
    "s2": [
        T("## 2.1 Main model: TweetBiDAF (trained from scratch)"), M("model.py"), C(MODEL_SUMMARY),
        T("## 2.2 Training procedure"), M("fasttext_init.py", "in-domain word vectors trained only on the training split"),
        M("train.py"),
        T("## 2.3 Baselines (separate from the main model)"), M("baselines/b1_lexical.py"),
        M("baselines/b3_extractive_ft.py"), M("baselines/b4_t5_ft.py"), M("baselines/b5_llm.py"),
        T("## 2.4 Training runs and saved training logs"), C(TRAIN_ALL), C(TRAIN_LOGS),
        T("## 2.5 Baseline runs"), C(B1_RUN), C(BASELINE_LOGS),
    ],
    "s3": [
        M("analysis.py"),
        T("## 3.1 Model selection (validation)"), C(SELECTION),
        T("## 3.2 Ablation studies (validation)"), C(ABLATION_TABLE),
        T("## 3.3 Final comparison on the test set"), C(TEST_EVAL),
        T("## 3.4 Qualitative analysis of attention"), C(FIGURES),
        T("## 3.5 Error analysis"), C(ERRORS),
    ],
}
