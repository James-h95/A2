"""Append-only experiment log: every reported number must come from a row in results/runs.csv."""
import csv
import json
import os
import time
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("A2_RESULTS", Path(__file__).resolve().parents[1] / "results"
                                  if "__file__" in globals() else "results"))
RUNS_CSV = RESULTS_DIR / "runs.csv"
FIELDS = ["run_id", "timestamp", "model", "config", "seed", "split",
          "bleu1", "meteor", "rougeL", "em", "f1", "train_time_s", "notes"]


def log_run(model, config, seed, split, scores, train_time_s=0.0, run_id=None, notes=""):
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RUNS_CSV.exists()
    run_id = run_id or f"{model}-{split}-s{seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    row = dict(run_id=run_id, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), model=model,
               config=json.dumps(config, sort_keys=True, default=str), seed=seed, split=split,
               train_time_s=round(train_time_s, 1), notes=notes,
               **{m: round(scores[m], 4) for m in ("bleu1", "meteor", "rougeL", "em", "f1")})
    with open(RUNS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return run_id


def read_runs():
    if not RUNS_CSV.exists():
        return []
    with open(RUNS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))
