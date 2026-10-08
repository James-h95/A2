"""Stage 4: apply the adoption rule to the one-factor sweep.

The base config has 3 seeds (main_v1 s1, sw_base s2/s3). A one-factor variant
(seed 1) is adopted only if its val ROUGE-L beats the *maximum* of the base
seed range. Usage: python scripts/stage4_select.py [--final <name>]
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.runlog import read_runs  # noqa: E402

runs = [r for r in read_runs() if r["split"] == "val"]
base = [r for r in runs if r["model"] in ("main_v1", "sw_base")]
br = [float(r["rougeL"]) for r in base]
lo, hi, mu = min(br), max(br), statistics.mean(br)
print(f"base seeds {[r['seed'] for r in base]}: rougeL {br} -> range [{lo:.2f}, {hi:.2f}] mean {mu:.2f} "
      f"sd {statistics.stdev(br):.2f}")
adopted = {}
for r in runs:
    if not r["model"].startswith("sw_") or r["model"] == "sw_base":
        continue
    v = float(r["rougeL"])
    cfg, base_cfg = json.loads(r["config"]), json.loads(base[0]["config"])
    diff = {k: cfg[k] for k in cfg if cfg.get(k) != base_cfg.get(k)}
    verdict = "ADOPT" if v > hi else ("within range" if v >= lo else "worse")
    print(f"{r['model']:14s} {json.dumps(diff):32s} rougeL={v:.2f} f1={float(r['f1']):.2f} "
          f"best_ep={r['notes'].split()[0]} -> {verdict}")
    if verdict == "ADOPT":
        adopted[r["model"]] = (v, diff)
print("adopted:", adopted or "none")
