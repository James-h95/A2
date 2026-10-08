"""Stage 2: B1 lexical baseline. Parameters are chosen on train, results reported on val."""
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src.baselines import b1_lexical as B1  # noqa: E402
from src.evaluate import evaluate, fmt  # noqa: E402
from src.runlog import log_run  # noqa: E402

train_rows, val_rows = D.get_splits()
tr, va = D.preprocess(train_rows), D.preprocess(val_rows)

t0 = time.time()
grid = []
for max_len, lp, tb in itertools.product([1, 2, 3, 4, 5], [0.0, 0.1, 0.3], [0.0, 1.0]):
    s = evaluate(B1.b1_predict(tr, max_len=max_len, length_penalty=lp, type_bonus=tb),
                 [e["answers"] for e in tr], metrics=("rougeL",))
    grid.append((s["rougeL"], dict(max_len=max_len, length_penalty=lp, type_bonus=tb)))
    print(f"train grid {grid[-1][1]} rougeL={s['rougeL']:.2f}")
grid.sort(key=lambda x: -x[0])
cfg = grid[0][1]
print("chosen on train:", cfg)
for name, exs in [("train", tr), ("val", va)]:
    s = evaluate(B1.b1_predict(exs, **cfg), [e["answers"] for e in exs])
    print(f"B1 {name}: {fmt(s)}")
    log_run("B1_lexical", cfg, 0, name, s, time.time() - t0, notes="params chosen on train ROUGE-L")
cfg_nt = {**cfg, "type_bonus": 0.0}
s = evaluate(B1.b1_predict(va, **cfg_nt), [e["answers"] for e in va])
print(f"B1 val without type rules: {fmt(s)}")
log_run("B1_lexical_notype", cfg_nt, 0, "val", s, notes="B1 with answer-type bonus disabled")
