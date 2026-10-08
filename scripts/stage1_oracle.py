"""Stage 1: split sizes, silver-span statistics and the extractive oracle ceiling."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data as D  # noqa: E402
from src.evaluate import evaluate, fmt  # noqa: E402
from src.runlog import log_run  # noqa: E402

train_rows, val_rows = D.get_splits()
print(f"split sizes: train={len(train_rows)} ({len({r['tweet'] for r in train_rows})} tweets) "
      f"val={len(val_rows)} ({len({r['tweet'] for r in val_rows})} tweets)")
(ROOT / "results" / "splits.json").write_text(json.dumps(
    dict(seed=D.SPLIT_SEED, val_frac=D.VAL_FRAC,
         train_row_ids=[r["row_id"] for r in train_rows], val_row_ids=[r["row_id"] for r in val_rows])))

summary = {}
for prep in [dict(max_span_len=15), dict(max_span_len=10), dict(max_span_len=20),
             dict(max_span_len=15, use_normaliser=False)]:
    t0 = time.time()
    tr = D.preprocess(train_rows, prep)
    va = D.preprocess(val_rows, prep)
    full = {**D.DEFAULT_PREP, **prep}
    for name, exs in [("train", tr), ("val", va)]:
        s = evaluate(D.oracle_predictions(exs), [e["answers"] for e in exs])
        n_tr = sum(e["trainable"] for e in exs)
        n_any = sum(e["best"] is not None for e in exs)
        mean_soft = sum(len(e["soft"]) for e in exs if e["trainable"]) / max(1, n_tr)
        line = (f"oracle {name:5s} {json.dumps(full)}: {fmt(s)} | span>0: {100*n_any/len(exs):.1f}% "
                f"| trainable (F1>=tau): {100*n_tr/len(exs):.1f}% | soft spans/ex: {mean_soft:.1f}")
        print(line)
        summary[f"{name}|{json.dumps(full)}"] = dict(scores=s, trainable=n_tr / len(exs), any=n_any / len(exs))
        log_run("oracle", full, 0, name, s, notes="best silver span vs gold (extractive ceiling)")
    print(f"  ({time.time()-t0:.1f}s)")

# vocab / OOV statistics for the default preprocessing
tr = D.preprocess(train_rows)
va = D.preprocess(val_rows)
wv, cv = D.build_vocabs(tr)
oov_tok = sum(wv(t["word"]) == 1 for e in va for t in e["t_tokens"]) / sum(len(e["t_tokens"]) for e in va)
ans_oov = 0
for e in va:
    if e["best"] is not None:
        i, j = e["best"]
        ans_oov += any(wv(t["word"]) == 1 for t in e["t_tokens"][i:j + 1])
print(f"vocab: words={len(wv)} chars={len(cv)} | val tweet-token OOV={100*oov_tok:.1f}% "
      f"| val examples whose best span has an OOV token={100*ans_oov/len(va):.1f}%")
(ROOT / "results" / "stage1_summary.json").write_text(json.dumps(
    dict(summary=summary, vocab_words=len(wv), vocab_chars=len(cv), val_tok_oov=oov_tok,
         val_span_oov=ans_oov / len(va)), indent=1))
