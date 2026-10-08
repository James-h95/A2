"""Quick validation diagnostics for a checkpoint: lengths, per-question-type and per-slice scores."""
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D, train as T
from src.evaluate import evaluate, norm_tokens
ck = sys.argv[1]
model, cfg, wv, cv = T.load_checkpoint(ck)
_, val_rows = D.get_splits()
va = D.encode(D.preprocess(val_rows, dict(use_normaliser=cfg["use_normaliser"], max_span_len=cfg["max_span_len"], tau=cfg["tau"])), wv, cv)
preds, spans = T.predict(model, va, T.get_device(), max_span_len=cfg["max_span_len"])
s, per = evaluate(preds, [e["answers"] for e in va], return_per_example=True)
print("overall", {k: round(v, 2) for k, v in s.items()})
pl = [len(norm_tokens(p)) for p in preds]; gl = [len(norm_tokens(e["answers"][0])) for e in va]
print(f"mean pred len {sum(pl)/len(pl):.2f} | gold {sum(gl)/len(gl):.2f} | oracle-best len {sum(j-i+1 for e in va if e['best'] for i,j in [e['best']])/sum(1 for e in va if e['best']):.2f}")
groups = collections.defaultdict(list)
for e, p in zip(va, per):
    groups["qw=" + D._question_word(e["question"])].append(p["rougeL"])
    groups["trainable" if e["trainable"] else "not-trainable"].append(p["rougeL"])
    if e["best"]:
        groups["gold-in-byline" if e["best"][0] >= e["byline_from"] else "gold-in-body"].append(p["rougeL"])
for k in sorted(groups):
    v = groups[k]; print(f"{k:18s} n={len(v):4d} rougeL={100*sum(v)/len(v):.1f}")
pred_in_by = sum(1 for e, (i, j) in zip(va, spans) if i >= e["byline_from"])
print(f"predicted span starts in byline: {100*pred_in_by/len(va):.1f}%  | gold best in byline: {100*sum(1 for e in va if e['best'] and e['best'][0]>=e['byline_from'])/len(va):.1f}%")
