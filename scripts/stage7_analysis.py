"""Stage 7: qualitative and error analysis on the validation split (frozen model, seed 1)."""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as A  # noqa: E402
from src import data as D  # noqa: E402
from src import train as T  # noqa: E402
from src.evaluate import evaluate, score_example  # noqa: E402
from src.runlog import read_runs  # noqa: E402

# A fresh Colab workspace does not contain these output directories yet.
# Create them before saving figures or the question-type LaTeX table.
A.FIG_DIR.mkdir(parents=True, exist_ok=True)
A.TAB_DIR.mkdir(parents=True, exist_ok=True)


def ckpt_for(model_name, seed=1):
    rs = [r for r in read_runs() if r["model"] == model_name and r["seed"] == str(seed) and r["split"] == "val"]
    return ROOT / "checkpoints" / f"{rs[-1]['run_id']}.pt", rs[-1]["run_id"]


device = T.get_device()
full_ck, full_id = ckpt_for("A5_single_span", 1)
noa3_ck, noa3_id = ckpt_for("S_A3_no_selfattn", 1)
model, cfg, wv, cv = T.load_checkpoint(full_ck, device)
model_off, cfg_off, wv_off, cv_off = T.load_checkpoint(noa3_ck, device)
assert wv.itos == wv_off.itos, "A3 model must share the vocabulary (same preprocessing and seed)"
_, val_rows = D.get_splits()
va = D.encode(D.preprocess(val_rows, dict(use_normaliser=cfg["use_normaliser"], max_span_len=cfg["max_span_len"],
                                          tau=cfg["tau"])), wv, cv)
preds, spans = T.predict(model, va, device, max_span_len=cfg["max_span_len"])
scores, per = evaluate(preds, [e["answers"] for e in va], return_per_example=True)
print("full model s1 val:", {k: round(v, 2) for k, v in scores.items()}, full_id)
pred_file = ROOT / "results" / "preds" / f"{full_id}.json"

# ---------------------------------------------------------------- pick examples (deterministic rules)
def short(e):
    return len(e["t_tokens"]) <= 32 and len(e["q_tokens"]) <= 12


def qtok_missing_but_similar(e):
    """A question content word that is not in the tweet but shares a 4-char prefix with a tweet token
    (a slang, misspelt or inflected keyword)."""
    tw = {t["word"] for t in e["t_tokens"]}
    for t in e["q_tokens"]:
        w = t["word"]
        if len(w) >= 4 and w not in D._STOP and w not in tw and any(x[:4] == w[:4] and x != w for x in tw):
            return True
    return False


rows = list(zip(va, preds, spans, per))
picked = {}
succ = [(e, p, s, sc) for e, p, s, sc in rows if sc["em"] == 1 and short(e)]
for qw in ("who", "what", "where"):
    c = [x for x in succ if A.D._question_word(x[0]["question"]) == qw]
    if c:
        picked[f"success_{qw}"] = c[0]
fail = [(e, p, s, sc) for e, p, s, sc in rows if sc["rougeL"] < 0.5 and short(e)]
rules = {
    "fail_username": lambda e: e["best"] is not None and (e["best"][0] >= e["byline_from"] or any(
        t["kind"] == "mention" for t in e["t_tokens"][e["best"][0]:e["best"][1] + 1])),
    "fail_abstractive": lambda e: not e["trainable"],
    "fail_slang_misspelt": qtok_missing_but_similar,
}
for name, rule in rules.items():
    c = [x for x in fail if rule(x[0]) and x[0]["row_id"] not in {v[0]["row_id"] for v in picked.values()}]
    if c:
        picked[name] = c[0]

for name, (e, p, s, sc) in picked.items():
    A.attention_figure(model, e, device, A.FIG_DIR / f"attn_{name}", max_span_len=cfg["max_span_len"])
    print(f"[{name}] Q: {e['question']} | gold: {e['answers']} | pred: {p} | rougeL={100*sc['rougeL']:.0f}")

# self-attention with vs without A3: an example whose answer is far from the question words or in the byline,
# and which the full model answers correctly
far = [x for x in rows if x[3]["em"] == 1 and A._answer_far_or_byline(x[0]) and short(x[0])]
if far:
    e = far[0][0]
    A.self_attention_figure(model, model_off, e, device, A.FIG_DIR / "self_attn_A3")
    print(f"[self_attn_A3] Q: {e['question']} | gold: {e['answers']}")

# ---------------------------------------------------------------- training curves for the frozen seeds
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
for r in [r for r in read_runs() if r["model"] == "A5_single_span"]:
    h = json.loads((ROOT / "results" / "history" / f"{r['run_id']}.json").read_text())["history"]
    ax[0].plot([x["epoch"] for x in h], [x["train_loss"] for x in h], label=f"seed {r['seed']}")
    ax[1].plot([x["epoch"] for x in h], [x["rougeL"] for x in h], label=f"seed {r['seed']}")
ax[0].set_title("training loss (single silver span)"); ax[1].set_title("validation ROUGE-L")
for a in ax:
    a.set_xlabel("epoch"); a.legend(fontsize=7)
for ext in ("pdf", "png"):
    fig.savefig(A.FIG_DIR / f"training_curves.{ext}", bbox_inches="tight", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- per-question-type table
by = collections.defaultdict(list)
for e, sc in zip(va, per):
    by[A.D._question_word(e["question"])].append(sc)
qt = {k: dict(n=len(v), **{m: 100 * sum(s[m] for s in v) / len(v) for m in ("rougeL", "f1", "em")})
      for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}
for k, v in qt.items():
    print(f"qtype {k:6s} n={v['n']:4d} rougeL={v['rougeL']:.1f} f1={v['f1']:.1f} em={v['em']:.1f}")
lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lrccc}", r"\toprule",
         r"Question word & $n$ & ROUGE-L & F1 & EM \\", r"\midrule"]
lines += [f"{k} & {v['n']} & {v['rougeL']:.1f} & {v['f1']:.1f} & {v['em']:.1f} \\\\" for k, v in qt.items()]
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Validation scores of the full model (seed 1) by question word.}", r"\label{tab:qtype}",
          r"\end{table}"]
(A.TAB_DIR / "qtype.tex").write_text("\n".join(lines))

# ---------------------------------------------------------------- error sample (+ automatic pre-tags)
n_err = A.export_error_sample(va, pred_file, ROOT / "results" / "error_sample.csv")
with open(ROOT / "results" / "error_sample.csv", encoding="utf-8") as f:
    err = list(csv.DictReader(f))
by_id = {e["row_id"]: e for e in va}
pred_by_id = {e["row_id"]: (p, s) for e, p, s in zip(va, preds, spans)}
auto = collections.Counter()
for r in err:
    e = by_id[int(r["row_id"])]
    p, (i, j) = pred_by_id[e["row_id"]]
    f1 = score_example(p, e["answers"], ("f1",))["f1"]
    if not e["trainable"]:
        tag = "abstractive gold (no good span)"
    elif 0 < f1:
        tag = "boundary / partial overlap"
    elif e["best"] and (e["best"][0] >= e["byline_from"]) != (i >= e["byline_from"]):
        tag = "byline vs body confusion"
    else:
        tag = "wrong span (no overlap)"
    r["auto_tag"] = tag
    auto[tag] += 1
with open(ROOT / "results" / "error_sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(err[0].keys()))
    w.writeheader()
    w.writerows(err)
print(f"errors (EM=0) on val: {n_err}/{len(va)}; automatic pre-tags of the 100-sample:", dict(auto))
(ROOT / "results" / "stage7_summary.json").write_text(json.dumps(dict(
    full_run=full_id, a3_run=noa3_id, val_scores=scores, picked={k: v[0]["row_id"] for k, v in picked.items()},
    qtype=qt, n_errors=n_err, auto_tags=auto), indent=1))
