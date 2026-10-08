"""Analysis utilities: hypothesis slices, ablation tables, attention heatmaps, error sampling."""
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch

try:
    from . import data as D
    from .evaluate import METRICS, evaluate, normalize_answer, score_example
    from .runlog import RESULTS_DIR, read_runs
except ImportError:
    import data as D
    from evaluate import METRICS, evaluate, normalize_answer, score_example
    from runlog import RESULTS_DIR, read_runs

FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
FAR_N = 8   # A3 slice: answer more than FAR_N tokens from the nearest question-word match


# ------------------------------------------------------------------ slices

def _answer_far_or_byline(ex):
    if ex["best"] is None:
        return False
    i, j = ex["best"]
    if i >= ex["byline_from"]:
        return True
    qwords = {w for t in ex["q_tokens"] for w in normalize_answer(t["raw"]).split()} - D._STOP
    anchors = [p for p, t in enumerate(ex["t_tokens"])
               if set(normalize_answer(t["raw"]).split()) & qwords and not (i <= p <= j)]
    if not anchors:
        return True
    return min(i - p if p < i else p - j for p in anchors) > FAR_N


def _answer_is_handle_or_tag(ex):
    if ex["best"] is None:
        return any(a.strip().startswith(("@", "#")) for a in ex["answers"])
    i, j = ex["best"]
    toks = ex["t_tokens"][i:j + 1]
    return (any(t["kind"] in ("mention", "hashtag") for t in toks) or i >= ex["byline_from"]
            or any(a.strip().startswith(("@", "#")) for a in ex["answers"]))


def slices(examples, wvocab):
    """row_id sets for the hypothesis-driven slices (validation examples)."""
    out = defaultdict(set)
    for ex in examples:
        rid = ex["row_id"]
        out["all"].add(rid)
        ans_words = {t["word"] for a in ex["answers"] for t in D.normalise_tokens(D.tokenize(a))}
        if any(wvocab(w) == 1 for w in ans_words if normalize_answer(w)):
            out["A1_answer_oov"].add(rid)
        if _answer_is_handle_or_tag(ex):
            out["A2_handle_tag_username"].add(rid)
        if _answer_far_or_byline(ex):
            out["A3_far_or_byline"].add(rid)
        if not any(a.lower().strip() in ex["tweet"].lower() for a in ex["answers"]):
            out["A5_not_substring"].add(rid)
    return dict(out)


def slice_scores(pred_file, examples, slice_sets, metric="rougeL"):
    preds = {p["row_id"]: p["pred"] for p in json.loads(Path(pred_file).read_text())}
    per = {ex["row_id"]: score_example(preds.get(ex["row_id"], ""), ex["answers"], (metric,))[metric]
           for ex in examples}
    return {name: 100 * sum(per[r] for r in ids) / max(1, len(ids)) for name, ids in slice_sets.items()}


# ------------------------------------------------------------------ tables

def mean_sd(xs):
    return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)


def runs_by_name(names, split="val"):
    groups = defaultdict(list)
    for r in read_runs():
        if r["model"] in names and r["split"] == split:
            groups[r["model"]].append(r)
    return groups


def latex_table(rows, caption, label, cols=METRICS, header=None):
    """rows: [(label, {metric: (mean, sd) or value})]"""
    header = header or ["BLEU-1", "METEOR", "ROUGE-L", "EM", "F1"]
    lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{l" + "c" * len(cols) + "}",
             r"\toprule", "Model & " + " & ".join(header) + r" \\", r"\midrule"]
    for name, vals in rows:
        if name == "---":
            lines.append(r"\midrule")
            continue
        cells = []
        for c in cols:
            v = vals.get(c)
            if v is None:
                cells.append("--")
            elif isinstance(v, tuple):
                cells.append(f"{v[0]:.1f}$\\pm${v[1]:.1f}")
            else:
                cells.append(f"{v:.1f}")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ figures

def _short(s, n=110):
    return s if len(s) <= n else s[:n - 1] + "…"


@torch.no_grad()
def attention_figure(model, ex, device, path, title_extra="", max_span_len=15):
    """Tweet x question similarity S and C2Q weights, plus Q2C and start/end probability strips."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from .model import decode
    except ImportError:
        from model import decode
    model.eval()
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in D.collate([ex]).items()}
    o = model(batch, return_attn=True)
    (i, j), = decode(o["start_logits"], o["end_logits"], max_span_len)[0].tolist()
    pred = D.span_text(ex, i, j)
    t_lab = [t["raw"] for t in ex["t_tokens"]]
    q_lab = [t["raw"] for t in ex["q_tokens"]]
    c2q = o["c2q"][0].cpu().numpy()
    ps = torch.softmax(o["start_logits"][0], -1).cpu().numpy()
    pe = torch.softmax(o["end_logits"][0], -1).cpu().numpy()
    q2c = o["q2c"][0].cpu().numpy() if o["q2c"] is not None else None
    n_strips = 3 if q2c is not None else 2
    T = len(t_lab)
    fig = plt.figure(figsize=(max(8, 0.32 * T + 2), 3.2 + 0.28 * len(q_lab) + 0.35 * n_strips))
    gs = fig.add_gridspec(1 + n_strips, 1, height_ratios=[max(2, 0.28 * len(q_lab))] + [0.35] * n_strips, hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.imshow(c2q.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(q_lab)), q_lab, fontsize=8)
    ax.set_xticks([])
    ax.set_ylabel("question (C2Q weights)", fontsize=8)
    strips = [("start", ps), ("end", pe)] + ([("Q2C", q2c)] if q2c is not None else [])
    for k, (name, v) in enumerate(strips):
        a = fig.add_subplot(gs[1 + k], sharex=ax)
        a.imshow(v[None, :T], aspect="auto", cmap="Oranges" if name != "Q2C" else "Greens", vmin=0)
        a.set_yticks([0], [name], fontsize=8)
        if k < len(strips) - 1:
            a.set_xticks([])
        else:
            a.set_xticks(range(T), t_lab, rotation=75, fontsize=7)
    for x in (i - 0.5, j + 0.5):
        ax.axvline(x, color="red", lw=1)
    gold = " | ".join(ex["answers"])
    ok = score_example(pred, ex["answers"], ("rougeL",))["rougeL"]
    fig.suptitle(f"Q: {_short(ex['question'])}\nTweet: {_short(ex['tweet'], 150)}\n"
                 f"gold: {gold}   pred: {pred}   (ROUGE-L {100*ok:.0f}) {title_extra}", fontsize=8, x=0.01, ha="left")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return pred


@torch.no_grad()
def self_attention_figure(model_on, model_off, ex, device, path):
    """Self-attention map of the full model, beside the start/end distributions with and without A3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in D.collate([ex]).items()}
    model_on.eval(); model_off.eval()
    on = model_on(batch, return_attn=True)
    off = model_off(batch, return_attn=True)
    t_lab = [t["raw"] for t in ex["t_tokens"]]
    T = len(t_lab)
    sa = on["self_attn"][0].cpu().numpy()[:T, :T]
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 0.5 * T), max(6, 0.3 * T)),
                             gridspec_kw=dict(width_ratios=[3, 1.2]))
    axes[0].imshow(sa, cmap="Purples")
    axes[0].set_xticks(range(T), t_lab, rotation=80, fontsize=7)
    axes[0].set_yticks(range(T), t_lab, fontsize=7)
    axes[0].set_title("gated self-attention (full model): row = query token", fontsize=9)
    for name, o, c in (("full: start", on, "C0"), ("full: end", on, "C1"), ("no self-attn: start", off, "C2"),
                       ("no self-attn: end", off, "C3")):
        key = "start_logits" if "start" in name else "end_logits"
        axes[1].plot(torch.softmax(o[key][0], -1).cpu().numpy()[:T], range(T), label=name, color=c,
                     ls="-" if "full" in name else "--")
    axes[1].invert_yaxis()
    axes[1].set_yticks(range(T), t_lab, fontsize=7)
    axes[1].legend(fontsize=7)
    axes[1].set_title("span probabilities", fontsize=9)
    fig.suptitle(f"Q: {ex['question']}   gold: {' | '.join(ex['answers'])}", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ errors

def export_error_sample(examples, pred_file, path, n=100, seed=4012):
    preds = {p["row_id"]: p["pred"] for p in json.loads(Path(pred_file).read_text())}
    errs = []
    for ex in examples:
        p = preds.get(ex["row_id"], "")
        s = score_example(p, ex["answers"], ("rougeL", "em"))
        if s["em"] < 1:
            errs.append((ex, p, s["rougeL"]))
    random.Random(seed).shuffle(errs)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "question", "tweet", "gold", "pred", "rougeL", "extractive_gold",
                     "question_type", "error_cause"])
        for ex, p, r in errs[:n]:
            w.writerow([ex["row_id"], ex["question"], ex["tweet"], " | ".join(ex["answers"]), p, f"{100*r:.0f}",
                        int(ex["trainable"]), D._question_word(ex["question"]), ""])
    return len(errs)
