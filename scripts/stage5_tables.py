"""Stage 5: ablation table (mean ± sd over 3 seeds) and hypothesis-slice breakdowns from logged runs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as A  # noqa: E402
from src import data as D  # noqa: E402
from src.evaluate import METRICS  # noqa: E402

FROZEN = json.loads((ROOT / "results" / "FROZEN_CONFIG.json").read_text())
ABL = [("A5_single_span", "Full model v2"), ("S_A1_no_char", "A1: $-$ char-CNN"),
       ("S_A2_no_norm_flags", "A2: $-$ normaliser \\& flags"),
       ("S_A3_no_selfattn", "A3: $-$ self-attention"),
       ("S_A4_c2q_only", "A4: C2Q only (no Q2C)"),
       ("cand_lr5e-4_ft", "A5: wide soft targets (v1)"),
       ("B2_vanilla", "B2: all task-specific parts off")]

groups = A.runs_by_name([n for n, _ in ABL])
_, val_rows = D.get_splits()
prep = dict(use_normaliser=True, max_span_len=FROZEN["max_span_len"], tau=FROZEN["tau"])
va = D.preprocess(val_rows, prep)
tr_rows, _ = D.get_splits()
wv, _ = D.build_vocabs(D.preprocess(tr_rows, prep), min_freq=FROZEN.get("min_freq", 2))
sl = A.slices(va, wv)
print({k: len(v) for k, v in sl.items()})

rows, slice_rows, summary = [], [], {}
for name, label in ABL:
    rs = groups.get(name, [])
    if not rs:
        print("missing runs for", name)
        continue
    vals = {m: A.mean_sd([float(r[m]) for r in rs]) for m in METRICS}
    rows.append((f"{label} (n={len(rs)})", vals))
    per_slice = {}
    for r in rs:
        pf = ROOT / "results" / "preds" / f"{r['run_id']}.json"
        if pf.exists():
            for k, v in A.slice_scores(pf, va, sl).items():
                per_slice.setdefault(k, []).append(v)
    sv = {k: A.mean_sd(v) for k, v in per_slice.items()}
    slice_rows.append((label, sv))
    summary[name] = dict(metrics=vals, slices=sv, seeds=[r["seed"] for r in rs], run_ids=[r["run_id"] for r in rs])
    print(f"{name:18s} " + " ".join(f"{m}={vals[m][0]:.2f}±{vals[m][1]:.2f}" for m in METRICS))
    print(" " * 19 + " ".join(f"{k}={v[0]:.1f}" for k, v in sv.items()))

A.TAB_DIR.mkdir(parents=True, exist_ok=True)
(A.TAB_DIR / "ablations.tex").write_text(A.latex_table(
    rows, "Ablations on the validation split (mean $\\pm$ sd over 3 seeds; frozen configuration).", "tab:ablations"))
cols = ["all", "A1_answer_oov", "A2_handle_tag_username", "A3_far_or_byline", "A5_not_substring"]
hdr = [f"All ({len(sl['all'])})", f"OOV ans. ({len(sl['A1_answer_oov'])})",
       f"@/\\#/user ({len(sl['A2_handle_tag_username'])})", f"Far/byline ({len(sl['A3_far_or_byline'])})",
       f"Non-substr. ({len(sl['A5_not_substring'])})"]
(A.TAB_DIR / "ablation_slices.tex").write_text(A.latex_table(
    slice_rows, "Validation ROUGE-L on hypothesis slices (mean $\\pm$ sd over 3 seeds).", "tab:slices",
    cols=cols, header=hdr))
(ROOT / "results" / "stage5_summary.json").write_text(json.dumps(
    dict(summary=summary, slice_sizes={k: len(v) for k, v in sl.items()}), indent=1, default=str))
print("wrote tables/ablations.tex, tables/ablation_slices.tex")
