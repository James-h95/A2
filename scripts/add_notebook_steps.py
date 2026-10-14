"""Add explicit Step headings to the experiment notebooks without clearing outputs."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPORT_STEPS = {
    "step02c_attention_results_report.ipynb": ["Load saved results", "Aggregate attention ablation", "Inspect attention diagnostics", "Review error categories", "Review representative examples", "Write the conclusion"],
    "boundary_step02c_attention_results_report.ipynb": ["Load the artifact", "Compare boundary objectives", "Inspect boundary diagnostics", "Review paired examples", "Write the conclusion"],
    "step07b_attention_variants_report.ipynb": ["Load and validate the artifact", "Compare attention variants", "Run paired bootstrap analysis", "Inspect question-type slices", "Write the conclusion"],
    "step05b_joint_span_results_report.ipynb": ["Load the artifact", "Compare component scores", "Estimate component effects", "Check optimisation histories", "Write the conclusion"],
    "step09b_linear_span_reranker_report.ipynb": ["Load and validate the artifact", "Compare reranker metrics", "Run paired bootstrap analysis", "Inspect slices and selected ranks", "Inspect feature effects", "Write the conclusion"],
    "step10b_oof_span_reranker_report.ipynb": ["Load and validate the artifact", "Compare OOF reranker metrics", "Run paired bootstrap analysis", "Inspect validation slices", "Inspect ranks and feature importance", "Write the conclusion"],
    "step04b_pointer_generator_results_report.ipynb": ["Load the artifact", "Compare generation and extraction", "Inspect extractive-status slices", "Write the conclusion"],
    "step06b_quality_weight_3seed_report.ipynb": ["Load and validate the artifact", "Compare three-seed results", "Run paired bootstrap analysis", "Write the conclusion"],
    "step08b_topk_span_oracle_report.ipynb": ["Load and validate the artifact", "Measure the Top-K oracle ceiling", "Inspect question and region slices", "Inspect the required candidate rank", "Write the conclusion"],
}


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text + "\n"]}


def normalise_numbered_headings(nb):
    pattern = re.compile(r"^(##)\s+(\d+)\.\s*(.*)$")
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        lines = []
        for line in "".join(cell.get("source", [])).splitlines(True):
            stripped = line.rstrip("\r\n")
            ending = line[len(stripped):]
            match = pattern.match(stripped)
            if match:
                stripped = f"## Step {match.group(2)} — {match.group(3)}"
            lines.append(stripped + ending)
        cell["source"] = lines


def label_report_code_cells(nb, labels):
    cells, code_index = [], 0
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            label = labels[code_index]
            expected = f"## Step {code_index + 1} — {label}"
            previous = "".join(cells[-1].get("source", [])).strip() if cells else ""
            if not previous.startswith("## Step"):
                cells.append(markdown(expected))
            else:
                cells[-1]["source"] = [expected + "\n"]
            code_index += 1
        cells.append(cell)
    if code_index != len(labels):
        raise ValueError(f"expected {len(labels)} code cells, found {code_index}")
    nb["cells"] = cells


def main():
    for path in sorted(ROOT.glob("*.ipynb")):
        if path.name == "template.ipynb":
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        normalise_numbered_headings(nb)
        if path.name in {"step01a_gpu_runner.ipynb", "step01b_gpu_runner_run2.ipynb"}:
            for cell in nb["cells"]:
                if cell.get("cell_type") != "markdown":
                    continue
                text = "".join(cell.get("source", []))
                text = text.replace("## Project code", "## Step 1 — Write the project code")
                text = text.replace("## Run", "## Step 2 — Run the experiment queue")
                cell["source"] = text.splitlines(True)
        if path.name in REPORT_STEPS:
            label_report_code_cells(nb, REPORT_STEPS[path.name])
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("updated", path.name)


if __name__ == "__main__":
    main()
