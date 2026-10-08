"""Assemble the single submission notebook from template.ipynb and the src/ modules.

  python build_notebook.py --group GROUPID --hf-repo <user>/<repo> [--execute]

- Every src/ module is inlined as code cells. Local relative imports and CLI
  `__main__` blocks are stripped, and the module aliases used in the code
  (D = data, A = analysis, ...) point at the notebook namespace.
- TRAIN = False (default) loads checkpoints, histories and logs from the public
  Hugging Face repo instead of retraining. TRAIN = True retrains everything.
- With --execute the notebook runs top to bottom (cwd = A2/, local artifacts)
  and is saved with outputs.
"""
import argparse
import re
import runpy
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

_LOCAL_TRY = re.compile(
    r"^(?P<i>[ ]*)try:\n(?:(?P=i)    from \.[^\n]*\n)+(?P=i)except ImportError:[^\n]*\n(?:(?P=i)    (?:from|import) [^\n]*\n)+",
    re.M)
_DROP_LINES = re.compile(r"^(?:sys\.path\.insert\(0, str\(ROOT\)\)|from src[ .][^\n]*|ROOT = Path\(__file__\)[^\n]*)\n", re.M)


def inline(path):
    s = Path(path).read_text(encoding="utf-8")
    s = _LOCAL_TRY.sub("", s)
    s = _DROP_LINES.sub("", s)
    k = s.find('\nif __name__ == "__main__":')
    if k >= 0:
        s = s[:k] + "\n"
    return s.strip() + "\n"


def md(text):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text):
    return nbformat.v4.new_code_cell(text.strip())


def build(group, hf_repo):
    tpl = nbformat.read(ROOT / "template.ipynb", as_version=4)
    heads = {"title": None, "readme": None, "s1": None, "s2": None, "s3": None}
    for c in tpl.cells:
        src = "".join(c.source)
        if src.startswith("# 2026 CITS4012 Project"):
            heads["title"] = c
        elif src.startswith("# Readme"):
            heads["readme"] = c
        elif src.startswith("# 1.Dataset Processing"):
            heads["s1"] = c
        elif src.startswith("# 2. Model Implementation"):
            heads["s2"] = c
        elif src.startswith("# 3.Testing and Evaluation"):
            heads["s3"] = c
    assert all(heads.values()), "template headings changed"

    ns = runpy.run_path(str(ROOT / "notebook_parts" / "cells.py"))
    readme = ns["README"].replace("{HF_REPO}", hf_repo).replace("{GROUP}", group)
    parts = ns["PARTS"]

    def expand(items):
        out = []
        for it in items:
            if it["type"] == "md":
                out.append(md(it["text"]))
            elif it["type"] == "code":
                out.append(code(it["text"].replace("{HF_REPO}", hf_repo).replace("{GROUP}", group)))
            elif it["type"] == "module":
                out.append(md(f"**`{it['path']}`** ({it.get('note', '')})" if it.get("note") else f"**`{it['path']}`**"))
                out.append(code(f"# ---- {it['path']} ----\n" + inline(SRC / it["path"])))
        return out

    title = heads["title"]
    title.source = f"# 2026 CITS4012 Project\n**Group {group}**: TweetQA with TweetBiDAF, an extractive QA model trained from scratch."
    rd = heads["readme"]
    rd.source = "# Readme\n" + readme
    cells = [title, rd, heads["s1"]] + expand(parts["s1"]) + [heads["s2"]] + expand(parts["s2"]) + \
            [heads["s3"]] + expand(parts["s3"])
    nb = nbformat.v4.new_notebook()
    nb.metadata = tpl.metadata
    nb.metadata["accelerator"] = "GPU"
    nb.cells = cells
    return nb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--hf-repo", required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    nb = build(a.group, a.hf_repo)
    out = ROOT / f"CITS4012_{a.group}.ipynb"
    nbformat.write(nb, out)
    print("wrote", out, len(nb.cells), "cells")
    if a.execute:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=a.timeout, kernel_name="cits4012-a2",
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbformat.write(nb, out)
        print("executed and saved with outputs:", out)
