"""Build the three-seed quality-weight confirmation and its offline report notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_OUT = ROOT / "colab_quality_weight_3seed.ipynb"
REPORT_OUT = ROOT / "quality_weight_3seed_report.ipynb"
FILES = ["src/__init__.py", "src/evaluate.py", "src/data.py", "src/runlog.py",
         "src/model.py", "src/fasttext_init.py", "src/train.py"]

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}

SETUP = r'''import os,sys,subprocess,shutil,json,csv,zipfile,hashlib,statistics
from pathlib import Path
assert "google.colab" in sys.modules
subprocess.run([sys.executable,"-m","pip","install","-q","gensim","emoji","nltk"],check=True)
import torch
assert torch.cuda.is_available(); print("GPU",torch.cuda.get_device_name(0),"torch",torch.__version__)
WORK=Path("/content/A2_quality_confirm"); RESULTS=WORK/"quality_results"; CKPT=WORK/"quality_checkpoints"
for p in (WORK/"src",WORK/"data",RESULTS,CKPT): p.mkdir(parents=True,exist_ok=True)
os.environ["A2_RESULTS"]=str(RESULTS); os.environ["A2_CKPT"]=str(CKPT); os.chdir(WORK)
from google.colab import drive
drive.mount("/content/drive")
BACKUP=Path("/content/drive/MyDrive/CITS4012_A2_quality_confirm"); BACKUP.mkdir(parents=True,exist_ok=True)
for name,dst in (("quality_results",RESULTS),("quality_checkpoints",CKPT)):
    if (BACKUP/name).exists(): shutil.copytree(BACKUP/name,dst,dirs_exist_ok=True)
def backup():
    for src,name in ((RESULTS,"quality_results"),(CKPT,"quality_checkpoints")):
        if src.exists(): shutil.copytree(src,BACKUP/name,dirs_exist_ok=True)
    print("Backed up to",BACKUP)
'''

DATA = r'''target=WORK/"data/train.csv"
if not target.exists():
    source=next((p for p in (Path("/content/train.csv"),Path("/content/data/train.csv")) if p.exists()),None)
    if source: shutil.copy2(source,target)
    else:
        from google.colab import files
        up=files.upload(); assert "train.csv" in up; target.write_bytes(up["train.csv"])
sha=hashlib.sha1(target.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
assert sha=="8a52bb7dff56b11fdad1ab6fd3b848fbadc22849"; print("DATA READY",sha)
'''

JOBS = r'''frozen=json.loads(Path("results/FROZEN_CONFIG.json").read_text())
BASE={k:v for k,v in frozen.items() if not k.startswith("_")}
COMMON=dict(use_quality_weight=False,use_joint_span_scorer=False,use_joint_span_loss=False,
            use_hard_negative=False,joint_span_weight=0.5,hard_negative_weight=0.2,
            hard_negative_margin=1.0,span_scorer_dim=64)
EXPERIMENTS={"ObjectiveControl":{},"QualityWeight":{"use_quality_weight":True}}
CONFIGS={name:{**BASE,**COMMON,**over} for name,over in EXPERIMENTS.items()}
SEEDS=(1,2,3)
def completed(name,seed):
    p=RESULTS/"runs.csv"
    return p.exists() and any(r["model"]==name and r["seed"]==str(seed) for r in csv.DictReader(p.open(encoding="utf-8")))
def run(name,cfg,seed):
    if completed(name,seed): print("SKIP",name,seed); return
    args=[sys.executable,"-m","src.train","--name",name,"--seed",str(seed),"--set"]
    args += [f"{k}={json.dumps(v)}" for k,v in cfg.items()]
    print("RUN",name,"seed",seed,flush=True)
    subprocess.run(args,check=True,env={**os.environ,"PYTHONUNBUFFERED":"1"}); backup()
'''

RUN = r'''smoke=[sys.executable,"-m","src.train","--name","quality_smoke","--smoke","--set"]
smoke += [f"{k}={json.dumps(v)}" for k,v in CONFIGS["QualityWeight"].items()]
subprocess.run(smoke,check=True,env={**os.environ,"PYTHONUNBUFFERED":"1"})
for seed in SEEDS:
    for name,cfg in CONFIGS.items(): run(name,cfg,seed)
'''

ANALYSE = r'''from src.data import get_splits
from src.evaluate import score_example
runs=[r for r in csv.DictReader((RESULTS/"runs.csv").open(encoding="utf-8"))
      if r["model"] in EXPERIMENTS and int(r["seed"]) in SEEDS]
assert len(runs)==len(EXPERIMENTS)*len(SEEDS), len(runs)
metrics=("bleu1","meteor","rougeL","em","f1")
by={(r["model"],int(r["seed"])):r for r in runs}
summary={"seeds":list(SEEDS),"runs":{},"aggregate":{},"delta_quality_minus_control":{}}
for name in EXPERIMENTS:
    summary["runs"][name]={str(s):{m:float(by[name,s][m]) for m in metrics}|{"run_id":by[name,s]["run_id"]} for s in SEEDS}
    summary["aggregate"][name]={m:{"mean":statistics.mean(float(by[name,s][m]) for s in SEEDS),
                                      "std":statistics.stdev(float(by[name,s][m]) for s in SEEDS)} for m in metrics}
for m in metrics:
    ds=[float(by["QualityWeight",s][m])-float(by["ObjectiveControl",s][m]) for s in SEEDS]
    summary["delta_quality_minus_control"][m]={"by_seed":ds,"mean":statistics.mean(ds),"std":statistics.stdev(ds)}

_,val=get_splits(); gold={e["row_id"]:e["answers"] for e in val}; qids={e["row_id"]:e["qid"] for e in val}
delta_rows=[]
for seed in SEEDS:
    pred={}
    for name in EXPERIMENTS:
        rid=by[name,seed]["run_id"]
        items=json.loads((RESULTS/"preds"/f"{rid}.json").read_text())
        pred[name]={int(x["row_id"]):x["pred"] for x in items}
    assert set(pred["ObjectiveControl"])==set(pred["QualityWeight"])==set(gold)
    for row_id in sorted(gold):
        row={"seed":seed,"row_id":row_id,"qid":qids[row_id]}
        for m in ("rougeL","f1","em"):
            row[m]=score_example(pred["QualityWeight"][row_id],gold[row_id],(m,))[m]-score_example(pred["ObjectiveControl"][row_id],gold[row_id],(m,))[m]
        delta_rows.append(row)
with (RESULTS/"paired_deltas.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=("seed","row_id","qid","rougeL","f1","em")); w.writeheader(); w.writerows(delta_rows)
(RESULTS/"quality_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); backup()
'''

PACKAGE = r'''archive=Path("/content/quality_weight_3seed_artifacts.zip")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for folder in (RESULTS,CKPT):
        for p in folder.rglob("*"):
            if p.is_file(): z.write(p,p.relative_to(WORK))
shutil.copy2(archive,BACKUP/archive.name); print("SAVED",BACKUP/archive.name,f"{archive.stat().st_size/1e6:.1f} MB")
'''

LOAD = r'''import csv,io,json,zipfile,sys,statistics
from pathlib import Path
from IPython.display import display,Markdown
NAME="quality_weight_3seed_artifacts.zip"
paths=[Path(NAME),Path("/content")/NAME,Path("/content/drive/MyDrive/CITS4012_A2_quality_confirm")/NAME]
archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import drive; drive.mount("/content/drive"); archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import files
    up=files.upload(); assert NAME in up; archive=Path("/content")/NAME; archive.write_bytes(up[NAME])
assert archive is not None
zf=zipfile.ZipFile(archive); assert zf.testzip() is None
summary=json.loads(zf.read("quality_results/quality_summary.json")); print("Loaded",archive,"— no training or inference")
'''

TABLE = r'''metrics=("bleu1","meteor","rougeL","em","f1")
lines=["| Model | Seed | BLEU-1 | METEOR | ROUGE-L | EM | F1 |","|---|---:|---:|---:|---:|---:|---:|"]
for name in ("ObjectiveControl","QualityWeight"):
    for seed in summary["seeds"]:
        v=summary["runs"][name][str(seed)]
        lines.append(f"| {name} | {seed} | "+" | ".join(f"{v[m]:.2f}" for m in metrics)+" |")
    a=summary["aggregate"][name]
    lines.append(f"| **{name} mean ± SD** | — | "+" | ".join(f"**{a[m]['mean']:.2f} ± {a[m]['std']:.2f}**" for m in metrics)+" |")
display(Markdown("## Three-seed results\n\n"+"\n".join(lines)))
d=summary["delta_quality_minus_control"]
display(Markdown("## QualityWeight − Control\n\n"+"\n".join([
    "| Metric | Seed 1 | Seed 2 | Seed 3 | Mean ± SD |","|---|---:|---:|---:|---:|",
    *[f"| {m} | "+" | ".join(f"{x:+.2f}" for x in d[m]["by_seed"])+f" | **{d[m]['mean']:+.2f} ± {d[m]['std']:.2f}** |" for m in metrics]])))
'''

BOOTSTRAP = r'''import numpy as np
rows=list(csv.DictReader(io.StringIO(zf.read("quality_results/paired_deltas.csv").decode())))
rng=np.random.default_rng(4012); lines=["| Metric | Mean paired Δ | 95% bootstrap CI | P(Δ≤0) |","|---|---:|---:|---:|"]
for metric in ("rougeL","f1","em"):
    # Cluster bootstrap: sample validation questions, keeping their three seed results together.
    grouped={}
    for r in rows: grouped.setdefault(r["row_id"],[]).append(float(r[metric]))
    values=np.array([grouped[k] for k in sorted(grouped)],dtype=float)
    means=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(10000)])*100
    lo,hi=np.quantile(means,[.025,.975]); point=values.mean()*100
    lines.append(f"| {metric} | {point:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {(means<=0).mean():.3f} |")
display(Markdown("## Paired validation comparison\n\n"+"\n".join(lines)+"\n\nThe bootstrap resamples validation questions and keeps all three seed results for each question together."))
'''

CONCLUSION = r'''r=d["rougeL"]; f=d["f1"]
decision="supported" if r["mean"]>0 and all(x>0 for x in r["by_seed"]) else "not yet supported"
text=f"""# Quality-weight confirmation

Across three matched seeds, quality weighting changed ROUGE-L by **{r['mean']:+.2f} ± {r['std']:.2f}**
and F1 by **{f['mean']:+.2f} ± {f['std']:.2f}** relative to the unchanged objective.
The improvement is **{decision}** by the predeclared practical rule: positive mean and positive ROUGE-L
effect in every seed. Results are validation-only; the locked test set was not accessed.
"""
Path("quality_weight_3seed_report.md").write_text(text,encoding="utf-8"); display(Markdown("## Report-ready conclusion\n\n"+text)); zf.close()
'''

def notebook(cells, gpu=False):
    meta={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"colab":{"provenance":[]}}
    if gpu: meta["accelerator"]="GPU"; meta["colab"]["gpuType"]="T4"
    return {"cells":cells,"metadata":meta,"nbformat":4,"nbformat_minor":5}

def build():
    train=[md("""# Confirm silver-span quality weighting over three seeds

This is the confirmatory comparison selected by the earlier six-condition study. It trains only the unchanged objective and quality-weighted objective at seeds 1, 2, and 3. Architecture, split, vocabulary, optimiser, and early stopping remain fixed. The locked test set is not loaded. Results and checkpoints are saved to a new MyDrive folder: `CITS4012_A2_quality_confirm`."""),
           md("## 1. Runtime and separate Drive backup"),code(SETUP),md("## 2. Upload train.csv"),code(DATA),md("## 3. Embedded source")]
    for rel in FILES:
        train.append(code(f"%%writefile {rel}\n"+(ROOT/rel).read_text(encoding="utf-8")))
    frozen=(ROOT/"results/FROZEN_CONFIG.json").read_text(encoding="utf-8")
    train += [code("Path('results').mkdir(exist_ok=True)\nPath('results/FROZEN_CONFIG.json').write_text('''"+frozen+"''')"),
              md("## 4. Two matched conditions × three seeds"),code(JOBS),md("## 5. Smoke test and training"),code(RUN),
              md("## 6. Aggregate and save paired per-example deltas"),code(ANALYSE),md("## 7. Package for download"),code(PACKAGE)]
    report=[md("""# Quality-weight three-seed report

Upload or locate `quality_weight_3seed_artifacts.zip`. This notebook performs no training or inference and does not need the dataset."""),
            md("## 0. Load and validate artifact"),code(LOAD),code(TABLE),code(BOOTSTRAP),code(CONCLUSION)]
    TRAIN_OUT.write_text(json.dumps(notebook(train,True),ensure_ascii=False,indent=1),encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(notebook(report),ensure_ascii=False,indent=1),encoding="utf-8")
    print("wrote",TRAIN_OUT.name,len(train),"cells")
    print("wrote",REPORT_OUT.name,len(report),"cells")

if __name__ == "__main__": build()
