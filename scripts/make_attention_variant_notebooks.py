"""Build a controlled one-seed study of self-attention variants and its report."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_OUT = ROOT / "notebooks" / "experiments" / "step07a_colab_attention_variants.ipynb"
REPORT_OUT = ROOT / "notebooks" / "experiments" / "step07b_attention_variants_report.ipynb"
FILES = ["src/__init__.py", "src/evaluate.py", "src/data.py", "src/runlog.py",
         "src/model.py", "src/fasttext_init.py", "src/train.py"]

def md(text): return {"cell_type":"markdown","metadata":{},"source":text.splitlines(True)}
def code(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(True)}

SETUP = r'''import os,sys,subprocess,shutil,json,csv,zipfile,hashlib,statistics
from pathlib import Path
assert "google.colab" in sys.modules
subprocess.run([sys.executable,"-m","pip","install","-q","gensim","emoji","nltk"],check=True)
import torch
assert torch.cuda.is_available(); print("GPU",torch.cuda.get_device_name(0),"torch",torch.__version__)
WORK=Path("/content/A2_attention_variants"); RESULTS=WORK/"attention_variant_results"; CKPT=WORK/"attention_variant_checkpoints"
for p in (WORK/"src",WORK/"data",RESULTS,CKPT): p.mkdir(parents=True,exist_ok=True)
os.environ["A2_RESULTS"]=str(RESULTS); os.environ["A2_CKPT"]=str(CKPT); os.chdir(WORK)
from google.colab import drive
drive.mount("/content/drive")
BACKUP=Path("/content/drive/MyDrive/CITS4012_A2_attention_variants"); BACKUP.mkdir(parents=True,exist_ok=True)
for name,dst in (("attention_variant_results",RESULTS),("attention_variant_checkpoints",CKPT)):
    if (BACKUP/name).exists(): shutil.copytree(BACKUP/name,dst,dirs_exist_ok=True)
def backup():
    for src,name in ((RESULTS,"attention_variant_results"),(CKPT,"attention_variant_checkpoints")):
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
            use_hard_negative=False,condition_end_on_start=False)
EXPERIMENTS={
 "AttnNone":{"use_self_attn":False,"self_attn_mode":"gated"},
 "CurrentGated":{"use_self_attn":True,"self_attn_mode":"gated"},
 "ResidualAttn":{"use_self_attn":True,"self_attn_mode":"residual"},
 "QuestionGated":{"use_self_attn":True,"self_attn_mode":"question_gated"},
}
CONFIGS={name:{**BASE,**COMMON,**over} for name,over in EXPERIMENTS.items()}; SEED=1
def completed(name):
    p=RESULTS/"runs.csv"
    return p.exists() and any(r["model"]==name and r["seed"]==str(SEED) for r in csv.DictReader(p.open(encoding="utf-8")))
def run(name,cfg):
    if completed(name): print("SKIP",name); return
    args=[sys.executable,"-m","src.train","--name",name,"--seed",str(SEED),"--set"]
    args += [f"{k}={json.dumps(v)}" for k,v in cfg.items()]
    print("RUN",name,flush=True); subprocess.run(args,check=True,env={**os.environ,"PYTHONUNBUFFERED":"1"}); backup()
'''

RUN = r'''for name in ("CurrentGated","ResidualAttn","QuestionGated"):
    smoke=[sys.executable,"-m","src.train","--name",f"{name}_smoke","--smoke","--set"]
    smoke += [f"{k}={json.dumps(v)}" for k,v in CONFIGS[name].items()]
    subprocess.run(smoke,check=True,env={**os.environ,"PYTHONUNBUFFERED":"1"})
for name,cfg in CONFIGS.items(): run(name,cfg)
'''

ANALYSE = r'''from src.data import get_splits,_question_word
from src.evaluate import score_example
runs=[r for r in csv.DictReader((RESULTS/"runs.csv").open(encoding="utf-8")) if r["model"] in EXPERIMENTS and r["seed"]==str(SEED)]
assert len(runs)==len(EXPERIMENTS),len(runs)
by={r["model"]:r for r in runs}; metrics=("bleu1","meteor","rougeL","em","f1")
summary={"seed":SEED,"models":{n:{m:float(by[n][m]) for m in metrics}|{"run_id":by[n]["run_id"],"params":int(by[n]["notes"].split("params=")[1].split()[0])} for n in EXPERIMENTS}}
summary["vs_current"]={n:{m:float(by[n][m])-float(by["CurrentGated"][m]) for m in metrics} for n in EXPERIMENTS if n!="CurrentGated"}
_,val=get_splits(); gold={e["row_id"]:e["answers"] for e in val}; qtype={e["row_id"]:_question_word(e["question"]) for e in val}
pred={}
for name in EXPERIMENTS:
    rid=by[name]["run_id"]; items=json.loads((RESULTS/"preds"/f"{rid}.json").read_text())
    pred[name]={int(x["row_id"]):x["pred"] for x in items}
assert all(set(p)==set(gold) for p in pred.values())
rows=[]
for name in EXPERIMENTS:
    if name=="CurrentGated": continue
    for row_id in sorted(gold):
        row={"model":name,"row_id":row_id,"question_type":qtype[row_id]}
        for m in ("rougeL","f1","em"):
            row[m]=score_example(pred[name][row_id],gold[row_id],(m,))[m]-score_example(pred["CurrentGated"][row_id],gold[row_id],(m,))[m]
        rows.append(row)
with (RESULTS/"paired_deltas.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=("model","row_id","question_type","rougeL","f1","em")); w.writeheader(); w.writerows(rows)
(RESULTS/"attention_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); backup()
'''

PACKAGE = r'''archive=Path("/content/attention_variants_artifacts.zip")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for folder in (RESULTS,CKPT):
        for p in folder.rglob("*"):
            if p.is_file(): z.write(p,p.relative_to(WORK))
shutil.copy2(archive,BACKUP/archive.name); print("SAVED",BACKUP/archive.name,f"{archive.stat().st_size/1e6:.1f} MB")
'''

LOAD = r'''import csv,io,json,zipfile,sys,collections
from pathlib import Path
from IPython.display import display,Markdown
NAME="attention_variants_artifacts.zip"
paths=[Path(NAME),Path("/content")/NAME,Path("/content/drive/MyDrive/CITS4012_A2_attention_variants")/NAME]
archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import drive; drive.mount("/content/drive"); archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import files
    up=files.upload(); assert NAME in up; archive=Path("/content")/NAME; archive.write_bytes(up[NAME])
assert archive is not None
zf=zipfile.ZipFile(archive); assert zf.testzip() is None
summary=json.loads(zf.read("attention_variant_results/attention_summary.json")); print("Loaded",archive,"— no training or inference")
'''

TABLE = r'''labels={"AttnNone":"No self-attention","CurrentGated":"Current gated","ResidualAttn":"Ungated residual","QuestionGated":"Question-conditioned gated"}; order=list(labels); ms=("bleu1","meteor","rougeL","em","f1")
lines=["| Model | Params | BLEU-1 | METEOR | ROUGE-L | EM | F1 | ΔROUGE-L vs current |","|---|---:|---:|---:|---:|---:|---:|---:|"]
for n in order:
    v=summary["models"][n]; delta=v["rougeL"]-summary["models"]["CurrentGated"]["rougeL"]
    lines.append(f"| {labels[n]} | {v['params']:,} | "+" | ".join(f"{v[m]:.2f}" for m in ms)+f" | {delta:+.2f} |")
display(Markdown("## Controlled attention results\n\n"+"\n".join(lines)))
'''

BOOTSTRAP = r'''import numpy as np
rows=list(csv.DictReader(io.StringIO(zf.read("attention_variant_results/paired_deltas.csv").decode()))); rng=np.random.default_rng(4012)
lines=["| Variant vs current | Metric | Mean Δ | 95% paired bootstrap CI | P(Δ≤0) |","|---|---|---:|---:|---:|"]
for model in ("AttnNone","ResidualAttn","QuestionGated"):
    subset=[r for r in rows if r["model"]==model]
    for metric in ("rougeL","f1","em"):
        d=np.array([float(r[metric]) for r in subset]); means=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(10000)])*100
        lo,hi=np.quantile(means,[.025,.975]); lines.append(f"| {labels[model]} | {metric} | {d.mean()*100:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {(means<=0).mean():.3f} |")
display(Markdown("## Paired validation comparison\n\n"+"\n".join(lines)))
'''

SLICES = r'''lines=["| Variant | Question type | N | ΔROUGE-L |","|---|---|---:|---:|"]
for model in ("AttnNone","ResidualAttn","QuestionGated"):
    subset=[r for r in rows if r["model"]==model]
    groups=collections.defaultdict(list)
    for r in subset: groups[r["question_type"]].append(float(r["rougeL"])*100)
    for qt in sorted(groups): lines.append(f"| {labels[model]} | {qt} | {len(groups[qt])} | {sum(groups[qt])/len(groups[qt]):+.2f} |")
display(Markdown("## ROUGE-L effect by question type\n\n"+"\n".join(lines)+"\n\nSlice results are exploratory; small groups should not drive selection."))
'''

CONCLUSION = r'''current=summary["models"]["CurrentGated"]; candidates=[n for n in summary["models"] if n!="CurrentGated"]
best=max(candidates,key=lambda n:summary["models"][n]["rougeL"]); gain=summary["models"][best]["rougeL"]-current["rougeL"]
text=f"""# Attention-variant experiment

The strongest challenger was **{labels[best]}**, with validation ROUGE-L {summary['models'][best]['rougeL']:.2f}
({gain:+.2f} versus the current gated attention). This is a one-seed screening result. Only a challenger
that clearly exceeds the current model should proceed to a matched three-seed confirmation.
The locked test set was not accessed.
"""
Path("attention_variants_report.md").write_text(text,encoding="utf-8"); display(Markdown("## Report-ready conclusion\n\n"+text)); zf.close()
'''

def notebook(cells,gpu=False):
    meta={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"colab":{"provenance":[]}}
    if gpu: meta["accelerator"]="GPU"; meta["colab"]["gpuType"]="T4"
    return {"cells":cells,"metadata":meta,"nbformat":4,"nbformat_minor":5}

def build():
    train=[md("""# Self-attention variants: controlled screening

This notebook compares no self-attention, the current R-Net-style gated self-attention, ungated residual self-attention, and question-conditioned gated self-attention. All runs use seed 1 and the frozen TweetBiDAF v2 training setup. Quality weighting and earlier output-head experiments are disabled so the attention effect is isolated. The locked test set is not loaded. Artifacts use the new Drive folder `CITS4012_A2_attention_variants`."""),md("## 1. Runtime and Drive backup"),code(SETUP),md("## 2. Upload train.csv"),code(DATA),md("## 3. Embedded source")]
    for rel in FILES: train.append(code(f"%%writefile {rel}\n"+(ROOT/rel).read_text(encoding="utf-8")))
    frozen=(ROOT/"results/FROZEN_CONFIG.json").read_text(encoding="utf-8")
    train += [code("Path('results').mkdir(exist_ok=True)\nPath('results/FROZEN_CONFIG.json').write_text('''"+frozen+"''')"),md("## 4. Four attention conditions"),code(JOBS),md("## 5. Smoke tests and training"),code(RUN),md("## 6. Metrics and paired deltas"),code(ANALYSE),md("## 7. Package for download"),code(PACKAGE)]
    report=[md("""# Attention-variant report

Upload or locate `attention_variants_artifacts.zip`. This notebook performs no training or inference and does not require the dataset."""),md("## 0. Load and validate artifact"),code(LOAD),code(TABLE),code(BOOTSTRAP),code(SLICES),code(CONCLUSION)]
    TRAIN_OUT.write_text(json.dumps(notebook(train,True),ensure_ascii=False,indent=1),encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(notebook(report),ensure_ascii=False,indent=1),encoding="utf-8")
    print("wrote",TRAIN_OUT.name,len(train),"cells"); print("wrote",REPORT_OUT.name,len(report),"cells")

if __name__=="__main__": build()
