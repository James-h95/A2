"""Build controlled joint-span training and reporting notebooks."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TRAIN_OUT=ROOT/"step05a_colab_joint_span_experiment.ipynb"
REPORT_OUT=ROOT/"step05b_joint_span_results_report.ipynb"
FILES=["src/__init__.py","src/evaluate.py","src/data.py","src/runlog.py",
       "src/model.py","src/fasttext_init.py","src/train.py"]
def md(s):return {"cell_type":"markdown","metadata":{},"source":s.splitlines(True)}
def code(s):return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.splitlines(True)}

SETUP=r'''import os,sys,subprocess,shutil,json,csv,zipfile,hashlib,statistics
from pathlib import Path
assert "google.colab" in sys.modules
subprocess.run([sys.executable,"-m","pip","install","-q","gensim","emoji","nltk"],check=True)
import torch
assert torch.cuda.is_available(); print("GPU",torch.cuda.get_device_name(0),"torch",torch.__version__)
WORK=Path("/content/A2_joint_span"); RESULTS=WORK/"joint_results"; CKPT=WORK/"joint_checkpoints"
for p in (WORK/"src",WORK/"data",RESULTS,CKPT):p.mkdir(parents=True,exist_ok=True)
os.environ["A2_RESULTS"]=str(RESULTS);os.environ["A2_CKPT"]=str(CKPT);os.chdir(WORK)
from google.colab import drive
drive.mount("/content/drive")
BACKUP=Path("/content/drive/MyDrive/CITS4012_A2_joint_span");BACKUP.mkdir(parents=True,exist_ok=True)
for name,dst in (("joint_results",RESULTS),("joint_checkpoints",CKPT)):
    if (BACKUP/name).exists():shutil.copytree(BACKUP/name,dst,dirs_exist_ok=True)
def backup():
    for src,name in ((RESULTS,"joint_results"),(CKPT,"joint_checkpoints")):
        if src.exists():shutil.copytree(src,BACKUP/name,dirs_exist_ok=True)
    print("Backed up to",BACKUP)
'''
DATA=r'''target=WORK/"data/train.csv"
if not target.exists():
    source=next((p for p in (Path("/content/train.csv"),Path("/content/data/train.csv")) if p.exists()),None)
    if source:shutil.copy2(source,target)
    else:
        from google.colab import files
        up=files.upload();assert "train.csv" in up;target.write_bytes(up["train.csv"])
sha=hashlib.sha1(target.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
assert sha=="8a52bb7dff56b11fdad1ab6fd3b848fbadc22849";print("DATA READY",sha)
'''
JOBS=r'''frozen=json.loads(Path("results/FROZEN_CONFIG.json").read_text())
BASE={k:v for k,v in frozen.items() if not k.startswith("_")}
COMMON=dict(use_quality_weight=False,use_joint_span_scorer=False,use_joint_span_loss=False,
            use_hard_negative=False,joint_span_weight=0.5,hard_negative_weight=0.2,
            hard_negative_margin=1.0,span_scorer_dim=64)
EXPERIMENTS={
 "ObjectiveControl":{},
 "QualityWeight":{"use_quality_weight":True},
 "JointSpan":{"use_joint_span_scorer":True,"use_joint_span_loss":True},
 "JointSpan_Quality":{"use_joint_span_scorer":True,"use_joint_span_loss":True,"use_quality_weight":True},
 "JointSpan_HardNeg":{"use_joint_span_scorer":True,"use_joint_span_loss":True,"use_hard_negative":True},
 "JointSpan_All":{"use_joint_span_scorer":True,"use_joint_span_loss":True,"use_quality_weight":True,"use_hard_negative":True},
}
CONFIGS={name:{**BASE,**COMMON,**over} for name,over in EXPERIMENTS.items()}
def completed(name):
    p=RESULTS/"runs.csv"
    return p.exists() and any(r["model"]==name and r["seed"]=="1" for r in csv.DictReader(p.open(encoding="utf-8")))
def run(name,cfg):
    if completed(name):print("SKIP",name);return
    args=[sys.executable,"-m","src.train","--name",name,"--seed","1","--set"]
    args += [f"{k}={json.dumps(v)}" for k,v in cfg.items()]
    print("RUN",name,flush=True);subprocess.run(args,check=True,env={**os.environ,"PYTHONUNBUFFERED":"1"});backup()
'''
RUN=r'''# Smoke-test the most complex configuration first.
smoke=[sys.executable,"-m","src.train","--name","joint_smoke","--smoke","--set"]
smoke += [f"{k}={json.dumps(v)}" for k,v in CONFIGS["JointSpan_All"].items()]
subprocess.run(smoke,check=True,env={**os.environ,"PYTHONUNBUFFERED":"1"})
for name,cfg in CONFIGS.items():run(name,cfg)
'''
ANALYSE=r'''runs=[r for r in csv.DictReader((RESULTS/"runs.csv").open(encoding="utf-8")) if r["model"] in EXPERIMENTS]
assert len(runs)==len(EXPERIMENTS)
by={r["model"]:r for r in runs};metrics=("bleu1","meteor","rougeL","em","f1")
summary={"models":{n:{m:float(by[n][m]) for m in metrics}|{"run_id":by[n]["run_id"],"params":int(by[n]["notes"].split("params=")[1].split()[0])} for n in EXPERIMENTS}}
contrasts={
 "quality_only":("QualityWeight","ObjectiveControl"),
 "joint_only":("JointSpan","ObjectiveControl"),
 "quality_given_joint":("JointSpan_Quality","JointSpan"),
 "hard_negative_given_joint":("JointSpan_HardNeg","JointSpan"),
 "hard_negative_given_joint_quality":("JointSpan_All","JointSpan_Quality"),
 "all_vs_control":("JointSpan_All","ObjectiveControl"),
}
summary["contrasts"]={k:{m:float(by[a][m])-float(by[b][m]) for m in metrics} for k,(a,b) in contrasts.items()}
(RESULTS/"joint_summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));backup()
'''
PACKAGE=r'''archive=Path("/content/joint_span_artifacts.zip")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for folder in (RESULTS,CKPT):
        for p in folder.rglob("*"):
            if p.is_file():z.write(p,p.relative_to(WORK))
shutil.copy2(archive,BACKUP/archive.name);print("SAVED",BACKUP/archive.name,f"{archive.stat().st_size/1e6:.1f} MB")
'''

LOAD=r'''import json,zipfile,sys
from pathlib import Path
from IPython.display import display,Markdown
NAME="joint_span_artifacts.zip"
paths=[Path(NAME),Path("/content")/NAME,Path("/content/drive/MyDrive/CITS4012_A2_joint_span")/NAME]
archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
 from google.colab import drive;drive.mount("/content/drive");archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
 from google.colab import files
 up=files.upload();assert NAME in up;archive=Path("/content")/NAME;archive.write_bytes(up[NAME])
assert archive is not None
zf=zipfile.ZipFile(archive);summary=json.loads(zf.read("joint_results/joint_summary.json"));print("Loaded",archive,"— no training or inference")
'''
TABLE=r'''labels={"ObjectiveControl":"Current objective","QualityWeight":"Quality weighting only","JointSpan":"Joint span only","JointSpan_Quality":"Joint + quality","JointSpan_HardNeg":"Joint + hard negative","JointSpan_All":"Joint + quality + hard negative"}
order=list(labels);ms=("bleu1","meteor","rougeL","em","f1")
lines=["| Model | Params | BLEU-1 | METEOR | ROUGE-L | EM | F1 |","|---|---:|---:|---:|---:|---:|---:|"]
for n in order:
 v=summary["models"][n];lines.append("| "+labels[n]+f" | {v['params']:,} | "+" | ".join(f"{v[m]:.2f}" for m in ms)+" |")
display(Markdown("## One-seed controlled results\n\n"+"\n".join(lines)))
'''
EFFECTS=r'''names={"quality_only":"Quality weighting alone","joint_only":"Joint span scorer alone","quality_given_joint":"Quality weighting added to joint","hard_negative_given_joint":"Hard negative added to joint","hard_negative_given_joint_quality":"Hard negative added to joint + quality","all_vs_control":"All additions versus control"}
lines=["| Controlled contrast | ΔROUGE-L | ΔEM | ΔF1 |","|---|---:|---:|---:|"]
for k,label in names.items():
 v=summary["contrasts"][k];lines.append(f"| {label} | {v['rougeL']:+.2f} | {v['em']:+.2f} | {v['f1']:+.2f} |")
display(Markdown("## Component effects\n\n"+"\n".join(lines)))
best=max(order,key=lambda n:summary["models"][n]["rougeL"]);control=summary["models"]["ObjectiveControl"];b=summary["models"][best]
display(Markdown(f"Best configuration: **{labels[best]}**, ROUGE-L **{b['rougeL']:.2f}** "
                 f"({b['rougeL']-control['rougeL']:+.2f} versus control). Because this is one seed, a promising gain must be confirmed with three seeds."))
'''
CURVES=r'''# Summarise the saved best epoch and whether later overfitting occurred.
rows=[]
for n in order:
 rid=summary["models"][n]["run_id"];h=json.loads(zf.read(f"joint_results/history/{rid}.json"))
 hist=h["history"];rows.append((labels[n],h["best_epoch"],len(hist),hist[0]["train_loss"],hist[-1]["train_loss"]))
lines=["| Model | Best epoch | Epochs run | First loss | Last loss |","|---|---:|---:|---:|---:|"]
for x in rows:lines.append(f"| {x[0]} | {x[1]} | {x[2]} | {x[3]:.3f} | {x[4]:.3f} |")
display(Markdown("## Optimisation check\n\n"+"\n".join(lines)))
'''
CONCLUSION=r'''v=summary["contrasts"];text=f"""# Joint-span objective experiment

In a one-seed controlled comparison, quality weighting alone changed ROUGE-L by
{v['quality_only']['rougeL']:+.2f}, joint span scoring alone by {v['joint_only']['rougeL']:+.2f},
and the complete joint + quality + hard-negative configuration by
{v['all_vs_control']['rougeL']:+.2f} relative to the current objective.

These validation-only results isolate each component under the same split, seed, and frozen
hyperparameters. Any selected configuration should be confirmed over three seeds before use.
The locked test set was not accessed.
"""
Path("joint_span_report.md").write_text(text,encoding="utf-8");display(Markdown("## Report-ready conclusion\n\n"+text));zf.close()
'''
def nb(cells,gpu=False):
 meta={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"colab":{"provenance":[]}}
 if gpu:meta["accelerator"]="GPU";meta["colab"]["gpuType"]="T4"
 return {"cells":cells,"metadata":meta,"nbformat":4,"nbformat_minor":5}
def build():
 tc=[md("""# Joint-span objective: controlled component study

This notebook keeps TweetBiDAF v2 fixed and isolates three output-objective changes:
silver-span quality weighting, globally scored candidate spans, and hard-negative ranking.
All six configurations use seed 1 and the same validation split. The test set is not loaded.
Artifacts use a new MyDrive folder."""),md("## 1. Runtime and backup"),code(SETUP),md("## 2. Upload train.csv"),code(DATA),md("## 3. Embedded source")]
 for rel in FILES:tc.append(code(f"%%writefile {rel}\n"+(ROOT/rel).read_text(encoding="utf-8")))
 frozen=(ROOT/"results/FROZEN_CONFIG.json").read_text(encoding="utf-8")
 tc += [code("Path('results').mkdir(exist_ok=True)\nPath('results/FROZEN_CONFIG.json').write_text('''"+frozen+"''')"),md("## 4. Factorial jobs"),code(JOBS),md("## 5. Smoke test and six one-seed runs"),code(RUN),md("## 6. Component contrasts"),code(ANALYSE),md("## 7. Package"),code(PACKAGE)]
 rc=[md("""# Joint-span component results

This notebook reads the downloaded ZIP and reports each component's controlled effect. It performs no training, inference, or test-set access."""),md("## 0. Load artifact"),code(LOAD),code(TABLE),code(EFFECTS),code(CURVES),code(CONCLUSION)]
 TRAIN_OUT.write_text(json.dumps(nb(tc,True),ensure_ascii=False,indent=1),encoding="utf-8")
 REPORT_OUT.write_text(json.dumps(nb(rc),ensure_ascii=False,indent=1),encoding="utf-8")
 print("wrote",TRAIN_OUT.name,len(tc),"cells");print("wrote",REPORT_OUT.name,len(rc),"cells")
if __name__=="__main__":build()
