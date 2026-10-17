"""Build Colab training and lightweight reporting notebooks for the pointer-generator."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_OUT = ROOT / "notebooks" / "experiments" / "step04a_colab_pointer_generator_experiment.ipynb"
REPORT_OUT = ROOT / "notebooks" / "experiments" / "step04b_pointer_generator_results_report.ipynb"
FILES = ["src/__init__.py", "src/evaluate.py", "src/data.py", "src/runlog.py",
         "src/model.py", "src/fasttext_init.py", "src/generative.py", "src/train_generative.py"]


def md(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(True)}
def code(s): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.splitlines(True)}


SETUP = r'''import os,sys,subprocess,shutil,json,csv,zipfile,hashlib
from pathlib import Path
assert "google.colab" in sys.modules
subprocess.run([sys.executable,"-m","pip","install","-q","gensim","emoji","nltk"],check=True)
import torch
assert torch.cuda.is_available(), "Enable a GPU runtime"
print("GPU",torch.cuda.get_device_name(0),"torch",torch.__version__)
WORK=Path("/content/A2_pointer_generator")
RESULTS=WORK/"generator_results"; CKPT=WORK/"generator_checkpoints"
for p in (WORK/"src",WORK/"data",RESULTS,CKPT): p.mkdir(parents=True,exist_ok=True)
os.environ["A2_RESULTS"]=str(RESULTS); os.environ["A2_CKPT"]=str(CKPT)
os.chdir(WORK)
from google.colab import drive
drive.mount("/content/drive")
BACKUP=Path("/content/drive/MyDrive/CITS4012_A2_pointer_generator"); BACKUP.mkdir(parents=True,exist_ok=True)
for name,dst in (("generator_results",RESULTS),("generator_checkpoints",CKPT)):
    if (BACKUP/name).exists(): shutil.copytree(BACKUP/name,dst,dirs_exist_ok=True)
def backup():
    for src,name in ((RESULTS,"generator_results"),(CKPT,"generator_checkpoints")):
        if src.exists(): shutil.copytree(src,BACKUP/name,dirs_exist_ok=True)
    print("Backed up to",BACKUP)
'''

DATA = r'''target=WORK/"data/train.csv"
if not target.exists():
    source=next((p for p in [Path("/content/train.csv"),Path("/content/data/train.csv")] if p.exists()),None)
    if source: shutil.copy2(source,target)
    else:
        from google.colab import files
        uploaded=files.upload(); assert "train.csv" in uploaded; target.write_bytes(uploaded["train.csv"])
sha=hashlib.sha1(target.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
assert sha=="8a52bb7dff56b11fdad1ab6fd3b848fbadc22849"
print("DATA READY",target,sha)
'''

CONFIG = r'''frozen=json.loads(Path("results/FROZEN_CONFIG.json").read_text())
cfg={k:v for k,v in frozen.items() if not k.startswith("_")}
cfg.update(epochs=20,patience=4,batch_size=32,lr=0.0005,grad_clip=5.0,
           decoder_emb=100,gen_min_freq=2,max_answer_len=16)
Path("generator_config.json").write_text(json.dumps(cfg,indent=2))
print(json.dumps(cfg,indent=2))
'''

RUN = r'''runs=RESULTS/"runs.csv"
done=runs.exists() and any(r["model"]=="PointerGenerator" and r["seed"]=="1"
                           for r in csv.DictReader(runs.open(encoding="utf-8")))
if not done:
    subprocess.run([sys.executable,"-m","src.train_generative","--smoke"],check=True,
                   env={**os.environ,"PYTHONUNBUFFERED":"1"})
    subprocess.run([sys.executable,"-m","src.train_generative"],check=True,
                   env={**os.environ,"PYTHONUNBUFFERED":"1"})
else: print("SKIP completed PointerGenerator seed 1")
backup()
'''

ANALYSE = r'''from src import data as D
from src.evaluate import evaluate,token_f1
run=next(r for r in csv.DictReader((RESULTS/"runs.csv").open(encoding="utf-8"))
         if r["model"]=="PointerGenerator" and r["seed"]=="1")
preds=json.loads((RESULTS/"preds"/f"{run['run_id']}.json").read_text())
_,val_rows=D.get_splits(); exs=D.preprocess(val_rows)
by_id={e["row_id"]:e for e in exs}; by_pred={p["row_id"]:p["pred"] for p in preds}
groups={"all":exs,"fully_extractive":[e for e in exs if e["best_f1"]==1.0],
        "partially_or_non_extractive":[e for e in exs if e["best_f1"]<1.0],
        "no_good_span":[e for e in exs if not e["trainable"]]}
summary={"run":run,"slices":{},"examples":[]}
for name,items in groups.items():
    ps=[by_pred[e["row_id"]] for e in items]
    summary["slices"][name]={"n":len(items),**evaluate(ps,[e["answers"] for e in items])}
for e in exs:
    p=by_pred[e["row_id"]]; score=max(token_f1(p,a) for a in e["answers"])
    source={t["raw"].lower() for t in e["t_tokens"]}; pt=[x[0].lower() for x in D.tokenize(p)]
    summary["examples"].append(dict(row_id=e["row_id"],question=e["question"],tweet=e["tweet"],
        gold=e["answers"],prediction=p,f1=round(100*score,2),best_span_f1=round(100*e["best_f1"],2),
        generated_tokens=[t for t in pt if t not in source]))
summary["examples"].sort(key=lambda x:(x["best_span_f1"],-x["f1"]))
(RESULTS/"generator_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary["slices"],indent=2)); backup()
'''

PACKAGE = r'''archive=Path("/content/pointer_generator_artifacts.zip")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for folder in (RESULTS,CKPT):
        for p in folder.rglob("*"):
            if p.is_file(): z.write(p,p.relative_to(WORK))
shutil.copy2(archive,BACKUP/archive.name)
print("SAVED",BACKUP/archive.name,f"{archive.stat().st_size/1e6:.1f} MB")
'''

LOAD = r'''import json,csv,io,zipfile,sys
from pathlib import Path
from IPython.display import display,Markdown
NAME="pointer_generator_artifacts.zip"
candidates=[Path(NAME),Path("/content")/NAME,Path("/content/drive/MyDrive/CITS4012_A2_pointer_generator")/NAME]
archive=next((p for p in candidates if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive"); archive=next((p for p in candidates if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import files
    up=files.upload(); assert NAME in up; archive=Path("/content")/NAME; archive.write_bytes(up[NAME])
assert archive is not None
zf=zipfile.ZipFile(archive); names=set(zf.namelist())
required={"generator_results/runs.csv","generator_results/generator_summary.json"}
assert required<=names,f"Missing {required-names}"
summary=json.loads(zf.read("generator_results/generator_summary.json"))
print("Loaded",archive,"— no training, model loading, or inference")
'''

TABLE = r'''s=summary["slices"]
lines=["| Validation slice | n | BLEU-1 | METEOR | ROUGE-L | EM | F1 |","|---|---:|---:|---:|---:|---:|---:|"]
for k,v in s.items(): lines.append(f"| {k} | {v['n']} | {v['bleu1']:.2f} | {v['meteor']:.2f} | {v['rougeL']:.2f} | {v['em']:.2f} | {v['f1']:.2f} |")
display(Markdown("## Validation results — one seed\n\n"+"\n".join(lines)))
overall=s["all"]
display(Markdown(f"""The from-scratch pointer-generator obtains **ROUGE-L {overall['rougeL']:.2f}**.
For context, the previous three-seed extractive v2 experiment obtained 44.22 ± 0.16, while the
pretrained T5 baseline obtained 56.93. Those are historical references rather than paired runs."""))
'''

EXAMPLES = r'''xs=summary["examples"]
nonext=[x for x in xs if x["best_span_f1"]<100]
best=sorted(nonext,key=lambda x:x["f1"],reverse=True)[:12]
worst=sorted(nonext,key=lambda x:x["f1"])[:12]
def show(items,title):
    lines=["| F1 | Best extractive F1 | Question | Gold | Prediction | Tokens absent from tweet |","|---:|---:|---|---|---|---|"]
    for x in items:
        clean=lambda y:str(y).replace("|","/").replace("\n"," ")
        lines.append(f"| {x['f1']:.1f} | {x['best_span_f1']:.1f} | {clean(x['question'])} | {clean(x['gold'])} | {clean(x['prediction'])} | {clean(x['generated_tokens'])} |")
    display(Markdown("### "+title+"\n\n"+"\n".join(lines)))
show(best,"Strongest partially/non-extractive cases")
show(worst,"Weakest partially/non-extractive cases")
'''

CONCLUSION = r'''text=f"""# Pointer-generator experiment

The one-seed from-scratch pointer-generator achieved validation ROUGE-L {overall['rougeL']:.2f},
F1 {overall['f1']:.2f}, and EM {overall['em']:.2f}. Its performance on the
partially/non-extractive and no-good-span slices indicates whether generation and copying
address the extractive model's structural limitation. Because this is one seed and uses a
different output architecture, it is an exploratory result. The locked test set was not used.
"""
Path("pointer_generator_report.md").write_text(text,encoding="utf-8")
display(Markdown("## Report-ready conclusion\n\n"+text)); zf.close()
'''


def nb(cells,gpu=False):
    meta={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"colab":{"provenance":[]}}
    if gpu: meta.update(accelerator="GPU"); meta["colab"]["gpuType"]="T4"
    return {"cells":cells,"metadata":meta,"nbformat":4,"nbformat_minor":5}


def build():
    tc=[md("""# TweetQA pointer-generator — one-seed experiment

This notebook trains a small encoder–decoder from scratch. At every answer step it mixes
a vocabulary distribution with an attention-based copy distribution over the tweet. It can
therefore copy rare names and generate short words absent from the tweet. It uses train.csv
only; the locked test set is never loaded. Outputs are isolated in a new MyDrive folder."""),
        md("## 1. Runtime and separate Drive backup"),code(SETUP),md("## 2. Upload train.csv"),code(DATA),md("## 3. Embedded source")]
    for rel in FILES: tc.append(code(f"%%writefile {rel}\n"+(ROOT/rel).read_text(encoding="utf-8")))
    frozen=(ROOT/"results/FROZEN_CONFIG.json").read_text(encoding="utf-8")
    tc += [code("Path('results').mkdir(exist_ok=True)\nPath('results/FROZEN_CONFIG.json').write_text('''"+frozen+"''')"),
           md("## 4. Generator configuration"),code(CONFIG),md("## 5. Smoke test and one-seed training"),code(RUN),
           md("## 6. Extractive/non-extractive analysis"),code(ANALYSE),md("## 7. Package results"),code(PACKAGE)]
    rc=[md("""# Pointer-generator results

This lightweight notebook reads the downloaded generator ZIP. It performs no training,
checkpoint loading, inference, or test-set access."""),md("## 0. Load artifact"),code(LOAD),code(TABLE),
        md("## Qualitative non-extractive cases"),code(EXAMPLES),code(CONCLUSION)]
    TRAIN_OUT.write_text(json.dumps(nb(tc,True),ensure_ascii=False,indent=1),encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(nb(rc),ensure_ascii=False,indent=1),encoding="utf-8")
    print("wrote",TRAIN_OUT.name,len(tc),"cells"); print("wrote",REPORT_OUT.name,len(rc),"cells")


if __name__=="__main__": build()
