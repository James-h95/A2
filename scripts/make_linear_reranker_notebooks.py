"""Build the fixed linear Top-10 reranker experiment and offline report."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_OUT = ROOT / "notebooks" / "experiments" / "step09a_colab_linear_span_reranker.ipynb"
REPORT_OUT = ROOT / "notebooks" / "experiments" / "step09b_linear_span_reranker_report.ipynb"
FILES = ["src/__init__.py", "src/evaluate.py", "src/data.py", "src/runlog.py",
         "src/model.py", "src/fasttext_init.py", "src/train.py", "src/rerank.py"]

def md(text): return {"cell_type":"markdown","metadata":{},"source":text.splitlines(True)}
def code(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(True)}

SETUP = r'''import os,sys,subprocess,shutil,json,csv,zipfile,hashlib,io,collections,statistics
from pathlib import Path
assert "google.colab" in sys.modules
subprocess.run([sys.executable,"-m","pip","install","-q","gensim","emoji","nltk","scikit-learn","joblib","lightgbm"],check=True)
import torch,numpy as np,joblib
assert torch.cuda.is_available(); DEVICE="cuda"; print("GPU",torch.cuda.get_device_name(0),"torch",torch.__version__)
WORK=Path("/content/A2_linear_reranker"); OUT=WORK/"reranker_results"
for p in (WORK/"src",WORK/"data",OUT): p.mkdir(parents=True,exist_ok=True)
os.chdir(WORK)
from google.colab import drive
drive.mount("/content/drive")
BACKUP=Path("/content/drive/MyDrive/CITS4012_A2_reranker"); BACKUP.mkdir(parents=True,exist_ok=True)
SOURCE_NAME="attention_variants_artifacts.zip"
source_paths=[Path("/content")/SOURCE_NAME,Path("/content/drive/MyDrive/CITS4012_A2_attention_variants")/SOURCE_NAME]
SOURCE=next((p for p in source_paths if p.exists()),None)
if SOURCE is None:
    from google.colab import files
    up=files.upload(); assert SOURCE_NAME in up; SOURCE=Path("/content")/SOURCE_NAME; SOURCE.write_bytes(up[SOURCE_NAME])
with zipfile.ZipFile(SOURCE) as z: assert z.testzip() is None
print("SOURCE READY",SOURCE)
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

LOAD = r'''from src import data as D
from src.train import load_checkpoint
from src.evaluate import evaluate,score_example
from src.rerank import (topk_candidates,fit_pairwise_reranker,fit_lambdamart_reranker,
                        rerank,linear_weights,tree_importances,span_region)
with zipfile.ZipFile(SOURCE) as z:
    runs=list(csv.DictReader(io.StringIO(z.read("attention_variant_results/runs.csv").decode("utf-8-sig"))))
    run=next(r for r in runs if r["model"]=="CurrentGated" and r["seed"]=="1")
    RUN_ID=run["run_id"]; ckpt=WORK/f"{RUN_ID}.pt"
    ckpt.write_bytes(z.read(f"attention_variant_checkpoints/{RUN_ID}.pt"))
model,cfg,wv,cv=load_checkpoint(ckpt,DEVICE)
train_rows,val_rows=D.get_splits(); prep=dict(use_normaliser=cfg["use_normaliser"],max_span_len=cfg["max_span_len"],tau=cfg["tau"],soft_tau=cfg.get("soft_tau"))
train=D.preprocess(train_rows,prep); val=D.preprocess(val_rows,prep); D.encode(train,wv,cv); D.encode(val,wv,cv)
print("MODEL",RUN_ID,"train",len(train),"val",len(val))
'''

CANDIDATES = r'''def save_jsonl(path,records):
    with path.open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
def load_jsonl(path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
def cached_candidates(name,examples):
    cache=BACKUP/f"{name}_top10_candidates.jsonl"
    if cache.exists():
        records=load_jsonl(cache); print("CACHE",name,len(records))
    else:
        records=topk_candidates(model,examples,DEVICE,top_k=10,batch_size=64,max_span_len=cfg["max_span_len"])
        save_jsonl(cache,records); print("GENERATED",name,len(records),"and cached",cache)
    assert len(records)==len(examples)
    return records
train_candidates=cached_candidates("train",train)
val_candidates=cached_candidates("val",val)
'''

FIT = r'''# Both settings are fixed here and are not tuned on validation.
REGULARIZATION=1.0
linear_model,linear_fit=fit_pairwise_reranker(train,train_candidates,regularization=REGULARIZATION)
tree_model,tree_fit=fit_lambdamart_reranker(train,train_candidates)
joblib.dump(linear_model,OUT/"linear_reranker.joblib"); joblib.dump(tree_model,OUT/"lambdamart_reranker.joblib")
weights=linear_weights(linear_model); importances=tree_importances(tree_model)
(OUT/"feature_weights.json").write_text(json.dumps(weights,indent=2),encoding="utf-8")
(OUT/"tree_importances.json").write_text(json.dumps(importances,indent=2),encoding="utf-8")
print("LINEAR FIT",linear_fit); print("LAMBDAMART FIT",tree_fit)
print("TOP LINEAR WEIGHTS",weights[:10]); print("TOP TREE IMPORTANCES",importances[:10])
'''

EVALUATE = r'''golds=[ex["answers"] for ex in val]; base_preds=[r["candidates"][0]["text"] for r in val_candidates]
linear_preds,linear_selections=rerank(linear_model,val,val_candidates)
tree_preds,tree_selections=rerank(tree_model,val,val_candidates)
oracle_preds=[]
for ex,r in zip(val,val_candidates):
    oracle_preds.append(max(r["candidates"],key=lambda c:score_example(c["text"],ex["answers"],("rougeL",))["rougeL"])["text"])
scores={"BaseTop1":evaluate(base_preds,golds),"LinearReranker":evaluate(linear_preds,golds),
        "LambdaMART":evaluate(tree_preds,golds),"Top10Oracle":evaluate(oracle_preds,golds)}
summary={"source_run":RUN_ID,"top_k":10,"regularization":REGULARIZATION,
         "fit":{"LinearReranker":linear_fit,"LambdaMART":tree_fit},"scores":scores,
         "delta_vs_base":{name:{m:scores[name][m]-scores["BaseTop1"][m] for m in scores["BaseTop1"]} for name in ("LinearReranker","LambdaMART")},
         "selected_rank_counts":{name:dict(collections.Counter(str(x["selected_rank"]) for x in sels)) for name,sels in (("LinearReranker",linear_selections),("LambdaMART",tree_selections))},
         "feature_weights":weights,"tree_importances":importances}
rows=[]
for ex,bp,lp,tp,op,lsel,tsel in zip(val,base_preds,linear_preds,tree_preds,oracle_preds,linear_selections,tree_selections):
    row={"row_id":ex["row_id"],"qid":ex["qid"],"question_type":D._question_word(ex["question"]),
         "silver_region":span_region(ex,*ex["best"]) if ex["best"] is not None else "no_span",
         "linear_rank":lsel["selected_rank"],"linear_region":lsel["region"],
         "tree_rank":tsel["selected_rank"],"tree_region":tsel["region"]}
    for name,pred in (("linear",lp),("tree",tp)):
        for m in ("rougeL","f1","em"):
            base=score_example(bp,ex["answers"],(m,))[m]; new=score_example(pred,ex["answers"],(m,))[m]
            row[f"{name}_{m}"]=new-base
    rows.append(row)
with (OUT/"paired_deltas.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with (OUT/"val_predictions.jsonl").open("w",encoding="utf-8") as f:
    for ex,bp,lp,tp,op,lsel,tsel in zip(val,base_preds,linear_preds,tree_preds,oracle_preds,linear_selections,tree_selections):
        f.write(json.dumps(dict(row_id=ex["row_id"],qid=ex["qid"],base=bp,linear=lp,lambdamart=tp,oracle=op,linear_selection=lsel,tree_selection=tsel),ensure_ascii=False)+"\n")
(OUT/"reranker_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
'''

PACKAGE = r'''archive=Path("/content/linear_span_reranker_artifacts.zip")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file(): z.write(p,p.relative_to(WORK))
shutil.copy2(archive,BACKUP/archive.name); print("SAVED",BACKUP/archive.name,f"{archive.stat().st_size/1e6:.1f} MB")
'''

REPORT_LOAD = r'''import json,csv,io,zipfile,sys,collections
from pathlib import Path
from IPython.display import display,Markdown
NAME="linear_span_reranker_artifacts.zip"
paths=[Path(NAME),Path("/content")/NAME,Path("/content/drive/MyDrive/CITS4012_A2_reranker")/NAME]
archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import drive; drive.mount("/content/drive"); archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import files
    up=files.upload(); assert NAME in up; archive=Path("/content")/NAME; archive.write_bytes(up[NAME])
assert archive is not None
zf=zipfile.ZipFile(archive); assert zf.testzip() is None
summary=json.loads(zf.read("reranker_results/reranker_summary.json")); rows=list(csv.DictReader(io.StringIO(zf.read("reranker_results/paired_deltas.csv").decode())))
print("Loaded",archive,"source",summary["source_run"],"— no training or inference")
'''

TABLE = r'''metrics=("bleu1","meteor","rougeL","em","f1"); lines=["| System | BLEU-1 | METEOR | ROUGE-L | EM | F1 |","|---|---:|---:|---:|---:|---:|"]
for name in ("BaseTop1","LinearReranker","LambdaMART","Top10Oracle"):
    v=summary["scores"][name]; lines.append(f"| {name} | "+" | ".join(f"{v[m]:.2f}" for m in metrics)+" |")
display(Markdown("## Validation results\n\n"+"\n".join(lines)))
for name in ("LinearReranker","LambdaMART"):
    d=summary["delta_vs_base"][name]; display(Markdown(f"**{name} − base:** "+", ".join(f"{m} {d[m]:+.2f}" for m in metrics)))
'''

BOOTSTRAP = r'''import numpy as np
rng=np.random.default_rng(4012); lines=["| Model vs base | Metric | Mean paired Δ | 95% bootstrap CI | P(Δ≤0) | Win / tie / loss |","|---|---|---:|---:|---:|---:|"]
for prefix,label in (("linear","Linear"),("tree","LambdaMART")):
 for metric in ("rougeL","f1","em"):
    d=np.array([float(r[f"{prefix}_{metric}"]) for r in rows]); means=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(10000)])*100
    lo,hi=np.quantile(means,[.025,.975]); lines.append(f"| {label} | {metric} | {d.mean()*100:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {(means<=0).mean():.3f} | {(d>1e-12).sum()} / {(abs(d)<=1e-12).sum()} / {(d < -1e-12).sum()} |")
display(Markdown("## Paired bootstrap\n\n"+"\n".join(lines)))
'''

SLICES = r'''for field,title in (("question_type","Question type"),("silver_region","Silver-span region")):
    groups=collections.defaultdict(list)
    for r in rows: groups[r[field]].append(r)
    lines=[f"| {title} | N | Linear ΔROUGE-L | LambdaMART ΔROUGE-L |","|---|---:|---:|---:|"]
    for name,v in sorted(groups.items()): lines.append(f"| {name} | {len(v)} | {sum(float(x['linear_rougeL']) for x in v)*100/len(v):+.2f} | {sum(float(x['tree_rougeL']) for x in v)*100/len(v):+.2f} |")
    display(Markdown(f"## {title} effects\n\n"+"\n".join(lines)))
for model in ("LinearReranker","LambdaMART"):
 rank_counts={int(k):v for k,v in summary["selected_rank_counts"][model].items()}; total=sum(rank_counts.values())
 display(Markdown(f"## {model} selected base ranks\n\n"+"\n".join(["| Rank | Count | Share |","|---:|---:|---:|"]+[f"| {k} | {rank_counts.get(k,0)} | {100*rank_counts.get(k,0)/total:.1f}% |" for k in range(1,11)])))
'''

WEIGHTS = r'''lines=["| Feature | Standardized coefficient |","|---|---:|"]+[f"| {name} | {weight:+.3f} |" for name,weight in summary["feature_weights"]]
display(Markdown("## Linear model weights\n\n"+"\n".join(lines)+"\n\nCoefficients are in standardized pair-difference space; magnitude indicates influence, not causality."))
lines=["| Feature | LambdaMART split importance |","|---|---:|"]+[f"| {name} | {weight} |" for name,weight in summary["tree_importances"]]
display(Markdown("## LambdaMART feature importances\n\n"+"\n".join(lines)+"\n\nSplit counts indicate usage, not causal effect."))
'''

CONCLUSION = r'''best=max(("LinearReranker","LambdaMART"),key=lambda n:summary["scores"][n]["rougeL"]); d=summary["delta_vs_base"][best]; gain=d["rougeL"]
decision="supports" if gain>0 else "does not support"
text=f"""# Linear Top-10 span reranker

The stronger fixed reranker was **{best}**. It changed validation ROUGE-L by **{gain:+.2f}**, F1 by **{d['f1']:+.2f}**,
and EM by **{d['em']:+.2f}** versus the unchanged Top-1 decoder. This result **{decision}** reranking
with the current feature set. The reranker was fitted only from training-split labels; validation labels
were used only for evaluation. The locked test set was not accessed.
"""
Path("linear_span_reranker_report.md").write_text(text,encoding="utf-8"); display(Markdown("## Report-ready conclusion\n\n"+text)); zf.close()
'''

def notebook(cells,gpu=False):
    meta={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"colab":{"provenance":[]}}
    if gpu: meta["accelerator"]="GPU"; meta["colab"]["gpuType"]="T4"
    return {"cells":cells,"metadata":meta,"nbformat":4,"nbformat_minor":5}

def build():
    train=[md("""# Lightweight linear Top-10 span reranker

This notebook restores the fixed seed-1 CurrentGated checkpoint, generates Top-10 spans, and fits two fixed lightweight rankers on the training split: pairwise logistic regression and shallow LambdaMART. There is no validation-driven hyperparameter search. Validation labels are used only once for evaluation, and the locked test set is not loaded. Candidate generation is cached in the separate Drive folder `CITS4012_A2_reranker`."""),md("## 1. Runtime, checkpoint archive, and Drive"),code(SETUP),md("## 2. Upload train.csv"),code(DATA),md("## 3. Embedded source")]
    for rel in FILES: train.append(code(f"%%writefile {rel}\n"+(ROOT/rel).read_text(encoding="utf-8")))
    train += [md("## 4. Restore model and fixed split"),code(LOAD),md("## 5. Generate or restore Top-10 candidate caches"),code(CANDIDATES),md("## 6. Fit the fixed linear pairwise reranker"),code(FIT),md("## 7. Evaluate once on validation"),code(EVALUATE),md("## 8. Package"),code(PACKAGE)]
    report=[md("""# Linear span-reranker report

Upload or locate `linear_span_reranker_artifacts.zip`. This notebook performs no training or inference and does not require the dataset."""),md("## 0. Load and validate artifact"),code(REPORT_LOAD),code(TABLE),code(BOOTSTRAP),code(SLICES),code(WEIGHTS),code(CONCLUSION)]
    TRAIN_OUT.write_text(json.dumps(notebook(train,True),ensure_ascii=False,indent=1),encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(notebook(report),ensure_ascii=False,indent=1),encoding="utf-8")
    print("wrote",TRAIN_OUT.name,len(train),"cells"); print("wrote",REPORT_OUT.name,len(report),"cells")

if __name__=="__main__": build()
