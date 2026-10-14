"""Build a leakage-controlled 3-fold OOF reranker experiment and report."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_OUT = ROOT / "colab_oof_span_reranker.ipynb"
REPORT_OUT = ROOT / "oof_span_reranker_report.ipynb"
FILES = ["src/__init__.py", "src/evaluate.py", "src/data.py", "src/runlog.py",
         "src/model.py", "src/fasttext_init.py", "src/train.py", "src/rerank.py"]

def md(text): return {"cell_type":"markdown","metadata":{},"source":text.splitlines(True)}
def code(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(True)}

SETUP = r'''import os,sys,subprocess,shutil,json,csv,zipfile,hashlib,io,collections,statistics,time
from pathlib import Path
assert "google.colab" in sys.modules
subprocess.run([sys.executable,"-m","pip","install","-q","gensim","emoji","nltk","scikit-learn","joblib","lightgbm"],check=True)
import torch,numpy as np,joblib
assert torch.cuda.is_available(); DEVICE="cuda"; print("GPU",torch.cuda.get_device_name(0),"torch",torch.__version__)
WORK=Path("/content/A2_oof_reranker"); RESULTS=WORK/"oof_base_results"; CKPT=WORK/"oof_base_checkpoints"; OUT=WORK/"oof_reranker_results"
for p in (WORK/"src",WORK/"data",RESULTS,CKPT,OUT): p.mkdir(parents=True,exist_ok=True)
os.environ["A2_RESULTS"]=str(RESULTS); os.environ["A2_CKPT"]=str(CKPT); os.chdir(WORK)
from google.colab import drive
drive.mount("/content/drive")
BACKUP=Path("/content/drive/MyDrive/CITS4012_A2_reranker_oof"); BACKUP.mkdir(parents=True,exist_ok=True)
for name,dst in (("oof_base_results",RESULTS),("oof_base_checkpoints",CKPT),("oof_reranker_results",OUT)):
    if (BACKUP/name).exists(): shutil.copytree(BACKUP/name,dst,dirs_exist_ok=True)
def backup():
    for src,name in ((RESULTS,"oof_base_results"),(CKPT,"oof_base_checkpoints"),(OUT,"oof_reranker_results")):
        if src.exists(): shutil.copytree(src,BACKUP/name,dirs_exist_ok=True)
SOURCE_NAME="attention_variants_artifacts.zip"
paths=[Path("/content")/SOURCE_NAME,Path("/content/drive/MyDrive/CITS4012_A2_attention_variants")/SOURCE_NAME]
SOURCE=next((p for p in paths if p.exists()),None)
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

LOAD = r'''from sklearn.model_selection import StratifiedGroupKFold,GroupShuffleSplit
from src import data as D
from src import train as T
from src.evaluate import evaluate,score_example
from src.rerank import (topk_candidates,fit_pairwise_reranker,fit_lambdamart_reranker,
                        rerank,linear_weights,tree_importances,span_region)
with zipfile.ZipFile(SOURCE) as z:
    source_runs=list(csv.DictReader(io.StringIO(z.read("attention_variant_results/runs.csv").decode("utf-8-sig"))))
    source_run=next(r for r in source_runs if r["model"]=="CurrentGated" and r["seed"]=="1")
    SOURCE_RUN_ID=source_run["run_id"]; full_ckpt=WORK/f"{SOURCE_RUN_ID}.pt"
    full_ckpt.write_bytes(z.read(f"attention_variant_checkpoints/{SOURCE_RUN_ID}.pt"))
full_model,full_cfg,full_wv,full_cv=T.load_checkpoint(full_ckpt,DEVICE)
train_rows,val_rows=D.get_splits(); print("SOURCE",SOURCE_RUN_ID,"train",len(train_rows),"val",len(val_rows))
'''

FOLDS = r'''# Split by tweet, stratified approximately by question type. No tweet crosses folds.
labels=np.array([D._question_word(r["question"]) for r in train_rows]); groups=np.array([r["tweet"] for r in train_rows],dtype=object)
splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=4012)
folds=[]
for fold,(fit_idx,hold_idx) in enumerate(splitter.split(np.zeros(len(train_rows)),labels,groups),1):
    fit_rows=[train_rows[i] for i in fit_idx]; hold_rows=[train_rows[i] for i in hold_idx]
    assert not ({r["tweet"] for r in fit_rows}&{r["tweet"] for r in hold_rows})
    folds.append((fit_rows,hold_rows))
    print("fold",fold,"outer fit",len(fit_rows),"OOF holdout",len(hold_rows),"tweets",len({r['tweet'] for r in hold_rows}))
assert sorted(r["row_id"] for _,h in folds for r in h)==sorted(r["row_id"] for r in train_rows)
'''

OOF = r'''def save_jsonl(path,records):
    with path.open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
def load_jsonl(path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
def completed_run(name):
    p=RESULTS/"runs.csv"
    if not p.exists(): return None
    found=[r for r in csv.DictReader(p.open(encoding="utf-8")) if r["model"]==name]
    return found[-1] if found else None
oof_candidates=[]; fold_manifest=[]
for fold,(outer_fit,holdout) in enumerate(folds,1):
    cache=BACKUP/f"fold{fold}_oof_top10.jsonl"; name=f"OOFBaseFold{fold}"
    if cache.exists():
        candidates=load_jsonl(cache); print("CACHE",name,len(candidates))
    else:
        # Inner dev comes only from outer_fit and is grouped by tweet. The OOF holdout remains unseen.
        inner_groups=np.array([r["tweet"] for r in outer_fit],dtype=object)
        inner_split=GroupShuffleSplit(n_splits=1,test_size=0.10,random_state=5000+fold)
        inner_train_idx,inner_dev_idx=next(inner_split.split(np.zeros(len(outer_fit)),groups=inner_groups))
        inner_train=[outer_fit[i] for i in inner_train_idx]; inner_dev=[outer_fit[i] for i in inner_dev_idx]
        assert not ({r["tweet"] for r in inner_train}&{r["tweet"] for r in inner_dev})
        existing=completed_run(name)
        if existing and (CKPT/f"{existing['run_id']}.pt").exists():
            fold_model,fold_cfg,wv,cv=T.load_checkpoint(CKPT/f"{existing['run_id']}.pt",DEVICE); run_id=existing["run_id"]
            print("RESTORED",run_id)
        else:
            prepared=T.prepare_data(full_cfg,train_rows=inner_train,val_rows=inner_dev)
            result=T.train(full_cfg,seed=1,name=name,data=prepared,save=True,log=lambda s:print(s,flush=True))
            run_id=result["run_id"]; fold_model,fold_cfg,wv,cv=T.load_checkpoint(result["ckpt"],DEVICE); backup()
        prep=dict(use_normaliser=fold_cfg["use_normaliser"],max_span_len=fold_cfg["max_span_len"],tau=fold_cfg["tau"],soft_tau=fold_cfg.get("soft_tau"))
        hold=D.preprocess(holdout,prep); D.encode(hold,wv,cv)
        candidates=topk_candidates(fold_model,hold,DEVICE,top_k=10,batch_size=64,max_span_len=fold_cfg["max_span_len"])
        save_jsonl(cache,candidates); backup(); print("GENERATED",name,len(candidates),cache)
        del fold_model; torch.cuda.empty_cache()
    assert {r["row_id"] for r in candidates}=={r["row_id"] for r in holdout}
    oof_candidates.extend(candidates); fold_manifest.append(dict(fold=fold,n=len(candidates)))
assert len(oof_candidates)==len(train_rows)==len({r["row_id"] for r in oof_candidates})
oof_candidates.sort(key=lambda r:r["row_id"]); print("OOF COMPLETE",len(oof_candidates))
'''

VAL = r'''prep=dict(use_normaliser=full_cfg["use_normaliser"],max_span_len=full_cfg["max_span_len"],tau=full_cfg["tau"],soft_tau=full_cfg.get("soft_tau"))
train=D.preprocess(train_rows,prep); val=D.preprocess(val_rows,prep); D.encode(train,full_wv,full_cv); D.encode(val,full_wv,full_cv)
val_cache=BACKUP/"val_top10_candidates.jsonl"
if val_cache.exists(): val_candidates=load_jsonl(val_cache); print("CACHE val",len(val_candidates))
else:
    val_candidates=topk_candidates(full_model,val,DEVICE,top_k=10,batch_size=64,max_span_len=full_cfg["max_span_len"])
    save_jsonl(val_cache,val_candidates); print("GENERATED val",len(val_candidates))
assert len(val_candidates)==len(val)
'''

FIT = r'''# Fixed settings; no validation-driven tuning.
linear_model,linear_fit=fit_pairwise_reranker(train,oof_candidates,regularization=1.0)
tree_model,tree_fit=fit_lambdamart_reranker(train,oof_candidates)
joblib.dump(linear_model,OUT/"oof_linear_reranker.joblib"); joblib.dump(tree_model,OUT/"oof_lambdamart_reranker.joblib")
weights=linear_weights(linear_model); importances=tree_importances(tree_model)
print("LINEAR",linear_fit); print("LAMBDAMART",tree_fit); print("TOP FEATURES",importances[:10])
'''

EVALUATE = r'''golds=[ex["answers"] for ex in val]; base_preds=[r["candidates"][0]["text"] for r in val_candidates]
linear_preds,linear_sel=rerank(linear_model,val,val_candidates); tree_preds,tree_sel=rerank(tree_model,val,val_candidates)
oracle_preds=[max(r["candidates"],key=lambda c:score_example(c["text"],ex["answers"],("rougeL",))["rougeL"])["text"] for ex,r in zip(val,val_candidates)]
scores={"BaseTop1":evaluate(base_preds,golds),"OOFLinear":evaluate(linear_preds,golds),"OOFLambdaMART":evaluate(tree_preds,golds),"Top10Oracle":evaluate(oracle_preds,golds)}
summary={"source_run":SOURCE_RUN_ID,"folds":fold_manifest,"top_k":10,"scores":scores,
 "delta_vs_base":{n:{m:scores[n][m]-scores["BaseTop1"][m] for m in scores["BaseTop1"]} for n in ("OOFLinear","OOFLambdaMART")},
 "fit":{"OOFLinear":linear_fit,"OOFLambdaMART":tree_fit},"linear_weights":weights,"tree_importances":importances,
 "selected_rank_counts":{n:dict(collections.Counter(str(x["selected_rank"]) for x in sel)) for n,sel in (("OOFLinear",linear_sel),("OOFLambdaMART",tree_sel))}}
rows=[]
for ex,bp,lp,tp,ls,ts in zip(val,base_preds,linear_preds,tree_preds,linear_sel,tree_sel):
    row={"row_id":ex["row_id"],"qid":ex["qid"],"question_type":D._question_word(ex["question"]),"silver_region":span_region(ex,*ex["best"]) if ex["best"] is not None else "no_span","linear_rank":ls["selected_rank"],"tree_rank":ts["selected_rank"]}
    for prefix,pred in (("linear",lp),("tree",tp)):
        for m in ("rougeL","f1","em"): row[f"{prefix}_{m}"]=score_example(pred,ex["answers"],(m,))[m]-score_example(bp,ex["answers"],(m,))[m]
    rows.append(row)
with (OUT/"paired_deltas.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(OUT/"oof_reranker_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
(OUT/"linear_weights.json").write_text(json.dumps(weights,indent=2),encoding="utf-8"); (OUT/"tree_importances.json").write_text(json.dumps(importances,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2)); backup()
'''

PACKAGE = r'''archive=Path("/content/oof_span_reranker_artifacts.zip")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file(): z.write(p,p.relative_to(WORK))
shutil.copy2(archive,BACKUP/archive.name); print("SAVED",BACKUP/archive.name,f"{archive.stat().st_size/1e6:.1f} MB")
'''

REPORT_LOAD = r'''import json,csv,io,zipfile,sys,collections
from pathlib import Path
from IPython.display import display,Markdown
NAME="oof_span_reranker_artifacts.zip"; paths=[Path(NAME),Path("/content")/NAME,Path("/content/drive/MyDrive/CITS4012_A2_reranker_oof")/NAME]
archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
 from google.colab import drive; drive.mount("/content/drive"); archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
 from google.colab import files; up=files.upload(); assert NAME in up; archive=Path("/content")/NAME; archive.write_bytes(up[NAME])
assert archive is not None
zf=zipfile.ZipFile(archive); assert zf.testzip() is None
summary=json.loads(zf.read("oof_reranker_results/oof_reranker_summary.json")); rows=list(csv.DictReader(io.StringIO(zf.read("oof_reranker_results/paired_deltas.csv").decode())))
print("Loaded",archive,"source",summary["source_run"],"— no training or inference")
'''

TABLE = r'''metrics=("bleu1","meteor","rougeL","em","f1"); lines=["| System | BLEU-1 | METEOR | ROUGE-L | EM | F1 |","|---|---:|---:|---:|---:|---:|"]
for name in ("BaseTop1","OOFLinear","OOFLambdaMART","Top10Oracle"):
 v=summary["scores"][name]; lines.append(f"| {name} | "+" | ".join(f"{v[m]:.2f}" for m in metrics)+" |")
display(Markdown("## Validation results\n\n"+"\n".join(lines)))
'''

BOOTSTRAP = r'''import numpy as np
rng=np.random.default_rng(4012); lines=["| Model vs base | Metric | Mean Δ | 95% bootstrap CI | P(Δ≤0) | Win / tie / loss |","|---|---|---:|---:|---:|---:|"]
for prefix,label in (("linear","OOF Linear"),("tree","OOF LambdaMART")):
 for metric in ("rougeL","f1","em"):
  d=np.array([float(r[f"{prefix}_{metric}"]) for r in rows]); means=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(10000)])*100; lo,hi=np.quantile(means,[.025,.975])
  lines.append(f"| {label} | {metric} | {d.mean()*100:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {(means<=0).mean():.3f} | {(d>1e-12).sum()} / {(abs(d)<=1e-12).sum()} / {(d< -1e-12).sum()} |")
display(Markdown("## Paired bootstrap\n\n"+"\n".join(lines)))
'''

SLICES = r'''for field,title in (("question_type","Question type"),("silver_region","Silver-span region")):
 groups=collections.defaultdict(list)
 for r in rows: groups[r[field]].append(r)
 lines=[f"| {title} | N | OOF Linear ΔRL | OOF LambdaMART ΔRL |","|---|---:|---:|---:|"]
 for name,v in sorted(groups.items()): lines.append(f"| {name} | {len(v)} | {sum(float(x['linear_rougeL']) for x in v)*100/len(v):+.2f} | {sum(float(x['tree_rougeL']) for x in v)*100/len(v):+.2f} |")
 display(Markdown(f"## {title} effects\n\n"+"\n".join(lines)))
'''

DIAGNOSTICS = r'''for model in ("OOFLinear","OOFLambdaMART"):
 counts={int(k):v for k,v in summary["selected_rank_counts"][model].items()}; total=sum(counts.values())
 display(Markdown(f"## {model} selected ranks\n\n"+"\n".join(["| Base rank | Count | Share |","|---:|---:|---:|"]+[f"| {k} | {counts.get(k,0)} | {100*counts.get(k,0)/total:.1f}% |" for k in range(1,11)])))
lines=["| Feature | OOF LambdaMART importance |","|---|---:|"]+[f"| {n} | {v} |" for n,v in summary["tree_importances"]]
display(Markdown("## Feature importance\n\n"+"\n".join(lines)))
'''

CONCLUSION = r'''best=max(("OOFLinear","OOFLambdaMART"),key=lambda n:summary["scores"][n]["rougeL"]); d=summary["delta_vs_base"][best]
text=f"""# Three-fold OOF span reranker

The stronger OOF reranker was **{best}**, changing validation ROUGE-L by **{d['rougeL']:+.2f}**,
F1 by **{d['f1']:+.2f}**, and EM by **{d['em']:+.2f}** versus unchanged Top-1 decoding.
Every reranker-training candidate was generated by a base model that had not seen that question or tweet.
Validation labels were used only for evaluation, and the locked test set was not accessed.
"""
Path("oof_span_reranker_report.md").write_text(text,encoding="utf-8"); display(Markdown("## Report-ready conclusion\n\n"+text)); zf.close()
'''

def notebook(cells,gpu=False):
    meta={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"colab":{"provenance":[]}}
    if gpu: meta["accelerator"]="GPU"; meta["colab"]["gpuType"]="T4"
    return {"cells":cells,"metadata":meta,"nbformat":4,"nbformat_minor":5}

def build():
    train=[md("""# Three-fold out-of-fold Top-10 reranker

This notebook tests whether the weak reranker result came from in-sample candidate generation. The original training split is divided into three tweet-grouped folds. Each OOF candidate set is produced by a base model that never saw that fold; early stopping uses a separate inner-dev drawn only from the other two folds. Fixed validation remains untouched until final evaluation. Linear and shallow LambdaMART settings are identical to the previous experiment, with no validation tuning. Intermediate checkpoints and candidate files are resumable in `CITS4012_A2_reranker_oof`."""),md("## 1. Runtime, Drive, and source checkpoint"),code(SETUP),md("## 2. Upload train.csv"),code(DATA),md("## 3. Embedded source")]
    for rel in FILES: train.append(code(f"%%writefile {rel}\n"+(ROOT/rel).read_text(encoding="utf-8")))
    train += [md("## 4. Load the fixed split and final base checkpoint"),code(LOAD),md("## 5. Define tweet-grouped outer folds"),code(FOLDS),md("## 6. Train or restore fold models and generate OOF candidates"),code(OOF),md("## 7. Generate fixed-validation candidates"),code(VAL),md("## 8. Fit unchanged rerankers on OOF candidates"),code(FIT),md("## 9. Evaluate once on fixed validation"),code(EVALUATE),md("## 10. Package"),code(PACKAGE)]
    report=[md("""# OOF reranker report

Upload or locate `oof_span_reranker_artifacts.zip`. This notebook performs no training or inference and does not require the dataset."""),md("## 0. Load"),code(REPORT_LOAD),code(TABLE),code(BOOTSTRAP),code(SLICES),code(DIAGNOSTICS),code(CONCLUSION)]
    TRAIN_OUT.write_text(json.dumps(notebook(train,True),ensure_ascii=False,indent=1),encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(notebook(report),ensure_ascii=False,indent=1),encoding="utf-8")
    print("wrote",TRAIN_OUT.name,len(train),"cells"); print("wrote",REPORT_OUT.name,len(report),"cells")

if __name__=="__main__": build()
