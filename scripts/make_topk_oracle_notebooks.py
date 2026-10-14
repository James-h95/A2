"""Build a no-training top-K span oracle analysis and offline report notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_OUT = ROOT / "step08a_colab_topk_span_oracle.ipynb"
REPORT_OUT = ROOT / "step08b_topk_span_oracle_report.ipynb"
FILES = ["src/__init__.py", "src/evaluate.py", "src/data.py", "src/runlog.py",
         "src/model.py", "src/fasttext_init.py", "src/train.py"]

def md(text): return {"cell_type":"markdown","metadata":{},"source":text.splitlines(True)}
def code(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":text.splitlines(True)}

SETUP = r'''import os,sys,subprocess,shutil,json,csv,zipfile,hashlib,io,collections,math
from pathlib import Path
assert "google.colab" in sys.modules
subprocess.run([sys.executable,"-m","pip","install","-q","gensim","emoji","nltk"],check=True)
import torch
assert torch.cuda.is_available(); DEVICE="cuda"; print("GPU",torch.cuda.get_device_name(0),"torch",torch.__version__)
WORK=Path("/content/A2_topk_oracle"); OUT=WORK/"topk_oracle_results"
for p in (WORK/"src",WORK/"data",OUT): p.mkdir(parents=True,exist_ok=True)
os.chdir(WORK)
from google.colab import drive
drive.mount("/content/drive")
BACKUP=Path("/content/drive/MyDrive/CITS4012_A2_oracle"); BACKUP.mkdir(parents=True,exist_ok=True)
SOURCE_NAME="attention_variants_artifacts.zip"
source_paths=[Path("/content")/SOURCE_NAME,Path("/content/drive/MyDrive/CITS4012_A2_attention_variants")/SOURCE_NAME]
SOURCE=next((p for p in source_paths if p.exists()),None)
if SOURCE is None:
    from google.colab import files
    up=files.upload(); assert SOURCE_NAME in up; SOURCE=Path("/content")/SOURCE_NAME; SOURCE.write_bytes(up[SOURCE_NAME])
with zipfile.ZipFile(SOURCE) as z:
    assert z.testzip() is None
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

LOAD_MODEL = r'''from src import data as D
from src.train import load_checkpoint,to_device
from src.evaluate import score_example
with zipfile.ZipFile(SOURCE) as z:
    runs=list(csv.DictReader(io.StringIO(z.read("attention_variant_results/runs.csv").decode("utf-8-sig"))))
    row=next(r for r in runs if r["model"]=="CurrentGated" and r["seed"]=="1")
    RUN_ID=row["run_id"]; ckpt_member=f"attention_variant_checkpoints/{RUN_ID}.pt"
    ckpt_path=WORK/f"{RUN_ID}.pt"; ckpt_path.write_bytes(z.read(ckpt_member))
    saved_preds={int(x["row_id"]):x["pred"] for x in json.loads(z.read(f"attention_variant_results/preds/{RUN_ID}.json"))}
model,cfg,wv,cv=load_checkpoint(ckpt_path,DEVICE)
_,val_rows=D.get_splits(); prep=dict(use_normaliser=cfg["use_normaliser"],max_span_len=cfg["max_span_len"],tau=cfg["tau"],soft_tau=cfg.get("soft_tau"))
val=D.preprocess(val_rows,prep); D.encode(val,wv,cv)
print("MODEL",RUN_ID,"val",len(val),"max_span_len",cfg["max_span_len"])
'''

INFER = r'''KS=(1,3,5,10,20); MAX_K=max(KS); records=[]
def region(ex,i,j):
    b=ex["byline_from"]
    if b is None or j<b: return "body"
    if i>=b: return "byline"
    return "cross"
model.eval()
with torch.no_grad():
    for batch in D.make_loader(val,batch_size=64):
        meta=batch["meta"]; dev=to_device(batch,DEVICE); out=model(dev)
        ls=torch.log_softmax(out["start_logits"],-1); le=torch.log_softmax(out["end_logits"],-1)
        for b,ex in enumerate(meta):
            T=len(ex["t_tokens"]); candidates=[]
            for i in range(T):
                for j in range(i,min(T,i+cfg["max_span_len"])):
                    text=D.span_text(ex,i,j); scores=score_example(text,ex["answers"],("rougeL","f1","em"))
                    candidates.append(dict(start=i,end=j,text=text,score=float(ls[b,i]+le[b,j]),region=region(ex,i,j),**scores))
            candidates.sort(key=lambda x:x["score"],reverse=True); candidates=candidates[:MAX_K]
            records.append(dict(row_id=ex["row_id"],qid=ex["qid"],question_type=D._question_word(ex["question"]),
                                byline_present=ex["byline_from"] is not None,
                                silver_region=region(ex,*ex["best"]) if ex["best"] is not None else "no_span",
                                candidates=candidates))
assert len(records)==len(val)
top1={r["row_id"]:r["candidates"][0]["text"] for r in records}
agreement=sum(top1[k]==v for k,v in saved_preds.items())/len(saved_preds)
assert agreement==1.0,agreement
with (OUT/"topk_candidates.jsonl").open("w",encoding="utf-8") as f:
    for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("INFERENCE COMPLETE",len(records),"top-1 agreement",agreement)
'''

SUMMARISE = r'''metrics=("rougeL","f1","em"); summary={"source_run":RUN_ID,"n":len(records),"ks":{},"headroom_vs_top1":{},"slices_k10":{}}
detail=[]
for r in records:
    row={k:r[k] for k in ("row_id","qid","question_type","byline_present","silver_region")}
    for K in KS:
        pool=r["candidates"][:K]
        for m in metrics: row[f"{m}@{K}"]=100*max(x[m] for x in pool)
        row[f"exact_found@{K}"]=int(any(x["em"]==1 for x in pool))
        best=max(pool,key=lambda x:(x["rougeL"],x["f1"],x["em"],x["score"]))
        row[f"oracle_region@{K}"]=best["region"]
    detail.append(row)
for K in KS:
    summary["ks"][str(K)]={m:sum(r[f"{m}@{K}"] for r in detail)/len(detail) for m in metrics}
    summary["ks"][str(K)]["exact_coverage"]=100*sum(r[f"exact_found@{K}"] for r in detail)/len(detail)
for K in KS[1:]:
    summary["headroom_vs_top1"][str(K)]={m:summary["ks"][str(K)][m]-summary["ks"]["1"][m] for m in metrics}
for field in ("question_type","silver_region"):
    groups=collections.defaultdict(list)
    for r in detail: groups[str(r[field])].append(r)
    summary["slices_k10"][field]={}
    for name,items in groups.items():
        summary["slices_k10"][field][name]={"n":len(items),**{m:sum(x[f"{m}@10"]-x[f"{m}@1"] for x in items)/len(items) for m in metrics}}
fields=list(detail[0])
with (OUT/"oracle_details.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(detail)
(OUT/"oracle_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
'''

PACKAGE = r'''archive=Path("/content/topk_span_oracle_artifacts.zip")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file(): z.write(p,p.relative_to(WORK))
shutil.copy2(archive,BACKUP/archive.name); print("SAVED",BACKUP/archive.name,f"{archive.stat().st_size/1e6:.1f} MB")
'''

REPORT_LOAD = r'''import json,csv,io,zipfile,sys,collections
from pathlib import Path
from IPython.display import display,Markdown
NAME="topk_span_oracle_artifacts.zip"
paths=[Path(NAME),Path("/content")/NAME,Path("/content/drive/MyDrive/CITS4012_A2_oracle")/NAME]
archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import drive; drive.mount("/content/drive"); archive=next((p for p in paths if p.exists()),None)
if archive is None and "google.colab" in sys.modules:
    from google.colab import files
    up=files.upload(); assert NAME in up; archive=Path("/content")/NAME; archive.write_bytes(up[NAME])
assert archive is not None
zf=zipfile.ZipFile(archive); assert zf.testzip() is None
summary=json.loads(zf.read("topk_oracle_results/oracle_summary.json")); details=list(csv.DictReader(io.StringIO(zf.read("topk_oracle_results/oracle_details.csv").decode())))
print("Loaded",archive,"source",summary["source_run"],"N",summary["n"],"— no training or inference")
'''

MAIN_TABLE = r'''lines=["| K | Oracle ROUGE-L | Oracle F1 | Oracle EM | Exact-answer coverage | ΔROUGE-L vs Top-1 |","|---:|---:|---:|---:|---:|---:|"]
for K in summary["ks"]:
    v=summary["ks"][K]; gain=v["rougeL"]-summary["ks"]["1"]["rougeL"]
    lines.append(f"| {K} | {v['rougeL']:.2f} | {v['f1']:.2f} | {v['em']:.2f} | {v['exact_coverage']:.2f}% | {gain:+.2f} |")
display(Markdown("## Top-K oracle ceiling\n\n"+"\n".join(lines)+"\n\nEach oracle metric independently selects the best candidate using gold answers; these are ceilings, not deployable scores."))
'''

SLICES = r'''for field,title in (("question_type","Question type"),("silver_region","Best silver-span region")):
    lines=[f"| {title} | N | Top-10 ΔROUGE-L | Top-10 ΔF1 | Top-10 ΔEM |","|---|---:|---:|---:|---:|"]
    for name,v in sorted(summary["slices_k10"][field].items()):
        lines.append(f"| {name} | {v['n']} | {v['rougeL']:+.2f} | {v['f1']:+.2f} | {v['em']:+.2f} |")
    display(Markdown(f"## {title} headroom\n\n"+"\n".join(lines)))
'''

RANKS = r'''# How deep must a reranker look to recover each example's best Top-20 ROUGE-L candidate?
records=[json.loads(line) for line in zf.read("topk_oracle_results/topk_candidates.jsonl").decode().splitlines()]
ranks=[]
for r in records:
    vals=[x["rougeL"] for x in r["candidates"]]; best=max(vals); ranks.append(next(i+1 for i,v in enumerate(vals) if v==best))
counts=collections.Counter(ranks); cumulative=0; lines=["| Rank needed | Examples whose Top-20 best first appears here | Cumulative |","|---:|---:|---:|"]
for rank in range(1,21):
    cumulative+=counts[rank]; lines.append(f"| {rank} | {counts[rank]} | {100*cumulative/len(ranks):.1f}% |")
display(Markdown("## Rank of the best available candidate\n\n"+"\n".join(lines)))
'''

CONCLUSION = r'''top1=summary["ks"]["1"]; top10=summary["ks"]["10"]; gain=top10["rougeL"]-top1["rougeL"]
decision="substantial" if gain>=3 else "limited" if gain<1 else "moderate"
text=f"""# Top-K span oracle analysis

For the current gated model, Top-1 ROUGE-L is **{top1['rougeL']:.2f}** and the Top-10 oracle ceiling is
**{top10['rougeL']:.2f}**, a **{gain:+.2f}** point gain. This indicates **{decision}** reranking headroom.
Top-10 exact-answer coverage is **{top10['exact_coverage']:.2f}%**. The oracle uses validation gold answers
only to measure candidate availability; it is not a deployable reranker. The locked test set was not accessed.
"""
Path("topk_span_oracle_report.md").write_text(text,encoding="utf-8"); display(Markdown("## Report-ready conclusion\n\n"+text)); zf.close()
'''

def notebook(cells,gpu=False):
    meta={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"},"colab":{"provenance":[]}}
    if gpu: meta["accelerator"]="GPU"; meta["colab"]["gpuType"]="T4"
    return {"cells":cells,"metadata":meta,"nbformat":4,"nbformat_minor":5}

def build():
    analysis=[md("""# Top-K span oracle analysis

This notebook reuses the saved seed-1 `CurrentGated` checkpoint from `attention_variants_artifacts.zip`. It does no training. It enumerates the model's top 20 legal spans and measures the best answer available within Top-1/3/5/10/20. Gold answers are used only for validation oracle measurement. The locked test set is not loaded. Outputs go to the new Drive folder `CITS4012_A2_oracle`."""),md("## 1. Runtime, source archive, and Drive"),code(SETUP),md("## 2. Upload train.csv to reconstruct the fixed validation split"),code(DATA),md("## 3. Embedded source")]
    for rel in FILES: analysis.append(code(f"%%writefile {rel}\n"+(ROOT/rel).read_text(encoding="utf-8")))
    analysis += [md("## 4. Restore CurrentGated checkpoint"),code(LOAD_MODEL),md("## 5. Enumerate Top-20 spans"),code(INFER),md("## 6. Oracle metrics and slices"),code(SUMMARISE),md("## 7. Package for download"),code(PACKAGE)]
    report=[md("""# Top-K span oracle report

Upload or locate `topk_span_oracle_artifacts.zip`. This notebook performs no training or inference and does not require the dataset."""),md("## 0. Load and validate artifact"),code(REPORT_LOAD),code(MAIN_TABLE),code(SLICES),code(RANKS),code(CONCLUSION)]
    ANALYSIS_OUT.write_text(json.dumps(notebook(analysis,True),ensure_ascii=False,indent=1),encoding="utf-8")
    REPORT_OUT.write_text(json.dumps(notebook(report),ensure_ascii=False,indent=1),encoding="utf-8")
    print("wrote",ANALYSIS_OUT.name,len(analysis),"cells"); print("wrote",REPORT_OUT.name,len(report),"cells")

if __name__=="__main__": build()
