"""Build ordered confirmation, ensemble and OOF feature experiments."""
import json
from pathlib import Path
import make_oof_reranker_notebooks as O

ROOT = O.ROOT

SETUP = r'''import os, sys, subprocess, json, zipfile, io, csv, shutil, hashlib
from pathlib import Path
assert "google.colab" in sys.modules
subprocess.run([sys.executable,"-m","pip","install","-q","gensim","emoji","nltk","scikit-learn","joblib","lightgbm"],check=True)
import torch, numpy as np, joblib
from google.colab import drive, files
drive.mount('/content/drive')
WORK=Path('/content/A2_reranker_next'); WORK.mkdir(exist_ok=True); os.chdir(WORK)
sys.path.insert(0,str(WORK))
OUT=WORK/'next_results'; OUT.mkdir(exist_ok=True)
BACKUP=Path('/content/drive/MyDrive/CITS4012_A2_reranker_next'); BACKUP.mkdir(parents=True,exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
def locate(name, folder):
    paths=[Path('/content')/name,Path('/content/drive/MyDrive')/folder/name]
    found=next((p for p in paths if p.exists()),None)
    if found is None:
        print('Upload',name); up=files.upload(); assert name in up
        found=WORK/name; found.write_bytes(up[name])
    return found
SOURCE=locate('attention_variants_artifacts.zip','CITS4012_A2_attention_variants')
V2=locate('a2_v2_colab_artifacts.zip','CITS4012_A2_v2')
OOF=locate('oof_span_reranker_artifacts.zip','CITS4012_A2_reranker_oof')
Path('src').mkdir(exist_ok=True); Path('data').mkdir(exist_ok=True)
results={}
def save():
    (OUT/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    archive=Path(shutil.make_archive(str(WORK/'reranker_next_artifacts'),'zip',OUT))
    shutil.copy2(archive,BACKUP/archive.name)
    print('Saved',BACKUP/archive.name)
'''

LOAD = r'''from src import data as D, train as T, rerank as R
from src.evaluate import evaluate, score_example
with zipfile.ZipFile(OOF) as z:
    prior=json.loads(z.read('oof_reranker_results/oof_reranker_summary.json'))
    model_path=WORK/'fixed_oof_tree.joblib'; model_path.write_bytes(z.read('oof_reranker_results/oof_lambdamart_reranker.joblib'))
fixed_tree=joblib.load(model_path)
train_rows,val_rows=D.get_splits()
def prepare(rows,cfg,wv,cv):
    ex=D.preprocess(rows,dict(use_normaliser=cfg['use_normaliser'],max_span_len=cfg['max_span_len'],tau=cfg['tau'],soft_tau=cfg.get('soft_tau')))
    D.encode(ex,wv,cv); return ex
paths=[]
with zipfile.ZipFile(SOURCE) as z:
    rid=prior['source_run']; p=WORK/(rid+'.pt'); p.write_bytes(z.read('attention_variant_checkpoints/'+rid+'.pt')); paths.append(('CurrentGated-s1',p))
with zipfile.ZipFile(V2) as z:
    for seed in (1,2,3):
        matches=[n for n in z.namelist() if n.startswith('checkpoints/A5_single_span-s'+str(seed)+'-') and n.endswith('.pt')]
        assert len(matches)==1,matches
        p=WORK/Path(matches[0]).name; p.write_bytes(z.read(matches[0])); paths.append(('V2-s'+str(seed),p))
seed_records={}; seed_examples={}; configs={}
def report(name,preds,examples):
    scores=evaluate(preds,[e['answers'] for e in examples]); results[name]=scores
    (OUT/(name+'_predictions.json')).write_text(json.dumps([dict(row_id=e['row_id'],pred=p) for e,p in zip(examples,preds)]),encoding='utf-8')
    print(name,scores); return scores
'''

CONFIRM = r'''# Fixed saved reranker. No fitting or validation-driven selection.
for name,path in paths:
    model,cfg,wv,cv=T.load_checkpoint(path,DEVICE)
    ex=prepare(val_rows,cfg,wv,cv)
    records=R.topk_candidates(model,ex,DEVICE,top_k=10,max_span_len=cfg['max_span_len'])
    assert [r['row_id'] for r in records]==[e['row_id'] for e in ex]
    seed_records[name]=records; seed_examples[name]=ex; configs[name]=cfg
    base=report(name+'_base',[r['candidates'][0]['text'] for r in records],ex)
    preds,_=R.rerank(fixed_tree,ex,records); reranked=report(name+'_fixed_oof',preds,ex)
    print('Delta ROUGE-L',reranked['rougeL']-base['rougeL'])
    if name=='CurrentGated-s1':
        assert abs(base['rougeL']-prior['scores']['BaseTop1']['rougeL'])<1e-4
        assert abs(reranked['rougeL']-prior['scores']['OOFLambdaMART']['rougeL'])<1e-4
    del model; torch.cuda.empty_cache()
save()
'''

ENSEMBLE = r'''# Average probabilities, keeping each checkpoint's own vocabulary.
from src.train import to_device
probabilities=[]
for name,path in paths[1:]:
    model,cfg,wv,cv=T.load_checkpoint(path,DEVICE); ex=prepare(val_rows,cfg,wv,cv)
    assert cfg==configs['V2-s1'], 'Ensemble requires identical configs'
    collected=[]
    with torch.no_grad():
        for batch in D.make_loader(ex,batch_size=64):
            out=model(to_device(batch,DEVICE))
            for i,e in enumerate(batch['meta']):
                n=len(e['t_tokens'])
                collected.append((e['row_id'],torch.softmax(out['start_logits'][i,:n],-1).cpu(),torch.softmax(out['end_logits'][i,:n],-1).cpu()))
    probabilities.append(collected); del model; torch.cuda.empty_cache()
ex=seed_examples['V2-s1']; records=[]
for index,e in enumerate(ex):
    items=[p[index] for p in probabilities]; assert all(x[0]==e['row_id'] for x in items)
    start=torch.stack([x[1] for x in items]).mean(0).clamp_min(1e-30).log()
    end=torch.stack([x[2] for x in items]).mean(0).clamp_min(1e-30).log()
    n=len(start); combined=start[:,None]+end[None,:]
    i=torch.arange(n)[:,None]; j=torch.arange(n)[None,:]
    legal=(j>=i)&(j-i<configs['V2-s1']['max_span_len']); combined[~legal]=-float('inf')
    values,indices=combined.flatten().topk(min(10,int(legal.sum())))
    candidates=[]
    for value,flat in zip(values.tolist(),indices.tolist()):
        a,b=divmod(flat,n)
        candidates.append(dict(start=a,end=b,text=D.span_text(e,a,b),start_logp=float(start[a]),end_logp=float(end[b]),base_score=value,region=R.span_region(e,a,b)))
    records.append(dict(row_id=e['row_id'],qid=e['qid'],candidates=candidates))
report('Ensemble_base',[r['candidates'][0]['text'] for r in records],ex)
preds,_=R.rerank(fixed_tree,ex,records); report('Ensemble_fixed_oof_exploratory',preds,ex)
results['ensemble_note']='Fixed reranker trained on single-model OOF candidates; exploratory transfer, not matched ensemble-OOF training.'
save()
'''

FEATURES = r'''# Fixed, label-free boundary and candidate-confidence features.
import re
from sklearn.model_selection import GroupShuffleSplit
from lightgbm import LGBMRanker
original_features=R.candidate_features
EXTRA_NAMES=['left_punctuation','right_punctuation','left_preposition','answer_has_capital','answer_has_digit','candidate_score_spread','candidate_entropy','span_support']
def extra_features(e,c,record):
    a,b=c['start'],c['end']; tokens=e['t_tokens']
    left=tokens[a-1]['raw'] if a else ''; right=tokens[b+1]['raw'] if b+1<len(tokens) else ''
    scores=np.array([x['base_score'] for x in record['candidates']]); p=np.exp(scores-scores.max()); p/=p.sum()
    support=np.mean([len(set(range(a,b+1))&set(range(x['start'],x['end']+1)))/len(set(range(a,b+1))|set(range(x['start'],x['end']+1))) for x in record['candidates']])
    return [float(bool(re.search(r'[^\w\s]',left))),float(bool(re.search(r'[^\w\s]',right))),float(left.lower() in {'in','on','at','from','to','by','of','for','with'}),float(any(x.isupper() for x in c['text'])),float(any(x.isdigit() for x in c['text'])),float(scores.std()),float(-(p*np.log(p+1e-30)).sum()),float(support)]
def matrices(examples,records,enhanced):
    by={e['row_id']:e for e in examples}; out=[]
    for r in records:
        e=by[r['row_id']]; rows=[]
        for rank,c in enumerate(r['candidates'],1):
            base=original_features(e,c,rank,r['candidates'][0]['base_score'])
            rows.append(np.r_[base,extra_features(e,c,r)] if enhanced else base)
        out.append(np.stack(rows))
    return out
# OOF caches are intentionally not in the compact Step 10 artifact ZIP.
CACHE=Path('/content/drive/MyDrive/CITS4012_A2_reranker_oof')
oof=[]
for fold in (1,2,3):
    path=CACHE/f'fold{fold}_oof_top10.jsonl'
    assert path.exists(),f'Missing {path}. Restore Step 10 Drive caches or rerun Step 10.'
    oof.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
by={r['row_id']:r for r in oof}; assert len(by)==len(oof)==len(train_rows)
assert set(by)=={r['row_id'] for r in train_rows}
model,cfg,wv,cv=T.load_checkpoint(paths[0][1],DEVICE)
train=prepare(train_rows,cfg,wv,cv); del model; torch.cuda.empty_cache()
oof=[by[e['row_id']] for e in train]
# Discovery only on grouped internal development. Fixed validation is not used to select features.
groups=np.array([e['tweet'] for e in train],dtype=object)
fit_idx,dev_idx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=40120).split(np.zeros(len(train)),groups=groups))
assert not (set(groups[fit_idx])&set(groups[dev_idx]))
def fit_tree(examples,records,enhanced):
    xs=matrices(examples,records,enhanced); labels=[]
    lookup={e['row_id']:e for e in examples}
    for r in records:
        labels.extend(min(4,int(round(4*score_example(c['text'],lookup[r['row_id']]['answers'],('rougeL',))['rougeL']))) for c in r['candidates'])
    m=LGBMRanker(objective='lambdarank',n_estimators=150,learning_rate=.05,num_leaves=15,max_depth=4,min_child_samples=50,reg_lambda=1.,random_state=4012,deterministic=True,force_col_wise=True,verbosity=-1,n_jobs=-1)
    m.fit(np.concatenate(xs),np.array(labels),group=[len(x) for x in xs]); return m
def predict_tree(m,examples,records,enhanced):
    return [r['candidates'][int(np.argmax(m.predict(x)))]['text'] for r,x in zip(records,matrices(examples,records,enhanced))]
for enhanced in (False,True):
    label='enhanced' if enhanced else 'original'
    fit_ex=[train[i] for i in fit_idx]; fit_rec=[oof[i] for i in fit_idx]
    dev_ex=[train[i] for i in dev_idx]; dev_rec=[oof[i] for i in dev_idx]
    m=fit_tree(fit_ex,fit_rec,enhanced)
    report('InternalDev_'+label,predict_tree(m,dev_ex,dev_rec,enhanced),dev_ex)
results['feature_names']=list(R.FEATURE_NAMES)+EXTRA_NAMES
results['internal_dev_note']='Exploratory split of existing OOF records: generating models may have seen other internal-dev rows. Not a fully nested independent estimate. Final validation remains separate.'
save()
'''

FINAL = r'''# Both predefined variants are evaluated; do not tune features on these results.
ex=seed_examples['CurrentGated-s1']; records=seed_records['CurrentGated-s1']
for enhanced in (False,True):
    label='enhanced' if enhanced else 'original'
    m=fit_tree(train,oof,enhanced)
    preds=predict_tree(m,ex,records,enhanced)
    report('FeatureStudy_'+label,preds,ex)
    joblib.dump(m,OUT/('feature_study_'+label+'.joblib'))
    names=list(R.FEATURE_NAMES)+(EXTRA_NAMES if enhanced else [])
    results['importance_'+label]=sorted(zip(names,m.feature_importances_.tolist()),key=lambda x:x[1],reverse=True)
save()
print('Completed. Test set was not loaded. Compare same-base deltas; ensemble transfer is exploratory.')
'''

def build():
    cells=[O.md('''# Step 11 — Confirm OOF, ensemble, then improve features

Run in order on Colab. No base-model training is performed. Inputs: `train.csv`, Step 01 v2 ZIP, Step 07 attention ZIP, Step 10 OOF ZIP, and Step 10 candidate caches on Drive. The test set is never loaded.

First reproduce the exact Step 10 checkpoint and apply its fixed reranker to the matched three-seed v2 family. Then test a probability ensemble. Finally compare a predefined boundary/confidence feature extension against unchanged LambdaMART settings. All validation comparisons are exploratory after previous project model selection; no new feature tuning on validation.
'''),O.code(SETUP),O.code(O.DATA)]
    for rel in O.FILES:
        cells.append(O.code('%%writefile '+rel+'\n'+(ROOT/rel).read_text(encoding='utf-8')))
    for title,code in [('Load checkpoints and fixed OOF model',LOAD),('1. Exact reproduction and three-seed confirmation',CONFIRM),('2. Probability ensemble',ENSEMBLE),('3. Boundary and confidence features: internal development',FEATURES),('4. Fixed feature comparison on validation',FINAL)]:
        cells.extend([O.md('## '+title),O.code(code)])
    out=ROOT/'step11_colab_reranker_confirmation_ensemble_features.ipynb'
    out.write_text(json.dumps(O.notebook(cells,True),ensure_ascii=False,indent=1),encoding='utf-8')
    print(out)

if __name__=='__main__': build()
