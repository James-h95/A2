"""Build a fixed-candidate OOF study of shared-BiLSTM similarity features."""
import ast
import json
import make_reranker_next_notebook as N
import make_ensemble_feature_notebook as E

O=N.O
SETUP=E.SETUP[E.SETUP.index('import os,sys'):].replace("if 'probabilities' in globals() and 'paths' in globals():\n    _step12_probability_source=[p.name for _,p in paths[1:]]\n",'')
SETUP=SETUP[:SETUP.index("STEP11=locate")].replace('A2_ensemble_features','A2_representation_reranker').replace('CITS4012_A2_ensemble_features','CITS4012_A2_representation_reranker')+r'''
SOURCE=locate('attention_variants_artifacts.zip','CITS4012_A2_attention_variants')
OOFZIP=locate('oof_span_reranker_artifacts.zip','CITS4012_A2_reranker_oof')
FOLD_ROOT=Path('/content/drive/MyDrive/CITS4012_A2_reranker_oof')
assert (FOLD_ROOT/'oof_base_results/runs.csv').exists(), 'Restore Step 10 Drive folder: fold checkpoints, runs.csv and candidate caches are required.'
'''

LOAD=r'''from src import data as D,train as T,rerank as R
from src.evaluate import evaluate,score_example
from lightgbm import LGBMRanker
from sklearn.model_selection import StratifiedGroupKFold
train_rows,val_rows=D.get_splits()
assert D.TEST_LOCKED
with zipfile.ZipFile(OOFZIP) as z:
    prior=json.loads(z.read('oof_reranker_results/oof_reranker_summary.json'))
with zipfile.ZipFile(SOURCE) as z:
    rid=prior['source_run']; full_path=WORK/(rid+'.pt')
    full_path.write_bytes(z.read('attention_variant_checkpoints/'+rid+'.pt'))
fold_runs=list(csv.DictReader((FOLD_ROOT/'oof_base_results/runs.csv').open(encoding='utf-8-sig')))
fold_paths={}
for fold in (1,2,3):
    matches=[r for r in fold_runs if r['model']==f'OOFBaseFold{fold}']
    assert matches, f'Missing fold {fold} run'
    p=FOLD_ROOT/'oof_base_checkpoints'/(matches[-1]['run_id']+'.pt')
    assert p.exists(),f'Missing {p}; restore Step 10 checkpoint. No automatic retraining.'
    assert (FOLD_ROOT/f'fold{fold}_oof_top10.jsonl').exists()
    fold_paths[fold]=p
labels=np.array([D._question_word(r['question']) for r in train_rows])
groups=np.array([r['tweet'] for r in train_rows],dtype=object)
folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=4012).split(np.zeros(len(train_rows)),labels,groups))
def prepare(rows,cfg,wv,cv):
    ex=D.preprocess(rows,dict(use_normaliser=cfg['use_normaliser'],max_span_len=cfg['max_span_len'],tau=cfg['tau'],soft_tau=cfg.get('soft_tau')))
    D.encode(ex,wv,cv); return ex
def read_jsonl(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
REP_NAMES=['question_span_mean_cos','question_context_mean_cos','question_left_mean_cos','question_right_mean_cos','question_to_span_maxcos_mean','span_to_question_maxcos_mean']
'''

REP=r'''import torch.nn.functional as F
def representation_features(h,u,start,end,window=3):
    # h/u are trimmed to real tokens: no padded vectors in pooling.
    assert 0<=start<=end<len(h) and len(u)>0
    q=u.mean(0); span=h[start:end+1]
    def cosine(v):
        return float(F.cosine_similarity(q[None],v.mean(0)[None],dim=-1,eps=1e-8)) if len(v) else 0.
    pair=F.normalize(u,dim=-1,eps=1e-8)@F.normalize(span,dim=-1,eps=1e-8).T
    values=[cosine(span),cosine(h[max(0,start-window):min(len(h),end+window+1)]),
            cosine(h[max(0,start-window):start]),cosine(h[end+1:min(len(h),end+window+1)]),
            float(pair.max(1).values.mean()),float(pair.max(0).values.mean())]
    assert len(values)==len(REP_NAMES) and np.isfinite(values).all()
    return values

# Small deterministic checks run before expensive fold inference.
_h=torch.tensor([[1.,0.],[0.,1.],[1.,0.]])
_u=torch.tensor([[1.,0.]])
_v=representation_features(_h,_u,0,0)
assert abs(_v[0]-1.)<1e-6 and _v[2]==0. and _v[4]==1. and _v[5]==1.
assert representation_features(_h,_u,2,2)[3]==0.
assert np.isfinite(representation_features(torch.zeros(2,2),torch.zeros(1,2),0,1)).all()
print('Representation feature sanity checks passed')

@torch.no_grad()
def enrich(model,examples,records,cfg):
    model.eval(); lookup={r['row_id']:r for r in records}
    assert len(lookup)==len(examples)==len(records)
    assert set(lookup)=={e['row_id'] for e in examples}
    # Verify checkpoint/candidate provenance before attaching representations.
    generated=R.topk_candidates(model,examples,DEVICE,top_k=10,batch_size=64,max_span_len=cfg['max_span_len'])
    for g in generated:
        old=lookup[g['row_id']]
        assert [(c['start'],c['end']) for c in g['candidates']]==[(c['start'],c['end']) for c in old['candidates']], 'Candidate/checkpoint mismatch'
        assert np.allclose([c['base_score'] for c in g['candidates']],[c['base_score'] for c in old['candidates']],atol=2e-4,rtol=1e-4), 'Candidate score mismatch'
    for batch in D.make_loader(examples,batch_size=64):
        b=T.to_device(batch,DEVICE)
        # Same shared encoder and embedding as model.forward, frozen in eval mode.
        h=model.drop(model.encoder(model.embed(b['t_words'],b['t_chars'],b['t_flags']),b['t_mask']))
        u=model.drop(model.encoder(model.embed(b['q_words'],b['q_chars'],b['q_flags']),b['q_mask']))
        for i,e in enumerate(batch['meta']):
            ht=h[i,:int(b['t_mask'][i].sum())]; ut=u[i,:int(b['q_mask'][i].sum())]
            for c in lookup[e['row_id']]['candidates']:
                c['representation_features']=representation_features(ht,ut,c['start'],c['end'])
    return [lookup[e['row_id']] for e in examples]

def checkpoint_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()
def enriched_cache(tag,path,rows,records):
    # Cache includes model and candidate fingerprints; stale values are rejected.
    key=hashlib.sha256((checkpoint_hash(path)+json.dumps(records,sort_keys=True)+json.dumps(REP_NAMES)+str(3)).encode()).hexdigest()
    cache=BACKUP/(tag+'_representations.json')
    model,cfg,wv,cv=T.load_checkpoint(path,DEVICE)
    examples=prepare(rows,cfg,wv,cv)
    payload=json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else None
    if payload and payload.get('fingerprint')==key:
        enriched=payload['records']; print('CACHE',tag)
        assert [r['row_id'] for r in enriched]==[e['row_id'] for e in examples]
        assert all(len(c['representation_features'])==len(REP_NAMES) and np.isfinite(c['representation_features']).all() for r in enriched for c in r['candidates'])
    else:
        enriched=enrich(model,examples,records,cfg)
        cache.write_text(json.dumps(dict(fingerprint=key,records=enriched)),encoding='utf-8')
        print('SAVED representations',tag,len(enriched))
    del model; torch.cuda.empty_cache()
    return examples,enriched
'''

GENERATE=r'''oof=[]; train=[]
for fold,(fit_idx,hold_idx) in enumerate(folds,1):
    assert not (set(groups[fit_idx])&set(groups[hold_idx]))
    rows=[train_rows[i] for i in hold_idx]
    records=read_jsonl(FOLD_ROOT/f'fold{fold}_oof_top10.jsonl')
    examples,enriched=enriched_cache(f'fold{fold}',fold_paths[fold],rows,records)
    train.extend(examples); oof.extend(enriched)
assert len(train)==len(oof)==len(train_rows)
assert len({e['row_id'] for e in train})==len(train_rows)
order=np.argsort([e['row_id'] for e in train]); train=[train[i] for i in order]; oof=[oof[i] for i in order]
model,cfg,wv,cv=T.load_checkpoint(full_path,DEVICE)
val=prepare(val_rows,cfg,wv,cv)
records=R.topk_candidates(model,val,DEVICE,top_k=10,batch_size=64,max_span_len=cfg['max_span_len'])
del model; torch.cuda.empty_cache()
val,val_records=enriched_cache('validation',full_path,val_rows,records)
print('OOF features ready',len(train),'validation',len(val))
'''

MATRIX=r'''def matrices(examples,records,enhanced):
    base=R.feature_matrix(examples,records)
    if not enhanced: return base
    return [np.column_stack((x,np.asarray([c['representation_features'] for c in r['candidates']]))) for x,r in zip(base,records)]
'''

FIT=r'''# Two predefined configurations, unchanged tree hyperparameters; no tuning.
predictions={'BaseTop1':[r['candidates'][0]['text'] for r in val_records]}
importances={}
for enhanced in (False,True):
    name='RepresentationOOF' if enhanced else 'OriginalOOF'
    model=fit_tree(train,oof,enhanced)
    predictions[name]=predict_tree(model,val,val_records,enhanced)
    joblib.dump(model,OUT/(name+'.joblib'))
    names=list(R.FEATURE_NAMES)+(REP_NAMES if enhanced else [])
    importances[name]=sorted(zip(names,model.feature_importances_.tolist()),key=lambda x:x[1],reverse=True)
golds=[e['answers'] for e in val]
scores={name:evaluate(p,golds) for name,p in predictions.items()}
for metric in ('bleu1','meteor','rougeL','em','f1'):
    assert abs(scores['OriginalOOF'][metric]-prior['scores']['OOFLambdaMART'][metric])<1e-4, 'Original OOF failed reproduction'
lines=['| System | BLEU-1 | METEOR | ROUGE-L | EM | F1 |','|---|---:|---:|---:|---:|---:|']
for name,s in scores.items(): lines.append('| '+name+' | '+' | '.join(f'{s[m]:.2f}' for m in ('bleu1','meteor','rougeL','em','f1'))+' |')
display(Markdown('\n'.join(lines)))
for name,p in predictions.items():
    (OUT/(name+'_predictions.json')).write_text(json.dumps([dict(row_id=e['row_id'],pred=a) for e,a in zip(val,p)]),encoding='utf-8')
(OUT/'feature_schema.json').write_text(json.dumps(dict(base=list(R.FEATURE_NAMES),representation=REP_NAMES,encoder='shared encoder before BiDAF attention',window=3),indent=2),encoding='utf-8')
ex=val
'''

def build():
    cells=[O.md('''# Step 13 — BiLSTM representation similarity in OOF reranking

Validation only. No base-model training and no test evaluation. Use a Colab GPU and run top to bottom. Requires `train.csv`, `attention_variants_artifacts.zip`, `oof_span_reranker_artifacts.zip` and the original Step 10 Drive folder `MyDrive/CITS4012_A2_reranker_oof` including fold checkpoints, `oof_base_results/runs.csv` and the three `fold*_oof_top10.jsonl` files. A preflight check stops if these are missing; it does not retrain models.

Add six fixed cosine features from the shared BiLSTM: question vs candidate / expanded context / left context / right context, and bidirectional mean-of-best token similarities. Empty left/right context maps to zero. No externally pretrained embeddings are introduced.

Every OOF feature comes from the same fold model that generated that candidate. Candidate positions and scores are checked before extraction. Validation features come from the exact Step 10 final checkpoint. Compare unchanged 30-feature LambdaMART with 36-feature LambdaMART on identical candidates. Prior eight boundary/confidence additions are excluded to isolate this hypothesis. Representation caches are saved on Drive after each fold. Bootstrap intervals are exploratory and do not correct for earlier validation-based model selection.
'''),O.code(SETUP.replace('import os,sys,subprocess,json,zipfile,shutil,hashlib,warnings','import os,sys,subprocess,json,zipfile,shutil,hashlib,warnings,csv')),O.code(O.DATA)]
    for rel in O.FILES: cells.append(O.code('%%writefile '+rel+'\n'+(O.ROOT/rel).read_text(encoding='utf-8')))
    funcs=[ast.get_source_segment(N.FEATURES,n) for n in ast.parse(N.FEATURES).body if isinstance(n,ast.FunctionDef) and n.name in ('fit_tree','predict_tree')]
    bootstrap=E.BOOT.replace("[('Ensemble_original','Ensemble_base'),('Ensemble_enhanced','Ensemble_original')]","[('OriginalOOF','BaseTop1'),('RepresentationOOF','OriginalOOF')]")
    bootstrap=bootstrap.replace('ensemble_feature_artifacts','representation_reranker_artifacts').replace('Validation-only exploratory transfer. Both rerankers were trained on single-model OOF candidates, not ensemble OOF. Intervals do not correct for previous validation-based model selection.','Validation-only fixed-candidate OOF representation study. Intervals do not correct for previous validation-based model selection.')
    bootstrap=bootstrap.replace("(OUT/'summary.json').write_text", "summary['feature_importances']=importances\nsummary['source_run']=prior['source_run']\n(OUT/'summary.json').write_text")
    bootstrap=bootstrap.replace('no training and no test evaluation','no base-model training and no test evaluation')
    for title,code in [('Load and verify required fold artifacts',LOAD),('Representation extraction',REP),('Extract fold-safe training and validation features',GENERATE),('Fixed reranker definitions',MATRIX+'\n\n'+'\n\n'.join(funcs)),('Fit and evaluate on unchanged candidates',FIT),('Tweet bootstrap and save',bootstrap)]:
        cells.extend([O.md('## '+title),O.code(code)])
    path=O.ROOT / "notebooks" / "experiments" / 'step13_colab_bilstm_representation_reranker.ipynb'
    path.write_text(json.dumps(O.notebook(cells,True),ensure_ascii=False,indent=1),encoding='utf-8'); print(path)
if __name__=='__main__': build()
