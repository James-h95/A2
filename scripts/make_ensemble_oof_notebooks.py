"""Candidate-union diagnosis followed by matched three-fold ensemble OOF."""
import json
import make_oof_reranker_notebooks as O

SETUP=r'''import os,sys,subprocess,json,zipfile,shutil,hashlib,csv,io,warnings
from pathlib import Path
assert 'google.colab' in sys.modules
subprocess.run([sys.executable,'-m','pip','install','-q','gensim','emoji','nltk','scikit-learn','joblib','lightgbm'],check=True)
import numpy as np,torch,joblib
from google.colab import drive,files
from IPython.display import display,Markdown
drive.mount('/content/drive')
WORK=Path('/content/A2_ensemble_oof'); WORK.mkdir(exist_ok=True); os.chdir(WORK); sys.path.insert(0,str(WORK))
BACKUP=Path('/content/drive/MyDrive/CITS4012_A2_ensemble_oof'); BACKUP.mkdir(parents=True,exist_ok=True)
CKPT=BACKUP/'checkpoints'; CKPT.mkdir(exist_ok=True)
RESULTS=BACKUP/'fold_results'; RESULTS.mkdir(exist_ok=True)
os.environ['A2_RESULTS']=str(RESULTS); os.environ['A2_CKPT']=str(CKPT)
Path('src').mkdir(exist_ok=True); Path('data').mkdir(exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMRanker was fitted with feature names',category=UserWarning)
def locate(name,folder):
    found=next((p for p in [Path('/content')/name,Path('/content/drive/MyDrive')/folder/name] if p.exists()),None)
    if found is None:
        print('Upload',name); up=files.upload(); assert name in up
        found=WORK/name; found.write_bytes(up[name])
    return found
V2=locate('a2_v2_colab_artifacts.zip','CITS4012_A2_v2')
STEP11=locate('reranker_next_artifacts.zip','CITS4012_A2_reranker_next')
'''

LOAD=r'''# Clear only already imported project modules, so new paths/env apply on reused runtimes.
for key in list(sys.modules):
    if key=='src' or key.startswith('src.'): del sys.modules[key]
from src import data as D,train as T,rerank as R
from src.evaluate import evaluate,score_example
from src.runlog import log_run
assert D.TEST_LOCKED
train_rows,val_rows=D.get_splits()
full_paths=[]
with zipfile.ZipFile(V2) as z:
    for seed in (1,2,3):
        names=[n for n in z.namelist() if n.startswith(f'checkpoints/A5_single_span-s{seed}-') and n.endswith('.pt')]
        assert len(names)==1,names
        p=WORK/Path(names[0]).name; p.write_bytes(z.read(names[0])); full_paths.append(p)
with zipfile.ZipFile(STEP11) as z:
    prior=json.loads(z.read('summary.json'))
    old_preds=json.loads(z.read('Ensemble_fixed_oof_exploratory_predictions.json'))
    p=WORK/'fixed_original.joblib'; p.write_bytes(z.read('feature_study_original.joblib'))
fixed_tree=joblib.load(p)
model,CFG,wv,cv=T.load_checkpoint(full_paths[0],DEVICE)
del model; torch.cuda.empty_cache()
def prepare(rows,cfg,wv,cv):
    ex=D.preprocess(rows,dict(use_normaliser=cfg['use_normaliser'],max_span_len=cfg['max_span_len'],tau=cfg['tau'],soft_tau=cfg.get('soft_tau')))
    D.encode(ex,wv,cv); return ex
def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def effective_config(cfg):
    # Older checkpoints omit settings later added with backward-compatible defaults.
    # JSON also makes tuple/list representations of char_widths equivalent.
    merged={**T.DEFAULT_TRAIN,**T._default_model(),**cfg}
    return json.loads(json.dumps(merged,sort_keys=True))
def assert_same_config(cfg):
    actual,expected=effective_config(cfg),effective_config(CFG)
    diff={k:dict(checkpoint=actual.get(k),expected=expected.get(k)) for k in actual.keys()|expected.keys() if actual.get(k)!=expected.get(k)}
    assert not diff, 'Effective checkpoint config differs: '+json.dumps(diff,sort_keys=True)
def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value),encoding='utf-8'); temp.replace(path)
'''

HELPERS=r'''@torch.no_grad()
def probabilities(path,rows,tag):
    key=fingerprint(dict(checkpoint=file_hash(path),rows=rows,method='trimmed-softmax-v1'))
    cache=BACKUP/(tag+'_probabilities.json')
    payload=json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else None
    model,cfg,wv,cv=T.load_checkpoint(path,DEVICE)
    assert_same_config(cfg)
    ex=prepare(rows,cfg,wv,cv)
    if payload and payload.get('fingerprint')==key:
        values=payload['values']; print('CACHE',tag)
    else:
        values=[]
        for batch in D.make_loader(ex,batch_size=64):
            out=model(T.to_device(batch,DEVICE))
            for i,e in enumerate(batch['meta']):
                n=len(e['t_tokens'])
                values.append(dict(row_id=e['row_id'],start=torch.softmax(out['start_logits'][i,:n],-1).cpu().tolist(),end=torch.softmax(out['end_logits'][i,:n],-1).cpu().tolist()))
        atomic_json(cache,dict(fingerprint=key,values=values)); print('SAVED',tag)
    assert len(values)==len(ex)
    assert all(v['row_id']==e['row_id'] and len(v['start'])==len(v['end'])==len(e['t_tokens']) for v,e in zip(values,ex))
    del model; torch.cuda.empty_cache()
    return ex,values

def candidates_from_probabilities(ex,values,top_k=10,union=False):
    # Equal-weight probability average matches Step 11; union scores use this same scale.
    records=[]
    assert len(values)==3
    for k,e in enumerate(ex):
        n=len(e['t_tokens']); items=[v[k] for v in values]
        assert all(x['row_id']==e['row_id'] for x in items)
        starts=torch.tensor([v['start'] for v in items],dtype=torch.float32)
        ends=torch.tensor([v['end'] for v in items],dtype=torch.float32)
        start=starts.mean(0).clamp_min(1e-30).log(); end=ends.mean(0).clamp_min(1e-30).log()
        scores=start[:,None]+end[None,:]
        pos=torch.arange(n); band=(pos[None,:]>=pos[:,None])&((pos[None,:]-pos[:,None])<CFG['max_span_len'])
        scores[~band]=-float('inf'); legal_count=int(band.sum())
        if union:
            selected=set()
            for s,t in zip(starts,ends):
                individual=s.clamp_min(1e-30).log()[:,None]+t.clamp_min(1e-30).log()[None,:]
                individual[~band]=-float('inf')
                selected.update(individual.flatten().topk(min(10,legal_count)).indices.tolist())
            flat=sorted(selected,key=lambda f:(-float(scores.flatten()[f]),f))
        else:flat=scores.flatten().topk(min(top_k,legal_count)).indices.tolist()
        candidates=[]
        for f in flat:
            a,b=divmod(f,n)
            candidates.append(dict(start=a,end=b,text=D.span_text(e,a,b),start_logp=float(start[a]),end_logp=float(end[b]),base_score=float(scores[a,b]),region=R.span_region(e,a,b)))
        records.append(dict(row_id=e['row_id'],qid=e['qid'],candidates=candidates))
    return records

def collect(paths,rows,prefix):
    all_values=[]; examples=None
    for seed,path in enumerate(paths,1):
        ex,values=probabilities(path,rows,f'{prefix}_seed{seed}')
        if examples is not None:
            assert [e['t_tokens'] for e in ex]==[e['t_tokens'] for e in examples], 'Token alignment differs'
        examples=ex; all_values.append(values)
    return examples,all_values

# Verify legal-span generation and union coverage before real inference.
_example=dict(row_id=-1,qid='sanity',tweet='a b c d e f',byline_from=None,
              t_tokens=[dict(start=2*i,end=2*i+1) for i in range(6)])
_rng=np.random.default_rng(4012); _values=[]
for _ in range(3):
    _s=_rng.random(6); _s/=_s.sum(); _e=_rng.random(6); _e/=_e.sum()
    _values.append([dict(row_id=-1,start=_s.tolist(),end=_e.tolist())])
_pooled=candidates_from_probabilities([_example],_values,30)[0]['candidates']
_union=candidates_from_probabilities([_example],_values,union=True)[0]['candidates']
_legal=sum(min(CFG['max_span_len'],6-i) for i in range(6))
assert len(_pooled)==min(30,_legal)
assert len(_union)<=30 and len({(c['start'],c['end']) for c in _union})==len(_union)
assert all(0<=c['start']<=c['end']<6 and c['end']-c['start']<CFG['max_span_len'] for c in _union)
assert set((c['start'],c['end']) for c in _union)<=set((c['start'],c['end']) for c in _pooled)
print('Candidate-generation sanity checks passed')
'''

ORACLE=r'''val,values=collect(full_paths,val_rows,'validation')
sets={'ProbabilityTop10':candidates_from_probabilities(val,values,10),
      'ProbabilityTop30':candidates_from_probabilities(val,values,30),
      'SeedTop10Union':candidates_from_probabilities(val,values,union=True)}
golds=[e['answers'] for e in val]; diagnosis={}; lines=['| Candidates | Mean count | Top-1 RL | Oracle RL | Exact answer coverage |','|---|---:|---:|---:|---:|']
for name,records in sets.items():
    oracle=[max(r['candidates'],key=lambda c:score_example(c['text'],e['answers'],('rougeL',))['rougeL'])['text'] for e,r in zip(val,records)]
    base=evaluate([r['candidates'][0]['text'] for r in records],golds)
    ceiling=evaluate(oracle,golds)
    coverage=100*np.mean([max(score_example(c['text'],e['answers'],('em',))['em'] for c in r['candidates']) for e,r in zip(val,records)])
    count=float(np.mean([len(r['candidates']) for r in records]))
    diagnosis[name]=dict(mean_candidates=count,top1=base,rouge_selected_oracle=ceiling,exact_coverage=float(coverage))
    lines.append(f"| {name} | {count:.1f} | {base['rougeL']:.2f} | {ceiling['rougeL']:.2f} | {coverage:.2f}% |")
assert abs(diagnosis['ProbabilityTop10']['top1']['rougeL']-prior['Ensemble_base']['rougeL'])<1e-4
display(Markdown('\n'.join(lines)))
print('Oracle uses gold answers: diagnostic ceiling only, not an achievable model score.')
print('Top30 controls candidate count; union has at most 30 distinct span positions.')
OUT=WORK/'diagnosis'; OUT.mkdir(exist_ok=True)
atomic_json(OUT/'summary.json',dict(diagnosis=diagnosis,test_accessed=False,note='Validation-only diagnosis. No reranker refitted and no automatic candidate-method selection. Step 15 keeps probability Top10 fixed.'))
(OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
archive=Path(shutil.make_archive(str(WORK/'candidate_union_oracle_artifacts'),'zip',OUT)); shutil.copy2(archive,BACKUP/archive.name)
print('SAVED',BACKUP/archive.name)
'''

FOLDS=O.FOLDS
TRAIN=r'''assert torch.cuda.is_available(), 'Select a GPU runtime for fold training'
from sklearn.model_selection import GroupShuffleSplit
# Keep three outer folds, one inner split, and original hyperparameters fixed.
# Step 10 seed-1 models can be reused only with matching config and split provenance.
OLD=Path('/content/drive/MyDrive/CITS4012_A2_reranker_oof')
old_runs=list(csv.DictReader((OLD/'oof_base_results/runs.csv').open(encoding='utf-8-sig'))) if (OLD/'oof_base_results/runs.csv').exists() else []
oof=[]; oof_examples=[]; manifest=[]
for fold,(outer_fit,holdout) in enumerate(folds,1):
    inner_groups=np.array([r['tweet'] for r in outer_fit],dtype=object)
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.10,random_state=5000+fold).split(np.zeros(len(outer_fit)),groups=inner_groups))
    inner_train=[outer_fit[i] for i in a]; inner_dev=[outer_fit[i] for i in b]
    assert not (set(r['tweet'] for r in inner_train)&set(r['tweet'] for r in inner_dev))
    assert not (set(r['tweet'] for r in outer_fit)&set(r['tweet'] for r in holdout))
    fold_paths=[]
    for seed in (1,2,3):
        name=f'EnsembleOOFFold{fold}'
        provenance=dict(config=CFG,seed=seed,train_ids=[r['row_id'] for r in inner_train],dev_ids=[r['row_id'] for r in inner_dev],holdout_ids=[r['row_id'] for r in holdout])
        key=fingerprint(provenance); state_path=BACKUP/f'fold{fold}_seed{seed}_state.json'
        state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
        path=Path(state['checkpoint']) if state and state.get('fingerprint')==key else None
        if path is not None and not path.exists(): path=None
        if path is not None and file_hash(path)!=state.get('checkpoint_sha256'): path=None
        if path is None and seed==1:
            matches=[r for r in old_runs if r['model']==f'OOFBaseFold{fold}' and r['seed']=='1']
            if matches:
                candidate=OLD/'oof_base_checkpoints'/(matches[-1]['run_id']+'.pt')
                if candidate.exists():
                    ck=torch.load(candidate,map_location='cpu',weights_only=False)
                    if effective_config(ck['cfg'])==effective_config(CFG):
                        # Existing Step 10 follows the identical deterministic outer/inner splits.
                        expected=T.prepare_data(CFG,train_rows=inner_train,val_rows=inner_dev)
                        if ck['vocab']==expected[2].itos and ck['chars']==expected[3].itos:
                            path=candidate; print('REUSE Step10',fold,seed,path.name)
                        del expected
                    del ck
        if path is None:
            data=T.prepare_data(CFG,train_rows=inner_train,val_rows=inner_dev)
            result=T.train(CFG,seed=seed,name=name,data=data,save=True,log=lambda s:print(s,flush=True))
            path=Path(result['ckpt']); del result,data; torch.cuda.empty_cache()
        atomic_json(state_path,dict(fingerprint=key,checkpoint=str(path),checkpoint_sha256=file_hash(path),provenance=provenance))
        fold_paths.append(path); manifest.append(dict(fold=fold,seed=seed,checkpoint=path.name,sha256=file_hash(path)))
    examples,values=collect(fold_paths,holdout,f'fold{fold}')
    records=candidates_from_probabilities(examples,values,10)
    atomic_json(BACKUP/f'fold{fold}_ensemble_top10.json',records)
    oof.extend(records); oof_examples.extend(examples)
    print('OOF FOLD COMPLETE',fold,len(records))
assert len(oof)==len(train_rows)==len({r['row_id'] for r in oof})
assert set(r['row_id'] for r in oof)==set(r['row_id'] for r in train_rows)
order=np.argsort([e['row_id'] for e in oof_examples]); oof_examples=[oof_examples[i] for i in order]; oof=[oof[i] for i in order]
print('ENSEMBLE OOF COMPLETE',len(oof))
'''

FIT=r'''OUT=WORK/'ensemble_oof_results'; OUT.mkdir(exist_ok=True)
# Same 30 features and exact original LambdaMART hyperparameters.
matched_tree,fit_info=R.fit_lambdamart_reranker(oof_examples,oof)
joblib.dump(matched_tree,OUT/'ensemble_oof_lambdamart.joblib')
val,values=collect(full_paths,val_rows,'validation')
records=candidates_from_probabilities(val,values,10)
golds=[e['answers'] for e in val]
predictions={'Ensemble_base':[r['candidates'][0]['text'] for r in records]}
predictions['OriginalOOF'],_=R.rerank(fixed_tree,val,records)
predictions['MatchedEnsembleOOF'],_=R.rerank(matched_tree,val,records)
old={r['row_id']:r['pred'] for r in old_preds}
assert set(old)=={e['row_id'] for e in val}
assert predictions['OriginalOOF']==[old[e['row_id']] for e in val], 'Step 11 baseline failed exact reproduction'
scores={n:evaluate(p,golds) for n,p in predictions.items()}
assert abs(scores['Ensemble_base']['rougeL']-prior['Ensemble_base']['rougeL'])<1e-4
lines=['| System | BLEU-1 | METEOR | ROUGE-L | EM | F1 |','|---|---:|---:|---:|---:|---:|']
for name,s in scores.items():
    lines.append('| '+name+' | '+' | '.join(f'{s[m]:.2f}' for m in ('bleu1','meteor','rougeL','em','f1'))+' |')
    log_run(name,dict(protocol='three-fold-three-seed-ensemble-oof',top_k=10),0,'val',s,notes='validation only; fold training logged separately; fixed hyperparameters')
display(Markdown('\n'.join(lines)))
for name,p in predictions.items():atomic_json(OUT/(name+'_predictions.json'),[dict(row_id=e['row_id'],pred=a) for e,a in zip(val,p)])
atomic_json(OUT/'manifest.json',dict(fold_models=manifest,fit=fit_info,feature_names=list(R.FEATURE_NAMES),importances=R.tree_importances(matched_tree),outer_seed=4012,inner_seeds=[5001,5002,5003]))
ex=val
'''

def build():
    import make_ensemble_feature_notebook as E
    common=[O.code(SETUP),O.code(O.DATA)]
    common.extend(O.code('%%writefile '+rel+'\n'+(O.ROOT/rel).read_text(encoding='utf-8')) for rel in O.FILES)
    common.extend([O.code(LOAD),O.code(HELPERS)])
    intro='''## Inputs and execution

Select GPU in Colab and run top to bottom. Requires `train.csv`, `a2_v2_colab_artifacts.zip`, and `reranker_next_artifacts.zip`. Files are located on Drive or requested for upload. No test data is loaded. Checkpoint and probability caches are stored in `MyDrive/CITS4012_A2_ensemble_oof`.
'''
    diagnosis=[O.md('''# Step 14 — Candidate union oracle diagnosis

Compare probability-averaged Top10, probability-averaged Top30, and the union of each seed's Top10 (at most 30 distinct spans). No training. Oracle answers use gold validation labels only to diagnose candidate availability, never as model predictions. The Top30 control separates candidate-count effects from candidate diversity. This diagnosis does not automatically change Step 15's fixed Top10 protocol.

'''+intro)]+common+[O.md('## Candidate ceiling comparison'),O.code(ORACLE)]
    boot=E.BOOT.replace("[('Ensemble_original','Ensemble_base'),('Ensemble_enhanced','Ensemble_original')]","[('OriginalOOF','Ensemble_base'),('MatchedEnsembleOOF','OriginalOOF')]")
    boot=boot.replace('ensemble_feature_artifacts','ensemble_oof_artifacts').replace('Validation-only exploratory transfer. Both rerankers were trained on single-model OOF candidates, not ensemble OOF. Intervals do not correct for previous validation-based model selection.','Validation-only comparison of fixed single-model OOF training and matched ensemble OOF training. Intervals do not correct for previous validation-based model selection.')
    boot=boot.replace("(OUT/'summary.json').write_text","summary['fit']=fit_info\nsummary['fold_models']=manifest\n(OUT/'summary.json').write_text")
    boot=boot.replace("archive=Path(shutil.make_archive", "shutil.copy2(RESULTS/'runs.csv',OUT/'fold_runs.csv') if (RESULTS/'runs.csv').exists() else None\narchive=Path(shutil.make_archive")
    boot=boot.replace('no training and no test evaluation','no test evaluation')
    training=[O.md('''# Step 15 — Matched 3-fold, 3-seed ensemble OOF reranking

Hold architecture, three outer folds, inner development splits, Top10, features and LambdaMART settings fixed. Change only OOF candidate generation from one model to three-seed probability averaging. Compare against the saved Step 11 reranker on the identical final ensemble candidates.

This notebook performs full GPU training: up to nine fold models, or six new models if the three compatible Step 10 seed-1 checkpoints can be reused. Each model early-stops on inner dev from its outer training portion; its OOF holdout and fixed validation are excluded from training and early stopping. Checkpoints and completed-fold probability caches are saved on Drive. Rerun after interruption to resume from completed models; an interrupted model restarts training. No 5-fold change and no new feature search.

Optional reuse folder: `MyDrive/CITS4012_A2_reranker_oof` with `oof_base_results/runs.csv` and `oof_base_checkpoints`. If missing or incompatible, train seed 1 as well. Reuse assumes these are the original unmodified Step 10 artifacts; config and vocabulary are checked.

'''+intro)]+common+[O.md('## Define unchanged grouped folds'),O.code('from sklearn.model_selection import StratifiedGroupKFold\n'+FOLDS),O.md('## Train or resume fold ensembles'),O.code(TRAIN),O.md('## Fit fixed LambdaMART and evaluate validation'),O.code(FIT),O.md('## Tweet bootstrap and package'),O.code(boot)]
    for name,cells in [('step14_colab_candidate_union_oracle.ipynb',diagnosis),('step15_colab_matched_ensemble_oof.ipynb',training)]:
        path=O.ROOT/"notebooks"/"experiments"/name; path.write_text(json.dumps(O.notebook(cells,True),ensure_ascii=False,indent=1),encoding='utf-8'); print(path)
if __name__=='__main__':build()
