"""Standalone validation-only combination of Step 11 ensemble and features."""
import ast
import json
import make_reranker_next_notebook as N

O=N.O
SETUP=r'''import os,sys,subprocess,json,zipfile,shutil,hashlib,warnings
from pathlib import Path
if 'probabilities' in globals() and 'paths' in globals():
    _step12_probability_source=[p.name for _,p in paths[1:]]
assert 'google.colab' in sys.modules
subprocess.run([sys.executable,'-m','pip','install','-q','gensim','emoji','nltk','scikit-learn','joblib','lightgbm'],check=True)
import torch,numpy as np,joblib,re
from google.colab import drive,files
from IPython.display import display,Markdown
drive.mount('/content/drive')
WORK=Path('/content/A2_ensemble_features'); WORK.mkdir(exist_ok=True); os.chdir(WORK)
sys.path.insert(0,str(WORK))
Path('src').mkdir(exist_ok=True); Path('data').mkdir(exist_ok=True)
OUT=WORK/'results'; OUT.mkdir(exist_ok=True)
BACKUP=Path('/content/drive/MyDrive/CITS4012_A2_ensemble_features'); BACKUP.mkdir(parents=True,exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMRanker was fitted with feature names',category=UserWarning)
def locate(name,folder):
    choices=[Path('/content')/name,Path('/content/drive/MyDrive')/folder/name]
    found=next((p for p in choices if p.exists()),None)
    if found is None:
        print('Upload',name); up=files.upload(); assert name in up
        found=WORK/name; found.write_bytes(up[name])
    return found
STEP11=locate('reranker_next_artifacts.zip','CITS4012_A2_reranker_next')
V2=locate('a2_v2_colab_artifacts.zip','CITS4012_A2_v2')
'''
LOAD=r'''from src import data as D,train as T,rerank as R
from src.evaluate import evaluate,score_example
train_rows,val_rows=D.get_splits()
def prepare(rows,cfg,wv,cv):
    ex=D.preprocess(rows,dict(use_normaliser=cfg['use_normaliser'],max_span_len=cfg['max_span_len'],tau=cfg['tau'],soft_tau=cfg.get('soft_tau')))
    D.encode(ex,wv,cv); return ex
with zipfile.ZipFile(STEP11) as z:
    prior=json.loads(z.read('summary.json'))
    for label in ('original','enhanced'):
        name='feature_study_'+label+'.joblib'; (WORK/name).write_bytes(z.read(name))
    baseline_predictions=json.loads(z.read('Ensemble_fixed_oof_exploratory_predictions.json'))
trees={label:joblib.load(WORK/('feature_study_'+label+'.joblib')) for label in ('original','enhanced')}
paths=[('unused',None)]
with zipfile.ZipFile(V2) as z:
    for seed in (1,2,3):
        matches=[n for n in z.namelist() if n.startswith('checkpoints/A5_single_span-s'+str(seed)+'-') and n.endswith('.pt')]
        assert len(matches)==1,matches
        p=WORK/Path(matches[0]).name; p.write_bytes(z.read(matches[0])); paths.append(('V2-s'+str(seed),p))
seed_examples={}; configs={}
model,cfg,wv,cv=T.load_checkpoint(paths[1][1],DEVICE)
seed_examples['V2-s1']=prepare(val_rows,cfg,wv,cv); configs['V2-s1']=cfg
del model; torch.cuda.empty_cache()
'''

def feature_code():
    tree=ast.parse(N.FEATURES)
    functions=[ast.get_source_segment(N.FEATURES,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('extra_features','matrices','predict_tree')]
    return 'original_features=R.candidate_features\n'+'\n\n'.join(functions)

EVAL=r'''assert prior['feature_names']==list(R.FEATURE_NAMES)+['left_punctuation','right_punctuation','left_preposition','answer_has_capital','answer_has_digit','candidate_score_spread','candidate_entropy','span_support']
ex=seed_examples['V2-s1']; golds=[e['answers'] for e in ex]
predictions={'Ensemble_base':[r['candidates'][0]['text'] for r in records]}
for label in ('original','enhanced'):
    predictions['Ensemble_'+label]=predict_tree(trees[label],ex,records,label=='enhanced')
old={r['row_id']:r['pred'] for r in baseline_predictions}
assert set(old)=={e['row_id'] for e in ex}
assert predictions['Ensemble_original']==[old[e['row_id']] for e in ex], 'Original predictions did not reproduce Step 11'
scores={name:evaluate(preds,golds) for name,preds in predictions.items()}
assert abs(scores['Ensemble_base']['rougeL']-prior['Ensemble_base']['rougeL'])<1e-4
lines=['| System | BLEU-1 | METEOR | ROUGE-L | EM | F1 |','|---|---:|---:|---:|---:|---:|']
for name,s in scores.items(): lines.append('| '+name+' | '+' | '.join(f'{s[m]:.2f}' for m in ('bleu1','meteor','rougeL','em','f1'))+' |')
display(Markdown('\n'.join(lines)))
for name,preds in predictions.items():
    (OUT/(name+'_predictions.json')).write_text(json.dumps([dict(row_id=e['row_id'],pred=p) for e,p in zip(ex,preds)]),encoding='utf-8')
'''
BOOT=r'''# Cluster bootstrap: all questions from a tweet are resampled together.
from collections import defaultdict
groups=defaultdict(list)
for i,e in enumerate(ex): groups[e['tweet']].append(i)
indices=list(groups.values()); rng=np.random.default_rng(4012); intervals={}
lines=['| Comparison | Metric | Delta | Tweet bootstrap 95% CI |','|---|---|---:|---:|']
for a,b in [('Ensemble_original','Ensemble_base'),('Ensemble_enhanced','Ensemble_original')]:
    intervals[a+'_vs_'+b]={}
    for metric in ('rougeL','f1','em'):
        d=np.array([score_example(p,g,(metric,))[metric]-score_example(q,g,(metric,))[metric] for p,q,g in zip(predictions[a],predictions[b],golds)])
        sums=np.array([d[ix].sum() for ix in indices]); sizes=np.array([len(ix) for ix in indices]); means=[]
        for _ in range(5000):
            chosen=rng.integers(0,len(indices),len(indices)); means.append(100*sums[chosen].sum()/sizes[chosen].sum())
        lo,hi=np.quantile(means,[.025,.975]); delta=float(d.mean()*100)
        intervals[a+'_vs_'+b][metric]=dict(delta=delta,low=float(lo),high=float(hi))
        lines.append(f'| {a} vs {b} | {metric} | {delta:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
display(Markdown('\n'.join(lines)))
summary=dict(scores=scores,paired_tweet_bootstrap=intervals,n_questions=len(ex),n_tweets=len(indices),test_accessed=False,note='Validation-only exploratory transfer. Both rerankers were trained on single-model OOF candidates, not ensemble OOF. Intervals do not correct for previous validation-based model selection.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(OUT/'report.md').write_text('\n'.join(lines)+'\n\n'+summary['note'],encoding='utf-8')
archive=Path(shutil.make_archive(str(WORK/'ensemble_feature_artifacts'),'zip',OUT))
shutil.copy2(archive,BACKUP/archive.name)
print('SAVED',BACKUP/archive.name)
print('Completed: validation only, no training and no test evaluation.')
'''

def build():
    cells=[O.md('''# Step 12 — Ensemble + enhanced OOF reranker (validation only)

Open this notebook in Colab, select GPU and run top to bottom. No training or test evaluation is performed. Inputs: `train.csv`, `a2_v2_colab_artifacts.zip`, and `reranker_next_artifacts.zip`. Step 10 candidate caches and the attention-variant ZIP are not needed.

Reuses Step 11 in-memory probability arrays when available; otherwise runs inference from the three saved checkpoints. Applies both saved rerankers to the same ensemble Top-10 and reproduces Step 11 predictions before comparison. Reports paired tweet-cluster bootstrap intervals. Results remain exploratory because rerankers were trained on single-model OOF candidates.
'''),O.code(SETUP),O.code(O.DATA)]
    for rel in O.FILES: cells.append(O.code('%%writefile '+rel+'\n'+(O.ROOT/rel).read_text(encoding='utf-8')))
    cells.extend([O.md('## Load saved models'),O.code(LOAD)])
    inference,combine=N.ENSEMBLE.split("ex=seed_examples['V2-s1']; records=[]",1)
    # Use old arrays only after checkpoint identity and row/length validation.
    loop=inference[inference.index('for name,path in paths[1:]:'):]
    cached="""cached=globals().get('probabilities')
expected=seed_examples['V2-s1']
valid=(isinstance(cached,list) and len(cached)==3 and all(len(p)==len(expected) and all(x[0]==e['row_id'] and len(x[1])==len(x[2])==len(e['t_tokens']) for x,e in zip(p,expected)) for p in cached))
# In-memory reuse requires explicit source identity from Step 11, not just matching shapes.
valid=valid and globals().get('_step12_probability_source')==[p.name for _,p in paths[1:]]
from src.train import to_device
if valid:
    print('Reusing verified probability cache')
else:
    probabilities=[]
"""+'\n'.join('    '+line for line in loop.splitlines())+"\n_step12_probability_source=[p.name for _,p in paths[1:]]\n"
    combine="ex=seed_examples['V2-s1']; records=[]"+combine.split("report('Ensemble_base'",1)[0]
    cells.extend([O.md('## Generate ensemble candidates'),O.code(cached+combine),O.md('## Apply saved feature variants'),O.code(feature_code()),O.code(EVAL),O.md('## Compare and save'),O.code(BOOT)])
    path=O.ROOT/'step12_colab_ensemble_enhanced_reranker.ipynb'
    path.write_text(json.dumps(O.notebook(cells,True),ensure_ascii=False,indent=1),encoding='utf-8'); print(path)
if __name__=='__main__': build()
