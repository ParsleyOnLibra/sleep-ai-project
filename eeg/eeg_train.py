"""v0.4: subject-separated experiments, causal context, offline transition analysis."""
import argparse
import csv
import json
import platform
from pathlib import Path
import sys
import time
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,f1_score
from data import get_nights, digest
from features import CLASSES,load_record,trailing_context
from transitions import evaluate,reference_events
from sustained import evaluate_sustained
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def split_subjects(subjects, seed):
    if len(subjects)<10:raise ValueError('정규 평가는 참가자 10명 이상이 필요합니다. 2명 점검은 --smoke')
    ids=np.random.default_rng(seed).permutation(sorted(subjects)).tolist()
    n=max(2,round(len(ids)*.2))
    return {'train':ids[2*n:], 'validation':ids[n:2*n], 'test':ids[:n]}


def region(record, mode):
    mask=np.ones(len(record['y']),dtype=bool)
    if mode=='sleep-window':
        sleep=np.flatnonzero(record['y']!=0)
        if not len(sleep):return np.zeros_like(mask)
        mask=(record['times']>=record['times'][sleep[0]]-1800)&(record['times']<=record['times'][sleep[-1]]+1800)
    return mask


def matrix(record, variant):
    if variant=='baseline':return record['baseline']
    if variant=='multimodal':return record['enhanced']
    return trailing_context(record['enhanced'],record['times'],record['valid'])


def train(records, ids, variant, mode, seed, trees, jobs):
    chosen=[r for r in records if r['subject'] in ids]
    x=np.concatenate([matrix(r,variant)[r['valid']&region(r,mode)] for r in chosen])
    y=np.concatenate([r['y'][r['valid']&region(r,mode)] for r in chosen])
    if len(np.unique(y))<5:raise ValueError('학습 분할에 5단계가 모두 필요합니다.')
    model=RandomForestClassifier(n_estimators=trees,min_samples_leaf=3,
                                 class_weight='balanced_subsample',random_state=seed,n_jobs=jobs)
    model.fit(x,y)
    return model


def probabilities(model,x):
    p=np.zeros((len(x),5));p[:,model.classes_.astype(int)]=model.predict_proba(x);return p


def metrics(y,pred):
    return {'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=range(5),average='macro',zero_division=0)),
            'report':classification_report(y,pred,labels=range(5),target_names=CLASSES,zero_division=0,output_dict=True),
            'confusion_matrix':confusion_matrix(y,pred,labels=range(5)).tolist(),'n_epochs':len(y)}


def evaluate_model(model,records,ids,variant,mode):
    ys=[];preds=[];per={}
    for subject in ids:
        sy=[];sp=[]
        for r in records:
            if r['subject']!=subject:continue
            mask=r['valid']&region(r,mode)
            if not mask.any():continue
            sy.extend(r['y'][mask]);sp.extend(model.predict(matrix(r,variant)[mask]))
        if not sy:raise ValueError(f'평가 가능한 구간 없음: subject={subject}')
        per[str(subject)]=metrics(sy,sp);ys.extend(sy);preds.extend(sp)
    result=metrics(ys,preds);result['per_subject']=per
    result['subject_mean_macro_f1']=float(np.mean([m['macro_f1'] for m in per.values()]))
    return result


def tune_threshold(model,records,ids,variant):
    candidates=[]
    for threshold in [.35,.5,.65,.8]:
        tp=fp=fn=0
        for r in records:
            if r['subject'] not in ids:continue
            s=evaluate(r['y'],probabilities(model,matrix(r,variant)),r['times'],r['valid'],threshold)
            tp+=s['matched'];fp+=s['alerts']-s['matched'];fn+=s['missed']
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        candidates.append({'threshold':threshold,'event_f1':f1,'tp':tp,'fp':fp,'fn':fn})
    # Threshold selected only on validation subjects; tie favors conservative value.
    best=max(candidates,key=lambda x:(x['event_f1'],x['threshold']))
    return best['threshold'],candidates


def plots(out, result):
    cm=np.asarray(result['confusion_matrix']);fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(cm,cmap='Blues');ax.set_xticks(range(5),CLASSES);ax.set_yticks(range(5),CLASSES)
    ax.set(xlabel='Predicted',ylabel='Reference',title='Held-out subjects (QC-passed epochs)')
    for i in range(5):
        for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--nights',type=int,nargs='+',choices=[1,2],default=[1])
    p.add_argument('--data-dir',type=Path,default=ROOT/'eeg_data')
    p.add_argument('--out',type=Path,default=ROOT/'output'/'eeg_v04')
    p.add_argument('--offline',action='store_true');p.add_argument('--download-only',action='store_true')
    p.add_argument('--smoke',action='store_true',help='2명 동작 점검: validation 없음, 성능 결론 금지')
    p.add_argument('--retries',type=int,default=3);p.add_argument('--timeout',type=float,default=60)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--trees',type=int,default=160)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--window',choices=['full','sleep-window'],default='full')
    p.add_argument('--learning-curve',action='store_true')
    args=p.parse_args()
    if len(set(args.subjects))!=len(args.subjects) or len(set(args.nights))!=len(args.nights):p.error('중복 참가자/박 지정')
    if args.retries<1 or args.timeout<=0 or args.trees<1:p.error('양수 인자 필요')
    if args.smoke:
        args.subjects=[0,1]; split={'train':[0],'validation':[],'test':[1]}
    else:split=split_subjects(args.subjects,args.seed)
    if args.out.exists() and any(args.out.iterdir()):p.error('출력 폴더가 비어 있지 않습니다. --out 새경로로 기존 결과를 보존하세요.')
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'run.json',{'status':'started','config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'split':split,'python':platform.python_version(),'versions':{'mne':mne.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'code_sha256':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}})
    paths=get_nights(args.subjects,args.nights,args.data_dir,offline=args.offline,retries=args.retries,timeout=args.timeout)
    write_json(args.out/'data_manifest.json',[{'subject':s,'night':n,'psg':str(a),'hypnogram':str(b),'psg_sha256':digest(a),'hypnogram_sha256':digest(b)} for s,n,a,b in paths])
    if args.download_only:
        write_json(args.out/'completion.json',{'status':'downloaded','nights':len(paths)});return
    records=[]
    for s,n,a,b in paths:
        print(f'특징 추출/캐시: 참가자 {s}, {n}박',flush=True)
        r=load_record(a,b,args.data_dir/'feature_cache_v03');r.update(subject=s,night=n);records.append(r)
    write_json(args.out/'quality.json',[{'subject':r['subject'],'night':r['night'],'scored_epochs':len(r['y']),
        'valid_epochs':int(r['valid'].sum()),'coverage':float(r['valid'].mean()),
        'excluded_by_stage':{CLASSES[c]:int(((r['y']==c)&~r['valid']).sum()) for c in range(5)}} for r in records])
    with (args.out/'quality_details.csv').open('w', newline='') as f:
        w=csv.writer(f);w.writerow(['subject','night','epoch_start_s','true_stage','qc_valid','reasons'])
        for r in records:
            for i, reason in enumerate(r.get('qc_reasons', ['unavailable']*len(r['y']))):
                w.writerow([r['subject'],r['night'],r['times'][i],CLASSES[r['y'][i]],int(r['valid'][i]),reason])
    write_json(args.out/'quality_reasons.json', [dict(subject=r['subject'],night=r['night'],
        reason_counts=dict(Counter(reason for row in r.get('qc_reasons',[]) for reason in row.split(';') if reason))) for r in records])
    variants=['baseline','multimodal','causal'];models={};validation={}
    for variant in variants:
        print('학습:',variant,flush=True)
        models[variant]=train(records,split['train'],variant,args.window,args.seed,args.trees,args.jobs)
        if split['validation']:
            validation[variant]=evaluate_model(models[variant],records,split['validation'],variant,args.window)
    selected=max(validation,key=lambda k:validation[k]['subject_mean_macro_f1']) if validation else 'causal'
    # Freeze variant and alert threshold BEFORE touching held-out test labels.
    threshold,tuning=tune_threshold(models[selected],records,split['validation'],selected) if validation else (.5,[])
    write_json(args.out/'selection.json',{'variant':selected,'threshold':threshold,'validation':validation,'threshold_tuning':tuning,'smoke':args.smoke})
    if args.learning_curve and validation:
        curve=[]
        for n in sorted(set([1,2,4,len(split['train'])])):
            if n>len(split['train']):continue
            model=train(records,split['train'][:n],selected,args.window,args.seed,args.trees,args.jobs)
            score=evaluate_model(model,records,split['validation'],selected,args.window)
            curve.append({'n_subjects':n,'subject_mean_macro_f1':score['subject_mean_macro_f1'],'balanced_accuracy':score['balanced_accuracy']})
        write_json(args.out/'learning_curve.json',curve)
        fig,ax=plt.subplots();ax.plot([r['n_subjects'] for r in curve],[r['subject_mean_macro_f1'] for r in curve],'o-')
        ax.set(xlabel='Training subjects',ylabel='Validation subject mean macro F1',title='Fixed validation set; single nested training order')
        fig.tight_layout();fig.savefig(args.out/'learning_curve.png',dpi=140);plt.close(fig)
    tests={v:evaluate_model(models[v],records,split['test'],v,args.window) for v in variants}
    write_json(args.out/'metrics.json',{'selected':selected,'smoke_only':args.smoke,'window':args.window,'test':tests})
    plots(args.out,tests[selected])
    transitions=[]; sustained=[]
    with (args.out/'predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['subject','night','epoch_start_s','decision_s','true_stage','prediction','qc_valid']+['p_'+c for c in CLASSES])
        for r in records:
            if r['subject'] not in split['test']:continue
            probs=probabilities(models[selected],matrix(r,selected))
            for i,prob in enumerate(probs):w.writerow([r['subject'],r['night'],r['times'][i],r['times'][i]+30,CLASSES[r['y'][i]],CLASSES[prob.argmax()] if r['valid'][i] else 'ABSTAIN',int(r['valid'][i])]+prob.tolist())
            sustained.append({'subject':r['subject'],'night':r['night'],'criteria':[evaluate_sustained(r['y'],probs,r['times'],r['valid'],m) for m in [2,5,10]]})
            transitions.append({'subject':r['subject'],'night':r['night'],**evaluate(r['y'],probs,r['times'],r['valid'],threshold)})
            fig,ax=plt.subplots(figsize=(12,4));ax.step(r['times']/3600,r['y'],where='post',label='Reference')
            pred=probs.argmax(axis=1).astype(float);pred[~r['valid']]=np.nan
            ax.step(r['times']/3600,pred,where='post',alpha=.6,label='Predicted');ax.set_yticks(range(5),CLASSES)
            ax.invert_yaxis();ax.set(xlabel='Hours from recording start',title=f'Subject {r["subject"]} night {r["night"]}');ax.legend()
            fig.tight_layout();fig.savefig(args.out/f'hypnogram_{r["subject"]}_{r["night"]}.png',dpi=120);plt.close(fig)
    write_json(args.out/'sustained_sleep.json',{'exploratory':True,'definition':'All sleep stages; strict continuity; QC/gaps reset; onset +/-120s and decision after reference confirmation, within reference bout. Not clinical SOL. No criterion selected on test.', 'records':sustained})
    write_json(args.out/'transitions.json',{'definition':'Adjacent reference Wake->N1; match alert END within (0,120] seconds after observing at least one reference epoch; one-to-one. Not clinical sleep latency.',
        'threshold':threshold,'n1_confirm_epochs':2,'records':transitions})
    # Training-only event-aligned summaries for hypothesis generation.
    with (args.out/'transition_profiles_train.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['subject','night','reference_start_s','offset_epochs','epoch_start_s']+records[0]['names'].tolist())
        for r in records:
            if r['subject'] not in split['train']:continue
            for event in reference_events(r['y'],r['times'],r['valid']):
                for offset in range(-10,5):
                    i=event+offset
                    if 0<=i<len(r['y']) and r['valid'][i] and r['times'][i]-r['times'][event]==offset*30:
                        w.writerow([r['subject'],r['night'],r['times'][event],offset,r['times'][i]]+r['enhanced'][i].tolist())
    joblib.dump({'model':models[selected],'variant':selected,'classes':CLASSES,'threshold':threshold,'split':split},args.out/'model.joblib')
    write_json(args.out/'completion.json',{'status':'completed','selected':selected,'smoke_only':args.smoke})
    print(json.dumps({k:tests[selected][k] for k in ['accuracy','balanced_accuracy','macro_f1']},indent=2))
    print('연구용 분류/전환 탐지 결과입니다. 수면 유도 효과 또는 임상 성능을 입증하지 않습니다.')


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\n사용자가 중단했습니다. 완료된 EDF/특징 캐시는 보존됩니다.',file=sys.stderr);sys.exit(130)
