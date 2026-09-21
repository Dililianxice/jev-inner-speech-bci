"""Forward-block, released-bin causal-prefix exploration of Stanford inner speech.
No provider API, no test-block centering, no temporal smoothing or test tuning.
"""
import argparse,csv,hashlib,json,re
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix,balanced_accuracy_score
from threadpoolctl import threadpool_limits
W=Path('data');O=Path('outputs')
MODELS=['shrinkage_lda','diagonal_lda'];DURATIONS=[.25,.5,1.]
THRESHOLDS=[0.,.5,.7,.9,.99,1.01]
CONFIG=dict(features='All binnedTX channels, rates; exclude spikePow in all participants because T12 duplicate; no region selection',
    split='Last max(1,floor(.2*nblocks)) blocks test; preceding one validation; all earlier blocks train; no refit with validation',
    models=MODELS,windows=DURATIONS,primary_seconds=.5,rounding='floor bins: requested250ms is240ms for20ms data',
    tasks=['imagined_word','attempted_word','listening_word','behavior'],priors='Equal per class; scores not calibrated probabilities',
    temporal_indexing='MATLAB inclusive epochs [s,e] -> Python [s-1,e); T12 listening aligns to DELAY as author Fig5_ABCD; all others GO. First bin included; latency relative to event-bin edge, not subjective onset',
    null='49 train-label permutations within original blocks, 500ms only; other tasks and test labels untouched; diagnostic, not adjusted confirmatory p-values',
    stream='Fixed500ms cue-condition classifier and imagined-word classifier, trained on first500ms source-aligned epoch features; update100ms within each block; no trial boundary resets',
    trigger='two consecutive same imagined word with behavior argmax imagined and score>=threshold; disarm until two nonimagined argmax updates AND1s refractory',
    thresholds=THRESHOLDS,selection='Validation only: <=1 event/min across all recorded time outside imagined GO epochs AND zero imagined-go duplicates; maximize correct first word within1s / all imagined trials; tie lowest threshold; 1.01 mute candidate',
    stream_scoring='First event in each imagined GO epoch determines outcome; fast<=1s, late separate; all trials denominator. False-output exposure is every recorded bin outside imagined GO epochs, including delays, ITIs, and attempted/listening/idle epochs. Source-aligned condition counts remain diagnostic. Attempted speech is non-target only under this imagined-only channel design, not absence of intention',
    limitations=['Go-relative decoding, not spontaneous thought onset','Behavior is cued task condition, not isolated volition','Released-bin extraction causal, upstream reference/filter/threshold calibration not verified causal','Sparse do-nothing exposure; no daily-use false activation guarantee','One day per participant; no cross-day transfer'],seed=20260921)

def write_csv(path, rows):
    """Write records with a stable header and newline convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'Cannot infer columns for empty table: {path}')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def forward_split(block_sequence):
    """Keep complete recording blocks in chronological train/validation/test sets."""
    blocks=list(dict.fromkeys(np.asarray(block_sequence).ravel().tolist()))
    ntest=max(1,len(blocks)//5)
    split={'train':blocks[:-ntest-1],'validation':[blocks[-ntest-1]],'test':blocks[-ntest:]}
    if not split['train']:
        raise ValueError('At least three chronologically ordered blocks are required')
    return split

def parse_cue(c):
    low=str(c).lower()
    if 'nothing' in low.replace('_',''):return 'idle',''
    word=re.match(r'[a-z]+',low).group()
    if 'imagined' in low:return 'imagined',word
    if 'attempted' in low or 'mimed' in low:return 'attempted',word
    if 'listen' in low or '$0' in low:return 'listening',word
    raise ValueError('Unmapped cue '+str(c))

def read_data(p):
    a=loadmat(p,simplify_cells=True,variable_names=['binnedTX','blockNum','goTrialEpochs','delayTrialEpochs','trialCues','cueList','binSize','chanSetNames'])
    x=np.asarray(a['binnedTX']);b=np.asarray(a['blockNum'],int).ravel();dt=float(a['binSize'])/1000
    g=np.asarray(a['goTrialEpochs'],int);d=np.asarray(a['delayTrialEpochs'],int)
    starts=g[:,0]-1;ends=g[:,1];delay=d[:,0]-1
    assert dt in [.01,.02] and len(b)==len(x) and np.isfinite(x).all()
    assert np.all(starts>=0) and np.all(ends<=len(x)) and np.all(ends>starts)
    assert np.all(starts[1:]>=ends[:-1]) and np.all(b[starts]==b[ends-1]) and np.all(b[delay]==b[starts])
    labels=np.asarray(a['trialCues'],int).ravel()-1;cues=list(map(str,np.asarray(a['cueList']).ravel()))
    mapped=[parse_cue(cues[i]) for i in labels];behavior=np.array([a[0] for a in mapped]);word=np.array([a[1] for a in mapped])
    alignment=np.full(len(starts),'go',dtype='<U5')
    if p.name.startswith('t12'):
        listening=behavior=='listening';starts[listening]=delay[listening];ends[listening]=d[listening,1];alignment[listening]='delay'
    assert np.all(starts[1:]>=ends[:-1]) and np.all(b[starts]==b[ends-1])
    blocks=list(dict.fromkeys(b.tolist()));split=forward_split(blocks)
    assert all(len(np.flatnonzero(b==k))==np.ptp(np.flatnonzero(b==k))+1 for k in blocks)
    for part,bb in split.items():
        for state in ['imagined','attempted','listening']:
            assert len(set(word[np.isin(b[starts],bb)&(behavior==state)]))==7
        assert set(behavior[np.isin(b[starts],bb)])=={'imagined','attempted','listening','idle'}
    return dict(subject=p.name[:3],file=p.name,x=x,b=b,dt=dt,start=starts,end=ends,delay=delay,block=b[starts],
                behavior=behavior,word=word,cues=cues,labels=labels,alignment=alignment,split=split,arrays=list(map(str,np.atleast_1d(a['chanSetNames']))))

def prefix(data,duration):
    n=int(np.floor(duration/data['dt']+1e-9));s=data['start'];assert np.all(s+n<=data['end'])
    return np.stack([data['x'][i:i+n].mean(0)/data['dt'] for i in s]),n*data['dt']

class DiagonalLDA:
    def fit(self,x,y):
        self.classes_=np.unique(y);self.means=np.stack([x[y==c].mean(0) for c in self.classes_])
        residual=np.vstack([x[y==c]-self.means[i] for i,c in enumerate(self.classes_)])
        v=(residual**2).sum(0)/(len(y)-len(self.classes_));self.var=np.maximum(v,max(1e-8,float(v.mean())*.001))
        return self
    def predict_proba(self,x):return softmax(-.5*((x[:,None,:]-self.means[None,:,:])**2/self.var).sum(2),axis=1)
    def predict(self,x):return self.classes_[self.predict_proba(x).argmax(1)]

def fit(name,x,y):
    if name=='diagonal_lda':return DiagonalLDA().fit(x,y)
    return LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=np.ones(len(np.unique(y)))/len(np.unique(y))).fit(x,y)

def trigger(times,state_prob,state_classes,words,threshold):
    # No epoch or cue input; caller resets only at recording-block boundaries.
    events=[];armed=True;inactive=0;last=-np.inf;streak=0;previous=None
    imagined=int(np.flatnonzero(state_classes=='imagined')[0])
    for t,p,w in zip(times,state_prob,words):
        if state_classes[p.argmax()]!='imagined':
            inactive+=1;streak=0;previous=None
            if inactive>=2 and t-last>=1.-1e-9:armed=True
            continue
        inactive=0
        if not armed or p[imagined]<threshold:streak=0;previous=None;continue
        streak=streak+1 if w==previous else 1;previous=w
        if streak>=2:
            events.append(dict(time=float(t),word=str(w),score=float(p[imagined])));armed=False;last=t;streak=0;previous=None
    return events

def stream_predictions(data,models,part):
    state,word=models;streams=[];n=int(.5/data['dt']);step=int(round(.1/data['dt']))
    for block in data['split'][part]:
        ix=np.flatnonzero(data['b']==block);x=data['x'][ix]
        cs=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(x,axis=0,dtype=float)])
        end=np.arange(n,len(x)+1,step);z=(cs[end]-cs[end-n])/(n*data['dt'])
        streams.append(dict(block=block,time=(ix[0]+end)*data['dt'],prob=state.predict_proba(z),words=word.predict(z),classes=state.classes_))
    return streams

def events_for(streams,threshold):
    return [dict(block=s['block'],**e) for s in streams for e in trigger(s['time'],s['prob'],s['classes'],s['words'],threshold)]

def score_events(data,events,part):
    ids=np.flatnonzero(np.isin(data['block'],data['split'][part]));rows=[];assigned=set()
    block_ids=data['split'][part]
    replay_seconds=float(np.isin(data['b'],block_ids).sum()*data['dt'])
    imagined_seconds=0.
    totals=dict(imagined_trials=0,fast_correct=0,fast_wrong=0,late_first=0,miss=0,full_go_correct=0,duplicates=0,nonimagined_events=0,nonimagined_seconds=0.,outside_scored_events=0,continuous_nontarget_events=0,continuous_nontarget_seconds=0.)
    by_state={s:dict(events=0,seconds=0.) for s in ['idle','listening','attempted']}
    for i in ids:
        start=data['start'][i]*data['dt'];end=data['end'][i]*data['dt'];state=data['behavior'][i]
        matches=[j for j,e in enumerate(events) if e['block']==data['block'][i] and start<=e['time']<end]
        assigned.update(matches)
        if state!='imagined':
            totals['nonimagined_events']+=len(matches);totals['nonimagined_seconds']+=end-start
            by_state[state]['events']+=len(matches);by_state[state]['seconds']+=end-start;continue
        totals['imagined_trials']+=1;imagined_seconds+=end-start;totals['duplicates']+=max(0,len(matches)-1)
        e=events[matches[0]] if matches else None;delay=e['time']-start if e else None
        if e is None:outcome='miss'
        elif delay>1.+1e-9:outcome='late_first'
        else:outcome='fast_correct' if e['word']==data['word'][i] else 'fast_wrong'
        totals[outcome]+=1
        if e and e['word']==data['word'][i]:totals['full_go_correct']+=1
        rows.append(dict(trial=int(i),block=int(data['block'][i]),truth=str(data['word'][i]),prediction=e['word'] if e else '',
            outcome=outcome,go_relative_seconds=delay if e else '',go_events=len(matches)))
    totals['outside_scored_events']=len(events)-len(assigned)
    totals['continuous_nontarget_events']=totals['nonimagined_events']+totals['outside_scored_events']
    totals['continuous_nontarget_seconds']=replay_seconds-imagined_seconds
    assert totals['continuous_nontarget_seconds']>=0
    for state,r in by_state.items():
        totals[state+'_events']=r['events'];totals[state+'_seconds']=r['seconds']
    return totals,rows

def choose_threshold(rows):
    eligible=[r for r in rows if r['continuous_nontarget_events']/(r['continuous_nontarget_seconds']/60)<=1 and r['duplicates']==0]
    return min(eligible,key=lambda r:(-r['fast_correct']/r['imagined_trials'],r['threshold']))['threshold']

def main(data_dir=None, output_dir=None):
    global W,O
    W=Path(data_dir) if data_dir is not None else W
    O=Path(output_dir) if output_dir is not None else O
    O.mkdir(parents=True,exist_ok=True);files=sorted(W.glob('*.mat'));assert len(files)==4, f'Expected four MAT files in {W}, found {len(files)}'
    manifest=dict(config=CONFIG,files={p.name:sha(p) for p in files},code_sha256=sha(Path(__file__)))
    mp=O/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2)+'\n')
    metrics=[];predictions=[];audits=[];stream_metrics=[];stream_trials=[];stream_events=[];validation=[];null_rows=[]
    for file in files:
        data=read_data(file);subject=data['subject'];train=np.isin(data['block'],data['split']['train']);test=np.isin(data['block'],data['split']['test'])
        audit=dict(subject=subject,dt=data['dt'],channels=data['x'].shape[1],arrays=data['arrays'],split=data['split'],cues=[dict(code=i+1,text=c,behavior=parse_cue(c)[0],word=parse_cue(c)[1]) for i,c in enumerate(data['cues'])],coverage=[])
        for part,bb in data['split'].items():
            for block in bb:
                for state in ['imagined','attempted','listening','idle']:
                    ids=np.flatnonzero((data['block']==block)&(data['behavior']==state))
                    audit['coverage'].append(dict(part=part,block=block,state=state,trials=len(ids),alignment=str(data['alignment'][ids[0]]),epoch_seconds=float(np.sum(data['end'][ids]-data['start'][ids])*data['dt'])))
        audits.append(audit)
        for duration in DURATIONS:
            x,effective=prefix(data,duration)
            for name in MODELS:
                trained={}
                for task in CONFIG['tasks']:
                    state=task.removesuffix('_word');use=np.ones(len(train),bool) if task=='behavior' else data['behavior']==state
                    y=data['behavior'] if task=='behavior' else data['word'];tr=train&use;te=test&use
                    clf=fit(name,x[tr],y[tr]);trained[task]=clf;pred=clf.predict(x[te]);classes=clf.classes_
                    meta=dict(subject=subject,model=name,task=task,requested_seconds=duration,effective_seconds=effective)
                    acc=float(np.mean(pred==y[te]));ba=float(balanced_accuracy_score(y[te],pred));null=[]
                    if duration==.5:
                        rng=np.random.default_rng(CONFIG['seed']);block=data['block'][tr]
                        for k in range(49):
                            yy=y[tr].copy()
                            for b in np.unique(block):
                                ix=np.flatnonzero(block==b);yy[ix]=rng.permutation(yy[ix])
                            pp=fit(name,x[tr],yy).predict(x[te]);nn=float(balanced_accuracy_score(y[te],pp));null.append(nn)
                            null_rows.append({**meta,'permutation':k,'balanced_accuracy':nn})
                    metrics.append({**meta,'train_n':int(tr.sum()),'test_n':int(te.sum()),'accuracy':acc,'balanced_accuracy':ba,'chance':1/len(classes),
                        'null_balanced_mean':float(np.mean(null)) if null else '', 'classes':json.dumps(classes.tolist()),'confusion':json.dumps(confusion_matrix(y[te],pred,labels=classes).tolist())})
                    for i,predicted in zip(np.flatnonzero(te),pred):predictions.append({**meta,'trial':int(i),'block':int(data['block'][i]),'truth':str(y[i]),'prediction':str(predicted)})
                if duration==.5:
                    models=trained['behavior'],trained['imagined_word'];streams=stream_predictions(data,models,'validation');vr=[]
                    for th in THRESHOLDS:
                        score,_=score_events(data,events_for(streams,th),'validation');vr.append(dict(subject=subject,model=name,threshold=th,**score))
                    chosen=choose_threshold(vr)
                    validation.extend([{**r,'selected':int(r['threshold']==chosen)} for r in vr])
                    events=events_for(stream_predictions(data,models,'test'),chosen);score,trials=score_events(data,events,'test')
                    meta=dict(subject=subject,model=name,threshold=chosen)
                    stream_metrics.append({**meta,**score});stream_trials.extend([{**meta,**r} for r in trials]);stream_events.extend([{**meta,**r} for r in events])
            print(subject,duration,'complete',flush=True)
        # Save completed subjects incrementally so subsequent interruption does not erase results.
        for filename,rows in [('metrics',metrics),('predictions',predictions),('nulls',null_rows),('stream_metrics',stream_metrics),('stream_trials',stream_trials),('stream_validation',validation)]:write_csv(O/(filename+'.csv'),rows)
        if stream_events:write_csv(O/'stream_events.csv',stream_events)
        else:
            with (O/'stream_events.csv').open('w',newline='') as f:csv.writer(f).writerow(['subject','model','threshold','block','time','word','score'])
        (O/'audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    (O/'COMPLETED.json').write_text(json.dumps(dict(subjects=[a['subject'] for a in audits],manifest_sha256=sha(mp),metric_rows=len(metrics)),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True,help='Extracted Dryad folder containing the four MAT files')
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'),help='Directory for derived tables (default: outputs)')
    args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.data_dir,args.output_dir)
