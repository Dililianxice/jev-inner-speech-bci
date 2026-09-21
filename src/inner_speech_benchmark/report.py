"""Reproducible tables and local scientific figures; no model fitting."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score
O=Path('outputs')

def write_csv(path, rows):
    if not rows: raise ValueError(f'Empty table: {path}')
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def read(name):return list(csv.DictReader((O/name).open()))

def main(output_dir=None, figure_dir=None):
    global O
    O=Path(output_dir) if output_dir is not None else O
    figure_dir=Path(figure_dir) if figure_dir is not None else O
    figure_dir.mkdir(parents=True,exist_ok=True)
    m=read('metrics.csv');p=read('predictions.csv');audit=json.loads((O/'audit.json').read_text())
    assert len(m)==96 and len(audit)==4
    groups=defaultdict(list)
    for r in p:groups[tuple(r[k] for k in ['subject','model','task','requested_seconds','effective_seconds','block'])].append(r)
    blockrows=[]
    for k,rr in groups.items():
        row=dict(zip(['subject','model','task','requested_seconds','effective_seconds','block'],k))
        row.update(n=len(rr),correct=sum(x['truth']==x['prediction'] for x in rr),accuracy=np.mean([x['truth']==x['prediction'] for x in rr]),balanced_accuracy=balanced_accuracy_score([x['truth'] for x in rr],[x['prediction'] for x in rr]));blockrows.append(row)
    write_csv(O/'block_metrics.csv',blockrows)
    recalls=[]
    for r in m:
        names=json.loads(r['classes']);c=np.array(json.loads(r['confusion']))
        for i,name in enumerate(names):recalls.append({k:r[k] for k in ['subject','model','task','requested_seconds','effective_seconds']}|dict(label=name,n=int(c[i].sum()),correct=int(c[i,i]),recall=float(c[i,i]/c[i].sum())))
    write_csv(O/'class_recalls.csv',recalls)
    fig,axes=plt.subplots(1,4,figsize=(12.6,4.4),sharey=True)
    for ax,a in zip(axes,audit):
        for model,color,style,label in [('shrinkage_lda','#0077BB','o-','Shrinkage LDA'),('diagonal_lda','#EE7733','s--','Diagonal LDA')]:
            rr=sorted([r for r in m if r['subject']==a['subject'] and r['model']==model and r['task']=='imagined_word'],key=lambda r:float(r['requested_seconds']))
            ax.plot([float(r['effective_seconds']) for r in rr],[100*float(r['accuracy']) for r in rr],style,c=color,lw=2,ms=5,label=label)
        n=rr[0]['test_n'];ax.set_title(f"{a['subject'].upper()} (test n={n})")
        ax.axhline(100/7,c='.5',ls=':',lw=1);ax.set_ylim(0,100);ax.set_xticks([.25,.5,1.]);ax.set_xlabel('Post-go prefix (s)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Seven-word accuracy (%)');axes[-1].legend(fontsize=8,loc='upper left',frameon=False)
    fig.suptitle('Inner-speech word identity: forward held-out recording blocks',fontsize=14)
    fig.text(.5,.02,'All released threshold-crossing channels; no new smoothing or test-block centering.\nT12/T16 use 240 ms for the nominal 250 ms window. These are go-aligned offline results, not autonomous thought-onset latency.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.26,top=.8,wspace=.15)
    for ext in ['png','pdf']:fig.savefig(figure_dir/('word_prefix.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(13,4.5))
    for ax,a in zip(axes,audit):
        r=next(r for r in m if r['subject']==a['subject'] and r['model']=='shrinkage_lda' and r['task']=='behavior' and r['requested_seconds']=='0.5')
        c=np.array(json.loads(r['confusion']));v=c/c.sum(1,keepdims=True)
        ax.imshow(v,cmap='Blues',vmin=0,vmax=1)
        for i in range(4):
            for j in range(4):ax.text(j,i,str(c[i,j]),ha='center',va='center',color='white' if v[i,j]>.5 else 'black',fontsize=9)
        ax.set_xticks(range(4),['Attempt','Idle','Imagine','Listen'],rotation=45,ha='right',fontsize=8)
        ax.set_yticks(range(4),['Attempt','Idle','Imagine','Listen'],fontsize=8)
        ax.set_title(f"{a['subject'].upper()} / macro recall {100*float(r['balanced_accuracy']):.1f}%",fontsize=10);ax.set_xlabel('Predicted condition')
    axes[0].set_ylabel('True instructed condition')
    fig.suptitle('500 ms condition decoding: counts shown, color is row-normalized',fontsize=14)
    fig.text(.5,.02,'T12 listening uses the source-defined DELAY epoch; other conditions use GO.\nCondition and task-phase cues may contribute. This does not isolate volition; each participant has only 4–5 test idle trials.',ha='center',fontsize=9)
    fig.subplots_adjust(bottom=.26,top=.82,wspace=.4)
    for ext in ['png','pdf']:fig.savefig(figure_dir/('condition_confusion.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    # Compact factual tables for the accompanying interpretation report.
    lines=['| 参与者 | 训练/测试内言语试次 | 240/250 ms | 500 ms | 1 s | 500 ms 块内置换均值 |','|---|---:|---:|---:|---:|---:|']
    for a in audit:
        rr=sorted([r for r in m if r['subject']==a['subject'] and r['model']=='shrinkage_lda' and r['task']=='imagined_word'],key=lambda r:float(r['requested_seconds']))
        lines.append(f"| {a['subject'].upper()} | {rr[0]['train_n']}/{rr[0]['test_n']} | {float(rr[0]['accuracy']):.1%} | {float(rr[1]['accuracy']):.1%} | {float(rr[2]['accuracy']):.1%} | {float(rr[1]['null_balanced_mean']):.1%} |")
    (O/'summary_table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'))
    parser.add_argument('--figure-dir',type=Path,default=Path('figures'))
    args=parser.parse_args();main(args.output_dir,args.figure_dir)
