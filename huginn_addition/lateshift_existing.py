"""CPU analysis of mon_main-20261110-* chunks, deduplicated by model/pid.

Uses saved ans_tok (the tokenizer output saved by run_collect). Computes accuracy
at T_STD and deep 48/64; Wilson 95% intervals for specified argmax changes;
8->32 correctness transitions and wrong->wrong changes; linear-interpolated
quartiles of top1-top2 log-probability margins for changed/unchanged answers;
and earliest sampled settling time through 32 (33 is deliberately excluded).
All available chunks are included. Conflicting duplicate observations fail.
Audits stored container accuracy counts, shard inventories, and readout NaNs.
"""
import argparse
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np

FAMS = ['add3','addsub','muladd','add4','sub3','add3w','arc_easy','arc_chal']
MODELS = ['L4','L8','L16','L32']
T_STD = [1,2,3,4,6,8,12,16,24,31,32,33]
PAIRS = [(4,32),(8,32),(16,32),(24,32),(8,16),(31,32),(32,33),(32,64)]

def rate(k, n):
    if not n:
        return dict(k=k,n=n,rate=None,ci95=None)
    p=k/n; z=NormalDist().inv_cdf(.975); d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return dict(k=k,n=n,rate=p,ci95=[float(c-h),float(c+h)])

def summarize(rows):
    out={'n':len(rows),'accuracy':{},'changes':{}}
    for t in T_STD+[48,64]:
        r=[x for x in rows if t in x['pred']]
        out['accuracy'][str(t)]=rate(sum(x['pred'][t]==x['ans'] for x in r),len(r))
    for a,b in PAIRS:
        r=[x for x in rows if a in x['pred'] and b in x['pred']]
        out['changes'][f'{a},{b}']=rate(sum(x['pred'][a]!=x['pred'][b] for x in r),len(r))
    tr={k:0 for k in ['right_right','right_wrong','wrong_right','wrong_wrong']}; wc=0
    margins={k:{str(t):[] for t in [8,32]} for k in ['changed','unchanged']}
    settle={str(t):0 for t in T_STD[:-1]}
    for x in rows:
        p=x['pred']; a=x['ans']; changed=p[8]!=p[32]
        key=('right' if p[8]==a else 'wrong')+'_'+('right' if p[32]==a else 'wrong'); tr[key]+=1
        wc+=int(key=='wrong_wrong' and changed)
        for t in [8,32]: margins['changed' if changed else 'unchanged'][str(t)].append(x['margin'][t])
        ts=T_STD[:-1]; first=next(t for j,t in enumerate(ts) if all(p[u]==p[32] for u in ts[j:]))
        for t in ts: settle[str(t)]+=int(first<=t)
    tr['wrong_wrong_changed']=rate(wc,tr['wrong_wrong']); out['transitions']=tr
    out['margins']={k:{t:dict(n=len(v),q25=float(np.quantile(v,.25)) if v else None,median=float(np.median(v)) if v else None,q75=float(np.quantile(v,.75)) if v else None) for t,v in d.items()} for k,d in margins.items()}
    out['settled_by']={t:rate(k,len(rows)) for t,k in settle.items()}
    return out

def main(results_root,out_dir):
    root=Path(results_root); dest=Path(out_dir)
    data={m:{} for m in MODELS}; audit={'shards':[],'duplicates':0,'nonfinite_readouts':0,'accuracy_checks':0,'accuracy_mismatches':[],'missing_shards':[]}
    for run in sorted(root.glob('mon_main-20261110-*')):
        containers=json.loads((run/'containers.json').read_text()) if (run/'containers.json').exists() else {}
        for key in containers:
            m,s=key.split(':')
            if m in MODELS and not (run/m/('s'+s)).is_dir(): audit['missing_shards'].append(str(run/m/('s'+s)))
        for m in MODELS:
            for shard in sorted((run/m).glob('s*')):
                meta=json.loads((shard/'meta.json').read_text()) if (shard/'meta.json').exists() else {}
                counts={}; chunk_names=[]; n=0
                for path in sorted(shard.glob('c*_*.npz')):
                    chunk_names.append(path.name)
                    with np.load(path,allow_pickle=False) as z:
                        ts=list(map(int,z['T'])); assert set(T_STD)<=set(ts),path
                        ids=z['top_id'][:,:,0]; lp=z['top_lp']; ans=z['ans_tok']; pids=z['pid']
                        for k in ['top_lp','lp_ans']:
                            audit['nonfinite_readouts']+=int((~np.isfinite(z[k])).sum())
                        for i,pid in enumerate(pids):
                            pid=int(pid); fam=FAMS[pid//100000]; n+=1
                            row={'ans':int(ans[i]),'pred':dict(zip(ts,map(int,ids[i]))),'margin':dict(zip(ts,map(float,lp[i,:,0]-lp[i,:,1])))}
                            for t in T_STD:
                                c=counts.setdefault(fam,{}).setdefault(str(t),[0,0]); c[0]+=int(row['pred'][t]==row['ans']); c[1]+=1
                            if pid in data[m]:
                                audit['duplicates']+=1; old=data[m][pid]; assert old['ans']==row['ans'],(m,pid)
                                for t in old['pred'].keys() & row['pred'].keys():
                                    assert old['pred'][t]==row['pred'][t] and old['margin'][t]==row['margin'][t],(m,pid,t)
                                old['pred'].update(row['pred']); old['margin'].update(row['margin'])
                            else: data[m][pid]=row
                saved=containers.get(m+':'+shard.name[1:],meta).get('accuracy_counts',{})
                for fam,d in saved.items():
                    for t,c in d.items():
                        audit['accuracy_checks']+=1
                        if counts.get(fam,{}).get(t,[0,0])!=c: audit['accuracy_mismatches'].append([str(shard),fam,t,c,counts.get(fam,{}).get(t,[0,0])])
                audit['shards'].append(dict(path=str(shard),status=meta.get('status'),n=n,items_requested=meta.get('items_requested'),items_committed=meta.get('items_committed'),missing_chunks=sorted(set(meta.get('chunks',[]))-set(chunk_names))))
    result={'definitions':__doc__,'audit':audit,'models':{}}
    for m,rows in data.items():
        result['models'][m]={f:summarize([rows[p] for p in sorted(rows) if f=='pooled' or FAMS[p//100000]==f]) for f in FAMS+['pooled'] if f=='pooled' or any(FAMS[p//100000]==f for p in rows)}
    assert audit['nonfinite_readouts']==0,audit
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def pct(v,interval=True):
        if v['rate'] is None:return '—'
        s=f"{100*v['rate']:.1f}%"
        if interval:s+=f" [{100*v['ci95'][0]:.1f}, {100*v['ci95'][1]:.1f}]"
        return s
    lines=['# Existing-problem late shifts','', 'Intervals are 95% Wilson. Deep-only denominators appear in the sample-size table. Margins are log-probability differences; brackets give Q1, Q3. Settling ignores loop 33.','']
    families=[f for f in FAMS+['pooled'] if any(f in d for d in result['models'].values())]
    def table(title,fn):
        lines.extend(['## '+title,'','| Family | '+' | '.join(MODELS)+' |','|---|'+'---|'*len(MODELS)])
        for f in families: lines.append('| '+f+' | '+' | '.join(fn(result['models'][m][f]) if f in result['models'][m] else '—' for m in MODELS)+' |')
        lines.append('')
    table('Sample sizes (main / deep)',lambda x:f"{x['n']} / {x['accuracy']['64']['n']}")
    for t in T_STD+[48,64]:table(f'Accuracy at {t}',lambda x,t=t:pct(x['accuracy'][str(t)],False))
    for a,b in PAIRS:table(f'Answer change {a} → {b}',lambda x,a=a,b=b:pct(x['changes'][f'{a},{b}']))
    table('8 → 32 transitions (RR / RW / WR / WW)',lambda x:' / '.join(str(x['transitions'][k]) for k in ['right_right','right_wrong','wrong_right','wrong_wrong']))
    table('Changed among wrong → wrong',lambda x:pct(x['transitions']['wrong_wrong_changed']))
    for k in ['changed','unchanged']:
        for t in [8,32]:
            def fmt(x,k=k,t=t):
                v=x['margins'][k][str(t)]
                return '—' if not v['n'] else f"{v['median']:.4f} [{v['q25']:.4f}, {v['q75']:.4f}]"
            table(f'{k.capitalize()} answer margins at {t}',fmt)
    for t in [1,2,3,4,6,8,12,16,24]:table(f'Settled by {t}',lambda x,t=t:pct(x['settled_by'][str(t)],False))
    lines.extend(['## Audit','',f"Checked {audit['accuracy_checks']} stored accuracy counts; {len(audit['accuracy_mismatches'])} mismatches. Duplicate pids: {audit['duplicates']}. Nonfinite readouts: {audit['nonfinite_readouts']}. Missing declared shards: {len(audit['missing_shards'])}."])
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('results_root'); p.add_argument('out_dir'); a=p.parse_args(); main(a.results_root,a.out_dir)
