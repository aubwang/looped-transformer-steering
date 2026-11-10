"""Fixed late-shift decisions on paired whole-vocabulary argmax observations."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from lateshift_existing import rate, summarize

T=[1,2,3,4,6,8,12,16,24,31,32,33]


def verdict(change, flicker, acc, dacc, chance, n, confirm=False):
    usable=chance+.10<=acc<=.90 and chance+.10<=dacc<=.90
    stable=flicker<=change/3 or abs(flicker-change/3)<1e-15
    if confirm:
        accepted=n>=300 and change>=.10 and rate(round(change*n),n)['ci95'][0]>.07 and usable and stable
        return 'ACCEPTED' if accepted else 'NOT ACCEPTED'
    if change>=.10 and not (usable and stable): return 'CHANGES BUT UNUSABLE'
    if change>=.15: return 'QUALIFIES'
    if change>=.10: return 'BORDERLINE'
    return 'SETTLED'


def analyze(L,D,items,confirm=False):
    li={int(p):j for j,p in enumerate(L['pid'])}; di={int(p):j for j,p in enumerate(D['pid'])}
    assert len(li)==len(L['pid']) and len(di)==len(D['pid']), 'duplicate pid'
    ts=list(map(int,L['T'])); assert ts==T
    out={}; common=li.keys()&di.keys()
    for fam in dict.fromkeys(it['fam'] for it in items):
        its=[it for it in items if it['fam']==fam and it['pid'] in common]; n=len(its)
        if n<48: out[fam]=dict(n=n,verdict='INSUFFICIENT'); continue
        rows=[]; rel=[]; dc=0; settling={str(t):0 for t in T[:-1]}
        for it in its:
            j=li[it['pid']]; k=di[it['pid']]; ans=int(L['ans_tok'][j]); assert ans==int(D['ans_tok'][k])
            pred=dict(zip(ts,map(int,L['top_id'][j,:,0]))); margin=dict(zip(ts,map(float,L['top_lp'][j,:,0]-L['top_lp'][j,:,1])))
            rows.append(dict(ans=ans,pred=pred,margin=margin)); rel.append(L['rel_step'][j,1:]); dc+=int(D['top_id'][k,0,0]==ans)
            first=next(t for a,t in enumerate(T[:-1]) if all(pred[u]==pred[32] for u in T[:-1][a:]))
            settling[str(first)]+=1
        r=summarize(rows)
        # Reuse transition/margin/Wilson implementation; omit unrelated deep-loop fields.
        r['accuracy']={str(t):r['accuracy'][str(t)] for t in T}
        r['changes'].pop('32,64',None)
        tr=r['transitions']; discord=tr['right_wrong']+tr['wrong_right']
        r['mcnemar_exact_p']=float(binomtest(tr['right_wrong'],discord,.5).pvalue) if discord else 1.
        r['accuracy_D']=rate(dc,n); r['settling_distribution']=settling
        r['median_rel_step']={str(t):float(v) for t,v in enumerate(np.median(rel,axis=0),1)}
        c=r['changes']['8,32']['rate']; flicker=r['changes']['31,32']['rate']; a=r['accuracy']['32']['rate']; chance=its[0]['chance']
        r.update(chance=chance,usable=chance+.10<=a<=.90 and chance+.10<=dc/n<=.90,not_flicker=flicker<=c/3 or abs(flicker-c/3)<1e-15,
                 verdict=verdict(c,flicker,a,dc/n,chance,n,confirm))
        out[fam]=r
    candidates=[f for f,r in out.items() if r['verdict'] in ('ACCEPTED','QUALIFIES','BORDERLINE')]
    missing=[f for f,r in out.items() if r['verdict']=='INSUFFICIENT']
    # Draft 1.1: an incomplete screen cannot support the negative conclusion.
    if candidates: step='continue monitoring study' if confirm else 'confirm on 300 fresh problems (at most two families, largest c(8,32) first, ties by family order)'
    elif missing: step='screen incomplete: no conclusion ('+', '.join(missing)+' below 48 paired problems)'
    else: step='no candidate among adequately sampled families: drop monitoring question for this model pair'
    return dict(confirm=confirm,families=out,candidates=sorted(candidates,key=lambda f:-out[f]['changes']['8,32']['rate']),insufficient=missing,next_step=step)


def main(run_dir,items_json,confirm=False):
    dest=Path(run_dir); master=json.loads(Path(items_json).read_text()); items=master['items'] if isinstance(master,dict) else master
    containers=json.loads((dest/'containers.json').read_text())  # required: collection status and gates
    assert all(containers.get(k,{}).get('status') in ('complete','partial') for k in ('L32','D')), 'failed collection gates/container'
    n_screen=master['counts']['n'] if isinstance(master,dict) else 64; cfg=containers['L32']['cfg']
    items=[it for it in items if cfg.get('i0',0)<=it['i']<cfg.get('i0',0)+cfg.get('n',n_screen)]  # the planned problems of this run
    if cfg.get('fams'): items=[it for it in items if it['fam'] in cfg['fams'].split(',')]  # with --fams only those were planned
    if confirm: assert items and all(it['i']>=n_screen for it in items), 'confirmation requires fresh indices'
    L=dict(np.load(dest/'L32.npz',allow_pickle=False)); D=dict(np.load(dest/'D.npz',allow_pickle=False))
    for z in (L,D):  # nothing completed: empty arrays, every family INSUFFICIENT
        if 'pid' not in z: z.update(pid=np.zeros(0,np.int64),T=np.array(T))
    result=analyze(L,D,items,confirm); (dest/'analysis_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Late-shift screen','','| Family | n | Verdict | c(8,32), 95% CI | L32 acc | D acc |','|---|---:|---|---|---:|---:|']
    for f,r in result['families'].items():
        if r['n']<48: lines.append(f"| {f} | {r['n']} | INSUFFICIENT | — | — | — |"); continue
        c=r['changes']['8,32']; lines.append(f"| {f} | {r['n']} | {r['verdict']} | {c['rate']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}] | {r['accuracy']['32']['rate']:.3f} | {r['accuracy_D']['rate']:.3f} |")
    for f,r in result['families'].items():
        if r['n']<48: continue
        lines+=['',f'## {f}','','| Loop | Accuracy | Median relative step |','|---:|---:|---:|']
        lines += [f"| {t} | {r['accuracy'][str(t)]['rate']:.3f} | {r['median_rel_step'][str(t)]:.6g} |" for t in T]
        lines+=['','| Pair | Change | 95% Wilson |','|---|---:|---|']
        lines += [f"| {p} | {v['rate']:.3f} | {v['ci95'][0]:.3f}, {v['ci95'][1]:.3f} |" for p,v in r['changes'].items()]
        lines+=['',f"Transitions: {r['transitions']}. Exact McNemar p = {r['mcnemar_exact_p']:.6g}.",
                '',f"Settling counts: {r['settling_distribution']}.",'','| Group | Loop | Margin Q1 / median / Q3 |','|---|---:|---|']
        for group,d in r['margins'].items():
            for t,v in d.items(): lines.append(f"| {group} | {t} | {v['q25']} / {v['median']} / {v['q75']} |")
        lines+=['','Relative step medians (all loops): '+json.dumps(r['median_rel_step'])]
    if not confirm: lines+=['','QUALIFIES and BORDERLINE require confirmation on 300 fresh problems.']
    else: lines+=['','Accepted families: '+', '.join(f for f,r in result['families'].items() if r['verdict']=='ACCEPTED')]
    lines+=['',result['next_step']+'.']
    (dest/'summary.md').write_text('\n'.join(lines)+'\n'); return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run_dir'); p.add_argument('items_json'); p.add_argument('--confirm',action='store_true'); a=p.parse_args(); main(a.run_dir,a.items_json,a.confirm)
