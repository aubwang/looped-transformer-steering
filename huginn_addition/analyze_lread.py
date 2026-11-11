"""Fixed-design analysis: python -I huginn_addition/analyze_lread.py RUN_DIR."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ctrlprobe import evaluate, boot_median
from analyze_ouro import primary, coord_control, rotation

# Draft 1.2 wording (after the pre-run review, before any production run); the numerical classifier is unchanged.
STRONG = ('The sampled C3 directions have a strong signed deficit under this rule: L32 is on Huginn\'s side of the '
          'Huginn/Ouro contrast under this protocol. This is a signed self-gain relative to the trace; it does not '
          'establish damping, an explanation of the lever result, or a property of the whole read space.')
ZERO = ('The median probe-estimated trace is near zero and no C3 deficit was detected. This does not establish '
        'uniformly small self-gains (eigenvalues of opposite sign can cancel). The write-up must not present the '
        'one-loop loss of a readout-gradient push on L32 as something that distinguishes the readout from a typical '
        'direction, and must report the absolute self-gains, one-step norms and self-cosines of probes, C3 and task '
        'directions, and any POSITIVE EXCESS sub-label.')
OTHER = 'Reported as measured, with absolute self-gains, norms and the branch used; no claim of a shared mechanism.'
LIMIT = ('The comparison across models remains a comparison of protocols (Huginn and L32: settled '
         'state of a re-injected recurrent core read through a nonlinear coda; Ouro: last loop '
         'of a drifting whole-stack loop with a linear readout).')


def reading(ev):
    verdict, branch = primary(ev)
    near = ev['tau'][1] >= -.05 and ev['tau'][2] <= .05
    return dict(reading=verdict, branch=branch,
                sublabel='POSITIVE EXCESS' if ev['C3']['Delta'][1] > 0 else None,
                tau_near_zero=bool(near), meaning=STRONG if verdict=='STRONG DEFICIT' else
                ZERO if verdict=='NOT DETECTED' and near else OTHER)


def statistics(z, mask):
    g, u, x = (z[k][mask] for k in ('gram3','ugram','x1'))
    fam=z['fam']; d=z['G_task'].shape[-1]
    ev=evaluate(g,fam,d,u,x)
    # Reuse the exact bootstrap for tau, as for the family medians.
    ev['tau']=boot_median(g[:,(fam==4)|(fam==5),0,1].mean(1))
    control=(coord_control(ev) if ev['C3']['Delta'][0] != 0 else
             dict(ratio=None, reading='DESCRIPTIVE ONLY'))
    control['meaningful']=bool(ev['C3']['Delta'][2] < 0)
    if not control['meaningful']: control['reading']='DESCRIPTIVE ONLY'
    rot=rotation(g,u,x,fam,d)
    al=g[:,:,0,1]; P=(fam==4)|(fam==5)   # absolute self-gains (descriptive; required by the Draft 1.2 wording)
    absolute={name:dict(median=float(np.median(al[:,f])), q05=float(np.quantile(al[:,f],.05)), q95=float(np.quantile(al[:,f],.95)),
                        min=float(al[:,f].min()), max=float(al[:,f].max()), n=int(al[:,f].size))
              for name,f in (('task',fam==0),('C1',fam==1),('C2',fam==2),('C3',fam==3),('C3flip',fam==6),('probes',P))}
    adj={str(k):float(np.median(z['adjoint_rel_err'][mask][:,fam==k])) for k in np.unique(fam)}
    return dict(evaluate=ev, primary=reading(ev), coordinate_control=control, rotation=rot, absolute_alpha=absolute,
                adjoint_by_family=adj, adjoint_ok=bool(max(adj.values())<1e-4), n=int(mask.sum()),
                registered='primary: C3 reading; secondary: C1-C3 readings against task, coordinate control, tau; '
                           'evaluate()\'s task-span-projected ("perp") statistics and absolute_alpha are descriptive')


def identity(z, reference):
    """G rows use gradient_pid, not an assumed offset into problem metadata."""
    ref=np.load(Path(reference)/'llever.npz',allow_pickle=False)
    items=json.loads((Path(reference)/'items.json').read_text())
    bypid={int(p['pid']):p for p in items}
    from llever_app import make_items
    local={p['pid']:p for p in make_items() if p['split']=='eval'}
    row={int(p):i for i,p in enumerate(ref['gradient_pid'])}
    diffs=[]; state=[]; prow={int(p):i for i,p in enumerate(ref['pid'])} if 'pid' in ref.files else {}
    for j in np.flatnonzero(z['done']):
        pid=int(z['pid'][j]); old=bypid[pid]; new=local[pid]
        assert all(old[k]==new[k] for k in ('ctx','a','u','prompt','ans')), 'cell mismatch'
        assert old['split']=='eval'
        gold=ref['G'][row[pid]]
        diffs.extend((np.linalg.norm(z['G_task'][j]-gold,axis=-1)/np.maximum(np.linalg.norm(gold,axis=-1),1e-300)).tolist())
        # same base state as the lever run: x*_q itself (Draft 1.2; Draft 1.1 compared only its length)
        if 'xq' in ref.files and 'xq' in z:
            xo=ref['xq'][prow[pid]]; state.append(float(np.linalg.norm(z['xq'][j]-xo)/np.linalg.norm(xo)))
    return float(max(diffs)), (float(max(state)) if state else None)


def analyze_set(z, meta, name, reference=None):
    smoke=bool(meta['cfg'].get('smoke')); done=z['done'].astype(bool); n=int(done.sum())
    g=dict(n_items=len(done),n_done=n, complete=bool(done.all()), smoke=smoke)
    out=dict(gates=g, reading='SMOKE: no reading' if smoke else 'GATES FAILED: no reading')
    expected=meta['cfg'].get('n_'+name,40)
    if n<2 or (not smoke and (not done.all() or len(done)!=expected or (name=='A' and len(done)!=40))):
        g['PASS']=False; g['reason']='incomplete set'; return out
    if len(set(z['pid'][done].tolist()))!=n:
        g['PASS']=False; g['reason']='duplicate contexts'; return out
    if name=='A' and not smoke:
        from llever_app import make_items
        if sorted(z['pid'].tolist())!=sorted(p['pid'] for p in make_items() if p['split']=='eval'):
            g['PASS']=False; g['reason']='Set A is not exactly the 40 lever evaluation problems'; return out
    keys=('gram3','ugram','x1','x2','adjoint_rel_err','gnorm','cos_x','drift','xnorm','G_task','argmax','correct','C_tokens')+(('xq',) if 'xq' in z else ())
    g['finite']=bool(all(np.isfinite(z[k][done]).all() for k in keys))
    if not g['finite']:
        g['PASS']=False; return out
    a=z['gram3'][done]; f=z['fam']; p=(f==4)|(f==5)
    tau=a[:,p,0,1].mean(1)
    tb=(a[:,p,0,2].mean(1)+a[:,p,1,1].mean(1))/2-tau**2
    g['unit_error']=float(max(np.abs(a[:,:,0,0]-1).max(),np.abs(np.diagonal(z['ugram'][done],axis1=1,axis2=2)-1).max()))
    g['x1_error']=float(np.abs(np.diagonal(z['x1'][done],axis1=1,axis2=2)-a[:,:,0,1]).max())
    g['gnorm_min']=float(z['gnorm'][done][:,f<=3].min()); g['tB2_min']=float(tb.min())
    g['adjoint_by_family']={str(k):float(np.median(z['adjoint_rel_err'][done][:,f==k])) for k in np.unique(f)}
    vals=meta.get('timings',{}).get(name,{}).get('G1',meta.get('gates',{}).get('G1',[]))
    g['G1']=bool(vals and np.isfinite(vals).all() and max(vals)<=1e-6)
    g['PASS']=bool(g['G1'] and g['unit_error']<1e-9 and g['x1_error']<1e-9 and g['gnorm_min']>0 and
                   g['tB2_min']>0 and max(g['adjoint_by_family'].values())<1e-4)
    if name=='A':
        g['median_drift']=float(np.median(z['drift'][done])); g['max_drift']=float(z['drift'][done].max()); g['n_correct']=int(z['correct'][done].sum())
        try:
            g['identity_relative_error'],g['state_relative_error']=identity(z,reference)
        except (OSError,KeyError,ValueError,AssertionError,TypeError) as exc:
            g['identity_error']=str(exc); g['identity_relative_error']=None; g['state_relative_error']=None
        g['PASS'] &= bool(g['identity_relative_error'] is not None and g['identity_relative_error']<1e-6 and
                          g['state_relative_error'] is not None and g['state_relative_error']<1e-9 and
                          g['median_drift']<1e-5 and (smoke or g['n_correct']>=36))   # drift threshold: design Draft 1.1
    if not g['PASS']: return out
    out['main']=statistics(z,done)
    out['descr']={str(k):dict(abs_cos_x=float(np.median(np.abs(z['cos_x'][done][:,f==k]))),
                           gnorm=float(np.median(z['gnorm'][done][:,f==k])) if k<=3 else None) for k in np.unique(f)}
    out['reading']='SMOKE: no reading' if smoke else out['main']['primary']['reading']
    if smoke: out['main']['primary']=dict(reading='SMOKE: no reading')
    dr=z['drift'][done]
    out['behaviour']=dict(n_correct=int(z['correct'][done].sum()), n=n,
                          drift=dict(median=float(np.median(dr)), q90=float(np.quantile(dr,.9)), max=float(dr.max()),
                                     above_1e_3=int((dr>1e-3).sum()), above_1e_5=int((dr>1e-5).sum())))
    if True:   # drift sensitivity for both sets (Draft 1.1)
        keep=done & (z['drift']<=1e-3)
        out['drift_excluded']=int((done & ~keep).sum())
        out['sensitivity_drift_le_1e-3']=statistics(z,keep) if keep.sum()>=2 else dict(reading='TOO FEW CONTEXTS: no reading')
        if keep.sum()>=2 and not out['sensitivity_drift_le_1e-3']['adjoint_ok']:
            out['sensitivity_drift_le_1e-3']['primary']=dict(reading='GATES FAILED: no reading')
        if smoke: out['sensitivity_drift_le_1e-3']['primary']=dict(reading='SMOKE: no reading')
    return out


def clean(obj):
    if isinstance(obj,dict): return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [clean(v) for v in obj]
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(float,np.floating)): return float(obj) if np.isfinite(obj) else None
    return obj


def main(run):
    run=Path(run); meta=json.loads((run/'meta.json').read_text()); data=np.load(run/'lread.npz',allow_pickle=False)
    reference=Path(__file__).resolve().parent.parent/'results/llever_pred-20261110-172208'
    out={}
    for s in ('A','B'):
        z={k[2:]:data[k] for k in data.files if k.startswith(s+'_')}
        out[s]=analyze_set(z,meta,s,reference)
    out['meaning_for_writeup']=('SMOKE: no reading' if meta['cfg'].get('smoke') else
        out['A'].get('main',{}).get('primary',{}).get('meaning','GATES FAILED: no reading'))
    out['protocol_limit']=LIMIT
    out=clean(out); (run/'lread_summary.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps(out,indent=2,allow_nan=False)); return out


if __name__=='__main__': main(sys.argv[1])
