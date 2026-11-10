"""Prespecified llever gates, dose/time summaries and endpoint predictors. No refitting."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr


def ci(v):
    v = np.asarray(v); rng = np.random.default_rng(0)
    return np.quantile(v[rng.integers(len(v),size=(5000,len(v)))].mean(1),[.025,.975]).tolist()


def verdict(gates, r1, r2):
    if not gates: return 'INCONCLUSIVE'
    if not r2: return 'PATTERN NOT REPLICATED'
    return 'HUGINN PATTERN REPLICATES' if r1 else 'RESPONSE SHAPE PREDICTED, NO DEMONSTRATED LEVER ADVANTAGE'


def scores(pred, obs):
    pairs = [(p.ravel(),o.ravel()) for p,o in zip(pred,obs)]
    cor = np.array([np.corrcoef(p,o)[0,1] if np.std(p)>0 and np.std(o)>0 else 0. for p,o in pairs])
    # NOTE(design): slope includes intercept as in the Huginn analysis; both matrices are centred.
    slope = [np.polyfit(p,o,1)[0] if np.std(p)>0 else 0. for p,o in pairs]
    err = [np.linalg.norm(p-o)/max(np.linalg.norm(o),1e-300) for p,o in pairs]
    return cor, dict(median_corr=float(np.median(cor)), median_slope=float(np.median(slope)),
                     median_norm_error=float(np.median(err)))


def analyze(zt,zp,ze,metas,push):
    n = len(ze['pid']); assert n==40
    ev = np.where(zp['split']=='eval')[0]; tr = np.where(zp['split']=='train')[0]
    assert np.array_equal(zp['pid'][ev],ze['pid'])
    assert np.array_equal(zp['pid'][tr],zt['gradient_pid'])
    ans = ze['ans']; jj = np.arange(n)
    assert (ans[jj^1]==ans).all() and (ans[(jj+2)%n]!=ans).all()
    PL = push['P'][:10]; G = zp['G'][ev]; AP = zp['AP'][ev]
    lhs = np.einsum('nkd,md->nkm',zt['AtG'],PL)
    rhs = np.einsum('nkd,nmd->nkm',zp['G'][tr],zp['AP'][tr])
    adj = float(np.median(abs(lhs-rhs)/(abs(lhs)+abs(rhs)+1e-9)))
    tails = {k:dict(median=float(np.median(z['tail'])),q95=float(np.quantile(z['tail'],.95)))
             for k,z in [('train',zt),('pred',zp)]}
    finite = all(np.isfinite(v).all() for z in (zt,zp,ze,push) for v in z.values() if v.dtype.kind in 'fc')
    g1 = all(m.get('status')=='complete' and len(m.get('gates',{}).get('G1',[]))>=2
             and max(m['gates']['G1'])<=1e-6 for m in metas)
    correct = int((ze['top'][:,0]==ze['ans_token']).sum())
    drift = float(np.median(ze['last_drift'][:,0]))
    gates = dict(finite=bool(finite),G1=g1,tails=all(v['median']<1e-3 and v['q95']<1e-2 for v in tails.values()),
                 adjoint=adj<.05,sham_correct=correct>=36,sham_drift=drift<1e-6)
    out = dict(gates=gates,tails=tails,adjoint_error=adj,sham_correct=correct,sham_drift=drift,
               primary=dict(rho=.03,t=int(ze['times'][-1])), doses={})
    # Primary time = the endpoint T of the forced run (48, or 96 under the smoke rule; design Draft 1.1).
    ti = len(ze['times'])-1
    predictors = dict(FULL=np.einsum('nmd,nkd->nkm',G,AP),
        AVERAGED=np.einsum('nmd,kd->nkm',G,zp['AP'][tr].mean(0)),
        SWAPPED=np.einsum('nmd,nkd->nkm',G,AP[(jj+2)%n]),
        SWAPPED_SAME_CLASS=np.einsum('nmd,nkd->nkm',G,AP[jj^1]),
        IDENTITY=np.einsum('nmd,kd->nkm',G,PL))
    for rho in (.01,.03,.09):
        def indices(offset):
            return [int(np.flatnonzero((ze['row_push']==offset+k)&np.isclose(ze['row_rho'],rho))[0]) for k in range(10)]
        effects = {}; structure = {}; course = {}; responses = {}
        for fam,offset in [('L',0),('G',10),('C',20)]:
            if not all(np.any((ze['row_push']==offset+k)&np.isclose(ze['row_rho'],rho)) for k in range(10)): continue
            dz = ze['Z'][:,indices(offset)]-ze['Z'][:,0:1]
            responses[fam] = dz
            target = np.diagonal(dz,axis1=1,axis2=3).transpose(0,2,1) # [problem,k,time]
            effects[fam] = target[:,:,ti].mean(1)
            course[fam] = [dict(t=int(t),mean=float(target[:,:,i].mean()),ci=ci(target[:,:,i].mean(1))) for i,t in enumerate(ze['times'])]
            coeff = []
            for j in range(n):
                for k in range(10):
                    v = dz[j,k,ti,np.arange(10)!=k]
                    if np.std(v)>0:
                        coeff.append(spearmanr(v,-abs(np.arange(10)[np.arange(10)!=k]-k)).statistic)
            # NOTE(design): constant off-target responses have undefined Spearman; omit and report null if all are constant.
            structure[fam] = float(np.median(coeff)) if coeff else None
        p1 = dict(mean_target_effect={f:float(v.mean()) for f,v in effects.items()},
                  target_ci={f:ci(v) for f,v in effects.items()},L_minus_G_ci=ci(effects['L']-effects['G']))
        if 'C' in effects: p1['L_minus_C_ci'] = ci(effects['L']-effects['C'])
        obs = responses['L'][:,:,-1]; cor = {}; p3 = {}
        for name,p in predictors.items(): cor[name],p3[name] = scores(rho*float(push['m_ref'])*p,obs)
        diffs = {name:ci(cor['FULL']-cor[name]) for name in predictors if name!='FULL'}
        r1 = p1['L_minus_G_ci'][0]>0 and p1['target_ci']['L'][0]>0  # Draft 1.2: L must itself raise its target
        r2 = p3['FULL']['median_corr']>=.9 and diffs['IDENTITY'][0]>0
        r3 = diffs['SWAPPED'][0]>0 and diffs['AVERAGED'][0]>0
        targetL = np.diagonal(responses['L'],axis1=1,axis2=3).transpose(0,2,1)
        out['doses'][str(rho)] = dict(P1=p1,P3=p3,FULL_minus_ci=diffs,linear_distance_spearman=structure,
            time_course=course,first_step_reversal=float(((targetL[:,:,0]<0)&(targetL[:,:,ti]>0)).mean()),
            criteria=dict(R1=r1,R2=r2,R3=r3),
            prediction_time=int(ze['times'][-1]),
            verdict=verdict(all(gates.values()),r1,r2),primary=rho==.03)
    out['dose_ratios'] = {f: {str(r):dict(observed=out['doses'][str(r)]['P1']['mean_target_effect'][f]/
            max(abs(out['doses']['0.03']['P1']['mean_target_effect'][f]),1e-300)*
            np.sign(out['doses']['0.03']['P1']['mean_target_effect'][f]), expected=r/.03)
            for r in (.01,.09) if f in out['doses'][str(r)]['P1']['mean_target_effect']}
            for f in out['doses']['0.03']['P1']['mean_target_effect']}
    out['verdict'] = out['doses']['0.03']['verdict']
    return out


def json_clean(value):
    if isinstance(value,dict): return {k:json_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_clean(v) for v in value]
    if isinstance(value,np.generic): value=value.item()
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def main(train,pred,evaluation):
    roots = list(map(Path,(train,pred,evaluation)))
    metas = [json.loads((p/'meta.json').read_text()) for p in roots]
    assert all(m['status']=='complete' for m in metas)
    assert len({m['cfg']['T'] for m in metas})==1
    raw = (roots[2]/'pushes.npz').read_bytes(); sha = hashlib.sha256(raw).hexdigest()
    assert all(m['push_file_sha256']==sha for m in metas[1:])
    arrays = [dict(np.load(p/'llever.npz',allow_pickle=False)) for p in roots]
    push = dict(np.load(roots[2]/'pushes.npz',allow_pickle=False))
    assert int(arrays[2]['times'][-1])==int(metas[2]['cfg']['T'])==int(push['T']), 'endpoint time, run T and push-file T must agree'
    out = analyze(*arrays,metas,push)
    # Convert numpy booleans/scalars, prohibit accidental NaN JSON.
    text = json.dumps(json_clean(out),indent=2,allow_nan=False)
    (roots[2]/'analysis_summary.json').write_text(text+'\n'); print(text)


if __name__=='__main__': main(*sys.argv[1:4])
