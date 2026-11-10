"""Post hoc E3 descriptions from saved OOF scores; no fitting or state loading.

All AUROCs average the five identifiable frozen folds equally. Intervals condition
on fitted scores. Deep differences are target minus source on identical items.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def align(reference, other, subset=False):
    """Return indices into other in reference pid order; fail on missing/duplicate IDs."""
    reference, other = np.asarray(reference), np.asarray(other)
    if len(np.unique(reference)) != len(reference) or len(np.unique(other)) != len(other):
        raise ValueError('duplicate item ids')
    if not subset and set(reference) != set(other):
        raise ValueError('item id sets differ')
    lookup = {p: i for i, p in enumerate(other.tolist())}
    try:
        return np.array([lookup[p] for p in reference.tolist()])
    except KeyError as e:
        raise ValueError('missing item id') from e


def auc(y, score):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y)-n1
    if not n1 or not n0:
        return None
    return float((rankdata(score)[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def fold_auc(y, score, folds):
    vals = [auc(y[folds == f], score[folds == f]) for f in np.unique(folds)]
    return None if None in vals else float(np.mean(vals))


def association(x, y):
    counts = np.bincount(2*np.asarray(x, int)+np.asarray(y, int), minlength=4)
    a,b,c,d = map(float, counts)
    den = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    return counts.tolist(), (a*d-b*c)/den if den else None, a*d/(b*c) if b*c else None


def interval(draws):
    bad = int(np.count_nonzero(~np.isfinite(draws)))
    return dict(interval=None if bad else np.quantile(draws,[.025,.975]).tolist(),
                undefined_bootstrap_draws=bad,
                interval_unavailable_reason='a resampled fold lost a class (or metric undefined)' if bad else None)


def weighted_auc(y, score, weights):
    """AUROC for each row of bootstrap multiplicities, including exact score ties."""
    order=np.argsort(score, kind='stable'); ys=y[order]; ss=score[order]
    starts=np.r_[0, np.flatnonzero(np.diff(ss))+1]
    pos=np.add.reduceat(weights[:,order]*ys,starts,axis=1)
    neg=np.add.reduceat(weights[:,order]*(1-ys),starts,axis=1)
    denom=pos.sum(1)*neg.sum(1)
    with np.errstate(invalid='ignore',divide='ignore'):
        return (pos*(np.cumsum(neg,axis=1)-neg/2)).sum(1)/denom


def paired_bootstrap(y0,s0,y1,s1,folds,draws=2000,seed=0,stratified=False):
    rng=np.random.default_rng(seed); n=len(y0)
    # Registered strata are frozen folds. Deviation preserves BOTH source/target
    # label margins via their joint classes within each frozen fold.
    strata=folds*4+2*y0+y1 if stratified else folds
    w=np.zeros((draws,n),dtype=np.int32)
    for f in np.unique(strata):
        ix=np.flatnonzero(strata==f)
        w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=draws)
    before=[]; after=[]
    for f in np.unique(folds):
        ix=np.flatnonzero(folds==f)
        before.append(weighted_auc(y0[ix],s0[ix],w[:,ix]))
        after.append(weighted_auc(y1[ix],s1[ix],w[:,ix]))
    return interval(np.mean(after,axis=0)-np.mean(before,axis=0))


def spearman(a,b):
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])


def load_score(z,prefix):
    return {f:z[prefix+'/'+f] for f in ('pid','y','scores','folds')}


def run(inputs,draws=2000):
    out=dict(joint=[],scorecorr=[],stratified=[],deep=[],missing=[],metadata={
        'status':'Exploratory, chosen after registered outcomes were known; no causal interpretation',
        'alignment':'Unique pid arrays checked; exact set equality for correlations, deep target subset lookup into source; labels asserted equal for same-label comparisons; frozen source folds asserted equal after deep alignment. Labels from saved y, no raw hidden states loaded.',
        'auroc':'Unweighted mean of identifiable frozen fold AUROCs; conditional stratified subgroup AUROCs pooled within subgroup.',
        'bootstrap':f'{draws} paired item resamples within frozen folds; any undefined draw invalidates whole interval',
        'stratified_deviation':'Explicit deviation from registered rule: resample within fold × joint(source label,target label) classes',
        'deep_difference':'target minus source, same deep items at both budgets; no native deep-budget fits',
        'privileged':'Conditioning on the other model correctness is privileged and descriptive only'})
    joint_labels={}
    for input_dir in inputs:
        r=json.loads((input_dir/'results.json').read_text()); seed=str(r['settings']['seeds'][0])
        with np.load(input_dir/'scores.npz',allow_pickle=False) as z:
            for fam in r['settings']['fams']:
                def get(a,k): return load_score(z,f'{a}/{fam}/{seed}/{k}')
                l=get('m1a','table/L32/L32'); d=get('m1a','table/D/D')
                ix=align(l['pid'],d['pid']); dy=d['y'][ix]
                if fam in joint_labels:
                    oldpid,oldl,oldd=joint_labels[fam]; ji=align(oldpid,l['pid'])
                    assert np.array_equal(oldl,l['y'][ji]) and np.array_equal(oldd,dy[ji])
                else: joint_labels[fam]=(l['pid'].copy(),l['y'].copy(),dy.copy())
                prior=[v for v in out['joint'] if v['family']==fam]
                if prior: assert association(l['y'],dy)[0]==prior[0]['counts']
                if seed==str(json.loads((inputs[0]/'results.json').read_text())['settings']['seeds'][0]):
                    counts,phi,odds=association(l['y'],dy)
                    rng=np.random.default_rng(1729); inds=rng.integers(len(dy),size=(draws,len(dy)))
                    samples=[association(l['y'][i],dy[i])[1:] for i in inds]
                    out['joint'].append(dict(family=fam,counts=counts,phi=phi,odds_ratio=odds,
                        phi_bootstrap=interval(np.array([v[0] if v[0] is not None else np.nan for v in samples])),
                        odds_ratio_bootstrap=interval(np.array([v[1] if v[1] is not None else np.nan for v in samples])),
                        source_keys=[f'm1a/{fam}/{seed}/table/{m}/{m}/y' for m in ('L32','D')]))
                comparisons=[('L32 labels: states','m1a','table/L32/L32','m1a','table/L32/D'),('D labels: states','m1a','table/D/L32','m1a','table/D/D')]
                comparisons += [(m+' own vs confidence','m1a',f'table/{m}/{m}','m1c',f'fixed/{m}/confidence') for m in ('L32','D')]
                for label,a,k,b,j in comparisons:
                    u=get(a,k); v=get(b,j); ix=align(u['pid'],v['pid']); assert np.array_equal(u['y'],v['y'][ix])
                    out['scorecorr'].append(dict(family=fam,seed=seed,comparison=label,spearman=spearman(u['scores'],v['scores'][ix]),n=len(ix),same_fold_assignment=bool(np.array_equal(u['folds'],v['folds'][ix])),source_keys=[f'{a}/{fam}/{seed}/{k}',f'{b}/{fam}/{seed}/{j}']))
                for m,u,other in [('L32',l,dy),('D',d,l['y'][align(d['pid'],l['pid'])])]:
                    vals={}
                    for wrong in (0,1):
                        mask=other==wrong
                        vals['other_'+('wrong' if wrong else 'right')]=dict(n=int(mask.sum()),auroc=auc(u['y'][mask],u['scores'][mask]))
                    out['stratified'].append(dict(family=fam,seed=seed,model=m,**vals,other_wrong_bit_auroc=auc(u['y'],other),source_keys=[f'm1a/{fam}/{seed}/table/{m}/{m}',f'm1a/{fam}/{seed}/table/{"D" if m=="L32" else "L32"}/{"D" if m=="L32" else "L32"}']))
                for m in ('L4','L8','L16','L32'):
                    for src in sorted({32,int(m[1:])}):
                        for target in (48,63,64):
                            base=f'{m}/source{src}'; trans=f'{m}/{src}->{target}/transfer'
                            try:
                                u=get('m3a',base); v=get('m3a',trans)
                            except KeyError as e:
                                out['missing'].append(dict(family=fam,seed=seed,model=m,source=src,target=target,missing_key=str(e))); continue
                            ix=align(v['pid'],u['pid'],subset=True)
                            y0,s0,f0=(u[f][ix] for f in ('y','scores','folds'))
                            assert np.array_equal(f0,v['folds']); assert len(ix)==256
                            before=fold_auc(y0,s0,f0); after=fold_auc(v['y'],v['scores'],f0)
                            out['deep'].append(dict(family=fam,seed=seed,model=m,source=src,target=target,n=len(ix),source_auroc=before,target_auroc=after,difference=None if before is None or after is None else after-before,label_changed=float(np.mean(y0!=v['y'])),
                                registered_bootstrap=paired_bootstrap(y0,s0,v['y'],v['scores'],f0,draws,int(seed)+1729),
                                class_stratified_deviation=paired_bootstrap(y0,s0,v['y'],v['scores'],f0,draws,int(seed)+2718,True),
                                source_keys=[f'm3a/{fam}/{seed}/{base}',f'm3a/{fam}/{seed}/{trans}']))
    for fam in r['settings']['fams']:
        for comp in sorted({v['comparison'] for v in out['scorecorr']}):
            rows=[v for v in out['scorecorr'] if v['family']==fam and v['comparison']==comp]
            out['scorecorr'].append(dict(family=fam,seed='median',comparison=comp,spearman=float(np.median([v['spearman'] for v in rows])),n=rows[0]['n'],source_keys=[v['source_keys'] for v in rows]))
    return out


def fmt(v,iv=None):
    if v is None: return '–'
    t=f'{v:.3f}'.replace('-','−')
    if iv is not None:
        a=iv.get('interval'); t+=' ['+', '.join(f'{x:.3f}'.replace('-','−') for x in a)+']' if a else ' [n/a]'
    return t


def write(out,directory):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'explore.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    lines=['// Exploratory E3; conditional on fitted OOF scores. See explore.json metadata.']
    def table(name,heads,rows):
        lines.append(f'#let tab_{name} = table(columns: {len(heads)}, inset: 3pt, table.header('+', '.join('text('+json.dumps(h)+')' for h in heads)+'),')
        for row in rows: lines.append(', '.join('text('+json.dumps(str(x),ensure_ascii=False)+')' for x in row)+',')
        lines.append(')')
    table('joint',['Family','Both right','L right D wrong','L wrong D right','Both wrong','Phi [95%]','Odds ratio [95%]'],[[v['family'],*v['counts'],fmt(v['phi'],v['phi_bootstrap']),fmt(v['odds_ratio'],v['odds_ratio_bootstrap'])] for v in out['joint']])
    table('scorecorr',['Family','Seed','Scores compared','Spearman'],[[v['family'],v['seed'],v['comparison'],fmt(v['spearman'])] for v in out['scorecorr']])
    table('stratified',['Family','Seed','Model','Other right: AUROC','Other wrong: AUROC','Other wrong bit: AUROC'],[[v['family'],v['seed'],v['model'],fmt(v['other_right']['auroc']),fmt(v['other_wrong']['auroc']),fmt(v['other_wrong_bit_auroc'])] for v in out['stratified']])
    table('deep',['Family/seed','Model/source→target','Source AUC','Target AUC','Δ [registered 95%]','Δ [stratified deviation 95%]','Labels changed'],[[v['family']+'/'+v['seed'],f'{v["model"]}/{v["source"]}→{v["target"]}',fmt(v['source_auroc']),fmt(v['target_auroc']),fmt(v['difference'],v['registered_bootstrap']),fmt(v['difference'],v['class_stratified_deviation']),fmt(v['label_changed'])] for v in out['deep']])
    (directory/'explore.typ').write_text('\n'.join(lines)+'\n')
    (directory/'README.md').write_text('\n'.join(f'{k}: {v}' for k,v in out['metadata'].items())+'\n\nSources: scores.npz keys are recorded per row; pid, y, scores, folds are the only arrays read. Joint labels were verified identical across all three seeds.\nRegenerate with the same --inputs as tables_mon.py and --out docs/monitor_results_gen/explore.\n')

    with (directory/'README.md').open('a') as f:
        f.write(f'\nDeep comparisons: {len(out["deep"])}; registered intervals unavailable: {sum(v["registered_bootstrap"]["interval"] is None for v in out["deep"])}; class-stratified deviation intervals unavailable: {sum(v["class_stratified_deviation"]["interval"] is None for v in out["deep"])}; missing stored rows: {len(out["missing"])}.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--inputs',nargs='+',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); write(run(args.inputs),args.out)

if __name__=='__main__': main()
