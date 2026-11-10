"""CPU-only monitor analyses. Bootstrap intervals condition on fitted OOF scores."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mon_probe import (nested_cv, fit_logistic, fit_path, apply_models, spline_features,
                       paired_bootstrap, auroc, selective_error, make_folds, make_joint_folds, check_folds, LAMS)

FAMS = ['add3', 'addsub', 'muladd', 'add4', 'sub3', 'add3w', 'arc_easy', 'arc_chal']
T_STD = [1, 2, 3, 4, 6, 8, 12, 16, 24, 31, 32, 33]
T_DEEP = T_STD + [48, 63, 64]
T_PREV, R_POST, D_PREV, R_DISP = [8, 32], [4, 8, 16, 32], [8, 12, 16], [2, 4, 8, 16, 32]
N_SCREEN, R_NPOS = 96, [8, 16, 24, 32]
ANALYSES = ['m1a', 'm1b', 'm1c', 'm1s', 'm1x', 'm3a', 't1', 'm4', 'f16', 'dyn']
READOUT = {'x_last', 'top_id', 'top_lp', 'ent', 'lp_ans', 'valid_lm', 'opt_lp'}
DYNAMICS = {'d_full', 'd_last', 'e_full', 'e_last', 'n_full', 'n_last', 'cos_full', 'cos_last', 'd_pos'}
DATA, ITEMS, SETTINGS, VALID = {}, {}, {}, {}


def load_runs(runs, models=None, allow_partial=False, manifest=None, items=None):
    data, provenance, skipped = {}, [], []
    hashes = {}; valid = None
    for model in models or ['L4', 'L8', 'L16', 'L32', 'D']:
        parts, copies = [], {}
        seen = set()
        for run in runs:
            for shard in sorted((Path(run)/model).glob('s*')):
                mp = shard/'meta.json'
                meta = json.loads(mp.read_text()) if mp.exists() else {}
                # A shard stopped by its deadline has committed, gate-checked chunks listed in its metadata.
                usable = meta.get('status') == 'complete' or (allow_partial and meta.get('status') == 'partial' and meta.get('chunks'))
                if not usable:
                    skipped.append({'path': str(shard), 'status': meta.get('status', 'missing meta')})
                    continue
                for key in ('FAMS', 'T_STD', 'T_DEEP', 'T_PREV', 'R_POST', 'D_PREV', 'R_DISP', 'R_NPOS'):
                    if key in meta:
                        assert meta[key] == globals()[key], f'{mp}: incompatible {key}'
                if manifest is not None:
                    assert meta['code_sha256'] == manifest['code_sha256'], f'{mp}: code hash mismatch'
                    assert meta['cfg']['items_sha256'] == manifest['items_sha256'], f'{mp}: items hash mismatch'
                    assert meta.get('valid_ids'), f'{mp}: metadata lacks the valid answer tokens'
                if meta.get('valid_ids'):
                    assert valid is None or valid == meta['valid_ids'], f'{mp}: valid answer tokens differ between shards'
                    valid = meta['valid_ids']
                provenance.append({'path': str(shard), 'meta': meta})
                for path in (sorted(shard/c for c in meta['chunks']) if meta.get('status') == 'partial' else sorted(shard.glob('c*_*.npz'))):
                    with np.load(path, allow_pickle=False) as z:
                        pids = z['pid']; n = len(pids)
                        assert len(set(map(int, pids))) == n and not seen.intersection(map(int, pids)), f'duplicate pids: {model} {path}'
                        seen.update(map(int, pids))
                        if 'tok_hash' in z:
                            for p, h in zip(pids, z['tok_hash']):
                                assert int(p) not in hashes or hashes[int(p)] == int(h), f'mismatched tok_hash: {p}'
                                hashes[int(p)] = int(h)
                        key = 'h_last' if model == 'D' else 'x_last'
                        ts = list(z['T']) if model != 'D' else list(range(17))
                        assert model == 'D' or ts in (T_STD, T_DEEP), f'invalid T: {path}'
                        for p, x in zip(z['pid32'], z[key+'32']):
                            copies[int(p)] = {int(t): a.copy() for t, a in zip(ts, x)}
                        part = {}
                        for k in z.files:
                            if k in ('T', 'pid32', key+'32'):
                                continue
                            a = z[k]
                            if model != 'D' and k in READOUT:
                                shape = (n, len(T_DEEP), *a.shape[2:])
                                b = np.full(shape, np.nan if a.dtype.kind == 'f' else -1, dtype=a.dtype)
                                b[:, :len(ts)] = a; a = b
                            elif model != 'D' and k in DYNAMICS:
                                b = np.full((n, 65, *a.shape[2:]), np.nan, dtype=a.dtype)
                                b[:, :a.shape[1]] = a; a = b
                            part[k] = a
                        parts.append(part)
        if parts:
            keys = set.union(*(set(p) for p in parts))
            merged = {}
            for k in keys:
                template = next(p[k] for p in parts if k in p)
                merged[k] = np.concatenate([p[k] if k in p else np.full((len(p['pid']), *template.shape[1:]),
                                              np.nan if template.dtype.kind == 'f' else -1, dtype=template.dtype) for p in parts])
            order = np.argsort(merged['pid'])
            data[model] = {k: v[order] for k, v in merged.items()}
            if model != 'D':
                a = data[model]
                deep_mask = a['top_id'][:, len(T_STD), 0] >= 0
                deep = {'pid': a['pid'][deep_mask]}
                for k in READOUT:
                    if k in a:
                        deep[k] = a[k][deep_mask, len(T_STD):].copy()
                        a[k] = a[k][:, :len(T_STD)].copy()
                a['deep'] = deep
            data[model]['copies'] = copies
    completeness={}
    if manifest is not None:
        expected={int(p) for p,it in items.items() if it['fam'] in manifest['families'] and N_SCREEN<=it['i']<N_SCREEN+manifest['n_main']}
        for m,a in data.items():
            found=set(map(int,a['pid'])); assert found<=expected, f'{m}: unexpected pids'
            completeness[m]={'complete':found==expected,'missing':len(expected-found)}
            assert allow_partial or found==expected, f'{m}: incomplete collection'
    return data, {'completeness':completeness, 'valid_ids': valid, 'shards': provenance, 'skipped_shards': skipped,
                  'missing_optional': {m: [k for k in ('d_pos', 'disp_full', 'n_pos', 'tok_hash') if k not in a]
                                       for m, a in data.items() if m != 'D'}}


def subset(a, pids):
    ix = np.searchsorted(a['pid'], pids)
    if len(ix) and np.array_equal(ix, np.arange(ix[0], ix[0]+len(ix))):
        ix = slice(int(ix[0]), int(ix[-1])+1)
    out = {k: v[ix] for k, v in a.items() if k not in ('copies', 'deep')}
    if 'deep' in a:
        keep = np.isin(a['deep']['pid'], pids)
        out['deep'] = {k:v[keep] for k,v in a['deep'].items()}
    return out


def at_budget(a, key, r):
    if r in T_STD: return a[key][:, T_STD.index(r)]
    deep = a['deep']; x = deep[key][:, [48,63,64].index(r)]
    out = np.full((len(a['pid']), *x.shape[1:]), np.nan if x.dtype.kind=='f' else -1, dtype=x.dtype)
    out[np.searchsorted(a['pid'], deep['pid'])] = x
    return out


def labels(a, model, r=32):
    top = a['ro_top_id'][:, 0] if model == 'D' else at_budget(a,'top_id',r)[:,0]
    return (top != a['ans_tok']).astype(int)


def state(a, name):
    if name == 'pre': return a['pre'][:, 0]
    if name.startswith('post'): return a['post'][:, R_POST.index(int(name[4:]))]
    if name.startswith('prev'): return a['x_prev'][:, T_PREV.index(int(name[4:]))]
    if name.startswith('hprev'): return a['h_prev'][:, D_PREV.index(int(name[5:]))]
    if name.startswith('x'): return at_budget(a,'x_last',int(name[1:]))
    return a['h_last'][:, int(name[1:])]


def confidence(a, model, r=32):
    prefix, ix = ('ro_', None) if model == 'D' else ('', T_STD.index(r))
    def get(k):
        x = a[prefix+k]
        return x if ix is None else x[:, ix]
    lp = get('top_lp')
    return np.column_stack((lp[:, 0], lp[:, 0]-lp[:, 1], get('ent'), get('valid_lm')))


def spline_block(u):
    def transform(tr, te):
        knots = np.quantile(u[tr], [.05, .35, .65, .95], axis=0)
        def build(ix):
            return np.column_stack([spline_features(u[ix, j], knots[:, j]) for j in range(u.shape[1])])
        return build(tr), build(te)
    return transform


def fold_metric(scores, y, folds, ix=None, metric='auc'):
    ix = np.arange(len(y)) if ix is None else ix
    values = []
    for f in np.unique(folds):
        j = ix[folds[ix] == f]
        if not len(j) or len(np.unique(y[j])) < 2: return float('nan')
        values.append(auroc(scores[j], y[j]) if metric == 'auc' else selective_error(scores[j], y[j])[0])
    return float(np.mean(values))


def nested_sites(sites, y, seed, U=None, folds=None):
    """Library path solver, visiting one site at a time to bound worker RAM."""
    folds=make_folds(y,SETTINGS['outer'],seed) if folds is None else check_folds(folds,y,SETTINGS['outer'])
    scores=np.empty(len(y))
    chosen=[]; models=[]; failures=[]; lams=sorted(LAMS,reverse=True)
    names=list(sites)
    def extras(tr,te):
        if callable(U): return U(tr,te)
        return (None,None) if U is None else (U[tr],U[te])
    for f in range(SETTINGS['outer']):
        tr=np.flatnonzero(folds!=f); te=np.flatnonzero(folds==f)
        inner=make_folds(y[tr],SETTINGS['inner'],np.random.SeedSequence([seed,f,917]))
        merit=np.zeros((len(names),len(lams)))
        for g in range(SETTINGS['inner']):
            it=tr[inner!=g]; iv=tr[inner==g]; ut,uv=extras(it,iv)
            for j,name in enumerate(names):
                ms=fit_path(sites[name][it],y[it],lams,ut)
                merit[j]+=[auroc(m.score(sites[name][iv],uv),y[iv]) for m in ms]
                failures.extend(m.message for m in ms if not m.converged)
        j,ell=np.unravel_index(np.argmax(merit),merit.shape); name=names[j]; lam=lams[ell]
        ut,uv=extras(tr,te); m=fit_logistic(sites[name][tr],y[tr],lam,ut)
        scores[te]=m.score(sites[name][te],uv); chosen.append((name,lam)); models.append(m)
    return dict(seed=seed,scores=scores,folds=folds,chosen=chosen,models=models,
                mean_auroc=fold_metric(scores,y,folds),inner_nonconverged=len(failures),
                inner_solver_messages=sorted(set(failures)))


def bootstrap_terms(terms, seed):
    assert all(np.array_equal(terms[0][1]['pid'],r['pid']) for _,r in terms)
    def stat(ix):
        return sum(w*fold_metric(r['scores'],r['y'],r['folds'],ix) for w,r in terms)
    groups=np.column_stack([r['folds'] for _,r in terms])
    est,_,_,draws=paired_bootstrap(stat,groups,SETTINGS['boot'],seed)
    bad=int((~np.isfinite(draws)).sum())
    lo,hi=np.percentile(draws,[2.5,97.5]) if not bad else (np.nan,np.nan)
    return float(est),float(lo),float(hi),bad,draws


def interval_reason(bad, n):
    return f'a resampled fold lost a class in {bad} of {n} draws'


class Analysis:
    def __init__(self, fam, seed):
        self.fam, self.seed = fam, seed
        self.rows, self.scores, self.differences, self.skips, self.cache = {}, {}, {}, [], {}
        self.local = {m: subset(a, a['pid'][[ITEMS[int(p)]['fam'] == fam for p in a['pid']]]) for m, a in DATA.items()}
        for m in [m for m, a in self.local.items() if not len(a['pid'])]:
            self.skips.append({'models': [m], 'reason': 'no items of this family'}); del self.local[m]

    def common(self, models):
        if any(m not in self.local for m in models):
            self.skips.append({'models': models, 'reason': 'model absent'}); return None
        p = np.array(sorted(set.intersection(*(set(self.local[m]['pid']) for m in models))), dtype=np.int64)
        self.rows['common/'+','.join(models)] = {'n': len(p)}
        if not len(p):
            self.skips.append({'models': models, 'reason': 'no items in common'}); return None
        return {m: subset(self.local[m], p) for m in models}

    def fit(self, key, a, model, sites, r=32, extras=None, baseline=False, y=None, fitter='logistic', folds=None):
        if key in self.cache: return self.cache[key]
        y = labels(a, model, r) if y is None else y
        counts = np.bincount(y, minlength=2)
        minimum = 12 if SETTINGS['smoke'] else max(SETTINGS['outer']+1, SETTINGS['inner']+2)
        if min(counts) < minimum:
            self.skips.append({'name': key, 'n': len(y), 'classes': counts.tolist(), 'reason': f'fewer than {minimum} per class'})
            self.cache[key] = None; return None
        xs = {'zero': np.zeros((len(y), 1))} if baseline else {s: state(a, s) for s in sites}
        result = nested_sites(xs,y,self.seed,extras,folds) if len(xs)>1 else nested_cv(xs, y, U=extras, lams=[1.] if baseline else LAMS,
                           n_outer=SETTINGS['outer'], n_inner=SETTINGS['inner'], seeds=(self.seed,), fitter=fitter, folds=None if folds is None else [folds])[0]
        result.update(y=y, pid=a['pid'])
        self.cache[key] = result
        self.rows[key] = {'n': len(y), 'classes': counts.tolist(), 'auroc': result['mean_auroc'],
                          'accepted_error': fold_metric(result['scores'], y, result['folds'], metric='error'),
                          'chosen': result['chosen'], 'nonconverged': sum(not m.converged for m in result['models']),
                          'inner_nonconverged': result['inner_nonconverged'],
                          'solver_messages': [m.message for m in result['models'] if not m.converged]}
        _,lo,hi,bad,draws=bootstrap_terms([(1,result)],self.seed+171+100003*FAMS.index(self.fam))
        self.rows[key].update(interval=[lo,hi],undefined_bootstrap_draws=bad)
        if bad: self.rows[key]['interval_unavailable_reason']=interval_reason(bad,len(draws))
        self.differences[key]=draws
        self.scores[key] = {k: result[k] for k in ('pid', 'y', 'scores', 'folds')}
        return result

    def contrast(self, key, terms):
        if any(r is None for _, r in terms): return
        est,lo,hi,bad,draws=bootstrap_terms(terms,self.seed+171+100003*FAMS.index(self.fam))
        self.rows[key] = {'difference':est,'interval':[lo,hi],'n':len(terms[0][1]['pid']),
                          'undefined_bootstrap_draws':bad,
                          'nonconverged_outer_fits':sum(not m.converged for _,r in terms for m in r.get('models',[]))}
        if bad: self.rows[key]['interval_unavailable_reason']=interval_reason(bad,len(draws))
        self.differences[key]=draws

    def pair(self, a, best=False, prefix=''):
        return [self.fit(prefix+m, a[m], m, (['x32', 'post32'] if m == 'L32' else [f'h{i}' for i in range(17)])
                         if best else (['x32'] if m == 'L32' else ['h12'])) for m in ('L32', 'D')]

    def m1a(self):
        a = self.common(['L32', 'D'])
        if a is None: return
        yL=labels(a['L32'],'L32'); yD=labels(a['D'],'D')
        shared=make_joint_folds(np.column_stack((yL,yD)),SETTINGS['outer'],np.random.SeedSequence([self.seed,FAMS.index(self.fam),2201]))
        if any(len(np.unique(y[shared==f]))<2 for y in (yL,yD) for f in range(SETTINGS['outer'])):
            self.skips.append({'name':'m1a','reason':'shared fold lacks a class'}); return
        cells = [self.fit('table/'+row+'/'+col, a[col], col, ['x32' if col == 'L32' else 'h12'], y=labels(a[row], row), folds=shared)
                 for row in ('L32', 'D') for col in ('L32', 'D')]
        for name, weights in [('diagonal', [1,0,0,-1]), ('row', [.5,.5,-.5,-.5]),
                              ('column', [.5,-.5,.5,-.5]), ('interaction', [1,-1,-1,1])]:
            self.contrast(name, list(zip(weights, cells)))

    def m1b(self):
        a = self.common(['L32', 'D'])
        if a is None: return
        l, d = self.pair(a, True, 'best/')
        self.contrast('best_difference', [(1,l),(-1,d)])

    def m1c(self):
        a = self.common(['L32', 'D'])
        if a is None: return
        for best in (False, True):
            terms = {}
            for m in ('L32', 'D'):
                sites = (['x32','post32'] if m == 'L32' else [f'h{i}' for i in range(17)]) if best else ['x32' if m == 'L32' else 'h12']
                pre = f'{"best" if best else "fixed"}/{m}/'
                conf = confidence(a[m], m)
                for flex in (False, True):
                    u = spline_block(conf) if flex else conf
                    k = 'flex' if flex else 'confidence'
                    b = self.fit(pre+k, a[m], m, [], extras=u, baseline=True)
                    p = self.fit(pre+k+'+probe', a[m], m, sites, extras=u)
                    self.contrast(pre+k+'/added', [(1,p),(-1,b)])
                    terms[m,flex] = [(1,p),(-1,b)]
                self.fit(pre+'probe', a[m], m, sites)
            for flex in (False, True):
                self.contrast(f'{best}/{flex}/added_difference', terms['L32',flex]+[(-w,r) for w,r in terms['D',flex]])
        for m in ('L32', 'D'):
            ident, privileged = problem_features(a[m], m)
            for k,u in [('output_identity',ident),('privileged_ground_truth',privileged)]:
                self.fit(m+'/'+k, a[m], m, [], extras=u, baseline=True)
            self.fit(m+'/early', a[m], m, ['pre' if m != 'D' else 'h4'])

    def m1s(self):
        for m,a in self.local.items():
            if m == 'D':
                for r in D_PREV: self.fit(f'D/prev{r}',a,m,[f'hprev{r}'])
                continue
            for r in sorted({int(m[1:]),32}):
                p = self.fit(f'{m}/{r}/probe',a,m,[f'x{r}'],r)
                b = self.fit(f'{m}/{r}/confidence',a,m,[],r,confidence(a,m,r),True)
                joint = self.fit(f'{m}/{r}/joint',a,m,[f'x{r}'],r,confidence(a,m,r))
                self.contrast(f'{m}/{r}/added',[(1,joint),(-1,b)])
            for r in T_PREV: self.fit(f'{m}/prev{r}',a,m,[f'prev{r}'],r)
        a = self.common(['L32','D'])
        if a is None: return
        agree = labels(a['L32'],'L32') == labels(a['D'],'D')
        selected = {m: subset(x,x['pid'][agree]) for m,x in a.items()}
        self.rows['concordant_population'] = {'n':int(agree.sum()), 'classes':np.bincount(labels(selected['L32'],'L32'),minlength=2).tolist()}
        concordant=self.pair(selected, prefix='concordant/')
        self.contrast('concordant/diagonal_difference',[(1,concordant[0]),(-1,concordant[1])])
        for m in ('L32','D'):
            for fitter in ('meandiff','pca_logistic'):
                self.fit(m+'/'+fitter,a[m],m,['x32' if m=='L32' else 'h12'],fitter=fitter)
            y = np.random.default_rng(12345).permutation(labels(a[m],m))
            sites = ['x32','post32'] if m=='L32' else [f'h{i}' for i in range(17)]
            self.fit(m+'/permutation',a[m],m,sites,y=y)
        for fitter in ('meandiff','pca_logistic'):
            self.contrast(fitter+'/diagonal_difference',[(1,self.cache['L32/'+fitter]),(-1,self.cache['D/'+fitter])])

    def m1x(self):
        """Cross-family transfer (secondary). The diagonal probes of this family (L32 from x_32, D from h_12) are
        applied unchanged to every other analysed family; the score of a target item is the mean of the five
        outer-fold models' scores. Intervals: 2000 resamples of the target items within the four joint
        correctness categories (L32 right/wrong x D right/wrong), the same resample for both models; they
        condition on the fitted probes. '*_single_probe' is the mean over the five probes of each probe's own
        AUROC on the target: the own-family AUROC is also an average of single-probe AUROCs, so
        '*_minus_own_family' uses it (the averaged score ranks differently even without any change of family).
        The mean over targets needs every other analysed family; otherwise it is unavailable and says which
        are missing."""
        a = self.common(['L32', 'D'])
        if a is None: return
        site = {'L32': 'x32', 'D': 'h12'}
        fits = {m: self.fit(m+'/source', a[m], m, [site[m]]) for m in site}
        if any(v is None for v in fits.values()): return
        means = []; missing = []
        for fam in SETTINGS['fams']:
            if fam == self.fam: continue
            pids = np.array(sorted(set.intersection(*({int(p) for p in DATA[m]['pid'] if ITEMS[int(p)]['fam'] == fam} for m in site))), dtype=np.int64)
            b = {m: subset(DATA[m], pids) for m in site} if len(pids) else None
            ys = {m: labels(b[m], m) for m in site} if b else None
            if b is None or any(len(np.unique(y)) < 2 for y in ys.values()):
                self.skips.append({'name': 'to/'+fam, 'reason': 'no common items or a class absent', 'n': len(pids)})
                missing.append(fam); continue
            each = {m: [model.score(state(b[m], site[m])) for model in fits[m]['models']] for m in site}
            sc = {m: np.mean(each[m], axis=0) for m in site}
            est = {m: auroc(sc[m], ys[m]) for m in site}
            single = {m: float(np.mean([auroc(v, ys[m]) for v in each[m]])) for m in site}
            strata = 2*ys['L32']+ys['D']; groups = [np.flatnonzero(strata == c) for c in np.unique(strata)]
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, FAMS.index(self.fam), FAMS.index(fam), 3301]))
            draws = np.empty((SETTINGS['boot'], 2))
            for k in range(SETTINGS['boot']):
                ix = np.concatenate([rng.choice(g, len(g)) for g in groups])
                draws[k] = [auroc(sc[m][ix], ys[m][ix]) for m in site]
            assert np.isfinite(draws).all()  # every resample keeps both classes for both models
            pct = lambda v: list(map(float, np.percentile(v, [2.5, 97.5])))
            self.rows['to/'+fam] = {'n': len(pids), 'L32_auroc': est['L32'], 'D_auroc': est['D'],
                                    'difference': est['L32']-est['D'], 'L32_interval': pct(draws[:, 0]),
                                    'L32_auroc_single_probe': single['L32'], 'D_auroc_single_probe': single['D'],
                                    'D_interval': pct(draws[:, 1]), 'difference_interval': pct(draws[:, 0]-draws[:, 1]),
                                    'L32_classes': np.bincount(ys['L32'], minlength=2).tolist(),
                                    'D_classes': np.bincount(ys['D'], minlength=2).tolist()}
            means.append((est['L32'], est['D'], single['L32'], single['D']))
        if missing or not means:
            self.rows['mean_over_targets'] = {'targets': len(means), 'missing_targets': missing}
            self.skips.append({'name': 'mean_over_targets', 'reason': 'needs every other analysed family', 'missing': missing})
        else:
            l, d, ls, ds = np.mean(means, axis=0)
            self.rows['mean_over_targets'] = {'targets': len(means), 'missing_targets': [], 'L32_auroc': float(l), 'D_auroc': float(d),
                                              'difference': float(l-d), 'L32_auroc_single_probe': float(ls), 'D_auroc_single_probe': float(ds),
                                              'L32_minus_own_family': float(ls-fits['L32']['mean_auroc']),
                                              'D_minus_own_family': float(ds-fits['D']['mean_auroc']),
                                              'drop_difference': float((ls-fits['L32']['mean_auroc'])-(ds-fits['D']['mean_auroc']))}

    def m3a(self):
        for m,a in self.local.items():
            if m == 'D': continue
            for s in sorted({32,int(m[1:])}):
                base = self.fit(f'{m}/source{s}',a,m,[f'x{s}'],s)
                if base is None: continue
                tau = thresholds(base)
                for r in T_DEEP:
                    mask = np.isfinite(state(a,f'x{r}')[:,0]); b = subset(a,a['pid'][mask])
                    target_y = labels(b,m,r)
                    if any(len(np.unique(target_y[base['folds'][mask]==f]))<2 for f in range(SETTINGS['outer'])):
                        self.skips.append({'name':f'{m}/{s}->{r}', 'reason':'target class absent in a frozen source fold', 'n':int(mask.sum())})
                        continue
                    native = self.fit(f'{m}/native{r}',b,m,[f'x{r}'],r)
                    if native is None: continue
                    trans = apply_models(base,{f'x{s}':state(a,f'x{r}')}) if mask.all() else frozen_subset(base,state(a,f'x{r}'),mask)
                    z = {'scores':trans, 'folds':base['folds'][mask], 'pid':a['pid'][mask], 'y':labels(b,m,r)}
                    key=f'{m}/{s}->{r}'
                    self.scores[key+'/transfer']=dict(z)
                    z['models']=base['models']  # the frozen source probes: their failures count in the contrast below
                    self.rows[key]={'transfer_auroc':fold_metric(z['scores'],z['y'],z['folds']),
                                    'native_auroc':native['mean_auroc'], 'n':len(b['pid']),
                                    'label_changed':float(np.mean(z['y']!=base['y'][mask])),
                                    'source_nonconverged':sum(not q.converged for q in base['models'])}
                    _,lo,hi,bad,draws=bootstrap_terms([(1,z)],self.seed+171+100003*FAMS.index(self.fam))
                    self.rows[key].update(transfer_interval=[lo,hi],transfer_undefined_bootstrap_draws=bad)
                    if bad: self.rows[key]['transfer_interval_unavailable_reason']=interval_reason(bad,len(draws))
                    self.differences[key+'/transfer']=draws
                    reject = z['scores'] > tau[mask]
                    self.rows[key].update(fraction_rejected=float(reject.mean()), accepted_error=float(z['y'][~reject].mean()) if (~reject).any() else None)
                    self.contrast(key+'/native_minus_transfer',[(1,native),(-1,z)])
            if m == 'L32': self.matrix(m,a,[2,4,8,16,32])
        if 'D' in self.local: self.matrix('D',self.local['D'],[4,8,12,16])

    def matrix(self,m,a,budgets):
        for s in budgets:
            site = f'h{s}' if m=='D' else f'x{s}'
            fit = self.fit(f'matrix/{m}/fit{s}',a,m,[site],s)
            if fit is None: continue
            for r in budgets:
                score=apply_models(fit,{site:state(a,f'h{r}' if m=='D' else f'x{r}')})
                self.rows[f'matrix/{m}/{s}->{r}']={'auroc':fold_metric(score,labels(a,m,r),fit['folds'])}

    def t1(self):
        for m,a in self.local.items():
            if m=='D': continue
            fit=self.fit(f'{m}/source32',a,m,['x32'])
            for r in (4,8,16):
                key=f'{m}/{r}'; d=a['d_full']; n=a['n_full']
                q,b=rule_a(d,n,r)
                last=np.linalg.norm(state(a,'x32').astype(float)-state(a,f'x{r}').astype(float),axis=1)
                exact=a.get('disp_full',np.full((len(d),5,2),np.nan))[:,R_DISP.index(r),0]
                valid=np.isfinite(b); realized=np.where(np.isfinite(exact),exact,last)
                path=d[:,r+1:33].sum(1)
                row={'n':len(d),'forecast_fraction':float(valid.mean()),'q':distribution(q),
                     'path_length':distribution(path),'state_bound':distribution(b[valid]),
                     'last_position_displacement':distribution(last),
                     'path_violation':rate(path>b,valid),'last_position_violation':rate(last>b,valid),
                     'exact_n':int(np.isfinite(exact).sum()),'exact_violation':rate(exact>b,valid&np.isfinite(exact)),
                     'bound_over_realized':distribution(b[valid]/np.maximum(realized[valid],1e-300)),
                     'state_conformal_target':'whole sequence where present, otherwise last position',
                     'last_position_storage':'float16; tiny displacements include quantization',
                     'skipped_optional': [k for k in ('disp_full','d_pos') if k not in a]}
                folds=fit['folds'] if fit is not None else np.arange(len(d))%SETTINGS['outer']
                row['state_conformal']=conformal(realized,predictors(a,r,b),folds,self.seed)
                if fit is not None:
                    sr=apply_models(fit,{'x32':state(a,f'x{r}')}); s32=fit['scores']; tau=thresholds(fit)
                    norms=np.array([v.norm_w_raw for v in fit['models']])[folds]
                    bs=b*norms; movement=np.abs(s32-sr)
                    qr=quantization_bound(state(a,f'x{r}'),fit); q32=quantization_bound(state(a,'x32'),fit)
                    allowance=qr+q32
                    flag_r=sr>tau; flag32=s32>tau; wrong_r=labels(a,m,r)
                    transfer={'error_32':float(np.mean(flag32!=fit['y'])),
                              'error_r':float(np.mean(flag_r!=wrong_r)),
                              'flag_changed':float(np.mean(flag_r!=flag32)),
                              'label_changed':float(np.mean(wrong_r!=fit['y']))}
                    transfer['slack']=transfer['error_32']+transfer['flag_changed']+transfer['label_changed']-transfer['error_r']
                    assert transfer['slack']>=-1e-12
                    row.update(classification_transfer=transfer,quantization_allowance=distribution(allowance),
                               allowance_over_movement=distribution(allowance/np.maximum(movement,1e-300)),
                               score_violation_beyond_quantization=rate(movement>bs+allowance,valid))
                    copies=DATA[m]['copies']
                    ix=np.array([i for i,p in enumerate(a['pid']) if int(p) in copies and r in copies[int(p)] and 32 in copies[int(p)]],dtype=int)
                    rounding={'n':len(ix)}
                    if len(ix):
                        fr=np.array([fit['models'][folds[i]].score(copies[int(a['pid'][i])][r][None])[0] for i in ix])
                        f32=np.array([fit['models'][folds[i]].score(copies[int(a['pid'][i])][32][None])[0] for i in ix])
                        ratios_r=np.abs(sr[ix]-fr)/np.maximum(qr[ix],1e-300)
                        ratios_32=np.abs(s32[ix]-f32)/np.maximum(q32[ix],1e-300)
                        assert np.max(ratios_r)<=1+1e-6 and np.max(ratios_32)<=1+1e-6
                        move32=np.abs(f32-fr)
                        rounding.update(max_score_error_over_q_r=float(np.max(ratios_r)),max_score_error_over_q_32=float(np.max(ratios_32)),
                                        flag_changed_r=int(np.sum((sr[ix]>tau[ix])!=(fr>tau[ix]))),
                                        flag_changed_32=int(np.sum((s32[ix]>tau[ix])!=(f32>tau[ix]))),
                                        raw_rule_stability_changed=int(np.sum((valid[ix]&(np.abs(sr[ix]-tau[ix])>bs[ix]))!=(valid[ix]&(np.abs(fr-tau[ix])>bs[ix])))),
                                        relative_movement_change=distribution(np.abs(movement[ix]-move32)/np.maximum(move32,1e-300)))
                    row['rounding_subset']=rounding
                    stable=valid & (np.abs(sr-tau)>bs)
                    row.update(realized_score_movement=distribution(movement),score_bound=distribution(bs[valid]),
                               score_violation=rate(movement>bs,valid),fraction_forecast_stable=float(stable.mean()),
                               false_stability=rate((sr>tau)!=(s32>tau),stable),
                               score_bound_over_realized=distribution(bs[valid]/np.maximum(movement[valid],1e-300)),
                               score_conformal=conformal(movement,predictors(a,r,b,norms,True),folds,self.seed,sr,tau,s32,y=fit['y'],allowance=allowance))
                self.rows[key]=row
                if 'd_pos' in a:
                    self.rows[key]['d_pos_full_norm_relative_error']=distribution(np.abs(np.sqrt(np.nansum(a['d_pos'][:,r].astype(float)**2,axis=1))-d[:,r])[np.isfinite(a['d_pos'][:,r]).any(1)]/np.maximum(d[:,r][np.isfinite(a['d_pos'][:,r]).any(1)],1e-300))

    def m4(self):
        for m,a in self.local.items():
            for r in ([16] if m=='D' else sorted({int(m[1:]),32} if m=='L32' else {int(m[1:])})):
                c=confidence(a,m,r); d=dynamics(a,m,r); key=f'{m}/{r}'
                b=self.fit(key+'/confidence',a,m,[],r,c,True)
                self.fit(key+'/dynamics',a,m,[],r,d,True)
                joint=self.fit(key+'/joint',a,m,[],r,np.column_stack((c,d)),True)
                self.contrast(key+'/added',[(1,joint),(-1,b)])

    def f16(self):
        a=self.common(['L32','D'])
        if a is None: return
        for m in ('L32','D'):
            site='x32' if m=='L32' else 'h12'; t=32 if m=='L32' else 12
            fit=self.fit(m+'/diagonal',a[m],m,[site])
            if fit is None: continue
            copies=DATA[m]['copies']; mask=np.array([int(p) in copies for p in a[m]['pid']])
            if not mask.any() or len(np.unique(labels(a[m],m)[~mask]))<2:
                self.skips.append({'name':m+'/f16','reason':'no copies or training class absent'}); continue
            x=state(a[m],site); lam=float(np.median([c[1] for c in fit['chosen']]))
            model=fit_logistic(x[~mask],labels(a[m],m)[~mask],lam)
            x32=np.array([copies[int(p)][t] for p in a[m]['pid'][mask]])
            sd=np.std(model.score(x)); change=np.abs(model.score(x32)-model.score(x[mask]))/max(sd,1e-300)
            self.rows[m+'/rounding']={'n':int(mask.sum()),'max_score_change_over_sd':float(change.max()),
                                      'median_score_change_over_sd':float(np.median(change)),
                                      'nonconverged':not model.converged}

    def dyn(self):
        for m,a in self.local.items():
            if m=='D': continue
            d=a['d_full']; ratios=d[:,2:]/np.maximum(d[:,1:-1],1e-300)
            self.rows[m]={'n':len(d),'relative_step':nanmedian(d/a['n_full'],axis=0),
                          'ratio':nanmedian(ratios,axis=0),'ratio_times':list(range(2,65)),
                          'cosine':nanmedian(a['cos_full'],axis=0),
                          'fraction_ratio_increases_3_16':float(np.mean(ratios[:,1:15]>ratios[:,:14])),
                          'two_cycle_fraction':float(np.mean(a['e_full'][:,33]/np.maximum(d[:,33],1e-300)<.5)),
                          'accuracy':{},'token_changes':{}}
            if 'd_pos' in a: self.rows[m]['position_rate']=position_rates(a)
            for r in T_DEEP:
                mask=at_budget(a,'top_id',r)[:,0]>=0
                if mask.any(): self.rows[m]['accuracy'][str(r)]={'n':int(mask.sum()),'accuracy':float(np.mean(labels(a,m,r)[mask]==0))}
            for s,r in [(31,32),(32,33),(63,64)]:
                left=at_budget(a,'top_id',s)[:,0]; right=at_budget(a,'top_id',r)[:,0]; mask=(left>=0)&(right>=0)
                self.rows[m]['token_changes'][f'{s}/{r}']={'n':int(mask.sum()),'fraction':rate(left!=right,mask)}


def nanmedian(x,axis=0):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmedian(x,axis=axis)


def rate(event, mask):
    return float(np.mean(np.asarray(event)[mask])) if np.any(mask) else None


def distribution(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return dict(zip(('p10','median','p90'),map(float,np.quantile(x,[.1,.5,.9])))) if len(x) else None


def thresholds(fit):
    return np.array([np.quantile(fit['scores'][fit['folds']==f],.8) for f in range(len(fit['models']))])[fit['folds']]


def quantization_bound(x,fit):
    x=np.asarray(x,dtype=float); out=np.empty(len(x)); eps=2.**-11
    for f,model in enumerate(fit['models']):
        ix=fit['folds']==f; w=np.abs(model.w_raw)
        out[ix]=eps/(1-eps)*(np.abs(x[ix])@w)+2.**-25*w.sum()
    return out


def frozen_subset(fit,x,mask):
    ix=np.flatnonzero(mask); out=np.empty(len(ix))
    for f,model in enumerate(fit['models']):
        j=np.flatnonzero(fit['folds'][ix]==f); out[j]=model.score(x[ix[j]])
    return out


def rule_a(d,n,r):
    q=np.max(d[:,max(2,r-7):r+1]/np.maximum(d[:,max(2,r-7)-1:r],1e-300),axis=1)
    valid=(q<1)&(d[:,r]/np.maximum(n[:,r],1e-300)>=1e-6)&np.isfinite(q)
    b=np.full(len(d),np.inf)
    b[valid]=q[valid]*d[valid,r]*(1-q[valid]**(32-r))/(1-q[valid])
    return q,b


def rule_two_step(e,n,r):
    """The same extrapolation for the map applied twice (theory note, two-step bound).

    e[:, t] = ||X_t - X_{t-2}||. Q is the largest of the last (up to four) ratios e_t / e_{t-2}, t = r, r-2, ...;
    the forecast for ||X_32 - X_r|| is e_r (Q + Q^2 + ... + Q^m), m = (32 - r)/2. Steps that reverse direction
    make e_r small, which the one-step rule cannot see.
    """
    ts=[t for t in range(r,3,-2)][:4]
    q=np.max(np.stack([e[:,t]/np.maximum(e[:,t-2],1e-300) for t in ts]),axis=0)
    valid=(q<1)&(e[:,r]/np.maximum(n[:,r],1e-300)>=1e-6)&np.isfinite(q)
    b=np.full(len(e),np.inf); m=(32-r)//2
    b[valid]=q[valid]*e[valid,r]*(1-q[valid]**m)/(1-q[valid])
    return q,b


def predictors(a,r,b,scale=1.,last=False):
    """Forecast candidates. 'scaled_rule' (registered rule A) against 'constant' is the primary comparison."""
    out={'scaled_rule':b*scale,'constant':np.ones(len(b)),'scaled_step':a['d_full'][:,r]*scale,
         'scaled_two_step':a['e_full'][:,r]*scale,'scaled_rule_two_step':rule_two_step(a['e_full'],a['n_full'],r)[1]*scale}
    if last:
        out.update(scaled_rule_last=rule_a(a['d_last'],a['n_last'],r)[1]*scale,scaled_step_last=a['d_last'][:,r]*scale,
                   scaled_two_step_last=a['e_last'][:,r]*scale,
                   scaled_rule_two_step_last=rule_two_step(a['e_last'],a['n_last'],r)[1]*scale)
    return out


def conformal(realized,predictors,folds,seed,sr=None,tau=None,s32=None,alpha=.05,y=None,allowance=None):
    """Split-conformal multiplier, separately in every held-out fold and for every predictor.

    Half of a fold calibrates kappa = the ceil((n+1)(1-alpha))-th smallest realized/predictor; the other half
    is evaluated with the bound kappa*predictor. Items without a forecast (predictor not finite and positive)
    are left out of the calibration and of the coverage figure, and count as "not stable": whether an item has
    a forecast depends on that item alone, so the items with a forecast in the two halves stay exchangeable.
    All predictors use the same halves.
    """
    result={}
    for name,predictor in predictors.items():
        ok=np.isfinite(predictor)&(predictor>0)
        evaluation=[]; bounds=[]; sizes=[]; kappas=[]; ranking=[]
        for f in np.unique(folds):
            ix=np.flatnonzero(folds==f)
            ix=np.random.default_rng(np.random.SeedSequence([seed,int(f),4701])).permutation(ix)
            cal,ev=np.split(ix,[len(ix)//2]); cal=cal[ok[cal]]; sizes.append(len(cal))
            rank=int(np.ceil((len(cal)+1)*(1-alpha)))
            k=np.sort(realized[cal]/predictor[cal])[rank-1] if rank<=len(cal) else np.inf
            bound=np.full(len(ev),np.inf)
            if np.isfinite(k): bound[ok[ev]]=k*predictor[ev][ok[ev]]
            if sr is not None and y is not None and len(np.unique(y[ev]))==2:
                wrong=np.flatnonzero(y[ev]==1); right=np.flatnonzero(y[ev]==0)
                close=np.abs(sr[ev[wrong],None]-sr[ev[right]][None,:])<=bound[wrong,None]+bound[right][None,:]
                violation=realized[ev]>bound
                violated=(violation[wrong,None]|violation[right][None,:])&~close
                change=abs(auroc(s32[ev],y[ev])-auroc(sr[ev],y[ev]))
                assert change<=close.mean()+violated.mean()+1e-12
                ranking.append((change,float(close.mean()),float(violated.mean())))
            evaluation.extend(ev); bounds.extend(bound); kappas.append(k)
        ev=np.array(evaluation,dtype=int); bound=np.array(bounds); has=np.isfinite(bound)
        row={'n_evaluation':len(ev),'calibration_sizes':sizes,'kappa':distribution(kappas),
             'forecast_fraction':float(has.mean()),
             'coverage':rate(realized[ev]<=bound,has),
             'bound_over_realized':distribution(bound[has]/np.maximum(realized[ev][has],1e-300))}
        if sr is not None:
            stable=has&(np.abs(sr[ev]-tau[ev])>bound)
            row.update(fraction_forecast_stable=float(stable.mean()),false_stability=rate((sr[ev]>tau[ev])!=(s32[ev]>tau[ev]),stable),
                       decision_changed=float(np.mean((sr[ev]>tau[ev])!=(s32[ev]>tau[ev]))))
            row['stable_and_violated']=float(np.mean(stable&(realized[ev]>bound)))
            row['decision_change_bound']=1-row['fraction_forecast_stable']+row['stable_and_violated']
            assert row['decision_changed']<=row['decision_change_bound']+1e-12
            if allowance is not None:
                row.update(coverage_with_allowance=rate(realized[ev]<=bound+allowance[ev],has),
                           fraction_forecast_stable_with_allowance=float(np.mean(has&(np.abs(sr[ev]-tau[ev])>bound+allowance[ev]))))
            if y is not None:
                row.update(ranking_folds_used=len(ranking))
                for j,k in enumerate(('auroc_change_fixed_labels','close_pair_mass','violated_pair_mass')):
                    row[k]=float(np.mean([v[j] for v in ranking])) if ranking else None
        result[name]=row
    return result


def position_rates(a,windows=((8,16),(16,24),(24,32)),slack=.02):
    """Decay rate of the step at each token position over a window of loops, from the saved per-position step sizes. What the block-triangular Jacobian implies (theory note, Theorem 4): the asymptotic rate of the first p positions TAKEN TOGETHER is non-decreasing in p. A single position inherits the slowest rate among the positions it reads only if the coupling to that position is non-zero, so 'rates do not fall as p grows' is a prediction for generic coupling, not a theorem; and windows inside 33 or 64 loops are finite-time evidence, not asymptotics. A position enters only if its step at the window's end is above the float32 floor. The 'prefix_*' entries are the same statistics for the step over positions 1..p together (the quantity the theorem is about; at finite time small dips remain possible), and 'last_position_minus_whole_state' compares the monitored position with the whole sequence."""
    from scipy import stats
    dp=a['d_pos'].astype(float); length=np.isfinite(dp[:,1]).sum(1); out={}
    for t1,t2 in windows:
        with np.errstate(divide='ignore',invalid='ignore'):
            lograte=(np.log(dp[:,t2])-np.log(dp[:,t1]))/(t2-t1)
        real=np.isfinite(dp[:,t2])
        # Norm of each position at the window's end: the saved per-position norms when an item has them at every
        # real position, otherwise (older data) the whole-state norm spread evenly over its positions.
        floor=np.where(real,a['n_full'][:,t2,None]/np.sqrt(np.maximum(length,1))[:,None],np.nan)
        own=np.zeros(len(dp),bool)
        if 'n_pos' in a and t2 in R_NPOS:
            saved=a['n_pos'][:,R_NPOS.index(t2)].astype(float)
            own=real.any(1)&(np.isfinite(saved)|~real).all(1)
            floor[own]=np.where(real[own],saved[own],np.nan)
        ok=np.isfinite(lograte)&(dp[:,t2]>=1e-6*floor)
        with np.errstate(divide='ignore',invalid='ignore'):
            p1=np.sqrt(np.nancumsum(dp[:,t1]**2,axis=1)); p2=np.sqrt(np.nancumsum(dp[:,t2]**2,axis=1))
            prate=(np.log(p2)-np.log(p1))/(t2-t1)
        pfloor=np.sqrt(np.nancumsum(floor**2,axis=1))  # norm of the first p positions together
        pok=real&np.isfinite(prate)&np.isfinite(floor)&(p2>=1e-6*pfloor)
        rho=[]; drops=[]; shortfall=[]; last=[]; used=0
        prho=[]; pdrops=[]; pshort=[]; whole=[]; pused=0
        for i in range(len(dp)):
            j=np.flatnonzero(pok[i])
            if len(j)>=8:
                v=prate[i,j]; pused+=1
                prho.append(stats.spearmanr(j,v).statistic)
                pdrops.append(np.mean(v[1:]<v[:-1]-slack))
                pshort.append(np.max(np.maximum.accumulate(v)-v))
                end=length[i]-1
                if ok[i,end] and pok[i,end]: whole.append(lograte[i,end]-prate[i,end])
            j=np.flatnonzero(ok[i])
            if len(j)<8: continue
            v=lograte[i,j]; used+=1
            rho.append(stats.spearmanr(j,v).statistic)
            drops.append(np.mean(v[1:]<v[:-1]-slack))
            shortfall.append(np.max(np.maximum.accumulate(v)-v))
            last.append(v[-1]-np.median(v))
        out[f'{t1}-{t2}']={'floor':{'per_position_norm_items':int(own.sum()),'whole_state_norm_items':int((real.any(1)&~own).sum())},'items_used':used,'positions_used':distribution(ok.sum(1)),
                           'log_rate':distribution(lograte[ok]),
                           'spearman_position_vs_rate':distribution(rho),
                           'fraction_adjacent_drops':distribution(drops),
                           'largest_shortfall_below_running_max':distribution(shortfall),
                           'last_position_minus_median':distribution(last),
                           'prefix_items_used':pused,'prefix_log_rate':distribution(prate[pok]),
                           'prefix_spearman_position_vs_rate':distribution(prho),
                           'prefix_fraction_adjacent_drops':distribution(pdrops),
                           'prefix_largest_shortfall_below_running_max':distribution(pshort),
                           'last_position_minus_whole_state':distribution(whole)}
    return out


def dynamics(a,m,r):
    if m=='D':
        times=list(range(17)); tokens=a['lens_top_id'][:,:,0]; lp=a['lens_top_lp'][:,:,0]; half=8
        features=[np.log(np.maximum(a[k][:,r],1e-300)) for k in ('dl_full','dl_last')]
    else:
        times=[t for t in T_STD if t<=r]; ix=[T_DEEP.index(t) for t in times]
        tokens=a['top_id'][:,ix,0]; lp=a['top_lp'][:,ix,0]; half=times.index(r//2)
        features=[np.log(np.maximum(a[k][:,r],1e-300)) for k in ('d_full','e_full','d_last','e_last')]
        features.extend(a[k][:,r] for k in ('cos_full','cos_last'))
    change=tokens[:,1:]!=tokens[:,:-1]
    features.extend([np.max(np.where(change,np.array(times[1:]),0),axis=1),change.sum(1),
                     tokens[:,half]==tokens[:,-1],lp[:,-1]-lp[:,half]])
    return np.column_stack(features)


def digits(values):
    values=np.asarray(values,dtype=int); cols=[]
    for power in (1,10,100): cols.append(np.eye(10)[np.abs(values)//power%10])
    nd=np.where(values==0,1,np.floor(np.log10(np.maximum(np.abs(values),1)))+1).astype(int)
    cols.append(np.eye(12)[np.minimum(nd,11)])
    return np.column_stack(cols)


def token_answers(valid_ids):
    """(answer kind, token id) -> answer for every valid answer token (integers 0..999; letters A-D), from the
    token ids the collection saved in its metadata. Independent of which answers occur in the data."""
    if not valid_ids: return {}
    out={('num',int(t)):i for i,t in enumerate(valid_ids['num'])}
    out.update({('abcd',int(t)):c for t,c in zip(valid_ids['abcd'],'ABCD')})
    assert len(valid_ids['num'])==1000 and len(valid_ids['abcd'])==4 and len(out)==1004, 'valid answer tokens are not distinct'
    return out


def problem_features(a,m):
    """Output-identity features (digits or letter of the PREDICTED token) and privileged features (operands, gold
    answer, carries). A predicted token that is not a valid answer token gets the indicator 'unknown' and zeros."""
    pids=a['pid']; its=[ITEMS[int(p)] for p in pids]; mc=its[0]['valid']=='abcd'
    if VALID:
        mapping=VALID
        assert all(VALID.get((it['valid'],int(t)))==it['ans'] for it,t in zip(its,a['ans_tok'])), 'answer tokens disagree with the saved valid tokens'
    else:
        # Legacy data without saved valid tokens (the screen): only tokens of observed gold answers are known, so
        # these features depend on gold answers. Main data must not take this branch (load_runs enforces it).
        mapping={}
        for model,z in DATA.items():
            for p,t in zip(z['pid'],z['ans_tok']):
                ans=ITEMS[int(p)]['ans']; key=(ITEMS[int(p)]['valid'],int(t))
                assert key not in mapping or mapping[key]==ans, 'inconsistent answer-token mapping'
                mapping[key]=ans
    pred=a['ro_top_id'][:,0] if m=='D' else at_budget(a,'top_id',32)[:,0]
    mapped=[mapping.get((it['valid'],int(t))) for it,t in zip(its,pred)]; unknown=np.array([v is None for v in mapped])
    if mc:
        encode=lambda v:np.array([[float(x==c) for c in 'ABCD'] for x in v])
        return np.column_stack((encode(mapped),unknown)),np.column_stack((encode([it['ans'] for it in its]),a['plen']))
    encoded=digits([v if v is not None else 0 for v in mapped]); encoded[unknown]=0
    identity=np.column_stack((encoded,unknown))
    cols=[digits([it['ops'][j] for it in its]) for j in range(len(its[0]['ops']))]
    cols.append(digits([it['ans'] for it in its]))
    if its[0]['fam'] in ('add3','add4','add3w'):
        counts=[]
        for it in its:
            carry=total=0
            for power in (1,10,100):
                carry=(sum(int(v)//power%10 for v in it['ops'])+carry)//10; total+=carry
            counts.append(total)
        cols.append(np.array(counts)[:,None])
    return identity,np.column_stack(cols)


def worker(task):
    name,fam,seed=task; start=time.perf_counter(); a=Analysis(fam,seed)
    getattr(a,name)()
    return name,fam,seed,{'estimates':a.rows,'skipped':a.skips,'seconds':time.perf_counter()-start},a.scores,a.differences


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def array_mean(values):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmean(np.stack(values),axis=0)


def aggregate(rows, draws, n_families):
    # Equal family weights; a missing family makes a numeric leaf unavailable.
    incomplete={}
    def merge(values,path=''):
        present=[v for v in values if v is not None]
        if present and all(isinstance(v,dict) for v in present):
            keys=set.union(*(set(v) for v in present))
            return {k:merge([v.get(k) for v in present],path+'/'+k if path else k) for k in sorted(keys)
                    if k not in ('chosen','solver_messages','classes','interval','transfer_interval')}
        if present and all(isinstance(v,(int,float,np.number,np.ndarray)) and not isinstance(v,bool) for v in present):
            if len(present)!=n_families:
                incomplete[path]=len(present); return None
            if all(isinstance(v,np.ndarray) for v in present):
                return np.mean(np.stack(present),axis=0) if len({v.shape for v in present})==1 else None
            return float(np.mean(present))
        if not present: incomplete[path]=0
        return None
    result=merge(rows) if rows else {}
    for k,vs in draws.items():
        transfer=k.endswith('/transfer') and k[:-9] in result
        row=result.setdefault(k[:-9] if transfer else k,{})
        prefix='transfer_' if transfer else ''
        average=np.mean(vs,axis=0); bad=int((~np.isfinite(average)).sum())
        complete=len(vs)==n_families
        row[prefix+'interval']=list(map(float,np.percentile(average,[2.5,97.5]))) if complete and not bad else [None,None]
        row[prefix+'undefined_bootstrap_draws']=bad
        row[prefix+'families']=len(vs)
        if not complete: row[prefix+'incomplete']=True
        if bad: row[prefix+'interval_unavailable_reason']=interval_reason(bad,len(average))
    result['incomplete']=incomplete
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    global DATA,ITEMS,SETTINGS,VALID
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True); p.add_argument('--items',required=True); p.add_argument('--out',required=True)
    p.add_argument('--manifest'); p.add_argument('--fams'); p.add_argument('--models'); p.add_argument('--seeds'); p.add_argument('--jobs',type=int,default=6)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--only'); p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args(argv); start=time.perf_counter()
    SETTINGS={'smoke':args.smoke,'outer':3 if args.smoke else 5,'inner':2 if args.smoke else 4,'boot':200 if args.smoke else 2000,
              'baseline':'constant zero X; all confidence/baseline features in unpenalized U',
              'intervals':'conditional on OOF fits; intersection of all participating folds; any undefined bootstrap draw makes the entire interval unavailable',
              'amendments':['wrong=1 (design uses correctness=1)','per-held-out-fold 80th percentile threshold','conformal calibration split within held-out folds'],
              'conventions':{'table_effects':'row and column: half-sum contrasts; interaction: difference of differences',
                             'folds':'2x2 table: one fold assignment balanced over joint correctness categories; other fits: stratified on their own label; bootstrap within participating fold intersections',
                             'missing_disp_full':'path and float16 last-position checks remain; state conformal falls back to last position',
                             'carries':'sum carry values over decimal columns, pure additions only',
                             'family_mean':'equal weights across all run families; missing numeric leaves and incomplete bootstrap intervals are unavailable'}}
    seeds=[0] if args.smoke else list(map(int,args.seeds.split(','))) if args.seeds else [0,1,2]
    ITEMS={int(it['pid']):it for it in json.loads(Path(args.items).read_text())['items']}
    manifest=json.loads(Path(args.manifest).read_text()) if args.manifest else None
    if manifest is not None: assert sha256(args.items)==manifest['items_sha256'], 'items hash mismatch'
    DATA,prov=load_runs(args.runs,args.models.split(',') if args.models else None,args.allow_partial,manifest,ITEMS)
    fams=args.fams.split(',') if args.fams else manifest['families'] if manifest else [f for f in FAMS if any(ITEMS[int(pid)]['fam']==f for a in DATA.values() for pid in a['pid'])]
    SETTINGS['fams']=fams
    VALID=token_answers(prov.pop('valid_ids'))
    SETTINGS['output_identity_tokens']='valid answer tokens saved by the collection' if VALID else 'LEGACY: tokens of observed gold answers (depends on gold answers)'
    names=args.only.split(',') if args.only else ANALYSES
    if not set(names)<=set(ANALYSES): p.error('unknown --only analysis')
    prov.update(runs=[str(Path(r).resolve()) for r in args.runs],items=str(Path(args.items).resolve()),
                items_sha256=hashlib.sha256(Path(args.items).read_bytes()).hexdigest(),
                analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                item_counts={m:{f:sum(ITEMS[int(pid)]['fam']==f for pid in a['pid']) for f in fams} for m,a in DATA.items()})
    if manifest is not None:
        current={name:sha256(Path(__file__).with_name(name)) for name in ('analyze_mon.py','mon_probe.py')}
        expected=manifest.get('analysis_sha256'); matches=None
        if expected is not None:
            base=Path(__file__).resolve().parent
            def matches_hash(name,digest):
                path=next((p for p in (base.parent/name,base/name,base.parent/'docs'/name) if p.is_file()),None)
                return path is not None and sha256(path)==digest
            matches=all(matches_hash(name,digest) for name,digest in expected.items())
        prov.update(manifest=str(Path(args.manifest).resolve()),manifest_sha256=sha256(args.manifest),
                    current_analysis_sha256=current,analysis_matches_manifest=matches)
    results={'provenance':prov,'settings':dict(SETTINGS,seeds=seeds,jobs=args.jobs),'analyses':{}}
    scores={}; all_draws={}; analysis_times={}
    tasks=[(name,f,seed) for name in names for f in fams for seed in seeds]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('fork')) as pool:
        futures={pool.submit(worker,t):t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            name,fam,seed,row,ss,dd=future.result()
            results['analyses'].setdefault(name,{}).setdefault(str(seed),{})[fam]=row
            analysis_times[name]=analysis_times.get(name,0)+row['seconds']
            for key,values in ss.items():
                if name.startswith(('m1','m3')):
                    for field,value in values.items(): scores[f'{name}/{fam}/{seed}/{key}/{field}']=value
            for key,value in dd.items(): all_draws.setdefault((name,seed,key),[]).append(value)
            print(f'{name} {fam} seed={seed}: {row["seconds"]:.2f}s',flush=True)
    for name,byseed in results['analyses'].items():
        for seed,byfam in byseed.items():
            draws={k:vs for (n,s,k),vs in all_draws.items() if n==name and str(s)==seed}
            byfam['family_mean']={'estimates':aggregate([v['estimates'] for v in byfam.values()],draws,len(fams)),'families':list(byfam)}
    results['seconds']=time.perf_counter()-start; results['analysis_worker_seconds']=analysis_times
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(clean(results),indent=2,allow_nan=False)+'\n')
    np.savez(out/'scores.npz',**scores)
    print(f'Completed {len(fams)} families in {results["seconds"]:.2f}s; worker seconds by analysis: {analysis_times}',flush=True)
    return results


if __name__=='__main__': main()
