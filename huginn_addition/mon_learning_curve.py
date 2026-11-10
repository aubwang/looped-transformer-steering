"""Exploratory E1 learning curves; CPU only, conditional held-out bootstrap."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import resource
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_mon as mon
from mon_probe import LAMS, auroc, check_folds, fit_path, fit_logistic, make_folds, make_joint_folds, paired_bootstrap

MODELS = ('L32', 'D')
FITS = ('probe', 'confidence', 'confidence+probe')
FROZEN = ('huginn_addition/analyze_mon.py', 'huginn_addition/mon_probe.py',
          'docs/oversight_monitor.md', 'docs/mon_spec.md')
REFERENCE = Path('results/mon_analysis-20261110/seed0/results.json')
TASK_DATA = {}


def training_subset(tr, y, fraction, seed, family, model, fold, draw):
    """Classwise prefixes share one permutation across fractions and fit types."""
    if fraction == 1:
        return tr.copy()  # preserve nested_cv's ordering, hence exact solver inputs
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, mon.FAMS.index(family), MODELS.index(model), fold, draw, 8147]))
    chosen = [rng.permutation(tr[y[tr] == c])[:int(np.ceil(fraction * np.sum(y[tr] == c)))]
              for c in (0, 1)]
    return np.sort(np.concatenate(chosen))


def reduced_cv(X, y, folds, fraction, draw, seed=0, family='add3', model='L32',
               U=None, lams=LAMS, n_outer=5, n_inner=4):
    """Mirror nested_cv's single-site logistic inner selection, including tie order.

    Nonconverged fits remain in selection and scoring. If a subset cannot form
    registered inner folds, its held-out scores remain NaN and the reason is saved.
    """
    check_folds(folds, y, n_outer)
    lams = sorted(set(float(x) for x in lams), reverse=True)
    scores = np.full(len(y), np.nan)
    counts = dict(inner_nonconverged=0, nonconverged=0, inner_class_failures=0)
    reasons, chosen, sizes = [], [], []
    for f in range(n_outer):
        tr = training_subset(np.flatnonzero(folds != f), y, fraction, seed, family, model, f, draw)
        te = np.flatnonzero(folds == f)
        sizes.append(len(tr))
        try:
            inner = make_folds(y[tr], n_inner, np.random.SeedSequence([seed, f, 917]))
        except ValueError as e:
            counts['inner_class_failures'] += 1
            reasons.append(dict(fold=f, reason=str(e), classes=np.bincount(y[tr], minlength=2).tolist()))
            chosen.append(None)
            continue
        merit = np.zeros(len(lams))
        for g in range(n_inner):
            it, iv = tr[inner != g], tr[inner == g]
            ms = fit_path(X[it], y[it], lams, None if U is None else U[it])
            merit += [auroc(m.score(X[iv], None if U is None else U[iv]), y[iv]) for m in ms]
            counts['inner_nonconverged'] += sum(not m.converged for m in ms)
            reasons.extend(dict(fold=f, inner=g, reason=m.message) for m in ms if not m.converged)
        lam = lams[int(np.argmax(merit))]
        m = fit_logistic(X[tr], y[tr], lam, None if U is None else U[tr])
        counts['nonconverged'] += int(not m.converged)
        if not m.converged:
            reasons.append(dict(fold=f, reason=m.message))
        scores[te] = m.score(X[te], None if U is None else U[te])
        chosen.append(lam)
    value = metric(scores, y, folds)
    return dict(scores=scores, folds=folds, y=y, auroc=value, training_items=float(np.mean(sizes)),
                training_items_by_fold=sizes, chosen=chosen, failures=reasons, **counts)


def metric(scores, y, folds, ix=None):
    ix = np.arange(len(y)) if ix is None else ix
    if not np.isfinite(scores[ix]).all():
        return float('nan')
    return mon.fold_metric(scores, y, folds, ix)


def interval(draws):
    bad = int(np.sum(~np.isfinite(draws)))
    return dict(interval=[None, None] if bad else np.percentile(draws, [2.5, 97.5]).tolist(),
                undefined_bootstrap_draws=bad,
                interval_unavailable_reason='a resampled fold lost a class or fitted scores are unavailable' if bad else None)


def bootstrap_gain(terms, n_boot, seed):
    # Each signed term is a draw, already weighted by 1 / number of subset draws.
    # All terms share item order. Resample within intersections of participating folds.
    groups = np.column_stack([r['folds'] for _, r in terms])
    def stat(ix):
        return sum(w * metric(r['scores'], r['y'], r['folds'], ix) for w, r in terms)
    est, _, _, draws = paired_bootstrap(stat, groups, n_boot, seed)
    return dict(estimate=float(est), **interval(draws)), draws


def worker(task):
    fam, model, fit, fraction, draw, seed = task
    start = time.perf_counter()
    d = TASK_DATA[fam, model]
    baseline = fit == 'confidence'
    X = np.zeros((len(d['y']), 1)) if baseline else d['X']
    r = reduced_cv(X, d['y'], d['joint'] if fit == 'probe' else d['own'], fraction, draw,
                   seed, fam, model, None if fit == 'probe' else d['U'], [1.] if baseline else LAMS)
    r.update(pid=d['pid'], seconds=time.perf_counter()-start,
             peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return task, r


def cell_summary(rows):
    values = [r['auroc'] for r in rows]
    return dict(training_items=float(np.mean([r['training_items'] for r in rows])),
                auroc=float(np.mean(values)), draw_aurocs=values,
                draw_spread_std=float(np.std(values)), draw_range=[float(np.min(values)), float(np.max(values))],
                draws=len(rows), **{k:sum(r[k] for r in rows) for k in
                ('inner_nonconverged', 'nonconverged', 'inner_class_failures')})


def summarize(records, fams, fractions, n_boot, seed):
    curves, derived, boot_draws = {}, {}, {}
    for fam in fams:
        curves[fam], derived[fam] = {}, {}
        for model in MODELS:
            curves[fam][model] = {fit:{} for fit in FITS}
            derived[fam][model] = {'added':{}, 'last_doubling':{}}
            for fit in FITS:
                for phi in fractions:
                    rows = records[fam, model, fit, phi]
                    curves[fam][model][fit][str(phi)] = cell_summary(rows)
            for phi in fractions:
                c = curves[fam][model]
                a = np.array(c['confidence+probe'][str(phi)]['draw_aurocs']) - c['confidence'][str(phi)]['draw_aurocs']
                derived[fam][model]['added'][str(phi)] = dict(auroc=float(a.mean()), draw_values=a.tolist(), draw_spread_std=float(a.std()))
            for quantity in ('probe', 'added'):
                if 0.5 not in fractions or 1. not in fractions:
                    derived[fam][model]['last_doubling'][quantity] = dict(estimate=None, interval=[None,None], interval_unavailable_reason='fractions 0.5 and 1 required')
                    continue
                terms = []
                for phi, sign in ((1., 1), (.5, -1)):
                    fits = [('probe', 1)] if quantity == 'probe' else [('confidence+probe', 1), ('confidence', -1)]
                    for fit, weight in fits:
                        rows = records[fam, model, fit, phi]
                        terms.extend((sign*weight/len(rows), r) for r in rows)
                row, draws = bootstrap_gain(terms, n_boot, seed+171+100003*mon.FAMS.index(fam))
                derived[fam][model]['last_doubling'][quantity] = row
                boot_draws[fam, model, quantity] = draws
        derived[fam]['L32_minus_D'] = {q:{str(phi):
            (curves[fam]['L32']['probe'][str(phi)]['auroc']-curves[fam]['D']['probe'][str(phi)]['auroc']) if q == 'probe' else
            (derived[fam]['L32']['added'][str(phi)]['auroc']-derived[fam]['D']['added'][str(phi)]['auroc'])
            for phi in fractions} for q in ('probe', 'added')}
    # Equal family weights; NaN propagates, never silently average incomplete results.
    def average(values):
        if isinstance(values[0], dict):
            return {k:average([v[k] for v in values]) for k in values[0]}
        if isinstance(values[0], list):
            return np.mean(np.asarray(values, dtype=float), axis=0).tolist()
        if values[0] is None or isinstance(values[0], str):
            return None
        return float(np.mean(values))
    curves['family_mean'] = average([curves[f] for f in fams])
    derived['family_mean'] = average([derived[f] for f in fams])
    # Spread of the family-mean draws (rather than mean of family spreads).
    for model in MODELS:
        for fit in FITS:
            for phi in fractions:
                row = curves['family_mean'][model][fit][str(phi)]
                row['draw_spread_std'] = float(np.std(row['draw_aurocs']))
                row['draw_range'] = [float(np.min(row['draw_aurocs'])),float(np.max(row['draw_aurocs']))]
        for phi in fractions:
            row = derived['family_mean'][model]['added'][str(phi)]
            row['draw_spread_std'] = float(np.std(row['draw_values']))
        for q in ('probe', 'added'):
            if all((f,model,q) in boot_draws for f in fams):
                derived['family_mean'][model]['last_doubling'][q].update(interval(np.mean([boot_draws[f,model,q] for f in fams], axis=0)))
    return curves, derived


def reproduction(curves, fams, seed):
    ref = json.loads(REFERENCE.read_text())
    checks = []
    for fam in fams:
        for model in MODELS:
            for fit in FITS:
                analysis, key = ('m1a', f'table/{model}/{model}') if fit == 'probe' else ('m1c', f'fixed/{model}/{fit}')
                expected = ref['analyses'][analysis]['0'][fam]['estimates'][key]['auroc']
                actual = curves[fam][model][fit].get('1.0', {}).get('auroc', float('nan'))
                checks.append(dict(family=fam, model=model, fit=fit, expected=expected, actual=actual,
                                   abs_difference=abs(actual-expected)))
    delta = float(np.max([c['abs_difference'] for c in checks]))
    return dict(reference=str(REFERENCE), reference_sha256=mon.sha256(REFERENCE), tolerance=1e-9,
                max_abs_difference=delta, passed=bool(seed == 0 and np.isfinite(delta) and delta <= 1e-9), checks=checks,
                note='seed-0 reference; other seeds are not expected to reproduce it')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True)
    for name in ('items', 'manifest', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--fams', nargs='+', help='space or comma separated; default: manifest families')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fractions', default='0.125,0.25,0.5,0.75,1')
    p.add_argument('--draws', type=int, default=3)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    fractions = sorted(set(map(float, args.fractions.split(','))))
    if args.jobs < 1 or args.draws < 1 or not fractions or any(not 0 < f <= 1 for f in fractions):
        p.error('jobs/draws must be positive and fractions in (0,1]')
    start = time.perf_counter()
    manifest = json.loads(Path(args.manifest).read_text())
    items = {int(it['pid']):it for it in json.loads(Path(args.items).read_text())['items']}
    assert mon.sha256(args.items) == manifest['items_sha256'], 'items hash mismatch'
    fams = [f for value in args.fams for f in value.split(',')] if args.fams else manifest['families']
    draws, n_boot = args.draws, 2000
    if args.smoke:
        fams, fractions, draws, n_boot = ['add3'], [.125, 1.], 1, 200
    data, provenance = mon.load_runs(args.runs, models=list(MODELS), manifest=manifest, items=items)
    for fam in fams:
        local = {m:mon.subset(a, a['pid'][[items[int(pid)]['fam'] == fam for pid in a['pid']]]) for m,a in data.items()}
        pids = np.array(sorted(set(local['L32']['pid']) & set(local['D']['pid'])))
        local = {m:mon.subset(a, pids) for m,a in local.items()}
        ys = {m:mon.labels(a,m) for m,a in local.items()}
        joint = make_joint_folds(np.column_stack([ys[m] for m in MODELS]), 5,
                                 np.random.SeedSequence([args.seed, mon.FAMS.index(fam), 2201]))
        for m,a in local.items():
            TASK_DATA[fam,m] = dict(X=mon.state(a,'x32' if m == 'L32' else 'h12'), U=mon.confidence(a,m),
                                    y=ys[m], pid=pids, joint=joint, own=make_folds(ys[m],5,args.seed))
    tasks = [(f,m,k,phi,d,args.seed) for f in fams for m in MODELS for k in FITS
             for phi in fractions for d in range(1 if phi == 1 else draws)]
    records, score_arrays, timings = {}, {}, []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context('fork')) as pool:
        for task,r in map(lambda future:future.result(), concurrent.futures.as_completed([pool.submit(worker,t) for t in tasks])):
            f,m,k,phi,d,_ = task
            records.setdefault((f,m,k,phi), {})[d] = r
            prefix = f'{f}/{m}/{k}/{phi}/{d}'
            for field in ('pid','y','folds','scores','training_items_by_fold'):
                score_arrays[prefix+'/'+field] = r[field]
            timings.append(dict(task=list(task), seconds=r['seconds'], peak_rss_kib=r['peak_rss_kib'],
                                chosen=r['chosen'], failures=r['failures']))
            print(f'{prefix}: {r["seconds"]:.2f}s peak_rss={r["peak_rss_kib"]} KiB', flush=True)
    records = {key:[v[d] for d in sorted(v)] for key,v in records.items()}
    curves, derived = summarize(records,fams,fractions,n_boot,args.seed)
    settings = dict(seed=args.seed, families=fams, fractions=fractions, draws=draws, outer=5, inner=4,
                    penalty_grid=LAMS.tolist(), confidence_penalties=[1.], bootstrap_draws=n_boot, jobs=args.jobs,
                    smoke=args.smoke, subsample_seed_components=['seed','family_index','model_index','fold','draw',8147],
                    folds={'probe':'registered m1a shared joint L32/D labels, SeedSequence([seed,family_index,2201])',
                           'confidence':'registered m1c own-label make_folds(y,5,seed)',
                           'confidence+probe':'registered m1c own-label make_folds(y,5,seed)'},
                    spread='population standard deviation and range of draw AUROCs; full size one draw',
                    bootstrap='mean AUROC over subset draws first, paired within participating fold intersections; family mean equal weights',
                    family_mean='equal weights over requested families; smoke has add3 only',
                    inner_class_failures='number of outer subsets unable to form four valid inner folds; scores unavailable',
                    nonconvergence='all fits retained, counts summed over subset draws; family mean counts averaged')
    provenance.update(hashes={path:mon.sha256(path) for path in (*FROZEN,args.items,args.manifest)},
                      script_sha256=mon.sha256(__file__), runs=args.runs)
    result = dict(curves=curves, derived=derived, reproduction=reproduction(curves,fams,args.seed),
                  provenance=provenance, settings=settings, task_timings=timings,
                  total_worker_seconds=sum(t['seconds'] for t in timings), seconds=time.perf_counter()-start,
                  parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'curve_scores.npz', **score_arrays)
    (out/'curve.json').write_text(json.dumps(mon.clean(result), indent=2, allow_nan=False)+'\n')
    print(f'Total worker-seconds: {result["total_worker_seconds"]:.2f}; wall: {result["seconds"]:.2f}s; reproduction: {result["reproduction"]["passed"]}, max difference {result["reproduction"]["max_abs_difference"]:.12g}', flush=True)
    return result


if __name__ == '__main__':
    main()
