"""Exploratory E4 (docs/mon_exploratory.md): paired comparison of the registered position -2 probes. No fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

FAMS = ('add3', 'add4', 'add3w', 'arc_easy')
L_SITE, D_SITES = 'L32/prev32', ('D/prev12', 'D/prev8', 'D/prev16')


def auroc(s, y):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fold_mean(s, y, folds, w=None):
    """Mean over folds of the AUROC; w = multiplicity of each problem in a resample (None: every problem once)."""
    out = []
    for f in np.unique(folds):
        idx = np.flatnonzero(folds == f)
        if w is not None:
            idx = np.repeat(idx, w[idx])
        out.append(auroc(s[idx], y[idx]))
    return float(np.mean(out))


def load(z, fam, seed, site):
    p = f'm1s/{fam}/{seed}/{site}/'
    order = np.argsort(z[p + 'pid'])
    return {k: z[p + k][order] for k in ('pid', 'y', 'scores', 'folds')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--draws', type=int, default=2000)
    a = ap.parse_args()
    rows, checks = [], []
    for d in a.inputs:
        d = Path(d)
        res = json.loads((d / 'results.json').read_text())['analyses']['m1s']
        seed = int(next(iter(res)))
        z = np.load(d / 'scores.npz')
        for dsite in D_SITES:
            point, boot = [], []
            for fi, fam in enumerate(FAMS):
                L, D = load(z, fam, seed, L_SITE), load(z, fam, seed, dsite)
                assert np.array_equal(L['pid'], D['pid'])
                pl, pd = fold_mean(L['scores'], L['y'], L['folds']), fold_mean(D['scores'], D['y'], D['folds'])
                for site, v in ((L_SITE, pl), (dsite, pd)):
                    checks.append(abs(v - res[str(seed)][fam]['estimates'][site]['auroc']))
                rng = np.random.default_rng(np.random.SeedSequence([seed, fi, 5521]))  # same resamples for every D site
                n = len(L['pid'])
                b = np.empty(a.draws)
                for i in range(a.draws):
                    w = np.bincount(rng.integers(0, n, n), minlength=n)
                    b[i] = fold_mean(L['scores'], L['y'], L['folds'], w) - fold_mean(D['scores'], D['y'], D['folds'], w)
                point.append((pl, pd)); boot.append(b)
                rows.append(dict(seed=seed, family=fam, d_site=dsite, L32=pl, D=pd, difference=pl - pd,
                                 interval=None if np.isnan(b).any() else [float(x) for x in np.quantile(b, [.025, .975])],
                                 undefined_draws=int(np.isnan(b).sum())))
            b = np.mean(boot, axis=0)
            pl, pd = np.mean(point, axis=0)
            rows.append(dict(seed=seed, family='family_mean', d_site=dsite, L32=float(pl), D=float(pd), difference=float(pl - pd),
                             interval=None if np.isnan(b).any() else [float(x) for x in np.quantile(b, [.025, .975])],
                             undefined_draws=int(np.isnan(b).sum())))
    out = dict(rows=rows, reproduction_max_abs_difference=float(max(checks)), draws=a.draws, l_site=L_SITE,
               note='Exploratory; chosen after seeing the registered rows. Resamples problems, monitors fixed.')
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print('reproduction max abs difference', out['reproduction_max_abs_difference'])
    for r in rows:
        if r['family'] == 'family_mean':
            iv = r['interval']  # None when some resample had only one class in a fold
            print(r['seed'], r['d_site'], round(r['difference'], 4), 'unavailable' if iv is None else [round(x, 4) for x in iv])


if __name__ == '__main__':
    main()
