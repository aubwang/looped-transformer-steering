"""Small synthetic checks; run directly, without experiment inputs or pytest."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('prev', Path(__file__).with_name('mon_explore_prev.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def pairwise(s, y):
    positive, negative = s[y == 1], s[y == 0]
    return np.mean((positive[:, None] > negative).astype(float)
                   + .5 * (positive[:, None] == negative))


def make_inputs(root, paired=False):
    inputs = []
    for seed in (0, 1):
        directory = root / f'seed{seed}'
        directory.mkdir(parents=True)
        arrays, estimates = {}, {}
        # Large balanced folds make undefined draws extremely unlikely, while
        # perfect/reversed/constant rankings give exact, known intervals.
        y = np.tile(np.repeat([0, 1], 40), 2)
        folds = np.repeat([0, 1], 80)
        pid = np.arange(len(y))
        for fi, fam in enumerate(m.FAMS):
            estimates[fam] = {'estimates': {}}
            for si, site in enumerate((m.L_SITE,) + m.D_SITES):
                if paired:
                    # Imperfect, tied rankings: independent resamples would
                    # generally produce nonzero differences here.
                    scores = (pid % 7).astype(float)
                else:
                    scores = y.astype(float) if si == 0 else (
                        -y.astype(float) if (fi + si) % 2 else np.zeros(len(y)))
                estimates[fam]['estimates'][site] = {
                    'auroc': float(np.mean([pairwise(scores[folds == f], y[folds == f])
                                           for f in np.unique(folds)]))}
                # Different on-disk orders exercise alignment by problem ID.
                order = np.random.default_rng(seed * 100 + fi * 10 + si).permutation(len(y))
                prefix = f'm1s/{fam}/{seed}/{site}/'
                for key, value in dict(pid=pid, y=y, scores=scores, folds=folds).items():
                    arrays[prefix + key] = value[order]
        np.savez(directory / 'scores.npz', **arrays)
        (directory / 'results.json').write_text(json.dumps({
            'analyses': {'m1s': {str(seed): estimates}}}))
        inputs.append(str(directory))
    return inputs


def run(inputs, output):
    with patch.object(sys, 'argv', ['mon_explore_prev.py', '--inputs', *inputs,
                                    '--out', str(output), '--draws', '24']):
        with contextlib.redirect_stdout(io.StringIO()):
            m.main()
    return json.loads(output.read_text())


def check_rows(result, paired=False):
    rows = result['rows']
    expected = {(seed, fam, site) for seed in (0, 1)
                for fam in (*m.FAMS, 'family_mean') for site in m.D_SITES}
    assert len(rows) == len(expected)
    assert {(r['seed'], r['family'], r['d_site']) for r in rows} == expected
    assert result['l_site'] == m.L_SITE and result['draws'] == 24
    assert result['reproduction_max_abs_difference'] < 1e-14
    for row in rows:
        assert np.isclose(row['difference'], row['L32'] - row['D'])
        assert row['undefined_draws'] == 0
        assert row['interval'][0] <= row['difference'] <= row['interval'][1]
        if paired:
            assert row['difference'] == 0 and row['interval'] == [0., 0.]
        else:
            assert row['L32'] == 1 and row['difference'] >= .5
        if row['family'] == 'family_mean':
            families = [r for r in rows if r['seed'] == row['seed']
                        and r['d_site'] == row['d_site'] and r['family'] in m.FAMS]
            for key in ('L32', 'D', 'difference'):
                assert np.isclose(row[key], np.mean([r[key] for r in families]))


def main():
    y = np.array([0, 1, 0, 1, 0, 1, 1])
    for s in (np.array([0., 2., 2., 1., 3., 3., 2.]), np.ones(len(y))):
        for order in (np.arange(len(y)), np.arange(len(y))[::-1]):
            assert np.isclose(m.auroc(s[order], y[order]), pairwise(s, y))
            assert np.isclose(m.auroc(-s[order], y[order]), pairwise(-s, y))
            assert np.isclose(m.auroc(s[order], 1 - y[order]), pairwise(s, 1 - y))
    assert np.isnan(m.auroc(np.ones(3), np.zeros(3)))
    assert np.isnan(m.auroc(np.ones(3), np.ones(3)))

    s = np.array([0., 2., 1., 3., 2., 1., 1., 0., 2., 2.])
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    folds = np.array([0] * 4 + [1] * 6)
    expected = np.mean([pairwise(s[folds == f], y[folds == f]) for f in (0, 1)])
    assert np.isclose(m.fold_mean(s, y, folds), expected)
    w = np.array([2, 1, 3, 1, 0, 2, 1, 3, 1, 2])
    repeated = np.repeat(np.arange(len(y)), w)
    assert np.isclose(m.fold_mean(s, y, folds, w),
                      m.fold_mean(s[repeated], y[repeated], folds[repeated]))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs = make_inputs(root / 'advantage')
        first = run(inputs, root / 'first.json')
        check_rows(first)
        assert first == run(inputs, root / 'second.json')
        assert (root / 'first.json').read_bytes() == (root / 'second.json').read_bytes()
        check_rows(run(make_inputs(root / 'paired', paired=True), root / 'paired.json'), paired=True)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
