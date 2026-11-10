"""Exploratory follow-up E1 of docs/llever_design.md (section 8): how much of the endpoint response is predicted by one
pass through the recurrent block, and by t passes, against the full operator sum. Reading fixed before the run.
Usage: python -I analyze_llever_partial.py <pred_partial_run> <pred_run> <eval_run>

Predictors for eval problem j, lever push L_k, class m, at dose rho:  rho * m_ref * G_j[m] . (A_t L_k)_j,
A_t = J + ... + J^t  (t = 1 is ONE-STEP; t = T is FULL). Scores per problem on the flattened 10 x 10 matrix: Pearson
correlation, least-squares slope, normalised error ||pred - obs|| / ||obs||.
  ENDPOINT:  every A_t against the observed response at the end of the forced run (t = T).
  MATCHED:   A_t against the observed response after the same number t of forced loops.
Intervals: bootstrap over the 40 problems (5,000 resamples, seed 0) of the mean paired difference to FULL.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def ci(v):
    v = np.asarray(v); rng = np.random.default_rng(0)
    return np.quantile(v[rng.integers(len(v), size=(5000, len(v)))].mean(1), [.025, .975]).tolist()


def score(pred, obs):
    p, o = pred.reshape(len(pred), -1), obs.reshape(len(obs), -1)
    cor = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(p, o)])
    slope = np.array([np.polyfit(a, b, 1)[0] for a, b in zip(p, o)])
    err = np.linalg.norm(p - o, axis=1) / np.linalg.norm(o, axis=1)
    return cor, slope, err


def analyze(zq, zp, ze, push):
    ev = np.where(zq['split'] == 'eval')[0]
    assert np.array_equal(zq['pid'][ev], ze['pid']) and np.array_equal(zq['pid'], zp['pid'])
    G = zq['G'][ev]; m = float(push['m_ref']); T = int(ze['times'][-1])
    ts = [int(t) for t in zq['partial_t']]
    A = {t: zq['AP_partial'][ev][:, i] for i, t in enumerate(ts)}; A[T] = zq['AP'][ev]
    rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
    out = dict(T=T, partial_t=ts, reproduces_pred_run=dict(AP=rel(zq['AP'], zp['AP']), G=rel(zq['G'], zp['G'])), doses={})
    for k in ('AP', 'G'):  # section 8: the full sum must reproduce the first prediction run
        assert np.isfinite(zq[k]).all() and np.max(np.abs(zq[k] - zp[k])) <= 1e-6 * np.max(np.abs(zp[k])), 'does not reproduce ' + k
    rp, rr = ze['row_push'], ze['row_rho']; times = [int(t) for t in ze['times']]
    for rho in (.01, .03, .09):
        rows = [int(np.flatnonzero((rp == k) & np.isclose(rr, rho))[0]) for k in range(10)]
        dz = ze['Z'][:, rows] - ze['Z'][:, 0:1]                       # [problem, k, time, class]
        end = dz[:, :, -1]; res = {}
        full = score(rho * m * np.einsum('jmd,jkd->jkm', G, A[T]), end)
        for t in sorted(A):
            pred = rho * m * np.einsum('jmd,jkd->jkm', G, A[t]); c, s, e = score(pred, end)
            r = dict(endpoint=dict(median_corr=float(np.median(c)), median_slope=float(np.median(s)), median_norm_error=float(np.median(e)),
                                   error_minus_FULL_ci=ci(e - full[2]), FULL_corr_minus_ci=ci(full[0] - c)))
            if t in times:
                c2, s2, e2 = score(pred, dz[:, :, times.index(t)])
                r['matched_time'] = dict(median_corr=float(np.median(c2)), median_slope=float(np.median(s2)), median_norm_error=float(np.median(e2)))
            res[str(t)] = r
        out['doses'][str(rho)] = res
    return out


def main(partial, pred, evaluation):
    roots = [Path(p) for p in (partial, pred, evaluation)]
    metas = [json.loads((p / 'meta.json').read_text()) for p in roots]
    assert all(mm['status'] == 'complete' for mm in metas) and len({mm['cfg']['T'] for mm in metas}) == 1
    raw = (roots[2] / 'pushes.npz').read_bytes(); sha = hashlib.sha256(raw).hexdigest()
    assert all(mm['push_file_sha256'] == sha for mm in metas), 'runs used different push files'
    assert [mm['mode'] for mm in metas] == ['pred', 'pred', 'eval']
    assert len({mm['model_revision'] for mm in metas}) == 1 and len({mm['cfg']['K0'] for mm in metas}) == 1
    zq, zp, ze = (dict(np.load(p / 'llever.npz', allow_pickle=False)) for p in roots)
    push = dict(np.load(roots[2] / 'pushes.npz', allow_pickle=False))
    assert int(ze['times'][-1]) == int(push['T']) == metas[0]['cfg']['T']
    out = analyze(zq, zp, ze, push)
    (roots[0] / 'partial_summary.json').write_text(json.dumps(out, indent=1, allow_nan=False) + '\n'); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
