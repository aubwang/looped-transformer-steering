"""Synthetic checks for E1; no data, network, or GPU required."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mon_learning_curve as lc
import numpy as np
from mon_probe import nested_cv, make_folds


def main():
    rng = np.random.default_rng(492)
    n, d = 800, 80
    y = np.tile([0, 1], n//2)
    X = rng.normal(size=(n,d))
    X[:,0] += 1.8*y
    U = rng.normal(size=(n,4))
    U[:,0] += .4*y
    folds = make_folds(y,5,0)
    tr = np.flatnonzero(folds != 0)
    for draw in range(3):
        prior = set()
        for phi in (.125,.25,.5,.75,1.):
            sub = lc.training_subset(tr,y,phi,0,'add3','L32',0,draw)
            assert prior <= set(sub) <= set(tr)
            assert all(np.sum(y[sub] == c) == int(np.ceil(phi*np.sum(y[tr] == c))) for c in (0,1))
            prior = set(sub)
    for xx, uu, lams in ((X,None,lc.LAMS), (X,U,lc.LAMS), (np.zeros((n,1)),U,[1.])):
        expected = nested_cv({'fixed':xx},y,U=uu,lams=lams,seeds=(0,),folds=[folds])[0]
        actual = lc.reduced_cv(xx,y,folds,1.,0,U=uu,lams=lams)
        np.testing.assert_array_equal(actual['scores'],expected['scores'])
        assert actual['auroc'] == expected['mean_auroc']
        assert actual['chosen'] == [lam for _,lam in expected['chosen']]
    small = [lc.reduced_cv(X,y,folds,.125,draw) for draw in range(3)]
    big = lc.reduced_cv(X,y,folds,1.,0)
    assert big['auroc'] > np.mean([r['auroc'] for r in small]) + .02
    print('Planted signal AUROC:',np.mean([r['auroc'] for r in small]),'->',big['auroc'])
    row, bs = lc.bootstrap_gain([(1,big)]+[(-1/3,r) for r in small],100,0)
    assert row['undefined_bootstrap_draws'] == 0 and row['interval'][0] is not None
    assert lc.interval(np.array([0.,np.nan]))['interval'] == [None,None]
    # A class lost in even one bootstrap draw must invalidate the whole interval.
    tiny = dict(scores=np.arange(4.), y=np.array([0,1,0,1]), folds=np.array([0,0,1,1]))
    row,_ = lc.bootstrap_gain([(1,tiny)],100,0)
    assert row['undefined_bootstrap_draws'] > 0 and row['interval'] == [None,None]
    # Insufficient subset class counts are recorded, never patched or dropped.
    yy = np.tile([0,1],10); ff = make_folds(yy,5,0)
    failed = lc.reduced_cv(np.arange(20.)[:,None],yy,ff,.125,0)
    assert failed['inner_class_failures'] == 5 and np.isnan(failed['scores']).all()
    curves, derived = lc.summarize({('add3',m,k,phi):([big] if phi==1 else small)
                                  for m in lc.MODELS for k in lc.FITS for phi in (.125,1.)},
                                 ['add3'],[.125,1.],10,0)
    assert derived['add3']['L32']['last_doubling']['probe']['interval'] == [None,None]
    # Full-size summaries and bootstrap means use every family with equal weight.
    records = {(fam,m,k,phi):([big] if phi==1 else small)
               for fam in ('add3','add4') for m in lc.MODELS for k in lc.FITS for phi in (.5,1.)}
    curves, derived = lc.summarize(records,['add3','add4'],[.5,1.],20,0)
    gain = big['auroc'] - np.mean([r['auroc'] for r in small])
    assert abs(derived['family_mean']['L32']['last_doubling']['probe']['estimate']-gain) < 1e-12
    assert derived['family_mean']['L32']['last_doubling']['probe']['interval'][0] is not None
    assert abs(derived['family_mean']['L32']['last_doubling']['added']['estimate']) < 1e-12
    # Averaging AUROCs over draws is explicit; raw scores need not share scales.
    shifted = dict(big, scores=big['scores'] + 100.)
    row,_ = lc.bootstrap_gain([(0.5,big),(0.5,shifted)],20,0)
    assert abs(row['estimate']-big['auroc']) < 1e-12
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
