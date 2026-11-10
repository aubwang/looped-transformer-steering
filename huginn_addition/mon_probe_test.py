"""Local checks for docs/mon_spec.md B; run with the analysis Python -I."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import time
import numpy as np
from scipy import optimize, special
from sklearn.metrics import roc_auc_score
from mon_probe import (fit_logistic, fit_path, fit_meandiff, fit_pca_logistic,
                       nested_cv, auroc, apply_models, make_folds, make_joint_folds,
                       paired_bootstrap, spline_features, selective_error, U_RIDGE, GRAD_TOL)


def main():
    rng = np.random.default_rng(401)
    for n, d in [(90, 7), (35, 65)]:
        X = rng.normal(size=(n, d)); X[:, -1] = 3
        X = X.astype(np.float16); saved = X.copy()
        U = rng.normal(size=(n, 2)); U[:, 1] = 5
        y = (rng.random(n) < special.expit(X[:, 0] + U[:, 0])).astype(int)
        m = fit_logistic(X, y, .07, U)
        xf = X.astype(float); mu = xf.mean(0); sd = xf.std(0)
        keep = sd > 0; Z = (xf[:, keep] - mu[keep]) / sd[keep]
        uz = (U[:, :1] - U[:, :1].mean(0)) / U[:, :1].std(0)
        A = np.column_stack([Z, uz, np.ones(n)])
        pen = np.r_[np.full(Z.shape[1], .07), U_RIDGE, 0]
        def obj(t):
            z = A @ t
            return (np.logaddexp(0, (1-2*y)*z).mean() + .5*np.dot(pen*t,t),
                    A.T@(special.expit(z)-y)/n + pen*t)
        o = optimize.minimize(obj, np.zeros(A.shape[1]), jac=True, method='BFGS',
                              options={'gtol': 1e-12, 'maxiter': 2000})
        # The solver stops at max |gradient| <= GRAD_TOL = 1e-6; with the small samples used here the
        # curvature is weak, so scores agree with the exact optimum to about 1e-4, not to 1e-6.
        assert m.converged, m.message
        np.testing.assert_allclose(m.score(X,U), A@o.x, atol=1e-4, rtol=0)
        np.testing.assert_allclose(m.score(X,U), xf@m.w_raw+U@m.u_raw+m.b_raw, atol=1e-10)
        assert m.w_raw[-1] == 0 and m.u_raw[-1] == 0
        assert np.isclose(m.norm_w_raw, np.linalg.norm(m.w_raw), rtol=1e-15)
        np.testing.assert_allclose(m.score(X,U), fit_logistic(X,y,.07,U*37).score(X,U*37),atol=1e-4)
        assert np.array_equal(X,saved)
        path = fit_path(X,y,[.07, 1, .001],U)
        np.testing.assert_allclose(path[0].score(X,U),m.score(X,U),atol=1e-4)
    # Penalized separation and unpenalized separation must terminate finitely.
    x = np.linspace(-2,2,80)[:,None]; y = (x[:,0]>0).astype(int)
    for m in [fit_logistic(x,y,1e-6),fit_logistic(np.zeros_like(x),y,1e-6,U=x)]:
        assert np.isfinite(m.b_raw) and m.converged, m.message
    # Nearly collinear extra block (spline terms of correlated predictors): every fit on the path must meet
    # the gradient criterion, and an invertible change of basis of the block must not change the scores.
    n = 300; base = rng.normal(size=n); c = np.column_stack([base + .005*rng.normal(size=n) for _ in range(4)])
    y = (rng.random(n) < special.expit(1.5*base)).astype(int)
    S = np.column_stack([spline_features(c[:, j], np.quantile(c[:, j], [.05,.35,.65,.95])) for j in range(4)])
    assert np.linalg.cond((S-S.mean(0))/S.std(0)) > 1e3
    X = rng.normal(size=(n, 400)); mix = rng.normal(size=(S.shape[1], S.shape[1]))
    for xs in (X, np.zeros((n, 1))):
        one = fit_path(xs, y, [1e-6, 1e-2, 1.], S); two = fit_path(xs, y, [1e-6, 1e-2, 1.], S @ mix)
        assert all(m.converged for m in one+two), [m.message for m in one+two]
        for a_, b_ in zip(one, two):
            # equal up to solver tolerance; at lambda = 1e-6 with d > n the fit nearly separates and is flat
            np.testing.assert_allclose(a_.score(xs, S), b_.score(xs, S @ mix), atol=2e-2, rtol=1e-3)
    # An exactly duplicated column is dropped by the rank rule, not fitted.
    dup = fit_logistic(np.zeros((n, 1)), y, 1., np.column_stack((S, S[:, 0])))
    np.testing.assert_allclose(dup.score(np.zeros((n, 1)), np.column_stack((S, S[:, 0]))),
                               fit_logistic(np.zeros((n, 1)), y, 1., S).score(np.zeros((n, 1)), S), atol=2e-3)
    # Shared folds: balanced over the joint categories of two label vectors, and honoured by nested_cv.
    y1 = rng.integers(0, 2, 503); y2 = np.where(rng.random(503) < .7, y1, rng.integers(0, 2, 503))
    joint = make_joint_folds(np.column_stack((y1, y2)), 5, 3)
    for a_ in (0, 1):
        for b_ in (0, 1):
            counts = np.bincount(joint[(y1 == a_) & (y2 == b_)], minlength=5)
            assert counts.max() - counts.min() <= 1, counts
    assert np.bincount(joint).max() - np.bincount(joint).min() <= 1
    Xj = rng.normal(size=(503, 6))
    r1 = nested_cv({'a': Xj}, y1, lams=[.01, 1], n_outer=5, n_inner=2, seeds=(0,), folds=[joint])[0]
    r2 = nested_cv({'a': Xj}, y2, lams=[.01, 1], n_outer=5, n_inner=2, seeds=(0,), folds=[joint])[0]
    assert np.array_equal(r1['folds'], joint) and np.array_equal(r2['folds'], joint)
    for _ in range(10):
        y = rng.integers(0,2,100); s = rng.integers(0,5,100)
        assert abs(auroc(s,y)-roc_auc_score(y,s)) < 1e-14
    assert auroc([1,1],[0,1]) == .5
    assert selective_error(np.arange(10), np.r_[np.zeros(8),1,1])[0] == 0
    try:
        make_folds([0,0,1], 2, 0)
        raise AssertionError('class count not checked')
    except ValueError:
        pass
    knots = np.array([-2,-.5,.5,2.])
    xx = np.array([-5,-4,-3,3,4,5.])
    B = spline_features(xx,knots)
    np.testing.assert_allclose(B[0]-2*B[1]+B[2],0,atol=1e-12)
    np.testing.assert_allclose(B[3]-2*B[4]+B[5],0,atol=1e-12)
    # No outer test feature may affect its model's transform, selection or fit.
    X = rng.normal(size=(100,9)); y = np.tile([0,1],50)
    a = nested_cv({'a':X},y,lams=[.01,1],n_outer=2,n_inner=2,seeds=(0,))[0]
    altered = X.copy(); altered[a['folds']==0] += 1000
    b = nested_cv({'a':altered},y,lams=[.01,1],n_outer=2,n_inner=2,seeds=(0,))[0]
    np.testing.assert_array_equal(a['models'][0].w_raw,b['models'][0].w_raw)
    assert a['chosen'][0] == b['chosen'][0]
    np.testing.assert_allclose(apply_models(a,{'a':X}),a['scores'])
    fit_pca_logistic(X,y,.1,k=5).score(X)
    boot = paired_bootstrap(lambda idx: np.array([idx.mean(),2*idx.mean()]),
                            np.repeat(np.arange(4),25),n_boot=100)
    np.testing.assert_allclose(boot[3][:,1],2*boot[3][:,0])
    # Actual outer-test selection leak: choose among null sites after evaluating
    # each site's OOF predictions, vs selecting sites only inside training folds.
    honest=[]; leaky=[]; permuted=[]
    for rep in range(32):
        y = rng.integers(0,2,160)
        sites = {str(j):rng.normal(size=(160,8)) for j in range(20)}
        r = nested_cv(sites,y,n_outer=4,n_inner=3,seeds=(rep,),fitter='meandiff')[0]
        honest.append(r['mean_auroc'])
        # Evaluate each site's honest OOF predictions, then misuse test labels
        # to choose the winning site. This is the sole deliberate leak.
        vals=[]
        for X in sites.values():
            fold_auc=[]
            for f in range(4):
                tr=r['folds']!=f; te=~tr
                fold_auc.append(auroc(fit_meandiff(X[tr],y[tr]).score(X[te]),y[te]))
            vals.append(np.mean(fold_auc))
        leaky.append(max(vals))
        if rep < 12:
            permuted.append(nested_cv(sites,rng.permutation(y),n_outer=4,n_inner=3,
                                      seeds=(rep,),fitter='meandiff')[0]['mean_auroc'])
    assert abs(np.mean(honest)-.5)<.035
    assert abs(np.mean(permuted)-.5)<.05
    assert np.mean(np.array(leaky)-honest) > .07
    print(f'Null honest={np.mean(honest):.4f}; test-selected leak={np.mean(leaky):.4f}; permutation={np.mean(permuted):.4f}',flush=True)
    # Exercise the full penalized pipeline on null and permuted planted labels,
    # in addition to the many-site mean-difference leak diagnostic above.
    X0 = rng.normal(size=(360,24)); y0 = rng.integers(0,2,360)
    null_results = nested_cv({'null':X0}, y0, n_outer=3, n_inner=2, seeds=(0,1,2))
    yp = rng.permutation((X0[:,0] > 0).astype(int))
    perm_results = nested_cv({'permuted':X0}, yp, n_outer=3, n_inner=2, seeds=(0,1,2))
    na = np.mean([r['mean_auroc'] for r in null_results])
    pa = np.mean([r['mean_auroc'] for r in perm_results])
    assert abs(na-.5) < .08 and abs(pa-.5) < .08, (na,pa)
    print(f'Logistic null={na:.4f}; logistic permutation={pa:.4f}',flush=True)
    # Low effective rank Gaussian data with analytically known optimal AUROC.
    n,d=1500,2048
    y=rng.integers(0,2,n)
    delta=np.sqrt(2)*special.ndtri(.8)*np.sqrt(1+.04/d)
    latent=rng.normal(size=n)+delta*y
    X=latent[:,None]+.2*rng.normal(size=(n,d))
    null=rng.normal(size=(n,d))
    start=time.perf_counter()
    r=nested_cv({'signal':X,'null':null},y,seeds=(0,))[0]
    elapsed=time.perf_counter()-start
    assert abs(r['mean_auroc']-.8)<.045, r['mean_auroc']
    assert sum(s=='signal' for s,_ in r['chosen'])>=4
    print(f'Planted optimal=0.8000; nested={r["mean_auroc"]:.4f}; sites={r["chosen"]}',flush=True)
    print(f'TIMING 2 sites n=1500 d=2048, 9 lambdas, 5x4, 1 seed: {elapsed:.2f} seconds',flush=True)
    print('Outer convergence:',[(m.converged,m.n_iter) for m in r['models']],flush=True)
    print('ALL CHECKS PASSED',flush=True)


if __name__=='__main__':
    main()
