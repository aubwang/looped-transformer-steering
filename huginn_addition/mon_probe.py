"""CPU monitors implementing docs/oversight_monitor.md §§5–7 and docs/mon_spec.md B.

Labels are 1 for wrong answers. All fitted transforms belong to their training
part. Spline knots must likewise be computed on that training part by callers.
Results are lists of seed dictionaries; fit_path returns models in input order.
"""
from dataclasses import dataclass
import numpy as np
from scipy import linalg, optimize, special, stats

LAMS = 10.0 ** np.arange(-6, 3)


def _labels(y):
    y = np.asarray(y)
    if y.ndim != 1 or not np.all((y == 0) | (y == 1)):
        raise ValueError('y must be a vector of binary labels (1 = wrong)')
    return y.astype(float)


def auroc(score, y):
    """Mann–Whitney AUROC with half credit for ties; undefined -> NaN."""
    y = _labels(y)
    s = np.asarray(score, dtype=float)
    if s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError('scores must be finite and match y')
    n1 = y.sum()
    n0 = len(y) - n1
    if not n0 or not n1:
        return float('nan')
    return float((stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def selective_error(score, y, reject_frac=0.2):
    """Reject floor(n*fraction) items; ties resolved by original item order.

    Threshold is the lowest rejected score (inf if none). For tied scores the
    exact coverage requires the stable rank rule, not a scalar threshold alone.
    """
    y = _labels(y)
    s = np.asarray(score, dtype=float)
    if s.shape != y.shape or not np.isfinite(s).all() or not 0 <= reject_frac < 1:
        raise ValueError('invalid scores or reject_frac')
    nr = int(len(y) * reject_frac)
    order = np.argsort(-s, kind='stable')
    return float(y[order[nr:]].mean()), float(s[order[nr - 1]]) if nr else np.inf


def _standardize(X):
    X = np.array(X, dtype=np.float64, copy=True)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('features must be a finite matrix')
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    keep = sd > 0
    scale = np.maximum(sd[keep], 1e-6 * sd.max(initial=0))
    return (X[:, keep] - mean[keep]) / scale, mean, scale, keep


@dataclass
class LinearModel:
    w_raw: np.ndarray
    b_raw: float
    u_raw: np.ndarray
    converged: bool = True
    message: str = ''
    n_iter: int = 0

    @property
    def norm_w_raw(self):
        return float(linalg.norm(self.w_raw))

    def score(self, X, U=None):
        X = np.asarray(X, dtype=np.float64)
        s = X @ self.w_raw + self.b_raw
        if self.u_raw.size:
            if U is None:
                raise ValueError('this model requires U')
            s = s + np.asarray(U, dtype=np.float64) @ self.u_raw
        return s


U_RIDGE = 1e-6     # frozen ridge on the whitened unpenalized block: a numerical safeguard, the same in every model
U_RANK_TOL = 1e-8  # frozen: directions of the standardized block below this fraction of the largest are dropped
GRAD_TOL = 1e-6    # frozen convergence criterion: largest absolute gradient component of the mean objective


def _whiten(U):
    """Standardize, then replace the columns by an orthogonal basis with unit mean square (training part only).

    Returns (V, back): V = standardized(U) @ back. An invertible change of basis leaves an unpenalized fit
    unchanged; it only removes the ill-conditioning of nearly collinear columns such as spline terms.
    """
    VU, um, us, uk = _standardize(U)
    if not VU.shape[1]:
        return VU, np.zeros((0, 0)), um, us, uk
    _, sing, vt = linalg.svd(VU, full_matrices=False, check_finite=False)
    keep = sing > U_RANK_TOL * sing[0]
    back = vt[keep].T * (np.sqrt(len(VU)) / sing[keep])
    return VU @ back, back, um, us, uk


def fit_path(X, y, lams, U=None):
    """One thin SVD (only if d>n), then descending-lambda warm starts.

    Objective: mean log loss + lam/2 ||w||^2 (standardized X) + U_RIDGE/2 ||u||^2 (whitened U); free intercept.
    L2 weights lie in the row space even with an extra block, and SVD directions are orthonormal, so their
    coefficient penalty is exactly L2. Solver: L-BFGS, then, if the gradient criterion is not met, a
    trust-region Newton method with exact Hessian-vector products started from the L-BFGS point.
    converged means max |gradient| <= GRAD_TOL, checked here and not taken from the solver's own flag.
    """
    y = _labels(y)
    if not 0 < y.sum() < len(y):
        raise ValueError('fitting requires both classes')
    Z, xm, xs, xk = _standardize(X)
    if len(Z) != len(y):
        raise ValueError('X and y lengths differ')
    if U is None:
        U = np.empty((len(y), 0))
    VU, back, um, us, uk = _whiten(U)
    if len(VU) != len(y):
        raise ValueError('U and y lengths differ')
    basis = None
    if Z.shape[1] > len(y):
        q, singular, vt = linalg.svd(Z, full_matrices=False, check_finite=False)
        # Zero singular directions have optimal penalized coefficient zero.
        rank = singular > singular[0] * np.finfo(float).eps * max(Z.shape)
        basis = vt[rank]
        Z = q[:, rank] * singular[rank]
    p = Z.shape[1]
    A = np.ascontiguousarray(np.column_stack((Z, VU, np.ones(len(y)))))
    theta = np.zeros(A.shape[1])
    theta[-1] = special.logit(y.mean())
    lams = np.asarray(list(lams), dtype=float)
    if not np.isfinite(lams).all() or np.any(lams <= 0):
        raise ValueError('lambdas must be finite and positive')
    models = [None] * len(lams)
    for j in np.argsort(-lams, kind='stable'):
        penalties = np.r_[np.full(p, lams[j]), np.full(VU.shape[1], U_RIDGE), 0.]
        def objective(t):
            z = A @ t
            # logaddexp form avoids overflow and cancellation for either label.
            loss = np.logaddexp(0, (1 - 2*y) * z).mean()
            grad = A.T @ (special.expit(z) - y) / len(y)
            return loss + np.dot(t * penalties, t) / 2, grad + penalties * t
        def hessp(t, v):
            pr = special.expit(A @ t)
            return A.T @ (pr * (1 - pr) * (A @ v)) / len(y) + penalties * v
        opt = optimize.minimize(objective, theta, jac=True, method='L-BFGS-B',
                                options={'ftol': 1e-15, 'gtol': GRAD_TOL / 2,
                                         'maxiter': 1000, 'maxls': 40, 'maxcor': 20})
        theta = opt.x; n_iter = int(opt.nit); how = 'lbfgs'
        if np.abs(objective(theta)[1]).max() > GRAD_TOL:
            opt = optimize.minimize(objective, theta, jac=True, hessp=hessp, method='trust-krylov',
                                    options={'gtol': GRAD_TOL / 2, 'maxiter': 200})
            theta = opt.x; n_iter += int(opt.nit); how = 'lbfgs+newton'
        g = float(np.abs(objective(theta)[1]).max())
        w = theta[:p] if basis is None else basis.T @ theta[:p]
        wr = np.zeros(len(xm)); wr[xk] = w / xs
        ur = np.zeros(len(um)); ur[uk] = (back @ theta[p:-1]) / us
        models[j] = LinearModel(wr, float(theta[-1] - xm @ wr - um @ ur), ur, g <= GRAD_TOL,
                                f'{how}: max |gradient| {g:.1e}', n_iter)
    return models


def fit_logistic(X, y, lam, U=None):
    """Fit mean log loss + lam/2 ||w_standardized||²."""
    return fit_path(X, y, [lam], U)[0]


def fit_meandiff(X, y):
    """Difference of standardized class means; positive points toward errors."""
    y = _labels(y)
    Z, mean, scale, keep = _standardize(X)
    if not 0 < y.sum() < len(y):
        raise ValueError('fitting requires both classes')
    w = Z[y == 1].mean(0) - Z[y == 0].mean(0)
    raw = np.zeros(len(mean)); raw[keep] = w / scale
    return LinearModel(raw, float(-mean @ raw), np.empty(0))


def _pca_path(X, y, lams, k=128, U=None):
    X = np.asarray(X, dtype=float)
    mean = X.mean(0)
    q, s, vt = linalg.svd(X - mean, full_matrices=False, check_finite=False)
    k = min(k, len(s))
    projected = q[:, :k] * s[:k]
    models = fit_path(projected, y, lams, U)
    for m in models:
        m.w_raw = vt[:k].T @ m.w_raw
        m.b_raw -= float(mean @ m.w_raw)
    return models


def fit_pca_logistic(X, y, lam, k=128):
    """Training-centered, unwhitened PCA followed by standardized logistic."""
    return _pca_path(X, y, [lam], k)[0]


def spline_features(u, knots):
    """Natural cubic regression spline [u, d0-d2, d1-d2] for four knots.

    Knots are the training quantiles (.05,.35,.65,.95). Repeated knots are
    coalesced, reducing dimension for discrete/constant predictors. No intercept.
    """
    u = np.asarray(u, dtype=float)
    t = np.unique(np.asarray(knots, dtype=float))
    if u.ndim != 1 or t.ndim != 1 or not len(t) or not np.isfinite(t).all():
        raise ValueError('supply a vector and finite training knots')
    cols = [u]
    if len(t) >= 3:
        def d(j):
            return (np.maximum(u-t[j], 0)**3 - np.maximum(u-t[-1], 0)**3) / (t[-1]-t[j])
        ref = d(len(t)-2)
        cols.extend(d(j)-ref for j in range(len(t)-2))
    return np.column_stack(cols)


def make_folds(y, n_folds, seed):
    """Return item-wise fold numbers; each class is shuffled independently."""
    y = _labels(y)
    if n_folds < 2 or min((y == 0).sum(), (y == 1).sum()) < n_folds:
        raise ValueError('each class must have at least n_folds members; n_folds >= 2')
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def check_folds(folds, y, n_folds):
    """Validate a supplied fold vector: numbers 0..n_folds-1, both classes present in every fold."""
    folds = np.asarray(folds); y = _labels(y)
    if folds.shape != y.shape or folds.dtype.kind not in 'iu' or set(np.unique(folds)) != set(range(n_folds)):
        raise ValueError('folds must assign every item to one of 0..n_folds-1, all used')
    if any(len(np.unique(y[folds == f])) < 2 for f in range(n_folds)):
        raise ValueError('every fold needs both classes')
    return folds.astype(int)


def make_joint_folds(strata, n_folds, seed):
    """One fold assignment balanced over joint categories (rows of strata, e.g. two label vectors).

    Items are ordered by category, at random within a category, and dealt out in turn starting from a
    random fold: every category, and every union of adjacent categories, is spread as evenly as possible.
    """
    strata = np.asarray(strata)
    if strata.ndim == 1:
        strata = strata[:, None]
    rng = np.random.default_rng(seed)
    _, codes = np.unique(strata, axis=0, return_inverse=True)
    order = np.lexsort((rng.random(len(codes)), codes.ravel()))
    folds = np.empty(len(codes), dtype=int)
    folds[order] = (np.arange(len(codes)) + rng.integers(n_folds)) % n_folds
    return folds


def nested_cv(sites, y, U=None, lams=LAMS, n_outer=5, n_inner=4,
              seeds=(0, 1, 2), fitter='logistic', folds=None):
    """Joint site/lambda selection with all transforms fitted inside each fold.

    folds, if given, is a list (one entry per seed) of outer fold numbers 0..n_outer-1 to use instead of
    make_folds(y, n_outer, seed): fits on different label vectors can then share one assignment. Every
    fold and every training part must contain both classes.

    U is a raw feature matrix, standardized within each fitting part. If using
    splines, pass a callable U(train_indices, score_indices) returning the two
    feature matrices; it must fit its knots only on train_indices.
    """
    y = _labels(y)
    sites = {name: np.asarray(X, dtype=float) for name, X in sites.items()}
    if not sites or any(len(X) != len(y) for X in sites.values()):
        raise ValueError('sites must be nonempty and match y')
    lams = sorted(set(float(x) for x in lams), reverse=True)
    if fitter not in ('logistic', 'pca_logistic', 'meandiff'):
        raise ValueError('unknown fitter')
    if fitter == 'meandiff':
        lams = [lams[0]]
        if U is not None:
            raise ValueError('mean difference does not support U')
    def extras(tr, te):
        if callable(U):
            return U(tr, te)
        if U is None:
            return None, None
        return np.asarray(U)[tr], np.asarray(U)[te]
    def path(X, labels, u):
        if fitter == 'meandiff':
            return [fit_meandiff(X, labels)]
        if fitter == 'pca_logistic':
            return _pca_path(X, labels, lams, U=u)
        return fit_path(X, labels, lams, u)
    results = []
    given = folds
    if given is not None and len(given) != len(tuple(seeds)):
        raise ValueError('supply one fold vector per seed')
    for k, seed in enumerate(seeds):
        folds = make_folds(y, n_outer, seed) if given is None else check_folds(given[k], y, n_outer)
        scores = np.empty(len(y)); models = []; chosen = []; fold_auc = []; failures = []
        for f in range(n_outer):
            tr = np.flatnonzero(folds != f); te = np.flatnonzero(folds == f)
            inner = make_folds(y[tr], n_inner, np.random.SeedSequence([int(seed), f, 917]))
            merit = np.zeros((len(sites), len(lams)))
            for g in range(n_inner):
                it = tr[inner != g]; iv = tr[inner == g]
                ut, uv = extras(it, iv)
                for j, X in enumerate(sites.values()):
                    ms = path(X[it], y[it], ut)
                    merit[j] += [auroc(m.score(X[iv], uv), y[iv]) for m in ms]
                    failures.extend(m.message for m in ms if not getattr(m, 'converged', True))
            j, ell = np.unravel_index(np.argmax(merit), merit.shape)
            name = list(sites)[j]; lam = lams[ell]
            ut, uv = extras(tr, te)
            if fitter == 'logistic':
                model = fit_logistic(sites[name][tr], y[tr], lam, ut)
            elif fitter == 'pca_logistic':
                model = _pca_path(sites[name][tr], y[tr], [lam], U=ut)[0]
            else:
                model = fit_meandiff(sites[name][tr], y[tr])
            scores[te] = model.score(sites[name][te], uv)
            models.append(model); chosen.append((name, lam))
            fold_auc.append(auroc(scores[te], y[te]))
        results.append(dict(seed=seed, scores=scores, folds=folds, chosen=chosen,
                            fold_auroc=np.asarray(fold_auc), mean_auroc=float(np.mean(fold_auc)),
                            models=models, inner_nonconverged=len(failures),
                            inner_solver_messages=sorted(set(failures))))
    return results


def apply_models(result, sites_other):
    """Frozen per-item transfer; result is one seed dict (or a list thereof).

    For augmented models supply sites_other['U'] as raw extra features, or a
    callable taking the outer fold number and item indices for frozen transforms.
    """
    if isinstance(result, list):
        return [apply_models(r, sites_other) for r in result]
    scores = np.empty(len(result['folds']))
    for f, (model, (site, _)) in enumerate(zip(result['models'], result['chosen'])):
        ix = np.flatnonzero(result['folds'] == f)
        u = sites_other.get('U')
        u = u(f, ix) if callable(u) else (None if u is None else np.asarray(u)[ix])
        scores[ix] = model.score(np.asarray(sites_other[site])[ix], u)
    return scores


def paired_bootstrap(stat_fn, groups, n_boot=2000, seed=0):
    """Paired within-group percentile bootstrap; stat_fn receives item indices.

    Groups may be a vector or a matrix of group keys (e.g. family, outer fold).
    This conditions on fitted predictions; it does not include refit uncertainty.
    """
    groups = np.asarray(groups)
    if groups.ndim == 1:
        _, codes = np.unique(groups, return_inverse=True)
    else:
        _, codes = np.unique(groups, axis=0, return_inverse=True)
    blocks = [np.flatnonzero(codes == c) for c in np.unique(codes)]
    rng = np.random.default_rng(seed)
    estimate = np.asarray(stat_fn(np.arange(len(groups))))
    draws = np.asarray([stat_fn(np.concatenate([rng.choice(b, len(b)) for b in blocks]))
                        for _ in range(n_boot)])
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return estimate, low, high, draws
