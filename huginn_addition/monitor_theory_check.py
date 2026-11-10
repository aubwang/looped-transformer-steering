"""Numerical checks of every statement in docs/monitor_theory.typ (CPU, numpy only, a few seconds).

Each check draws random instances and asserts the stated inequality or identity; the statement
numbers match the note. Run: python monitor_theory_check.py
"""
import numpy as np

rng = np.random.default_rng(20261109)
TOL = 1e-9


def geo(q, h):
    """q + q^2 + ... + q^h."""
    return sum(q ** k for k in range(1, h + 1))


def orbit(step, x0, n):
    xs = [x0]
    for _ in range(n):
        xs.append(step(xs[-1]))
    return np.array(xs)


def linear(a, n):
    """Steps of X -> AX + c computed directly (delta_{t+1} = A delta_t), avoiding cancellation near the limit.

    Returns the positions X_0..X_n (up to the irrelevant constant X_0 = 0) and their consecutive differences.
    """
    deltas = [rng.normal(size=len(a))]
    for _ in range(n - 1):
        deltas.append(a @ deltas[-1])
    deltas = np.array(deltas)
    return np.vstack([np.zeros(len(a)), np.cumsum(deltas, axis=0)]), deltas


def sym(eigs):
    q, _ = np.linalg.qr(rng.normal(size=(len(eigs), len(eigs))))
    return q @ np.diag(eigs) @ q.T


def theorem1():
    """A-posteriori bound for a nonlinear contraction with known factor q."""
    worst = 0
    for _ in range(200):
        n = 6
        w = rng.normal(size=(n, n)); w *= rng.uniform(.3, .9) / np.linalg.norm(w, 2)
        q = np.linalg.norm(w, 2)                      # tanh is 1-Lipschitz, so x -> tanh(Wx + c) is a q-contraction
        c = rng.normal(size=n)
        xs = orbit(lambda x: np.tanh(w @ x + c), rng.normal(size=n) * 3, 60)
        d = np.linalg.norm(np.diff(xs, axis=0), axis=1)          # d[t-1] = d_t
        for r in (2, 4, 8):
            for h in (1, 5, 30):
                path = d[r:r + h].sum(); disp = np.linalg.norm(xs[r + h] - xs[r]); bound = d[r - 1] * geo(q, h)
                assert disp <= path * (1 + TOL) and path <= bound * (1 + TOL)
                worst = max(worst, path / bound)
            assert np.linalg.norm(xs[-1] - xs[r]) <= d[r - 1] * q / (1 - q) * (1 + TOL)
    return f"largest path/bound {worst:.3f} (must be <= 1)"


def theorem2():
    """Symmetric (and normal) linear maps: ratios rise, so the extrapolated forecast is a lower bound."""
    gaps = []
    for trial in range(300):
        n = 8
        positive = trial % 2 == 0
        eigs = rng.uniform(0 if positive else -1, 1, n) * .95
        a = sym(eigs)
        xs, deltas = linear(a, 45)
        d = np.linalg.norm(deltas, axis=1)
        ratio = d[1:] / d[:-1]
        assert np.all(np.diff(ratio[:30]) >= -1e-9 * ratio[:29]), "ratios must not decrease"   # (a)
        for r in (3, 5, 9):
            qhat = max(ratio[max(0, r - 9):r - 1])                                       # last eight ratios up to d_r/d_{r-1}
            assert abs(qhat - d[r - 1] / d[r - 2]) <= 1e-9 * qhat
            for h in (1, 6, 28):
                forecast = d[r - 1] * geo(qhat, h)
                path = d[r:r + h].sum(); disp = np.linalg.norm(xs[r + h] - xs[r])
                assert forecast <= path * (1 + 1e-7)                                     # (b)
                if positive:
                    assert forecast <= disp * (1 + 1e-7)                                 # (c)
                    gaps.append(disp / forecast)
    # a single mode gives equality
    xs, deltas = linear(.7 * np.eye(5), 20); d = np.linalg.norm(deltas, axis=1)
    assert abs(d[5:15].sum() - d[4] * geo(d[4] / d[3], 10)) <= 1e-9 * d[4]
    # normal but not symmetric (rotation-scalings): (a), (b) hold; (c) fails
    failures = 0
    for _ in range(100):
        blocks = []
        for _ in range(3):
            rad, ang = rng.uniform(.3, .95), rng.uniform(0, np.pi)
            blocks.append(rad * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]))
        a = np.zeros((6, 6))
        for i, blk in enumerate(blocks):
            a[2 * i:2 * i + 2, 2 * i:2 * i + 2] = blk
        qmat, _ = np.linalg.qr(rng.normal(size=(6, 6))); a = qmat @ a @ qmat.T
        assert np.allclose(a @ a.T, a.T @ a)
        xs, deltas = linear(a, 40)
        d = np.linalg.norm(deltas, axis=1); ratio = d[1:] / d[:-1]
        assert np.all(np.diff(ratio[:25]) >= -1e-9 * ratio[:24])
        forecast = d[4] * geo(ratio[3], 20)
        assert forecast <= d[5:25].sum() * (1 + 1e-7)
        failures += forecast > np.linalg.norm(xs[25] - xs[5])
    assert failures > 0, "expected the displacement bound (c) to fail for rotations"
    # not normal: ratios can fall (Jordan block), so no sign is guaranteed
    a = np.array([[.8, 0], [1., .8]]); xs = orbit(lambda x: a @ x, np.array([1., 0]), 30)
    d = np.linalg.norm(np.diff(xs, axis=0), axis=1); ratio = d[1:] / d[:-1]
    assert np.any(np.diff(ratio) < -1e-6)
    return (f"positive spectra: displacement/forecast median {np.median(gaps):.2f}, max {np.max(gaps):.1f} (always >= 1); "
            f"rotations break (c) in {failures}/100 trials")


def theorem3():
    """Two-step bound, its comparison with the one-step bound, and the two-sided corollary."""
    gain = []
    for _ in range(200):
        n = 6
        w = rng.normal(size=(n, n)); w *= rng.uniform(.3, .9) / np.linalg.norm(w, 2); q = np.linalg.norm(w, 2)
        c = rng.normal(size=n)
        xs = orbit(lambda x: np.tanh(w @ x + c), rng.normal(size=n) * 3, 70)
        for r in (2, 4, 8):
            e = np.linalg.norm(xs[r] - xs[r - 2]); d = np.linalg.norm(xs[r] - xs[r - 1])
            for m in (1, 4, 14):
                disp = np.linalg.norm(xs[r + 2 * m] - xs[r])
                b2 = e * geo(q * q, m); b1 = d * geo(q, 2 * m)
                assert disp <= b2 * (1 + TOL)
                assert abs(b2 - b1 * (e / d) * q / (1 + q)) <= 1e-9 * b1                 # identity B2 = B (e/d) q/(1+q)
    # pure modes: reversing steps gain the factor (1+q)/(1-q); aligned steps give equality of the two bounds
    for q in (.5, .83):
        for sign in (-1, 1):
            xs = orbit(lambda x: sign * q * x + 1, np.array([5.]), 40)
            r, m = 6, 10
            e = abs(xs[r] - xs[r - 2])[0]; d = abs(xs[r] - xs[r - 1])[0]; disp = abs(xs[r + 2 * m] - xs[r])[0]
            b2 = e * geo(q * q, m); b1 = d * geo(q, 2 * m)
            assert abs(b2 - disp) <= 1e-9 * disp                                         # two-step bound is exact
            assert abs(b1 / b2 - ((1 + q) / (1 - q) if sign < 0 else 1)) <= 1e-9
            gain.append(b1 / b2)
    # corollary: symmetric linear map with eigenvalues of both signs
    for _ in range(300):
        eigs = rng.uniform(-1, 1, 8) * .95
        xs, _ = linear(sym(eigs), 45)
        for r in (4, 8, 16):
            e = np.linalg.norm(xs[r] - xs[r - 2]); e_before = np.linalg.norm(xs[r - 2] - xs[r - 4])
            for m in (1, 4, 12):
                disp = np.linalg.norm(xs[r + 2 * m] - xs[r])
                assert e * geo(e / e_before, m) <= disp * (1 + 1e-7) <= e * geo(max(eigs ** 2), m) * (1 + 1e-6)
    return f"one-step/two-step bound in pure modes: {[round(float(g), 2) for g in gain]}"


def proposition_period_two():
    """A map with an attracting two-cycle: parities converge separately; the one-step ratio tends to 1."""
    f = lambda x: np.array([-1.2 * np.tanh(2 * x[0]), .5 * x[1] + .1])
    xs = orbit(f, np.array([.3, 2.]), 200)
    a, b = xs[-2], xs[-1]
    assert np.linalg.norm(f(a) - b) < 1e-12 and np.linalg.norm(f(b) - a) < 1e-9 and np.linalg.norm(a - b) > 1
    d = np.linalg.norm(np.diff(xs, axis=0), axis=1)
    e = np.linalg.norm(xs[2:] - xs[:-2], axis=1)
    assert abs(d[60] / d[59] - 1) < 1e-6 and e[60] < 1e-6 * d[60]
    w = np.array([1., 0]); sa, sb = w @ a, w @ b
    flags = (xs[100:] @ w > (sa + sb) / 2)
    assert np.all(flags[:-1:2] != flags[1::2])                                             # threshold between: flag alternates forever
    assert len(set(xs[100:] @ w > max(sa, sb) + .1)) == 1                                 # threshold outside: constant
    return f"|X_a - X_b| = {np.linalg.norm(a - b):.3f}, step ratio -> {d[60] / d[59]:.6f}, two-step difference -> {e[60]:.1e}"


def theorem4():
    """Positions: block lower-triangular derivative."""
    out = []
    for _ in range(50):
        L, d = 5, 3
        j = rng.normal(size=(L * d, L * d)) * .6
        for p in range(L):
            j[p * d:(p + 1) * d, (p + 1) * d:] = 0                                       # position p reads positions <= p only
            blk = j[p * d:(p + 1) * d, p * d:(p + 1) * d]
            j[p * d:(p + 1) * d, p * d:(p + 1) * d] = blk * rng.uniform(.5, .95) / max(abs(np.linalg.eigvals(blk)))
        radii = [max(abs(np.linalg.eigvals(j[p * d:(p + 1) * d, p * d:(p + 1) * d]))) for p in range(L)]
        full = np.sort_complex(np.linalg.eigvals(j))
        parts = np.sort_complex(np.concatenate([np.linalg.eigvals(j[p * d:(p + 1) * d, p * d:(p + 1) * d]) for p in range(L)]))
        assert np.allclose(full, parts, atol=1e-7)                                       # (b)
        # (a): the prefix evolves by itself
        x = rng.normal(size=L * d); y = x.copy(); y[2 * d:] = rng.normal(size=(L - 2) * d)
        assert np.allclose(np.linalg.matrix_power(j, 7).dot(x)[:2 * d], np.linalg.matrix_power(j, 7).dot(y)[:2 * d])
        # (c): late per-position rates equal the running maximum of the block radii
        delta = rng.normal(size=L * d); t1, t2 = 200, 400   # radii >= 0.5 keep 0.5^400 above underflow
        a1 = np.linalg.matrix_power(j, t1) @ delta; a2 = np.linalg.matrix_power(j, t2) @ delta
        rate = np.array([(np.linalg.norm(a2[p * d:(p + 1) * d]) / np.linalg.norm(a1[p * d:(p + 1) * d])) ** (1 / (t2 - t1)) for p in range(L)])
        target = np.maximum.accumulate(radii)
        assert np.all(np.abs(rate - target) < .02), (rate, target)
        assert np.all(np.diff(rate) > -.02)
        # (d): not normal
        assert not np.allclose(j @ j.T, j.T @ j)
        out.append(np.linalg.norm(j @ j.T - j.T @ j))
    # the two-scalar cascade x -> ax, y -> ay + x, started where y's first step is zero:
    # y's step is (t-1) a^(t-2) times x's first step: it grows for about 1/log(1/a) loops before decaying
    a = np.array([[.9, 0], [1., .9]]); xs = orbit(lambda x: a @ x, np.array([1., 10.]), 60)
    step = abs(np.diff(xs, axis=0))
    peak = int(np.argmax(step[:, 1]))
    assert step[0, 1] < 1e-12 and peak + 1 in (10, 11) and step[peak, 1] > 3.8 * step[0, 0]
    assert np.allclose(step[1:40, 1], [(t - 1) * .9 ** (t - 2) * .1 for t in range(2, 41)])
    return f"cascade example: step of the later coordinate peaks at loop {peak + 1}, {step[peak, 1] / step[0, 0]:.1f} times the first step of the earlier one"


def theorem5():
    """Split-conformal multiplier: coverage between 1 - alpha and 1 - alpha + 1/(n+1), for a useless forecast too."""
    alpha, n, reps = .05, 150, 20000
    res = {}
    for name in ("informative", "useless"):
        realized = rng.lognormal(size=(reps, n + 1))
        forecast = realized * rng.lognormal(sigma=.3, size=(reps, n + 1)) if name == "informative" else rng.lognormal(size=(reps, n + 1))
        ratio = realized / forecast
        k = int(np.ceil((n + 1) * (1 - alpha)))
        kappa = np.sort(ratio[:, :n], axis=1)[:, k - 1]
        cover = np.mean(ratio[:, n] <= kappa)
        assert 1 - alpha - .005 <= cover <= 1 - alpha + 1 / (n + 1) + .005, cover
        res[name] = (cover, np.median(kappa * forecast[:, n] / realized[:, n]))
    return "; ".join(f"{k}: coverage {v[0]:.3f}, median bound/realized {v[1]:.1f}" for k, v in res.items())


def theorem6():
    """Transfer inequality and the ranking version, on simulated scores."""
    n = 200000
    wrong_r = rng.random(n) < .3
    s_r = rng.normal(size=n) + 1.2 * wrong_r
    beta = rng.uniform(.05, .4, n)
    ok = rng.random(n) < .95
    s_new = s_r + np.where(ok, rng.uniform(-1, 1, n) * beta, rng.normal(size=n) * 2)
    wrong_new = wrong_r ^ (rng.random(n) < .03)
    tau = np.quantile(s_r, .8)
    err = lambda s, y: np.mean((s > tau) != y)
    lhs = err(s_new, wrong_new)
    flip = np.mean((s_new > tau) != (s_r > tau)); churn = np.mean(wrong_new != wrong_r)
    assert lhs <= err(s_r, wrong_r) + flip + churn + 1e-12
    assert flip <= np.mean(np.abs(s_r - tau) <= beta) + np.mean(np.abs(s_new - s_r) > beta) + 1e-12
    # ranking version (labels fixed, every item within its bound)
    m = 3000
    y = rng.random(m) < .3; s = rng.normal(size=m) + y; b = rng.uniform(0, .3, m); s2 = s + rng.uniform(-1, 1, m) * b
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    auc = lambda v: np.mean((v[pos][:, None] > v[neg][None]) + .5 * (v[pos][:, None] == v[neg][None]))
    close = np.mean(np.abs(s[pos][:, None] - s[neg][None]) <= b[pos][:, None] + b[neg][None])
    assert abs(auc(s2) - auc(s)) <= close + 1e-12
    return (f"error at new budget {lhs:.3f} <= {err(s_r, wrong_r):.3f} + {flip:.3f} + {churn:.3f}; "
            f"AUROC change {abs(auc(s2) - auc(s)):.4f} <= close-pair mass {close:.3f}")


if __name__ == "__main__":
    for fn in (theorem1, theorem2, theorem3, proposition_period_two, theorem4, theorem5, theorem6):
        print(f"{fn.__name__}: OK. {fn()}")
    print("ALL CHECKS PASSED")
