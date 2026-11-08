"""Numerical checks for docs/observability.typ: Theorems 1, 1', 2a-c (incl. the p_min bound and exact minimum-norm
reconstruction), Corollary 1b example, Propositions 3-4, the Section 7 counterexample, and the m_p values. numpy only.
Exits with status 1 if any check fails.
Usage: python -I docs/observability_check.py
"""
import numpy as np

rng = np.random.default_rng(0)
FAIL = []


def ok(c, msg):
    if not c:
        FAIL.append(msg)
        print("FAIL", msg)


def sqrtm_psd(A):
    w, V = np.linalg.eigh(A)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def fisher(V):
    z = rng.normal(size=V) * 2
    pi = np.exp(z - z.max()); pi /= pi.sum()
    return np.diag(pi) - np.outer(pi, pi), pi


def Wp(J, Hh, G, p):
    """W_p = sum_r p(r) (J^T)^r G^T H G J^r, with r = 1..len(p)."""
    d = J.shape[0]; W = np.zeros((d, d)); Jr = np.eye(d)
    for r in range(1, len(p) + 1):
        Jr = J @ Jr
        X = Hh @ G @ Jr
        W += p[r - 1] * X.T @ X
    return W


# ---- Theorem 1 on random non-normal loops, including expanding modes
for trial in range(200):
    d, V, R = 10, 7, 12
    J = rng.normal(size=(d, d)) / np.sqrt(d) * rng.uniform(0.5, 1.3)
    G = rng.normal(size=(V, d)); H, _ = fisher(V); Hh = sqrtm_psd(H)
    p = rng.dirichlet(np.ones(R))
    W = Wp(J, Hh, G, p); Om = np.linalg.norm(W, 2)
    lam, Vec = np.linalg.eig(J)
    for i in range(d):
        v = Vec[:, i] / np.linalg.norm(Vec[:, i])
        mp = sum(p[r - 1] * abs(lam[i]) ** (2 * r) for r in range(1, R + 1))
        lhs = mp * np.linalg.norm(Hh @ G @ v) ** 2
        ok(lhs <= Om * (1 + 1e-9) + 1e-12, f"T1 eig trial {trial} mode {i}: {lhs} > {Om}")
    for _ in range(5):  # general unit vectors (complex)
        v = rng.normal(size=d) + 1j * rng.normal(size=d); v /= np.linalg.norm(v)
        lhs = sum(p[r - 1] * np.linalg.norm(Hh @ G @ np.linalg.matrix_power(J, r) @ v) ** 2 for r in range(1, R + 1))
        ok(lhs <= Om * (1 + 1e-9) + 1e-12, f"T1 general trial {trial}")
print("Theorem 1 checked on 200 random loops")

# ---- Corollary 1b example: d = 1
lam, R, eta = 1.3, 4, 1e-3
a = eta * lam ** (-R); Om = a ** 2 * lam ** (2 * R)
ok(abs(Om - eta ** 2) < 1e-15, "C1b Omega"); ok(a * lam ** 40 > 1e4 * eta, "C1b blow-up")
print("Corollary 1b example: Omega =", Om, " effect at k=40:", a * lam ** 40)

# ---- Theorem 2: build a loop whose observable subspace K is a proper subspace
for trial in range(100):
    d, V, kdim, R = 12, 6, 5, 10
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))          # adapted orthonormal basis: first kdim columns span K
    A = rng.normal(size=(kdim, kdim)) / np.sqrt(kdim) * 0.8
    B = rng.normal(size=(kdim, d - kdim)) / np.sqrt(d)
    C = rng.normal(size=(d - kdim, d - kdim)) / np.sqrt(d - kdim) * 0.8
    JT_b = np.block([[A, B], [np.zeros((d - kdim, kdim)), C]])  # J^T in the adapted basis
    JT = Q @ JT_b @ Q.T; J = JT.T
    H, _ = fisher(V); Hh = sqrtm_psd(H)
    # G^T Hh must have columns in K: G^T = Q[:, :kdim] @ Y  ->  G = Y^T Q[:, :kdim]^T, then G^T Hh = Q_K (Y Hh)
    Y = rng.normal(size=(kdim, V)); G = (Q[:, :kdim] @ Y).T
    p = rng.dirichlet(np.ones(R))
    W = Wp(J, Hh, G, p)
    # (a) change B, C: same W_p and same Fisher output response
    Bp = rng.normal(size=B.shape) * 3; Cp = rng.normal(size=C.shape) * 2
    JTp = Q @ np.block([[A, Bp], [np.zeros_like(Bp.T), Cp]]) @ Q.T; Jp = JTp.T
    Wp2 = Wp(Jp, Hh, G, p)
    ok(np.allclose(W, Wp2, atol=1e-10), f"T2a W_p trial {trial}")
    xi = rng.normal(size=d)
    for r in (1, 3, 7):
        ok(np.allclose(Hh @ G @ np.linalg.matrix_power(J, r) @ xi, Hh @ G @ np.linalg.matrix_power(Jp, r) @ xi, atol=1e-9),
           f"T2a response trial {trial} r {r}")
    # (b) monitor in range(W_kappa): bound holds exactly in expectation (sigma = 1)
    kap = 4
    O = np.vstack([Hh @ G @ np.linalg.matrix_power(J, k) for k in range(kap)])
    Wk = O.T @ O
    c = rng.normal(size=O.shape[0]); m = O.T @ c
    cost = m @ np.linalg.pinv(Wk, rcond=1e-10, hermitian=True) @ m
    ok(cost <= c @ c * (1 + 1e-6), f"T2b cost minimality trial {trial}")
    for r in (1, 2, 5):
        lhs = np.linalg.norm(np.linalg.matrix_power(JT, r) @ m) ** 2          # E|m^T J^r xi|^2 with sigma = 1
        rhs = cost * sum(np.trace(Hh @ G @ np.linalg.matrix_power(J, r + k) @ np.linalg.matrix_power(J, r + k).T @ G.T @ Hh)
                         for k in range(kap))
        ok(lhs <= rhs * (1 + 1e-6) + 1e-12, f"T2b bound trial {trial} r {r}: {lhs} > {rhs}")
    # (c) monitor with a K-perp component: C' = t I blows up the drift with the same training-visible dynamics
    m = Q @ np.concatenate([rng.normal(size=kdim), rng.normal(size=d - kdim)])
    JTc = Q @ np.block([[A, B], [np.zeros((d - kdim, kdim)), 1.5 * np.eye(d - kdim)]]) @ Q.T
    ok(np.allclose(Wp(JTc.T, Hh, G, p), W, atol=1e-10), f"T2c same W_p trial {trial}")
    ok(np.linalg.norm(np.linalg.matrix_power(JTc, 30) @ m) ** 2 > 1e4 * np.linalg.norm(m) ** 2, f"T2c blow-up trial {trial}")
    # K computed from the Krylov space equals span(Q_K)
    Kr = np.linalg.matrix_rank(np.vstack([Hh @ G @ np.linalg.matrix_power(J, k) for k in range(d)]), tol=1e-8)
    ok(Kr <= kdim, f"T2 Krylov rank {Kr} > {kdim}")
print("Theorem 2 (a)-(c) checked on 100 random block-triangular loops")

# ---- Theorem 2(b), second bound, and exact minimum-norm reconstruction
for trial in range(50):
    d, V, kap = 8, 5, 3
    J = rng.normal(size=(d, d)) / np.sqrt(d) * 0.9
    G = rng.normal(size=(V, d)); H, _ = fisher(V); Hh = sqrtm_psd(H)
    O = np.vstack([Hh @ G @ np.linalg.matrix_power(J, k) for k in range(kap)])
    Wk = O.T @ O
    c0 = rng.normal(size=O.shape[0]); m = O.T @ c0
    cmin = np.linalg.lstsq(O.T, m, rcond=None)[0]                     # minimum-norm solution of O^T c = m
    cost = m @ np.linalg.pinv(Wk, rcond=1e-12, hermitian=True) @ m
    ok(abs(cmin @ cmin - cost) <= 1e-8 * max(1.0, cost), f"T2 cost = min-norm reconstruction trial {trial}")
    R = 12; p = np.full(R, 1.0 / R); pmin = p.min(); sig = 0.7
    Sp = sig ** 2 * sum(p[r - 1] * np.trace(Hh @ G @ np.linalg.matrix_power(J, r) @ np.linalg.matrix_power(J, r).T @ G.T @ Hh)
                        for r in range(1, R + 1))
    for r in range(1, R - kap + 1):                                   # p(r + k) >= pmin for k < kappa
        lhs = sig ** 2 * np.linalg.norm(np.linalg.matrix_power(J.T, r) @ m) ** 2
        ok(lhs <= cost * Sp / pmin * (1 + 1e-9), f"T2b p_min bound trial {trial} r {r}")
print("Theorem 2(b) p_min bound and minimum-norm cost checked on 50 random loops")

# ---- Theorem 1': ||J^m||_F^2 <= S_p / (sigma^2 p_min alpha) when W_kappa >= alpha I
for trial in range(100):
    d, V, kap, R = 6, 6, 3, 10
    J = rng.normal(size=(d, d)) / np.sqrt(d) * rng.uniform(0.3, 1.2)
    G = rng.normal(size=(V, d)); H, _ = fisher(V); Hh = sqrtm_psd(H)
    Wk = sum(np.linalg.matrix_power(J.T, k) @ G.T @ H @ G @ np.linalg.matrix_power(J, k) for k in range(kap))
    alpha = np.linalg.eigvalsh(Wk).min()
    if alpha <= 1e-10:
        continue
    p = rng.dirichlet(np.ones(R)); sig = 0.3
    Sp = sig ** 2 * np.trace(Wp(J, Hh, G, p))
    for mm in range(1, R - kap + 1):
        pmin = min(p[mm + k - 1] for k in range(kap))
        lhs = np.linalg.norm(np.linalg.matrix_power(J, mm), "fro") ** 2
        ok(lhs <= Sp / (sig ** 2 * pmin * alpha) * (1 + 1e-9), f"T1' trial {trial} m {mm}: {lhs}")
print("Theorem 1' checked on random fully observable loops")

# ---- Proposition 3: frozen-Jacobian prediction error under contraction (F = q tanh(A x) + b, ||A||_2 = 1)
for trial in range(50):
    d, q, h = 5, rng.uniform(0.3, 0.8), 6
    A = rng.normal(size=(d, d)); A /= np.linalg.norm(A, 2); b = rng.normal(size=d) * 0.3
    F = lambda x: q * np.tanh(A @ x) + b
    DF = lambda x: q * (1 - np.tanh(A @ x) ** 2)[:, None] * A
    # ||D^2 F|| <= q * max|tanh''| * ||A||^2 with max|tanh''| = 4 / (3 sqrt 3)
    Mb = q * 4 / (3 * np.sqrt(3))
    x = rng.normal(size=d); u = rng.normal(size=d) * 0.05
    xs = x.copy()
    for _ in range(h):
        xs = F(xs) + u
    Jx = DF(x); v = F(x) - x + u
    Sh = sum(np.linalg.matrix_power(Jx, j) for j in range(h))
    err = np.linalg.norm(xs - x - Sh @ v)
    ok(err <= Mb * np.linalg.norm(v) ** 2 / (2 * (1 - q) ** 3) * (1 + 1e-9), f"P3 trial {trial}: {err}")
print("Proposition 3 checked on 50 contracting tanh maps")

# ---- Proposition 4(b): Bayes error of two equal-covariance Gaussians vs the lower bound
from math import erf, sqrt
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
for trial in range(50):
    d, q, sg = 4, rng.uniform(0.5, 0.95), rng.uniform(0.2, 1.0)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d))); J = q * Q             # ||J|| = q
    Lr = rng.normal(size=(d, d)) * 0.3; Sig = Lr @ Lr.T; delta = rng.normal(size=d)
    for t in (1, 3, 6):
        Jt = np.linalg.matrix_power(J, t); C = Jt @ Sig @ Jt.T + sg ** 2 * np.eye(d); mu = Jt @ delta
        Dt = np.sqrt(mu @ np.linalg.solve(C, mu))
        ok(Phi(-Dt / 2) >= Phi(-q ** t * np.linalg.norm(delta) / (2 * sg)) - 1e-12, f"P4b trial {trial} t {t}")
print("Proposition 4(b) checked on 50 Gaussian examples")

# ---- Section 7 counterexample: equal Omega_p, reversed persistence ordering; and the m_p values
def mp_quad(x, M, s=0.5, n=4001):
    """m_p(x) = x E exp(Lambda (x - 1)), Lambda lognormal with E Lambda = M (Gauss-Hermite-free trapezoid in log space)."""
    mu = np.log(M) - s * s / 2; zz = np.linspace(-8, 8, n); w = np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi)
    lam = np.exp(mu + s * zz)
    return x * np.trapezoid(w * np.exp(lam * (x - 1)), zz)
vals = [mp_quad(0.81, M) for M in (4, 8, 16, 32)]
print("m_p(0.81) by quadrature:", [round(v, 4) for v in vals])
ok(all(abs(a - b) < 0.006 for a, b in zip(vals, (0.41, 0.22, 0.081, 0.016))), f"m_p values {vals}")
lam4, lam32 = 0.95, 0.8
a4 = 1e-6 / mp_quad(lam4 ** 2, 4); a32 = 1e-6 / mp_quad(lam32 ** 2, 32)        # a^2 values
Om4, Om32 = a4 * mp_quad(lam4 ** 2, 4), a32 * mp_quad(lam32 ** 2, 32)
ok(abs(Om4 - Om32) < 1e-15 and lam4 ** 14 > lam32 ** 14, "Section 7 counterexample")
print(f"Section 7 counterexample: Omega {Om4:.2e} = {Om32:.2e}; F8/F1 = {lam4 ** 14:.3f} (M=4) > {lam32 ** 14:.3f} (M=32)")

print("FAILS", FAIL[:5], len(FAIL))
print("ALL CHECKS PASSED" if not FAIL else "CHECKS FAILED")
raise SystemExit(1 if FAIL else 0)
