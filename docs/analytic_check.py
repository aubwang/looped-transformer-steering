"""Numerical verification of docs/analytic.typ. numpy only. Exits non-zero on any failure.

Run: python -I docs/analytic_check.py
Part A: identities and inequalities on random stable matrices (all four families; absolute/relative tolerances,
        np.allclose with rtol=0).
Part B: deterministic edge cases and counterexamples (each documents why a hypothesis is needed).
Part C: the worked examples table (Section 6) and the 3x3 Huginn-like toy.
"""
import sys

import numpy as np

FAIL = []
INFORMATIVE_RESID = [0]
SYMCAP = [0]
NEGSIG = [0]


def ok(cond, msg):
    if not cond:
        FAIL.append(msg)


def basis(gh):
    """Orthonormal U spanning gh-perp (deterministic: Householder reflection mapping e1 to gh)."""
    d = len(gh)
    v = gh - np.eye(d)[0]
    H = np.eye(d) if np.linalg.norm(v) < 1e-14 else np.eye(d) - 2 * np.outer(v, v) / (v @ v)
    return H[:, 1:]


def quantities(J, g):
    d = len(g)
    gh = g / np.linalg.norm(g)
    U = basis(gh)
    al = gh @ J @ gh
    l, b, K = U.T @ J @ gh, U.T @ J.T @ gh, U.T @ J @ U
    Ik = np.eye(d - 1)
    f = np.linalg.solve(Ik - K.T, b)
    psi = al + l @ f
    sig = 1 - psi
    a = np.linalg.solve(np.eye(d) - J.T, J.T @ g)
    q = U.T @ ((J - J.T) / 2) @ gh
    E, C = b @ f, q @ f
    om = np.linalg.eigvalsh((K + K.T) / 2).max() if d > 1 else -np.inf
    na = np.linalg.norm(a)
    c0 = gh @ a / na if na > 0 else np.nan
    c1 = gh @ J @ a / na if na > 0 else np.nan
    return dict(gh=gh, U=U, al=al, l=l, b=b, K=K, f=f, psi=psi, sig=sig, a=a, q=q, E=E, C=C, om=om, c0=c0, c1=c1,
                rho=np.max(np.abs(np.linalg.eigvals(J))))


def random_J(rng, d, kind):
    M = rng.standard_normal((d, d))
    if kind == 0:  # generic
        return M / np.max(np.abs(np.linalg.eigvals(M))) * rng.uniform(.2, .95)
    if kind == 1:  # strongly non-normal
        T = np.triu(rng.standard_normal((d, d)) * 3, 1) + np.diag(rng.uniform(-.9, .9, d))
        Qo, _ = np.linalg.qr(rng.standard_normal((d, d)))
        return Qo @ T @ Qo.T
    if kind == 2:  # symmetric
        S = (M + M.T) / 2
        return S / np.max(np.abs(np.linalg.eigvalsh(S))) * rng.uniform(.2, .95)
    # normal with complex spectrum
    Qo, _ = np.linalg.qr(rng.standard_normal((d, d)))
    B = np.zeros((d, d))
    i = 0
    while i < d:
        if i + 1 < d and rng.random() < .6:
            r, th = rng.uniform(.1, .95), rng.uniform(0, np.pi)
            B[i:i + 2, i:i + 2] = r * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            i += 2
        else:
            B[i, i] = rng.uniform(-.95, .95)
            i += 1
    return Qo @ B @ Qo.T


def part_a():
    rng = np.random.default_rng(0)
    n = 0
    for trial in range(4000):
        d = int(rng.integers(2, 12))
        kind = trial % 4
        J = random_J(rng, d, kind)
        if np.max(np.abs(np.linalg.eigvals(J))) >= 1:
            continue
        g = rng.standard_normal(d) * rng.uniform(.5, 3)
        Q = quantities(J, g)
        I, Ik = np.eye(d), np.eye(d - 1)
        if np.linalg.cond(Ik - Q['K']) > 1e6 or np.linalg.cond(I - J) > 1e6 or np.linalg.norm(Q['a']) < 1e-8:
            continue
        n += 1
        gh, U, f, psi, sig, a = Q['gh'], Q['U'], Q['f'], Q['psi'], Q['sig'], Q['a']
        ng, nf = np.linalg.norm(g), np.linalg.norm(f)
        tol = 1e-8
        # Lemma 1: Green's function and determinant identity
        ok(abs(sig * (gh @ np.linalg.solve(I - J, gh)) - 1) < tol, f"green {trial}")
        ok(abs(np.linalg.det(I - J) - sig * np.linalg.det(Ik - Q['K'])) < tol * max(1, abs(np.linalg.det(I - J))), f"det {trial}")
        # Lemma 1: renewal equation (all kinds)
        T = 10
        r = [gh @ np.linalg.matrix_power(J, t) @ gh for t in range(T)]
        ps = [None, Q['al']] + [Q['b'] @ np.linalg.matrix_power(Q['K'], m - 2) @ Q['l'] for m in range(2, T)]
        for t in range(1, T):
            scale = sum(abs(ps[m] * r[t - m]) for m in range(1, t + 1)) + abs(r[t])
            ok(abs(r[t] - sum(ps[m] * r[t - m] for m in range(1, t + 1))) < 1e-10 * max(1, scale), f"renewal {trial},{t}")
        # Theorem 1: lever formula, cosine, signed tangent, orthogonal part
        a_pred = ng / sig * (psi * gh + U @ f)
        ok(np.linalg.norm(a - a_pred) < tol * np.linalg.norm(a), f"lever {trial}")
        D = np.hypot(psi, nf)
        ok(abs(Q['c0'] - np.sign(sig) * psi / D) < tol, f"cos {trial}")
        th = np.arccos(np.clip(Q['c0'], -1, 1))
        if abs(psi) > 1e-6 and abs(np.cos(th)) > 1e-6:
            ok(abs(np.tan(th) - np.sign(sig) * nf / psi) < 1e-6 * max(1, abs(np.tan(th))), f"tan sign {trial}")
        a_perp = a - (gh @ a) * gh
        ok(np.linalg.norm(a_perp - ng / sig * (U @ f)) < tol * max(1, np.linalg.norm(a)), f"a_perp {trial}")
        # Theorem 2
        ok(np.linalg.norm(Q['l'] - Q['b'] - 2 * Q['q']) < 1e-12, f"l=b+2q {trial}")
        ok(abs(psi - (Q['al'] + Q['E'] + 2 * Q['C'])) < tol * max(1, abs(psi)), f"twoforce {trial}")
        ok(abs(Q['E'] - f @ (Ik - Q['K']) @ f) < tol * max(1, abs(Q['E'])), f"E quad {trial}")
        # Theorem 3
        ok(abs(Q['c1'] - (Q['al'] * Q['c0'] + np.sign(sig) * Q['E'] / D)) < tol, f"reorient {trial}")
        om = Q['om']
        if om < 1 and nf > 1e-9:
            ok(Q['E'] >= (1 - om) * nf ** 2 - 1e-10, f"E bound {trial}")
            # sigma > 0 is implied by Re spec(J) < 1 and omega(K) < 1
            ok(sig > 0, f"sigma>0 from det {trial}")
            # Cor 3a
            sinth = nf / D
            ok(Q['c1'] - Q['al'] * Q['c0'] >= (1 - om) * nf * sinth - 1e-10, f"cor3a {trial}")
            # Cor 3b (Elman-type), omega(K) < 1 required
            cbf = Q['b'] @ f / (np.linalg.norm(Q['b']) * nf)
            ok(cbf >= (1 - om) / np.linalg.norm(Ik - Q['K'], 2) - 1e-10, f"cor3b {trial}")
        # Cor 2b: c0 < 0 and sigma > 0  =>  2C < -(alpha + E); with alpha >= 0, omega < 1: ||q|| > (1-omega)||f||/2
        if Q['c0'] < 0 and sig > 0:
            ok(2 * Q['C'] < -(Q['al'] + Q['E']) + 1e-10, f"cor2b {trial}")
            if Q['al'] >= 0 and om < 1:
                ok(np.linalg.norm(Q['q']) > (1 - om) * nf / 2 - 1e-10, f"cor2b q {trial}")
        if kind == 2:
            lam = np.linalg.eigvalsh(J)
            ok(np.linalg.norm(Q['q']) < 1e-12, f"sym q {trial}")
            ok(sig > 0, f"sym sigma {trial}")
            ok(Q['E'] >= (1 - lam.max()) * nf ** 2 - 1e-10, f"sym E {trial}")
            ok(nf >= np.linalg.norm(Q['b']) / (1 - lam.min()) - 1e-10, f"sym f lower {trial}")
            ok(g @ J @ a > 0, f"sym first-step positive {trial}")  # Prop 4
            if Q['al'] >= 0:
                ok(Q['c0'] > 0, f"sym no reversal {trial}")
                if nf > 1e-9:
                    ok(np.tan(th) <= 1 / ((1 - lam.max()) * nf) + 1e-8, f"sym tan {trial}")
            if lam.min() > 0:  # Kantorovich bound
                m, Mx = lam.min() / (1 - lam.min()), lam.max() / (1 - lam.max())
                ok(Q['c0'] >= 2 * np.sqrt(m * Mx) / (m + Mx) - 1e-10, f"kantorovich {trial}")
        # Corollary 4a identities, all signs of sigma: sigma = 1/(1+G cos th), ||f|| = G sin th / |1+G cos th|
        Gam = np.linalg.norm(a) / ng
        den = 1 + Gam * np.cos(th)
        ok(abs(sig * den - 1) < 1e-7 * max(1, abs(sig)), f"cor4a sigma {trial}")
        ok(abs(nf * abs(den) - Gam * np.sin(th)) < 1e-7 * max(1, Gam), f"cor4a f {trial}")
        if sig < 0:
            NEGSIG[0] += 1
        # Section 6 bounds. Theorem 4: 2||q|| >= alpha/||f|| + (1 - omega)||f|| - cot(theta)   (sigma > 0, f != 0)
        if sig > 0 and nf > 1e-9 and 1e-9 < th < np.pi - 1e-9:
            cot = np.cos(th) / np.sin(th)
            nq, al = np.linalg.norm(Q['q']), Q['al']
            rhs = al / nf + (1 - om) * nf - cot
            sc = max(1, abs(al / nf) + abs(1 - om) * nf + abs(cot) + 2 * nq)
            ok(2 * nq >= rhs - 1e-9 * sc, f"thm4 {trial}")
            if al >= 0:
                ok(2 * nq >= (1 - om) * Gam * np.sin(th) / (1 + Gam * np.cos(th)) - cot - 1e-9 * sc, f"cor4a {trial}")
                if om < 1:
                    ok(2 * nq >= 2 * np.sqrt(al * (1 - om)) - cot - 1e-9 * sc, f"cor4b {trial}")
                    if nq < np.sqrt(al * (1 - om)) * (1 - 1e-9):
                        ok(Q['c0'] > 0, f"cor4b no reversal {trial}")
            # Proposition 5: one-step lever angle
            nb = np.linalg.norm(Q['b'])
            if nb > 1e-9:
                JTg = J.T @ gh
                ang = np.arccos(np.clip(JTg @ a / (np.linalg.norm(JTg) * np.linalg.norm(a)), -1, 1))
                abf = np.arccos(np.clip(Q['b'] @ f / (nb * nf), -1, 1))
                ok(ang <= np.arctan(abs(al) / nb) + abf + abs(np.pi / 2 - th) + 1e-9, f"prop5 tri {trial}")
                rK = np.linalg.norm(Q['K'].T @ f) / nf
                if rK < 1:
                    ok(abf < np.pi / 2 and np.sin(abf) <= rK + 1e-9, f"prop5 sin {trial}")
        if kind == 2 and Q['al'] >= 0 and 0 < th <= np.pi / 2:
            lp = np.linalg.eigvalsh(J).max()
            if (1 - lp) * np.sin(th) ** 2 > np.cos(th) ** 2:
                cap = np.cos(th) / ((1 - lp) * np.sin(th) ** 2 - np.cos(th) ** 2)
                ok(np.linalg.norm(a) / ng <= cap * (1 + 1e-9) + 1e-12, f"cor4c {trial}")
                SYMCAP[0] += 1
        # Prop 4: relaxation
        be = rng.uniform(.1, 1.)
        Jb = (1 - be) * I + be * J
        Qb = quantities(Jb, g)
        ok(np.allclose(Qb['f'], f, rtol=0, atol=1e-8 * max(1, nf)), f"relax f {trial}")
        ok(abs(Qb['sig'] - be * sig) < 1e-8 * max(1, abs(sig)), f"relax sig {trial}")
        ok(abs(Qb['C'] - be * Q['C']) < 1e-8 * max(1, abs(Q['C'])), f"relax C {trial}")
        ok(np.allclose(Qb['a'], (a + g) / be - g, rtol=0, atol=1e-7 * max(1, np.linalg.norm(a) / be)), f"relax a {trial}")
        # Generating function: gh^T (I - zJ)^{-1} gh = 1/(1 - Psi(z)), Psi(z) = z alpha + z^2 b^T (I - zK)^{-1} l
        for z in (0.3, -0.7, -1.0):
            if np.linalg.cond(I - z * J) < 1e6 and np.linalg.cond(Ik - z * Q['K']) < 1e6:
                Psi = z * Q['al'] + z * z * Q['b'] @ np.linalg.solve(Ik - z * Q['K'], Q['l'])
                G = gh @ np.linalg.solve(I - z * J, gh)
                ok(abs(G * (1 - Psi) - 1) < 1e-8, f"genfun {trial},{z}")
        # Cor 3b cosine identity on the alpha = 0 projection of J
        J0 = J - Q['al'] * np.outer(gh, gh)
        if np.max(np.abs(np.linalg.eigvals(J0))) < 1:
            Q0 = quantities(J0, g)
            nf0 = np.linalg.norm(Q0['f'])
            if Q0['sig'] > 0 and nf0 > 1e-9 and np.linalg.norm(Q0['a']) > 1e-9:
                lhs = (J0.T @ gh) @ Q0['a'] / (np.linalg.norm(J0.T @ gh) * np.linalg.norm(Q0['a']))
                rhs = (nf0 / np.hypot(Q0['psi'], nf0)) * (Q0['b'] @ Q0['f']) / (np.linalg.norm(Q0['b']) * nf0)
                ok(abs(lhs - rhs) < 1e-8, f"cor3b identity {trial}")
        # Section 8 finite-T: residual and scalar diagnostic identities
        Tn = int(rng.integers(3, 30))
        aT = sum(np.linalg.matrix_power(J.T, t) @ g for t in range(1, Tn + 1))
        rT = (I - J.T) @ aT - J.T @ g
        # floating-point scale of the power sum: sum_t ||J||^t ||g|| (a priori bound on intermediate magnitudes)
        scaleT = sum(np.linalg.norm(J, 2) ** t for t in range(0, Tn + 2)) * ng
        errT = np.linalg.norm(rT + np.linalg.matrix_power(J.T, Tn + 1) @ g)
        tolT = 1e-12 * scaleT
        if tolT <= 1e-6 * max(1, np.linalg.norm(aT)):  # count only cases where the roundoff scale is informative
            INFORMATIVE_RESID[0] += 1
            ok(errT < tolT, f"resid {trial}: {errT:.2e} tol {tolT:.2e}")
        xT = gh @ aT / ng
        if abs(1 + xT) > 1e-6:
            sT, pT = 1 / (1 + xT), xT / (1 + xT)
            fT = sT * (aT - (gh @ aT) * gh) / ng
            qa = 0.5 * ((J @ gh - J.T @ gh) - (gh @ (J @ gh - J.T @ gh)) * gh)
            ET, CT = (J.T @ gh) @ fT, qa @ fT
            ok(abs((pT - Q['al'] - ET - 2 * CT) - sT / ng * (gh @ rT)) < 1e-8 * max(1, abs(pT), abs(ET)), f"scalar diag {trial}")
        # Section 7 measurement reconstruction from the ambient lever
        x = gh @ a / ng
        sig_m, psi_m = 1 / (1 + x), x / (1 + x)
        f_amb = sig_m * (a - (gh @ a) * gh) / ng
        q_amb = 0.5 * ((J @ gh - J.T @ gh) - (gh @ (J @ gh - J.T @ gh)) * gh)
        ok(abs(psi_m - psi) < 1e-7 * max(1, abs(psi)), f"meas psi {trial}")
        ok(abs((J.T @ gh) @ f_amb - Q['E']) < 1e-7 * max(1, abs(Q['E'])), f"meas E {trial}")
        ok(abs(q_amb @ f_amb - Q['C']) < 1e-7 * max(1, abs(Q['C'])), f"meas C {trial}")
    # Multi-dimensional readout (Remark)
    for trial in range(200):
        d, k = int(rng.integers(4, 10)), int(rng.integers(2, 4))
        J = random_J(rng, d, trial % 4)
        Qo, _ = np.linalg.qr(rng.standard_normal((d, d)))
        Jb = Qo.T @ J @ Qo
        A, Bt, L, K = Jb[:k, :k], Jb[:k, k:], Jb[k:, :k], Jb[k:, k:]
        F = np.linalg.solve(np.eye(d - k) - K.T, Bt.T)
        Sig = np.eye(k) - A - Bt @ np.linalg.solve(np.eye(d - k) - K, L)
        v = rng.standard_normal(k)
        a = np.linalg.solve(np.eye(d) - Jb.T, Jb.T @ np.r_[v, np.zeros(d - k)])
        SiT = np.linalg.inv(Sig).T
        ok(np.allclose(a, np.r_[(SiT - np.eye(k)) @ v, F @ SiT @ v], rtol=0, atol=1e-8 * max(1, np.linalg.norm(a))), f"multidim {trial}")
    ok(INFORMATIVE_RESID[0] >= 1000, f"too few informative residual cases: {INFORMATIVE_RESID[0]}")
    print(f"Part A: {n} random cases checked ({INFORMATIVE_RESID[0]} with an informative residual tolerance, "
          f"{SYMCAP[0]} symmetric-cap cases, {NEGSIG[0]} with sigma < 0)")


def part_b():
    e = lambda d: np.eye(d)[0]
    R = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    # sigma can be negative for a stable (nilpotent) J; the sgn(sigma) factors are needed
    Q = quantities(np.array([[-2., 1], [-4, 2]]), e(2))
    ok(Q['sig'] < 0 and Q['rho'] < 1e-6, "B1 sigma<0")
    # I - K singular although J is nilpotent: invertibility of I - K is a real hypothesis
    J = np.array([[-1., 1], [-1, 1]])
    ok(abs(J[1, 1] - 1) < 1e-12 and np.allclose(J @ J, 0, rtol=0, atol=1e-14), "B2 K=1")
    # symmetric reversal through negative self-gain (alpha < 0): Cor 2b needs alpha + E > 0
    Q = quantities(np.diag([-.5, 0.]), e(2))
    ok(abs(Q['C']) < 1e-12 and abs(Q['c0'] + 1) < 1e-12 and abs(Q['c1'] - .5) < 1e-12, "B3 symmetric reversal")
    # normal rotation with alpha > 0 still reverses
    Q = quantities(.9 * R(np.pi / 3), e(2))
    ok(Q['al'] > 0 and abs(Q['c0'] + 0.4193139) < 1e-6, "B4 rotation 60")
    # f = 0: Cor 3a strict inequality fails, so f != 0 is required
    Q = quantities(np.diag([.5, 0.]), e(2))
    ok(np.linalg.norm(Q['f']) == 0 and abs(Q['c1'] - Q['al'] * Q['c0']) < 1e-12, "B5 f=0")
    # small alpha in absolute terms is not enough: must be small relative to E/D
    Q = quantities(np.array([[.01, .001], [-20, 0]]), e(2))
    ok(Q['c0'] < -.99 and Q['c1'] < 0 and Q['om'] < 1 and Q['sig'] > 0, "B6 no flip")
    # Cor 3b bound fails without omega(K) < 1
    J = np.array([[0, 1, -2], [0, 0, 4], [0, 0, 0.]])
    Q = quantities(J, e(3))
    cbf = Q['b'] @ Q['f'] / (np.linalg.norm(Q['b']) * np.linalg.norm(Q['f']))
    ok(Q['om'] > 1 and cbf < (1 - Q['om']) / np.linalg.norm(np.eye(2) - Q['K'], 2), "B7 Cor3b needs omega<1")
    # omega(K) < 1 is sufficient, not necessary, for a good one-step lever (and not sufficient by itself)
    J = np.array([[0, .001], [-500, 0]])
    a = np.linalg.solve(np.eye(2) - J.T, J.T @ e(2))
    ok(abs((J.T @ e(2)) @ a / np.linalg.norm(J.T @ e(2)) / np.linalg.norm(a)) < .01, "B8 bad one-step lever")
    # symmetric, positive spectrum, nearly orthogonal lever (C = 0): circulation is not needed for near-orthogonality
    eps = 1e-4
    g = np.array([np.sqrt(1 - eps), np.sqrt(eps)])
    Q = quantities(np.diag([eps, .5]), g)
    ok(abs(Q['C']) < 1e-12 and abs(Q['c0'] - 2 * np.sqrt(eps * (1 - eps))) < 1e-9, "B9 symmetric near-orthogonal")
    # truncation: rho = 0.78 does not make T = 48 negligible for strongly non-normal J
    d = 50
    J = .78 * np.eye(d) + np.eye(d, k=1)
    aT = sum(np.linalg.matrix_power(J.T, t) @ e(d) for t in range(1, 49))
    ai = np.linalg.solve(np.eye(d) - J.T, J.T @ e(d))
    ok(np.linalg.norm(aT - ai) / np.linalg.norm(ai) > .9, "B10 truncation")
    # negative passage term: alpha = C = 0, E < 0, stable, c0 < 0
    Q = quantities(np.array([[0, .1, -.2], [.1, 0, 4], [-.2, 0, 0]]), e(3))
    ok(abs(Q['al']) < 1e-12 and abs(Q['C']) < 1e-12 and abs(Q['E'] + .03) < 1e-12 and abs(Q['sig'] - 1.03) < 1e-12
       and abs(Q['c0'] + .133) < 1e-3 and Q['rho'] < .47, "B11 negative passage")
    # alpha < -E but circulation supplies most of the negative return
    Q = quantities(np.array([[-.02, .1], [-5, 0]]), e(2))
    ok(abs(Q['al'] + .02) < 1e-12 and abs(Q['E'] - .01) < 1e-12 and abs(2 * Q['C'] + .51) < 1e-12
       and abs(Q['c0'] + .982) < 1e-3 and abs(Q['c1'] - .0385) < 1e-3 and Q['rho'] < 1, "B12 alpha<-E")
    # symmetric but lambda_max > 1 (resolvent only): alpha > 0 yet c0 = -1, c1 = -2
    Q = quantities(np.diag([2., 0]), e(2))
    ok(abs(Q['c0'] + 1) < 1e-12 and abs(Q['c1'] + 2) < 1e-12, "B13 symmetric lambda>1")
    # unobserved nilpotent block: omega(K) = 2 while a = J^T g exactly
    J = np.zeros((4, 4)); J[0, 1] = 1; J[2, 3] = 4
    Q = quantities(J, e(4))
    ok(abs(Q['om'] - 2) < 1e-12 and np.allclose(Q['a'], J.T @ e(4), rtol=0, atol=1e-14), "B14 unobserved block")
    # z = -1: I + J invertible but I + K singular
    J = np.array([[1., 1], [-1, -1]])
    ok(abs(np.linalg.det(np.eye(2) + J)) > .5 and abs(1 + J[1, 1]) < 1e-14, "B15 genfun singular K")
    # Theorem 4 is sharp: every 2x2 with sigma > 0 and q f <= 0 attains equality (stable examples)
    for al, k, b, l in [(.05, .5, .3, -.4), (.2, .1, 1., .2), (0., .6, 1.5, -.3)]:
        J = np.array([[al, b], [l, k]])
        Q = quantities(J, e(2))
        f = abs(Q['f'][0]); th = np.arccos(Q['c0']); cot = np.cos(th) / np.sin(th)
        ok(Q['rho'] < 1 and Q['sig'] > 0 and Q['q'][0] * Q['f'][0] <= 0
           and abs(2 * abs(Q['q'][0]) - (al / f + (1 - k) * f - cot)) < 1e-12, f"B16 thm4 sharp {al}")
    # Theorem 4 equality in dimension 4: f on the top eigenvector of sym(K), q = -c f  (equality);
    # the same with q rotated off -f (strict)
    rng = np.random.default_rng(7)
    n16 = [0, 0]
    for trial in range(200):
        d = 4
        M = rng.standard_normal((d - 1, d - 1)) * .3
        K = (M + M.T) / 2 * .8 + (M - M.T) / 2
        w, V = np.linalg.eigh((K + K.T) / 2)
        v = V[:, -1] * rng.uniform(.2, 2)
        b = (np.eye(d - 1) - K.T) @ v                      # so f = v
        for strict in (False, True):
            qv = -rng.uniform(0, 2) * v
            if strict:
                pv = rng.standard_normal(d - 1); pv -= (pv @ v) / (v @ v) * v
                qv = qv + .5 * pv / np.linalg.norm(pv) * np.linalg.norm(qv)
            l = b + 2 * qv
            J = np.zeros((d, d)); J[0, 0] = rng.uniform(0, .5); J[0, 1:] = b; J[1:, 0] = l; J[1:, 1:] = K
            Q = quantities(J, e(d))
            if Q['sig'] <= 0 or np.linalg.cond(np.eye(d - 1) - K) > 1e6:
                continue
            f = np.linalg.norm(Q['f']); th = np.arccos(Q['c0']); cot = np.cos(th) / np.sin(th)
            gap = 2 * np.linalg.norm(Q['q']) - (Q['al'] / f + (1 - Q['om']) * f - cot)
            sc = max(1, abs(Q['al'] / f) + abs(1 - Q['om']) * f + abs(cot))
            ok(np.allclose(Q['f'], v) and (gap > 1e-6 * sc if strict else abs(gap) < 1e-9 * sc),
               f"B16 thm4 equality d=4 strict={strict} {trial}")
            n16[strict] += 1
    ok(min(n16) >= 100, f"B16 d=4 coverage {n16}")
    print(f"B16: Theorem 4 equality/strict cases in d=4: {n16}")
    # Corollary 4a needs sigma > 0: stable, sigma = -1.6, the displayed inequality fails (1.5 < 1.52)
    J = np.array([[.1, 1], [-.5, 1.2]]); Q = quantities(J, e(2))
    th = np.arccos(Q['c0']); Gam = np.linalg.norm(Q['a']); den = 1 + Gam * np.cos(th)
    rhs = (1 - Q['om']) * Gam * np.sin(th) / den - np.cos(th) / np.sin(th)
    ok(Q['rho'] < 1 and abs(Q['sig'] + 1.6) < 1e-12 and abs(2 * np.linalg.norm(Q['q']) - 1.5) < 1e-12
       and abs(rhs - 1.52) < 1e-9 and abs(np.linalg.norm(Q['f']) - Gam * np.sin(th) / abs(den)) < 1e-12,
       "B17 cor4a needs sigma > 0")
    # Theorem 4's right side can be negative (equality then impossible): RHS = -0.1, 2||q|| = 0.1
    J = np.array([[0., .1], [.2, 0]]); Q = quantities(J, e(2))
    th = np.arccos(Q['c0']); f = np.linalg.norm(Q['f'])
    ok(Q['sig'] > 0 and abs((1 - Q['om']) * f - np.cos(th) / np.sin(th) + .1) < 1e-12
       and abs(2 * np.linalg.norm(Q['q']) - .1) < 1e-12, "B18 thm4 negative rhs")
    # Corollary 4c needs a spectral gap: symmetric, eps = 1e-4, Gamma ~ 100 at theta ~ 89.43 deg, hypothesis fails
    ep = 1e-4; J = np.diag([ep, 1 - ep]); Q = quantities(J, np.array([np.sqrt(1 - ep), np.sqrt(ep)]))
    th = np.arccos(Q['c0']); Gam = np.linalg.norm(Q['a'])
    ok(99.9 < Gam < 100 and 89.4 < np.degrees(th) < 89.45 and np.linalg.norm(Q['q']) < 1e-14
       and Q['al'] > 0 and ep * np.sin(th) ** 2 <= np.cos(th) ** 2, "B19 cor4c no gap")
    # Proposition 5: small |alpha| is not enough, |alpha|/||b|| matters. sigma = 1, theta = 90 deg, K = 0, b || f
    J = np.array([[.001, 1e-6], [-1000, 0]]); Q = quantities(J, e(2))
    JTg = J.T @ e(2); ang = np.degrees(np.arccos(JTg @ Q['a'] / (np.linalg.norm(JTg) * np.linalg.norm(Q['a']))))
    ok(Q['rho'] < .04 and abs(Q['sig'] - 1) < 1e-12 and abs(Q['c0']) < 1e-12 and 89.94 < ang < 89.95,
       "B20 prop5 needs |alpha| << ||b||")
    # Corollary 4b with f = 0: b = 0, a along g-hat, theta = 0, c0 = 1
    Q = quantities(np.diag([.5, .3, -.2]), e(3))
    ok(np.linalg.norm(Q['f']) == 0 and Q['al'] > 0 and abs(Q['c0'] - 1) < 1e-12, "B21 cor4b f = 0")
    print("Part B: edge cases checked")


def show(name, J):
    Q = quantities(J, np.eye(len(J))[0])
    Jg = J @ np.eye(len(J))[0]
    print(f"{name:10s} eig={np.round(np.linalg.eigvals(J), 3)} alpha={Q['al']:.3f} l={np.round(Q['l'], 3)} "
          f"b={np.round(Q['b'], 3)} f={np.round(Q['f'], 3)} psi={Q['psi']:.3f} sigma={Q['sig']:.3f} E={Q['E']:.3f} "
          f"C={Q['C']:.3f} c0={Q['c0']:.4f} c1={Q['c1']:.4f} omegaK={Q['om']:.3f} |Jg|={np.linalg.norm(Jg):.3f}")
    return Q


def part_c():
    R = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rows = {
        "toy": np.array([[.1, 2], [0, .5]]),
        "reversal": np.array([[.1, 2], [-.2, .5]]),
        "rot60": .9 * R(np.pi / 3),
        "sym": .6 * np.array([[.1, 1], [1, .5]]),
        "sym-neg": np.diag([-.5, 0.]),
        "huginn3": np.array([[0, 1.5, 1.5], [-.9, .6, 0], [.74, -.4, .2]]),
    }
    out = {k: show(k, J) for k, J in rows.items()}
    # displayed table: alpha, |J gh|, E, C, psi, c0, c1 (checked to the displayed rounding)
    # displayed table, as printed strings so trailing zeros fix the rounding precision:
    # eigenvalues, alpha, |J gh|, E, C, psi, c0, c1
    exp = {"toy": ([0.1, 0.5], "0.1", "0.10", "8", "-4", "0.1", "0.025", "2.00"),
           "reversal": (["0.3+0.6j", "0.3-0.6j"], "0.1", "0.22", "8", "-4.4", "-0.7", "-0.17", "1.95"),
           "rot60": (["0.45+0.78j", "0.45-0.78j"], "0.45", "0.90", "1.10", "-1.10", "-0.65", "-0.42", "0.52"),
           "sym": (["-0.43", "0.79"], "0.06", "0.60", "0.51", "0", "0.57", "0.56", "0.53"),
           "sym-neg": (["-0.5", "0"], "-0.5", "0.50", "0", "0", "-0.5", "-1", "0.5"),
           "huginn3": (["0.6", "0.1+0.48j", "0.1-0.48j"], "0", "1.17", "5.63", "-2.96", "-0.30", "-0.11", "2.11")}

    def dec(txt):
        txt = txt.replace("j", "")
        parts = [p for p in txt.replace("+", " ").replace("-", " ").split() if p]
        return min(len(p.split(".")[1]) if "." in p else 0 for p in parts)

    for k, (eigs, *vals) in exp.items():
        Q = out[k]
        got = (Q['al'], np.linalg.norm(rows[k][:, 0]), Q['E'], Q['C'], Q['psi'], Q['c0'], Q['c1'])
        for name, gv, ev in zip(("alpha", "|Jg|", "E", "C", "psi", "c0", "c1"), got, vals):
            ok(abs(gv - float(ev)) <= .5 * 10 ** (-dec(ev)) + 1e-9, f"table {k} {name}: {gv} vs {ev}")
        ev_true = sorted(np.linalg.eigvals(rows[k]), key=lambda z: (round(z.real, 6), z.imag))
        ev_exp = sorted((complex(str(e)) for e in eigs), key=lambda z: (round(z.real, 6), z.imag))
        for zt, (ze, txt) in zip(ev_true, zip(ev_exp, sorted(map(str, eigs), key=lambda t: (complex(t).real, complex(t).imag)))):
            tol = .5 * 10 ** (-dec(txt)) + 1e-9
            ok(abs(zt.real - ze.real) <= tol and abs(zt.imag - ze.imag) <= tol, f"table {k} eig {zt} vs {txt}")
    Q = out["huginn3"]
    ok(abs(Q['rho'] - .6) < 1e-9 and abs(Q['om'] - .683) < 5e-4, "table huginn3 rho/omega")
    J = rows["huginn3"]
    gh = np.eye(3)[0]
    a = out["huginn3"]['a']
    print(f"huginn3: |J g|={np.linalg.norm(J @ gh):.3f} rho={out['huginn3']['rho']:.3f} "
          f"cos(J^T g, a)={(J.T @ gh) @ a / np.linalg.norm(J.T @ gh) / np.linalg.norm(a):.3f}")
    ok(abs((J.T @ gh) @ a / np.linalg.norm(J.T @ gh) / np.linalg.norm(a) - .994) < 5e-4, "huginn3 one-step cos")
    # huginn3 toy through the Section 6 bounds
    Q = out["huginn3"]
    th = np.arccos(Q['c0']); Gam = np.linalg.norm(a)
    need = .5 * ((1 - Q['om']) * Gam * np.sin(th) / (1 + Gam * np.cos(th)) - np.cos(th) / np.sin(th))
    print(f"huginn3 bounds: Gamma={Gam:.3f} theta={np.degrees(th):.1f}deg omegaK={Q['om']:.3f} "
          f"required ||q|| >= {need:.3f}, actual ||q|| = {np.linalg.norm(Q['q']):.3f}")
    ok(abs(Gam - 2.053) < 1e-3 and abs(need - .477) < 1e-3 and abs(np.linalg.norm(Q["q"]) - 1.259) < 1e-3, "huginn3 bounds")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):", FAIL[:20])
        sys.exit(1)
    print("ALL CHECKS PASSED")
