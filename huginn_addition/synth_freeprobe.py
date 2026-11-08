"""Synthetic calibration and power checks for analyze_freeprobe.py (no GPU). Builds fake `freeprobe` outputs from explicit
matrices J (d x d), runs the analysis and prints its verdicts.
Usage: python -I huginn_addition/synth_freeprobe.py <kind> <d> <n_items> [param] [--T 48] [--out /tmp/synth_freeprobe]
Kinds (readout gh uniformly random unless stated; K = 4 probes):
  free       J = gamma I + beta gh gh^T + U diag(sv) U^T O  (gamma ~ U[0.08, 0.12], beta = 0.04, sv ~ U[0.4, 0.86])
             -> should pass P1 and P2
  symmetric  J = Q diag(ev) Q^T, ev ~ U[-0.6, 0.8]  -> should fail P1 (C = 0, so (E + 2C)/E = 1)
  slowmode   free + param * u u^T (u random unit; outlier eigenvalue near gamma + param; default 0.85; needs larger --T)
  cyclic     weighted cyclic shift, alternating weights b = 0.9, c = param (default 0.5): no shift, chain orthogonal for
             random starts, but |w_2|^2 -> b^2 c^2 != s^4 -> should fail P1c (norm profile)
  block      [[0, 1/2], [1/2, 0]] (+) (1/2) O_{d-2}, readout gh = e_1: random probes miss the block (P1 passes) but the
             readout's lever has c0 = 1/sqrt(5) while the free prediction is 0 -> should fail P2
The adjoint errors (1e-4) and the dstar reference (built from the same chain) are placeholders: the GATES that use them are
not tested here. Each run writes to <out>/<kind>_d<d>_n<n>_p<param>_T<T>/ with synth_meta.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_freeprobe as af  # noqa: E402


def haar(d, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def make_J(kind, d, rng, param):
    gh = rng.standard_normal(d); gh /= np.linalg.norm(gh)
    if kind == "symmetric":
        ev = rng.uniform(-0.6, 0.8, d); Qm = haar(d, rng)
        return (Qm * ev) @ Qm.T, gh
    if kind == "cyclic":
        assert d % 2 == 0
        wts = np.where(np.arange(d) % 2 == 0, 0.9, param)
        J = np.zeros((d, d)); J[(np.arange(d) + 1) % d, np.arange(d)] = wts
        return J, gh
    if kind == "block":
        J = np.zeros((d, d)); J[0, 1] = J[1, 0] = 0.5; J[2:, 2:] = 0.5 * haar(d - 2, rng)
        e1 = np.zeros(d); e1[0] = 1
        return J, e1
    gam, beta = 0.08 + 0.04 * rng.random(), 0.04
    sv = rng.uniform(0.4, 0.86, d); Om = haar(d, rng); Um = haar(d, rng)
    J = gam * np.eye(d) + beta * np.outer(gh, gh) + (Um * sv) @ Um.T @ Om
    if kind == "slowmode":
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        J = J + param * np.outer(u, u)
    else:
        assert kind == "free", kind
    return J, gh


def make(kind, d, n, K, T, seed, param, out, nr=1):
    """nr > 1 exercises the digits path: readouts 2..nr are further random unit vectors, written as digits_k = [0, 5, ..]
    into a dglever-like reference (G, AtG with 10 readout slots)."""
    rng = np.random.default_rng(seed); S = nr + K; V = S * (T + 1)
    GR = np.zeros((n, V, V)); W1 = np.zeros((n, S, d), np.float32); AT = np.zeros((n, S, d), np.float32)
    G = np.zeros((n, nr, d), np.float32); A_d = np.zeros((n, nr, d), np.float32)
    for i in range(n):
        J, gh = make_J(kind, d, rng, param)
        extra = [x / np.linalg.norm(x) for x in rng.standard_normal((nr - 1, d))]
        starts = [gh] + extra + [x / np.linalg.norm(x) for x in rng.standard_normal((K, d))]
        Jt = J.T; C = []
        for m, u in enumerate(starts):
            w = u.copy(); ch = [w]
            for _ in range(T):
                w = Jt @ w; ch.append(w)
            C += ch
            W1[i, m] = ch[1]; AT[i, m] = np.sum(ch[1:], 0)
        C = np.array(C); GR[i] = C @ C.T
        for r_ in range(nr):
            gn = 2.0 + rng.random(); G[i, r_] = gn * starts[r_]; A_d[i, r_] = gn * AT[i, r_]
    os.makedirs(out, exist_ok=True)
    uid = np.arange(n) + 100000
    np.savez_compressed(f"{out}/freeprobe.npz", gram=GR, w1=W1, aT=AT, u0=np.zeros((n, S, 1)), G=G, drift=np.zeros(n),
                        adjoint_rel_err=np.full((n, S), 1e-4), done=np.ones(n, bool), uid=uid)
    if nr == 1:
        json.dump(dict(cfg=dict(T=T, K=K)), open(f"{out}/meta.json", "w"))
        np.savez_compressed(f"{out}/dstar.npz", G=G[:, 0], A=A_d[:, 0], uid=uid)
    else:
        dk = [0, 5, 1, 2, 3, 4, 6, 7, 8, 9][:nr]
        Gref = np.zeros((n, 10, d), np.float32); Aref = np.zeros((n, 10, d), np.float32)
        Gref[:, dk] = G; Aref[:, dk] = A_d
        json.dump(dict(cfg=dict(T=T, K=K, task="digits", digits_k=dk)), open(f"{out}/meta.json", "w"))
        np.savez_compressed(f"{out}/dstar.npz", G=Gref, AtG=Aref, uid=uid)
    json.dump(dict(kind=kind, d=d, n=n, K=K, T=T, seed=seed, param=param, nr=nr, synthetic=True), open(f"{out}/synth_meta.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind"); ap.add_argument("d", type=int); ap.add_argument("n", type=int)
    ap.add_argument("param", type=float, nargs="?", default=None)
    ap.add_argument("--T", type=int, default=48); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="/tmp/synth_freeprobe"); ap.add_argument("--nr", type=int, default=1)
    a = ap.parse_args()
    param = a.param if a.param is not None else {"slowmode": 0.85, "cyclic": 0.5}.get(a.kind, 0.0)
    out = f"{a.out}/{a.kind}_d{a.d}_n{a.n}_p{param}_T{a.T}_nr{a.nr}"
    make(a.kind, a.d, a.n, 4, a.T, a.seed, param, out, nr=a.nr)
    r = af.main(out, f"{out}/dstar.npz", min_done=1)
    p1, p2 = r.get("P1", {}), r.get("P2", {})
    print("VERDICT", json.dumps(dict(kind=a.kind, d=a.d, n=a.n, param=param, T=a.T, gates=r["gates"]["PASS"],
                                     P1=p1.get("verdict"), P1_pass=p1.get("pass"), P2=p2.get("pass"),
                                     P2_c0_err=p2.get("c0", {}).get("med_err"), Q3=r.get("Q3", {}).get("reading"))))
