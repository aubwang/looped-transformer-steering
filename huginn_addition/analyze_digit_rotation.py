"""Digit-circle rotation pilot analysis (intervene_app.py `dgcollect`). Prespecified before the run; a 10-level
generalisation of the frozen weekday analysis (analyze_weekday.py, rules reviewed by GPT-6 Astra).
Usage: python -I huginn_addition/analyze_digit_rotation.py results/<dg_run> [--start a|b]

INTERPRETATION (frozen after Astra's review, before data): this is CARRY-CONDITIONED UNITS PREDICTION. The tens answer
token T at q is teacher-forced and encodes the carry, so carry information is supplied from initialisation; carry is
absorbed into the U effects (it is linear in the a, b, U main effects), so it contributes to the plane-defining centroids,
and carry x operand interactions can contribute to non-additivity. No claim about unassisted addition dynamics.
Qualification is digit-restricted accuracy with oracle T (full-vocab accuracy and digit mass also reported).

Design: units digits a ("start", first operand) x b ("offset") in 0..9, full factorial, x 2 contexts (template A/B);
answer U = (a + b) mod 10; carry = [a + b >= 10] is a linear function of the a, b, U main effects (absorbed). State =
recurrent state at q (the tens answer token, predicting U), every loop 0..96. Folds: fold = (a - b) mod 10 (each fold has
every a and every b once per context; answers appear twice each; both contexts of a cell share a fold).
--start b swaps the roles of a and b for the rotation test (secondary; primary = a).

0. BEHAVIOURAL QUALIFICATION (gate for the rotation verdict): digit-restricted argmax at loop 96 == U for >= 80% of all
   rows and >= 60% within every offset value. Otherwise verdict NOT-QUALIFIED (everything else descriptive).
1. Behaviour: accuracy by context and offset; digit mass; accuracy by loop.
2. Endpoint geometry (loop 96, fp32 final state `fin[:, 0]`; all rows, correct rows as diagnostic): OLS template + a + b + U (sum-to-zero), residual =
   state - intercept/template/a/b effects; top-20 PCs of training residuals; held-out R^2 of codes of U: canonical
   circle; the x3 circle (order 0,3,6,9,2,5,8,1,4,7); 5000 random cyclic orders (seeded; canonical rank among them);
   categorical; canonical line and best of 2000 random linear orders; 200 random rank-2 codes (centred, orthonormalised,
   circle-scaled). Ridge 1e-3 on code weights, intercept unpenalised.
3. Dynamics in a fixed plane (cross-fitted per fold): U-plane = top-2 PCs of the training adjusted answer centroids at
   loop 96, canonical order counter-clockwise; validity gate per fold: canonical cyclic order, every gap >= 0.3 x 2pi/10,
   s2/s1 >= 0.3. Held-out state at loop t centred by the training mean at loop t for its context, projected; phase in
   digit units by piecewise-linear interpolation between centroid angles; radius relative to the median loop-96 centroid
   radius. Conditional decoding of a, b, U per loop (other factors removed with training effects). Additive null per loop.
4. Rotation vs direct. Reliable samples: radius >= 0.3; JUMP if a later sample has radius < 0.3 or consecutive samples
   differ by > 2 digits; AMBIGUOUS-WINDING if |wrap(phase(t0) - start)| > 0.5; NO-ENDPOINT if |wrap(phase(96) - U)| > 0.5
   (the path does not end at the answer; cannot vote); else PRIMARY. traversed = unwrapped
   phase(96) - phase(t0); rotation prediction = offset - wrap(phase(t0) - start); direct = wrap(U - phase(t0)); wrap into
   [-5, 5); closer = rotation / direct / tie (|difference| < 0.25). Primary contrast: offsets 6..9 (rotation forward
   6..9 vs direct backward 4..1). Offset 5 is excluded (+-5 tie).
   Non-additivity (valid folds only): na(t) = held-out in-plane SSE of the additive prediction / mean held-out in-plane
   total variance over the last 16 loops; 3-loop moving average over the SIGNAL WINDOW (loops >= 1 with pooled median
   radius >= 0.3); statistic = window max - median na over the last 16 loops.
   VERDICT: NOT-QUALIFIED (gate 0 fails); INCONCLUSIVE if primary offset-6..9 rows < 40 (of 80), span < 7 folds or one
   context, or no signal window; CROSS-FADE-COMPATIBLE if statistic < 0.10; else ROTATION-LIKE if >= 2/3 of primary
   offset-6..9 rows are closer to rotation, DIRECT-ANGULAR if >= 2/3 closer to direct, else MIXED.
   Smooth angular motion alone does not establish rotational computation.
"""
from __future__ import annotations

import json
import sys

import numpy as np

NL = 10
R = 20
TAU = 2 * np.pi
HALF = NL / 2


def wrap(x):
    return ((np.asarray(x) + HALF) % NL) - HALF


def dummies(v, k):
    D = np.zeros((len(v), k - 1))
    for i, x in enumerate(v):
        if x < k - 1:
            D[i, x] = 1
        else:
            D[i, :] = -1
    return D


BLK = {"intercept": slice(0, 1), "template": slice(1, 2), "a": slice(2, 11), "b": slice(11, 20), "ans": slice(20, 29)}


def design(P):
    t = [0 if p["template"] == "A" else 1 for p in P]
    return np.hstack([np.ones((len(P), 1)), dummies(t, 2), dummies([p["start"] for p in P], NL),
                      dummies([p["offset"] for p in P], NL), dummies([p["ans"] for p in P], NL)])


def fit(P, X):
    D = design(P)
    B, *_ = np.linalg.lstsq(D, X, rcond=None)
    return D, B


def levels(B, name):
    e = B[BLK[name]]
    return np.vstack([e, -e.sum(0)])


def partial(D, B, names):
    keep = np.zeros(D.shape[1], bool)
    for nm in names:
        keep[BLK[nm]] = True
    return D[:, keep] @ B[keep]


def code_from_order(order):
    pos = np.zeros(NL)
    for i, d in enumerate(order):
        pos[d] = i
    return np.stack([np.cos(TAU * pos / NL), np.sin(TAU * pos / NL)], 1)


def sse_of(Ztr, Ytr, Zte, Yte, lam=1e-3):
    zm, ym = Ztr.mean(0), Ytr.mean(0)
    A = Ztr - zm
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    return float(((Yte - ((Zte - zm) @ W + ym)) ** 2).sum())


def plane_from_centroids(cent):
    cc = cent - cent.mean(0)
    _, sv, Vt = np.linalg.svd(cc, full_matrices=False)
    U = Vt[:2].T
    ang = np.angle((cc @ U) @ np.array([1, 1j]))
    if np.sum(np.sin(np.diff(np.concatenate([ang, ang[:1]])))) < 0:
        U[:, 1] *= -1
        ang = np.angle((cc @ U) @ np.array([1, 1j]))
    rel = (ang - ang[0]) % TAU
    gaps = np.diff(np.concatenate([rel, [TAU]]))
    canon = bool(np.all(np.diff(rel) > 0))
    valid = bool(canon and gaps.min() >= 0.3 * TAU / NL and sv[1] / sv[0] >= 0.3)
    crad = float(np.median(np.abs((cc @ U) @ np.array([1, 1j]))))
    return U, ang, crad, valid, dict(gaps_deg=np.degrees(gaps).round(1).tolist(), sv_ratio=float(sv[1] / sv[0]), canonical_cyclic=canon)


def to_unit(th, ang):
    a0 = ang[0]
    rel = (ang - a0) % TAU
    rel_ext = np.concatenate([rel, [TAU]])
    x = (np.asarray(th) - a0) % TAU
    k = np.clip(np.searchsorted(rel_ext, x, side="right") - 1, 0, NL - 1)
    w = (x - rel_ext[k]) / (rel_ext[k + 1] - rel_ext[k])
    return (k + w) % NL


def traverse(ph, rad, start, off, ans):
    ok = rad >= 0.3
    if not ok.any() or not ok[-1]:
        return dict(reliable=False)
    t0 = int(np.argmax(ok))
    seg = ph[t0:][ok[t0:]]
    steps = wrap(np.diff(seg))
    jump = bool((~ok[t0:]).any() or (np.abs(steps) > 2.0).any())
    trav = float(steps.sum())
    p0 = float(seg[0])
    dev0 = float(wrap(p0 - start))
    rot = off - dev0
    direct = float(wrap(ans - p0))
    cls = "tie" if abs(abs(trav - rot) - abs(trav - direct)) < 0.25 else ("rotation" if abs(trav - rot) < abs(trav - direct) else "direct")
    end_ok = abs(float(wrap(float(seg[-1]) - ans))) <= 0.5
    path = "jump" if jump else ("ambiguous_winding" if abs(dev0) > 0.5 else ("no_endpoint" if not end_ok else "primary"))
    return dict(reliable=True, t0=t0, phase_t0=round(p0, 2), dev0=round(dev0, 2), traversed=round(trav, 2),
                pred_rotation=round(rot, 2), pred_direct=round(direct, 2), closer=cls, path=path)


def main(run, start_role="a"):
    z = np.load(f"{run}/dg.npz", allow_pickle=True)
    P = json.load(open(f"{run}/problems.json"))
    assert z["done"].all(), "incomplete primary collection"
    n = len(P)
    traj = z["traj"].astype(np.float64)
    nT = traj.shape[1]
    SL = list(range(nT))
    X96 = z["fin"][:, 0].astype(np.float64)
    assert np.allclose(X96, traj[:, -1], rtol=1e-3, atol=1e-3 * np.abs(X96).max())
    A_ = np.array([p["start"] for p in P]); B_ = np.array([p["offset"] for p in P]); ans = np.array([p["ans"] for p in P])
    start, off = (A_, B_) if start_role == "a" else (B_, A_)
    tmpl = np.array([p["template"] for p in P]); folds = (A_ - B_) % NL
    out = {"start_role": start_role}
    # 0/1. behaviour
    corr = z["ro_lp"][:, -1].argmax(-1) == ans
    full_ok = z["ro_am"][:, -1] == z["dig_tokens"][ans]
    by_off = {int(o): float(corr[B_ == o].mean()) for o in range(NL)}
    qualified = bool(corr.mean() >= 0.8 and min(by_off.values()) >= 0.6)
    out["behaviour"] = dict(acc=float(corr.mean()), acc_full=float(full_ok.mean()), acc_by_offset_b=by_off,
                            acc_by_context={t: float(corr[tmpl == t].mean()) for t in ("A", "B")},
                            median_digit_mass=float(np.median(z["ro_mass"][:, -1])), qualified=qualified,
                            acc_by_loop={int(k): round(float((z["ro_lp"][:, i].argmax(-1) == ans).mean()), 3) for i, k in enumerate(z["ro_loops"])})
    late = z["drift"][:, -8:, :].mean(1).max(-1)
    out["late8_drift"] = dict(median=float(np.median(late)), max=float(late.max()))
    # 2. geometry
    rng = np.random.default_rng(0)
    canon = tuple(range(NL)); x3 = tuple((3 * i) % NL for i in range(NL))
    rand_orders = [tuple([0] + list(rng.permutation(np.arange(1, NL)))) for _ in range(5000)]
    lin_orders = [tuple(range(NL))] + [tuple(rng.permutation(NL)) for _ in range(2000)]
    scale = np.linalg.norm(code_from_order(canon))
    rand_codes = []
    for i in range(200):
        c = np.random.default_rng(100 + i).normal(size=(NL, 2)); c -= c.mean(0); q_, _ = np.linalg.qr(c)
        rand_codes.append(q_ * scale / np.linalg.norm(q_))

    def geometry(mask):
        sse = dict(canon=0.0, x3=0.0, ro=np.zeros(len(rand_orders)), cat=0.0, lin=np.zeros(len(lin_orders)), rc=np.zeros(200))
        sst = 0.0
        for f in range(NL):
            tr, te = (folds != f) & mask, (folds == f) & mask
            if te.sum() == 0 or tr.sum() < 40:
                continue
            Ptr = [P[i] for i in np.where(tr)[0]]; Pte = [P[i] for i in np.where(te)[0]]
            Dtr, B = fit(Ptr, X96[tr])
            nuis = ["intercept", "template", "a", "b"]
            rtr = X96[tr] - partial(Dtr, B, nuis); rte = X96[te] - partial(design(Pte), B, nuis)
            mu = rtr.mean(0); V = np.linalg.svd(rtr - mu, full_matrices=False)[2][:R].T
            Ytr, Yte = (rtr - mu) @ V, (rte - mu) @ V
            a, b = ans[tr], ans[te]
            c = code_from_order(canon); sse["canon"] += sse_of(c[a], Ytr, c[b], Yte)
            c = code_from_order(x3); sse["x3"] += sse_of(c[a], Ytr, c[b], Yte)
            for j, o in enumerate(rand_orders):
                c = code_from_order(o); sse["ro"][j] += sse_of(c[a], Ytr, c[b], Yte)
            oh = np.eye(NL)[:, :NL - 1]; sse["cat"] += sse_of(oh[a], Ytr, oh[b], Yte)
            for j, o in enumerate(lin_orders):
                c = np.zeros((NL, 1)); c[list(o), 0] = np.arange(NL); sse["lin"][j] += sse_of(c[a], Ytr, c[b], Yte)
            for j, c in enumerate(rand_codes):
                sse["rc"][j] += sse_of(c[a], Ytr, c[b], Yte)
            sst += float(((Yte - Ytr.mean(0)) ** 2).sum())
        r2 = lambda e: 1 - np.asarray(e) / sst
        rc_ = r2(sse["canon"]); ro = r2(sse["ro"]); rcd = r2(sse["rc"])
        return dict(n_rows=int(mask.sum()), R2_canonical_circle=float(rc_), R2_x3_circle=float(r2(sse["x3"])),
                    canonical_rank_among_5000_random_cyclic=int((ro > rc_).sum() + 1),
                    R2_random_cyclic=dict(median=float(np.median(ro)), q99=float(np.quantile(ro, 0.99)), max=float(ro.max())),
                    R2_categorical=float(r2(sse["cat"])), R2_line_canonical=float(r2(sse["lin"][0])),
                    R2_line_best_of_2001=float(r2(sse["lin"]).max()),
                    R2_random_rank2=dict(median=float(np.median(rcd)), q95=float(np.quantile(rcd, 0.95)), max=float(rcd.max())))
    out["geometry_all_rows"] = geometry(np.ones(n, bool))
    out["geometry_correct_rows_diagnostic"] = geometry(corr)
    # 3/4. dynamics
    phase = np.full((n, nT), np.nan); radius = np.full((n, nT), np.nan)
    nphase = np.full((n, nT), np.nan); nradius = np.full((n, nT), np.nan)
    res_ss = np.zeros(nT); tot_ip = np.zeros(nT)
    acc = {k: np.zeros(nT) for k in ("a", "b", "ans")}
    plane_info = {}
    for f in range(NL):
        tr, te = folds != f, folds == f
        Ptr = [P[i] for i in np.where(tr)[0]]; Pte = [P[i] for i in np.where(te)[0]]
        Dtr, Dte = design(Ptr), design(Pte)
        _, B96 = fit(Ptr, X96[tr])
        U, ang, crad, valid, info = plane_from_centroids(levels(B96, "ans"))
        plane_info[f] = dict(valid=valid, **info)
        for ti in range(nT):
            Xt = traj[:, ti]
            _, B = fit(Ptr, Xt[tr])
            add_te = Dte @ B
            for t in ("A", "B"):
                m_tr = tr & (tmpl == t); loc = tmpl[te] == t
                mean_t = Xt[m_tr].mean(0)
                Y = (Xt[te][loc] - mean_t) @ U; Yn = (add_te[loc] - mean_t) @ U
                idx = np.where(te)[0][loc]
                if valid:
                    res_ss[ti] += float(((Y - Yn) ** 2).sum()); tot_ip[ti] += float(((Y - Y.mean(0)) ** 2).sum())
                    zc, zn = Y @ np.array([1, 1j]), Yn @ np.array([1, 1j])
                    radius[idx, ti] = np.abs(zc) / crad; phase[idx, ti] = to_unit(np.angle(zc), ang)
                    nradius[idx, ti] = np.abs(zn) / crad; nphase[idx, ti] = to_unit(np.angle(zn), ang)
            if ti % 4 == 0 or ti == nT - 1:  # conditional decoding diagnostics (every 4th loop to save time)
                for name, lab in (("a", A_), ("b", B_), ("ans", ans)):
                    others = [k for k in ("intercept", "template", "a", "b", "ans") if k != name]
                    adj_tr = Xt[tr] - partial(Dtr, B, others); adj_te = Xt[te] - partial(Dte, B, others)
                    mu = adj_tr.mean(0); V = np.linalg.svd(adj_tr - mu, full_matrices=False)[2][:R].T
                    C = levels(B, name)  # adj = target effect + noise; compare uncentred observations to uncentred effects
                    pred = (((adj_te[:, None] - C[None]) @ V) ** 2).sum(-1).argmin(1)
                    acc[name][ti] += (pred == lab[te]).sum()
    for k in acc:
        acc[k] /= n
    na = res_ss / max(float(np.mean(tot_ip[-16:])), 1e-30)
    med_rad = np.array([np.nanmedian(radius[:, i]) if np.isfinite(radius[:, i]).any() else np.nan for i in range(nT)])
    window = (np.arange(nT) >= 1) & (np.nan_to_num(med_rad) >= 0.3)
    sm = np.convolve(na, np.ones(3) / 3, mode="same"); sm[0], sm[-1] = na[0], na[-1]
    na_stat = float(sm[window].max() - np.median(na[-16:])) if window.any() else float("nan")
    cdist = lambda x, y: np.minimum(np.abs(x - y) % NL, NL - np.abs(x - y) % NL)
    hit = lambda lab: np.where(np.isnan(phase), np.nan, (cdist(phase, lab[:, None]) < 0.5).astype(float))
    with np.errstate(all="ignore"):
        ip_s = np.nanmean(hit(start), 0); ip_a = np.nanmean(hit(ans), 0)
    show = [i for i in range(nT) if i % 4 == 0 or i == nT - 1]
    out["plane_validity"] = plane_info
    out["dynamics_by_loop"] = {i: dict(dec_a=round(float(acc["a"][i]), 3), dec_b=round(float(acc["b"][i]), 3), dec_ans=round(float(acc["ans"][i]), 3),
                                       inplane_start=None if not np.isfinite(ip_s[i]) else round(float(ip_s[i]), 3),
                                       inplane_ans=None if not np.isfinite(ip_a[i]) else round(float(ip_a[i]), 3),
                                       median_radius=None if not np.isfinite(med_rad[i]) else round(float(med_rad[i]), 3),
                                       nonadditive_vs_late=round(float(na[i]), 3)) for i in show}
    rows = []
    for i in range(n):
        if np.isnan(phase[i]).all():
            rows.append(dict(i=i, reliable=False, reason="invalid plane")); continue
        ob = traverse(phase[i], radius[i], start[i], off[i], ans[i]); nb = traverse(nphase[i], nradius[i], start[i], off[i], ans[i])
        rows.append(dict(i=i, offset=int(off[i]), start=int(start[i]), ans=int(ans[i]), template=str(tmpl[i]), fold=int(folds[i]),
                         correct=bool(corr[i]), **ob, null=nb))
    rel = [r for r in rows if r.get("reliable")]

    def tally(rs):
        return dict(n=len(rs), median_traversed=float(np.median([r["traversed"] for r in rs])) if rs else None,
                    rotation=sum(r["closer"] == "rotation" for r in rs), direct=sum(r["closer"] == "direct" for r in rs),
                    tie=sum(r["closer"] == "tie" for r in rs),
                    null_median_traversed=float(np.median([r["null"]["traversed"] for r in rs if r["null"].get("reliable")]))
                    if any(r["null"].get("reliable") for r in rs) else None)
    prim = [r for r in rel if r["offset"] >= 6 and r["path"] == "primary"]
    cov = dict(n=len(prim), folds=len({r["fold"] for r in prim}), templates=len({r["template"] for r in prim}))
    t69 = tally(prim)
    if not qualified:
        verdict = "NOT-QUALIFIED"
    elif cov["n"] < 40 or cov["folds"] < 7 or cov["templates"] < 2 or not np.isfinite(na_stat):
        verdict = "INCONCLUSIVE"
    elif na_stat < 0.10:
        verdict = "CROSS-FADE-COMPATIBLE"
    elif t69["rotation"] >= 2 / 3 * t69["n"]:
        verdict = "ROTATION-LIKE"
    elif t69["direct"] >= 2 / 3 * t69["n"]:
        verdict = "DIRECT-ANGULAR"
    else:
        verdict = "MIXED"
    out["rotation_test"] = dict(
        verdict=verdict, nonadditive_stat=na_stat, signal_window=[int(i) for i in np.where(window)[0]], coverage_6_9=cov,
        primary_offsets_6_9=t69, path_counts={k: sum(r["path"] == k for r in rel) for k in ("primary", "jump", "ambiguous_winding", "no_endpoint")},
        primary_by_offset={o: tally([r for r in rel if r["offset"] == o and r["path"] == "primary"]) for o in range(NL)},
        jump_by_offset={o: tally([r for r in rel if r["offset"] == o and r["path"] == "jump"]) for o in range(NL)},
        ambiguous_by_offset={o: tally([r for r in rel if r["offset"] == o and r["path"] == "ambiguous_winding"]) for o in range(NL)},
        caveat="smooth angular motion alone is not rotational computation; no operation-conditioned transition map is fitted")
    json.dump(dict(summary=out, rows=rows), open(f"{run}/analysis_summary_start_{start_role}.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    args = sys.argv[1:]
    role = "a"
    if "--start" in args:
        role = args[args.index("--start") + 1]
    main(args[0].rstrip("/"), role)
