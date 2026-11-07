"""Weekday pilot analysis (intervene_app.py `wkcollect`). Prespecified before seeing data (revised after Astra's code
review, still before data).
Usage: python -I huginn_addition/analyze_weekday.py results/<wk_run>

Design: 7 starts x 7 offsets (1..7) x 2 templates; answer = (start + offset) mod 7 (Latin square). State = recurrent state
at the answer-prediction position q (last prompt token), saved at every loop 0..96. Folds: 7 transversal folds,
fold(start, offset) = (2 start + offset) mod 7; each fold holds one cell per start, per offset and per answer (x 2
templates); both templates of a cell share a fold. The primary dataset must be complete (all `done`), else abort.

1. Behaviour: accuracy (argmax over the 7 day tokens; full-vocab argmax == answer token) at loop 96 by template and
   offset; day-token mass; 7-way accuracy by loop.
2. Endpoint geometry (loop 96, all rows; correct-only rows repeated as a diagnostic). Per training split: OLS of the state
   on template + start + offset + answer (sum-to-zero dummies); additive effects are identifiable in this Latin square,
   interactions are not. Residual = state - (intercept + template + start + offset effects of that full fit); the answer
   effects stay in the residual. Work in the top-R PCs (R=20) of the training residuals. Scores = held-out R^2 of the
   residual (PC space) from a code of the answer, ridge 1e-3 on the code weights only (intercept unpenalised):
   - circle code under each of the 360 undirected cyclic orders (canonical rank and best order; rank is descriptive,
     not a calibrated test);
   - categorical (one-hot) answer means; line (canonical rank code; best of all 5040 permutations, reversals duplicated);
   - 200 random rank-2 codes (Gaussian, centred and orthonormalised over the 7 days, scaled like the circle code).
   Fraction of categorical held-out R^2 captured by the canonical circle.
3. Dynamics in a FIXED plane (cross-fitted per fold): U = top-2 PCs of the training adjusted answer centroids at loop 96,
   oriented so the canonical order runs counter-clockwise. PHASE VALIDITY GATE per fold: centroid angles in canonical
   cyclic order, every adjacent gap >= 0.3 x 2pi/7, and singular-value ratio s2/s1 >= 0.3; rows of failing folds get no
   phase. Held-out state at loop t: centred by the training mean at loop t for its template, projected on U. Day-phase =
   angle mapped to fractional day units by piecewise-linear interpolation between the training centroid angles.
   Radius relative to the median loop-96 centroid radius.
   Decoding diagnostics per loop (CONDITIONAL: they use the held-out rows' known other-factor labels): for target
   T in {start, answer}, held-out state minus the training intercept/template and the two other factors' effects, nearest
   T-effect in the top-20 PCs of the training data with the same other-factor adjustment (target signal retained).
   In-plane: round(day-phase) vs start and vs answer.
   ADDITIVE NULL: at each loop t, the training additive fit (template + start + offset + answer effects at loop t)
   predicts each held-out row's state; its plane projection gives an additive-null phase path. Additive effects can only
   produce paths built from fading/growing start, offset and answer components, not a start x offset interaction.
   Report per loop the fraction of held-out in-plane variance not explained by the additive prediction.
4. Rotation vs direct relaxation. Per held-out row with valid phase: reliable samples = radius >= 0.3; t0 = first
   reliable sample. Path class JUMP if any later sample has radius < 0.3 or two consecutive samples differ by > 1.5 days;
   AMBIGUOUS-WINDING if |wrap(phase(t0) - start)| > 0.5 (start branch not established); else PRIMARY.
   traversed = unwrapped phase(96) - phase(t0) (days). Predictions: rotation = offset - wrap(phase(t0) - start);
   direct = wrap(answer - phase(t0)); wrap into (-3.5, 3.5]. closer = rotation / direct / tie (|difference| < 0.25).
   Primary contrast: offsets 5, 6, 7 among PRIMARY rows. Jump and ambiguous rows are reported separately.
   DECISION RULE (fixed after synthetic validation and Astra's second review, before inspecting real data):
   - Non-additivity is computed over VALID-PLANE folds only: na(t) = held-out in-plane SSE of the additive prediction at
     loop t divided by the held-out in-plane total variance averaged over the last 16 loops (a FIXED late denominator, so
     weak early signal cannot inflate it), 3-loop moving average, restricted to the SIGNAL WINDOW: loops >= 1 whose
     pooled median in-plane radius (valid folds) is >= 0.3. Statistic = max over the signal window minus the median of
     na over the last 16 loops. (A per-loop-fraction version falsely flagged a weak-early-signal cross-fade world, 0.116;
     see the synthetic checks in the README.)
   - Coverage: primary offset-5..7 rows must number >= 21 (of 42), span >= 5 folds and both templates; else INCONCLUSIVE.
   - If coverage holds: statistic < 0.10 -> CROSS-FADE-COMPATIBLE (paths not interpretable as angular motion); else
     ROTATION-LIKE if >= 2/3 of primary offset-5..7 rows are closer to rotation, DIRECT-ANGULAR if >= 2/3 closer to
     direct, else MIXED. The
   additive-null path's own traversal is reported but is NOT a discriminator (on synthetic rotation data the additive
   fit's projected path also rotates). Smooth angular motion alone does not establish rotational computation (no
   operation-conditioned transition map is fitted here).
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

R = 20
TAU = 2 * np.pi


def fold_of(p):
    return (2 * p["start"] + p["offset"]) % 7


def wrap(x):
    return ((np.asarray(x) + 3.5) % 7) - 3.5


def dummies(v, k):  # sum-to-zero coding, k levels -> k-1 columns
    D = np.zeros((len(v), k - 1))
    for i, x in enumerate(v):
        if x < k - 1:
            D[i, x] = 1
        else:
            D[i, :] = -1
    return D


BLK = {"intercept": slice(0, 1), "template": slice(1, 2), "start": slice(2, 8), "offset": slice(8, 14), "answer": slice(14, 20)}


def design(P):
    t = [0 if p["template"] == "A" else 1 for p in P]
    return np.hstack([np.ones((len(P), 1)), dummies(t, 2), dummies([p["start"] for p in P], 7),
                      dummies([p["offset"] - 1 for p in P], 7), dummies([p["ans"] for p in P], 7)])


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


def cyclic_orders():
    return [(0,) + perm for perm in itertools.permutations(range(1, 7)) if perm[0] < perm[-1]]  # 360


def code_from_order(order):
    pos = np.zeros(7)
    for i, d in enumerate(order):
        pos[d] = i
    return np.stack([np.cos(TAU * pos / 7), np.sin(TAU * pos / 7)], 1)


def sse_of(Ztr, Ytr, Zte, Yte, lam=1e-3):
    """Ridge map code -> PC-space residual, intercept unpenalised (centre on training means). Returns held-out SSE."""
    zm, ym = Ztr.mean(0), Ytr.mean(0)
    A = Ztr - zm
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    pred = (Zte - zm) @ W + ym
    return float(((Yte - pred) ** 2).sum())


def plane_from_centroids(cent):
    cc = cent - cent.mean(0)
    _, sv, Vt = np.linalg.svd(cc, full_matrices=False)
    U = Vt[:2].T
    ang = np.angle((cc @ U) @ np.array([1, 1j]))
    if np.sum(np.sin(np.diff(np.concatenate([ang, ang[:1]])))) < 0:  # canonical order counter-clockwise
        U[:, 1] *= -1
        ang = np.angle((cc @ U) @ np.array([1, 1j]))
    rel = (ang - ang[0]) % TAU
    gaps = np.diff(np.concatenate([rel, [TAU]]))
    valid = bool(np.all(np.diff(rel) > 0) and gaps.min() >= 0.3 * TAU / 7 and sv[1] / sv[0] >= 0.3)
    crad = float(np.median(np.abs((cc @ U) @ np.array([1, 1j]))))
    return U, ang, crad, valid, dict(gaps_deg=np.degrees(gaps).round(1).tolist(), sv_ratio=float(sv[1] / sv[0]),
                                     canonical_cyclic=bool(np.all(np.diff(rel) > 0)))


def to_day(th, ang):
    """Map angles to fractional day units using the 7 centroid angles (assumed in canonical CCW order: gate)."""
    a0 = ang[0]
    rel = (ang - a0) % TAU
    rel_ext = np.concatenate([rel, [TAU]])
    x = (np.asarray(th) - a0) % TAU
    k = np.clip(np.searchsorted(rel_ext, x, side="right") - 1, 0, 6)
    w = (x - rel_ext[k]) / (rel_ext[k + 1] - rel_ext[k])
    return (k + w) % 7


def traverse(ph, rad, start, off, ans):
    ok = rad >= 0.3
    if not ok.any() or not ok[-1]:
        return dict(reliable=False)
    t0 = int(np.argmax(ok))
    seg = ph[t0:][ok[t0:]]
    steps = wrap(np.diff(seg))
    jump = bool((~ok[t0:]).any() or (np.abs(steps) > 1.5).any())
    trav = float(steps.sum())
    p0 = float(seg[0])
    dev0 = float(wrap(p0 - start))
    rot = off - dev0
    direct = float(wrap(ans - p0))
    cls = "tie" if abs(abs(trav - rot) - abs(trav - direct)) < 0.25 else ("rotation" if abs(trav - rot) < abs(trav - direct) else "direct")
    path = "jump" if jump else ("ambiguous_winding" if abs(dev0) > 0.5 else "primary")
    return dict(reliable=True, t0=t0, phase_t0=round(p0, 2), dev0=round(dev0, 2), traversed=round(trav, 2),
                pred_rotation=round(rot, 2), pred_direct=round(direct, 2), closer=cls, path=path)


def main(run):
    z = np.load(f"{run}/wk.npz", allow_pickle=True)
    P = json.load(open(f"{run}/problems.json"))
    assert z["done"].all(), "incomplete primary collection"
    n = len(P)
    SL = list(z["save_loops"])
    assert SL == list(range(SL[0], SL[-1] + 1))
    traj = z["traj"].astype(np.float64)
    nT = len(SL)
    X96 = traj[:, -1]
    ans = np.array([p["ans"] for p in P]); start = np.array([p["start"] for p in P]); off = np.array([p["offset"] for p in P])
    tmpl = np.array([p["template"] for p in P]); folds = np.array([fold_of(p) for p in P])
    out = {}
    # 1. behaviour
    pred7 = z["ro_lp"][:, -1].argmax(-1)
    full_ok = z["ro_am"][:, -1] == z["day_tokens"][ans]
    corr = pred7 == ans
    out["behaviour"] = {t: dict(acc7=float(corr[tmpl == t].mean()), acc_full=float(full_ok[tmpl == t].mean()),
                                acc7_by_offset={int(o): float(corr[(tmpl == t) & (off == o)].mean()) for o in range(1, 8)},
                                median_day_mass=float(np.median(z["ro_mass"][tmpl == t, -1]))) for t in ("A", "B")}
    out["late8_drift"] = dict(median=float(np.nanmedian(z["late8"])), max=float(np.nanmax(z["late8"])),
                              n_extended=int(len(z["ext_idx"])), n_ext_done=int(z["ext_done"].sum()) if len(z["ext_idx"]) else 0)
    out["acc7_by_loop"] = {int(SL[i]): round(float((z["ro_lp"][:, i].argmax(-1) == ans).mean()), 3) for i in range(0, nT, 4)}

    # 2. endpoint geometry
    orders = cyclic_orders()
    canon_idx = orders.index(tuple(range(7)))
    lin_orders = list(itertools.permutations(range(7)))
    canon_lin = lin_orders.index(tuple(range(7)))
    circ_scale = np.linalg.norm(code_from_order(tuple(range(7))))
    rand_codes = []
    for i in range(200):
        c = np.random.default_rng(100 + i).normal(size=(7, 2)); c -= c.mean(0)
        q_, _ = np.linalg.qr(c); rand_codes.append(q_ * circ_scale / np.linalg.norm(q_))

    def geometry(mask):
        sse = dict(circle=np.zeros(len(orders)), cat=0.0, line=np.zeros(len(lin_orders)), rand=np.zeros(len(rand_codes)))
        sst = 0.0
        for f in range(7):
            tr, te = (folds != f) & mask, (folds == f) & mask
            if te.sum() == 0:
                continue
            Ptr = [P[i] for i in np.where(tr)[0]]; Pte = [P[i] for i in np.where(te)[0]]
            Dtr, B = fit(Ptr, X96[tr])
            nuis = ["intercept", "template", "start", "offset"]
            res_tr = X96[tr] - partial(Dtr, B, nuis); res_te = X96[te] - partial(design(Pte), B, nuis)
            mu = res_tr.mean(0)
            V = np.linalg.svd(res_tr - mu, full_matrices=False)[2][:R].T
            Ytr, Yte = (res_tr - mu) @ V, (res_te - mu) @ V
            a_tr, a_te = ans[tr], ans[te]
            for j, o in enumerate(orders):
                c = code_from_order(o); sse["circle"][j] += sse_of(c[a_tr], Ytr, c[a_te], Yte)
            oh = np.eye(7)[:, :6]
            sse["cat"] += sse_of(oh[a_tr], Ytr, oh[a_te], Yte)
            for j, o in enumerate(lin_orders):
                c = np.zeros((7, 1)); c[list(o), 0] = np.arange(7); sse["line"][j] += sse_of(c[a_tr], Ytr, c[a_te], Yte)
            for j, c in enumerate(rand_codes):
                sse["rand"][j] += sse_of(c[a_tr], Ytr, c[a_te], Yte)
            sst += float(((Yte - Ytr.mean(0)) ** 2).sum())
        r2 = lambda e: 1 - np.asarray(e) / sst
        circ, rnd = r2(sse["circle"]), r2(sse["rand"])
        return dict(n_rows=int(mask.sum()), R2_canonical_circle=float(circ[canon_idx]),
                    canonical_rank_of_360=int((circ > circ[canon_idx]).sum() + 1),
                    top5_orders=[(list(orders[j]), round(float(circ[j]), 4)) for j in np.argsort(-circ)[:5]],
                    R2_categorical=float(r2(sse["cat"])), R2_line_canonical=float(r2(sse["line"][canon_lin])),
                    R2_line_best_of_5040=float(r2(sse["line"]).max()),
                    R2_random_rank2=dict(median=float(np.median(rnd)), q95=float(np.quantile(rnd, 0.95)), max=float(rnd.max())),
                    circle_over_categorical=float(circ[canon_idx] / r2(sse["cat"])) if r2(sse["cat"]) > 0 else None)
    out["geometry_all_rows"] = geometry(np.ones(n, bool))
    out["geometry_correct_rows_diagnostic"] = geometry(corr)

    # 3./4. dynamics
    phase = np.full((n, nT), np.nan); radius = np.full((n, nT), np.nan)
    nphase = np.full((n, nT), np.nan); nradius = np.full((n, nT), np.nan)
    resid_frac = np.zeros(nT); tot_ip = np.zeros(nT)
    acc = {"start": np.zeros(nT), "answer": np.zeros(nT)}
    plane_info = {}
    for f in range(7):
        tr, te = folds != f, folds == f
        Ptr = [P[i] for i in np.where(tr)[0]]; Pte = [P[i] for i in np.where(te)[0]]
        Dtr, Dte = design(Ptr), design(Pte)
        _, B96 = fit(Ptr, X96[tr])
        U, ang, crad, valid, info = plane_from_centroids(levels(B96, "answer"))
        plane_info[f] = dict(valid=valid, **info)
        for ti in range(nT):
            Xt = traj[:, ti]
            _, B = fit(Ptr, Xt[tr])
            add_te = Dte @ B  # additive-null prediction for held-out rows at this loop
            for t in ("A", "B"):
                m_tr = tr & (tmpl == t); m_te_local = tmpl[te] == t
                mean_t = Xt[m_tr].mean(0)
                Y = (Xt[te][m_te_local] - mean_t) @ U
                Yn = (add_te[m_te_local] - mean_t) @ U
                idx = np.where(te)[0][m_te_local]
                if valid:  # non-additivity and phases from valid-plane folds only
                    resid_frac[ti] += float(((Y - Yn) ** 2).sum()); tot_ip[ti] += float(((Y - Y.mean(0)) ** 2).sum())
                    zc, zn = Y @ np.array([1, 1j]), Yn @ np.array([1, 1j])
                    radius[idx, ti] = np.abs(zc) / crad; phase[idx, ti] = to_day(np.angle(zc), ang)
                    nradius[idx, ti] = np.abs(zn) / crad; nphase[idx, ti] = to_day(np.angle(zn), ang)
            # conditional decoding diagnostics
            for name, lab in (("start", start), ("answer", ans)):
                others = [k for k in ("intercept", "template", "start", "offset", "answer") if k != name]
                adj_tr = Xt[tr] - partial(Dtr, B, others); adj_te = Xt[te] - partial(Dte, B, others)
                mu = adj_tr.mean(0)
                V = np.linalg.svd(adj_tr - mu, full_matrices=False)[2][:R].T
                C = levels(B, name)
                pred = ((((adj_te - mu)[:, None] - (C - C.mean(0))[None]) @ V) ** 2).sum(-1).argmin(1)
                acc[name][ti] += (pred == lab[te]).sum()
    for k in acc:
        acc[k] /= n
    resid_frac = resid_frac / max(float(np.mean(tot_ip[-16:])), 1e-30)  # fixed late denominator
    cd = lambda a, b: np.minimum(np.abs(a - b) % 7, 7 - np.abs(a - b) % 7)
    hit = lambda lab: np.where(np.isnan(phase), np.nan, (cd(phase, lab[:, None]) < 0.5).astype(float))
    ip_start = np.nanmean(hit(start), 0); ip_ans = np.nanmean(hit(ans), 0)
    med_rad = np.nanmedian(radius, 0)
    window = (np.array(SL) >= 1) & (med_rad >= 0.3)
    sm = np.convolve(resid_frac, np.ones(3) / 3, mode="same"); sm[0], sm[-1] = resid_frac[0], resid_frac[-1]
    na_stat = float(sm[window].max() - np.median(resid_frac[-16:])) if window.any() else float("nan")
    na_loop = int(np.array(SL)[window][int(sm[window].argmax())]) if window.any() else None
    show = list(range(0, 17)) + list(range(20, nT, 4)) + [nT - 1]
    out["plane_validity"] = plane_info
    out["dynamics_by_loop"] = {int(SL[i]): dict(dec_start=round(float(acc["start"][i]), 3), dec_answer=round(float(acc["answer"][i]), 3),
                                                inplane_start=round(float(ip_start[i]), 3), inplane_answer=round(float(ip_ans[i]), 3),
                                                median_radius=round(float(np.nanmedian(radius[:, i])), 3),
                                                inplane_nonadditive_vs_late=round(float(resid_frac[i]), 3)) for i in sorted(set(show))}
    rows = []
    for i in range(n):
        if np.isnan(phase[i]).all():
            rows.append(dict(i=i, offset=int(off[i]), reliable=False, reason="invalid plane")); continue
        ob = traverse(phase[i], radius[i], start[i], off[i], ans[i])
        nb = traverse(nphase[i], nradius[i], start[i], off[i], ans[i])
        rows.append(dict(i=i, offset=int(off[i]), template=str(tmpl[i]), start=int(start[i]), answer=int(ans[i]), correct=bool(corr[i]),
                         **ob, null=nb))
    rel = [r for r in rows if r.get("reliable")]

    def tally(rs):
        return dict(n=len(rs), median_traversed=float(np.median([r["traversed"] for r in rs])) if rs else None,
                    rotation=sum(r["closer"] == "rotation" for r in rs), direct=sum(r["closer"] == "direct" for r in rs),
                    tie=sum(r["closer"] == "tie" for r in rs),
                    null_median_traversed=float(np.median([r["null"]["traversed"] for r in rs if r["null"].get("reliable")]))
                    if any(r["null"].get("reliable") for r in rs) else None,
                    null_rotation=sum(r["null"].get("closer") == "rotation" for r in rs))
    out["rotation_test"] = dict(
        n_rows=n, n_reliable=len(rel), path_counts={k: sum(r["path"] == k for r in rel) for k in ("primary", "jump", "ambiguous_winding")},
        primary_by_offset={o: tally([r for r in rel if r["offset"] == o and r["path"] == "primary"]) for o in range(1, 8)},
        primary_offsets_5_6_7=tally([r for r in rel if r["offset"] >= 5 and r["path"] == "primary"]),
        jump_by_offset={o: tally([r for r in rel if r["offset"] == o and r["path"] == "jump"]) for o in range(1, 8)},
        ambiguous_by_offset={o: tally([r for r in rel if r["offset"] == o and r["path"] == "ambiguous_winding"]) for o in range(1, 8)},
        t0_loop_median=float(np.median([SL[r["t0"]] for r in rel])) if rel else None,
        nonadditive_stat=na_stat, nonadditive_peak_loop=na_loop, signal_window_loops=[int(x) for x in np.array(SL)[window]],
        caveat="smooth angular motion alone is not rotational computation; no operation-conditioned transition map is fitted")
    rt = out["rotation_test"]; p567 = rt["primary_offsets_5_6_7"]
    prim567 = [r for r in rel if r["offset"] >= 5 and r["path"] == "primary"]
    cov = dict(n=len(prim567), folds=len({int(folds[r["i"]]) for r in prim567}), templates=len({r["template"] for r in prim567}))
    rt["coverage_567"] = cov
    if cov["n"] < 21 or cov["folds"] < 5 or cov["templates"] < 2 or not np.isfinite(na_stat):
        verdict = "INCONCLUSIVE"
    elif na_stat < 0.10:
        verdict = "CROSS-FADE-COMPATIBLE"
    elif p567["n"] and p567["rotation"] >= 2 / 3 * p567["n"]:
        verdict = "ROTATION-LIKE"
    elif p567["n"] and p567["direct"] >= 2 / 3 * p567["n"]:
        verdict = "DIRECT-ANGULAR"
    else:
        verdict = "MIXED"
    rt["verdict"] = verdict
    json.dump(dict(summary=out, rows=rows), open(f"{run}/analysis_summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
