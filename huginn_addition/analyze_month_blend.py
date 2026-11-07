"""Month shape + context-blend experiment (intervene_app.py `mblend`). Prespecified before the run.
Usage: python -I huginn_addition/analyze_month_blend.py results/<mblend_run>

Rows: 24 pure prompts (2 templates x 12 months, "What month comes after M?") and 216 blends E_a = (1-a) E(m0) + a E(m1)
(6 opposite pairs m, m+6; 6 quarter pairs m, m+3; a = 0.1..0.9; both templates), all from one common init per template.
State = the answer-position state at loop 96 (and at checkpoints 1, 2, 4, 8, 16, 32, 64 for the loop-role analysis).

0. QUALIFICATION: on pure prompts, 12-way argmax at loop 96 == successor month for >= 9/12 months in each template.
   STATIONARITY reported (max-position mean drift over the last 8 loops < 2e-5); blend rows that are not stationary are
   flagged and excluded from path classification.
1. SHAPE (pure fixed points): template-centred states; circle code of the month (canonical order) vs 5000 random cyclic
   orders, line (canonical; best of 2000 random), categorical, 200 random rank-2 codes; held-out R^2 by leave-one-month-out
   (both templates of the held-out month), top-10 PCs of the training rows, ridge 1e-3 on code weights.
2. PLANE (from pure fixed points, per template-centred data): U = top-2 PCs of the 12 month means (average of the two
   templates); oriented so the canonical order runs counter-clockwise; VALID if canonical cyclic order, every gap >= 0.3 x
   30 deg, s2/s1 >= 0.3. Month-phase of a point = piecewise-linear interpolation of its angle between month-mean angles;
   radius relative to the median month-mean radius. If invalid: no path verdict (INCONCLUSIVE).
3. PATHS (PRIMARY: the 12 opposite-pair paths = 6 pairs x 2 templates; quarter pairs descriptive). Per path, using the
   loop-96 state at a = 0.1..0.9 plus the pure endpoints, projected (template-centred) on U:
   ARC   if radius >= 0.6 for every a in {0.3, 0.4, 0.5, 0.6, 0.7} and the phase at a = 0.5 is within 1.5 months of
         m0 + 3 or m0 - 3 (the two quadrature months);
   CHORD if radius at a = 0.5 < 0.3;
   else INTERMEDIATE. Paths with a non-stationary point in a = 0.3..0.7 are UNRESOLVED (excluded; reported).
   VERDICT: ARC-LIKE if >= 2/3 of the resolved opposite paths are ARC (and >= 8 resolved), CHORD-LIKE if >= 2/3 CHORD,
   else MIXED; INCONCLUSIVE if not qualified, plane invalid, or < 8 resolved paths.
   Behavioural counterpart (descriptive): at a = 0.5 on opposite paths, the 12-way readout mass on the successors of the
   endpoints (m0+1, m1+1) vs on the successors of the quadrature months (m0+4, m0-2) +- 1 month.
REVISIONS after Astra's review (frozen before inspecting data):
 R-a STATIONARY REFERENCE: all 24 pure rows must be stationary, else INCONCLUSIVE for settled-state claims.
 R-b Path classes (opposite pairs) replace ARC/CHORD:
     ARC-TRAVEL: radius >= 0.6 for every a in 0.3..0.7, phase at a=0.5 within 1.5 months of a quadrature month, AND the
                 unwrapped phase progresses monotonically from a=0 to a=1 with no step between consecutive a > 2 months;
     QUADRATURE-JUMP: the radius/quadrature conditions hold but progression is non-monotone or has a step > 2 months;
     LINEAR-BLEND: relative distance of the a=0.5 state from the linear blend of the endpoint states,
                 ||x_0.5 - (x_0 + x_1)/2|| / ||x_1 - x_0|| < 0.2 (full state space);
     LOW-RADIUS-NONLINEAR: radius at 0.5 < 0.3 but not LINEAR-BLEND; else INTERMEDIATE.
     VERDICT: ARC-TRAVEL-LIKE if >= 2/3 of eligible resolved opposite paths are ARC-TRAVEL, LINEAR-LIKE if >= 2/3
     LINEAR-BLEND, else MIXED; INCONCLUSIVE if not qualified, plane invalid, R-a fails, or < 8 eligible paths.
 R-c Endpoint check per path and template: the in-plane angular separation of the two pure endpoints must be 6 +- 1.5
     months, else the path is NOT-OPPOSITE and ineligible. Reported: plane variance explained (top-2 share of the pure
     month-mean variance), off-plane fraction of the blended a=0.5 displacement, radius at a=0.5 of the explicit linear
     state blend of the endpoints (baseline).
 R-d Readout baselines: uniform expectations 0.5 (six quadrature-successor months) vs 1/6 (two endpoint successors);
     absolute month mass at a=0.5 reported.
 R-e Loop role: numerator (blend radius) and denominator (pure median radius) reported separately per checkpoint.
 R-f Shape CV uses training-month-only template means.
4. LOOP ROLE (descriptive): for opposite pairs at a = 0.5, the radius of the state at each checkpoint loop, relative to
   the median radius of the pure states at that same loop (template-centred, same plane U): does recurrence move the
   blended state onto the ring?
"""
from __future__ import annotations

import json
import sys

import numpy as np

TAU = 2 * np.pi
R = 10


def code(order, nl=12):
    pos = np.zeros(nl)
    for i, d in enumerate(order):
        pos[d] = i
    return np.stack([np.cos(TAU * pos / nl), np.sin(TAU * pos / nl)], 1)


def sse(Ztr, Ytr, Zte, Yte, lam=1e-3):
    zm, ym = Ztr.mean(0), Ytr.mean(0)
    A = Ztr - zm
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    return float(((Yte - ((Zte - zm) @ W + ym)) ** 2).sum())


def wrap12(x):
    return ((np.asarray(x) + 6) % 12) - 6


def main(run):
    z = np.load(f"{run}/mblend.npz", allow_pickle=True)
    meta = json.load(open(f"{run}/meta.json"))
    spec = json.load(open(f"{run}/rows.json"))
    rows = spec["rows"]
    assert z["done"].all(), "incomplete run"
    CK = list(z["checkpoints"]); S = z["states"].astype(np.float64)
    i96 = CK.index(96)
    tm = np.array([r["template"] for r in rows]); pure = np.array([r["kind"] == "pure" for r in rows])
    out = {"runtime_s": meta["runtime_s"]}
    # 0. qualification
    succ = {}
    for t in ("A", "B"):
        idx = [i for i, r in enumerate(rows) if r["kind"] == "pure" and r["template"] == t]
        succ[t] = int(sum(z["ro_lp"][i, -1].argmax() == (rows[i]["m0"] + 1) % 12 for i in idx))
    qualified = all(v >= 9 for v in succ.values())
    stat = z["late8"] < 2e-5
    out["qualification"] = dict(successor_correct=succ, qualified=qualified, stationary_pure=int(stat[pure].sum()),
                                stationary_blend=int(stat[~pure].sum()), n_blend=int((~pure).sum()),
                                median_month_mass_pure=float(np.median(z["ro_mass"][pure, -1])))
    # 1. shape (pure)
    pid = np.where(pure)[0]
    month = np.array([rows[i]["m0"] for i in pid]); Xraw = S[pid, i96].copy(); X = Xraw.copy()
    for t in ("A", "B"):
        m = tm[pid] == t
        X[m] -= X[m].mean(0)  # used for the plane (all pure rows)
    rng = np.random.default_rng(0)
    canon = tuple(range(12))
    rand_orders = [tuple([0] + list(rng.permutation(np.arange(1, 12)))) for _ in range(5000)]
    lin = [canon] + [tuple(rng.permutation(12)) for _ in range(2000)]
    scale = np.linalg.norm(code(canon))
    rc = []
    for i in range(200):
        c = np.random.default_rng(100 + i).normal(size=(12, 2)); c -= c.mean(0); q_, _ = np.linalg.qr(c); rc.append(q_ * scale / np.linalg.norm(q_))
    s_c = 0.0; s_ro = np.zeros(5000); s_lin = np.zeros(len(lin)); s_cat = 0.0; s_rc = np.zeros(200); sst = 0.0
    for hm in range(12):
        tr, te = month != hm, month == hm
        Xc = Xraw.copy()
        for t in ("A", "B"):
            mt = tm[pid] == t
            Xc[mt] -= Xraw[mt & tr].mean(0)  # training-month-only template means
        mu = Xc[tr].mean(0); V = np.linalg.svd(Xc[tr] - mu, full_matrices=False)[2][:R].T
        Ytr, Yte = (Xc[tr] - mu) @ V, (Xc[te] - mu) @ V
        a, b = month[tr], month[te]
        c = code(canon); s_c += sse(c[a], Ytr, c[b], Yte)
        for j, o in enumerate(rand_orders):
            c = code(o); s_ro[j] += sse(c[a], Ytr, c[b], Yte)
        for j, o in enumerate(lin):
            c = np.zeros((12, 1)); c[list(o), 0] = np.arange(12); s_lin[j] += sse(c[a], Ytr, c[b], Yte)
        for j, c in enumerate(rc):
            s_rc[j] += sse(c[a], Ytr, c[b], Yte)
        oh = np.eye(12)[:, :11]
        # categorical cannot predict an unseen month: report the centred-mean baseline (R^2 = 0 reference) instead
        sst += float(((Yte - Ytr.mean(0)) ** 2).sum())
    r2 = lambda e: 1 - np.asarray(e) / sst
    ro = r2(s_ro); rr = r2(s_rc)
    out["shape"] = dict(R2_canonical_circle=float(r2(s_c)), canonical_rank_among_5000=int((ro > r2(s_c)).sum() + 1),
                        R2_random_cyclic=dict(median=float(np.median(ro)), q99=float(np.quantile(ro, .99)), max=float(ro.max())),
                        R2_line_canonical=float(r2(s_lin[0])), R2_line_best_of_2001=float(r2(s_lin).max()),
                        R2_random_rank2=dict(median=float(np.median(rr)), max=float(rr.max())),
                        note="leave-one-month-out: categorical cannot predict an unseen level, so it is not reported")
    # 2. plane from pure month means (template-centred)
    MM = np.stack([X[month == m].mean(0) for m in range(12)])
    MMc = MM - MM.mean(0)
    _, sv, Vt = np.linalg.svd(MMc, full_matrices=False)
    U = Vt[:2].T
    ang = np.angle((MMc @ U) @ np.array([1, 1j]))
    if np.sum(np.sin(np.diff(np.concatenate([ang, ang[:1]])))) < 0:
        U[:, 1] *= -1; ang = np.angle((MMc @ U) @ np.array([1, 1j]))
    rel = (ang - ang[0]) % TAU
    gaps = np.diff(np.concatenate([rel, [TAU]]))
    canonical = bool(np.all(np.diff(rel) > 0))
    valid = bool(canonical and gaps.min() >= 0.3 * TAU / 12 and sv[1] / sv[0] >= 0.3)
    crad = float(np.median(np.abs((MMc @ U) @ np.array([1, 1j]))))
    out["plane"] = dict(valid=valid, canonical_cyclic=canonical, min_gap_deg=float(np.degrees(gaps.min())), sv_ratio=float(sv[1] / sv[0]),
                        top2_variance_share=float((sv[:2] ** 2).sum() / (sv ** 2).sum()))
    ref_stationary = bool(stat[pure].all())
    rel_ext = np.concatenate([rel, [TAU]])

    def phase_rad(v, t, ck=i96):
        """v: raw state; centre by the pure-prompt mean of template t at checkpoint INDEX ck; returns (phase in months, radius)."""
        cen = S[[i for i in pid if tm[i] == t], ck].mean(0)
        zc = ((v - cen) @ U) @ np.array([1, 1j])
        x = (np.angle(zc) - ang[0]) % TAU
        kk = int(np.clip(np.searchsorted(rel_ext, x, side="right") - 1, 0, 11))
        w = (x - rel_ext[kk]) / (rel_ext[kk + 1] - rel_ext[kk])
        return float((kk + w) % 12), float(np.abs(zc) / crad)
    # 3. paths
    paths = []
    for t in ("A", "B"):
        for kind in ("opposite", "quarter"):
            pairs = sorted({(r["m0"], r["m1"]) for r in rows if r["kind"] == kind and r["template"] == t})
            for m0, m1 in pairs:
                idx = sorted([i for i, r in enumerate(rows) if r["kind"] == kind and r["template"] == t and r["m0"] == m0 and r["m1"] == m1],
                             key=lambda i: rows[i]["a"])
                e0 = [i for i in pid if tm[i] == t and rows[i]["m0"] == m0][0]; e1 = [i for i in pid if tm[i] == t and rows[i]["m0"] == m1][0]
                seq = [(0.0, e0)] + [(rows[i]["a"], i) for i in idx] + [(1.0, e1)]
                pr = [(a_, *phase_rad(S[i, i96], t), bool(stat[i])) for a_, i in seq]
                mid = [p for p in pr if 0.25 < p[0] < 0.75]
                unresolved = not all(p[3] for p in mid)
                p5 = [p for p in pr if abs(p[0] - 0.5) < 1e-6][0]
                cls = None
                i5 = [i for a_, i in seq if abs(a_ - 0.5) < 1e-6][0]
                sep = abs(wrap12(pr[-1][1] - pr[0][1]))
                eligible = kind != "opposite" or abs(sep - 6) <= 1.5
                x0, x1, x5 = S[e0, i96], S[e1, i96], S[i5, i96]
                lin_dev = float(np.linalg.norm(x5 - (x0 + x1) / 2) / np.linalg.norm(x1 - x0))
                lin_rad = phase_rad((x0 + x1) / 2, t)[1]
                disp = x5 - (x0 + x1) / 2
                off_plane = float(1 - np.linalg.norm(disp @ U) ** 2 / max(np.linalg.norm(disp) ** 2, 1e-30))
                steps = wrap12(np.diff([p[1] for p in pr]))
                monotone = bool(np.all(steps > 0) or np.all(steps < 0))
                if kind == "opposite" and not unresolved and eligible:
                    near_q = min(abs(wrap12(p5[1] - (m0 + 3))), abs(wrap12(p5[1] - (m0 - 3)))) <= 1.5
                    if all(p[2] >= 0.6 for p in mid) and near_q:
                        cls = "ARC-TRAVEL" if (monotone and np.abs(steps).max() <= 2) else "QUADRATURE-JUMP"
                    elif lin_dev < 0.2:
                        cls = "LINEAR-BLEND"
                    elif p5[2] < 0.3:
                        cls = "LOW-RADIUS-NONLINEAR"
                    else:
                        cls = "INTERMEDIATE"
                pm = np.exp(z["ro_lp"][i5, -1]); pm = pm / pm.sum()
                endp = pm[(m0 + 1) % 12] + pm[(m1 + 1) % 12]
                quad = sum(pm[(c + d) % 12] for c in (m0 + 4, m0 - 2) for d in (-1, 0, 1))
                loop_r = {}
                if kind == "opposite":
                    for ci, ck in enumerate(CK):
                        pr_ck = phase_rad(S[i5, ci], t, ci)[1]
                        pure_r = np.median([phase_rad(S[i, ci], t, ci)[1] for i in pid if tm[i] == t])
                        loop_r[int(ck)] = dict(rel=round(pr_ck / max(pure_r, 1e-9), 3), blend=round(pr_ck, 3), pure_median=round(float(pure_r), 3))
                paths.append(dict(template=t, kind=kind, m0=m0, m1=m1, cls=cls, unresolved=unresolved, eligible=eligible,
                                  endpoint_separation_months=round(float(sep), 2), linear_blend_deviation=round(lin_dev, 3),
                                  linear_blend_radius_a05=round(float(lin_rad), 3), off_plane_fraction_a05=round(off_plane, 3),
                                  month_mass_a05=round(float(z["ro_mass"][i5, -1]), 3), monotone_phase=monotone,
                                  path=[(round(a_, 1), round(ph, 2), round(rd, 3), st) for a_, ph, rd, st in pr],
                                  readout_a05=dict(argmax=int(pm.argmax()), mass_endpoint_successors=round(float(endp), 3),
                                                   mass_quadrature_successors=round(float(quad), 3)),
                                  radius_a05_by_loop_rel_pure=loop_r))
    opp = [p for p in paths if p["kind"] == "opposite"]
    res = [p for p in opp if not p["unresolved"] and p["eligible"]]
    n_arc = sum(p["cls"] == "ARC-TRAVEL" for p in res); n_ch = sum(p["cls"] == "LINEAR-BLEND" for p in res)
    if not qualified or not valid or not ref_stationary or len(res) < 8:
        verdict = "INCONCLUSIVE"
    elif n_arc >= 2 / 3 * len(res):
        verdict = "ARC-TRAVEL-LIKE"
    elif n_ch >= 2 / 3 * len(res):
        verdict = "LINEAR-LIKE"
    else:
        verdict = "MIXED"
    out["paths"] = dict(verdict=verdict, reference_stationary=ref_stationary, eligible_resolved_opposite=len(res),
                        classes={c: sum(p["cls"] == c for p in res) for c in ("ARC-TRAVEL", "QUADRATURE-JUMP", "LINEAR-BLEND", "LOW-RADIUS-NONLINEAR", "INTERMEDIATE")},
                        not_opposite=sum(not p["eligible"] for p in opp), unresolved=sum(p["unresolved"] for p in opp),
                        median_radius_a05_opposite=float(np.median([[x for x in p["path"] if abs(x[0] - 0.5) < 1e-6][0][2] for p in opp])),
                        median_linear_blend_radius_a05=float(np.median([p["linear_blend_radius_a05"] for p in opp])),
                        median_linear_blend_deviation=float(np.median([p["linear_blend_deviation"] for p in opp])),
                        median_off_plane_fraction_a05=float(np.median([p["off_plane_fraction_a05"] for p in opp])),
                        readout_a05=dict(median_mass_quadrature_successors=float(np.median([p["readout_a05"]["mass_quadrature_successors"] for p in opp])),
                                         median_mass_endpoint_successors=float(np.median([p["readout_a05"]["mass_endpoint_successors"] for p in opp])),
                                         uniform_baselines=dict(quadrature=0.5, endpoint=1 / 6),
                                         median_month_mass=float(np.median([p["month_mass_a05"] for p in opp]))),
                        loop_role_a05={int(ck): dict(rel=float(np.median([p["radius_a05_by_loop_rel_pure"][int(ck)]["rel"] for p in opp])),
                                                     blend=float(np.median([p["radius_a05_by_loop_rel_pure"][int(ck)]["blend"] for p in opp])),
                                                     pure_median=float(np.median([p["radius_a05_by_loop_rel_pure"][int(ck)]["pure_median"] for p in opp])))
                                       for ck in CK})
    json.dump(dict(summary=out, paths=paths), open(f"{run}/analysis_summary.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
