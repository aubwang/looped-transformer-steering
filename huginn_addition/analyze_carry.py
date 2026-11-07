"""Does a carry column first represent the pre-carry digit and later move one notch to the final digit?
Usage: python -I huginn_addition/analyze_carry.py results/<carry_run> results/<collect_run>

Rows: answer positions pos >= 1 with col >= 1, excluding the overflow digit (col == n_digits: its 'pre-carry digit' 0
is not a column sum). carry_in = 1 rows ("carry") vs carry_in = 0 rows ("no-carry"). pre = (digit - carry_in) mod 10.
No-carry rows are reweighted to the carry rows' (n_digits, col) distribution (unsupported strata dropped and reported).

(i) Readout lens (teacher-forced coda readout at every step t = 0..K): carry P(final), P(pre = final-1); no-carry
    P(final), P(final-1) (generic predecessor baseline), P(final+1). Time at which the readout settles on final.
(ii) Per-step decoders trained ONLY on no-carry rows (pre == final), cross-fitted by problem (5 grouped folds), applied
    to held-out rows: (a) ridge on [256-d random projection + nuisance dummies] -> (cos, sin) of the digit (inner grouped
    CV for alpha; standardisation inside the training fold); (b) fixed late circle plane: calibrated circle coords
    a = Gram^-1 U s, OLS a ~ [1, nuisance] + W [cos, sin] on training rows; held-out decoded = W^-1 (a - nuisance part).
    Signed phase offset relative to theta_final (a pre-then-final story predicts ~ -36 deg early -> 0 late); rows with
    decoded radius < 0.3 are excluded from angle summaries. Bootstrap by problem.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from analyze_gradients import CAT, INK2, wrap  # noqa: E402
from shape_analysis import nuisance_design  # noqa: E402

RNG = np.random.default_rng(3)
ALPHAS = [1e-1, 1, 10, 100, 1e3, 1e4]
RAD_MIN = 0.3


def group_folds(groups, k, seed):
    u = np.unique(groups)
    r = np.random.default_rng(seed).permutation(u)
    f = {g: i % k for i, g in enumerate(r)}
    return np.array([f[g] for g in groups])


def ridge(X, Y, a):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + a * np.eye(d), X.T @ Y)


def fit_ridge_cv(X, Y, groups):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    inner = group_folds(groups, 4, 1)
    err = []
    for a in ALPHAS:
        e = 0.0
        for f in range(4):
            tr, te = inner != f, inner == f
            if te.sum() == 0 or tr.sum() == 0:
                continue
            mi, si = X[tr].mean(0), X[tr].std(0) + 1e-8  # scaling fitted on the inner-training rows only
            B = ridge((X[tr] - mi) / si, Y[tr] - Y[tr].mean(0), a)
            e += ((Y[te] - Y[tr].mean(0) - ((X[te] - mi) / si) @ B) ** 2).sum()
        err.append(e)
    a = ALPHAS[int(np.argmin(err))]
    m_ = Xs.mean(0)
    B = ridge(Xs - m_, Y - ym, a)
    return lambda Xn: ym + ((Xn - mu) / sd - m_) @ B


def boot_curve(vals, pids, w=None, B=500):
    """vals (n, T); problem-cluster bootstrap of the (weighted) mean curve."""
    w = np.ones(len(vals)) if w is None else w
    up = np.unique(pids)
    idx = {p: np.where(pids == p)[0] for p in up}
    m = (w[:, None] * vals).sum(0) / w.sum()
    bs = []
    for _ in range(B):
        sel = np.concatenate([idx[p] for p in RNG.choice(up, len(up))])
        bs.append((w[sel, None] * vals[sel]).sum(0) / w[sel].sum())
    bs = np.array(bs)
    return m, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def main(run, collect_run):
    out = f"{run}/analysis"
    os.makedirs(out, exist_ok=True)
    z = np.load(f"{run}/carry_traj.npz")
    probs = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    recs = []
    for r, (uid, j) in enumerate(z["rows"]):
        p = probs[int(uid)]
        di = p["digit_info"][j]
        prev = "eq" if j == 0 else (f"sp{p['digit_info'][0]['digit']}" if j == 1 else f"d{p['digit_info'][j - 1]['digit']}")
        recs.append(dict(row=r, uid=int(uid), pid=p["pid"], pos=int(j), digit=di["digit"], carry=int(di["carry_in"]),
                         col=di["col"], dep=di["dep_len"], n_answer=len(p["digit_info"]), n_digits=p["n_digits"],
                         overflow=p["overflow"], demo=p["demo"], family=p["family"], prev=prev))
    S = pd.DataFrame(recs)
    S = S[(S["pos"] >= 1) & (S["col"] >= 1) & (S["col"] < S["n_digits"])].reset_index(drop=True)
    S["pre"] = (S["digit"] - S["carry"]) % 10
    rows = S["row"].to_numpy()
    lp, proj, plane = z["lp"][rows], z["proj"][rows].astype(np.float64), z["plane"][rows].astype(np.float64)
    T = lp.shape[1]
    tt = np.arange(T)
    car = S["carry"].to_numpy() == 1
    d, pre, pid = S["digit"].to_numpy(), S["pre"].to_numpy(), S["pid"].to_numpy()
    ar = np.arange(len(S))
    fid = z["full_is_digit"][rows].astype(bool)  # full-vocab argmax is a digit token
    correct_K = (lp[:, -1].argmax(-1) == d) & fid[:, -1]

    # reweight no-carry rows to carry (n_digits, col) distribution
    S["stratum"] = S["n_digits"].astype(str) + "_" + S["col"].astype(str)
    stv = S["stratum"].to_numpy()

    def strat_weights(subset):  # no-carry rows in subset reweighted to the carry rows' strata within the same subset
        w_ = np.zeros(len(S))
        cc_ = pd.Series(stv[subset & car]).value_counts()
        nc_ = pd.Series(stv[subset & ~car]).value_counts()
        w_[subset & car] = 1.0
        for st_, cnt in cc_.items():
            if st_ in nc_:
                w_[subset & ~car & (stv == st_)] = cnt / nc_[st_]
            else:
                w_[subset & car & (stv == st_)] = 0.0  # drop unsupported carry strata
        return w_, [st_ for st_ in cc_.index if st_ not in nc_]

    w, unsupported = strat_weights(np.ones(len(S), bool))
    summ = {"rows": int(len(S)), "carry_rows": int(car.sum()), "nocarry_rows": int((~car).sum()),
            "problems": int(S["pid"].nunique()), "unsupported_carry_strata": unsupported,
            "correct_at_K_carry": float(correct_K[car].mean()), "correct_at_K_nocarry": float(correct_K[~car].mean())}

    # (i) readout lens (rows correct at the end; weights recomputed within that subset)
    P = np.exp(lp)
    w, unsupported_c = strat_weights(correct_K)
    summ["unsupported_carry_strata_correct_rows"] = unsupported_c
    sel_c, sel_n = car & correct_K & (w > 0), (~car) & correct_K & (w > 0)
    lens = {
        "carry_P_final": boot_curve(P[sel_c][np.arange(sel_c.sum()), :, d[sel_c]], pid[sel_c]),
        "carry_P_pre": boot_curve(P[sel_c][np.arange(sel_c.sum()), :, pre[sel_c]], pid[sel_c]),
        "carry_P_final_plus1": boot_curve(P[sel_c][np.arange(sel_c.sum()), :, (d[sel_c] + 1) % 10], pid[sel_c]),
        "nocarry_P_final": boot_curve(P[sel_n][np.arange(sel_n.sum()), :, d[sel_n]], pid[sel_n], w[sel_n]),
        "nocarry_P_final_minus1": boot_curve(P[sel_n][np.arange(sel_n.sum()), :, (d[sel_n] - 1) % 10], pid[sel_n], w[sel_n]),
        "nocarry_P_final_plus1": boot_curve(P[sel_n][np.arange(sel_n.sum()), :, (d[sel_n] + 1) % 10], pid[sel_n], w[sel_n]),
    }
    show = sorted({t for t in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48) if t < T} | {T - 1})
    summ["lens_correct_at_K"] = {k: {"t": show, "mean": v[0][show].round(4).tolist(), "lo": v[1][show].round(4).tolist(),
                                     "hi": v[2][show].round(4).tolist()} for k, v in lens.items()}
    # argmax identity fractions
    am = lp.argmax(-1)
    summ["argmax_is_pre_carry"] = (am[sel_c] == pre[sel_c, None]).mean(0)[show].round(3).tolist()
    summ["argmax_is_final_minus1_nocarry_weighted"] = ((w[sel_n, None] * (am[sel_n] == ((d[sel_n] - 1) % 10)[:, None])).sum(0) / w[sel_n].sum())[show].round(3).tolist()

    # settle time: first t after which argmax == final for all later t (correct-at-K rows)
    ok = (am == d[:, None]) & fid
    settle = np.array([T - 1 - np.argmax(~ok[i, ::-1]) + 1 if not ok[i].all() else 0 for i in range(len(S))])
    st_c, st_n = settle[sel_c], settle[sel_n]
    summ["settle_step_median"] = {"carry": float(np.median(st_c)), "nocarry_weighted_mean": float((w[sel_n] * st_n).sum() / w[sel_n].sum()),
                                  "carry_mean": float(st_c.mean()), "nocarry_median": float(np.median(st_n))}
    dep = S["dep"].to_numpy()
    summ["settle_step_carry_by_dep"] = {str(k): {"n": int(((dep == k) & sel_c).sum()), "mean": float(settle[(dep == k) & sel_c].mean())}
                                        for k in sorted(set(dep[sel_c]))}
    # among rows that pass through pre before final: was pre ever the argmax before settling?
    summ["frac_rows_argmax_pre_before_settle"] = {
        "carry": float(np.mean([(am[i, : max(settle[i], 1)] == pre[i]).any() for i in np.where(sel_c)[0]])),
        "nocarry_final_minus1_weighted": float(np.sum([w[i] * (am[i, : max(settle[i], 1)] == (d[i] - 1) % 10).any() for i in np.where(sel_n)[0]]) / w[sel_n].sum()),
        "nocarry_final_plus1_weighted": float(np.sum([w[i] * (am[i, : max(settle[i], 1)] == (d[i] + 1) % 10).any() for i in np.where(sel_n)[0]]) / w[sel_n].sum()),
        "carry_final_plus1": float(np.mean([(am[i, : max(settle[i], 1)] == (d[i] + 1) % 10).any() for i in np.where(sel_c)[0]]))}

    # (ii) decoders
    N = nuisance_design(S)
    th_f = 2 * np.pi * d / 10
    Yt = np.stack([np.cos(th_f), np.sin(th_f)], 1)
    folds = group_folds(pid, 5, 0)
    U = np.load(f"{collect_run}/directions.npz")["U"].astype(np.float64)
    Gm = U @ U.T
    acoord = np.einsum("ij,ntj->nti", np.linalg.inv(Gm), plane)  # calibrated circle coords (n, T, 2)
    dec = {"proj": np.full((len(S), T, 2), np.nan), "plane": np.full((len(S), T, 2), np.nan)}
    for t in range(T):
        for f in range(5):
            tr = (folds != f) & (~car)
            te = folds == f
            Xtr = np.column_stack([proj[tr, t], N[tr]])
            pred = fit_ridge_cv(Xtr, Yt[tr], pid[tr])
            dec["proj"][te, t] = pred(np.column_stack([proj[te, t], N[te]]))
            # fixed plane: OLS a ~ [1, N, cos, sin]
            Dtr = np.column_stack([np.ones(tr.sum()), N[tr], Yt[tr]])
            coef, *_ = np.linalg.lstsq(Dtr, acoord[tr, t], rcond=None)
            W = coef[-2:]  # (2 [cos,sin], 2 [a])
            nuis = np.column_stack([np.ones(te.sum()), N[te]]) @ coef[:-2]
            dec["plane"][te, t] = np.linalg.lstsq(W.T, (acoord[te, t] - nuis).T, rcond=None)[0].T
    def late_contrast(off, okr, cS, nS, B=2000):  # joint problem bootstrap of mean late offset (t >= 32), carry - no-carry
        lo_ = np.where(okr[:, 32:], off[:, 32:], np.nan)
        lo_ = np.nanmean(lo_, 1)
        up = np.unique(pid)
        def stat(ps):
            k = np.isin(pid, ps)
            a, b = lo_[k & cS], lo_[k & nS]
            wb = w[k & nS]
            return np.nanmean(a) - np.nansum(np.nan_to_num(b) * wb) / np.sum(wb * ~np.isnan(b))
        m = stat(up)
        bs = [stat(RNG.choice(up, len(up))) for _ in range(B)]
        return [round(float(m), 1), round(float(np.nanpercentile(bs, 2.5)), 1), round(float(np.nanpercentile(bs, 97.5)), 1)]

    summ["decoders"] = {}
    curves = {}
    for nm, Dd in dec.items():
        ph = np.degrees(np.arctan2(Dd[..., 1], Dd[..., 0]))
        rad = np.linalg.norm(Dd, axis=-1)
        off = wrap(ph - np.degrees(th_f)[:, None])  # signed offset from final
        okr = rad >= RAD_MIN
        nc_err = np.where(okr[~car], np.abs(off[~car]), np.nan)
        nc_acc = (np.round(ph / 36) % 10 == d[:, None])[~car].mean(0)
        cS, nS = car & correct_K & (w > 0), (~car) & correct_K & (w > 0)  # rows whose final readout is correct
        c_off = np.where(okr[cS], off[cS], np.nan)
        n_off = np.where(okr[nS], off[nS], np.nan)
        c_all = np.where(okr[car], off[car], np.nan)

        def med_boot(v, pids, wts=None):  # mean over problems of (weighted, NaN-aware) problem means; bootstrap CI
            wts = np.ones(len(v)) if wts is None else wts
            up = np.unique(pids)
            pm = []
            for p_ in up:
                k = pids == p_
                vv, ww = v[k], wts[k][:, None] * ~np.isnan(v[k])
                pm.append(np.nansum(np.nan_to_num(vv) * ww, 0) / np.where(ww.sum(0) > 0, ww.sum(0), np.nan))
            pm = np.array(pm)
            m = np.nanmean(pm, 0)
            bs = np.array([np.nanmean(pm[RNG.integers(0, len(pm), len(pm))], 0) for _ in range(300)])
            return m, np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)

        cm = med_boot(c_off, pid[cS])
        nm_ = med_boot(n_off, pid[nS], w[nS])
        call = med_boot(c_all, pid[car])
        cW = car & ~correct_K
        cwr = med_boot(np.where(okr[cW], off[cW], np.nan), pid[cW])
        curves[nm] = (cm, nm_)
        closer_pre = (np.abs(wrap(ph - np.degrees(2 * np.pi * pre / 10)[:, None])) < np.abs(off))
        closer_pre = closer_pre & correct_K[:, None]
        summ["decoders"][nm] = {
            "t": show,
            "nocarry_heldout_digit_acc": nc_acc[show].round(3).tolist(),
            "nocarry_median_abs_err_deg": np.nanmedian(nc_err, 0)[show].round(1).tolist(),
            "frac_rows_radius_ok_carry": okr[car].mean(0)[show].round(3).tolist(),
            "carry_mean_signed_offset_deg_correct_rows": cm[0][show].round(1).tolist(),
            "carry_mean_signed_offset_deg_all_rows": call[0][show].round(1).tolist(),
            "carry_wrong_rows_mean_signed_offset_deg": cwr[0][show].round(1).tolist(),
            "carry_wrong_rows_ci": [cwr[1][show].round(1).tolist(), cwr[2][show].round(1).tolist()],
            "n_carry_correct_wrong": [int(cS.sum()), int(cW.sum())],
            "late_offset_t32_64_carry_minus_nocarry": late_contrast(off, okr, cS, nS),
            "carry_offset_ci": [cm[1][show].round(1).tolist(), cm[2][show].round(1).tolist()],
            "nocarry_mean_signed_offset_deg_weighted_correct_rows": nm_[0][show].round(1).tolist(),
            "nocarry_offset_ci": [nm_[1][show].round(1).tolist(), nm_[2][show].round(1).tolist()],
            "carry_correct_frac_closer_to_pre_rate": (np.where(okr[cS], closer_pre[cS], False).sum(0) / np.maximum(okr[cS].sum(0), 1))[show].round(3).tolist(),
        }
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))

    # figures
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax = axs[0]
    for k, c, ls, lab in (("carry_P_final", CAT[1], "-", "carry: P(final)"), ("carry_P_pre", CAT[1], "--", "carry: P(final−1) = 'no carry' answer"),
                          ("carry_P_final_plus1", CAT[1], ":", "carry: P(final+1)"),
                          ("nocarry_P_final", CAT[0], "-", "no carry: P(final)"), ("nocarry_P_final_plus1", CAT[0], "--", "no carry: P(final+1) = 'carry' answer"),
                          ("nocarry_P_final_minus1", CAT[0], ":", "no carry: P(final−1)")):
        m, lo, hi = lens[k]
        ax.plot(tt, m, color=c, ls=ls, label=lab)
        ax.fill_between(tt, lo, hi, color=c, alpha=0.12, lw=0)
    ax.set_xscale("symlog", linthresh=4); ax.set_xlabel("loop step"); ax.set_ylabel("readout probability (mean)")
    ax.set_title("Readout over loops (rows correct at the end)"); ax.legend(fontsize=7)
    for ax, nm, title in ((axs[1], "proj", "Per-step decoder (trained on no-carry rows)"), (axs[2], "plane", "Fixed late circle plane")):
        (cm, nm_) = curves[nm]
        ax.plot(tt, cm[0], color=CAT[1], label="carry rows (final readout correct)")
        ax.fill_between(tt, cm[1], cm[2], color=CAT[1], alpha=0.15, lw=0)
        ax.plot(tt, nm_[0], color=CAT[0], label="no-carry rows (held out, correct)")
        ax.fill_between(tt, nm_[1], nm_[2], color=CAT[0], alpha=0.15, lw=0)
        ax.axhline(-36, color=INK2, ls=":", lw=1, label="pre-carry digit (−36°)")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xscale("symlog", linthresh=4); ax.set_ylim(-90, 60)
        ax.set_xlabel("loop step"); ax.set_ylabel("decoded phase − final digit phase (deg)")
        ax.set_title(title); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/carry.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"))
