"""Stricter audit of the late carry offset (no model run; uses the saved carry_traj data).
Usage: python -I huginn_addition/analyze_carry_audit.py results/<carry_run> results/<collect_run>

Worry: the original decoders were trained only on no-carry rows, where the column's input sum (a+b mod 10) and the final
digit are identical, so any input-sum or input-digit information in the state is rewarded and then pulls carry rows
toward the pre-carry digit. Late features = mean over steps t = 32..64 (states are near-stationary there).
Rows as in analyze_carry.py (pos >= 1, col >= 1, no overflow digit); main analyses use rows whose final readout is correct.

A. Decoder-free geometry ("fraction of the way toward the pre-carry digit"). Cross-fitted by problem (5 folds). In the
   training fold: residualise features on nuisance dummies (prev token, pos, col, n_answer, n_digits, overflow, demo,
   family) [+ optionally one-hot input digits of this column and the lower column], with final-digit one-hots and a carry
   dummy in the same regression so their effects are not removed; class centroids c_k = mean of training no-carry rows
   with final digit k. Held-out row with final d: f_pre = <x - c_d, c_{d-1} - c_d> / ||c_{d-1} - c_d||^2 (0 = at its own
   digit, 1 = at the pre-carry digit's centroid); control f_next toward d+1. Carry vs no-carry, problem bootstrap; and a
   strictly matched version (cells = final digit x previous token x n_digits x col containing both groups).
   Spaces: 256-d random projection, fitted circle plane (calibrated coords), and an inputs-only "state" (one-hot input
   digits, no model state) as a baseline for how much input-digit information alone could produce.
B. Decoders (late features, angle offset from the final digit, as before): no-carry-trained (replication), no-carry-trained
   on input-residualised features ("can't see the input digits"), trained on all rows (no extrapolation), and an
   inputs-only decoder (no state).
C. Wrong carry rows: what the model output, and where the state sits relative to its output and the truth.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from analyze_carry import fit_ridge_cv, group_folds  # noqa: E402
from analyze_gradients import wrap  # noqa: E402
from shape_analysis import nuisance_design  # noqa: E402

RNG = np.random.default_rng(4)
T0 = 32
NB = 2000


def onehots(df, cols):
    return pd.get_dummies(df[cols].astype(str), drop_first=True).to_numpy(float)


def load(run, collect_run):
    z = np.load(f"{run}/carry_traj.npz")
    probs = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    recs = []
    for r, (uid, j) in enumerate(z["rows"]):
        p = probs[int(uid)]
        di = p["digit_info"][j]
        prev = "eq" if j == 0 else (f"sp{p['digit_info'][0]['digit']}" if j == 1 else f"d{p['digit_info'][j - 1]['digit']}")
        c = di["col"]
        recs.append(dict(row=r, uid=int(uid), pid=p["pid"], pos=int(j), digit=di["digit"], carry=int(di["carry_in"]), col=c,
                         n_answer=len(p["digit_info"]), n_digits=p["n_digits"], overflow=p["overflow"], demo=p["demo"],
                         family=p["family"], prev=prev,
                         a_c=(p["a"] // 10 ** c) % 10, b_c=(p["b"] // 10 ** c) % 10,
                         a_l=(p["a"] // 10 ** (c - 1)) % 10 if c >= 1 else -1, b_l=(p["b"] // 10 ** (c - 1)) % 10 if c >= 1 else -1))
    S = pd.DataFrame(recs)
    S = S[(S["pos"] >= 1) & (S["col"] >= 1) & (S["col"] < S["n_digits"])].reset_index(drop=True)
    S["pre"] = (S["digit"] - S["carry"]) % 10
    assert ((S["a_c"] + S["b_c"] + S["carry"]) % 10 == S["digit"]).all(), "column digits inconsistent with answer"
    rows = S["row"].to_numpy()
    lp = z["lp"][rows]
    fid = z["full_is_digit"][rows].astype(bool)
    U = np.load(f"{collect_run}/directions.npz")["U"].astype(np.float64)
    acoord = np.einsum("ij,ntj->nti", np.linalg.inv(U @ U.T), z["plane"][rows].astype(np.float64))
    feats = {"proj": z["proj"][rows][:, T0:].astype(np.float64).mean(1), "plane": acoord[:, T0:].mean(1)}
    out_digit = lp[:, -1].argmax(-1)
    correct = (out_digit == S["digit"].to_numpy()) & fid[:, -1]
    return S, feats, out_digit, correct, fid[:, -1]


def boot_diff(a, ga, b, gb, B=NB):
    """Row-mean(a) - row-mean(b), problem-cluster bootstrap (problems resampled jointly)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    up = np.unique(np.concatenate([ga, gb]))
    ia = {p: a[ga == p] for p in up}
    ib = {p: b[gb == p] for p in up}
    m = a.mean() - b.mean()
    bs = []
    for _ in range(B):
        sel = RNG.choice(up, len(up))
        aa = np.concatenate([ia[p] for p in sel])
        bb = np.concatenate([ib[p] for p in sel])
        if len(aa) and len(bb):
            bs.append(aa.mean() - bb.mean())
    return [round(float(m), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def boot_mean(a, ga, B=NB):
    a = np.asarray(a, float)
    up = np.unique(ga)
    ia = {p: a[ga == p] for p in up}
    bs = [np.concatenate([ia[p] for p in RNG.choice(up, len(up))]).mean() for _ in range(B)]
    return [round(float(a.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def residualise(X, tr, nuis, keep):
    """Fit X ~ [1, nuis, keep] on rows tr; return X minus the fitted nuis part (intercept and keep effects retained)."""
    D = np.column_stack([np.ones(len(X)), nuis, keep])
    coef, *_ = np.linalg.lstsq(D[tr], X[tr], rcond=None)
    k = nuis.shape[1]
    return X - nuis @ coef[1 : 1 + k]


def fractions(X, S, correct, folds, nuis):
    d, car = S["digit"].to_numpy(), S["carry"].to_numpy() == 1
    keep = np.column_stack([np.eye(10)[d][:, 1:], car.astype(float)])
    f_pre, f_next = np.full(len(S), np.nan), np.full(len(S), np.nan)
    for f in np.unique(folds):
        tr, te = (folds != f) & correct, folds == f
        Xr = residualise(X, tr, nuis, keep)
        cen = np.stack([Xr[tr & ~car & (d == k)].mean(0) for k in range(10)])
        for i in np.where(te)[0]:
            x, c = Xr[i], cen[d[i]]
            for arr, nb in ((f_pre, cen[(d[i] - 1) % 10]), (f_next, cen[(d[i] + 1) % 10])):
                ch = nb - c
                arr[i] = (x - c) @ ch / (ch @ ch)
    return f_pre, f_next


def matched(vals, S, sel_c, sel_n, by="full"):
    """Mean over matched cells of (carry mean - no-carry mean), weighted by carry count; problem-cluster bootstrap with
    problems resampled jointly. by='full': final digit x prev token x n_digits x col; by='digit': final digit only."""
    if by == "full":
        cell = (S["digit"].astype(str) + "|" + S["prev"] + "|" + S["n_digits"].astype(str) + "|" + S["col"].astype(str)).to_numpy()
    elif by == "digit_fold":  # same estimated reference geometry (fold-specific centroids) for both groups
        cell = (S["digit"].astype(str) + "|" + S["fold"].astype(str)).to_numpy()
    else:
        cell = S["digit"].astype(str).to_numpy()
    pid = S["pid"].to_numpy()
    vals = np.asarray(vals, float)
    sel_c, sel_n = sel_c & ~np.isnan(vals), sel_n & ~np.isnan(vals)

    def stat(idx):
        c_, v_ = cell[idx], vals[idx]
        sc, sn = sel_c[idx], sel_n[idx]
        num, den = 0.0, 0
        for k in np.unique(c_[sc]):
            a, b = v_[sc & (c_ == k)], v_[sn & (c_ == k)]
            if len(b):
                num += len(a) * (a.mean() - b.mean()); den += len(a)
        return num / den if den else np.nan, den

    allidx = np.arange(len(S))
    m, n = stat(allidx)
    up = np.unique(pid)
    byp = {p: np.where(pid == p)[0] for p in up}
    bsn = [stat(np.concatenate([byp[p] for p in RNG.choice(up, len(up))])) for _ in range(NB)]
    bs = [b[0] for b in bsn]
    good = {k for k in np.unique(cell[sel_c]) if (sel_n & (cell == k)).any()}
    mrows = sel_c & np.isin(cell, list(good))
    return {"diff": [round(float(m), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)],
            "n_carry_rows_matched": int(n), "n_cells": int(len(good)), "n_problems_matched_carry": int(len(np.unique(pid[mrows]))),
            "boot_matched_carry_rows_median_frac": round(float(np.median([b[1] for b in bsn]) / max(n, 1)), 3)}


def decode(X, S, train_mask, folds, nuis_in, resid_nuis=None):
    """Cross-fitted ridge decoder of (cos, sin) of the final digit from [X (optionally residualised), nuis_in]."""
    d = S["digit"].to_numpy()
    th = 2 * np.pi * d / 10
    Y = np.stack([np.cos(th), np.sin(th)], 1)
    pid = S["pid"].to_numpy()
    car = S["carry"].to_numpy() == 1
    keep = np.column_stack([np.eye(10)[d][:, 1:], car.astype(float)])
    out = np.full((len(S), 2), np.nan)
    for f in np.unique(folds):
        tr, te = (folds != f) & train_mask, folds == f
        Xf = X if resid_nuis is None else residualise(X, tr, resid_nuis, keep)
        F = np.column_stack([Xf, nuis_in]) if nuis_in is not None and nuis_in.shape[1] else Xf
        pred = fit_ridge_cv(F[tr], Y[tr], pid[tr])
        out[te] = pred(F[te])
    ph = np.degrees(np.arctan2(out[:, 1], out[:, 0]))
    return wrap(ph - np.degrees(th)), np.linalg.norm(out, axis=1)


def main(run, collect_run):
    out = f"{run}/analysis_audit"
    os.makedirs(out, exist_ok=True)
    S, feats, out_digit, correct, fid_K = load(run, collect_run)
    N = nuisance_design(S)
    I = onehots(S, ["a_c", "b_c", "a_l", "b_l"])
    NI = np.column_stack([N, I])
    NL = np.column_stack([N, onehots(S, ["a_l", "b_l"])])  # lower (carry-generating) column digits only
    car = S["carry"].to_numpy() == 1
    pid = S["pid"].to_numpy()
    d, pre = S["digit"].to_numpy(), S["pre"].to_numpy()
    folds = group_folds(pid, 5, 0)
    S["fold"] = folds
    sel_c, sel_n = car & correct, ~car & correct
    def rank_diag(nuis):  # rank of the residualisation design [1, nuis, final one-hots, carry] on training-fold correct rows
        keep = np.column_stack([np.eye(10)[d][:, 1:], car.astype(float)])
        D = np.column_stack([np.ones(len(S)), nuis, keep])
        return [int(np.linalg.matrix_rank(D[(folds != f) & correct])) for f in range(5)] + [int(D.shape[1])]

    summ = {"NOTE": "Intervals are problem-cluster bootstraps CONDITIONAL on the fitted residualisers / centroids / decoders "
                    "(not refitted per replicate); exploratory. 'input-dummy-adjusted' = residualised on one-hot operand digits "
                    "(this + lower column); its split from the final-digit effect is not fully identified (see rank diagnostics). "
                    "asym = f_pre - f_next is a directional contrast intended to reduce (not guaranteed to cancel) shared centroid-noise bias.",
            "rank_diag_geometry_all_correct_rows [per fold..., n_cols]": {"nuisance": rank_diag(N), "nuisance+input_digits": rank_diag(NI)},
            "rows": int(len(S)), "carry_correct": int(sel_c.sum()), "nocarry_correct": int(sel_n.sum()),
            "carry_wrong": int((car & ~correct).sum()), "problems": int(len(np.unique(pid))), "late_window": f"t={T0}..64 mean"}

    # A. geometry
    th_pre = 2 * np.pi * pre / 10
    spaces = {"proj": feats["proj"], "plane": feats["plane"], "inputs_only": onehots(S, ["a_c", "b_c", "a_l", "b_l"]),
              # reference scale, not a null: a state that stored ONLY the column sum (a_c + b_c) mod 10 (= pre-carry digit)
              "column_sum_only_reference": np.column_stack([np.eye(10)[pre], np.cos(th_pre), np.sin(th_pre)])}
    A = {}
    fp_store = {}
    for sp, X in spaces.items():
        for rn, nuis in (("nuisance", N), ("nuisance+lower_col_digits", NL), ("nuisance+input_digits", NI)):
            if sp in ("inputs_only", "column_sum_only_reference") and rn != "nuisance":
                continue
            fp, fn = fractions(X, S, correct, folds, nuis)
            fp_store[(sp, rn)] = fp
            g = fp - fn  # asymmetry: toward pre-carry neighbour minus toward the other neighbour (cancels shared centroid-noise bias)
            A[f"{sp} | resid {rn}"] = {
                "asym_pre_minus_next_digit_matched": matched(g, S, sel_c, sel_n, by="digit"),
                "asym_pre_minus_next_digit_x_fold_matched": matched(g, S, sel_c, sel_n, by="digit_fold"),
                "asym_pre_minus_next_full_matched": matched(g, S, sel_c, sel_n),
                "carry_asym": boot_mean(g[sel_c], pid[sel_c]), "nocarry_asym": boot_mean(g[sel_n], pid[sel_n]),
                "f_pre_digit_matched": matched(fp, S, sel_c, sel_n, by="digit"),
                "f_next_digit_matched": matched(fn, S, sel_c, sel_n, by="digit"),
                "carry_f_pre": boot_mean(fp[sel_c], pid[sel_c]), "nocarry_f_pre": boot_mean(fp[sel_n], pid[sel_n]),
                "carry_f_next": boot_mean(fn[sel_c], pid[sel_c]), "nocarry_f_next": boot_mean(fn[sel_n], pid[sel_n]),
                "carry_minus_nocarry_f_pre": boot_diff(fp[sel_c], pid[sel_c], fp[sel_n], pid[sel_n]),
                "carry_minus_nocarry_f_next": boot_diff(fn[sel_c], pid[sel_c], fn[sel_n], pid[sel_n]),
                "matched_cells_f_pre": matched(fp, S, sel_c, sel_n),
                "matched_cells_f_next": matched(fn, S, sel_c, sel_n),
                "wrong_carry_f_pre": boot_mean(fp[car & ~correct & ~np.isnan(fp)], pid[car & ~correct & ~np.isnan(fp)]),
            }
    summ["A_fraction_toward_pre_carry_centroid"] = A
    # fold-seed sensitivity of the digit-matched asymmetry (point estimates over 10 fold assignments)
    sens = {}
    for sp in ("proj", "plane"):
        for rn, nuis in (("nuisance", N), ("nuisance+input_digits", NI)):
            vals = []
            for sd in range(10):
                fo = group_folds(pid, 5, 100 + sd)
                fp_, fn_ = fractions(feats[sp], S, correct, fo, nuis)
                S_ = S.assign(fold=fo)
                vals.append(matched(fp_ - fn_, S_, sel_c, sel_n, by="digit")["diff"][0])
            sens[f"{sp} | resid {rn}"] = {"min": min(vals), "median": float(np.median(vals)), "max": max(vals)}
    summ["A_fold_seed_sensitivity_asym_digit_matched"] = sens
    # chord lengths between adjacent no-carry centroids (nuisance-residualised, all correct rows): small chords make f unstable
    ch = {}
    for sp in ("proj", "plane"):
        keep = np.column_stack([np.eye(10)[d][:, 1:], car.astype(float)])
        Xr = residualise(feats[sp], correct, N, keep)
        cen = np.stack([Xr[sel_n & (d == k)].mean(0) for k in range(10)])
        L_ = np.linalg.norm(np.roll(cen, -1, 0) - cen, axis=1)
        ch[sp] = {"adjacent_chord_len": L_.round(3).tolist(), "max_over_min": round(float(L_.max() / L_.min()), 2)}
    summ["A_adjacent_centroid_chords"] = ch
    # per final digit (proj, input-residualised): is the effect consistent across digits?
    fp = fp_store[("proj", "nuisance+input_digits")]
    summ["A_per_final_digit_proj_inputresid"] = {int(k): {"n_carry": int((sel_c & (d == k)).sum()), "n_nocarry": int((sel_n & (d == k)).sum()),
                                                         "carry_mean": round(float(np.nanmean(fp[sel_c & (d == k)])), 3) if (sel_c & (d == k)).any() else None,
                                                         "nocarry_mean": round(float(np.nanmean(fp[sel_n & (d == k)])), 3) if (sel_n & (d == k)).any() else None}
                                                 for k in range(10)}

    # B. decoders
    Bd = {}
    X = feats["proj"]
    variants = {
        "nocarry_trained (replication)": dict(train=~car & correct, nuis_in=N, resid=None),
        "nocarry_trained, input-dummy-adjusted [INCONCLUSIVE: not identified]": dict(train=~car & correct, nuis_in=None, resid=NI),
        "nocarry_trained, lower-col-dummy-adjusted [INCONCLUSIVE: not identified]": dict(train=~car & correct, nuis_in=None, resid=NL),
        "all_rows_trained (nuisance as inputs, no residualisation)": dict(train=correct, nuis_in=N, resid=None),
        "all_rows_trained, input-dummy-adjusted": dict(train=correct, nuis_in=None, resid=NI),
        "inputs_only (no state), nocarry_trained": dict(train=~car & correct, nuis_in=None, resid=None, X=NI),
    }
    for nm, v in variants.items():
        off, rad = decode(v.get("X", X), S, v["train"], folds, v["nuis_in"], v["resid"])
        ok = rad >= 0.3
        c_, n_ = sel_c & ok, sel_n & ok
        Bd[nm] = {"carry_offset_deg": boot_mean(off[c_], pid[c_]), "nocarry_offset_deg": boot_mean(off[n_], pid[n_]),
                  "carry_minus_nocarry_deg": boot_diff(off[c_], pid[c_], off[n_], pid[n_]),
                  "matched_cells_deg": matched(np.where(ok, off, np.nan), S, c_, n_),
                  "digit_matched_deg": matched(np.where(ok, off, np.nan), S, c_, n_, by="digit"),
                  "nocarry_heldout_acc": round(float((np.round((off[sel_n] + np.degrees(2 * np.pi * d[sel_n] / 10)) / 36) % 10 == d[sel_n]).mean()), 3),
                  "frac_radius_ok": round(float(ok[sel_c | sel_n].mean()), 3)}
        if nm.startswith("nocarry_trained (rep"):
            wr = car & ~correct & ok
            od = out_digit[wr]
            # offset relative to the model's own (wrong) output digit
            off_out = wrap(off[wr] + np.degrees(2 * np.pi * d[wr] / 10) - np.degrees(2 * np.pi * od / 10))
            isp = od == pre[wr]
            Bd[nm]["wrong_carry"] = {"n_radius_ok": int(wr.sum()), "n_total": int((car & ~correct).sum()),
                                     "frac_output_is_pre": round(float((od == pre[wr]).mean()), 3),
                                     "median_abs_offset_vs_own_output_deg": round(float(np.median(np.abs(off_out))), 1),
                                     "frac_within_18deg_of_own_output": round(float((np.abs(off_out) <= 18).mean()), 3),
                                     "frac_within_36deg_of_own_output": round(float((np.abs(off_out) <= 36).mean()), 3),
                                     "output_pre_subset_offset_vs_own_output_deg": boot_mean(off_out[isp], pid[wr][isp]),
                                     "output_other_subset_offset_vs_own_output_deg": boot_mean(off_out[~isp], pid[wr][~isp]) if (~isp).sum() > 1 else None,
                                     "frac_output_is_final_plus1": round(float((od == (d[wr] + 1) % 10).mean()), 3),
                                     "frac_output_not_digit": round(float((~fid_K[wr]).mean()), 3),
                                     "offset_vs_truth_deg": boot_mean(off[wr], pid[wr]),
                                     "offset_vs_own_output_deg": boot_mean(off_out, pid[wr])}
    summ["B_decoders_late"] = Bd
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"))
