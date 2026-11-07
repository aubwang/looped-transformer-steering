"""Digit-centered logit-gradient reanalysis (no model run; Astra's suggestion).
Usage: python -I huginn_addition/analyze_centered.py results/<grad_run> results/<collect_run>

grad lp_q = grad z_q - sum_k p_k grad z_k; the normaliser term is shared across q, so subtracting the across-digit mean of
the 10 digit log-prob gradients gives exact gradients of digit-centered logits  z~_q = z_q - mean_k z_k  (k over digits).
All quantities live in the circle plane, stored in "U c" form (U @ vector, 2-d); the induced state-space metric on in-plane
gradient components is inv(Gram(U)).

1. Centroid geometry: nuisance-adjusted class means of the collect run (the circle-fit problems; disjoint from the gradient problems), projected into the plane.
   Least-squares circle coordinates a_k = Gram^-1 U mu_k vs the uniform circle.
2. Which geometry predicts the local contrast gradient grad(z_d+j - z_d)?  (a) uniform chord Gram (f(t)-f(d));
   (b) empirical centroid chord U (mu_t - mu_d); (c) pure "away from d" -Gram f(d).  Preferred-angle error per row.
3. Split of the contrast into a target-specific part T = grad z~_t and a source part S = -grad z~_d (exact, T + S = contrast):
   metric share of each, and whether T points toward t (uniform or empirical centroid).
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
from analyze_gradients import CAT, DIR, GRID, INK, INK2, PHI, circ_mean, f_, th, wrap  # noqa: E402  (also sets rcParams)
from shape_analysis import nuisance_design  # noqa: E402

RNG = np.random.default_rng(1)


def centroids(collect_run):
    z = np.load(f"{collect_run}/states.npz")
    X, rows = z["states"], z["rows"]
    probs = {p["uid"]: p for p in json.load(open(f"{collect_run}/problems.json"))}
    df = pd.DataFrame(rows, columns=["uid", "seed", "pos"])
    df["row"] = np.arange(len(df))
    recs = []
    for (uid, pos), g in df.groupby(["uid", "pos"]):  # same construction as fit_circle.py
        p = probs[uid]
        di = p["digit_info"][pos]
        prev = "eq" if pos == 0 else (f"sp{p['digit_info'][0]['digit']}" if pos == 1 else f"d{p['digit_info'][pos - 1]['digit']}")
        recs.append(dict(uid=uid, pid=p["pid"], pos=pos, digit=di["digit"], col=di["col"], n_answer=len(p["digit_info"]),
                         n_digits=p["n_digits"], overflow=p["overflow"], demo=p["demo"], family=p["family"], prev=prev,
                         rows=g["row"].tolist(), a=p["a"], b=p["b"]))
    S = pd.DataFrame(recs)
    S = S[S["pos"] >= 1].reset_index(drop=True)
    Y = np.stack([X[r].mean(0) for r in S["rows"]]).astype(np.float64)
    N1 = np.column_stack([np.ones(len(S)), nuisance_design(S)])
    coef, *_ = np.linalg.lstsq(N1, Y, rcond=None)
    Yr = Y - N1 @ coef
    d = S["digit"].to_numpy()
    mu = np.stack([Yr[d == k].mean(0) for k in range(10)])
    return mu, set(zip(S["a"], S["b"])), np.bincount(d, minlength=10)


def main(run, collect_run):
    out = f"{run}/analysis_centered"
    os.makedirs(out, exist_ok=True)
    U = np.load(f"{collect_run}/directions.npz")["U"].astype(np.float64)
    Gm = U @ U.T
    M = np.linalg.inv(Gm)
    dirnorm = np.sqrt(np.einsum("mi,ij,mj->m", DIR, Gm, DIR))
    pref = lambda v: np.rad2deg(PHI[((v @ DIR.T) / dirnorm[None]).argmax(1)])  # v: (n, 2) in "U c" form
    strength = lambda v: np.sqrt(np.einsum("ni,ij,nj->n", v, M, v))  # state-space norm of the in-plane component

    z = np.load(f"{run}/gradients.npz")
    d, pid, base, gU = z["digit"], z["pid"], z["base_lp"], z["g_U"].astype(np.float64)
    elig = (base.argmax(1) == d) & z["full_argmax_is_digit"]
    n = len(d)
    ar = np.arange(n)
    gprobs = json.load(open(f"{run}/problems.json"))
    mu, fit_pairs, cnt = centroids(collect_run)
    overlap = sum((p["a"], p["b"]) in fit_pairs for p in gprobs)

    # 1. centroid geometry in circle coordinates
    P = (U @ mu.T).T  # (10, 2) "U c" form
    A = P @ M  # LS circle coords (M symmetric)
    ang = (np.rad2deg(np.arctan2(A[:, 1], A[:, 0])) + 360) % 360
    rad = np.linalg.norm(A, axis=1)
    summary = {"rows": int(n), "eligible_rows": int(elig.sum()), "grad_problems_overlapping_fit_pairs": int(overlap),
               "centroid_counts_collect": cnt.tolist(),
               "centroid_angle_deg": ang.round(1).tolist(), "centroid_angle_minus_uniform_deg": wrap(ang - 36 * np.arange(10)).round(1).tolist(),
               "centroid_radius": rad.round(3).tolist(),
               "centroid_step_deg": wrap(np.roll(ang, -1) - ang).round(1).tolist()}

    # 2. which geometry predicts the contrast gradient
    C = gU - gU.mean(1, keepdims=True)  # centered-logit gradients (n, 10, 2)
    rows = []
    for j in range(1, 10):
        t = (d + j) % 10
        con = C[ar, t] - C[ar, d]  # == gU[t] - gU[d]
        obs = pref(con)
        preds = {"uniform_chord": (f_(t) - f_(d)) @ Gm, "centroid_chord": P[t] - P[d], "away_from_d": -f_(d) @ Gm}
        T, S_ = C[ar, t], -C[ar, d]
        cc = np.einsum("ni,ij,nj->n", con, M, con)
        shareS = np.einsum("ni,ij,nj->n", S_, M, con) / cc
        tobs = pref(T)
        for r in np.where(elig)[0]:
            rec = {"row": r, "pid": pid[r], "digit": d[r], "j": j, "shareS": shareS[r], "T_over_S": strength(T[r:r + 1])[0] / strength(S_[r:r + 1])[0]}
            for k, v in preds.items():
                rec[f"err_{k}"] = wrap(obs[r] - pref(v[r:r + 1] if v.ndim == 2 else v[None])[0])
            rec["T_err_uniform"] = wrap(tobs[r] - pref((f_(t[r:r + 1])) @ Gm)[0])
            rec["T_err_centroid"] = wrap(tobs[r] - pref((P[t[r]] - P.mean(0))[None])[0])
            rows.append(rec)
    R = pd.DataFrame(rows)

    def pl_med_abs(g, col):  # problem-level: median |err| within problem, then median over problems
        return float(g.groupby("pid")[col].apply(lambda a: np.abs(a).median()).median())

    def boot_diff(g, c1, c2, B=2000):  # problem-cluster bootstrap of mean(|err c1| - |err c2|)
        pm = g.groupby("pid").apply(lambda x: (x[c1].abs() - x[c2].abs()).mean()).to_numpy()
        bs = np.array([RNG.choice(pm, len(pm)).mean() for _ in range(B)])
        return float(pm.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    per_j = []
    for j, g in R.groupby("j"):
        rec = {"j": j}
        for k in ("uniform_chord", "centroid_chord", "away_from_d"):
            rec[f"medabs_{k}"] = pl_med_abs(g, f"err_{k}")
        rec["centroid_minus_uniform_abs_err"] = boot_diff(g, "err_centroid_chord", "err_uniform_chord")
        rec["median_source_share"] = float(g.groupby("pid")["shareS"].median().median())
        rec["median_T_over_S"] = float(g.groupby("pid")["T_over_S"].median().median())
        rec["T_medabs_err_uniform"] = pl_med_abs(g, "T_err_uniform")
        rec["T_medabs_err_centroid"] = pl_med_abs(g, "T_err_centroid")
        per_j.append(rec)
    PJ = pd.DataFrame(per_j)
    summary["per_offset"] = PJ.round(3).to_dict("records")
    allj = {}
    for k in ("uniform_chord", "centroid_chord", "away_from_d"):
        allj[k] = pl_med_abs(R, f"err_{k}")
    allj["centroid_minus_uniform_abs_err_all_j"] = boot_diff(R, "err_centroid_chord", "err_uniform_chord")
    summary["all_offsets_median_abs_err"] = allj
    # source term direction (independent of j)
    Sang = wrap(pref(-C[ar, d]) - np.rad2deg(th(d)))[elig]
    summary["source_term_angle_rel_theta_d"] = {"circ_mean": circ_mean(Sang % 360)[0], "R": circ_mean(Sang % 360)[1]}

    # ---- figures
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3))
    ax = axs[0]
    tt = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(tt), np.sin(tt), color=GRID, lw=1.5, zorder=0)
    ax.scatter(np.cos(th(np.arange(10))), np.sin(th(np.arange(10))), s=30, facecolor="none", edgecolor=INK2, label="uniform circle")
    ax.scatter(A[:, 0], A[:, 1], s=36, color=CAT[0], label="digit centroids (circle-fit problems)", zorder=3)
    for k in range(10):
        ax.annotate(str(k), A[k], xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_aspect("equal"); ax.set_title("Digit centroids in circle coordinates"); ax.legend(fontsize=7, loc="lower left")
    ax.set_xlabel("cos coordinate"); ax.set_ylabel("sin coordinate")

    ax = axs[1]
    js = PJ["j"].to_numpy()
    for k, c, lab in (("uniform_chord", INK2, "uniform circle chord"), ("centroid_chord", CAT[0], "centroid chord"),
                      ("away_from_d", CAT[1], "pure 'away from current digit'")):
        ax.plot(js, PJ[f"medabs_{k}"], marker="o", ms=5, color=c, label=lab)
    ax.set_xticks(js); ax.set_xlabel("digit offset j"); ax.set_ylabel("median |angle error| (deg, problem-level)")
    ax.set_title("Which geometry predicts\nthe local readout direction?"); ax.legend(fontsize=7.5)

    ax = axs[2]
    ax.bar(js, PJ["median_source_share"], color=CAT[1], width=0.6, label="source part  (−∇z̃_current)")
    ax.bar(js, 1 - PJ["median_source_share"], bottom=PJ["median_source_share"], color=CAT[0], width=0.6, label="target part  (∇z̃_target)")
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(js); ax.set_xlabel("digit offset j"); ax.set_ylabel("share of in-plane contrast (median)")
    ax.set_title("What drives the push direction:\ntarget or source?"); ax.legend(fontsize=7.5, loc="upper center"); ax.set_ylim(0, 1.15)
    fig.tight_layout(); fig.savefig(f"{out}/centered_readout.png", dpi=150); plt.close(fig)

    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"))
