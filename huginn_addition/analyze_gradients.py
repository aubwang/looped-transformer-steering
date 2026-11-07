"""Local readout-gradient analysis.
Usage: python -I huginn_addition/analyze_gradients.py results/<grad_run> results/<collect_run> [results/<sweep_run>]

For each eligible row (baseline full-vocab argmax == correct digit token) and offset j, the gradient of the
target-vs-source log odds L_j = lp_{d+j} - lp_d w.r.t. the position's state, restricted to the circle plane, gives the
locally preferred push angle (in the same circle-coordinate parameterisation as the sweep: v(phi) = [cos phi, sin phi] U,
normalised). Compared with the geometric chord prediction, per row, per source digit, and with the finite-size sweep.
Finite differences (eps 0.25 and 2.0 along orthonormal plane axes) validate the gradients and measure curvature.
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
from matplotlib.colors import LinearSegmentedColormap

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
})
RNG = np.random.default_rng(0)
th = lambda x: 2 * np.pi * np.asarray(x) / 10.0
f_ = lambda x: np.stack([np.cos(th(x)), np.sin(th(x))], -1)
PHI = np.deg2rad(np.arange(0, 360, 0.5))
DIR = np.stack([np.cos(PHI), np.sin(PHI)], 1)  # (M, 2)


def circ_mean(a_deg, w=None):
    a = np.deg2rad(a_deg)
    w = np.ones_like(a) if w is None else w
    c, s = (w * np.cos(a)).sum() / w.sum(), (w * np.sin(a)).sum() / w.sum()
    return (np.rad2deg(np.arctan2(s, c)) + 360) % 360, float(np.hypot(c, s))


def wrap(x):
    return (x + 180) % 360 - 180


def main(run, collect_run, sweep_run=None):
    out = f"{run}/analysis"
    os.makedirs(out, exist_ok=True)
    z = np.load(f"{run}/gradients.npz")
    U = np.load(f"{collect_run}/directions.npz")["U"].astype(np.float64)
    Gm = U @ U.T
    dirnorm = np.sqrt(np.einsum("mi,ij,mj->m", DIR, Gm, DIR))  # ||[cos,sin] U|| over the angle grid
    d = z["digit"]
    base = z["base_lp"]
    elig = (base.argmax(1) == d) & z["full_argmax_is_digit"]
    pid = z["pid"]
    gU, gQ, gN, fd, eps = z["g_U"], z["g_Q"], z["g_norm"], z["fd_lp"], z["eps"]
    n = len(d)
    summary = {"rows": int(n), "eligible_rows": int(elig.sum()), "eligible_problems": int(len(np.unique(pid[elig])))}

    def pref_angle(vec_U):  # vec_U: (n, 2) gradient in circle coords; returns preferred phi (deg) per row
        score = (vec_U @ DIR.T) / dirnorm[None]  # (n, M)
        return np.rad2deg(PHI[score.argmax(1)])

    rows = []
    for j in range(1, 10):
        t = (d + j) % 10
        gj = gU[np.arange(n), t] - gU[np.arange(n), d]  # U grad L_j
        obs = (pref_angle(gj) - np.rad2deg(th(d))) % 360
        chord = (f_(t) - f_(d)) @ Gm  # geometric prediction in circle coords
        pred = (pref_angle(chord) - np.rad2deg(th(d))) % 360
        mag = np.linalg.norm(gQ[np.arange(n), t] - gQ[np.arange(n), d], axis=1)
        for r in range(n):
            rows.append({"row": r, "pid": pid[r], "digit": d[r], "j": j, "obs": obs[r], "pred": pred[r], "err": wrap(obs[r] - pred[r]),
                         "mag": mag[r], "elig": elig[r]})
    R = pd.DataFrame(rows)
    E = R[R["elig"]]

    per_j = []
    for j, g in E.groupby("j"):
        pm = g.groupby("pid")["obs"].apply(lambda a: circ_mean(a.to_numpy())[0])  # problem-level circular means
        m, Rl = circ_mean(pm.to_numpy())
        bs = [circ_mean(RNG.choice(pm.to_numpy(), len(pm)))[0] for _ in range(1000)]
        dev = wrap(np.array(bs) - m)
        pred = circ_mean(g["pred"].to_numpy())[0]
        per_j.append({"j": j, "obs_mean": m, "ci_lo": m + np.percentile(dev, 2.5), "ci_hi": m + np.percentile(dev, 97.5),
                      "resultant": Rl, "pred": pred, "err": wrap(m - pred), "median_abs_row_err": float(g["err"].abs().median()),
                      "frac_rows_within_30deg": float((g["err"].abs() < 30).mean()), "median_mag": float(g["mag"].median())})
    PJ = pd.DataFrame(per_j)
    summary["per_offset"] = PJ.round(2).to_dict("records")

    # per source digit x offset: circular-mean error
    H = np.full((10, 9), np.nan)
    Hn = np.zeros((10, 9), int)
    for (dd, j), g in E.groupby(["digit", "j"]):
        pm = g.groupby("pid")["obs"].apply(lambda a: circ_mean(a.to_numpy())[0]).to_numpy()  # equal weight per problem
        H[dd, j - 1] = wrap(circ_mean(pm)[0] - circ_mean(g["pred"].to_numpy())[0])
        Hn[dd, j - 1] = g["pid"].nunique()
    summary["per_digit_problem_counts"] = {int(k): int(v) for k, v in enumerate(Hn[:, 0])}

    # in-plane fraction of per-digit gradient norm (target and source digits separately)
    fr = (np.linalg.norm(gQ, axis=-1) / gN) ** 2  # (n, 10)
    summary["inplane_sqnorm_frac_of_digit_gradients"] = {
        "median_own_digit": float(np.median(fr[np.arange(n), d][elig])),
        "median_all_digits": float(np.median(fr[elig])), "random_2d_plane_expectation": 2 / U.shape[1]}

    # finite-difference validation and curvature (target-vs-source log odds along the two orthonormal axes)
    fdv = {}
    for ei, e_ in enumerate(eps):
        slopes_fd, slopes_ad, curv = [], [], []
        for j in range(1, 10):
            t = (d + j) % 10
            L = lambda arr: arr[np.arange(n), t] - arr[np.arange(n), d]
            L0 = L(base)
            for ax in range(2):
                Lp, Lm = L(fd[:, ei, ax, 0]), L(fd[:, ei, ax, 1])
                slopes_fd.append(((Lp - Lm) / (2 * e_))[elig])
                slopes_ad.append((gQ[np.arange(n), t, ax] - gQ[np.arange(n), d, ax])[elig])
                curv.append(((Lp + Lm) / 2 - L0)[elig])
        a, b = np.concatenate(slopes_fd), np.concatenate(slopes_ad)
        fdv[float(e_)] = {"corr_fd_vs_autograd": float(np.corrcoef(a, b)[0, 1]),
                          "median_rel_err": float(np.median(np.abs(a - b) / (np.abs(b) + 1e-6))),
                          "mean_symmetric_part_nats": float(np.concatenate(curv).mean())}
    summary["finite_difference_check"] = fdv

    # digit-balanced pooled mean per offset (each source digit weighted equally, problem-level within digit)
    bal = {}
    for j, g in E.groupby("j"):
        dm = [circ_mean(gg.groupby("pid")["obs"].apply(lambda a: circ_mean(a.to_numpy())[0]).to_numpy())[0] for _, gg in g.groupby("digit")]
        bal[int(j)] = round(circ_mean(np.array(dm))[0], 1)
    summary["digit_balanced_obs_mean"] = bal

    # target vs source decomposition of grad L_j = grad lp_t - grad lp_d (in circle coords)
    gd = gU[np.arange(n), d]  # source term (enters with minus sign)
    src_ang = (pref_angle(-gd) - np.rad2deg(th(d))) % 360
    src_pred = (pref_angle(-f_(d) @ Gm) - np.rad2deg(th(d))) % 360
    dec = []
    for j in range(1, 10):
        t = (d + j) % 10
        gt = gU[np.arange(n), t]
        tg_ang = (pref_angle(gt) - np.rad2deg(th(d))) % 360
        tg_pred = (pref_angle(f_(t) @ Gm) - np.rad2deg(th(d))) % 360  # "toward t" radial direction
        mt = ((gt @ DIR.T) / dirnorm[None]).max(1)  # max directional derivative per unit push
        ms = ((-gd @ DIR.T) / dirnorm[None]).max(1)
        k = elig
        dec.append({"j": j, "target_term_angle": circ_mean(tg_ang[k])[0], "target_term_pred": circ_mean(tg_pred[k])[0],
                    "target_term_R": circ_mean(tg_ang[k])[1],
                    "median_target_to_source_strength": float(np.median(mt[k] / ms[k]))})
    summary["source_term"] = {"angle": circ_mean(src_ang[elig])[0], "pred_away_from_d": circ_mean(src_pred[elig])[0],
                              "R": circ_mean(src_ang[elig])[1]}
    summary["target_term"] = pd.DataFrame(dec).round(3).to_dict("records")

    # absolute Taylor residuals |L(+-eps) - L0 -+ eps*grad|, relative to |eps*grad|
    tay = {}
    for ei, e_ in enumerate(eps):
        res, lin = [], []
        for j in range(1, 10):
            t = (d + j) % 10
            L = lambda arr: arr[np.arange(n), t] - arr[np.arange(n), d]
            L0 = L(base)
            for ax_ in range(2):
                gl = gQ[np.arange(n), t, ax_] - gQ[np.arange(n), d, ax_]
                for si, sg in enumerate((1, -1)):
                    res.append(np.abs(L(fd[:, ei, ax_, si]) - L0 - sg * e_ * gl)[elig]); lin.append(np.abs(e_ * gl)[elig])
        r, l_ = np.concatenate(res), np.concatenate(lin)
        tay[float(e_)] = {"median_abs_residual_nats": float(np.median(r)), "p95_abs_residual_nats": float(np.percentile(r, 95)),
                          "median_residual_over_linear_term": float(np.median(r / (l_ + 1e-6))),
                          "median_abs_linear_term_nats": float(np.median(l_))}
    summary["taylor_residual_abs"] = tay

    # compare with sweep (finite size)
    sw = None
    if sweep_run and os.path.exists(f"{sweep_run}/analysis/summary.json"):
        ss = json.load(open(f"{sweep_run}/analysis/summary.json"))["angular"]
        sw = {k: v["psi_obs"] for k, v in ss.items()}
        summary["sweep_psi_for_comparison"] = sw

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    unwrap = lambda a: np.where(np.asarray(a) < 60, np.asarray(a) + 360, np.asarray(a))
    js = PJ["j"].to_numpy()
    ax.plot(js, unwrap(PJ["pred"]), color=INK2, ls="--", lw=1.5, label="circle-geometry prediction")
    jit = RNG.uniform(-0.18, 0.18, len(E))
    ax.scatter(E["j"] + jit, unwrap(E["obs"]), s=5, color=CAT[0], alpha=0.12, edgecolor="none", label="each row (problem × position × demo)")
    ax.errorbar(js, unwrap(PJ["obs_mean"]), yerr=[PJ["obs_mean"] - PJ["ci_lo"], PJ["ci_hi"] - PJ["obs_mean"]], color=CAT[0], marker="o",
                ms=7, lw=0, elinewidth=2, label="local gradient, mean over problems (95% CI)")
    if sw:
        for (k, v), c in zip(sorted(sw.items()), (CAT[1], CAT[3])):
            ax.plot(js, unwrap(v), color=c, marker="s", ms=4, lw=1, label=f"finite push sweep ({float(k):g}× notch)")
    ax.set_xticks(js); ax.set_xlabel("digit offset j (target = d + j)")
    ax.set_ylabel("push angle that most favors d+j\n(degrees, relative to current digit)")
    ax.set_title("Locally preferred push direction vs circle prediction")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}/1_local_preferred_angle.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    im = ax.imshow(H, cmap=DIV, vmin=-90, vmax=90, aspect="auto")
    for dd in range(10):
        for j in range(9):
            if not np.isnan(H[dd, j]):
                ax.text(j, dd, f"{H[dd, j]:.0f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(9), range(1, 10)); ax.set_yticks(range(10), [f"{k} (n={Hn[k, 0]})" for k in range(10)])
    ax.set_xlabel("digit offset j"); ax.set_ylabel("current (source) digit, n problems")
    ax.set_title("Local angle error by source digit (problem-weighted; 0 = matches circle)"); ax.grid(False)
    fig.colorbar(im, ax=ax, label="observed − predicted (deg)")
    fig.tight_layout(); fig.savefig(f"{out}/2_error_by_source_digit.png", dpi=150); plt.close(fig)

    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"), sys.argv[3].rstrip("/") if len(sys.argv) > 3 else None)
