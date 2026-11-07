"""Analyze the angular sweep.  Usage: python -I huginn_addition/analyze_sweep.py results/<sweep_run> results/<collect_run>

Eligible rows: baseline full-vocab argmax == correct digit token (fixed in advance; all-row sensitivity reported).
Observed gain map: G(phi_rel, j) = paired change in log-odds of digit d+j vs d (push minus baseline), problem-weighted.
Geometric prediction (circle hypothesis): the change toward d+j is proportional to the push direction's alignment with
the chord from d to d+j in circle coordinates, [cos phi, sin phi] Gram(U) (f(d+j) - f(d)), whose peak angle relative to
the current digit is 90 + 18 j degrees for a circle. Tests:
  1. correlation of observed vs predicted maps, against predictions from scrambled digit orders (1000 permutations);
  2. per-offset sinusoid fits: G_j(phi) ~ a_j + b_j cos(phi - psi_j); does psi_j step by 18 degrees per offset?
  3. scrambled-chord decomposition: full vs in-plane projection vs orthogonal residual.
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
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])  # red <- gray -> blue
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
})
RNG = np.random.default_rng(0)
th = lambda x: 2 * np.pi * np.asarray(x) / 10.0
f_ = lambda x: np.stack([np.cos(th(x)), np.sin(th(x))], -1)


def main(run, collect_run):
    out = f"{run}/analysis"
    os.makedirs(out, exist_ok=True)
    df = pd.read_parquet(f"{run}/sweep.parquet")
    U = np.load(f"{collect_run}/directions.npz")["U"].astype(np.float64)
    Gram = U @ U.T
    LP = df[[f"lp{q}" for q in range(10)]].to_numpy()
    base = df[df["kind"] == "baseline"].set_index("uid")
    elig = set(base.index[base["full_argmax_is_digit"] & (base["digit_argmax"] == base["digit"])])
    bLP = base[[f"lp{q}" for q in range(10)]]
    summary = {"run": run, "prompts": int(df["uid"].nunique()), "eligible_prompts": len(elig),
               "eligible_problems": int(base.loc[list(elig), "pid"].nunique())}

    # ---------------- observed gain map
    A = df[df["kind"] == "angle"].copy()
    aLP = LP[A.index]
    bl = bLP.loc[A["uid"]].to_numpy()
    d = A["digit"].to_numpy()
    rows = np.arange(len(A))
    for j in range(1, 10):
        t = (d + j) % 10
        A[f"g{j}"] = (aLP[rows, t] - aLP[rows, d]) - (bl[rows, t] - bl[rows, d])
    A["elig"] = A["uid"].isin(elig)
    phis = np.sort(A["phi_rel"].unique())
    sizes = np.sort(A["size"].unique())

    def gain_map(sub):
        per = sub.groupby(["pid", "phi_rel"])[[f"g{j}" for j in range(1, 10)]].mean()
        return per.groupby("phi_rel").mean().reindex(phis).to_numpy()  # (phi, 9)

    # ---------------- predicted maps
    def pred_rows(sub, order=None):
        """Per-row predicted gain (phi, j) under a digit->angle order (None = true order); returns (n, 9)."""
        dd = sub["digit"].to_numpy()
        ph = th(dd) + np.deg2rad(sub["phi_rel"].to_numpy())
        v = np.stack([np.cos(ph), np.sin(ph)], 1) @ Gram  # (n, 2)
        o = np.arange(10) if order is None else order
        return np.stack([(v * (f_(o[(dd + j) % 10]) - f_(o[dd % 10]))).sum(1) for j in range(1, 10)], 1)

    def pred_map(sub, order=None):
        P = pd.DataFrame(pred_rows(sub, order), columns=[f"g{j}" for j in range(1, 10)])
        P["pid"], P["phi_rel"] = sub["pid"].to_numpy(), sub["phi_rel"].to_numpy()
        return P.groupby(["pid", "phi_rel"]).mean().groupby("phi_rel").mean().reindex(phis).to_numpy()

    res = {}
    for size in sizes:
        sub = A[(A["size"] == size) & A["elig"]].reset_index(drop=True)
        obs = gain_map(sub)
        pt = pred_map(sub)
        r_true = float(np.corrcoef(obs.ravel(), pt.ravel())[0, 1])
        r_perm = np.array([np.corrcoef(obs.ravel(), pred_map(sub, RNG.permutation(10)).ravel())[0, 1] for _ in range(1000)])
        r_rot = [float(np.corrcoef(obs.ravel(), pred_map(sub, (np.arange(10) + c) % 10).ravel())[0, 1]) for c in range(10)]
        # sinusoid fits per offset: g = a + b cos(phi) + c sin(phi)
        Xs = np.column_stack([np.ones(len(phis)), np.cos(np.deg2rad(phis)), np.sin(np.deg2rad(phis))])
        coef = np.linalg.lstsq(Xs, obs, rcond=None)[0]  # (3, 9)
        psi = (np.rad2deg(np.arctan2(coef[2], coef[1])) + 360) % 360
        amp = np.hypot(coef[1], coef[2])
        # bootstrap psi by problem
        pids = sub["pid"].unique()
        per = {p: g for p, g in sub.groupby("pid")}
        bpsi = []
        for _ in range(300):
            bs = pd.concat([per[p].assign(pid=i) for i, p in enumerate(RNG.choice(pids, len(pids)))])
            o_ = gain_map(bs)
            c_ = np.linalg.lstsq(Xs, o_, rcond=None)[0]
            bpsi.append((np.rad2deg(np.arctan2(c_[2], c_[1])) + 360) % 360)
        bpsi = np.array(bpsi)
        # circular CI: deviations from point estimate
        dev = (bpsi - psi + 180) % 360 - 180
        lo, hi = psi + np.percentile(dev, 2.5, 0), psi + np.percentile(dev, 97.5, 0)
        pred_psi = (90 + 18 * np.arange(1, 10)) % 360
        cpt = np.linalg.lstsq(Xs, pt, rcond=None)[0]
        pred_psi_gram = (np.rad2deg(np.arctan2(cpt[2], cpt[1])) + 360) % 360
        err = (psi - pred_psi_gram + 180) % 360 - 180
        res[float(size)] = {
            "r_true_order": r_true, "perm_p": float((1 + (r_perm >= r_true).sum()) / 1001),
            "perm_r_quantiles": np.percentile(r_perm, [50, 95, 100]).tolist(),
            "r_by_rotation_offset": r_rot,
            "psi_obs": psi.tolist(), "psi_ci": np.stack([lo, hi], 1).tolist(), "psi_pred_circle": pred_psi.tolist(),
            "psi_pred_gram": pred_psi_gram.tolist(), "psi_abs_err_deg": np.abs(err).tolist(),
            "amp": amp.tolist(), "offset_const": coef[0].tolist(),
            "frac_variance_explained_by_sinusoids": float(1 - ((obs - Xs @ coef) ** 2).sum() / ((obs - obs.mean(0)) ** 2).sum()),
        }
        if size == sizes.max():
            OBS, PRED, PSI, PCI, PPG, RPERM, RTRUE = obs, pt, psi, (lo, hi), pred_psi_gram, r_perm, r_true
    summary["angular"] = res
    # all-row sensitivity
    summary["r_true_all_rows"] = {float(s_): float(np.corrcoef(gain_map(A[A["size"] == s_]).ravel(), pred_map(A[A["size"] == s_].reset_index(drop=True)).ravel())[0, 1]) for s_ in sizes}

    # ---------------- scrambled decomposition
    Sx = df[df["kind"] == "scr_part"].copy()
    Sx["shift"], Sx["scr"] = Sx["shift"].astype(int), Sx["scr"].astype(int)
    sLP = LP[Sx.index]
    blS = bLP.loc[Sx["uid"]].to_numpy()
    dS = Sx["digit"].to_numpy()
    tS = (dS + Sx["shift"].to_numpy().astype(int)) % 10
    r_ = np.arange(len(Sx))
    Sx["dlogodds"] = (sLP[r_, tS] - sLP[r_, dS]) - (blS[r_, tS] - blS[r_, dS])
    Sx = Sx[Sx["uid"].isin(elig)]
    nrm = Sx.pivot_table(index=["uid", "scr", "shift", "size"], columns="part", values="applied_norm")
    frac = ((nrm["inplane"] / nrm["full"]) ** 2).groupby(["scr", "shift"]).mean()
    tab = Sx.groupby(["scr", "shift", "size", "part"]).agg(dlogodds=("dlogodds", "mean"), inplane_frac=("inplane_frac", "first"),
                                                           norm=("applied_norm", "mean")).reset_index()
    tab.to_csv(f"{out}/scrambled_parts.csv", index=False)
    piv = tab[tab["size"] == 1.0].pivot_table(index=["scr", "shift"], columns="part", values="dlogodds")
    piv["inplane_plus_residual"] = piv["inplane"] + piv["residual"]
    summary["scrambled_parts_alpha1"] = piv.round(3).reset_index().to_dict("records")
    summary["scrambled_inplane_sqnorm_frac_eligible"] = {f"{k[0]},{k[1]}": round(float(v), 3) for k, v in frac.items()}

    # ---------------- figures
    vmax = np.abs(OBS).max()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, M, t in ((axes[0], OBS, f"Observed (push = {sizes.max():g}× one-notch chord)"), (axes[1], PRED / np.abs(PRED).max() * vmax, "Circle-geometry prediction (arbitrary scale)")):
        im = ax.imshow(M.T, aspect="auto", origin="lower", cmap=DIV, vmin=-vmax, vmax=vmax,
                       extent=[phis[0] - 7.5, phis[-1] + 7.5, 0.5, 9.5])
        ax.set_xlabel("push angle relative to the current digit (degrees)"); ax.set_ylabel("digit offset j (target = d + j)")
        ax.set_yticks(range(1, 10)); ax.set_xticks(range(0, 360, 45)); ax.grid(False); ax.set_title(t)
    fig.colorbar(im, ax=axes, shrink=0.85, label="observed: change in log-odds of d+j vs d")
    fig.savefig(f"{out}/1_gain_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    js = np.arange(1, 10)
    unwrap = lambda a: np.where(a < 60, a + 360, a)
    axes[0].plot(js, unwrap(PPG), color=INK2, ls="--", lw=1.5, label="circle prediction")
    axes[0].errorbar(js, unwrap(PSI), yerr=[PSI - PCI[0], PCI[1] - PSI], color=CAT[0], marker="o", ms=6, lw=0, elinewidth=1.5,
                     label=f"observed peak of fitted sinusoid (push {sizes.max():g}×, 95% CI)")
    axes[0].set_xlabel("digit offset j"); axes[0].set_ylabel("push angle that most favors digit d+j (deg)")
    axes[0].set_xticks(js); axes[0].set_title("Preferred push angle: matches for near digits, flat for far digits"); axes[0].legend(fontsize=8)
    axes[1].hist(RPERM, bins=40, color=CAT[1], alpha=0.6, label="predictions from scrambled digit orders (1000)")
    axes[1].axvline(RTRUE, color=INK, lw=2)
    axes[1].annotate("true numerical order", (RTRUE, axes[1].get_ylim()[1] * 0.8), xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=9)
    axes[1].set_xlabel("correlation between observed and predicted gain maps"); axes[1].set_ylabel("count")
    axes[1].set_title(f"True order vs scrambled-label predictions (push {sizes.max():g}×)"); axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}/2_phase_and_null.png", dpi=150); plt.close(fig)

    t1 = tab[tab["size"] == 1.0]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels, x = [], 0
    for (i, sh), g in t1.groupby(["scr", "shift"]):
        for part, c in (("full", CAT[1]), ("inplane", CAT[0]), ("residual", INK2)):
            v = g[g["part"] == part]["dlogodds"].iloc[0]
            ax.bar(x, v, color=c, width=0.8, label=part if not labels else None)
            x += 1
        labels.append((x - 2, f"scr #{i}\n→ d{sh:+d}"))
        x += 1
    ax.set_xticks([l[0] for l in labels], [l[1] for l in labels], fontsize=8)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel("change in log-odds toward target")
    ax.set_title("Scrambled pushes (1×): most of the effect comes from the true-plane part")
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_[:3], ["full scrambled push", "its part inside the true circle plane", "its leftover (outside) part"], fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/3_scrambled_parts.png", dpi=150); plt.close(fig)

    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"))
