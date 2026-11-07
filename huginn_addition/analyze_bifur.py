"""Bifurcation of the '=' token's loop dynamics along a family of contextual inputs (intervene_app.py `bifur`).
Usage: python -I huginn_addition/analyze_bifur.py results/<bifur coarse run> results/<bifur_zoom run>

Family: E_a = (1 - a) E(49B, stationary) + a E(49A, cycling), whole aligned prelude-output tensor; a = 0 / 1 are the real
prompts. Source labels refer to the '=' position. A row counts as an '='-sourced cycle if its in-run label is period-2
and the earliest late-moving position (drift > 1e-4) is '='. Rows whose earliest moving position is token 1 (the first
demo token, whose interpolated input is a blend of two different tokens) are a separate, off-manifold phenomenon.
Checks: tracked eigenvalue lambda(a) through -1 (gated: fixed-point prefix residual < 1e-5, FD eigen-residual < 1e-2,
eps-consistency, adjacent-mode overlap > 0.9); cycle amplitude A^2 vs mu = -1 - lambda (supercritical flip: linear through
0); two-step multiplier vs 1 - 4 mu; antisymmetric push response rate vs |lambda|; endpoint-start runs (basin dependence).
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_gradients import CAT, GRID, INK2  # noqa: E402

MOVE = 1e-4


def rows(z, tag, spec_tag):
    q = int(z["q"])
    a = z[f"{tag}__alpha"]
    d32 = z[f"{tag}__drift_last32_all"].copy()
    if f"{tag}__ext_idx" in z.files:  # final-window drift for extended rows
        d32[z[f"{tag}__ext_idx"]] = z[f"{tag}__ext_drift_last32_all"]
    first = np.array([(np.where(d32[i] > MOVE)[0][0] if (d32[i] > MOVE).any() else -1) for i in range(len(a))])
    lab = z[f"{tag}__label_final"]
    return dict(alpha=a, lam=z[f"{spec_tag}__lam_track"], elig=z[f"{spec_tag}__eligible"].astype(bool), label=lab, first=first,
                amp=z[f"{tag}__amp_rel"][:, q], tail=z[f"{tag}__tail_final"], fp_ok=z[f"{tag}__fp_ok_prefix"].astype(bool),
                argA=z[f"{tag}__lp_A"].argmax(-1), argB=z[f"{tag}__lp_B"].argmax(-1), q=q)  # digit-restricted argmax


def main(run_c, run_z):
    out = f"{run_z}/analysis"
    os.makedirs(out, exist_ok=True)
    zc, zz = np.load(f"{run_c}/bifur.npz", allow_pickle=True), np.load(f"{run_z}/bifur.npz", allow_pickle=True)
    C, Z = rows(zc, "coarse", "coarse_spec"), rows(zz, "fine", "fine_spec")
    q = C["q"]
    # crossing from the zoom grid (eligible adjacent pairs)
    a, lam, el = Z["alpha"], Z["lam"], Z["elig"]
    br = [a[i] + (lam[i] + 1) / (lam[i] - lam[i + 1]) * (a[i + 1] - a[i]) for i in range(len(a) - 1)
          if el[i] and el[i + 1] and lam[i] > -1 >= lam[i + 1]]
    a_star = br[0] if br else np.nan
    # '='-sourced cycles used for scaling (both runs; exclude the near-critical a=0.71 row: mu below FD resolution)
    pts = []
    for R, src in ((C, "coarse"), (Z, "zoom")):
        for i in range(len(R["alpha"])):
            mu = -1 - R["lam"][i]
            if R["label"][i] == 1 and R["first"][i] == q and R["elig"][i] and mu > 0.002:
                pts.append((src, float(R["alpha"][i]), float(mu), float(R["amp"][i])))
    mu_ = np.array([p[2] for p in pts]); A2 = np.array([p[3] ** 2 for p in pts])
    slope_origin = float((mu_ @ A2) / (mu_ @ mu_))
    coef = np.polyfit(mu_, A2, 1)
    # two-step multipliers
    ts = {"alpha": zz["twostep__alpha"].tolist(), "mult": zz["twostep__mult"].round(4).tolist(),
          "pred_1_minus_4mu": (1 - 4 * (-1 - lam[zz["twostep__rows"]])).round(4).tolist(),
          "fd_resid_eps0.1": zz["twostep__fd_resid"][:, 1].round(4).tolist(), "axis_overlap": zz["twostep__axis_ov"].round(4).tolist()}
    # pushes: antisymmetric response decay/growth rate over the first 64 steps
    pr, prow, psg, pmg = zz["push__proj"], zz["push__row"], zz["push__sign"], zz["push__mag"]
    pushes = []
    for r in np.unique(prow):
        for mg in (1e-3, 1e-4):
            pp = pr[(prow == r) & (pmg == mg) & (psg == 1)][0]
            pm = pr[(prow == r) & (pmg == mg) & (psg == -1)][0]
            anti = (pp - pm) / 2
            k = np.arange(1, 65)
            rate = float(np.exp(np.polyfit(k, np.log(np.abs(anti[1:65])), 1)[0]))
            pushes.append({"alpha": float(a[r]), "lam": float(lam[r]), "mag": mg, "abs_rate": round(rate, 4),
                           "alternates": bool(np.all(np.sign(anti[1:33]) != np.sign(anti[:32]))),
                           "control_max": float(np.abs(pr[(prow == r) & (pmg == 0)][0]).max())})
    summ = {
        "family": "E_a = (1-a) E(uid %s) + a E(uid %s), whole aligned prelude-output tensor; source/q = '=' (position %d)" % (
            json.load(open(f"{run_z}/meta.json"))["cfg"]["uid0"], json.load(open(f"{run_z}/meta.json"))["cfg"]["uid1"], q),
        "endpoint_lambda": {"a=0 (stationary)": float(C["lam"][0]), "a=1 (cycling)": float(C["lam"][-1])},
        "coarse": [{"alpha": round(float(C["alpha"][i]), 4), "lam": round(float(C["lam"][i]), 4), "eligible": bool(C["elig"][i]),
                    "label": int(C["label"][i]), "first_moving": int(C["first"][i]), "amp_rel_q": round(float(C["amp"][i]), 5)}
                   for i in range(len(C["alpha"]))],
        "zoom": [{"alpha": round(float(a[i]), 4), "lam": round(float(lam[i]), 4), "eligible": bool(el[i]), "label": int(Z["label"][i]),
                  "first_moving": int(Z["first"][i]), "amp_rel_q": round(float(Z["amp"][i]), 5),
                  "readout_same_both_phases": bool((Z["argA"][i] == Z["argB"][i]).all())} for i in range(len(a))],
        "alpha_star_zoom": float(a_star),
        "scaling_points": pts, "A2_over_mu_slope_through_origin": slope_origin, "A2_vs_mu_linear_fit": coef.tolist(),
        "A2_over_mu_each": (A2 / mu_).round(4).tolist(),
        "two_step": ts, "pushes": pushes,
        "endpoint_start": {"alpha": a.tolist(), "label": zz["hyst__label"].tolist(), "amp_rel_q_after_256": zz["hyst__amp_rel_q"].round(5).tolist(),
                           "common_init_amp_rel_q": Z["amp"].round(5).tolist()},
        "token1_offmanifold_cycles_coarse_alpha": [round(float(C["alpha"][i]), 3) for i in range(len(C["alpha"])) if C["label"][i] == 1 and C["first"][i] == 1],
    }
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: summ[k] for k in ("endpoint_lambda", "alpha_star_zoom", "A2_over_mu_slope_through_origin", "A2_vs_mu_linear_fit",
                                           "A2_over_mu_each", "two_step", "token1_offmanifold_cycles_coarse_alpha")}, indent=1, default=float))
    for p in pushes:
        print(p)

    # ---------------- figure
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.3))
    # A: bifurcation diagram (projection of the last 64 states at '=' on the endpoint cycle axis, centred per alpha)
    ax = axs[0]
    for R, mk in ((C, "o"), (Z, "o")):
        for i in range(len(R["alpha"])):
            y = R["tail"][i][:, 0]
            if len(y) == 0:
                continue
            y = y - y.mean()
            col = CAT[1] if (R["label"][i] == 1 and R["first"][i] == q) else (GRID if R["first"][i] == 1 else INK2)
            ax.scatter(np.full(len(y), R["alpha"][i]), y, s=6, color=col, zorder=3)
    if np.isfinite(a_star):
        ax.axvline(a_star, color=CAT[1], lw=0.8, ls=":")
    ax.scatter([], [], color=CAT[1], s=12, label="cycle at “=” (two branches)")
    ax.scatter([], [], color=INK2, s=12, label="settled / unresolved")
    ax.scatter([], [], color=GRID, s=12, label="other cycle: blended 1st demo token")
    ax.set_xlabel("blend α  (0 = stationary prompt 49B, 1 = cycling prompt 49A)")
    ax.set_ylabel("“=” state on cycle axis, last 64 loops\n(tail-mean-centred projection)")
    ax.set_title("Bifurcation diagram"); ax.legend(fontsize=6.8, loc="upper left")
    # B: tracked eigenvalue
    ax = axs[1]
    ax.plot(C["alpha"], C["lam"], color=INK2, lw=1, marker="o", ms=4, label="coarse grid (most rows not gated)")
    ax.plot(a, lam, color=CAT[1], lw=1.2, marker="o", ms=4, label="zoom grid (all pass checks)")
    ax.axhline(-1, color=CAT[1], lw=0.8, ls=":")
    ax.set_xlabel("blend α"); ax.set_ylabel("flip eigenvalue of the “=” token")
    ax.set_title(f"Eigenvalue crosses −1 near α ≈ {a_star:.2f}"); ax.legend(fontsize=7)
    # C: amplitude^2 vs mu
    ax = axs[2]
    mm = np.linspace(0, mu_.max() * 1.05, 50)
    ax.plot(mm, slope_origin * mm, color=INK2, ls=":", lw=1, label=f"A² = {slope_origin:.4f} μ")
    for src, col in (("coarse", INK2), ("zoom", CAT[1])):
        sel = np.array([p[0] == src for p in pts])
        if not sel.any():
            continue
        ax.scatter(mu_[sel], A2[sel], color=col, s=30, zorder=3, label=f"{src} grid")
    ax.set_xlabel("μ = −1 − eigenvalue  (distance past −1)"); ax.set_ylabel("cycle amplitude²  (relative, at “=”)")
    ax.set_title("Amplitude² ≈ linear in μ"); ax.legend(fontsize=7)
    # D: two-step multiplier vs mu
    ax = axs[3]
    mu_t = -1 - lam[zz["twostep__rows"]]
    mm = np.linspace(0, max(mu_t.max(), 0.01) * 1.1, 50)
    ax.plot(mm, 1 - 4 * mm, color=INK2, ls=":", lw=1, label="flip normal form: 1 − 4μ")
    ax.scatter(mu_t, zz["twostep__mult"], color=CAT[1], s=34, zorder=3, label="measured (two-step map, “=” block)")
    ax.axhline(1, color=GRID, lw=0.8)
    ax.set_xlabel("μ"); ax.set_ylabel("cycle multiplier"); ax.set_title("Cycle stability → 1 at onset"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/bifurcation.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"))
