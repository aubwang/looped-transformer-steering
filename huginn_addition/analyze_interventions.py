"""Analyze circle-intervention results.  Usage: python -I huginn_addition/analyze_interventions.py results/<intervene_run>

Eligibility (fixed in advance): rows whose unperturbed model prediction at step K (full-vocabulary argmax) is the
correct digit token. All-row sensitivity is reported too.
Primary endpoint (pre-declared): paired change in target-vs-source log odds,
    Δ = [lp_target − lp_d]_intervention − [lp_target − lp_d]_baseline   (same step),
with the target defined by each kind: circle{k} -> d+k; scrT{k}_i -> d+k; radial/random/pca (matched to circle1)
-> d+1, or d−1 when matched '@circle-1'. Secondary: target hit rate (full argmax == target digit token),
unchanged rate, non-digit rate. 95% CIs bootstrap whole problems (both demo sets together).
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

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
})
RNG = np.random.default_rng(0)


def target_shift(kind):
    base, _, ref = kind.partition("@")
    if base == "none":
        return 1  # sham evaluated toward d+1 (should be ~0)
    if base.startswith("circle"):
        return int(base[6:])
    if base.startswith("scrT"):
        return int(base[4:].split("_")[0])
    return int(ref[6:]) if ref else 1


def label(kind):
    base, _, ref = kind.partition("@")
    names = {"radial": "radial (outward)", "none": "sham (no push)"}
    if base.startswith("circle"):
        k = int(base[6:]); return f"circle {k:+d} notch"
    if base.startswith("scrT"):
        k, i = base[4:].split("_"); return f"scrambled-order circle #{i} (→ d{int(k):+d})"
    for p, n in (("random", "random direction"), ("pca", "random top-PC direction")):
        if base.startswith(p):
            name = f"{n} #{base.split('_')[1]}"
            break
    else:
        name = names.get(base, base)
    return name + (f" (sized like {int(ref[6:]):+d})" if ref else "")


def boot(df, col, n=2000):
    per = df.groupby("pid")[col].mean()
    v = per.to_numpy()
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    bs = RNG.choice(v, (n, len(v))).mean(1)
    return v.mean(), *np.percentile(bs, [2.5, 97.5]), len(v)


def main(run):
    out = f"{run}/analysis"
    os.makedirs(out, exist_ok=True)
    df = pd.read_parquet(f"{run}/interventions.parquet")
    meta = json.load(open(f"{run}/meta.json"))
    K, TF = meta["cfg"]["K"], meta["cfg"]["T_FINAL_EXTRA"]
    lp = df[[f"lp{q}" for q in range(10)]].to_numpy()

    base = df[df["timing"] == "baseline"].set_index(["uid", "step"])
    # eligibility: baseline full-vocab argmax is the correct digit token at step K
    bK = base.xs(K, level="step")
    eligible = set(bK.index[(bK["full_argmax_is_digit"]) & (bK["digit_argmax"] == bK["digit"])])
    summary = {"run": run, "n_prompts": int(df["uid"].nunique()), "n_problems": int(df["pid"].nunique()),
               "eligible_prompts": len(eligible), "eligible_problems": int(df[df["uid"].isin(eligible)]["pid"].nunique()),
               "validation": meta["validation"]}

    d = df["digit"].to_numpy()
    sh = df["kind"].map(target_shift).to_numpy()
    tgt = (d + sh) % 10
    df["target"] = tgt
    df["lo_int"] = lp[np.arange(len(df)), tgt] - lp[np.arange(len(df)), d]
    bl = base[[f"lp{q}" for q in range(10)]]
    key = list(zip(df["uid"], df["step"]))
    blp = bl.reindex(key).to_numpy()
    df["lo_base"] = blp[np.arange(len(df)), tgt] - blp[np.arange(len(df)), d]
    df["dlogodds"] = df["lo_int"] - df["lo_base"]
    df["hit"] = (df["full_argmax_is_digit"]) & (df["digit_argmax"] == df["target"])
    df["unchanged"] = (df["full_argmax_is_digit"]) & (df["digit_argmax"] == df["digit"])
    df["nondigit"] = ~df["full_argmax_is_digit"]
    df["eligible"] = df["uid"].isin(eligible)
    E = df[df["eligible"] & (df["timing"] != "baseline")]

    # ---------------- readout-only table
    ro = E[E["timing"] == "readout"]
    rows = []
    for (kind, alpha, normp), g in ro.groupby(["kind", "alpha", "norm_preserve"]):
        m, lo, hi, n = boot(g, "dlogodds")
        rows.append({"kind": kind, "label": label(kind), "alpha": alpha, "norm_preserve": normp, "dlogodds": m,
                     "lo": lo, "hi": hi, "problems": n, "hit_rate": g["hit"].mean(), "unchanged": g["unchanged"].mean(),
                     "nondigit": g["nondigit"].mean()})
    RO = pd.DataFrame(rows)
    RO.to_csv(f"{out}/readout_only.csv", index=False)
    summary["readout_only_alpha1"] = RO[(RO["alpha"] == 1.0) & (~RO["norm_preserve"])].round(3).drop(columns=["label"]).to_dict("records")

    # all-row sensitivity for the primary comparison
    allro = df[(df["timing"] == "readout") & (df["alpha"] == 1.0) & (~df["norm_preserve"])]
    summary["readout_only_alpha1_all_rows"] = {k: round(boot(g, "dlogodds")[0], 3) for k, g in allro.groupby("kind")}

    # ---------------- dynamic curves
    dyn = E[E["timing"].isin(["mid", "final"])]
    curves = []
    for (timing, kind, alpha, normp, step), g in dyn.groupby(["timing", "kind", "alpha", "norm_preserve", "step"]):
        m, lo, hi, n = boot(g, "dlogodds")
        curves.append({"timing": timing, "kind": kind, "alpha": alpha, "norm_preserve": normp, "step": step,
                       "dlogodds": m, "lo": lo, "hi": hi, "hit_rate": g["hit"].mean(), "unchanged": g["unchanged"].mean(),
                       "retention_median": g["retention"].median(), "rel_dev_median": g["rel_dev"].median(),
                       "abs_dev_max": g["abs_dev"].max()})
    C = pd.DataFrame(curves)
    C.to_csv(f"{out}/dynamic.csv", index=False)
    summary["sham_max_abs_dev"] = float(dyn[dyn["kind"] == "none"]["abs_dev"].max())
    fin_end = C[(C["timing"] == "final") & (C["step"] == K + TF)]
    summary["final_push_after_32_steps"] = fin_end.round(3).to_dict("records")
    mid_end = C[(C["timing"] == "mid") & (C["step"] == K)]
    summary["mid_push_at_K"] = mid_end.round(3).to_dict("records")
    bsc = df[(df["timing"] == "baseline") & df["eligible"]]
    summary["baseline_step_change_median"] = {int(k): float(v) for k, v in bsc.groupby("step")["step_change"].median().items()}

    # ---------------- figures
    # 1. readout-only, alpha 1, primary
    sel = RO[(RO["alpha"] == 1.0) & (~RO["norm_preserve"])].copy()
    order = ["circle1", "circle-1", "circle2", "radial", "radial@circle-1", "scrT1_0", "scrT1_1", "scrT1_2", "scrT-1_0", "scrT-1_1",
             "pca_0", "pca_1", "pca_2", "pca_0@circle-1", "random_0", "random_1", "random_2", "random_0@circle-1"]
    sel = sel.set_index("kind").reindex([k for k in order if k in set(sel["kind"])]).reset_index()
    colors = [CAT[0] if k.startswith("circle") else CAT[1] if k.startswith("scr") else CAT[2] if k.startswith("radial")
              else CAT[6] if k.startswith("pca") else INK2 for k in sel["kind"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    y = np.arange(len(sel))[::-1]
    axes[0].barh(y, sel["dlogodds"], color=colors, height=0.6)
    axes[0].errorbar(sel["dlogodds"], y, xerr=[sel["dlogodds"] - sel["lo"], sel["hi"] - sel["dlogodds"]], fmt="none", ecolor=INK, lw=1)
    axes[0].axvline(0, color=INK2, lw=1)
    axes[0].set_yticks(y, [label(k) for k in sel["kind"]], fontsize=8.5)
    axes[0].set_xlabel("change in log-odds of target digit vs original digit\n(push vs no push, same state)")
    axes[0].set_title("Push the settled state, read the answer immediately")
    axes[1].barh(y, sel["hit_rate"], color=colors, height=0.6)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("share of answers that switch to the target digit")
    axes[1].set_title("Does the predicted digit actually change?")
    for yi, v in zip(y, sel["hit_rate"]):
        axes[1].annotate(f"{v:.0%}", (v, yi), textcoords="offset points", xytext=(4, -3), fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(f"{out}/1_readout_only.png", dpi=150); plt.close(fig)

    # 2. dose-response (primary endpoint, with CIs)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for kind, c in (("circle1", CAT[0]), ("circle-1", "#1c5cab"), ("circle2", "#0d366b"), ("radial", CAT[2]), ("scrT1_1", CAT[1]),
                    ("pca_0", CAT[6]), ("random_0", INK2)):
        g = RO[(RO["kind"] == kind) & (~RO["norm_preserve"])].sort_values("alpha")
        if len(g):
            ax.errorbar(g["alpha"], g["dlogodds"], yerr=[g["dlogodds"] - g["lo"], g["hi"] - g["dlogodds"]], color=c, marker="o",
                        ms=5, capsize=0, label=label(kind) + (" (larger chord, no matched controls)" if kind == "circle2" else ""))
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("push size (× chord length)"); ax.set_ylabel("change in target-vs-original log-odds")
    ax.set_xticks([0.5, 1, 2]); ax.set_title("Dose-response (readout-only, 95% CI by problem)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/2_dose_response.png", dpi=150); plt.close(fig)

    # 3. dynamics after a push
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    for row, timing, t0_ in ((0, "final", K), (1, "mid", meta["cfg"]["T_MID"])):
        for kind, c in (("circle1", CAT[0]), ("circle-1", "#1c5cab"), ("radial", CAT[2]), ("scrT1_0", CAT[1]), ("pca_0", CAT[6]), ("random_0", INK2)):
            for normp, ls in ((False, "-"), (True, ":")):
                g = C[(C["timing"] == timing) & (C["kind"] == kind) & (C["alpha"] == 1.0) & (C["norm_preserve"] == normp)].sort_values("step")
                if not len(g):
                    continue
                lab = label(kind) + (" (norm kept)" if normp else "")
                axes[row, 0].plot(g["step"] - t0_, g["retention_median"], color=c, ls=ls, label=lab)
                if not normp:
                    axes[row, 0].plot(g["step"] - t0_, g["rel_dev_median"], color=c, ls="--", lw=1, alpha=0.7)
                axes[row, 1].plot(g["step"] - t0_, g["hit_rate"], color=c, ls=ls, marker="o", ms=3, label=lab)
        axes[row, 0].axhline(0, color=INK2, lw=1)
        axes[row, 0].set_ylabel("solid: component along the original push\ndashed: total displacement (both / push size, median)")
        axes[row, 1].set_ylabel("share answering the target digit"); axes[row, 1].set_ylim(0, 1)
        for a in axes[row]:
            a.set_xlabel(f"iterations after the push (pushed at step {t0_})")
        axes[row, 0].set_title(f"{'Converged state' if timing == 'final' else 'Mid-trajectory (baseline still moving)'}: does the push decay?")
        axes[row, 0].set_xlim(-0.5, 16)
        axes[row, 1].set_title("…and does the answer stay changed?")
    axes[0, 1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{out}/3_dynamics.png", dpi=150); plt.close(fig)

    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
