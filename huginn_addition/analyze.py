"""Analyze a Huginn addition-dynamics run.

Usage:  python -I huginn_addition/analyze.py results/<run_id>

Writes figures (PNG) and summary.json into results/<run_id>/analysis/.
All accuracy numbers are reported per *problem* (pid) with bootstrap CIs resampling problems,
since seeds / digits / demo sets / steps are repeated measurements of the same problem.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reference palette (dataviz skill, light mode)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal blue ramp (>= step 250)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def load(run):
    def cat(name):
        fs = sorted(glob.glob(f"{run}/{name}_c*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

    probs = pd.DataFrame(json.load(open(f"{run}/problems.json")))
    digits = []
    for _, p in probs.iterrows():
        for j, di in enumerate(p["digit_info"]):
            digits.append({"uid": p["uid"], "pos": j, **di, "n_answer": len(p["digit_info"])})
    digits = pd.DataFrame(digits)
    meta = json.load(open(f"{run}/meta.json"))
    return dict(
        probs=probs.drop(columns=["digit_info", "prompt"]), digits=digits, meta=meta,
        steps=cat("steps"), pairs=cat("pairs"), lags=cat("lags"), phase=cat("phase"),
        to_final=cat("to_final"), pca=cat("pca"),
        greedy=pd.read_parquet(f"{run}/greedy.parquet") if os.path.exists(f"{run}/greedy.parquet") else None,
    )


def boot_ci(df, value, group_cols, by="pid", n=1000, seed=0):
    """Mean of `value` per group, with a bootstrap CI that resamples problems."""
    rng = np.random.default_rng(seed)
    per = df.groupby(group_cols + [by])[value].mean().reset_index()
    out = []
    for g, sub in per.groupby(group_cols):
        v = sub[value].to_numpy()
        bs = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
        out.append((*(g if isinstance(g, tuple) else (g,)), v.mean(), *np.percentile(bs, [2.5, 97.5]), len(v)))
    return pd.DataFrame(out, columns=group_cols + ["mean", "lo", "hi", "n"])


def stable_step(df, flag, keys):
    """First step after which `flag` is True for every remaining step (NaN if never stable)."""
    df = df.sort_values(keys + ["step"])
    K = df["step"].max()

    def f(g):
        x = g[flag].to_numpy()
        if not x[-1]:
            return np.nan
        bad = np.nonzero(~x)[0]
        return g["step"].to_numpy()[bad[-1] + 1] if len(bad) else g["step"].min()

    return df.groupby(keys).apply(f, include_groups=False).rename(f"stable_{flag}").reset_index(), K


def savefig(fig, out, name):
    fig.tight_layout()
    fig.savefig(f"{out}/{name}.png", dpi=150)
    plt.close(fig)


def main(run):
    out = f"{run}/analysis"
    os.makedirs(out, exist_ok=True)
    D = load(run)
    probs, digits, st = D["probs"], D["digits"], D["steps"]
    st = st.merge(probs[["uid", "pid", "family", "demo", "n_digits", "max_dep", "n_carries", "overflow"]], on="uid")
    st = st.merge(digits[["uid", "pos", "col", "carry_in", "chain_in", "dep_len", "ctype", "n_answer"]], on=["uid", "pos"])
    K = int(st["step"].max())
    summary = {"run": run, "K": K, "n_prompts": int(probs["uid"].nunique()), "n_problems": int(probs["pid"].nunique()),
               "validation": D["meta"]["validation"], "runtime_s": D["meta"]["runtime_s"]}

    # ---------------------------------------------------------------- accuracy vs loops
    allc = st.groupby(["uid", "pid", "seed", "step", "family", "n_digits", "max_dep", "n_carries"])["correct"].all().reset_index()
    allc["correct"] = allc["correct"].astype(float)
    acc = boot_ci(allc, "correct", ["step"])
    g = D["greedy"]
    if g is not None:
        g = g.merge(probs[["uid", "pid", "family", "n_digits", "max_dep"]], on="uid")
        g["exact_and_terminated"] = g["exact_and_terminated"].astype(float)
        gacc = boot_ci(g, "exact_and_terminated", ["depth"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(acc["step"], acc["mean"], color=CAT[0], label="teacher-forced, all digits correct")
    ax.fill_between(acc["step"], acc["lo"], acc["hi"], color=CAT[0], alpha=0.15, lw=0)
    if g is not None:
        ax.plot(gacc["depth"], gacc["mean"], color=CAT[1], marker="o", ms=5, label="greedy generation, exact + stops")
        ax.vlines(gacc["depth"], gacc["lo"], gacc["hi"], color=CAT[1], lw=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("recurrent iterations (log scale)")
    ax.set_ylabel("accuracy (per problem)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Addition accuracy vs. recurrent iterations")
    ax.legend(loc="lower right")
    savefig(fig, out, "01_accuracy_vs_loops")
    summary["accuracy_tf"] = acc[acc["step"].isin([1, 2, 4, 8, 16, 32, 64, 128, K])].round(3).to_dict("records")
    if g is not None:
        summary["accuracy_greedy"] = gacc.round(3).to_dict("records")

    # by dependency length (dep family) and carries (carries family), faceted by digits
    for fam, col, lab in (("dep", "max_dep", "max carry dependency"), ("carries", "n_carries", "number of carries")):
        sub = allc[allc["family"] == fam]
        nd = sorted(sub["n_digits"].unique())
        fig, axes = plt.subplots(1, len(nd), figsize=(4.0 * len(nd), 3.6), sharey=True)
        for ax, n in zip(np.atleast_1d(axes), nd):
            s2 = sub[sub["n_digits"] == n]
            levels = sorted(s2[col].unique())
            for li, lv in enumerate(levels):
                a = boot_ci(s2[s2[col] == lv], "correct", ["step"])
                c = SEQ[min(li + (len(SEQ) - len(levels)) // 2, len(SEQ) - 1)]
                ax.plot(a["step"], a["mean"], color=c, label=f"{lv}")
            ax.set_xscale("log", base=2)
            ax.set_title(f"{n}-digit operands")
            ax.set_xlabel("iterations (log)")
            ax.legend(title=lab, loc="lower right", fontsize=8, title_fontsize=8)
        np.atleast_1d(axes)[0].set_ylabel("all-digits correct (TF)")
        savefig(fig, out, f"02_accuracy_by_{col}")

    # ---------------------------------------------------------------- iterations needed per digit
    ss, _ = stable_step(st, "correct", ["uid", "pid", "seed", "pos"])
    ss = ss.merge(digits, on=["uid", "pos"]).merge(probs[["uid", "n_digits", "overflow", "family", "demo"]], on="uid")
    ss["first_token"] = (ss["pos"] == 0).astype(int)
    summary["never_stably_correct_frac"] = float(ss["stable_correct"].isna().mean())
    ok = ss.dropna(subset=["stable_correct"]).copy()
    ok["log2_steps"] = np.log2(ok["stable_correct"])
    # OLS with problem-cluster bootstrap: log2(steps-to-stable) ~ dep_len + chain_in + col + pos + first + overflow + n_digits
    # NB: pos is omitted - col + pos == n_digits - 1 + overflow exactly, so including all four is unidentifiable
    feats = ["dep_len", "chain_in", "carry_in", "col", "first_token", "overflow", "n_digits"]
    def ols(d):
        X = np.column_stack([np.ones(len(d))] + [d[f].to_numpy(float) for f in feats])
        beta, *_ = np.linalg.lstsq(X, d["log2_steps"].to_numpy(), rcond=None)
        return beta
    beta = ols(ok)
    rng = np.random.default_rng(0)
    pids = ok["pid"].unique()
    groups = {p: d for p, d in ok.groupby("pid")}
    bs = np.array([ols(pd.concat([groups[p] for p in rng.choice(pids, len(pids))])) for _ in range(300)])
    coef = pd.DataFrame({"feature": ["intercept"] + feats, "beta": beta,
                         "lo": np.percentile(bs, 2.5, 0), "hi": np.percentile(bs, 97.5, 0)})
    summary["steps_to_stable_regression_log2"] = coef.round(3).to_dict("records")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, (key, lab) in zip(axes, (("dep_len", "carry dependency length into column"), ("chain_in", "consecutive carry run into column"))):
        a = boot_ci(ok, "log2_steps", [key])
        ax.errorbar(a[key], 2 ** a["mean"], yerr=[2 ** a["mean"] - 2 ** a["lo"], 2 ** a["hi"] - 2 ** a["mean"]],
                    color=CAT[0], marker="o", ms=6, lw=2, capsize=0)
        for _, r in a.iterrows():
            ax.annotate(f"n={int(r['n'])}", (r[key], 2 ** r["hi"]), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color=INK2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel(lab)
        ax.set_xticks(sorted(a[key].unique()))
    axes[0].set_ylabel("iterations until digit is stably correct\n(geometric mean, per problem)")
    axes[0].set_title("Causal dependency (propagate chain)\nunadjusted; only digits correct at final step")
    axes[1].set_title("Raw carry run (counts independent carries)")
    savefig(fig, out, "03_steps_to_stable_by_dependency")

    # ---------------------------------------------------------------- what converges
    st["rel_delta"] = st["delta"] / st["norm"]
    pairs = D["pairs"].merge(probs[["uid", "pid"]], on="uid")
    pairs["rel_dist"] = pairs["dist"] / 76.36  # raw norm is pinned by the core's final RMSNorm (checked below)
    summary["raw_norm_range"] = [float(st["norm"].min()), float(st["norm"].max())]
    panels = [
        ("rel_delta", st, "state change per step\n‖s_k − s_{k−1}‖ / ‖s_k‖"),
        ("delta_ln", st, "change of normalized state\n‖ln_f(s_k) − ln_f(s_{k−1})‖"),
        ("kl_prev", st, "readout change per step\nKL(p_k ‖ p_{k−1}) [nats]"),
        ("rel_dist", pairs, "cross-seed state distance\n(relative)"),
        ("tv", pairs, "cross-seed readout distance\n(total variation)"),
        ("entropy", st, "readout entropy [nats]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    conv = {}
    for ax, (m, df, lab) in zip(axes.ravel(), panels):
        q = df.groupby("step")[m].quantile([0.1, 0.5, 0.9]).unstack()
        ax.plot(q.index, q[0.5], color=CAT[0])
        ax.fill_between(q.index, q[0.1].clip(lower=1e-9), q[0.9].clip(lower=1e-9), color=CAT[0], alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(lab, fontsize=9.5)
        ax.set_xlabel("iterations (log)")
        conv[m] = {int(k): float(q.loc[k, 0.5]) for k in [1, 4, 16, 64, K] if k in q.index}
    fig.suptitle("What converges? median and 10–90% band over (prompt, seed, answer digit)", fontsize=11, fontweight="bold")
    savefig(fig, out, "04_what_converges")
    summary["convergence_medians"] = conv

    # readout stabilizes vs state stabilizes
    st["argmax_same"] = st.groupby(["uid", "seed", "pos"])["argmax"].transform(lambda x: x == x.iloc[-1])
    sa, _ = stable_step(st, "argmax_same", ["uid", "seed", "pos"])
    for thr in (1e-2, 1e-3):
        st[f"still_{thr}"] = st["rel_delta"] < thr
        sb, _ = stable_step(st, f"still_{thr}", ["uid", "seed", "pos"])
        summary[f"state_rel_delta_below_{thr}"] = {
            "frac_trajectories_reaching": float(sb.iloc[:, -1].notna().mean()),
            "median_step": float(sb.iloc[:, -1].median()) if sb.iloc[:, -1].notna().any() else None,
        }
    summary["argmax_final_stable_median_step"] = float(sa["stable_argmax_same"].median())

    # ---------------------------------------------------------------- fixed point / orbit / drift
    lags = D["lags"]
    lp = lags.pivot_table(index=["uid", "seed", "pos", "window"], columns="lag", values="D_rel").reset_index()
    lagcols = [c for c in lp.columns if isinstance(c, (int, np.integer))]
    L = np.array(lagcols)
    Dm = lp[lagcols].to_numpy()
    d1 = Dm[:, 0]
    dmin_ge2 = Dm[:, 1:].min(1)
    lag_at_min = L[1:][Dm[:, 1:].argmin(1)]
    slope = np.polyfit(np.log(L), np.log(Dm.T + 1e-12), 1)[0]  # growth exponent of D(p) in p
    lp["D1"], lp["dip_ratio"], lp["dip_lag"], lp["growth_exp"] = d1, dmin_ge2 / d1, lag_at_min, slope
    lp["Dmax"] = Dm.max(1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for wi, (w, c) in enumerate((("mid", CAT[1]), ("late", CAT[0]))):
        sub = lags[lags["window"] == w]
        q = sub.groupby("lag")["D_rel"].quantile([0.1, 0.5, 0.9]).unstack()
        axes[0].plot(q.index, q[0.5], color=c, label=f"{w} window")
        axes[0].fill_between(q.index, q[0.1], q[0.9], color=c, alpha=0.12, lw=0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("lag p (iterations)")
    axes[0].set_ylabel("D(p) = mean ‖s_k − s_{k−p}‖ / ‖s_k‖")
    axes[0].set_title("Lag profile: ~0 = stationary, flat > 0 = jitter, dip = orbit, rising = drift")
    axes[0].legend()
    late = lp[lp["window"] == "late"]
    axes[1].scatter(late["D1"], late["growth_exp"], s=9, color=CAT[0], alpha=0.5, edgecolor="none")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("late-window D(1) (step size)")
    axes[1].set_ylabel("growth exponent of D(p) in p\n(0 = flat, 1 = linear drift)")
    axes[1].set_title("Per trajectory, late window")
    savefig(fig, out, "05_lag_profiles")
    mid = lp[lp["window"] == "mid"].set_index(["uid", "seed", "pos"])
    latei = late.set_index(["uid", "seed", "pos"])
    shrink = (latei["D1"] / mid.loc[latei.index, "D1"])
    summary["lag_profile"] = {
        "late_D1_quantiles": {str(k): float(v) for k, v in late["D1"].quantile([0.1, 0.5, 0.9]).items()},
        "late_Dmax_quantiles": {str(k): float(v) for k, v in late["Dmax"].quantile([0.1, 0.5, 0.9]).items()},
        "late_growth_exp_quantiles": {str(k): float(v) for k, v in late["growth_exp"].quantile([0.1, 0.5, 0.9]).items()},
        "frac_dip_below_half_D1": float((late["dip_ratio"] < 0.5).mean()),
        "late_over_mid_D1_quantiles": {str(k): float(v) for k, v in shrink.quantile([0.1, 0.5, 0.9]).items()},
    }
    lp.to_parquet(f"{out}/lag_features.parquet", index=False)

    # ---------------------------------------------------------------- phase alignment across seeds
    ph = D["phase"]
    if len(ph):
        fig, ax = plt.subplots(figsize=(5, 4.4))
        ax.scatter(ph["resid_tau0"], ph["resid_best"], s=8, color=CAT[0], alpha=0.4, edgecolor="none")
        lim = [min(ph["resid_best"].min(), ph["resid_tau0"].min()) * 0.8, ph["resid_tau0"].max() * 1.2]
        ax.plot(lim, lim, color=INK2, lw=1, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cross-seed distance, no time shift")
        ax.set_ylabel("cross-seed distance, best time shift")
        ax.set_title("Same orbit at a different phase?\n(points far below the diagonal would say yes)")
        savefig(fig, out, "06_phase_alignment")
        summary["phase"] = {
            "resid_tau0_quantiles": {str(k): float(v) for k, v in ph["resid_tau0"].quantile([0.1, 0.5, 0.9]).items()},
            "frac_tau_nonzero": float((ph["tau_best"] != 0).mean()),
            "frac_best_below_half_tau0": float((ph["resid_best"] < 0.5 * ph["resid_tau0"]).mean()),
        }

    # ---------------------------------------------------------------- example PCA trajectories
    pca = D["pca"]
    if len(pca):
        ex = probs[(probs["demo"] == "A") & (probs["family"] == "dep")].sort_values(["n_digits", "max_dep"])
        ex = ex.groupby("max_dep").head(1).head(4)
        fig, axes = plt.subplots(1, len(ex), figsize=(3.4 * len(ex), 3.4))
        for ax, (_, p) in zip(np.atleast_1d(axes), ex.iterrows()):
            sub = pca[(pca["uid"] == p["uid"]) & (pca["pos"] == 0) & (pca["basis"] == "all")]
            for si, (sd, tr) in enumerate(sub.groupby("seed")):
                tr = tr.sort_values("step")
                ax.plot(tr["pc1"], tr["pc2"], color=CAT[si], lw=1.2, alpha=0.9, label=f"seed {sd}")
                ax.scatter(tr["pc1"].iloc[0], tr["pc2"].iloc[0], s=16, color=CAT[si], marker="o")
                ax.scatter(tr["pc1"].iloc[-1], tr["pc2"].iloc[-1], s=40, color=CAT[si], marker="*", edgecolor=SURF, lw=0.6)
            ax.set_title(f"{p['a']} + {p['b']} (dep {p['max_dep']})\nfirst answer digit", fontsize=9)
            ax.set_xlabel(f"PC1 ({sub['evr1'].iloc[0]:.0%})"); ax.set_ylabel(f"PC2 ({sub['evr2'].iloc[0]:.0%})")
        np.atleast_1d(axes)[0].legend(fontsize=7, loc="best")
        savefig(fig, out, "07_pca_examples")

    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
