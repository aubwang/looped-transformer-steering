"""First-pass geometry of Huginn's late answer-position states (offline, existing pilot data).

Usage:  python -I huginn_addition/shape_analysis.py results/<pilot_run>

Plan revised after review by GPT-6 Astra. Key design choices:
  * Primary cohort: answer positions after the first (pos >= 1). The first answer token is a space-prefixed
    digit, is never 0 and is dominated by 1 and 9, so it cannot test a full ten-digit structure.
  * Unit of independence = problem (pid). Both demo sets, every position and every seed of a pid stay in the same
    CV fold / bootstrap cluster. Static analyses average the seeds; dynamics keep individual seeds.
  * Geometry is tested by predicting held-out *states* from encodings of the digit (encoding models), not by
    training a probe to output a circle. Every model includes the same nuisance covariates (previous token incl.
    space-prefix flag, answer position, column, answer length, operand length, overflow, demo set, family).
    Ridge penalties are tuned inside training folds.
  * Cyclic order is tested against nulls: random digit->angle permutations, random 2-D ten-point codes,
    and (for the line model) random digit orderings.
  * The answer-subspace settling test is cross-fitted and compared with equal-rank random and PCA subspaces.

Limits: states are a 64-d Gaussian sketch (random projection) of the 5280-d bf16 state, stored in fp16.
A positive result is meaningful; a negative one does not rule structure out in the full state. All results
are conditional on teacher forcing (the correct answer prefix is visible). Saved states are pre-coda.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
})

LATE = (192, 256)  # inclusive step window for the late-state average
ALPHAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
N_FOLDS, N_INNER, N_PERM = 5, 4, 200
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- data
def load(run):
    probs = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    rows, trajs, K = [], [], None
    for f in sorted(glob.glob(f"{run}/traj_c*.npz")):
        z = np.load(f)
        proj, dlp, valid, uids, seeds = z["proj"], z["digit_logp"], z["valid"], z["uids"], z["seeds"]
        K = proj.shape[0] - 1
        for r in range(proj.shape[1]):
            p = probs[int(uids[r])]
            for j in range(proj.shape[2]):
                if not valid[r, j]:
                    continue
                di = p["digit_info"][j]
                tr = proj[:, r, j].astype(np.float32)  # (K+1, 64)
                prev = "eq" if j == 0 else (f"sp{p['digit_info'][0]['digit']}" if j == 1 else f"d{p['digit_info'][j - 1]['digit']}")
                rows.append(dict(
                    uid=p["uid"], pid=p["pid"], seed=int(seeds[r]), pos=j, digit=di["digit"], col=di["col"],
                    carry_in=di["carry_in"], dep_len=di["dep_len"], n_answer=len(p["digit_info"]), n_digits=p["n_digits"],
                    overflow=p["overflow"], demo=p["demo"], family=p["family"], prev=prev,
                    pred_final=int(np.argmax(dlp[K - 1, r, j])),
                ))
                trajs.append(tr)
    meta = pd.DataFrame(rows)
    T = np.stack(trajs)  # (rows, K+1, 64)
    return meta, T, K


def nuisance_design(df):
    cats = ["prev", "pos", "col", "n_answer", "n_digits", "overflow", "demo", "family"]
    return pd.get_dummies(df[cats].astype(str), drop_first=True).to_numpy(float)


def digit_features(d, kind, code=None):
    d = np.asarray(d)
    if kind == "none":
        return np.zeros((len(d), 0))
    if kind == "line":
        return ((d - 4.5) / 2.87)[:, None]
    if kind == "circle":
        th = 2 * np.pi * d / 10
        return np.column_stack([np.cos(th), np.sin(th)])
    if kind == "line+circle":
        return np.column_stack([digit_features(d, "line"), digit_features(d, "circle")])
    if kind == "categorical":
        return np.eye(10)[d][:, 1:]
    if kind == "code":  # arbitrary 10 x m code table
        return code[d]
    raise ValueError(kind)


# ----------------------------------------------------------------------------- ridge with grouped nested CV
def ridge_fit(X, Y, alpha):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
    return mx, my, beta


def ridge_pred(model, X):
    mx, my, beta = model
    return (X - mx) @ beta + my


def group_folds(groups, k, seed):
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ug)
    fold_of = {g: i % k for i, g in enumerate(perm)}
    return np.array([fold_of[g] for g in groups])


def cv_sse(X, Y, groups, folds):
    """Per-row held-out squared error with alpha tuned by inner grouped CV."""
    sse = np.zeros(len(Y))
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        inner = group_folds(groups[tr], N_INNER, seed=100 + f)
        best, best_a = np.inf, ALPHAS[0]
        for a in ALPHAS:
            e = 0.0
            for g in range(N_INNER):
                itr, ite = inner != g, inner == g
                m = ridge_fit(X[tr][itr], Y[tr][itr], a)
                e += ((ridge_pred(m, X[tr][ite]) - Y[tr][ite]) ** 2).sum()
            if e < best:
                best, best_a = e, a
        m = ridge_fit(X[tr], Y[tr], best_a)
        sse[te] = ((ridge_pred(m, X[te]) - Y[te]) ** 2).sum(1)
    return sse


def boot_share(sse0, ssek, ssecat, groups, n=2000):
    """Share of the categorical digit gain captured by model k, bootstrap over problems."""
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    est = (sse0.sum() - ssek.sum()) / (sse0.sum() - ssecat.sum())
    bs = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in RNG.choice(ug, len(ug))])
        den = sse0[s].sum() - ssecat[s].sum()
        bs.append((sse0[s].sum() - ssek[s].sum()) / den if den > 0 else np.nan)
    return est, np.nanpercentile(bs, [2.5, 97.5])


# ----------------------------------------------------------------------------- main
def main(run):
    out = f"{run}/shape"
    os.makedirs(out, exist_ok=True)
    meta, T, K = load(run)
    lo, hi = LATE
    late = T[:, lo : hi + 1].mean(1)  # (rows, 64), per seed
    meta["readout_correct"] = meta["pred_final"] == meta["digit"]
    summary = {"run": run, "late_window": LATE, "sketch_dim": int(T.shape[-1])}

    # seed-averaged static dataset, primary cohort pos >= 1
    meta["row"] = np.arange(len(meta))
    keys = ["uid", "pos"]
    g = meta.groupby(keys)
    S = g.first().reset_index()
    S["state_rows"] = g["row"].apply(list).values
    S["pred_mode"] = g["pred_final"].agg(lambda x: x.mode().iloc[0]).values
    Y_all = np.stack([late[r].mean(0) for r in S["state_rows"]])
    prim = S["pos"] >= 1
    D, Y = S[prim].reset_index(drop=True), Y_all[prim.to_numpy()]
    groups = D["pid"].to_numpy()
    folds = group_folds(groups, N_FOLDS, seed=0)
    N0 = nuisance_design(D)
    summary["cohort"] = {"rows_uid_pos": int(len(D)), "problems": int(D["pid"].nunique()),
                         "digit_counts": D["digit"].value_counts().sort_index().to_dict(),
                         "readout_accuracy": float((D["pred_mode"] == D["digit"]).mean())}
    sst = ((Y - Y.mean(0)) ** 2).sum()

    # ---------------- encoding models (true digit and model-predicted digit)
    res = {}
    for label_name, labels in (("true", D["digit"].to_numpy()), ("predicted", D["pred_mode"].to_numpy())):
        sse = {}
        for kind in ("none", "line", "circle", "line+circle", "categorical"):
            X = np.column_stack([N0, digit_features(labels, kind)])
            sse[kind] = cv_sse(X, Y, groups, folds)
        rows = []
        for kind in ("line", "circle", "line+circle", "categorical"):
            share, ci = boot_share(sse["none"], sse[kind], sse["categorical"], groups)
            rows.append({"model": kind, "heldout_R2": 1 - sse[kind].sum() / sst,
                         "gain_over_nuisance_R2": (sse["none"].sum() - sse[kind].sum()) / sst,
                         "share_of_categorical_gain": share, "share_ci": list(ci)})
        res[label_name] = {"nuisance_only_R2": 1 - sse["none"].sum() / sst, "models": rows}
        if label_name == "true":
            sse_true = sse
    summary["encoding_models"] = res

    # ---------------- cyclic-order nulls (true labels)
    d = D["digit"].to_numpy()
    def gain(Xd):
        return (sse_true["none"].sum() - cv_sse(np.column_stack([N0, Xd]), Y, groups, folds).sum()) / sst
    true_circle = (sse_true["none"].sum() - sse_true["circle"].sum()) / sst
    true_line = (sse_true["none"].sum() - sse_true["line"].sum()) / sst
    perm_circle, perm_line, rand_code = [], [], []
    for i in range(N_PERM):
        pi = RNG.permutation(10)
        perm_circle.append(gain(digit_features(pi[d], "circle")))
        perm_line.append(gain(digit_features(pi[d], "line")))
        rand_code.append(gain(digit_features(d, "code", code=RNG.standard_normal((10, 2)))))
    perm_circle, perm_line, rand_code = map(np.array, (perm_circle, perm_line, rand_code))
    summary["order_nulls"] = {
        "circle_true_gain": true_circle, "circle_perm_p": float((1 + (perm_circle >= true_circle).sum()) / (1 + N_PERM)),
        "circle_perm_quantiles": np.percentile(perm_circle, [50, 95, 100]).tolist(),
        "line_true_gain": true_line, "line_perm_p": float((1 + (perm_line >= true_line).sum()) / (1 + N_PERM)),
        "line_perm_quantiles": np.percentile(perm_line, [50, 95, 100]).tolist(),
        "random2d_p": float((1 + (rand_code >= true_circle).sum()) / (1 + N_PERM)),
        "random2d_quantiles": np.percentile(rand_code, [50, 95, 100]).tolist(),
    }

    # ---------------- unconstrained categorical decoder: which digits get confused?
    pred = np.zeros(len(D), dtype=int)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        m = ridge_fit(N0[tr], Y[tr], 10.0)  # partial out nuisance (training-only)
        Rtr, Rte = Y[tr] - ridge_pred(m, N0[tr]), Y[te] - ridge_pred(m, N0[te])
        clf = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit(Rtr, d[tr])
        pred[te] = clf.predict(Rte)
    err = pred != d
    cdist = np.minimum((pred - d) % 10, (d - pred) % 10)
    lin_dist = np.abs(pred - d)
    conf = pd.crosstab(pd.Series(d, name="true"), pd.Series(pred, name="pred")).reindex(index=range(10), columns=range(10), fill_value=0)
    # expected error-distance distribution if errors were spread uniformly over the other 9 digits
    exp_cd = {k: (2 if k < 5 else 1) / 9 for k in range(1, 6)}
    obs_cd = {int(k): float(v) for k, v in pd.Series(cdist[err]).value_counts(normalize=True).sort_index().items()}
    wrap = int(((d == 9) & (pred == 0)).sum() + ((d == 0) & (pred == 9)).sum())
    summary["decoder"] = {
        "balanced_accuracy": float(balanced_accuracy_score(d, pred)), "chance": 0.1,
        "error_circular_distance_share": obs_cd, "uniform_expectation": exp_cd,
        "wraparound_9_0_errors": wrap, "errors_total": int(err.sum()),
        "frac_errors_adjacent_linear(|d-p|=1)": float((lin_dist[err] == 1).mean()),
        "frac_errors_adjacent_circular": float((cdist[err] == 1).mean()),
    }
    conf.to_csv(f"{out}/decoder_confusion.csv")

    # ---------------- held-out geometry of nuisance-adjusted digit means (split-half by problem)
    def class_means(idx):
        m = ridge_fit(N0[idx], Y[idx], 10.0)
        R = Y[idx] - ridge_pred(m, N0[idx])
        return np.stack([R[d[idx] == k].mean(0) for k in range(10)])
    ug = np.unique(groups)
    adj_match, var2, splits = [], [], []
    for sp in range(50):
        A = RNG.permutation(ug)[: len(ug) // 2]
        ia, ib = np.isin(groups, A), ~np.isin(groups, A)
        Ma, Mb = class_means(ia), class_means(ib)
        Ma_c = Ma - Ma.mean(0)
        U, sv, Vt = np.linalg.svd(Ma_c, full_matrices=False)
        var2.append(float((sv[:2] ** 2).sum() / (sv**2).sum()))
        Zb = (Mb - Ma.mean(0)) @ Vt[:2].T
        ang = np.arctan2(Zb[:, 1], Zb[:, 0])
        order = np.argsort(ang)
        nb = sum(((order[(i + 1) % 10] - order[i]) % 10) in (1, 9) for i in range(10))
        adj_match.append(nb)
        splits.append((Ma @ Vt[:2].T - Ma.mean(0) @ Vt[:2].T, Zb))
    # null for adjacency: random arrangement of 10 labels around a loop
    null_adj = [sum(((p[(i + 1) % 10] - p[i]) % 10) in (1, 9) for i in range(10)) for p in (RNG.permutation(10) for _ in range(20000))]
    summary["class_mean_geometry"] = {
        "top2_variance_share_median": float(np.median(var2)),
        "heldout_angular_neighbors_numerical_median_of_10": float(np.median(adj_match)),
        "heldout_angular_neighbors_mean": float(np.mean(adj_match)),
        "null_mean": float(np.mean(null_adj)), "null_p_ge_median": float(np.mean(np.array(null_adj) >= np.median(adj_match))),
    }

    # ---------------- cross-fitted settling of the answer subspace vs random / PCA subspaces
    rows_primary = meta[meta["pos"] >= 1].reset_index(drop=True)
    fold_of_pid = dict(zip(D["pid"], folds))
    rows_primary["fold"] = rows_primary["pid"].map(fold_of_pid)
    curves = {k: [] for k in ("answer r=2", "answer r=9", "random r=9", "PCA r=9")}
    dec_curve = []
    steps = np.arange(K + 1)
    for f in range(N_FOLDS):
        tr = folds != f
        # answer subspace from training class means (nuisance-adjusted), orthonormal
        m = ridge_fit(N0[tr], Y[tr], 10.0)
        Rtr = Y[tr] - ridge_pred(m, N0[tr])
        M = np.stack([Rtr[d[tr] == k].mean(0) for k in range(10)])
        _, _, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
        _, _, Pt = np.linalg.svd(Y[tr] - Y[tr].mean(0), full_matrices=False)
        subs = {"answer r=2": Vt[:2].T, "answer r=9": Vt[:9].T, "PCA r=9": Pt[:9].T}
        Q, _ = np.linalg.qr(RNG.standard_normal((Y.shape[1], 9)))
        subs["random r=9"] = Q
        te_rows = rows_primary[rows_primary["fold"] == f]
        Tr = T[te_rows["row"].to_numpy()]  # (n, K+1, 64), individual seeds
        ref = late[te_rows["row"].to_numpy()]
        for name, B in subs.items():
            scale = np.sqrt(((Y[tr] - Y[tr].mean(0)) @ B).var(0).sum())  # between-problem spread in subspace
            dist = np.linalg.norm(np.einsum("nkd,dr->nkr", Tr - ref[:, None], B), axis=-1) / scale
            curves[name].append(dist)
        clf = LogisticRegression(C=0.1, max_iter=5000).fit(Y[tr], d[tr])
        dt = te_rows["digit"].to_numpy()
        acc = np.stack([(clf.predict(Tr[:, k]) == dt) for k in steps], 1)
        dec_curve.append(acc)
    curves = {k: np.concatenate(v) for k, v in curves.items()}
    dec_curve = np.concatenate(dec_curve)

    def settle(dist, thr=0.1):
        ok = dist < thr
        last_bad = np.array([np.nonzero(~o)[0].max() if (~o).any() else -1 for o in ok])
        s = (last_bad + 1).astype(float)
        s[last_bad == dist.shape[1] - 1] = np.nan
        return s
    summary["settling"] = {name: {"median_step_below_0.1": float(np.nanmedian(settle(c))),
                                  "median_dist_at": {int(k): float(np.median(c[:, k])) for k in (4, 8, 16, 32, 64)}}
                           for name, c in curves.items()}
    summary["probe_decoding_accuracy_at"] = {int(k): float(dec_curve[:, k].mean()) for k in (1, 4, 8, 16, 32, 64, 256)}

    # ---------------------------------------------------------------- figures
    # A. encoding models + nulls
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    mods = res["true"]["models"]
    names = [m_["model"] for m_ in mods]
    sh = [m_["share_of_categorical_gain"] for m_ in mods]
    lo_ = [m_["share_of_categorical_gain"] - m_["share_ci"][0] for m_ in mods]
    hi_ = [m_["share_ci"][1] - m_["share_of_categorical_gain"] for m_ in mods]
    axes[0].bar(range(4), sh, color=CAT[0], width=0.55)
    axes[0].errorbar(range(4), sh, yerr=[lo_, hi_], fmt="none", ecolor=INK2, lw=1.5)
    axes[0].set_xticks(range(4), ["number line", "circle", "line + circle", "lookup table\n(any 10 points)"])
    axes[0].set_ylabel("share of digit information captured\n(held-out, vs. lookup table = 1)")
    axes[0].set_title("How well does each shape explain the states?")
    for i, v in enumerate(sh):
        axes[0].annotate(f"{v:.0%}", (i, v), textcoords="offset points", xytext=(10, 3), fontsize=9, color=INK2)
    bins = np.linspace(min(perm_circle.min(), rand_code.min(), 0), max(true_circle, perm_circle.max()) * 1.05, 40)
    axes[1].hist(perm_circle, bins=bins, color=CAT[0], alpha=0.55, label="circle, digits in scrambled order")
    axes[1].hist(rand_code, bins=bins, color=CAT[1], alpha=0.45, label="random 2-D layout of the 10 digits")
    axes[1].axvline(true_circle, color=INK, lw=2)
    axes[1].annotate("circle in true order\n0→1→…→9→0", (true_circle, axes[1].get_ylim()[1] * 0.45), xytext=(-8, 0),
                     textcoords="offset points", ha="right", fontsize=9)
    axes[1].set_xlabel("held-out variance explained beyond nuisance (R² gain)")
    axes[1].set_ylabel("count (200 each)")
    axes[1].set_title("Is the numerical cyclic order special?")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{out}/A_shape_models.png", dpi=150); plt.close(fig)

    # B. held-out class means in a plane fit on the other half of problems
    Za, Zb = splits[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for Z, c, lab, mk in ((Za, CAT[0], "fit half (defines plane)", "o"), (Zb, CAT[1], "held-out half", "s")):
        loop = np.vstack([Z, Z[:1]])
        ax.plot(loop[:, 0], loop[:, 1], color=c, lw=1.2, alpha=0.7)
        ax.scatter(Z[:, 0], Z[:, 1], color=c, s=46, marker=mk, edgecolor=SURF, lw=1, label=lab, zorder=3)
        for k in range(10):
            ax.annotate(str(k), Z[k], textcoords="offset points", xytext=(5, 4), fontsize=9, color=INK)
    ax.set_title("Average state per answer digit (nuisance removed)\nconnected in numerical order 0→1→…→9→0")
    ax.set_xlabel("plane axis 1"); ax.set_ylabel("plane axis 2"); ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(f"{out}/B_digit_means_plane.png", dpi=150); plt.close(fig)

    # C. decoder error distances
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ks = np.arange(1, 6)
    ax.bar(ks - 0.18, [obs_cd.get(int(k), 0) for k in ks], width=0.34, color=CAT[0], label="observed")
    ax.bar(ks + 0.18, [exp_cd[int(k)] for k in ks], width=0.34, color=GRID, edgecolor=INK2, label="if errors were random")
    ax.set_xticks(ks, ["±1", "±2", "±3", "±4", "5"])
    ax.set_xlabel("how far off the decoder's wrong guesses are (around the 0–9 loop)")
    ax.set_ylabel("share of errors")
    ax.set_title("When a decoder confuses digits, are they neighbors?")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/C_decoder_errors.png", dpi=150); plt.close(fig)

    # D. settling
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    cols = {"answer r=2": CAT[1], "answer r=9": CAT[0], "PCA r=9": CAT[2], "random r=9": INK2}
    for name, cv in curves.items():
        med = np.median(cv, 0)
        axes[0].plot(steps[1:], med[1:], color=cols[name], label=name.replace("answer", "answer subspace").replace("r=", "dim "))
    axes[0].axhline(0.1, color=INK2, lw=1, ls=":")
    axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
    axes[0].set_xlabel("recurrent iterations (log)")
    axes[0].set_ylabel("distance to own late state\n(relative to spread between problems)")
    axes[0].set_title("Does the answer part settle before the rest?")
    axes[0].legend(fontsize=8)
    axes[1].plot(steps[1:], dec_curve.mean(0)[1:], color=CAT[0], label="probe decoding (held-out problems)")
    axes[1].set_xscale("log", base=2); axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("recurrent iterations (log)"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Digit decodable from the state")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{out}/D_settling.png", dpi=150); plt.close(fig)

    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
