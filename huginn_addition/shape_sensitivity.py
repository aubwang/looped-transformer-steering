"""Nuisance-sensitivity check for the cyclic digit structure (suggested by GPT-6 Astra's results review).

Usage: python -I huginn_addition/shape_sensitivity.py results/<pilot_run>

Repeats the encoding-model comparison under progressively stronger controls for answer-prefix structure:
  base:      additive nuisance (as in shape_analysis.py)
  strong:    + previous-token x position, previous-token x column, previous-token x family interactions;
             digit features AND states are residualized on the nuisance inside each training fold (Frisch-Waugh)
  strong_no9prev: strong, excluding rows whose previous answer digit is 9 (the main source of digit dependence)
For each, reports the circle / line share of the categorical gain and a scrambled-order null (100 permutations).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__file__))
from shape_analysis import ALPHAS, LATE, N_FOLDS, N_INNER, digit_features, group_folds, load, ridge_fit, ridge_pred  # noqa: E402

N_PERM = 100
rng = np.random.default_rng(1)


def design(df, strong):
    cats = ["prev", "pos", "col", "n_answer", "n_digits", "overflow", "demo", "family"]
    X = df[cats].astype(str)
    if strong:
        X = X.assign(prev_pos=df["prev"] + "_" + df["pos"].astype(str), prev_col=df["prev"] + "_" + df["col"].astype(str),
                     prev_fam=df["prev"] + "_" + df["family"])
    return pd.get_dummies(X, drop_first=True).to_numpy(float)


def cv_sse_fw(N, F, Y, groups, folds):
    """Held-out SSE. Residualize Y and digit features F on nuisance N in each training fold (OLS-ish ridge, small
    alpha), then ridge Y_res ~ F_res with inner grouped CV. F may have zero columns (nuisance-only model)."""
    sse = np.zeros(len(Y))
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        mY = ridge_fit(N[tr], Y[tr], 1e-2)
        Ytr, Yte = Y[tr] - ridge_pred(mY, N[tr]), Y[te] - ridge_pred(mY, N[te])
        if F.shape[1] == 0:
            sse[te] = (Yte**2).sum(1)
            continue
        mF = ridge_fit(N[tr], F[tr], 1e-2)
        Ftr, Fte = F[tr] - ridge_pred(mF, N[tr]), F[te] - ridge_pred(mF, N[te])
        inner = group_folds(groups[tr], N_INNER, seed=100 + f)
        best, best_a = np.inf, ALPHAS[0]
        for a in ALPHAS:
            e = 0.0
            for g in range(N_INNER):
                i_tr, i_te = inner != g, inner == g
                m = ridge_fit(Ftr[i_tr], Ytr[i_tr], a)
                e += ((ridge_pred(m, Ftr[i_te]) - Ytr[i_te]) ** 2).sum()
            if e < best:
                best, best_a = e, a
        m = ridge_fit(Ftr, Ytr, best_a)
        sse[te] = ((ridge_pred(m, Fte) - Yte) ** 2).sum(1)
    return sse


def main(run):
    meta, T, K = load(run)
    lo, hi = LATE
    late = T[:, lo : hi + 1].mean(1)
    meta["row"] = np.arange(len(meta))
    g = meta.groupby(["uid", "pos"])
    S = g.first().reset_index()
    S["state_rows"] = g["row"].apply(list).values
    Yall = np.stack([late[r].mean(0) for r in S["state_rows"]])
    out = {}
    for name, strong, mask_fn in (
        ("base_FW", False, lambda D: D["pos"] >= 1),
        ("strong", True, lambda D: D["pos"] >= 1),
        ("strong_no9prev", True, lambda D: (D["pos"] >= 1) & ~D["prev"].isin(["d9", "sp9"])),
    ):
        m = mask_fn(S).to_numpy()
        D, Y = S[m].reset_index(drop=True), Yall[m]
        groups = D["pid"].to_numpy()
        folds = group_folds(groups, N_FOLDS, seed=0)
        N = design(D, strong)
        d = D["digit"].to_numpy()
        sst = ((Y - Y.mean(0)) ** 2).sum()
        sse = {k: cv_sse_fw(N, digit_features(d, k), Y, groups, folds) for k in ("none", "line", "circle", "categorical")}
        cat_gain = sse["none"].sum() - sse["categorical"].sum()
        res = {"rows": int(len(D)), "problems": int(D["pid"].nunique()), "nuisance_cols": int(N.shape[1]),
               "categorical_gain_R2": float(cat_gain / sst)}
        for k in ("line", "circle"):
            res[f"{k}_share_of_categorical"] = float((sse["none"].sum() - sse[k].sum()) / cat_gain)
        true_gain = sse["none"].sum() - sse["circle"].sum()
        perms = np.array([sse["none"].sum() - cv_sse_fw(N, digit_features(rng.permutation(10)[d], "circle"), Y, groups, folds).sum()
                          for _ in range(N_PERM)])
        res["circle_vs_scrambled_p"] = float((1 + (perms >= true_gain).sum()) / (1 + N_PERM))
        res["circle_gain_over_best_scrambled"] = float(true_gain / perms.max())
        out[name] = res
        print(name, json.dumps(res), flush=True)
    json.dump(out, open(f"{run}/shape/sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
