"""Steering experiment inputs from the digit-rotation run (results/dg-20261105-200850).
Split the 100 (a, b) cells: FIT cells with (a - b) mod 10 < 5, TEST cells otherwise (both contexts of a cell together).
On FIT rows: OLS of the loop-96 fp32 answer-position state on context + a + b + U (sum-to-zero), answer-level centroids
C[U] (10 x d; effects, so differences C[U+1] - C[U] are the adjusted one-notch chords). TEST prompts: 50 rows, 5 per answer
digit (seeded), alternating contexts. Saved: results/steer_dirs.npz (C) and results/steer_prompts.json.
Usage: python -I huginn_addition/make_steer_dirs.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_digit_rotation import design, fit, levels  # noqa: E402

RUN = "results/dg-20261105-200850"


def main():
    z = np.load(f"{RUN}/dg.npz", allow_pickle=True)
    P = json.load(open(f"{RUN}/problems.json"))
    X = z["fin"][:, 0].astype(np.float64)
    fitm = np.array([((p["start"] - p["offset"]) % 10) < 5 for p in P])
    Pf = [P[i] for i in np.where(fitm)[0]]
    _, B = fit(Pf, X[fitm])
    C = levels(B, "ans")
    rng = np.random.default_rng(0)
    test = [i for i in np.where(~fitm)[0]]
    sel = []
    for u in range(10):
        cand = [i for i in test if P[i]["ans"] == u]
        sel += sorted(rng.choice(cand, 5, replace=False).tolist())
    out = []
    for i in sel:
        q = dict(P[i]); q.pop("ids", None); q.pop("pred_pos", None); out.append(q)
    np.savez("results/steer_dirs.npz", C=C.astype(np.float32), fit_uids=np.array([p["uid"] for p in Pf]))
    json.dump(out, open("results/steer_prompts.json", "w"), indent=1)
    ch = np.linalg.norm(np.diff(np.vstack([C, C[:1]]), axis=0), axis=1)
    print("fit rows", int(fitm.sum()), "test prompts", len(out), "chord norms", ch.round(1), "state norm ~", np.linalg.norm(X, axis=1).mean().round(1))


if __name__ == "__main__":
    main()
