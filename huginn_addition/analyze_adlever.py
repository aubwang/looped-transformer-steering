"""Stage 2 of the adapter-pullback test: do fixed levers built from one-block pullbacks (no 48-step resolvent) steer
held-out prompts? (intervene_app.py `dglever` mode eval, config adlever_eval). Prespecified before the run.
Usage: python -I huginn_addition/analyze_adlever.py results/<adlever_eval_run> results/<dglever_eval_run> results/<adaptcov_run>

Pushes (results/adlever_pushes.npz, norm 3, built on TRAIN items by make_adlever_pushes.py; sha256 verified against
results/adlever_pushes.sha256): L_k (48-step resolvent levers, reference), A_k (adapter pullback, adapter path only),
W_k (adapter weights only), B_k (full one-step pullback J^T g). Eval: the 20 held-out dglever prompts, persistent forcing
s <- F(s + P) for 48 loops (x* itself still needs 96 loops). Target effect E_F(j, k) = z_k(F_k) - z_k(sham); per item the
MEAN over k (as in analyze_dglever.py P1). Summaries are MEANS over items with item-bootstrap 95% CIs of the mean (5000).
GATES (all required for labels): stage-1 gate passed; push file hash matches; all arrays finite; L replication vs the
original dglever eval run cell by cell: max |diff| over all 200 (item, k) L target cells < 1e-3 and sham z max |diff|
< 1e-3; L reference adequately positive: mean E_L CI lower bound > 0.5.
H1 (PRIMARY) ADAPTER PULLBACK: per-item E_A - 0.5 E_L. CI > 0 -> 'A RETAINS >= HALF OF L'; CI < 0 -> 'A BELOW HALF OF L';
   else INCONCLUSIVE. 'Half of L' is a practical efficacy benchmark, not a mechanistic threshold.
H2, H3 (SECONDARY, no multiplicity correction): E_A - E_W (CI > 0 'NORM-CHAIN GEOMETRY ADDS', < 0 'NORM CHAIN HURTS');
   E_B - 0.5 E_L (labels as H1 for B).
Descriptive: ratio of means E_F / E_L; endpoint displacement and final-step drift by family; original G/C effects and the
original random summary (copied from its analysis_summary.json). Interpretation limits: transferred steering efficacy,
not prediction of response matrices or causal necessity of the adapter; norm 3 linearity unverified.
PRE-DATA NOTE (moved here from the former file analyze_adlever.py.prenote; its content is unchanged):
  Recorded before stage-2 data (2026-11-06): stage 1 shows a ~ 2.76 w (cos(a, w) median 0.9987, min 0.9937; |a|/|w| 5-95%
  2.72-2.86), i.e. the residual/norm chain acts on these readout covectors as ~scalar. A_k and W_k pushes are therefore nearly
  the same direction, and H2 (A - W) is expected to be uninformative; it is reported but not interpreted.
"""
import hashlib
import json
import sys

import numpy as np


def boot(v, B=5000, seed=0):
    v = np.asarray(v); rng = np.random.default_rng(seed)
    b = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def tcells(Z, names, fam):
    idx = [names.index(f"{fam}{k}") + 1 for k in range(10)]  # +1: index 0 = sham
    return np.stack([Z[:, idx[k], k] - Z[:, 0, k] for k in range(10)], 1)  # (n, 10) target cells


def main(re, ro, ra):
    z = np.load(f"{re}/dglever.npz"); zo = np.load(f"{ro}/dglever.npz")
    assert (z["uid"] == zo["uid"]).all()
    h_ = hashlib.sha256(open("results/adlever_pushes.npz", "rb").read()).hexdigest()
    hash_ok = open("results/adlever_pushes.sha256").read().split()[0] == h_
    names = [str(x) for x in np.load("results/adlever_pushes.npz")["names"]]
    names_o = [str(x) for x in np.load("results/dglever_pushes.npz")["names"]]
    assert names == [f"{f}{k}" for f in "LAWB" for k in range(10)]
    Z, Zo = z["Z"].astype(np.float64), zo["Z"].astype(np.float64)
    finite = bool(np.isfinite(Z).all())
    T = {f: tcells(Z, names, f) for f in "LAWB"}
    To = {f: tcells(Zo, names_o, f) for f in "LGC"}
    E = {f: v.mean(1) for f, v in T.items()}
    rep_cell = float(np.max(np.abs(T["L"] - To["L"]))); rep_sham = float(np.max(np.abs(Z[:, 0] - Zo[:, 0])))
    stage1 = bool(json.load(open(f"{ra}/analysis_summary.json"))["gate_pass"])
    Lci = boot(E["L"])
    gates = dict(stage1=stage1, push_hash=hash_ok, finite=finite, L_replication_max_cell=rep_cell, sham_replication_max=rep_sham,
                 L_mean=float(E["L"].mean()), L_ci=Lci)
    gate_pass = bool(stage1 and hash_ok and finite and rep_cell < 1e-3 and rep_sham < 1e-3 and Lci[0] > 0.5)
    out = dict(gates=gates, gate_pass=gate_pass)
    out["target_effect_mean"] = {f: dict(mean=float(v.mean()), ci=boot(v)) for f, v in E.items()}
    out["ratio_of_means_to_L"] = {f: float(E[f].mean() / E["L"].mean()) for f in "AWB"}
    so = json.load(open(f"{ro}/analysis_summary.json"))["P1"]
    out["original_scale"] = dict(mean_target_effect=so["mean_target_effect"], random_median_abs=so["random_median_abs"])
    out["endpoint_displacement_median"] = {f: float(np.median(z["endpoint_displacement_q"][:, [names.index(f"{f}{k}") + 1 for k in range(10)]])) for f in "LAWB"}
    out["last_step_drift_median"] = {f: float(np.median(z["last_step_max_drift"][:, [names.index(f"{f}{k}") + 1 for k in range(10)]])) for f in "LAWB"}
    out["last_step_drift_median"]["sham"] = float(np.median(z["last_step_max_drift"][:, 0]))

    def lab(v, pos, neg):
        c = boot(v)
        return dict(mean=float(np.mean(v)), ci=c, label=(pos if c[0] > 0 else neg if c[1] < 0 else "INCONCLUSIVE"))
    if gate_pass:
        out["H1_primary_A_vs_half_L"] = lab(E["A"] - 0.5 * E["L"], "A RETAINS >= HALF OF L", "A BELOW HALF OF L")
        out["H2_secondary_A_minus_W"] = lab(E["A"] - E["W"], "NORM-CHAIN GEOMETRY ADDS", "NORM CHAIN HURTS")
        out["H3_secondary_B_vs_half_L"] = lab(E["B"] - 0.5 * E["L"], "B RETAINS >= HALF OF L", "B BELOW HALF OF L")
    else:
        out["labels"] = "SUPPRESSED (gate failure)"
    json.dump(out, open(f"{re}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*[a.rstrip("/") for a in sys.argv[1:4]])
