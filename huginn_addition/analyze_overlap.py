"""Overlap-matched control for the cheap levers (intervene_app.py `dglever` mode eval, config ovl_eval). Prespecified;
revised per Astra before any run.
Usage: python -I huginn_addition/analyze_overlap.py results/<ovl_eval_run_1> results/<ovl_eval_run_2> results/<adlever_eval_run>

Pushes (make_overlap_pushes.py; results/ovl_pushes.json holds the hashes): run 1 = Actrla, Bctrla, Aperp; run 2 = Actrlb,
Bctrlb, Bperp. Fctrl = same L_k overlap and norm as F_k with a random remainder orthogonal to L_k (2 independent draws per
(family, k)); Fperp = F_k's own non-L remainder (norm 3). A = the full adapter-path pullback family (A ~ W), B = the one-step
pullback. Same 20 held-out prompts, 48 loops of persistent forcing, centered digit logits; target effect per (item, k) =
z_k(push_k) - z_k(sham); per item MEAN over k; means over items; item-bootstrap CIs (5000). E_A, E_B, E_L come from the
adlever_eval run.
GATES: all three runs finite; each ovl run's recorded push_sha256 equals its file's hash in ovl_pushes.json; identical
items.json across the three runs; same model revision, K0, T and code hash across the two ovl runs; sham z identical
to the adlever_eval run (max |diff| < 1e-3). (The adlever_eval run predates push-hash recording; its pushes were
validated by exact L replication.)
H1 (A) and H2 (B), co-primary: excess = E_F - mean over the two draws of E_Fctrl, per item; 97.5% CIs (Bonferroni for 2).
   CI > 0 -> 'F ABOVE MATCHED CONTROL'; CI < 0 -> 'F BELOW MATCHED CONTROL'; otherwise 'NO EXCESS DETECTED (for these
   controls; CI includes zero; not evidence of equivalence)'. Conditional on the 20 fixed random remainders per family;
   item bootstrap excludes random-direction uncertainty. Per-draw excesses are reported (seed sensitivity).
Descriptive: E_Fperp (CI vs 0); linear overlap prediction c E_L (NOT an overlap-only expectation at norm 3: random
   remainders can have nonzero nonlinear effects); additivity check c E_L + sqrt(1 - c^2) E_Fperp vs E_F. Neither a
   positive excess nor Fperp steering establishes adapter-specific causality.
"""
import hashlib
import json
import sys

import numpy as np


def boot(v, q=(0.025, 0.975), B=5000, seed=0):
    v = np.asarray(v); rng = np.random.default_rng(seed)
    b = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
    return [float(np.quantile(b, x)) for x in q]


def tcells(Z, names, fam):
    idx = [names.index(f"{fam}{k}") + 1 for k in range(10)]
    return np.stack([Z[:, idx[k], k] - Z[:, 0, k] for k in range(10)], 1)


def main(r1, r2, ra):
    H = json.load(open("results/ovl_pushes.json"))
    runs = {r1: "results/ovl_pushes_1.npz", r2: "results/ovl_pushes_2.npz"}
    z = {r: np.load(f"{r}/dglever.npz") for r in (r1, r2, ra)}
    meta = {r: json.load(open(f"{r}/meta.json")) for r in (r1, r2, ra)}
    items = {r: open(f"{r}/items.json").read() for r in (r1, r2, ra)}
    g = dict(finite=all(bool(np.isfinite(z[r]["Z"]).all()) for r in z),
             push_hash=all(meta[r]["cfg"].get("push_sha256") == H[f] == hashlib.sha256(open(f, "rb").read()).hexdigest() for r, f in runs.items()),
             items_identical=len(set(items.values())) == 1,
             config_match=all(meta[r1]["cfg"][k] == meta[r2]["cfg"][k] for k in ("K0", "T", "code_sha256")) and meta[r1]["revision"] == meta[r2]["revision"] == meta[ra]["revision"]
             and all(meta[r]["cfg"]["K0"] == meta[ra]["cfg"]["K0"] and meta[r]["cfg"]["T"] == meta[ra]["cfg"]["T"] for r in (r1, r2)),
             sham_max_diff=float(max(np.max(np.abs(z[r]["Z"][:, 0] - z[ra]["Z"][:, 0])) for r in (r1, r2))))
    gate_pass = bool(g["finite"] and g["push_hash"] and g["items_identical"] and g["config_match"] and g["sham_max_diff"] < 1e-3)
    na = [str(x) for x in np.load("results/adlever_pushes.npz")["names"]]
    T = {f: tcells(z[ra]["Z"].astype(np.float64), na, f) for f in "LAB"}
    for r, f in runs.items():
        nv = [str(x) for x in np.load(f)["names"]]
        for fam in sorted({n[:-1] for n in nv}):
            T[fam] = tcells(z[r]["Z"].astype(np.float64), nv, fam)
    E = {f: v.mean(1) for f, v in T.items()}
    P = np.load("results/adlever_pushes.npz")["P"].astype(np.float64); nzf = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    Lu = nzf(P[[na.index(f"L{k}") for k in range(10)]])
    out = dict(gates=g, gate_pass=gate_pass, target_effect_mean={f: dict(mean=float(v.mean()), ci=boot(v)) for f, v in sorted(E.items())})
    for h, f in (("H1_A", "A"), ("H2_B", "B")):
        c = np.sum(nzf(P[[na.index(f"{f}{k}") for k in range(10)]]) * Lu, -1)
        ctrl = 0.5 * (E[f + "ctrla"] + E[f + "ctrlb"]); ex = E[f] - ctrl; ci = boot(ex, q=(0.0125, 0.9875))
        d = dict(excess_mean=float(ex.mean()), ci97_5=ci, per_draw_excess={s: float((E[f] - E[f + "ctrl" + s]).mean()) for s in "ab"},
                 cos_with_L_mean=float(c.mean()), observed_mean=float(E[f].mean()), control_mean=float(ctrl.mean()))
        if gate_pass:
            d["label"] = (f"{f} ABOVE MATCHED CONTROL" if ci[0] > 0 else f"{f} BELOW MATCHED CONTROL" if ci[1] < 0 else
                          "NO EXCESS DETECTED (for these controls; CI includes zero; not evidence of equivalence)")
        d["linear_overlap_prediction_mean"] = float((c[None] * T["L"]).mean(1).mean())
        d["additivity_prediction_mean"] = float((c[None] * T["L"] + np.sqrt(1 - c[None] ** 2) * T[f + "perp"]).mean(1).mean())
        d["perp_effect"] = dict(mean=float(E[f + "perp"].mean()), ci=boot(E[f + "perp"]))
        out[h] = d
    if not gate_pass:
        out["labels"] = "SUPPRESSED (gate failure)"
    json.dump(out, open(f"{r1}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*[a.rstrip("/") for a in sys.argv[1:4]])
