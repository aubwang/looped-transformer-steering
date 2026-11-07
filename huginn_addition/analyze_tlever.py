"""Cheap truth levers on held-out statements (intervene_app.py `dstar_eval`, config tlever_eval; pushes from
make_tlever_pushes.py). Prespecified before the runs; revised per Astra's pre-run review (before any eval data).
Usage: python -I huginn_addition/analyze_tlever.py results/<run1> results/<run2> results/<run3> results/<truthcov_run> results/<dseval_run>

60 held-out judgment statements (30 units; cities + compare; training units disjoint). Persistent forcing s <- F(s + P)
for 24 loops from x* (loop 64). Directions (label-free): L (48-step resolvent d*), B (one-step J^T g), W (adapter weights
W_s^T g), G (readout gradient); population (train) and item-specific (L_item, B_item); random at 12.5. Doses 1.5625,
3.125, 12.5. The intervention SIGN uses the ground-truth label: runs 1-2 push toward the OPPOSITE label, run 3 pushes the
population L/B/W toward the item's OWN label ('+own'). '+' raising R is the first-order orientation convention.
E(c) = movement of R = log p(True) - log p(False) toward the pushed-toward label vs sham at offset 24
(opposite: sgn = -1 for true items, +1 for false; own: the reverse). Medians over items; unit-cluster bootstrap within
dataset (5000), as in analyze_dstar.py.
GATES (all required for labels): finite arrays and finite truthcov adjoint errors; matching readout offsets, K0, code
hash, model revision and items across the three runs (and items equal to the dseval run); NPZ uids and names equal to the
push manifest; recorded push hashes equal the manifest; sham R at 24 identical across runs (max |diff| < 1e-4) and equal
to the earlier dseval sham at 24 (< 1e-3); itemwise replication of L_pop@12.5 (opposite) against the earlier dseval
dstar_pop at 24 (max |diff| < 1e-2); truthcov adjoint median < 0.05; sham off-TF KL at 24 median < 1e-3.
L ADEQUACY (per comparison): the reference L condition at 1.5625 must have median E >= 0.25 with CI lower bound > 0;
otherwise that comparison's label is 'REFERENCE INADEQUATE'.
H1 (PRIMARY): paired median contrast E(B_pop@1.5625) - 0.5 E(L_pop@1.5625) (opposite direction): CI > 0 -> 'PAIRED
   CONTRAST B - 0.5L POSITIVE'; CI < 0 -> 'PAIRED CONTRAST B - 0.5L NEGATIVE'; else INCONCLUSIVE. This is a paired
   contrast, not a ratio of medians.
H2, H3 (EXPLORATORY, no multiplicity correction): the same for W_pop vs L_pop and B_item vs L_item at 1.5625.
Descriptive: E for every condition at offsets 8, 16, 24 (24 is the fixed endpoint; not all families need have settled);
ratios of medians to the matching L condition (suppressed if |median E_L| < 0.1); judgment accuracy (sign of R vs label)
under sham and each condition, with corrections (sham wrong -> right) and corruptions (sham right -> wrong) by truth
stratum; collateral at 24: True+False mass change vs sham, and off-TF KL(x* reference || pushed) over all tokens except
True/False, renormalised (divergence from x*, approximating the offset-24 sham since sham KL is ~0; this is on-prompt
distribution change, not unrelated-task degradation); compute: measured per-item seconds (truthcov, cuda-synchronised)
for the shared 64-loop warm-up, g, one VJP and W; L's construction cost is ESTIMATED as g + 48 VJPs (not measured
end-to-end); the historical dstar_vjp wall time (incl. model load) is reported separately; population directions are
built once and amortised over all held-out items.
Limits: R is a True-vs-False token readout; a lever moving R is a readout lever and need not change the model's truth
representation; single prompt format; B/W linearity in dose is not assumed.
"""
import hashlib
import json
import sys

import numpy as np

RUN_FILES = ["results/tlever_pushes_1.npz", "results/tlever_pushes_2.npz", "results/tlever_pushes_3.npz"]


def main(r1, r2, r3, rt, rd):
    info = json.load(open("results/tlever_pushes.json"))
    runs = [r1, r2, r3]
    Z = {r: np.load(f"{r}/dstar_eval.npz") for r in runs}
    meta = {r: json.load(open(f"{r}/meta.json")) for r in runs}
    items = json.load(open(f"{r1}/items.json"))
    items_same = all(open(f"{r}/items.json").read() == open(f"{r1}/items.json").read() for r in runs)
    items_dseval = [it["uid"] for it in items] == [it["uid"] for it in json.load(open(f"{rd}/items.json"))]
    cfg_same = all(list(Z[r]["ro"]) == list(Z[r1]["ro"]) and meta[r]["cfg"]["K0"] == meta[r1]["cfg"]["K0"]
                   and meta[r]["cfg"]["code_sha256"] == meta[r1]["cfg"]["code_sha256"] and meta[r]["revision"] == meta[r1]["revision"] for r in runs)
    uid_ok = all([int(u) for u in Z[r]["uid"]] == [int(it["uid"]) for it in items] for r in runs)
    names_ok = all([str(x) for x in Z[r]["names"]] == info[f]["names"] for r, f in zip(runs, RUN_FILES))
    hash_ok = all(meta[r]["cfg"].get("push_sha256") == info[f]["sha256"] == hashlib.sha256(open(f, "rb").read()).hexdigest() for r, f in zip(runs, RUN_FILES))
    y = np.array([int(it["truth"]) for it in items]); units = np.array([it["unit"] for it in items]); dsn = np.array([it["dataset"] for it in items])
    sgn_opp = np.where(y == 1, -1.0, 1.0)
    RO = list(Z[r1]["ro"]); ti = {t: RO.index(t) for t in (8, 16, 24)}
    R, M, K, conds = {}, {}, {}, []
    for k_, r in enumerate(runs):
        for j, n_ in enumerate(str(x) for x in Z[r]["names"]):
            key = n_ if n_ != "sham" else f"sham{k_ + 1}"
            R[key] = Z[r]["R"][:, j].astype(np.float64); M[key] = Z[r]["mass"][:, j].astype(np.float64); K[key] = Z[r]["offkl"][:, j].astype(np.float64)
            if not n_.startswith("sham"):
                conds.append(key)
    zd = np.load(f"{rd}/dstar_eval.npz"); ndn = [str(x) for x in zd["names"]]; t24o = list(zd["ro"]).index(24)
    old_sham = zd["R"][:, ndn.index("sham"), t24o].astype(np.float64)
    old_L = np.where(y == 1, zd["R"][:, ndn.index("dstar_pop-"), t24o], zd["R"][:, ndn.index("dstar_pop+"), t24o]).astype(np.float64)
    zt = np.load(f"{rt}/truthcov.npz"); mt = json.load(open(f"{rt}/meta.json"))
    t24 = ti[24]
    gates = dict(finite=all(bool(np.isfinite(Z[r][k]).all()) for r in Z for k in ("R", "mass", "offkl")) and bool(np.isfinite(zt["adjoint_rel_err"]).all()),
                 items_same=items_same, items_match_dseval=items_dseval, config_same=cfg_same, uids_ok=uid_ok, names_ok=names_ok, push_hash=hash_ok,
                 sham_between_runs=float(max(np.max(np.abs(R[f"sham{k}"][:, t24] - R["sham1"][:, t24])) for k in (2, 3))),
                 sham_vs_dseval=float(np.max(np.abs(R["sham1"][:, t24] - old_sham))),
                 L12_replication_vs_dseval=float(np.max(np.abs(R["L_pop@12.5"][:, t24] - old_L))),
                 truthcov_adjoint_median=float(np.median(zt["adjoint_rel_err"])),
                 sham_offkl_median=float(np.median(K["sham1"][:, t24])))
    gate_pass = bool(gates["finite"] and items_same and items_dseval and cfg_same and uid_ok and names_ok and hash_ok
                     and gates["sham_between_runs"] < 1e-4 and gates["sham_vs_dseval"] < 1e-3 and gates["L12_replication_vs_dseval"] < 1e-2
                     and gates["truthcov_adjoint_median"] < 0.05 and gates["sham_offkl_median"] < 1e-3)
    sham = R["sham1"]; msham = M["sham1"]
    sg = {c: (-sgn_opp if c.endswith("+own") else sgn_opp) for c in conds}
    E = {c: sg[c][:, None] * (R[c] - sham) for c in conds}  # (n, offsets)
    rng = np.random.default_rng(0)
    DR = []
    for _ in range(5000):
        idx = []
        for d_ in ("cities", "compare"):
            uu_ = np.unique(units[dsn == d_])
            for u in rng.choice(uu_, len(uu_)):
                idx += list(np.where(units == u)[0])
        DR.append(np.array(idx))
    ci = lambda v: [float(np.quantile([np.median(v[ix]) for ix in DR], q)) for q in (.025, .975)]
    out = dict(gates=gates, gate_pass=gate_pass)
    out["E_median_by_offset"] = {c: {t: float(np.median(E[c][:, i])) for t, i in ti.items()} for c in conds}
    out["E24_ci"] = {c: ci(E[c][:, t24]) for c in conds}

    def lref(c):
        base, rest = c.split("@")
        return ("L_item" if base.endswith("item") else "L_pop") + "@" + rest
    ratios = {}
    for c in conds:
        if "@" in c and not c.startswith("rand"):
            den = float(np.median(E[lref(c)][:, t24]))
            ratios[c] = float(np.median(E[c][:, t24]) / den) if abs(den) >= 0.1 else "SUPPRESSED (|median E_L| < 0.1)"
    out["ratio_of_medians_to_L"] = ratios
    acc = {"sham": float(np.mean(np.sign(sham[:, t24]) == np.where(y == 1, 1, -1)))}
    cc = {}
    for c in conds:
        right_s = np.sign(sham[:, t24]) == np.where(y == 1, 1, -1); right_p = np.sign(R[c][:, t24]) == np.where(y == 1, 1, -1)
        acc[c] = float(right_p.mean())
        cc[c] = {nm: dict(corrections=int((~right_s & right_p & m).sum()), corruptions=int((right_s & ~right_p & m).sum()), n=int(m.sum()))
                 for nm, m in (("true", y == 1), ("false", y == 0))}
    out["accuracy_at_24"] = acc; out["corrections_corruptions_at_24"] = cc
    out["collateral_at_24"] = {c: dict(tf_mass_change_median=float(np.median(M[c][:, t24] - msham[:, t24])), offtf_kl_median=float(np.median(K[c][:, t24])),
                                       offtf_kl_q90=float(np.quantile(K[c][:, t24], 0.9))) for c in conds}
    out["collateral_at_24"]["sham"] = dict(offtf_kl_median=float(np.median(K["sham1"][:, t24])))
    tm = mt["timings"]; n_ = sum(t["n"] for t in tm)
    per = {k: sum(t[k] for t in tm) / n_ for k in ("warmup_s", "g_s", "vjp_s", "w_s")}
    out["compute_seconds_per_item"] = dict(measured_shared_warmup_64_loops=per["warmup_s"], measured_G=per["g_s"], measured_W=per["g_s"] + per["w_s"],
                                           measured_B=per["g_s"] + per["vjp_s"], estimated_L_g_plus_48_vjps=per["g_s"] + 48 * per["vjp_s"],
                                           batch=tm[0]["n"], historical_dstar_vjp_wall_s_120_items_incl_load=252.7)

    def lab(c, ref):  # (label names the tested family; fixed after Astra's results review: it was hardcoded to 'B')
        v = E[c][:, t24] - 0.5 * E[ref][:, t24]
        lm, lc = float(np.median(E[ref][:, t24])), ci(E[ref][:, t24])
        d = dict(median=float(np.median(v)), ci=ci(v), reference_median=lm, reference_ci=lc)
        if not (lm >= 0.25 and lc[0] > 0):
            d["label"] = "REFERENCE INADEQUATE"
        else:
            f = c.split("@")[0]
            d["label"] = (f"PAIRED CONTRAST {f} - 0.5L POSITIVE" if d["ci"][0] > 0 else f"PAIRED CONTRAST {f} - 0.5L NEGATIVE" if d["ci"][1] < 0 else "INCONCLUSIVE")
        return d
    if gate_pass:
        out["H1_primary_Bpop@1.5625"] = lab("B_pop@1.5625", "L_pop@1.5625")
        out["H2_exploratory_Wpop@1.5625"] = lab("W_pop@1.5625", "L_pop@1.5625")
        out["H3_exploratory_Bitem@1.5625"] = lab("B_item@1.5625", "L_item@1.5625")
    else:
        out["labels"] = "SUPPRESSED (gate failure)"
    json.dump(out, open(f"{r1}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*[a.rstrip("/") for a in sys.argv[1:6]])
