"""Pre-registered analysis of the Ouro-1.4B read-space run (ouro_app.py). Written before the production run (2026-11-07),
after the Huginn ctrlprobe results (SHARED read-space suppression in both Huginn contexts) were known.

Question: does a second looped LM, with a different architecture (whole 24-layer stack looped, final norm on the state,
linear readout; no adapter, no coda, no fixed point), also have low self-gain on the directions its output reads?

Per context and loop map J_k (k = 3: the map producing the read state h_4, PRIMARY; k = 2: the loop before, secondary),
the statistics of analyze_ctrlprobe.evaluate (tau from 16 Gaussian probes, Proposition R sd, excess = alpha - tau, Z,
families task / C1 / C2 / C3, readings TASK-SPECIFIC / SHARED / PARTIAL / UNCLEAR / NO TASK DEFICIT).
PRIMARY READING (C3, k = 3; the reading measures a SIGNED self-gain deficit relative to the trace, alpha - tau, within the
own-position block; it is not a statement about damping or about the complete read space), with
s = -median_i Delta_C3(i) / median_i tau_i:
  STRONG DEFICIT  CI(Delta_C3) entirely < 0, s >= 0.5 and >= 90% of C3 directions below all 16 probes of their context
                  (branch "trace": used if median tau > 0.05; branch "Z": otherwise median Z_C3 <= -10 replaces s >= 0.5;
                  the branch used is reported)
  NOT DETECTED    CI(Delta_C3) not entirely < 0 (no equivalence margin: this is not evidence of absence)
  WEAK DEFICIT    otherwise
COORDINATE CONTROL (pre-registered, C3flip = C3 with random coordinate signs, same coordinate energies): the deficit is
called COORDINATE-WISE if the C3flip deficit is at least 75% of the C3 deficit (median_i Delta_C3flip(i) / median_i
Delta_C3(i) >= 0.75), ALIGNMENT if it is at most 25% (C3 deficit not explained by coordinate energies), MIXED otherwise.
(Note after review: the ratio is computed as registered, but its deficit-attribution reading applies only when a C3
deficit is detected; with a positive C3 excess it is descriptive. Sign flips keep coordinate energies and remove the
off-diagonal part of <u, J^T u> only in expectation: E_S[(Su)^T J^T (Su)] = sum_i J_ii u_i^2.)
Descriptive: the same at k = 2; absolute alpha, one-step norms |J^T u| and self-cosines; the frozen-Jacobian second
moment <u, (J_k^T)^2 u> (needed by Proposition R; not a two-loop quantity) and, separately, the actual two-loop pullback
<u, J_2^T J_3^T u>; retention inside the SAMPLED read span (baseline = its numerical rank / d) and the compression of J^T
onto it vs the probe span (as analyze_readspace.py; "rotation away from the sampled read directions within the own-position
block", nothing about the full read space or prefix positions); the final-norm diagnostic (u^T N u for C3 vs probes, and
its diagonal-only part); state drift between loops; task behaviour (p(True) > p(False) against the label).
GATES: complete run (smoke runs, flagged in cfg, are exempt); loops include k = 3 and k = 2; finite arrays (all consumed,
incl. p_TF); unit starts (< 1e-5); ugram diagonal = 1 and diag(x1) = alpha from gram3 (< 1e-6); |J^T u| > 0; manual loop
states vs the model's own 4 states (max relative error < 1e-4); logits at h_4 vs the model's exit-at-last-step logits
(max abs < 1e-3); median adjoint error (FD step 0.01 |h|) < 5e-3 in every family and loop; tau(B^2) > 0.
Usage: python -I analyze_ouro.py <ouroprobe run dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ctrlprobe import boot_median, evaluate  # noqa: E402
from analyze_readspace import compress, retention  # noqa: E402


def primary(ev):
    c = ev["C3"]; tau = ev["tau"][0]; branch = "trace" if tau > 0.05 else "Z"
    if not c["Delta"][2] < 0:
        return "NOT DETECTED", branch
    big = (-c["Delta"][0] / tau >= 0.5) if branch == "trace" else (c["Z"][0] <= -10)
    return ("STRONG DEFICIT" if big and c["below_all"] >= 0.9 else "WEAK DEFICIT"), branch


def coord_control(ev):
    r = ev["C3flip"]["Delta"][0] / ev["C3"]["Delta"][0]
    return dict(ratio=float(r), reading="COORDINATE-WISE" if r >= 0.75 else ("ALIGNMENT" if r <= 0.25 else "MIXED"))


def rotation(g3, ug, x1, fam, d):
    n = len(g3); R = np.where(fam <= 3)[0]; P = np.where((fam == 4) | (fam == 5))[0]
    al = g3[:, :, 0, 1]; n1 = g3[:, :, 1, 1]; a2 = g3[:, :, 0, 2]; med = lambda v: float(np.median(v))
    out = {name: dict(norm1=med(np.sqrt(n1[:, f])), alpha=med(al[:, f]), alpha2=med(a2[:, f]), cos1=med(al[:, f] / np.sqrt(n1[:, f])))
           for name, f in (("task", fam == 0), ("C1", fam == 1), ("C2", fam == 2), ("C3", fam == 3), ("C3flip", fam == 6),
                           ("probes", (fam == 4) | (fam == 5)))}
    rk = lambda G: int(np.sum(np.linalg.eigvalsh(G) > 1e-8 * np.linalg.eigvalsh(G).max()))   # cutoff of retention()
    rank = int(np.median([rk(ug[i][np.ix_(R, R)]) for i in range(n)]))
    out["retention"] = dict(read_in_R=med([retention(ug[i], x1[i], n1[i], R, R) for i in range(n)]),
                            probes_in_P=med([retention(ug[i], x1[i], n1[i], P, P) for i in range(n)]), rank_R=rank, baseline_R=rank / d,
                            baseline_P=len(P) / d)
    cr = np.array([[np.trace(B) / len(B), np.linalg.eigvalsh((B + B.T) / 2).max()] for B in (compress(ug[i], x1[i], R) for i in range(n))])
    cp = np.array([[np.trace(B) / len(B), np.linalg.eigvalsh((B + B.T) / 2).max()] for B in (compress(ug[i], x1[i], P) for i in range(n))])
    out["compression"] = dict(trace_over_dim=dict(read=med(cr[:, 0]), probes=med(cp[:, 0])), max_sym_eig=dict(read=med(cr[:, 1]), probes=med(cp[:, 1])))
    return out


def main(run):
    z = np.load(f"{run}/ouroprobe.npz"); meta = json.load(open(f"{run}/meta.json")); man = json.load(open(f"{run}/manifest.json"))
    done = z["done"]; fam = z["fam"]; loops = [int(k) for k in z["loops"]]; d = man["d"]; smoke = bool(man["cfg"].get("smoke"))
    out = dict(run=run, smoke=smoke, gates=dict(n_items=int(len(done)), n_done=int(done.sum())))
    g = out["gates"]

    def fail(msg):
        g["PASS"] = False; print("GATES FAILED:", msg); json.dump(out, open(f"{run}/ouro_summary.json", "w"), indent=1); return out

    if done.sum() < 2 or (not smoke and not done.all()):
        return fail("incomplete run")
    if not (3 in loops and 2 in loops):
        return fail("loops must include k = 3 (primary) and k = 2")
    keys = ("gram3", "ugram", "x1", "x2", "adjoint_rel_err", "norm_only", "actual2", "p_TF", "val_state_rel", "logit_err", "state_drift")
    A = {k: z[k][done] for k in keys}
    g["finite"] = bool(all(np.isfinite(A[k]).all() for k in keys))
    if not g["finite"]:
        return fail("non-finite arrays")
    g3, ug, x1, adj = A["gram3"], A["ugram"], A["x1"], A["adjoint_rel_err"]
    g["unit_starts"] = float(np.abs(g3[..., 0, 0] - 1).max())
    g["ugram_diag"] = float(np.abs(np.diagonal(ug, axis1=1, axis2=2) - 1).max())
    g["x1_vs_gram3"] = float(np.abs(np.diagonal(x1, axis1=2, axis2=3) - g3[..., 0, 1]).max())
    g["min_norm1"] = float(np.sqrt(g3[..., 1, 1]).min())
    g["val_state_rel_max"] = float(A["val_state_rel"].max()); g["logit_err_max"] = float(A["logit_err"].max())
    ei = list(np.round(z["fd_eps"], 6)).index(0.01)
    g["adjoint_median_max"] = float(max(np.median(adj[:, li, ei][:, fam == f]) for li in range(len(loops)) for f in np.unique(fam)))
    g["adjoint_median_by_eps"] = {str(e): float(np.median(adj[:, :, j])) for j, e in enumerate(z["fd_eps"])}
    P = (fam == 4) | (fam == 5)
    tB2 = [(g3[:, li][:, P, 0, 2].mean(1) + g3[:, li][:, P, 1, 1].mean(1)) / 2 - g3[:, li][:, P, 0, 1].mean(1) ** 2 for li in range(len(loops))]
    g["tB2_min"] = float(min(t.min() for t in tB2))
    g["PASS"] = bool(g["unit_starts"] < 1e-5 and g["ugram_diag"] < 1e-6 and g["x1_vs_gram3"] < 1e-6 and g["min_norm1"] > 0
                     and g["val_state_rel_max"] < 1e-4 and g["logit_err_max"] < 1e-3 and g["adjoint_median_max"] < 5e-3 and g["tB2_min"] > 0)
    print("GATES", json.dumps(g, indent=1))
    if not g["PASS"]:
        return fail("no conclusions")
    pt = A["p_TF"]; tr = z["truth"][done]
    out["behaviour"] = dict(acc=float(np.mean((pt[:, 0] > pt[:, 1]) == (tr == 1))), mass_TF=boot_median(pt.sum(1)))
    out["state_drift"] = {f"k{k}->{k + 1}": dict(rel=float(np.median(A["state_drift"][:, k, 0])), cos=float(np.median(A["state_drift"][:, k, 1])))
                          for k in range(A["state_drift"].shape[1])}
    for li, k in enumerate(loops):
        ev = evaluate(g3[:, li], fam, d, ug, x1[:, li])
        ev["primary"], ev["branch"] = primary(ev)
        ev["coordinate_control"] = coord_control(ev)
        # EXPLORATORY (added after review): paired per-context C3 - C3flip median excess, context bootstrap
        ex = g3[:, li][:, :, 0, 1] - g3[:, li][:, P, 0, 1].mean(1)[:, None]
        ev["C3_minus_C3flip"] = boot_median(np.median(ex[:, fam == 3], 1) - np.median(ex[:, fam == 6], 1))
        ev["rotation"] = rotation(g3[:, li], ug, x1[:, li], fam, d)
        no = A["norm_only"][:, li]
        ev["norm_only"] = {name: dict(uNu=float(np.median(no[:, f, 0])), NTu=float(np.median(no[:, f, 1])), diag=float(np.median(no[:, f, 2])))
                           for name, f in (("task", fam == 0), ("C3", fam == 3), ("C3flip", fam == 6), ("probes", P))}
        out[f"loop_k{k}"] = ev
    a2 = A["actual2"]
    out["actual_two_loop"] = {name: dict(u_J2T_J3T_u=float(np.median(a2[:, f, 0])), norm=float(np.median(a2[:, f, 1])))
                              for name, f in (("task", fam == 0), ("C3", fam == 3), ("C3flip", fam == 6), ("probes", P))}
    out["PRIMARY"] = out["loop_k3"]["primary"]; out["COORDINATE_CONTROL"] = out["loop_k3"]["coordinate_control"]["reading"]
    for k in loops:
        e = out[f"loop_k{k}"]
        print(f"k={k}: tau {np.round(e['tau'], 3)}  task Delta {np.round(e['task']['Delta'], 3)} Z {np.round(e['task']['Z'], 1)}")
        for f in ("C1", "C2", "C3", "C3flip"):
            c = e[f]
            print(f"   {f}: Delta {np.round(c['Delta'], 3)} Z {np.round(c['Z'], 1)} below_all {c['below_all']:.2f} rho {np.round(c['rho'], 2)} {c['reading']}")
        print("   primary", e["primary"], e["branch"], "coordinate control", e["coordinate_control"])
        print("   rotation", json.dumps(e["rotation"]))
        print("   norm_only", json.dumps(e["norm_only"]))
    print("actual two-loop", json.dumps(out["actual_two_loop"]))
    print("behaviour", out["behaviour"], "drift", out["state_drift"])
    print("PRIMARY", out["PRIMARY"], "COORDINATE_CONTROL", out["COORDINATE_CONTROL"])
    json.dump(out, open(f"{run}/ouro_summary.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    main(sys.argv[1])
