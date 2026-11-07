"""Sublayer decomposition of the one-step readout response (intervene_app.py `blockdecomp`). Prespecified before the run.
Usage: python -I huginn_addition/analyze_blockdecomp.py results/<blockdecomp_run>

Items: the 20 held-out dglever eval prompts at x* (96 loops), q = last token (predicts the units digit). Pushes: frozen
population pushes at norm 3 (L_k levers, G_k readout gradients, C_k centroids, R0/R1 random). Target response of push
F_k = centered digit-logit readout k after ONE application of the core block: c1 = g_k . J P. Exact linear decomposition
c1 = sum_s mu_{k,s} . dw_s over 9 writes s: ADAPTER (state->hidden map, then residual/norm chain only), ATTN_l, MLP_l
(l = 1..4; each write's total tangent carried to the output through the residual/norm chain only).
Cells = (item, k); inference over ITEMS (item bootstrap, 5000) of per-item aggregate ratios
share_X(item) = sum_k X / sum_k c1 (aggregate over the 10 targets of that item).
D0 GATES (all required): replica and mixed-attention replica relative error < 1e-5; median |sum_s comp - c1_FD| /
   median |c1_FD| < 0.01; FD vs VJP (J^T g . P) median relative < 0.05; attention Q+KV split median relative error < 0.05;
   per-source sum vs query-path contribution median relative < 0.05.
D1 WHERE THE LEVER RESPONSE COMES FROM (L, target): shares of ADAPTER, ATTN (sum of 4), MLP (sum of 4), and each of the 9
   writes. Label <GROUP>-DRIVEN if that group's share has CI lower bound > 0.5; otherwise DISTRIBUTED. Groups with share
   CI upper bound < 0 are reported as OPPOSING.
D2 ATTENTION ROUTE (L, target): query-path share vs self key/value-path share of the ATTN contribution (ratio of sums).
D3 SOURCES OF THE QUERY PATH (L, target; summed over layers): share by source category (context = demo lines and BOS;
   symbols = '+', '=' and other final-line non-digits; op_samecol = operand digits in the predicted (units) column;
   op_othercol = other operand digits; ans_prev = earlier answer tokens; self = q). The units column has no lower
   column, so op_lowercol is empty and dropped. Convention: dalpha_j * u.(v_j - y) (centered on the head output);
   per-position attributions are convention-dependent, the category sum is exact.
D4 HOW G IS SUPPRESSED (G, target): gross = sum_s |comp_s|, net = sum_s comp_s. NOT TRANSMITTED if median gross/|c0| < 0.25;
   else CANCELLATION if median |net|/gross < 0.25; else PARTIAL.
REVISIONS (Astra pre-run review, before data): (a) D1 reports ALL groups with CI lower bound > 0.5; 'NO MAJORITY GROUP'
   replaces 'DISTRIBUTED' (it means no majority demonstrated, not evenly spread). (b) Shares are signed accounting ratios
   (values outside [0, 1] are legitimate), not causal responsibility. An item's ratio is DEFINED only if its denominator
   is not itself a cancellation: |sum_k den| >= 0.5 sum_k |den|; a share is reported as UNDEFINED if fewer than 15/20
   items qualify. Signed numerators are reported alongside. (c) Labels are emitted only if every D0 gate passes; D0 also
   requires step_vs_core < 1e-5, zero nonfinite source terms, and per-item maxima (L and G target cells) of
   |sum comp - c1_FD| / |c1_FD| < 0.05 and FD vs VJP < 0.2 (worst item); the source gate checks the CATEGORIZED sum.
   (d) D4: ratios per item (median over k) with item-bootstrap CIs; NOT TRANSMITTED means a small gross target-readout
   contribution under this decomposition (not absent state transmission); CANCELLATION means cancellation between the
   nine scalar contributions (cancellation within a write is invisible). (e) 'Adapter' = the adapter write carried
   through residual/norm derivatives only; adapter-driven changes feeding later sublayers are credited to those writes.
   This is an attribution convention, not ablation/total-effect evidence. The readout is the frozen gradient g(x*), as
   in the previous analyses.
Descriptive: the same shares for G and C; immediate c0; write tangent norms per sublayer; base attention mass of row q
by category.
"""
import json
import sys

import numpy as np

GROUPS = {"ADAPTER": [0], "ATTN": [1, 3, 5, 7], "MLP": [2, 4, 6, 8]}
WRITES = ["adapter"] + [f"{t}{l}" for l in range(1, 5) for t in ("attn", "mlp")]


def boot(v, B=5000, seed=0):
    v = np.asarray(v); rng = np.random.default_rng(seed)
    b = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def ratio(num, den, min_items=15):
    """num, den: (n_items, 10). Per-item sum_k num / sum_k den, defined only if |sum_k den| >= 0.5 sum_k |den|."""
    ok = np.abs(den.sum(1)) >= 0.5 * np.abs(den).sum(1)
    v = num.sum(1)[ok] / den.sum(1)[ok]
    if ok.sum() < min_items:
        return dict(status="UNDEFINED", n_defined=int(ok.sum()), median_signed_numerator=float(np.median(num.sum(1))))
    return dict(median=float(np.median(v)), ci=boot(v), n_defined=int(ok.sum()), median_signed_numerator=float(np.median(num.sum(1))))


def main(run):
    z = np.load(f"{run}/blockdecomp.npz"); meta = json.load(open(f"{run}/meta.json"))
    names = [str(x) for x in z["names"]]; cats = [str(x) for x in z["cats"]]
    assert int(z["n_done_push"]) == len(names), "incomplete run"
    comp, c1fd, c1v, c0 = z["comp"], z["c1fd"], z["c1vjp"], z["c0"]
    qp, kv, src = z["qpath"], z["kvpath"], z["src"]
    n = comp.shape[0]
    rel = lambda a, b: float(np.median(np.abs(a - b)) / np.median(np.abs(b)))
    gates = dict(replica=meta["replica_rel_err"], attn_mix=meta["attn_mix_rel_err"], sum_vs_fd=rel(comp.sum(-1), c1fd),
                 fd_vs_vjp=rel(c1fd, c1v), split=float(np.median(z["split_err"])), source_sum=float(np.median(np.abs(z["src_err"])) / np.median(np.abs(qp))))

    def target(arr, f):  # (n, 10, ...) target-cell values for family f
        idx = [names.index(f"{f}{k}") for k in range(10)]
        return np.stack([arr[:, idx[k], k] for k in range(10)], 1)
    gates["step_vs_core"] = meta["step_vs_core_rel_err"]; gates["nonfinite_source_terms"] = meta["nonfinite_source_terms"]
    for f in ("L", "G"):
        a, b, v_ = target(comp, f).sum(-1), target(c1fd, f), target(c1v, f)
        gates[f"worst_item_sum_vs_fd_{f}"] = float(np.max(np.median(np.abs(a - b), 1) / np.median(np.abs(b), 1)))
        gates[f"worst_item_fd_vs_vjp_{f}"] = float(np.max(np.median(np.abs(b - v_), 1) / np.median(np.abs(v_), 1)))
    gate_pass = bool(gates["replica"] < 1e-5 and gates["attn_mix"] < 1e-5 and gates["sum_vs_fd"] < 0.01 and gates["fd_vs_vjp"] < 0.05
                     and gates["split"] < 0.05 and gates["source_sum"] < 0.05 and gates["step_vs_core"] < 1e-5
                     and gates["nonfinite_source_terms"] == 0
                     and all(gates[f"worst_item_sum_vs_fd_{f}"] < 0.05 and gates[f"worst_item_fd_vs_vjp_{f}"] < 0.2 for f in ("L", "G")))
    out = dict(gates=gates, gate_pass=gate_pass)

    fams = {}
    for f in ("L", "G", "C"):
        Tc, T1, T0 = target(comp, f), target(c1fd, f), target(c0, f)
        d = dict(median_c0=float(np.median(T0)), median_c1=float(np.median(T1)),
                 median_write_contrib={w: float(np.median(Tc[..., s])) for s, w in enumerate(WRITES)})
        d["group_share"] = {g: ratio(Tc[..., ix].sum(-1), T1) for g, ix in GROUPS.items()}
        d["write_share"] = {w: ratio(Tc[..., s], T1) for s, w in enumerate(WRITES)}
        Tq, Tkv = target(qp, f), target(kv, f)
        att = Tc[..., GROUPS["ATTN"]].sum(-1)
        d["attn_query_share_of_attn"] = ratio(Tq.sum(-1), att)
        d["attn_selfkv_share_of_attn"] = ratio(Tkv.sum(-1), att)
        Ts = target(src, f).sum(2)  # (n, 10, ncat) summed over layers
        d["query_source_share"] = {c: ratio(Ts[..., ci], Ts.sum(-1)) for ci, c in enumerate(cats) if c != "op_lowercol"}
        d["query_source_share_of_c1"] = {c: ratio(Ts[..., ci], T1) for ci, c in enumerate(cats) if c != "op_lowercol"}
        gross = np.abs(Tc).sum(-1); net = Tc.sum(-1)
        r1 = np.median(gross / np.abs(T0), 1); r2 = np.median(np.abs(net) / gross, 1)
        d["gross_over_abs_c0"] = dict(median=float(np.median(r1)), ci=boot(r1))
        d["abs_net_over_gross"] = dict(median=float(np.median(r2)), ci=boot(r2))
        d["write_norm_median"] = {w: float(np.median(target(np.repeat(z["dwn"][:, :, None, :], 10, 2), f)[..., s])) for s, w in enumerate(WRITES)}
        fams[f] = d
    out["families"] = fams
    L = fams["L"]["group_share"]
    G = fams["G"]
    if gate_pass:
        lab = [g for g in GROUPS if "ci" in L[g] and L[g]["ci"][0] > 0.5]
        out["D1_label"] = [g + "-DRIVEN" for g in lab] if lab else "NO MAJORITY GROUP"
        out["D1_opposing"] = [g for g in GROUPS if "ci" in L[g] and L[g]["ci"][1] < 0]
        out["D4_label"] = ("NOT TRANSMITTED" if G["gross_over_abs_c0"]["median"] < 0.25 else
                           ("CANCELLATION" if G["abs_net_over_gross"]["median"] < 0.25 else "PARTIAL"))
    else:
        out["D1_label"] = out["D4_label"] = "SUPPRESSED (gate failure)"
    out["base_attention_mass_by_category"] = {c: [float(x) for x in np.median(z["attn_cat"][:, :, ci], 0)] for ci, c in enumerate(cats)}
    json.dump(out, open(f"{run}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
