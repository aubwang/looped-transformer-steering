"""One-shot state push vs context (E) edit (intervene_app.py `steer`). Prespecified before the run.
Usage: python -I huginn_addition/analyze_steer.py results/<steer_run>
50 held-out digit prompts (5 per answer digit U), answer position q after 96 warm-up loops. Directions: adjusted
one-notch chord C[U+1] - C[U] (fit on held-out cells) or a norm-matched random direction; scales 1 and 2.
TARGET GAIN G(t) = [log p(U+1) - log p(U)] - the same for sham at the same t (digit-restricted log-probs at q).
Readouts at t = 0 (one-shot: right after the push, before any further loop; context: identical to sham) and after
1, 2, 4, 8, 16, 32, 48 loops.
PREDICTIONS (per scale; prompt statistics; bootstrap 95% CIs resample (a, b) CELLS within answer strata, 5000
resamples; each CI computed once and reused):
  P1 one-shot chord UNDONE: G(0) CI > 0, |median G(48)| <= 0.1 median G(0) (two-sided), and the final state returns:
     median ||x_48 - x_sham,48|| / ||d_s|| <= 0.1.
  P2 context chord PERSISTENT: median G(48) CI > 0; if median G(8) <= 0, P2 fails; else median G(48) >= 0.5 median G(8).
     Claim limited to "persistent through 48 loops under continued forcing" unless the late consecutive-loop drift at q
     (loops 47 -> 48) is < 1e-4 and lag-2 closure does not indicate a cycle (closure > 0.5), reported per condition.
  P3 specificity: MEDIAN PAIRED DIFFERENCE context chord - context random at t = 48, CI excludes 0 (> 0); sham-to-target
     transitions (sham argmax == U at 48 and condition argmax == U+1) counted for chord vs random; baseline accuracy.
  P4 matched input: median paired difference context chord - one-shot chord at t = 48, CI > 0.
  CHECK: one-shot and context conditions with the same direction must agree at t = 1 (equivalence of the input change).
"""
import json
import sys

import numpy as np


def main(run):
    z = np.load(f"{run}/steer.npz", allow_pickle=True)
    conds = [str(c) for c in z["cond"]]; RO = list(z["ro_t"]); lp = z["lp"]; ans = z["ans"]; am = z["am"]; dig = z["dig_tokens"]
    n = len(ans)
    G = np.stack([(lp[:, c, :, :][np.arange(n), :, (ans + 1) % 10] - lp[:, c, :, :][np.arange(n), :, ans])
                  - (lp[:, 0][np.arange(n), :, (ans + 1) % 10] - lp[:, 0][np.arange(n), :, ans]) for c in range(len(conds))], 1)  # (n, C, T)
    rng = np.random.default_rng(0)
    P = {p["uid"]: p for p in json.load(open(f"{run}/problems.json"))}
    cell = np.array([P[int(u)]["start"] * 10 + P[int(u)]["offset"] for u in z["uid"]])
    strata = {u: sorted(set(cell[ans == u])) for u in range(10)}
    cache = {}

    def ci(key, v):
        if key in cache:
            return cache[key]
        b = []
        for _ in range(5000):
            idx = []
            for u in range(10):
                cs = rng.choice(strata[u], len(strata[u]))
                for c_ in cs:
                    idx += list(np.where(cell == c_)[0])
            b.append(np.median(v[idx]))
        cache[key] = [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
        return cache[key]
    t0, t1, t8, t48 = RO.index(0), RO.index(1), RO.index(8), RO.index(48)
    out = {"median_G_by_t": {c: {int(t): float(np.median(G[:, ci_, ti])) for ti, t in enumerate(RO)} for ci_, c in enumerate(conds)}}
    sham_ok = am[:, 0, t48] == dig[ans]
    out["baseline_accuracy_sham_t48"] = float(sham_ok.mean())
    res = {}
    for sc in sorted({c.split("|")[2] for c in conds if c != "sham|none|0.0"}):
        oc, orr, cc, cr = (conds.index(f"oneshot|chord|{sc}"), conds.index(f"oneshot|random|{sc}"),
                           conds.index(f"context|chord|{sc}"), conds.index(f"context|random|{sc}"))
        g_o0, g_o48 = np.median(G[:, oc, t0]), np.median(G[:, oc, t48])
        g_c8, g_c48 = np.median(G[:, cc, t8]), np.median(G[:, cc, t48])
        ret = float(np.median(z["disp"][:, oc] / z["dnorm"][:, oc]))
        tr_c = int((sham_ok & (am[:, cc, t48] == dig[(ans + 1) % 10])).sum()); tr_r = int((sham_ok & (am[:, cr, t48] == dig[(ans + 1) % 10])).sum())
        res[sc] = dict(
            P1=dict(G0=float(g_o0), G0_ci=ci(f"o0{sc}", G[:, oc, t0]), G48=float(g_o48), return_ratio=ret,
                    holds=bool(ci(f"o0{sc}", G[:, oc, t0])[0] > 0 and abs(g_o48) <= 0.1 * g_o0 and ret <= 0.1)),
            P2=dict(G8=float(g_c8), G48=float(g_c48), G48_ci=ci(f"c48{sc}", G[:, cc, t48]),
                    holds=bool(ci(f"c48{sc}", G[:, cc, t48])[0] > 0 and g_c8 > 0 and g_c48 >= 0.5 * g_c8),
                    late_drift_median=float(np.median(z["late_drift"][:, cc])), lag2_closure_median=float(np.median(z["lag2_closure"][:, cc])),
                    settled=bool(np.median(z["late_drift"][:, cc]) < 1e-4)),
            P3=dict(median_paired_diff=float(np.median(G[:, cc, t48] - G[:, cr, t48])), ci=ci(f"p3{sc}", G[:, cc, t48] - G[:, cr, t48]),
                    sham_to_target_chord=tr_c, sham_to_target_random=tr_r, n_sham_correct=int(sham_ok.sum()),
                    holds=bool(ci(f"p3{sc}", G[:, cc, t48] - G[:, cr, t48])[0] > 0)),
            P4=dict(median_paired_diff=float(np.median(G[:, cc, t48] - G[:, oc, t48])), ci=ci(f"p4{sc}", G[:, cc, t48] - G[:, oc, t48]),
                    holds=bool(ci(f"p4{sc}", G[:, cc, t48] - G[:, oc, t48])[0] > 0)),
            CHECK_t1_equivalence_max_abs_diff=float(np.abs(G[:, oc, t1] - G[:, cc, t1]).max()),
            displacement_over_push=dict(context_chord=float(np.median(z["disp"][:, cc] / z["dnorm"][:, cc])), oneshot_chord=ret,
                                        context_chord_along=float(np.median(z["along"][:, cc] / z["dnorm"][:, cc]))),
            wrap_9_to_0_context_G48=[float(x) for x in G[ans == 9, cc, t48]],
            other_digit_argmax_t48_context_chord=int(((am[:, cc, t48] != dig[(ans + 1) % 10]) & (am[:, cc, t48] != dig[ans])).sum()))
    out["by_scale"] = res
    out["e_resid_max"] = float(np.nanmax(z["e_resid"]))
    json.dump(out, open(f"{run}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
