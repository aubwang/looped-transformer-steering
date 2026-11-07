"""Truth stage 2: steering along the truth direction (intervene_app.py `truthsteer`). Prespecified before the run.
Usage: python -I huginn_addition/analyze_truthsteer.py results/<tsteer_run>

Items: 60 judgment-format statements (15 held-out units per dataset, both truth values), direction = the item's
fold-specific B_dot mass-mean truth direction (fit without the item's unit, loop 64), dose 1 = training class-mean
distance. Intervention at loop 64; CONTEXT edits active for loops 1..24 after it and removed afterwards (state not reset).
Judgment logit difference ld = log p(' True') - log p(' False') at the answer position.
PRIMARY ENDPOINT E(t): movement of ld toward the item's OPPOSITE label relative to sham at the same offset, under the
toward-opposite condition (sign -1 for true items, +1 for false items): E = -(ld - ld_sham) for true items, +(ld - ld_sham)
for false items. E_same(t): movement toward the item's own label under the other sign (saturation check).
Random control: |ld_rand - ld_sham| (absolute deviation; conservative).
Inference: medians over items; unit-cluster bootstrap 95% CIs (resample units within dataset, 5000).
PREDICTIONS:
  P1 one-shot toward opposite is ERASED: E(1) > 0 (CI) with median E(1) >= 0.5 (minimum meaningful effect in logit
     units; otherwise erasure is INCONCLUSIVE), and |median E(24)| <= 0.1 median E(1) (aggregate, not equivalence).
  P2 context toward opposite is PERSISTENT while applied: E(24) CI > 0 and E(24) >= 0.5 E(8).
  P3 SPECIFICITY and DOSE: paired [E_context(24) - |random context deviation|(24)] CI > 0; paired [E(dose 1) - E(dose 0.5)]
     at 24, CI > 0.
  P4 RECOVERY after removal: requires median E_context(24) >= 0.5; recovered if |median E(48)| <= 0.1 median E(24);
     median |E(48)| also reported. Failure means a residual effect after 24 removal loops (not established hysteresis).
  P2's retention clause applies only if median E(8) > 0 (else P2 fails). Bootstrap CIs are cached (one per quantity) and
  conditional on the fitted directions and the single random direction per item.
P5 (answer-site run, prespecified for tsteer_ans): MEANINGFUL BIDIRECTIONAL movement = median context E(24) >= 0.5 in
   BOTH truth strata (true items toward False AND false items toward True), each with unit-bootstrap CI > 0; immediate
   coda effects (offset 0, one-shot) are reported separately from effects surviving further loops.
Also: E at offset 0 (one-shot, before any further loop: readout through the coda only); judgment flips at 24 (sham sign
= own label -> condition sign = opposite label), by dataset x truth stratum (baseline is False-biased); results by dataset.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main(run):
    z = np.load(f"{run}/truthsteer.npz", allow_pickle=True)
    conds = [json.loads(c)[0] for c in z["conds"]]; RO = list(z["ro"]); ld = z["ld"].astype(np.float64)
    y = z["truth"]; ds = z["dataset"]; units = z["unit"]
    n = len(y)
    C = {c: i for i, c in enumerate(conds)}
    sham = ld[:, C["sham"]]
    sgn_opp = np.where(y == 1, -1.0, 1.0)  # direction of the opposite label in ld

    def pick(base):  # per item, the toward-opposite / toward-same condition of a family
        opp = np.where(y[:, None] == 1, ld[:, C[base + "-"]], ld[:, C[base + "+"]])
        same = np.where(y[:, None] == 1, ld[:, C[base + "+"]], ld[:, C[base + "-"]])
        return opp, same
    E = {}
    for fam in ("oneshot", "context"):
        opp, same = pick(fam)
        E[fam] = sgn_opp[:, None] * (opp - sham)
        E[fam + "_same"] = -sgn_opp[:, None] * (same - sham)
    opp_h = np.where(y[:, None] == 1, ld[:, C["context-half"]], ld[:, C["context+half"]])
    E["context_half"] = sgn_opp[:, None] * (opp_h - sham)
    E["context_rand_abs"] = np.abs(ld[:, C["context_rand"]] - sham)
    E["oneshot_rand_abs"] = np.abs(ld[:, C["oneshot_rand"]] - sham)
    rng = np.random.default_rng(0)
    ub = {d: sorted(set(units[ds == d])) for d in ("cities", "compare")}
    idx_by = {u: np.where(units == u)[0] for u in units}

    cache = {}

    def ci(v, key=None):
        if key is not None and key in cache:
            return cache[key]
        b = []
        for _ in range(5000):
            ix = np.concatenate([idx_by[u] for d in ub for u in rng.choice(ub[d], len(ub[d]))])
            b.append(np.median(v[ix]))
        res_ = [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
        if key is not None:
            cache[key] = res_
        return res_
    t = {k: RO.index(k) for k in RO}
    med = {k: {int(o): float(np.median(v[:, t[o]])) for o in RO} for k, v in E.items()}
    e1, e8, e24, e48 = (E["context"][:, t[x]] for x in (1, 8, 24, 48))
    o1, o24 = E["oneshot"][:, t[1]], E["oneshot"][:, t[24]]
    out = dict(median_E_by_offset=med, e_resid_max=float(np.nanmax(z["e_resid"])))
    m1 = float(np.median(o1))
    out["P1"] = dict(E1=m1, E1_ci=ci(o1, "o1"), E24=float(np.median(o24)), E0=float(np.median(E["oneshot"][:, t[0]])),
                     status=("INCONCLUSIVE (E1 < 0.5)" if m1 < 0.5 else ("HOLDS" if ci(o1, "o1")[0] > 0 and abs(np.median(o24)) <= 0.1 * m1 else "FAILS")))
    m8, m24 = float(np.median(e8)), float(np.median(e24))
    out["P2"] = dict(E8=m8, E24=m24, E24_ci=ci(e24, "e24"), holds=bool(ci(e24, "e24")[0] > 0 and m8 > 0 and m24 >= 0.5 * m8))
    spec = e24 - E["context_rand_abs"][:, t[24]]; dose = e24 - E["context_half"][:, t[24]]
    out["P3"] = dict(spec_median=float(np.median(spec)), spec_ci=ci(spec, "spec"), dose_median=float(np.median(dose)), dose_ci=ci(dose, "dose"),
                     holds=bool(ci(spec, "spec")[0] > 0 and ci(dose, "dose")[0] > 0))
    out["P4"] = dict(E24=m24, E48=float(np.median(e48)), E48_ci=ci(e48, "e48"), median_abs_E48=float(np.median(np.abs(e48))),
                     status=("INCONCLUSIVE (E24 < 0.5)" if m24 < 0.5 else ("RECOVERED" if abs(np.median(e48)) <= 0.1 * m24 else "RESIDUAL EFFECT")))
    flips = {}
    for d in ("cities", "compare"):
        for tv in (1, 0):
            m = (ds == d) & (y == tv)
            own = np.where(tv == 1, 1.0, -1.0)
            sham_own = np.sign(sham[m, t[24]]) == own
            opp = np.where(tv == 1, ld[m, C["context-"], t[24]], ld[m, C["context+"], t[24]])
            flips[f"{d}_{'true' if tv else 'false'}"] = dict(n=int(m.sum()), sham_says_own=int(sham_own.sum()),
                                                            flipped_to_opposite=int((sham_own & (np.sign(opp) == -own)).sum()),
                                                            median_E24=float(np.median(E["context"][m, t[24]])))
    out["flips_context_t24"] = flips
    p5 = {}
    for tv, name in ((1, "true_items_toward_false"), (0, "false_items_toward_true")):
        m = y == tv
        v = E["context"][m, t[24]]
        rng2 = np.random.default_rng(7); uu_ = sorted(set(units[m])); ib = {u: np.where(units[m] == u)[0] for u in uu_}
        bs = [np.median(v[np.concatenate([ib[u] for u in rng2.choice(uu_, len(uu_))])]) for _ in range(5000)]
        p5[name] = dict(median_E24=float(np.median(v)), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                        oneshot_E0=float(np.median(E["oneshot"][m, t[0]])), oneshot_E24=float(np.median(E["oneshot"][m, t[24]])))
    p5["bidirectional_meaningful"] = bool(all(p5[k]["median_E24"] >= 0.5 and p5[k]["ci"][0] > 0 for k in ("true_items_toward_false", "false_items_toward_true")))
    out["P5_bidirectional"] = p5
    out["by_dataset_median_E24"] = {d: dict(context=float(np.median(e24[ds == d])), oneshot=float(np.median(o24[ds == d])),
                                            context_after_removal_48=float(np.median(e48[ds == d]))) for d in ("cities", "compare")}
    json.dump(out, open(f"{run}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
