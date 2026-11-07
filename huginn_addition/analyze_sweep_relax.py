"""Threshold sweep of source under-relaxation (intervene_app.py `relaxsweep`). Prespecified before the run.
Usage: python -I huginn_addition/analyze_sweep_relax.py results/<rsweep_run>

Predictions are the alpha* values saved by results/relax-20261105-200208 (not recomputed). Relaxing the source with
a = r alpha* maps the flip multiplier to 1 - 2r: r < 1 predicted to DECAY (multiplier -0.92, -0.96, -0.98 for r = 0.96,
0.98, 0.99), r > 1 predicted to PERSIST (-1.02, -1.04, -1.08 for r = 1.01, 1.02, 1.04).
Per condition (source drift D_t at p0, T = 192 loops):
  late rate rho_late = exp(slope of log D_t over loops 128..191); D_late = mean D over the last 32 loops.
  DECAY   if D_late < 1e-6 or rho_late <= 0.995
  PERSIST if D_late >= 1e-5 and |rho_late - 1| < 0.002 and lag-2 closure < 0.05
  else UNCLEAR.
PRIMARY: per prompt, all six conditions on the predicted side; reported for the 24 prompts resolved in the relax run
and separately for the 2 unresolved ones (uid 20041, 20117).
THRESHOLD ESTIMATES per prompt (descriptive, two independent sides):
  below: decay rate rho(r) fitted log-linearly over loops 32..160 (points above 1e-8) for r in {0.96, 0.98, 0.99};
         predicted rho = 2r - 1; linear fit rho vs r, empirical threshold r_below = r at rho = 1; slope vs predicted 2.
  above: squared late drift D_late^2 (proportional to cycle amplitude^2) for r in {1.01, 1.02, 1.04}; supercritical
         normal form predicts D^2 linear in (r - 1); linear fit, empirical threshold r_above = zero crossing.
  Report median / IQR of r_below and r_above (1 = prediction exact) and the fraction within +-2%.
Revisions after Astra's review (before data): a missing late-rate fit is UNCLEAR unless D_late < 1e-6; every prompt must
have exactly the six conditions and T = 192 loops; the primary denominator stays 24 (incomplete prompts count as not
confirmed and are listed). Fit eligibility: below-side estimate only if all three decay rates exist, slope > 0 and the
crossing lies in [0.9, 1.1]; above-side estimate uses only PERSIST conditions (>= 2), slope > 0, crossing in [0.9, 1.1];
otherwise 'unavailable'. Both are descriptive three-point extrapolations sharing predictions and starting states (not
independent), not exact threshold localisation.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def fit_rate(y, t):
    t = t[: len(y)]
    m = y > 1e-8
    if m.sum() < 5:
        return None
    return float(np.exp(np.polyfit(t[m], np.log(y[m]), 1)[0]))


def main(run):
    z = np.load(f"{run}/sweep.npz", allow_pickle=True)
    P = json.load(open(f"{run}/problems.json"))
    meta = json.load(open(f"{run}/meta.json"))
    cr, rr, dn = z["cond_row"], z["cond_r"], z["done"]
    D, l2 = z["drift_p0"], z["lag2_p0"]
    T = D.shape[1]
    assert T == 192, T
    assert all((cr == j).sum() == 6 for j in range(len(P))), "expected six conditions per prompt"
    tl = np.arange(128, T); tb = np.arange(32, 161)
    rows = []
    for j, p in enumerate(P):
        ks = np.where(cr == j)[0]
        if not dn[ks].all():
            rows.append(dict(uid=p["uid"], status="INCOMPLETE", relax_all_ok=p["relax_all_ok"], all_ok=False)); continue
        conds = {}
        for k in ks:
            r = float(rr[k]); y = D[k]
            rho_late = fit_rate(y[128:], tl)
            dl = float(y[-32:].mean())
            if dl < 1e-6 or (rho_late is not None and rho_late <= 0.995):
                out = "DECAY"
            elif rho_late is not None and dl >= 1e-5 and abs(rho_late - 1) < 0.002 and l2[k] < 0.05:
                out = "PERSIST"
            else:
                out = "UNCLEAR"
            pred = "DECAY" if r < 1 else "PERSIST"
            conds[r] = dict(outcome=out, predicted=pred, ok=out == pred, rho_late=None if rho_late is None else round(rho_late, 5), D_late=dl,
                            rho_32_160=fit_rate(y[32:161], tb), lag2=float(l2[k]))
        below = [(r, c["rho_32_160"]) for r, c in conds.items() if r < 1 and c["rho_32_160"] is not None]
        above = [(r, c["D_late"] ** 2) for r, c in conds.items() if r > 1 and c["outcome"] == "PERSIST"]
        r_below = slope_b = r_above = None
        if len(below) == 3:
            sl, ic = np.polyfit([b[0] for b in below], [b[1] for b in below], 1)
            if sl > 0 and 0.9 <= (1 - ic) / sl <= 1.1:
                r_below, slope_b = float((1 - ic) / sl), float(sl)
        if len(above) >= 2:
            sl, ic = np.polyfit([a[0] for a in above], [a[1] for a in above], 1)
            if sl > 0 and 0.9 <= -ic / sl <= 1.1:
                r_above = float(-ic / sl)
        rows.append(dict(uid=p["uid"], relax_all_ok=p["relax_all_ok"], lam=p["lam_pred"], alpha_star=p["alpha_star_pred"],
                         all_ok=all(c["ok"] for c in conds.values()), conds={str(r): c for r, c in conds.items()},
                         r_below=r_below, slope_below=slope_b, r_above=r_above))
    res = [r for r in rows if r.get("relax_all_ok") and r.get("status") != "INCOMPLETE"]
    inc = [r for r in rows if r.get("status") == "INCOMPLETE"]
    unres = [r for r in rows if r.get("relax_all_ok") is False and r.get("status") != "INCOMPLETE"]

    def q(v):
        v = [x for x in v if x is not None]
        return dict(n=len(v), median=float(np.median(v)), q25=float(np.quantile(v, .25)), q75=float(np.quantile(v, .75)),
                    within_2pct=int(sum(abs(x - 1) <= 0.02 for x in v)), min=float(min(v)), max=float(max(v))) if v else None
    summ = dict(skipped=meta["skipped"], n=len(rows), n_incomplete=sum(r.get("status") == "INCOMPLETE" for r in rows),
                primary_resolved=dict(denominator=sum(1 for r in rows if r.get("relax_all_ok")), all_six_on_predicted_side=sum(r["all_ok"] for r in res),
                                      incomplete=[r["uid"] for r in inc]),
                threshold_estimates_available=dict(below=sum(r["r_below"] is not None for r in res), above=sum(r["r_above"] is not None for r in res)),
                unresolved=[(r["uid"], r.get("all_ok"), {k: c["outcome"] for k, c in r.get("conds", {}).items()}) for r in unres],
                by_r={str(rm): dict(ok=sum(r["conds"][str(rm)]["ok"] for r in res),
                                    outcomes={o: sum(r["conds"][str(rm)]["outcome"] == o for r in res) for o in ("DECAY", "PERSIST", "UNCLEAR")})
                      for rm in sorted({float(x) for x in rr})},
                r_below=q([r["r_below"] for r in res]), slope_below=q([r["slope_below"] for r in res]), r_above=q([r["r_above"] for r in res]),
                failures=[(r["uid"], {k: (c["outcome"], c["rho_late"], "%.1e" % c["D_late"]) for k, c in r["conds"].items() if not c["ok"]})
                          for r in res if not r["all_ok"]])
    json.dump(dict(summary=summ, rows=rows), open(f"{run}/analysis_summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
