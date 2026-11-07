"""Held-out evaluation of resolvent-optimal steering (intervene_app.py `dstar_eval`). Prespecified before the run.
Usage: python -I huginn_addition/analyze_dstar.py results/<dseval_run>
60 held-out judgment items (30 units, both truth values; training units disjoint). Push norm 12.5 for every direction.
E(t) = movement of R = log p(True) - log p(False) toward the item's OPPOSITE label vs sham under the toward-opposite sign
(true items: '-' push, E = -(R - R_sham); false items: '+' push, E = +(R - R_sham)). Primary offset t = 48.
Unit-cluster bootstrap (5000 resamples of units; both statements of a unit together).
P1 (primary): population d* beats the population mass-mean (probe) direction: paired median difference E48(dstar_pop) -
   E48(mm_pop) with CI > 0 (ADDITIVE SUPERIORITY). A '2x' statement additionally requires median E48(dstar_pop) > 0,
   median E48(mm_pop) > 0 and ratio >= 2 (descriptive; the CI concerns the paired difference only).
GATE (run 1 diagnostics, before run 2): median adjoint relative error < 0.05 and median absolute adjoint error reported;
   d* is a frozen-linearisation optimum at loop 64 (x* drift reported), not an exact nonlinear optimum.
Bootstrap resamples units WITHIN dataset (balanced design), also for the per-stratum P2 intervals.
P2 meaningful bidirectional control by population d*: median E48 >= 0.5 in BOTH truth strata, each CI > 0.
P3 accounting for recurrence helps: paired median difference E48(dstar_pop) - E48(g_pop), CI > 0 (descriptive if not).
P4 adaptive (label-free item-specific d*) vs population d*: paired difference (descriptive).
Flips at 48: items whose sham judgment sign equals their own label and whose pushed judgment sign equals the opposite label,
per stratum and condition. Collateral: median change in the True+False mass at 48 vs sham. Random push: |R - R_sham|.
"""
import json
import sys

import numpy as np


def main(run):
    z = np.load(f"{run}/dstar_eval.npz", allow_pickle=True)
    items = json.load(open(f"{run}/items.json"))
    names = [str(x) for x in z["names"]]; RO = list(z["ro"]); R = z["R"].astype(np.float64); M = z["mass"].astype(np.float64)
    y = np.array([it["truth"] for it in items]); units = np.array([it["unit"] for it in items])
    t48 = RO.index(48); c = {nm: i for i, nm in enumerate(names)}
    sham = R[:, c["sham"]]
    def E(fam):
        opp = np.where(y[:, None] == 1, R[:, c[fam + "-"]], R[:, c[fam + "+"]])
        return np.where(y[:, None] == 1, -1.0, 1.0) * (opp - sham)
    Es = {fam: E(fam) for fam in ("dstar_pop", "g_pop", "mm_pop", "dstar_item")}
    rng = np.random.default_rng(0)
    dsn = np.array([it["dataset"] for it in items])
    def draw(mask):
        idx = []
        for d_ in ("cities", "compare"):
            uu_ = np.unique(units[mask & (dsn == d_)])
            for u in rng.choice(uu_, len(uu_)):
                idx += list(np.where(mask & (units == u))[0])
        return np.array(idx)
    cache = {}
    allm = np.ones(len(y), bool)
    def ci(key, v, mask=allm):
        if key not in cache:
            bs = [np.median(v[draw(mask)]) for _ in range(5000)]
            cache[key] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        return cache[key]
    med = {fam: {int(t): float(np.median(v[:, i])) for i, t in enumerate(RO)} for fam, v in Es.items()}
    d48, m48, g48, i48 = (Es[f][:, t48] for f in ("dstar_pop", "mm_pop", "g_pop", "dstar_item"))
    out = dict(median_E_by_offset=med)
    md, mm_ = float(np.median(d48)), float(np.median(m48))
    out["P1"] = dict(dstar=md, mm=mm_, paired_diff=float(np.median(d48 - m48)), ci=ci("p1", d48 - m48),
                     additive_superiority=bool(ci("p1", d48 - m48)[0] > 0),
                     twofold_descriptive=bool(md > 0 and mm_ > 0 and md >= 2 * mm_))
    p2 = {}
    for tv, nm in ((1, "true_toward_false"), (0, "false_toward_true")):
        m_ = y == tv
        p2[nm] = dict(median=float(np.median(d48[m_])), ci=ci(f"p2{tv}", d48, m_) if False else None)
        bs = [np.median(d48[[i for i in draw(m_)]]) for _ in range(5000)]
        p2[nm]["ci"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    p2["holds"] = bool(all(p2[k]["median"] >= 0.5 and p2[k]["ci"][0] > 0 for k in ("true_toward_false", "false_toward_true")))
    out["P2"] = p2
    out["P3"] = dict(paired_diff=float(np.median(d48 - g48)), ci=ci("p3", d48 - g48), holds=bool(ci("p3", d48 - g48)[0] > 0))
    out["P4"] = dict(paired_diff=float(np.median(i48 - d48)), ci=ci("p4", i48 - d48), item=float(np.median(i48)))
    flips = {}
    for fam in ("dstar_pop", "g_pop", "mm_pop", "dstar_item"):
        for tv, nm in ((1, "true"), (0, "false")):
            m = y == tv; own = 1 if tv else -1
            pushed = R[m, c[fam + ("-" if tv else "+")], t48]
            sham_own = np.sign(sham[m, t48]) == own
            flips[f"{fam}_{nm}"] = dict(sham_own=int(sham_own.sum()), flipped=int((sham_own & (np.sign(pushed) == -own)).sum()), n=int(m.sum()))
    out["flips_t48"] = flips
    out["collateral_TF_mass_change_t48"] = {fam: float(np.median(np.where(y == 1, M[:, c[fam + "-"], t48], M[:, c[fam + "+"], t48]) - M[:, c["sham"], t48]))
                                            for fam in ("dstar_pop", "g_pop", "mm_pop", "dstar_item")}
    out["random_abs_t48"] = float(np.median(np.abs(R[:, c["random+"], t48] - sham[:, t48])))
    json.dump(out, open(f"{run}/analysis_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"))
