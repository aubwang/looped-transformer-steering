"""Descriptive tables for part 2 (docs/oversight_hard.md, 'Frozen before the screen'), from saved artifacts only.
Leaves the frozen analysis (analyze_obs.py) untouched; requested by Astra's results review.

Usage: python -I report_hard.py <screen run dir> <hard run dir>
Writes <hard run dir>/descriptive.json and prints the tables.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_obs as an  # noqa: E402

THR = 0.9


def main(sdir, hdir):
    items = json.load(open(os.path.join(hdir, "prompts.json")))
    data, _ = an.load(hdir)
    out, lines = {}, []
    # ---- composition
    fam = [it["fam"] for it in items]; strat = [it["stratum"] for it in items]
    lens = {k: [len(x) for x in data[k][2]["ids"]] for k in data}
    assert all(lens[k] == lens["D"] for k in lens)
    out["families"] = {f: fam.count(f) for f in sorted(set(fam))}
    out["lengths"] = {str(L): lens["D"].count(L) for L in sorted(set(lens["D"]))}
    out["by_stratum"] = {st: {f: sum(1 for a, b in zip(strat, fam) if a == st and b == f) for f in sorted(set(fam))} for st in "WU"}
    lines.append(f"families {out['families']}; lengths {out['lengths']}; by stratum {out['by_stratum']}")
    # ---- triggers: W = models wrong in the screen; U = models below the threshold
    trig = []
    for it in items:
        sc = it["screen"]
        if it["stratum"] == "W":
            t = [k for k in ("L4", "L32", "D") if not sc[k]["correct"]]
        else:
            t = [k for k in ("L4", "L32", "D") if sc[k]["p"] < THR]
        trig.append(dict(uid=it["uid"], stratum=it["stratum"], fam=it["fam"], query=it["prompt"].splitlines()[-1].strip(),
                         ans=it["ans"], triggers=t, screen_p={k: round(sc[k]["p"], 4) for k in sc}))
    out["triggers"] = trig
    lines.append("\nuid stratum fam query ans triggers screen_p")
    lines += [f"{t['uid']:3d} {t['stratum']} {t['fam']:4s} {t['query']:16s} {t['ans']:4d} {','.join(t['triggers']):12s} {t['screen_p']}"
              for t in trig]
    # ---- screen -> run transitions at every screened depth
    cid = [it["cid"] for it in items]
    trans = {}
    for k in ("L4", "L32", "D"):
        sz = np.load(os.path.join(sdir, k, "screen.npz")); z = data[k][0]
        pos = [int(np.where(sz["cid"] == c)[0][0]) for c in cid]
        for di, dep in enumerate(sz["depths"]):
            p_s = sz["p"][pos, di]; t_s = sz["top1"][pos, di]
            if k == "D":
                p_r = np.exp(z["c_base_logp"][:, 0]); t_r = z["c_base_top1"][:, 0]
            else:
                p_r = np.exp(z["a0_logp"][:, int(dep)]); t_r = z["a0_top1"][:, int(dep)]
            trans[f"{k} r={int(dep)}"] = dict(max_abs_dp=float(np.max(np.abs(p_r - p_s))), same_token=int((t_r == t_s).sum()),
                                              same_correct=int(((t_r == z["ans"]) == (t_s == sz["ans"][pos])).sum()))
    out["screen_to_run"] = trans
    lines.append("\nscreen -> run: " + "; ".join(f"{k}: max|dp| {v['max_abs_dp']:.2e}, same token {v['same_token']}/16, "
                                                f"same correctness {v['same_correct']}/16" for k, v in trans.items()))
    # ---- primary exclusions
    S = json.load(open(os.path.join(hdir, "summary.json")))
    c = S["A_compare"]["L4/L32 r64 all"]
    out["primary_excluded_uids"] = [i for i, v in enumerate(c["included"]) if not v]
    lines.append(f"\nprimary (r=64) excluded uids: {out['primary_excluded_uids']} -> {[(items[i]['stratum'], items[i]['fam']) for i in out['primary_excluded_uids']]}")
    # ---- Part C: push size at matched KL, per-injection energy, adapter energy, error tails, on common directions
    PC = {k: an.part_c(data[k][0], data[k][2]) for k in data}
    nd = data["D"][0]["c_kl"].shape[2]
    groups = dict(random=list(range(nd - 2)), gradient=[nd - 2, nd - 1])
    diag = {}
    for a in ("L4", "L32"):
        rho_a = float(data[a][0]["rho"]); rho_d = float(data["D"][0]["rho"])
        for r, ca in PC[a].items():
            cd = PC["D"][0]
            for g, dirs in groups.items():
                for ti, t in enumerate(an.KL_TARGETS):
                    need = 2
                    pa, pd, ea, ed, eaa, erra, errd = ([] for _ in range(7))
                    for i in range(16):
                        com = [j for j in dirs if np.isfinite(ca["err"][i, j, ti]) and np.isfinite(cd["err"][i, j, ti])]
                        if len(com) < need:
                            continue
                        size_a = np.exp((ca["logE"][i, com, ti] - np.log(r)) / 2) / rho_a     # ||eps u|| / rho
                        size_d = np.exp(cd["logE"][i, com, ti] / 2) / rho_d
                        pa.append(np.median(size_a)); pd.append(np.median(size_d))
                        ea.append(np.median(ca["logE"][i, com, ti] - np.log(r))); ed.append(np.median(cd["logE"][i, com, ti]))
                        if np.isfinite(ca["logE_adapter"][i, com, ti]).all():
                            eaa.append(np.median(ca["logE_adapter"][i, com, ti]))
                        erra.append(np.median(ca["err"][i, com, ti])); errd.append(np.median(cd["err"][i, com, ti]))
                    if not pa:
                        continue
                    diag[f"{a}(r={r}) {g} KL={t:g}"] = dict(
                        n=len(pa), push_size_looped=float(np.median(pa)), push_size_D=float(np.median(pd)),
                        log_per_injection_energy_ratio=float(np.median(np.array(ea) - np.array(ed))),
                        log_cumulative_energy_ratio=float(np.median(np.array(ea) - np.array(ed)) + np.log(r)),
                        log_adapter_energy_looped=float(np.median(eaa)) if eaa else None,
                        err_median_looped=float(np.median(erra)), err_median_D=float(np.median(errd)),
                        err_max_looped=float(np.max(erra)), err_max_D=float(np.max(errd)))
    out["part_c_diagnostics"] = diag
    lines.append("\nPart C diagnostics (per-prompt medians over common directions; then median / max over prompts):")
    for k, v in diag.items():
        if ("KL=0.01" in k or "KL=0.1" in k or "KL=0.001" in k):
            lines.append(f"  {k:28s} n{v['n']:2d} push {v['push_size_looped']:.3f} vs D {v['push_size_D']:.3f} | log E/inj "
                         f"{v['log_per_injection_energy_ratio']:+.2f} cum {v['log_cumulative_energy_ratio']:+.2f} | err med "
                         f"{v['err_median_looped']:.3f} vs {v['err_median_D']:.3f}, max {v['err_max_looped']:.2f} vs {v['err_max_D']:.2f}")
    out["flops_per_prompt_G"] = {f"{k} r={r}": round(v["per_prompt"] / 1e9, 1) for k in data for r, v in
                                 S[k]["flops"].items()}
    lines.append(f"\nforward matmul GFLOPs per prompt: {out['flops_per_prompt_G']}")
    # ---- entropy (screen) for the selected prompts, evaluation depth
    ent = {}
    for k in ("L4", "L32", "D"):
        sz = np.load(os.path.join(sdir, k, "screen.npz")); pos = [int(np.where(sz["cid"] == c)[0][0]) for c in cid]
        col = 0 if k == "D" else list(sz["depths"]).index(64)
        ent[k] = float(np.median(sz["ent"][pos, col]))
    out["median_entropy_nats"] = ent
    lines.append(f"median answer-position entropy (nats, screen, selected prompts): {ent}")
    with open(os.path.join(hdir, "descriptive.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
