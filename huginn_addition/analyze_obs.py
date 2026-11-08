"""Pre-registered summaries of the oversight pilot (docs/oversight_pilot.md, Draft 2 with the frozen analysis rules).
Reads <run>/<model>/{obs.npz, meta.json, manifest.json} as written by obs_app.py; writes <run>/summary.json and prints a
plain report.

Usage: python -I analyze_obs.py <run dir> [--ref <run dir that holds D>]
  --ref lets run2 (L8, L16) be compared with run1's D. Prompt ids, answers and the measurement settings must agree.

Frozen rules (Astra review of Draft 2; all are in the design document):
- Ratios between looped models are oriented by training depth: numerator = smaller M (the primary is L4 / L32).
- A is valid only if every F_k it averages (k = 1..32, all 4 directions) clears 100 x its floor. kappa = F_8 / F_1 needs
  positive values that clear the floor in every direction; per prompt it is the geometric mean over directions. State
  persistence uses (dx_8 / dx_1)^2.
- Part C: the base finite difference (dose 1e-3) must clear 100 x floor, else the direction is unusable. Brackets: the
  first adjacent pair of doses, scanning upward from 3e-3 to 3, with both doses valid, KL increasing and the target
  inside; log error and log dose are interpolated linearly in log KL; no extrapolation.
- Cross-model Part C comparisons use only directions matched in both models (same direction ids): at least 2 of the 4
  random directions, and both gradient signs. Per prompt: median of the interpolated errors (not of their logs) per
  model, then the log ratio. Energies are compared on the same directions. All six KL targets are exploratory.
- At least 12 jointly valid prompts for any claim ("eligible_n12" means only that; it says nothing about the CI). The two smoke prompts (uids 0, 1) reappear in every run, so every
  comparison is also reported without them.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

KL_TARGETS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
MIN_PAIRS = 12
FLOOR_MULT = 100.0
B_BOOT = 10_000
MDEP = {"L4": 4, "L8": 8, "L16": 16, "L32": 32, "D": 0}
SMOKE_UIDS = (0, 1)
# Llama-3.2-1B shapes, used only if a manifest has no parameter counts (smoke runs): one transformer layer, the adapter
# (2d -> d) and the output head, in parameters (matmul weights only).
LAYER_P = 2048 * 2048 * 2 + 2048 * 512 * 2 + 3 * 2048 * 8192
ADAPTER_P = 2 * 2048 * 2048
HEAD_P = 128256 * 2048


def fl(v):
    return None if v is None or not np.isfinite(v) else float(v)


def load_one(p):
    z = np.load(os.path.join(p, "obs.npz"))
    meta = json.load(open(os.path.join(p, "meta.json"))) if os.path.exists(os.path.join(p, "meta.json")) else {}
    man = json.load(open(os.path.join(p, "manifest.json"))) if os.path.exists(os.path.join(p, "manifest.json")) else {}
    return {k: z[k] for k in z.files}, meta, man


def load(run, ref=None):
    out = {}
    for key in sorted(os.listdir(run)):
        if key in MDEP and os.path.isfile(os.path.join(run, key, "obs.npz")):
            out[key] = load_one(os.path.join(run, key))
    notes = []
    if ref and "D" not in out:
        out["D"] = load_one(os.path.join(ref, "D"))
        notes.append(f"D taken from {ref}")
    # every loaded collection must share prompts, answers and measurement settings, however it was assembled
    keys = sorted(out); m0 = out[keys[0]][2]
    for k in keys[1:]:
        man = out[k][2]
        assert man["ids"] == m0["ids"] and man["ans"] == m0["ans"], f"prompt mismatch between {k} and {keys[0]}"
        for c in ("eps", "n_rand", "seed", "prompt_seed", "dtype", "T", "K", "sig", "ndir_a", "n_starts"):
            assert man["cfg"].get(c) == m0["cfg"].get(c), f"setting {c} differs between {k} and {keys[0]}"
    hashes = {k: out[k][2]["cfg"].get("code_sha256") for k in keys}
    if len(set(hashes.values())) > 1:
        notes.append(f"code hashes differ across collections: {hashes}")
    return out, notes


def boot_median(x, seed=0):
    """Median of the per-prompt values x with a percentile bootstrap 95% CI (prompts resampled)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, median=None, ci95=None, n_pos=0, eligible_n12=False)
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, x.size, (B_BOOT, x.size))], axis=1)
    return dict(n=int(x.size), median=float(np.median(x)), ci95=[float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))],
                n_pos=int((x > 0).sum()), eligible_n12=bool(x.size >= MIN_PAIRS))


def keep_mask(n, exclude):
    m = np.ones(n, bool)
    m[[u for u in exclude if u < n]] = False
    return m


# ------------------------------------------------------------------------------------------------ Part A
def part_a(z, M):
    done = z["doneA"]; RA = list(z["RA"]); SIG = list(z["SIG"]); s0 = SIG.index(1e-3)
    K = z["a2_fish"].shape[-1] - 1
    out = {}
    for ri, r in enumerate(RA):
        F = z["a2_fish"][:, ri, s0]                                           # (n, dirs, K + 1), divided by sigma^2
        raw = F * SIG[s0] ** 2
        floor = np.maximum(z["a2_dup_fish"][:, ri], z["a2_floor_fish"][:, ri])[:, None, :]
        valid = np.isfinite(raw) & (raw >= FLOOR_MULT * floor) & done[:, None, None]
        lowres = ((z["a2_base_res"][:, ri] < 1e-3) & (z["a2_base_res_q"][:, ri] < 1e-3)
                  & (np.nanmax(z["a2_base_drift"][:, ri], axis=-1) < 1e-2) & (z["a2_base_kl"][:, ri, K] < 1e-3)) & done
        allv = valid[:, :, 1:].all(axis=(1, 2))
        A = np.where(allv, F[:, :, 1:].mean(axis=(1, 2)), np.nan)
        Aabs = A / z["a2_xr_norm"][:, ri] ** 2
        dx = z["a2_dx_all"][:, ri, s0]
        okk = (valid[:, :, 1] & valid[:, :, 8] & (F[:, :, 1] > 0) & (F[:, :, 8] > 0)).all(axis=1)
        okx = okk & ((dx[:, :, 1] > 0) & (dx[:, :, 8] > 0) & np.isfinite(dx[:, :, [1, 8]]).all(-1)).all(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            kap = np.where(okk, np.exp(np.log(F[:, :, 8] / F[:, :, 1]).mean(axis=1)), np.nan)
            kap_x = np.where(okx, np.exp(np.log((dx[:, :, 8] / dx[:, :, 1]) ** 2).mean(axis=1)), np.nan)
        curve = np.where(valid[:, :, 1:].all(axis=1), F[:, :, 1:].mean(axis=1), np.nan)   # (n, K)
        lin = z["a2_fish"][:, ri, 1, :, 1] / z["a2_fish"][:, ri, 0, :, 1]
        out[int(r)] = dict(done=done, lowres=lowres, valid_A=allv, A=A, Aabs=Aabs, kappa=kap, kappa_state=kap_x,
                           curve=curve, frac_valid=fl(valid[done][:, :, 1:].mean()) if done.any() else None,
                           lin_k1=lin[done])
    T = z["a0_top1"].shape[1] - 1
    acc = {int(t): fl((z["a0_top1"][done, t] == z["ans"][done]).mean()) if done.any() and t <= T else None
           for t in sorted({M, 32, 64})}
    return out, acc


def compare_A(pa, pb, ra, rb, exclude=()):
    """Per-prompt log(A_a / A_b) over jointly valid prompts (both low residual, both A valid and positive)."""
    a, b = pa[ra], pb[rb]
    ok = a["lowres"] & b["lowres"] & a["valid_A"] & b["valid_A"] & (a["A"] > 0) & (b["A"] > 0) & keep_mask(a["A"].size, exclude)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(ok, np.log(a["A"] / b["A"]), np.nan)
        lra = np.where(ok, np.log(a["Aabs"] / b["Aabs"]), np.nan)
        lk = np.log(a["curve"][ok] / b["curve"][ok])
    medk = np.nanmedian(lk, axis=0) if ok.any() else np.full(a["curve"].shape[1], np.nan)
    s = np.sign(medk[np.isfinite(medk)])
    km = keep_mask(a["A"].size, exclude)                       # counts refer to the requested subset only
    excluded = dict(not_lowres_a=int((km & a["done"] & ~a["lowres"]).sum()), not_lowres_b=int((km & b["done"] & ~b["lowres"]).sum()),
                    invalid_A_a=int((km & a["done"] & ~a["valid_A"]).sum()), invalid_A_b=int((km & b["done"] & ~b["valid_A"]).sum()))
    return dict(relative=boot_median(lr), absolute=boot_median(lra), included=ok.tolist(),
                log_ratio_relative=[fl(v) for v in lr], log_ratio_absolute=[fl(v) for v in lra], excluded=excluded,
                curve_logratio_median=[fl(v) for v in medk],
                curves_cross=bool(s.size > 1 and (s > 0).any() and (s < 0).any()))


# ------------------------------------------------------------------------------------------------ Part C
def part_c(z, man):
    """Direction-level interpolants per budget: err, log energy (raw, adapter) at each KL target; NaN when unmatched."""
    done = z["doneC"]; EPS = np.array(z["EPS"], float); RC = list(z["RC"]); n = done.size
    nd = z["c_kl"].shape[2]; rho = float(z["rho"])
    floor = np.maximum(z["c_dup_fish"], z["c_floor_fish"])                 # (n, budgets)
    res = {}
    for ci, r in enumerate(RC):
        reps = max(int(r), 1)
        err = np.full((n, nd, len(KL_TARGETS)), np.nan); lE = err.copy(); lEa = err.copy()
        base_bad = 0
        for i in np.where(done)[0]:
            for j in range(nd):
                kl = z["c_kl"][i, ci, j]; e_ = z["c_linerr"][i, ci, j]; fi = z["c_fish"][i, ci, j]
                if not (np.isfinite(fi[0]) and fi[0] >= FLOOR_MULT * floor[i, ci]):
                    base_bad += 1
                    continue
                ok = np.isfinite(kl) & np.isfinite(e_) & (e_ > 0) & (kl > 0) & np.isfinite(fi) & (fi >= FLOOR_MULT * floor[i, ci])
                ok[0] = False                                               # zero linear error by construction
                pe = z["c_push_euclid"][i, ci, j] * rho                     # ||u|| (c_push_euclid stores ||u|| / rho)
                gain = z["c_adapter_gain"][i, ci, j] if "c_adapter_gain" in z else np.nan
                for ti, t in enumerate(KL_TARGETS):
                    for a in range(1, len(EPS) - 1):
                        if ok[a] and ok[a + 1] and kl[a + 1] > kl[a] and kl[a] <= t <= kl[a + 1]:
                            w = (np.log(t) - np.log(kl[a])) / (np.log(kl[a + 1]) - np.log(kl[a]))
                            err[i, j, ti] = np.exp((1 - w) * np.log(e_[a]) + w * np.log(e_[a + 1]))
                            leps = (1 - w) * np.log(EPS[a]) + w * np.log(EPS[a + 1])
                            lE[i, j, ti] = np.log(reps) + 2 * (leps + np.log(pe))
                            if np.isfinite(gain):
                                lEa[i, j, ti] = np.log(reps) + 2 * (leps + np.log(pe * gain))
                            break
        acc = fl((z["c_base_top1"][done, ci] == z["ans"][done]).mean()) if done.any() else None
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(z["c_dlogp"][done, ci] - z["c_dlogp_pred"][done, ci]) / np.abs(z["c_dlogp"][done, ci])
            dl_relerr = np.nanmedian(rel, axis=(0, 1)) if done.any() else None
        res[int(r)] = dict(err=err, logE=lE, logE_adapter=lEa, done=done, acc=acc, base_unusable=base_bad,
                           n_dirs_matched=[[int(np.isfinite(err[:, j, ti]).sum()) for ti in range(len(KL_TARGETS))]
                                           for j in range(nd)],
                           dlogp_relerr_by_dose=None if dl_relerr is None else [fl(v) for v in dl_relerr])
    return res


def compare_C(ca, cd, nd, exclude=()):
    """Looped (a) vs D at each KL target and direction group, on directions matched in both models."""
    groups = dict(random=(list(range(nd - 2)), 2), gradient=([nd - 2, nd - 1], 2))
    n = ca["err"].shape[0]
    km = keep_mask(n, exclude) & ca["done"] & cd["done"]
    out = {}
    for g, (dirs, need) in groups.items():
        for ti, t in enumerate(KL_TARGETS):
            lr = np.full(n, np.nan); ea = np.full(n, np.nan); ed = np.full(n, np.nan); eaa = np.full(n, np.nan)
            erra = np.full(n, np.nan); errd = np.full(n, np.nan)
            ndir = np.zeros(n, int)
            for i in np.where(km)[0]:
                common = [j for j in dirs if np.isfinite(ca["err"][i, j, ti]) and np.isfinite(cd["err"][i, j, ti])]
                ndir[i] = len(common)
                if len(common) < need:
                    continue
                erra[i] = np.median(ca["err"][i, common, ti]); errd[i] = np.median(cd["err"][i, common, ti])
                lr[i] = np.log(erra[i] / errd[i])
                ea[i] = np.median(ca["logE"][i, common, ti]); ed[i] = np.median(cd["logE"][i, common, ti])
                if np.isfinite(ca["logE_adapter"][i, common, ti]).all():
                    eaa[i] = np.median(ca["logE_adapter"][i, common, ti])
            out[f"{g} KL={t:g}"] = dict(log_linerr_ratio=boot_median(lr), per_prompt=[fl(v) for v in lr],
                                         abs_err_looped=boot_median(erra), abs_err_D=boot_median(errd),
                                         n_common_dirs=ndir.tolist(), log_energy_looped=boot_median(ea),
                                         log_energy_D=boot_median(ed), log_energy_ratio=boot_median(ea - ed),
                                         log_energy_adapter_looped=boot_median(eaa))
    return out


def correct_row(top, ans, valid, logp, items, strata, screen_key=None):
    """Correctness and answer probability per prompt; unmeasured prompts are None and excluded from denominators.
    Screen agreement is reported as correctness agreement and as identical greedy token."""
    valid = np.asarray(valid, bool) & (np.asarray(top) >= 0)
    corr = np.asarray(top) == np.asarray(ans)
    row = dict(n_measured=int(valid.sum()), all=fl(corr[valid].mean()) if valid.any() else None,
               per_prompt=[int(c) if v else None for c, v in zip(corr, valid)],
               p_answer=[fl(np.exp(lp)) if v else None for lp, v in zip(logp, valid)])
    for st in strata:
        m = valid & np.array([it.get("stratum") == st for it in items])
        row[st] = fl(corr[m].mean()) if m.any() else None
    if screen_key is not None:
        scr = [it.get("screen", {}).get(screen_key) for it in items]
        if all(x is not None for x in scr):
            row["correctness_agrees_with_screen"] = int(sum(bool(c) == bool(x["correct"]) for c, x, v in zip(corr, scr, valid) if v))
            if all("top1" in x for x in scr):
                row["same_greedy_token_as_screen"] = int(sum(int(t) == x["top1"] for t, x, v in zip(top, scr, valid) if v))
    return row


def flops(key, man, r):
    """Approximate forward matmul FLOPs: per token, and for the prompt (all L tokens through the trunk, one head call)."""
    pc = man.get("param_counts")
    if key == "D":
        trunk = pc["layers"] if pc else 16 * LAYER_P
        head = pc["lm_head"] if pc else HEAD_P
    else:
        trunk = (pc["prelude"] + pc["coda"] + r * (pc["core_block"] + pc["adapter"])) if pc else (8 * LAYER_P + r * (6 * LAYER_P + ADAPTER_P))
        head = pc["lm_head"] if pc else HEAD_P
    L = float(np.mean([len(x) for x in man["ids"]])) if man.get("ids") else np.nan
    return dict(per_token=2.0 * (trunk + head), per_prompt=2.0 * (L * trunk + head), mean_L=L,
                source="manifest" if pc else "Llama-3.2-1B shapes")


# ------------------------------------------------------------------------------------------------ report
def main(run, ref=None):
    data, notes = load(run, ref)
    S = dict(run=os.path.abspath(run), models=sorted(data), notes=notes, rules=__doc__.split("Frozen rules")[1].strip())
    lines = [f"run {run}: models {sorted(data)}"] + [f"NOTE {x}" for x in notes]
    PA, PC = {}, {}
    # subsets: all prompts; without the smoke prompts (from the run's own settings; part 1 used uids 0, 1); per stratum
    m0 = next(iter(data.values()))[2]
    smoke = list(m0.get("cfg", {}).get("smoke_uids", SMOKE_UIDS))
    subsets = [("all", ())] + ([("no_smoke", tuple(smoke))] if smoke else [])
    items = json.load(open(os.path.join(run, "prompts.json"))) if os.path.exists(os.path.join(run, "prompts.json")) else []
    strata = sorted({it["stratum"] for it in items if "stratum" in it})
    for st in strata:
        subsets.append((f"stratum_{st}", tuple(it["uid"] for it in items if it.get("stratum") != st)))
    S["subsets"] = {k: list(v) for k, v in subsets}
    for key, (z, meta, man) in data.items():
        M = MDEP[key]
        lines.append(f"\n== {key}: status {meta.get('status')}, gate_fail {meta.get('gate_fail')}, A done "
                     f"{int(z['doneA'].sum())}, C done {int(z['doneC'].sum())} / {z['doneC'].size}, runtime "
                     f"{meta.get('runtime_s', float('nan')):.0f}s, item_s {[round(v, 1) for v in meta.get('item_s', [])]}")
        lines.append(f"   G1 max {meta.get('g1_max')}, G2 {meta.get('g2')}, G4 {meta.get('g4')}")
        lines.append(f"   floors {meta.get('floors')}")
        S[key] = dict(status=meta.get("status"), gate_fail=meta.get("gate_fail"), g4=meta.get("g4"), floors=meta.get("floors"),
                      numerics=meta.get("numerics"),
                      flops={str(r): flops(key, man, r) for r in ([0] if key == "D" else sorted({M, 32}))})
        if z["doneA"].size and z["doneA"].any():
            pa, acc = part_a(z, M); PA[key] = pa
            S[key]["acc_loop"] = acc
            lines.append(f"   greedy accuracy at t = {sorted(acc)}: {list(acc.values())}")
            for r, d in pa.items():
                info = dict(n_done=int(d["done"].sum()), n_lowres=int(d["lowres"].sum()), n_validA=int(d["valid_A"].sum()),
                            frac_valid_F=d["frac_valid"], lowres=d["lowres"].tolist(), valid_A=d["valid_A"].tolist(),
                            A=[fl(v) for v in d["A"]], Aabs=[fl(v) for v in d["Aabs"]], kappa=[fl(v) for v in d["kappa"]],
                            kappa_state=[fl(v) for v in d["kappa_state"]],
                            kappa_state_over_obs=[fl(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else None
                                                  for a, b in zip(d["kappa_state"], d["kappa"])],
                            lin_k1_median=fl(np.nanmedian(d["lin_k1"])) if np.isfinite(d["lin_k1"]).any() else None)
                S[key][f"A_r{r}"] = info
                medA = fl(np.nanmedian(d["A"])) if np.isfinite(d["A"]).any() else None
                medk = fl(np.nanmedian(d["kappa"])) if np.isfinite(d["kappa"]).any() else None
                lines.append(f"   r={r}: low-residual {info['n_lowres']}/{info['n_done']}, A valid {info['n_validA']}, F valid "
                             f"frac {info['frac_valid_F']}, median A {medA}, median kappa {medk}, "
                             f"A2 linearity F1(1e-2)/F1(1e-3) median {info['lin_k1_median']}")
        if z["doneC"].any():
            pc = part_c(z, man); PC[key] = pc
            for r, d in pc.items():
                S[key][f"C_r{r}"] = dict(acc=d["acc"], base_unusable=d["base_unusable"], n_dirs_matched=d["n_dirs_matched"],
                                         dlogp_relerr_by_dose=d["dlogp_relerr_by_dose"])
                lines.append(f"   C budget {r}: greedy acc {d['acc']}, directions with unusable base difference "
                             f"{d['base_unusable']}, prompts matched per direction x target {d['n_dirs_matched']}")

    # ---- Part A: every looped pair, numerator = smaller training depth (primary: L4 / L32 at r = 64)
    looped = sorted(PA, key=lambda k: MDEP[k])
    S["A_compare"] = {}
    for ia, a in enumerate(looped):
        for b in looped[ia + 1:]:
            for tag, ra, rb in (("r64", 64, 64), ("r32", 32, 32), ("ownM", MDEP[a], MDEP[b])):
                for sub, exc in subsets:
                    c = compare_A(PA[a], PA[b], ra, rb, exc)
                    S["A_compare"][f"{a}/{b} {tag} {sub}"] = c
                    rel = c["relative"]
                    lines.append(f"\nA: log(A_{a}/A_{b}) at {tag} [{sub}]: n {rel['n']}, median {rel['median']}, CI {rel['ci95']}, "
                                 f"positive {rel['n_pos']}{'' if rel['eligible_n12'] else '  -> INCONCLUSIVE (< 12 pairs)'}; "
                                 f"abs-normalized median {c['absolute']['median']}; curves cross: {c['curves_cross']}; "
                                 f"excluded {c['excluded']}")
    # ---- Part C: looped versus D at matched KL, on common directions
    S["C_compare"] = {}
    if "D" in PC:
        nd = data["D"][0]["c_kl"].shape[2]
        for a in sorted([k for k in PC if k != "D"], key=lambda k: MDEP[k]):
            for r, d in PC[a].items():
                for sub, exc in subsets:
                    cc = compare_C(d, PC["D"][0], nd, exc)
                    S["C_compare"][f"{a}(r={r})/D {sub}"] = cc
                    for k, v in cc.items():
                        bm = v["log_linerr_ratio"]
                        lines.append(f"C: log(linerr {a}(r={r}) / D) [{sub}] {k}: n {bm['n']}, median {bm['median']}, CI "
                                     f"{bm['ci95']}{'' if bm['eligible_n12'] else '  -> INCONCLUSIVE (< 12 pairs)'}; log energy ratio median "
                                     f"{v['log_energy_ratio']['median']}")
    # ---- correctness per model, depth and stratum; agreement with the difficulty screen (different random starts)
    if items:
        S["correctness"] = {}
        for key, (z, meta, man) in data.items():
            if key == "D":
                cols = {"D": (z["c_base_top1"][:, 0], z["doneC"], z["c_base_logp"][:, 0])}
            else:
                cols = {f"t{t}": (z["a0_top1"][:, t], z["doneA"], z["a0_logp"][:, t]) for t in sorted({MDEP[key], 32, 64})}
            for tag, (top, valid, logp) in cols.items():
                row = correct_row(top, z["ans"], valid, logp, items, strata, key if tag in ("D", "t64") else None)
                S["correctness"][f"{key} {tag}"] = row
                lines.append(f"correct {key} {tag}: {row}")
    with open(os.path.join(run, "summary.json"), "w") as f:
        json.dump(S, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\n".join(lines))
    print(f"\nwrote {os.path.join(run, 'summary.json')}")


if __name__ == "__main__":
    args = sys.argv[1:]
    ref = None
    if "--ref" in args:
        k = args.index("--ref"); ref = args[k + 1]; del args[k:k + 2]
    main(args[0], ref)
