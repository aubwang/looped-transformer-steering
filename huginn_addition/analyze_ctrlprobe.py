"""Pre-registered analysis of the suppression control (ctrlprobe_app.py; docs/freeloop.pdf Section 6).
Written, and its readings fixed, before the production runs (2026-11-07).

Question. Huginn's task readouts g have self-gain alpha(g) = <g, J^T g> far below the loop's mean eigenvalue tau(J)
(240 of 240 readouts below every random probe). Is that deficit specific to the task readouts, or shared by the
directions the coda reads in general?

Per context i (16 Gaussian probes z: the 4 freeprobe probes + 12 new ones):
  tau_i = mean_z alpha(z);  tau(B^2)_i = (mean_z <z, (J^T)^2 z> + mean_z |J^T z|^2) / 2 - tau_i^2  (Proposition R);
  sd_i = sqrt(2 tau(B^2)_i / (d + 2));  excess(u) = alpha(u) - tau_i;  Z(u) = excess(u) / sd_i.
Families F: task (truth 1 readout, digits 10), C1 random-token contrasts, C2 plausible-token contrasts, C3 random logit
combinations (8 each per context). Delta_F(i) = median over the directions of F at context i of excess(u).
  rho_F = median_i Delta_F(i) / median_i Delta_task(i)   (share of the task deficit that the controls have)
  D_F(i) = Delta_task(i) - Delta_F(i)                     (paired: task readouts minus controls)
CIs: context bootstrap (5000), percentile 95%; rho_F's CI resamples contexts and recomputes both medians.
READINGS (per task and family). The primary family is C3: typical sensitivity-weighted coda contrasts (a direction
the coda reads, weighted by how strongly it reads it; it can miss weakly read directions, so no reading is a statement
about every direction the coda reads). C1 (vocabulary-wide) and C2 (plausible tokens) get the same rules (secondary).
tau(J) is a descriptive reference; the paired task - control comparison does not depend on it.
  NO TASK DEFICIT  CI(median_i Delta_task(i)) not entirely < 0 (nothing to compare)
  TASK-SPECIFIC    CI(D_F) entirely < 0 and CI(rho_F) entirely <= 0.25
  SHARED           CI(Delta_F) entirely < 0 and CI(rho_F) entirely >= 0.75   (shared by typical coda contrasts)
  PARTIAL          CI(Delta_F) entirely < 0, CI(D_F) entirely < 0 and CI(rho_F) inside (0.25, 0.75)
  UNCLEAR          otherwise (the point-estimate classification is reported alongside)
Overall (C3): the common reading if truth and digits agree; INCONCLUSIVE if either is UNCLEAR or NO TASK DEFICIT;
CONTEXT-DEPENDENT if they give different clear readings.
Descriptive: the share of directions below all 16 probes of their context, median Z per family, within-family |cos|
between directions (diversity), cos with x*_q, gradient norms; the same statistics without contexts whose drift exceeds
1e-3 (sensitivity). Exploratory (exact, offline from x1 and the start Gram): each control direction with the span of the
task readouts of its context projected out (pseudoinverse), then renormalized; its excess against the task's.
GATES (no conclusions if any fails): complete run (every item done; smoke runs may lower this); every used array finite;
unit starts (|gram3[0,0] - 1| < 1e-5); gradient norms > 0; tau(B^2) > 0 in every context; median adjoint error < 5e-3 in
every family; validation max_abs < 1e-3 and argmax agreement 1; eq_err < 1e-5; cross-run identity with the freeprobe run
of the same items: the 4 old-seed probes' alpha within 1e-4, the shared task readouts' G within 1e-4 (relative) and
alpha within 1e-4.

Usage: python -I analyze_ctrlprobe.py <ctrlprobe run dir> <freeprobe run dir> [min_done, smoke runs only]
       python -I analyze_ctrlprobe.py --overall <truth ctrl_summary.json> <digits ctrl_summary.json>
       python -I analyze_ctrlprobe.py --selftest      (synthetic worlds with known answers)
"""
import json
import sys

import numpy as np

FAMS = {1: "C1", 2: "C2", 3: "C3", 6: "C3flip"}  # 6: sign-flipped C3 (Ouro run only)


def boot_median(v, B=5000, seed=0):
    v = np.asarray(v); rng = np.random.default_rng(seed)
    b = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
    return [float(np.median(v)), float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def boot_ratio(a, b, B=5000, seed=1):
    a, b = np.asarray(a), np.asarray(b); rng = np.random.default_rng(seed); n = len(a)
    bs = []
    for _ in range(B):
        k = rng.integers(0, n, n); bs.append(np.median(a[k]) / np.median(b[k]))
    return [float(np.median(a) / np.median(b)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def reading(DT, DF, Dpair, rho):
    if not DT[2] < 0:
        return "NO TASK DEFICIT"
    if Dpair[2] < 0 and rho[2] <= 0.25:
        return "TASK-SPECIFIC"
    if DF[2] < 0 and rho[1] >= 0.75:
        return "SHARED"
    if DF[2] < 0 and Dpair[2] < 0 and rho[1] > 0.25 and rho[2] < 0.75:
        return "PARTIAL"
    return "UNCLEAR"


def point_reading(DT, DF, Dpair, rho):
    return reading(DT, DF, Dpair, [rho[0]] * 3)


def projected_alpha(ug, x1, t_idx, c_idx):
    """alpha of (c - Pi c) / |c - Pi c|, Pi the orthogonal projection onto span{u_t, t in t_idx}; exact from the start
    Gram ug[a, b] = <u_a, u_b> and x1[a, b] = <u_a, J^T u_b>. Returns (alpha_perp, |Pi c|^2)."""
    Gtt = ug[np.ix_(t_idx, t_idx)]; Gp = np.linalg.pinv(Gtt, rcond=1e-8, hermitian=True)
    out = []
    for c in c_idx:
        beta = Gp @ ug[t_idx, c]                                   # Pi c = sum_k beta_k u_k
        num = x1[c, c] - beta @ (x1[t_idx, c] + x1[c, t_idx]) + beta @ x1[np.ix_(t_idx, t_idx)] @ beta
        nn = ug[c, c] - ug[c, t_idx] @ beta
        out.append((num / nn, 1 - nn))
    return np.array(out)


def evaluate(gram3, fam, d, ugram=None, x1=None):
    alpha = gram3[:, :, 0, 1]; n1 = gram3[:, :, 1, 1]; m2 = gram3[:, :, 0, 2]
    P = (fam == 4) | (fam == 5)
    tau = alpha[:, P].mean(1)
    tB2 = (m2[:, P].mean(1) + n1[:, P].mean(1)) / 2 - tau ** 2
    sd = np.sqrt(2 * tB2 / (d + 2))
    exc = alpha - tau[:, None]; Z = exc / sd[:, None]
    below = alpha < alpha[:, P].min(1)[:, None]
    Dtask = np.median(exc[:, fam == 0], 1)
    DT = boot_median(Dtask)
    out = dict(n=int(len(alpha)), K_probes=int(P.sum()), tau=boot_median(tau),
               probe_sd=dict(pred_rms=float(np.sqrt((sd ** 2).mean())),
                             obs_rms=float(np.sqrt(alpha[:, P].var(1, ddof=1).mean()))),
               task=dict(Delta=DT, Z=boot_median(np.median(Z[:, fam == 0], 1)),
                         below_all=float(below[:, fam == 0].mean())))
    t_idx = np.where(fam == 0)[0]
    for f, name in FAMS.items():
        if not (fam == f).any():
            continue
        DF = np.median(exc[:, fam == f], 1)
        r = dict(Delta=boot_median(DF), D_pair=boot_median(Dtask - DF), rho=boot_ratio(DF, Dtask),
                 Z=boot_median(np.median(Z[:, fam == f], 1)), below_all=float(below[:, fam == f].mean()))
        r["reading"] = reading(DT, r["Delta"], r["D_pair"], r["rho"])
        r["point_reading"] = point_reading(DT, r["Delta"], r["D_pair"], r["rho"])
        idx = np.where(fam == f)[0]
        if ugram is not None:
            cs = [abs(ugram[i][a, b]) for i in range(len(alpha)) for k, a in enumerate(idx) for b in idx[k + 1:]]
            r["within_family_abs_cos"] = [float(np.median(cs)), float(np.quantile(cs, .9))]
        if ugram is not None and x1 is not None:   # exploratory: task span projected out
            pa = np.array([projected_alpha(ugram[i], x1[i], t_idx, idx) for i in range(len(alpha))])  # (n, nc, 2)
            DFp = np.median(pa[:, :, 0] - tau[:, None], 1)
            r["perp"] = dict(Delta=boot_median(DFp), rho=boot_ratio(DFp, Dtask), D_pair=boot_median(Dtask - DFp),
                             share_in_task_span=[float(np.median(pa[:, :, 1])), float(np.quantile(pa[:, :, 1], .9))])
        out[name] = r
    return out


def fail(out, run, msg):
    print(msg); out["gates"]["PASS"] = False; json.dump(out, open(f"{run}/ctrl_summary.json", "w"), indent=1); return out


def gates_and_eval(run, ref, min_done=None):
    z = np.load(f"{run}/ctrlprobe.npz"); meta = json.load(open(f"{run}/meta.json")); cfg = meta["cfg"]
    task = cfg["task"]; done = z["done"]; fam = z["fam"]; d = z["G"].shape[-1]
    out = dict(run=run, ref=ref, task=task, gates=dict(n_items=int(len(done)), n_done=int(done.sum())))
    g = out["gates"]
    g["min_done"] = min_done or len(done)
    if done.sum() < max(g["min_done"], 2):
        return fail(out, run, f"GATES FAILED: {done.sum()} of {len(done)} items done")
    g3, uid, ug, x1 = z["gram3"][done], z["uid"][done], z["ugram"][done], z["x1"][done]
    gn = z["gnorm"][done][:, fam <= 3]; adj = z["adjoint_rel_err"][done]
    g["finite"] = bool(np.isfinite(g3).all() and np.isfinite(ug).all() and np.isfinite(x1).all() and np.isfinite(gn).all()
                       and np.isfinite(adj).all() and all(np.isfinite(z[k][done]).all() for k in ("drift", "cos_x", "G")))
    if not g["finite"]:
        return fail(out, run, "GATES FAILED: non-finite arrays")
    g["gnorm_min"] = float(gn.min())
    g["unit_starts"] = float(np.abs(g3[:, :, 0, 0] - 1).max())
    a_ = g3[:, :, 0, 1][:, fam >= 4]
    tB2 = (g3[:, :, 0, 2][:, fam >= 4].mean(1) + g3[:, :, 1, 1][:, fam >= 4].mean(1)) / 2 - a_.mean(1) ** 2
    g["tB2_min"] = float(tB2.min())
    g["adjoint_median_by_family"] = {int(f): float(np.median(adj[:, fam == f])) for f in np.unique(fam)}
    g["adjoint_max"] = float(adj.max())
    g["val_max_abs"] = max(v["max_abs"] for v in meta["validation"]); g["val_argmax"] = min(v["argmax_agree"] for v in meta["validation"])
    g["eq_err"] = max(meta["eq_err"])
    # cross-run identity with freeprobe (T = 48 Gram, v = m (T + 1) + t; truth NR = 1, digits NR = 2 with k = 0, 5)
    F = np.load(f"{ref}/freeprobe.npz"); fm = json.load(open(f"{ref}/meta.json"))["cfg"]
    T, NRf = fm["T"], len(fm.get("digits_k", [None])); o = lambda m: m * (T + 1)
    idx = {u: k for k, u in enumerate(F["uid"])}; kk = np.array([idx[u] for u in uid])
    assert F["done"][kk].all()
    a_old = np.array([[F["gram"][k, o(m), o(m) + 1] for m in range(NRf, NRf + 4)] for k in kk])
    g["probe_alpha_vs_freeprobe"] = float(np.abs(g3[:, fam == 4, 0, 1] - a_old).max())
    shared = [0] if task == "truth" else list(fm["digits_k"])
    a_task_old = np.array([[F["gram"][k, o(m), o(m) + 1] for m in range(NRf)] for k in kk])
    Gold = F["G"][kk]; Gold = Gold[:, None] if Gold.ndim == 2 else Gold
    Gnew = z["G"][done][:, shared]
    g["G_vs_freeprobe"] = float((np.linalg.norm(Gnew - Gold, axis=-1) / np.linalg.norm(Gold, axis=-1)).max())
    g["task_alpha_vs_freeprobe"] = float(np.abs(g3[:, np.where(fam == 0)[0][shared], 0, 1] - a_task_old).max())
    g["PASS"] = bool(g["gnorm_min"] > 0 and g["unit_starts"] < 1e-5 and g["tB2_min"] > 0
                     and max(g["adjoint_median_by_family"].values()) < 5e-3
                     and g["val_max_abs"] < 1e-3 and g["val_argmax"] == 1 and g["eq_err"] < 1e-5
                     and g["probe_alpha_vs_freeprobe"] < 1e-4 and g["G_vs_freeprobe"] < 1e-4 and g["task_alpha_vs_freeprobe"] < 1e-4)
    print("GATES", json.dumps(g, indent=1))
    if not g["PASS"]:
        return fail(out, run, "GATES FAILED: no conclusions")
    drift = z["drift"][done]
    out["main"] = evaluate(g3, fam, d, ug, x1)
    keep = drift <= 1e-3
    out["drift_above_1e-3"] = int((~keep).sum())
    if (~keep).any() and keep.sum() >= 2:
        out["sensitivity_drift_le_1e-3"] = evaluate(g3[keep], fam, d)
    cx = z["cos_x"][done]; gnall = z["gnorm"][done]
    out["descr"] = {name: dict(abs_cos_x=float(np.median(np.abs(cx[:, fam == f]))),
                               gnorm=float(np.median(gnall[:, fam == f])) if f <= 3 else None)
                    for f, name in {0: "task", 1: "C1", 2: "C2", 3: "C3", 5: "probes"}.items()}
    print(json.dumps(out["main"], indent=1)); print("descr", json.dumps(out["descr"]))
    json.dump(out, open(f"{run}/ctrl_summary.json", "w"), indent=1)
    return out


def overall(s_truth, s_digits):
    a, b = (json.load(open(f))["main"]["C3"]["reading"] for f in (s_truth, s_digits))
    if a in ("UNCLEAR", "NO TASK DEFICIT") or b in ("UNCLEAR", "NO TASK DEFICIT"):
        res = "INCONCLUSIVE"
    else:
        res = a if a == b else "CONTEXT-DEPENDENT"
    print(json.dumps(dict(truth=a, digits=b, overall=res)))
    return res


def selftest():
    """Synthetic worlds (d = 800, 40 contexts): J = gamma I + A (non-normal, A Gaussian / sqrt(d) * 0.45) - delta P_R,
    R a random 24-dimensional 'read subspace'; the task readout is a random unit vector in R. Controls: in R
    (SHARED expected), random in R^d (TASK-SPECIFIC), half in R by squared mass (PARTIAL)."""
    d, n, nc, K = 800, 40, 8, 16; rng = np.random.default_rng(0)
    fam = np.array([0] + [1] * nc + [2] * nc + [3] * nc + [4] * 4 + [5] * (K - 4))
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    want = {"inside": "SHARED", "outside": "TASK-SPECIFIC", "half": "PARTIAL"}
    for world, exp in want.items():
        g3 = np.zeros((n, len(fam), 3, 3)); UGs = np.zeros((n, len(fam), len(fam))); X1s = np.zeros_like(UGs)
        for i in range(n):
            Qr = np.linalg.qr(rng.standard_normal((d, 24)))[0]
            J = (0.25 + 0.04 * rng.standard_normal()) * np.eye(d) + 0.45 * rng.standard_normal((d, d)) / np.sqrt(d) - 0.15 * Qr @ Qr.T
            inR = lambda m: unit(rng.standard_normal((m, 24)) @ Qr.T)
            if world == "inside":
                C = inR(3 * nc)
            elif world == "outside":
                C = unit(rng.standard_normal((3 * nc, d)))
            else:
                C = unit(inR(3 * nc) + unit(rng.standard_normal((3 * nc, d))))
            U = np.concatenate([inR(1), C, unit(rng.standard_normal((K, d)))])
            W1 = U @ J; W2 = W1 @ J                                                   # rows: (J^T u)^T = u^T J
            Cv = np.stack([U, W1, W2], 1); g3[i] = Cv @ Cv.transpose(0, 2, 1)
            UGs[i] = U @ U.T; X1s[i] = U @ W1.T
            if i == 0:  # projection formula against a direct computation
                c = U[3] - (U[3] @ U[0]) * U[0]; c /= np.linalg.norm(c)
                pa = projected_alpha(UGs[0], X1s[0], np.array([0]), np.array([3]))
                assert abs(pa[0, 0] - c @ J @ c) < 1e-10 and abs(pa[0, 1] - (U[3] @ U[0]) ** 2) < 1e-10
        r = evaluate(g3, fam, d, UGs, X1s)
        got = {name: r[name]["reading"] for name in FAMS.values() if name in r}
        print(world, "expected", exp, "got", got, "rho C3", [round(x, 2) for x in r["C3"]["rho"]],
              "perp rho C3", [round(x, 2) for x in r["C3"]["perp"]["rho"]],
              "probe sd pred/obs", round(r["probe_sd"]["pred_rms"], 4), round(r["probe_sd"]["obs_rms"], 4))
        assert all(v == exp for v in got.values()), (world, got)
    print("SELFTEST PASSED")


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
    elif sys.argv[1] == "--overall":
        overall(sys.argv[2], sys.argv[3])
    else:
        gates_and_eval(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)  # 3rd arg: smoke min_done
