"""EXPLORATORY (post hoc, 2026-11-07, after the ctrlprobe and Ouro results): does a start-invariance mechanism explain
Huginn's read-space deficit? Linear consequence (docs/freeloop.pdf, Section 6): if the output were exactly invariant to
the start state after m loop steps, G J^m = 0 at the fixed point, so the readout chains |(J^T)^t g| would vanish for
t >= m (and the mean self-gain over the read directions' Krylov hull would be 0). Test from saved data only:
  1. freeprobe runs (T = 48 chains, frozen J at the fixed point): per context, |(J^T)^t g_hat| (g_hat = g/|g|, as stored)
     / median over the 4 probes of |(J^T)^t z|, median over readouts and contexts, for t = 1..48; successive-pullback
     cosines; the count of readout chain norms below the probe median over t = 1..8;
  2. ctrlprobe runs (T = 2): |(J^T)^2 u| and cos(J^T u, (J^T)^2 u) for C3 vs probes.
Usage: python -I analyze_chains.py <freeprobe dir> <ctrlprobe dir> [<freeprobe dir> <ctrlprobe dir> ...]
"""
import json
import sys

import numpy as np


def chains(fdir):
    z = np.load(f"{fdir}/freeprobe.npz"); cfg = json.load(open(f"{fdir}/meta.json"))["cfg"]
    T = cfg["T"]; NR = len(cfg.get("digits_k", [None])); g = z["gram"][z["done"]]; n = len(g); M = g.shape[1] // (T + 1)
    nrm = np.sqrt(np.diagonal(g, axis1=1, axis2=2).reshape(n, M, T + 1))
    off = np.array([[g[:, m * (T + 1) + t, m * (T + 1) + t + 1] for t in range(T)] for m in range(M)]).transpose(2, 0, 1)
    cs = off / (nrm[:, :, :-1] * nrm[:, :, 1:])
    ratio = np.array([float(np.median(nrm[:, :NR, t] / np.median(nrm[:, NR:, t], 1)[:, None])) for t in range(1, T + 1)])
    frac_below = np.array([float(np.mean(nrm[:, :NR, t] < np.median(nrm[:, NR:, t], 1)[:, None])) for t in range(1, T + 1)])
    return dict(T=T, NR=NR, n=n, ratio=ratio.tolist(), ratio_min_1_32=float(ratio[:32].min()), ratio_max_1_32=float(ratio[:32].max()),
                frac_readout_below_probe_median_1_8=float(frac_below[:8].mean()),
                n_below_1_8=int(sum((nrm[:, :NR, t] < np.median(nrm[:, NR:, t], 1)[:, None]).sum() for t in range(1, 9))),
                n_obs_1_8=int(8 * n * NR),
                probe_norm_t32=float(np.median(nrm[:, NR:, 32])),
                cos_readout=[float(np.median(cs[:, :NR, t])) for t in range(T)], cos_probes=[float(np.median(cs[:, NR:, t])) for t in range(T)])


def twostep(cdir):
    z = np.load(f"{cdir}/ctrlprobe.npz"); g3 = z["gram3"][z["done"]]; fam = z["fam"]
    n1, n2 = g3[..., 1, 1], g3[..., 2, 2]; c12 = g3[..., 1, 2] / np.sqrt(n1 * n2)
    out = {}
    for name, f in (("task", fam == 0), ("C3", fam == 3), ("probes", (fam == 4) | (fam == 5))):
        out[name] = dict(norm2=float(np.median(np.sqrt(n2[:, f]))), cos12=float(np.median(c12[:, f])),
                         step2_over_step1=float(np.median(np.sqrt(n2[:, f] / n1[:, f]))))
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    for fdir, cdir in zip(a[::2], a[1::2]):
        c = chains(fdir)
        print(fdir, json.dumps({k: v for k, v in c.items() if k not in ("ratio", "cos_readout", "cos_probes")}))
        print("   ratio t=1,2,4,8,16,32,48:", [round(c["ratio"][t - 1], 3) for t in (1, 2, 4, 8, 16, 32, 48)])
        print(cdir, json.dumps(twostep(cdir)))
