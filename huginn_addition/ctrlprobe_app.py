"""Suppression control for docs/freeloop.pdf (Section 6): is the readouts' low self-gain alpha(g) = <g, J^T g> specific
to the task readouts, or shared by the directions the coda reads in general?

Same contexts and states as the freeprobe runs (truth: results/freeprobe-20261107-130520/items.json, 40 items;
digits: results/freeprobe_digits-20261107-133750/items.json, 20 items; same K0, same init, same chunking). Per item,
S unit start vectors u in the own block at the answer position q, in this order (family code in `fam`):
  0  task readouts: truth  g = grad of log p(' True') - log p(' False')                         (NR = 1)
                    digits g_k = grad of the centered digit logit z_k, k = 0..9 (as dglever)    (NR = 10)
  1  C1  n_c random-token contrasts   grad (l_a - l_b), a != b uniform over non-special tokens, task tokens excluded
  2  C2  n_c plausible-token contrasts grad (l_a - l_b), a != b among the 64 highest-logit non-special non-task tokens at x*
  3  C3  n_c random logit combinations grad (xi . log p), xi Gaussian on the non-special tokens, centered (so = xi . logits)
  4  the K_old = 4 Gaussian probes of freeprobe (same generator call: seed + uid), for the cross-run identity gate
  5  K_new further Gaussian probes (seed2 + uid)
(l = log-softmax at q; task tokens: ' True', ' False' / the 10 digit and 10 space-digit tokens.) For every start the chain
w_0 = u, w_t = J^T w_{t-1}, t = 1..T (T = 2; one VJP through core_block_forward per step, as freeprobe), and per start the
3 x 3 Gram of (w_0, w_1, w_2) (float64): alpha(u) = gram3[0, 1], |w_1|^2 = gram3[1, 1], <u, (J^T)^2 u> = gram3[0, 2].
Also saved: the S x S Gram of the starts, the cross products x1[a, b] = <u_a, J^T u_b>, x2[a, b] = <u_a, (J^T)^2 u_b>
(no extra passes; for offline projections, e.g. controls with the task-gradient span removed), the raw gradient norms, cos(u, x*_q), the token ids, G (task gradients, for the
cross-run gate), drift and the relative adjoint error of every start (central FD on a random unit r, as freeprobe).
Analysis (pre-registered before the production runs): analyze_ctrlprobe.py.

Run from huginn_addition/ (the image mounts intervene_app.py, whose helper class H computes the states):
  <venv>/bin/modal run ctrlprobe_app.py --mode ctrlprobe_smoke --probs ../results/freeprobe-20261107-130520/items.json
Outputs: Modal volume `huginn-results`, <mode>-<timestamp>/ (manifest.json and items.json first; chunk_*.npz per chunk;
ctrlprobe.npz and meta.json at the end).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import modal

import intervene_app as iv
from intervene_app import H, REVISION, hf_cache, results_vol

app = modal.App("huginn-ctrlprobe")
image = iv.image.add_local_python_source("intervene_app")


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=600)
def ctrlprobe(items_json: str, cfg: dict, run_id: str):
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    task = cfg["task"]
    if task == "truth":  # exactly as freeprobe (truth)
        seqs = []
        for r in items:
            ids_ = h.tok.encode(r["prompt_B"], add_special_tokens=True)
            seqs.append(dict(uid=r["uid"] + 500000, ids=ids_, pred_pos=[len(ids_) - 1]))
        val = h.validate(seqs[:20])
        TF = [h.tok.encode(w_, add_special_tokens=False)[0] for w_ in (" True", " False")]
        TASK_TOK = list(TF); NR = 1
    else:  # exactly as freeprobe (digits)
        assert task == "digits"
        h.prep(items)
        val = h.validate(items[:10])
        seqs = [dict(uid=p["uid"], ids=p["ids"], pred_pos=p["pred_pos"]) for p in items]
        dig = torch.tensor(h.DIG, device=h.dev)
        TASK_TOK = list(h.DIG) + list(h.SPD); NR = 10
    nc, K_old, K_new, T, d = cfg["n_c"], 4, cfg["K_new"], cfg["T"], h.d
    M = NR + 3 * nc; S = M + K_old + K_new
    fam = np.array([0] * NR + [1] * nc + [2] * nc + [3] * nc + [4] * K_old + [5] * K_new)
    Vv = h.model.lm_head.weight.shape[0]; Vt = len(h.tok)
    special = set(int(t) for t in h.tok.all_special_ids)
    real = np.array([t for t in range(min(Vv, Vt)) if t not in special])          # C3 support
    allowed = np.array([t for t in real if t not in set(TASK_TOK)])                # C1 support
    bad = torch.ones(Vv, dtype=torch.bool, device=h.dev); bad[torch.tensor(allowed, device=h.dev)] = False  # C2 mask
    supp = torch.zeros(Vv, device=h.dev); supp[torch.tensor(real, device=h.dev)] = 1.0
    n = len(items)
    GR3 = np.full((n, S, 3, 3), np.nan); UG = np.full((n, S, S), np.nan); GN = np.full((n, S), np.nan)
    X1 = np.full((n, S, S), np.nan); X2 = np.full((n, S, S), np.nan)
    CX = np.full((n, S), np.nan); TOK = np.full((n, 2 * nc, 2), -1, np.int64); G = np.zeros((n, NR, d), np.float32)
    ADJ = np.full((n, S), np.nan, np.float32); drift = np.full(n, np.nan, np.float32); DONE = np.zeros(n, bool)
    EQ = []; TM = []; MEM = []; aborted = False
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    with open(f"{o}/manifest.json", "w") as f_:  # written first, so a hard timeout still leaves the layout
        json.dump(dict(cfg=cfg, run_id=run_id, S=S, NR=NR, fam=fam.tolist(), n_items=n, validation=val, revision=REVISION), f_, indent=1)
    with open(f"{o}/items.json", "w") as f_:
        f_.write(items_json)
    results_vol.commit()
    print(f"setup {time.monotonic() - T0:.0f}s; S = {S}; |allowed| = {len(allowed)}, |real| = {len(real)}, V = {Vv}", flush=True)

    def vjp(xj, ej, qj, fc, w):
        """J^T w for R copies of one item's state (rows of w), as freeprobe."""
        R = w.shape[0]; arr = torch.arange(R, device=h.dev); qr = qj.expand(R)
        xr = xj.expand(R, -1, -1).clone(); er = ej.expand(R, -1, -1).clone()
        with torch.enable_grad():
            xq = xr[arr, qr].clone().requires_grad_(True)
            st = xr.index_put((arr, qr), xq)
            out = h.model.core_block_forward(st, er, fc, None, None, torch.tensor(0, dtype=torch.long),
                                             current_step=cfg["K0"])[0][arr, qr]
            return torch.autograd.grad((out * w).sum(), xq)[0].detach()

    def adjoint(xj, ej, qj, u, w1):
        """relative error of (J^T u) . r against u . (J r), J r by central FD (eps 0.1 on unit r), as freeprobe."""
        R = u.shape[0]; arr = torch.arange(R, device=h.dev); qr = qj.expand(R)
        with torch.no_grad():
            r_ = torch.randn((R, d), device=h.dev, generator=torch.Generator(device=h.dev).manual_seed(9))
            r_ = r_ / r_.norm(dim=-1, keepdim=True)
            xr = xj.expand(R, -1, -1); er = ej.expand(R, -1, -1).clone()
            ep = 0.1; sp, sm = xr.clone(), xr.clone(); sp[arr, qr] += ep * r_; sm[arr, qr] -= ep * r_
            jr = (h.step(er, sp, cfg["K0"])[arr, qr] - h.step(er, sm, cfg["K0"])[arr, qr]) / (2 * ep)
            lhs = (w1 * r_).sum(-1); rhs = (u * jr).sum(-1)
            return ((lhs - rhs).abs() / (lhs.abs() + rhs.abs()).clamp_min(1e-12)).cpu().numpy()

    for c0 in range(0, n, cfg["chunk"]):
        tc = time.monotonic()
        if (TM and tc - T0 + 1.3 * max(TM) > cfg["total_s"]) or len(TM) >= cfg.get("max_chunks", 10 ** 9):
            print(f"stopping before item {c0} (time or max_chunks)", flush=True)
            break
        ch = list(range(c0, min(n, c0 + cfg["chunk"]))); N = len(ch)
        with torch.no_grad():  # states exactly as freeprobe
            ids, e, s = h.batch([seqs[i] for i in ch], [0] * N)
            q = torch.tensor([seqs[i]["pred_pos"][-1] for i in ch], device=h.dev); ar = torch.arange(N, device=h.dev)
            if task == "digits":
                Lq = int(q.max()) + 1
                assert bool((q == Lq - 1).all())
                e, s = e[:, :Lq], s[:, :Lq]
            for k in range(cfg["K0"]):
                s = h.step(e, s, k)
            x = s.clone()
            fx = h.step(e, x, cfg["K0"])
            drift[ch] = ((fx - x)[ar, q].norm(dim=-1) / x[ar, q].norm(dim=-1)).cpu().numpy()
            lg0 = torch.log_softmax(h.coda_logits(x, q[:, None])[:, 0], -1)  # (N, V) at x*
        fc = h.model.freqs_cis[:, : e.shape[1]]
        with torch.no_grad():
            ref = h.step(e, x, cfg["K0"])
            alt = h.model.core_block_forward(x, e, fc, None, None, torch.tensor(0, dtype=torch.long), current_step=cfg["K0"])[0]
            EQ.append(float(((ref - alt).norm() / ref.norm()).item()))
            assert EQ[-1] < 1e-5, EQ[-1]
        res = {}
        for j, i in enumerate(ch):
            if time.monotonic() - T0 > cfg["total_s"]:
                aborted = True
                break
            uid = int(items[i]["uid"])
            xj, ej, qj = x[j : j + 1], e[j : j + 1], q[j : j + 1]
            # ---- control token choices (CPU generator, independent of the model)
            rng = np.random.default_rng(cfg["cseed"] + uid)
            c1 = np.array([rng.choice(allowed, 2, replace=False) for _ in range(nc)])
            top = torch.topk(lg0[j].masked_fill(bad, -float("inf")), 64).indices.cpu().numpy()
            c2 = np.array([top[rng.choice(64, 2, replace=False)] for _ in range(nc)])
            gx = torch.Generator(device=h.dev).manual_seed(cfg["cseed"] + uid)
            xi = torch.randn((nc, Vv), device=h.dev, generator=gx) * supp
            xi = xi - supp * (xi.sum(-1, keepdim=True) / supp.sum())                   # centered on the support
            # ---- the M coda gradients in one batched pass (row r only sees its own copy of x_q)
            arM = torch.arange(M, device=h.dev); qM = qj.expand(M)
            with torch.enable_grad():
                xs = xj.expand(M, -1, -1).clone()
                xq = xs[arM, qM].clone().requires_grad_(True)
                st = xs.index_put((arM, qM), xq)
                lgt = h.coda_logits(st, qM[:, None])[:, 0]; lg = torch.log_softmax(lgt, -1)  # (M, V)
                if task == "truth":
                    obj = [lg[0, TF[0]] - lg[0, TF[1]]]
                else:
                    zl = lg[:NR][:, dig]; zc = zl - zl.mean(-1, keepdim=True)
                    obj = [zc[k, k] for k in range(NR)]
                for r_, (a, b) in enumerate(np.concatenate([c1, c2])):
                    obj.append(lg[NR + r_, int(a)] - lg[NR + r_, int(b)])
                for k in range(nc):
                    obj.append((xi[k] * lgt[NR + 2 * nc + k]).sum())  # = xi . log p, as xi is centered on its support
                gall = torch.autograd.grad(torch.stack(obj).sum(), xq)[0].detach()     # (M, d)
            U = torch.zeros((S, d), device=h.dev)
            gn = gall.norm(dim=-1); U[:M] = gall / gn[:, None]
            gen = torch.Generator(device=h.dev).manual_seed(cfg["seed"] + uid)       # freeprobe's probe call
            z = torch.randn((K_old, d), device=h.dev, generator=gen); U[M : M + K_old] = z / z.norm(dim=-1, keepdim=True)
            gen2 = torch.Generator(device=h.dev).manual_seed(cfg["seed2"] + uid)
            z = torch.randn((K_new, d), device=h.dev, generator=gen2); U[M + K_old :] = z / z.norm(dim=-1, keepdim=True)
            # ---- pullback chains, rows in sub-batches of cfg["rows"]
            Wc = [U]; adj = np.zeros(S, np.float32)
            for t in range(T):
                nxt = torch.zeros_like(U)
                for b0 in range(0, S, cfg["rows"]):
                    sl = slice(b0, min(S, b0 + cfg["rows"]))
                    nxt[sl] = vjp(xj, ej, qj, fc, Wc[-1][sl])
                    if t == 0:
                        adj[sl] = adjoint(xj, ej, qj, U[sl], nxt[sl])
                Wc.append(nxt)
            with torch.no_grad():
                Cv = torch.stack(Wc, 1).double()                                       # (S, T + 1, d)
                xqv = xj[0, qj[0]].double()
                res[i] = dict(gram3=torch.bmm(Cv, Cv.transpose(1, 2)).cpu().numpy(), ug=(U.double() @ U.double().T).cpu().numpy(),
                              x1=(U.double() @ Wc[1].double().T).cpu().numpy(), x2=(U.double() @ Wc[2].double().T).cpu().numpy(),
                              gn=np.concatenate([gn.cpu().numpy(), np.full(S - M, np.nan)]),
                              cx=((U.double() @ xqv) / xqv.norm()).cpu().numpy(), tok=np.concatenate([c1, c2]),
                              G=gall[:NR].cpu().numpy(), adj=adj)
        for i, rr in res.items():  # items finished before a deadline inside the chunk are kept (done = True only for them)
            GR3[i], UG[i], X1[i], X2[i], GN[i], CX[i], TOK[i], G[i], ADJ[i] = (rr[k] for k in ("gram3", "ug", "x1", "x2", "gn", "cx", "tok", "G", "adj"))
            DONE[i] = True
        np.savez(f"{o}/chunk_{c0:03d}.npz", idx=np.array(ch), done=DONE[ch], gram3=GR3[ch], ugram=UG[ch], x1=X1[ch], x2=X2[ch], gnorm=GN[ch],
                 cos_x=CX[ch], tok=TOK[ch], G=G[ch], drift=drift[ch], adjoint_rel_err=ADJ[ch], uid=np.array([items[i]["uid"] for i in ch]))
        results_vol.commit()
        if aborted:
            print(f"deadline inside chunk at item {c0}: {len(res)} of {N} items kept", flush=True)
            break
        TM.append(time.monotonic() - tc)
        MEM.append(torch.cuda.max_memory_allocated() / 2 ** 30)
        print(f"items {ch[-1] + 1}/{n} {time.monotonic() - T0:.0f}s (chunk {TM[-1]:.0f}s, peak {MEM[-1]:.1f} GiB)", flush=True)
    np.savez_compressed(f"{o}/ctrlprobe.npz", gram3=GR3, ugram=UG, x1=X1, x2=X2, gnorm=GN, cos_x=CX, tok=TOK, G=G, drift=drift,
                        adjoint_rel_err=ADJ, done=DONE, fam=fam, uid=np.array([r["uid"] for r in items]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION,
            "eq_err": EQ, "chunk_s": TM, "peak_gib": MEM, "complete": bool(DONE.all()), "aborted_in_chunk": aborted, "S": S, "NR": NR,
            "n_allowed": int(len(allowed)), "n_real": int(len(real)), "V": int(Vv)}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    results_vol.commit()
    return meta


BASE = dict(T=2, n_c=8, K_new=12, seed=7_000_000, seed2=8_000_000, cseed=9_000_000, rows=25, total_s=530)
CTRL = {  # chunk sizes as freeprobe (truth 8, digits 6), so the warm-up batches are the same
    "ctrlprobe_smoke": dict(BASE, task="truth", K0=64, chunk=8, max_items=16, max_chunks=1),
    "ctrlprobe": dict(BASE, task="truth", K0=64, chunk=8),
    "ctrlprobe_digits_smoke": dict(BASE, task="digits", K0=96, chunk=6, max_items=12, max_chunks=1),
    "ctrlprobe_digits": dict(BASE, task="digits", K0=96, chunk=6),
}


@app.local_entrypoint()
def main(mode: str = "ctrlprobe_smoke", probs: str = ""):
    cfg = dict(CTRL[mode])
    pj = json.load(open(probs))
    if cfg.get("max_items"):
        pj = pj[: cfg.pop("max_items")]
    assert len({p["uid"] for p in pj}) == len(pj)
    here = os.path.dirname(os.path.abspath(__file__))
    sha = lambda f: hashlib.sha256(open(os.path.join(here, f), "rb").read()).hexdigest()[:16]
    cfg["code_sha256"] = sha("ctrlprobe_app.py"); cfg["intervene_sha256"] = sha("intervene_app.py")
    run_id = f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    print(run_id, len(pj), "items", cfg)
    print(json.dumps(ctrlprobe.remote(json.dumps(pj), cfg, run_id), indent=1))
