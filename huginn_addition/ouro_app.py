"""Read-space self-gain in a second looped LM: Ouro-1.4B (ByteDance; 24 shared layers looped 4 times; Apache-2.0).
Generalization test of the Huginn ctrlprobe result (docs/freeloop.pdf; ctrlprobe_app.py). Analysis: analyze_ouro.py.

Ouro's loop map (modeling_ouro.py, OuroModel.forward): h_{k+1} = F(h_k) = norm(L(h_k)), L = the 24 decoder layers
(pre-norm + sandwich norm on each sublayer output, residual add), norm = the model's final RMSNorm, applied after every
loop; h_0 = token embeddings, no input re-injection; logits at step k = lm_head(h_k) (linear; default exit = last step).
So the read covectors of a state are exactly unembedding contrasts. One loop is computed by the model's own forward with
inputs_embeds = h and total_ut_steps = 1 (gate: 4 such applications from the embeddings reproduce the model's 4 states).

Per item (truth judgments, prompt_B of results/freeprobe-20261107-130520/items.json, q = last token): states h_0..h_4
(fp32, one sequence, no padding). For loops k in cfg["loops"] (k = 3: the map producing the read state h_4; k = 2: the
loop before), the own-position Jacobian J_k = dF/dh at h_k, position q. Unit starts u (family code `fam`):
  0  task: E_True - E_False (normalized; ' True', ' False' single tokens)
  1  C1  n_c random-token contrasts E_a - E_b (non-special tokens, task tokens excluded)
  2  C2  n_c contrasts among the 64 highest-logit non-task tokens at h_4[q]
  3  C3  n_c random combinations E^T xi, xi Gaussian centered on the non-special tokens
  4/5  K Gaussian probes (code 4 for the first 4, 5 for the rest)
  6  C3flip: the C3 directions with random coordinate sign flips (same coordinate energies, alignment randomized): a
     control for coordinate-wise effects such as the final RMSNorm weights
Diagnostics: the final RMSNorm's exact Jacobian N at z = L(h_k)_q (u^T N u, |N^T u|, the diagonal-only sum_i w_i u_i^2 / r;
descriptive, since J = N dL); the actual two-loop pullback J_2^T J_3^T u (not the frozen square); state drift between loops;
adjoint checks at three FD steps (the 0.01 step is the gate).
Chains w_0 = u, w_t = J_k^T w_{t-1}, t = 1, 2 (VJP); saved per loop: gram3 (S, 3, 3), ugram (S, S), x1 = U^T J^T U,
x2 = U^T (J^T)^2 U, adjoint errors (central FD on a random unit r, step 1e-2 |h_k[q]|), cos(u, h_k[q]).
Run from huginn_addition/:  <venv>/bin/modal run ouro_app.py --mode ouro_smoke --probs ../results/freeprobe-20261107-130520/items.json
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import modal

OURO_ID = "ByteDance/Ouro-1.4B"
OURO_REV = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
app = modal.App("ouro-readspace")
hf_cache = modal.Volume.from_name("huginn-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("huginn-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.54.1", "safetensors", "huggingface_hub[hf_transfer]", "numpy")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=900)
def ouroprobe(items_json: str, cfg: dict, run_id: str):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    T0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(OURO_ID, revision=OURO_REV, trust_remote_code=True)
    M = AutoModelForCausalLM.from_pretrained(OURO_ID, revision=OURO_REV, torch_dtype=torch.float32,
                                             trust_remote_code=True).cuda().eval()
    hf_cache.commit()
    dev = torch.device("cuda"); d = M.config.hidden_size; NUT = M.model.total_ut_steps
    assert NUT == 4 and M.config.num_hidden_layers == 24
    E = M.lm_head.weight.detach()                                    # (V, d)
    Vv = E.shape[0]
    TFt = [tok.encode(w_, add_special_tokens=False) for w_ in (" True", " False")]
    assert all(len(t) == 1 for t in TFt), TFt
    TF = [t[0] for t in TFt]
    special = set(int(t) for t in tok.all_special_ids)
    real = np.array([t for t in range(min(Vv, len(tok))) if t not in special])
    allowed = np.array([t for t in real if t not in set(TF)])
    bad = torch.ones(Vv, dtype=torch.bool, device=dev); bad[torch.tensor(allowed, device=dev)] = False
    supp = torch.zeros(Vv, device=dev); supp[torch.tensor(real, device=dev)] = 1.0

    def F(h):
        """one loop, by the model's own forward: norm(layers(h)), h (B, L, d)."""
        M.model.total_ut_steps = 1
        try:
            return M.model(inputs_embeds=h, use_cache=False)[1][0]
        finally:
            M.model.total_ut_steps = NUT

    M.requires_grad_(False)
    normw = M.model.norm.weight.detach().double(); neps = M.model.norm.variance_epsilon
    captured = {}
    hk_ = M.model.norm.register_forward_hook(lambda mod, inp, out: captured.__setitem__("z", inp[0].detach()))
    import transformers
    versions = dict(torch=str(torch.__version__), transformers=str(transformers.__version__), attn=str(M.config._attn_implementation))

    items = json.loads(items_json); n = len(items)
    nc, K, T, LO = cfg["n_c"], cfg["K"], 2, list(cfg["loops"])
    assert 3 in LO and 2 in LO
    S = 1 + 4 * nc + K   # task, C1, C2, C3, probes (4 + K - 4), C3flip (code 6) last
    fam = np.array([0] + [1] * nc + [2] * nc + [3] * nc + [4] * 4 + [5] * (K - 4) + [6] * nc)
    i_c3 = np.where(fam == 3)[0]; i_fl = np.where(fam == 6)[0]; i_pr = np.where((fam == 4) | (fam == 5))[0]
    nL = len(LO); EPS = list(cfg["fd_eps"])
    GR3 = np.full((n, nL, S, 3, 3), np.nan); UG = np.full((n, S, S), np.nan); X1 = np.full((n, nL, S, S), np.nan)
    X2 = np.full((n, nL, S, S), np.nan); ADJ = np.full((n, nL, len(EPS), S), np.nan); ADJA = np.full((n, nL, len(EPS), S), np.nan)
    CX = np.full((n, nL, S), np.nan); NSG = np.full((n, nL, S, 3), np.nan)   # norm-only: u^T N u, |N^T u|, diag-only
    ACT2 = np.full((n, S, 2), np.nan)                                          # actual two-loop: <u, J_2^T J_3^T u>, |J_2^T J_3^T u|
    VAL = np.full((n, NUT), np.nan); LOGERR = np.full(n, np.nan); TOK = np.full((n, 2 * nc, 2), -1, np.int64)
    PT = np.full((n, 2), np.nan); HN = np.full((n, NUT + 1), np.nan); DRIFT = np.full((n, NUT, 2), np.nan); DONE = np.zeros(n, bool)
    o = f"/results/{run_id}"; os.makedirs(o, exist_ok=True)
    with open(f"{o}/manifest.json", "w") as f_:
        json.dump(dict(cfg=cfg, run_id=run_id, S=S, fam=fam.tolist(), n_items=n, model=OURO_ID, revision=OURO_REV, d=d, V=int(Vv),
                       TF=TF, n_allowed=int(len(allowed)), versions=versions), f_, indent=1)
    with open(f"{o}/items.json", "w") as f_:
        f_.write(items_json)
    results_vol.commit()
    print(f"setup {time.monotonic() - T0:.0f}s; d = {d}, V = {Vv}, S = {S}; {versions}", flush=True)
    TM = []

    def save(fn, compressed=False):
        arrs = dict(gram3=GR3, ugram=UG, x1=X1, x2=X2, adjoint_rel_err=ADJ, adjoint_abs_err=ADJA, fd_eps=np.array(EPS), cos_h=CX,
                    norm_only=NSG, actual2=ACT2, val_state_rel=VAL, logit_err=LOGERR, p_TF=PT, hnorm=HN, state_drift=DRIFT, tok=TOK,
                    done=DONE, fam=fam, loops=np.array(LO), uid=np.array([r["uid"] for r in items]),
                    truth=np.array([r["truth"] for r in items]))
        (np.savez_compressed if compressed else np.savez)(fn, **arrs)

    for i, it in enumerate(items):
        if time.monotonic() - T0 > cfg["total_s"] or i >= cfg.get("max_items", 10 ** 9):
            print(f"stopping before item {i}", flush=True)
            break
        tc = time.monotonic()
        ids = torch.tensor([tok.encode(it["prompt_B"], add_special_tokens=False)], device=dev)
        Lq = ids.shape[1]; q = Lq - 1
        with torch.no_grad():
            hs = [M.model.embed_tokens(ids)]; zs = [None]
            for k in range(NUT):
                hs.append(F(hs[-1])); zs.append(captured["z"][0, q].double())   # z_{k+1} = L(h_k)_q, the final norm's input
            ref = M.model(input_ids=ids, use_cache=False)[1]           # the model's own 4 states
            VAL[i] = [float((hs[k + 1] - ref[k]).norm() / ref[k].norm()) for k in range(NUT)]
            lg_ref = M(input_ids=ids, use_cache=False, exit_at_step=NUT - 1).logits[0, -1].float()
            lg4 = (hs[NUT][0, q] @ E.T).float()
            LOGERR[i] = float((lg4 - lg_ref).abs().max())
            assert VAL[i].max() < 1e-3 and LOGERR[i] < 1e-2, (VAL[i], LOGERR[i])  # fail fast (the analysis gate is stricter)
            lp = torch.log_softmax(lg4, -1); PT[i] = [float(lp[TF[0]].exp()), float(lp[TF[1]].exp())]
            HN[i] = [float(h_[0, q].norm()) for h_ in hs]
            DRIFT[i] = [[float((hs[k + 1][0, q] - hs[k][0, q]).norm() / hs[k][0, q].norm()),
                         float(torch.nn.functional.cosine_similarity(hs[k + 1][0, q], hs[k][0, q], dim=0))] for k in range(NUT)]
        # ---- directions (context-independent except C2, which uses the logits at h_4)
        uid = int(it["uid"]); rng = np.random.default_rng(cfg["cseed"] + uid)
        c1 = np.array([rng.choice(allowed, 2, replace=False) for _ in range(nc)])
        top = torch.topk(lg4.masked_fill(bad, -float("inf")), 64).indices.cpu().numpy()
        c2 = np.array([top[rng.choice(64, 2, replace=False)] for _ in range(nc)])
        gx = torch.Generator(device=dev).manual_seed(cfg["cseed"] + uid)
        xi = torch.randn((nc, Vv), device=dev, generator=gx) * supp
        xi = xi - supp * (xi.sum(-1, keepdim=True) / supp.sum())
        U = torch.zeros((S, d), device=dev)
        U[0] = E[TF[0]] - E[TF[1]]
        for r_, (a, b) in enumerate(np.concatenate([c1, c2])):
            U[1 + r_] = E[int(a)] - E[int(b)]
        U[i_c3] = xi @ E
        gp = torch.Generator(device=dev).manual_seed(cfg["seed"] + uid)
        U[i_pr] = torch.randn((K, d), device=dev, generator=gp)
        sg_ = torch.randint(0, 2, (nc, d), device=dev, generator=gp).float() * 2 - 1      # coordinate sign flips of C3
        U[i_fl] = U[i_c3] * sg_
        U = U / U.norm(dim=-1, keepdim=True)
        TOK[i] = np.concatenate([c1, c2])
        UG[i] = (U.double() @ U.double().T).cpu().numpy()
        # ---- VJP chains at each requested loop
        W1 = {}
        for li, k in enumerate(LO):
            hk = hs[k].detach()

            def vjp(w, hk=hk):
                R = w.shape[0]; arr = torch.arange(R, device=dev)
                xr = hk.expand(R, -1, -1).clone()
                with torch.enable_grad():
                    xq = xr[:, q].clone().requires_grad_(True)
                    st = xr.index_put((arr, torch.full((R,), q, device=dev)), xq)
                    out = F(st)[:, q]
                    return torch.autograd.grad((out * w).sum(), xq)[0].detach()

            def vjp_all(Win, vjp=vjp):
                nxt = torch.zeros_like(Win)
                for b0 in range(0, S, cfg["rows"]):
                    sl = slice(b0, min(S, b0 + cfg["rows"]))
                    nxt[sl] = vjp(Win[sl])
                return nxt

            Wc = [U]
            for t in range(T):   # frozen-Jacobian powers (J_k^T)^t u
                Wc.append(vjp_all(Wc[-1]))
            W1[k] = (Wc[1], vjp_all)
            with torch.no_grad():  # adjoint check: (J^T u) . r vs u . (J r), J r by central FD, steps eps * |h_k[q]|
                r_ = torch.randn((S, d), device=dev, generator=torch.Generator(device=dev).manual_seed(9))
                r_ = r_ / r_.norm(dim=-1, keepdim=True)
                for ei, eps in enumerate(EPS):
                    ep = eps * float(hk[0, q].norm()); jr = torch.zeros_like(U)
                    for b0 in range(0, S, cfg["rows"]):
                        sl = slice(b0, min(S, b0 + cfg["rows"])); R = sl.stop - sl.start
                        sp = hk.expand(R, -1, -1).clone(); sm = sp.clone()
                        sp[:, q] += ep * r_[sl]; sm[:, q] -= ep * r_[sl]
                        jr[sl] = (F(sp)[:, q] - F(sm)[:, q]) / (2 * ep)
                    lhs = (Wc[1] * r_).sum(-1); rhs = (U * jr).sum(-1)
                    ADJ[i, li, ei] = ((lhs - rhs).abs() / (lhs.abs() + rhs.abs()).clamp_min(1e-12)).cpu().numpy()
                    ADJA[i, li, ei] = (lhs - rhs).abs().cpu().numpy()
                Cv = torch.stack(Wc, 1).double()
                GR3[i, li] = torch.bmm(Cv, Cv.transpose(1, 2)).cpu().numpy()
                X1[i, li] = (U.double() @ Wc[1].double().T).cpu().numpy(); X2[i, li] = (U.double() @ Wc[2].double().T).cpu().numpy()
                hq = hk[0, q].double(); CX[i, li] = ((U.double() @ hq) / hq.norm()).cpu().numpy()
                # final-norm diagnostic at z = L(h_k)_q: N = diag(w)/r - (w * z) z^T / (d r^3), r = sqrt(|z|^2/d + eps)
                zq = zs[k + 1]; rr = torch.sqrt((zq ** 2).mean() + neps); Ud = U.double()
                Nu = Ud * normw / rr - ((Ud @ zq)[:, None] * (normw * zq)[None] / (d * rr ** 3))   # rows: (N u)^T
                NTu = Ud * normw / rr - ((Ud @ (normw * zq))[:, None] * zq[None] / (d * rr ** 3))  # rows: (N^T u)^T
                NSG[i, li] = torch.stack([(Ud * Nu).sum(-1), NTu.norm(dim=-1), (Ud ** 2 * normw).sum(-1) / rr], -1).cpu().numpy()
        A2 = W1[2][1](W1[3][0])     # actual two-loop pullback J_2^T J_3^T u (h_4 back to h_2)
        ACT2[i] = torch.stack([(U * A2).sum(-1), A2.norm(dim=-1)], -1).double().cpu().numpy()
        DONE[i] = True
        TM.append(time.monotonic() - tc)
        if cfg.get("max_items") or i % 8 == 7 or i == n - 1:
            save(f"{o}/partial.npz"); results_vol.commit()
            print(f"items {i + 1}/{n} {time.monotonic() - T0:.0f}s (item {TM[-1]:.1f}s, peak "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB); val {np.nanmax(VAL[: i + 1]):.1e}, logit {np.nanmax(LOGERR[: i + 1]):.1e}", flush=True)
    hk_.remove()
    save(f"{o}/ouroprobe.npz", compressed=True)
    meta = dict(cfg=cfg, run_id=run_id, runtime_s=time.monotonic() - T0, model=OURO_ID, revision=OURO_REV, S=S, d=d,
                complete=bool(DONE.all()), item_s=TM, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, versions=versions)
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    results_vol.commit()
    return meta


OURO = {
    "ouro_smoke": dict(n_c=8, K=16, loops=[2, 3], seed=7_000_000, cseed=9_000_000, rows=25, total_s=780, fd_eps=[0.003, 0.01, 0.03],
                       max_items=3, smoke=True),
    "ouro": dict(n_c=8, K=16, loops=[2, 3], seed=7_000_000, cseed=9_000_000, rows=25, total_s=780, fd_eps=[0.003, 0.01, 0.03]),
}


@app.local_entrypoint()
def main(mode: str = "ouro_smoke", probs: str = ""):
    cfg = dict(OURO[mode])
    pj = json.load(open(probs))
    assert len({p["uid"] for p in pj}) == len(pj)
    here = os.path.dirname(os.path.abspath(__file__))
    cfg["code_sha256"] = hashlib.sha256(open(os.path.join(here, "ouro_app.py"), "rb").read()).hexdigest()[:16]
    run_id = f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    print(run_id, len(pj), "items", cfg)
    print(json.dumps(ouroprobe.remote(json.dumps(pj), cfg, run_id), indent=1))
