"""Oversight-dynamics pilot (design: docs/oversight_pilot.md; theory: docs/observability.pdf). float64; RMSNorm
arithmetic patched to float64 (patch_norms); D pinned to SDPA attention.

Models: the retrofitted Recurrent-Llama-3.2 family (McLeish et al., arXiv 2511.07384) at mean training depths M = 4, 8,
16, 32, and its feedforward partner Llama-3.2-non-recurrent-posttrained (same data, same recipe).

Loop harness for the retrofitted (older Huginn) code. The model's own `iterate_one_step` refers to `self.freqs_cis`,
which this code version never creates, so one loop is re-implemented here, copied from RavenForCausalLM.forward:
  prelude:  e = prelude(wte(ids) * emb_scale), rotary (cos, sin) = model.rotary_emb(e, positions), no mask (causal SDPA)
  one loop: x <- core_block(adapter([x, e]))                      (test_time_noise = 0, asserted)
  readout:  logits = lm_head(ln_f(coda(x)))                       (forward applies ln_f once, after the coda)
Gate G1 checks it against model(ids, input_states=x0, num_steps=r) for r in {1, 4, 32}.

Part A (looped models only): A0 base trajectory, A1 start sensitivity (fresh starts from the training distribution),
A2 local perturbation response at depths r in {M, 32, 64}. Part C (all models): steering at the layer-4 output, position
q; linear-prediction error versus dose. Every comparison is made against an unperturbed row in the same batch; the
batch-versus-single-row difference is recorded as a noise floor.

Run from huginn_addition/:  <venv>/bin/modal run obs_app.py --mode smoke
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

import modal

MODELS = {
    "L4": ("smcleish/Recurrent-Llama-3.2-train-recurrence-4", "c466903f9ab35d827438d6a3fffcca95c47a5c1e", 4),
    "L8": ("smcleish/Recurrent-Llama-3.2-train-recurrence-8", "45f95e3b88202225e9f948db4441a188b38e7d15", 8),
    "L16": ("smcleish/Recurrent-Llama-3.2-train-recurrence-16", "157b949fdb897c0476551b8d27978310cd991268", 16),
    "L32": ("smcleish/Recurrent-Llama-3.2-train-recurrence-32", "a5f6f126e9d9302445346844f1d7d29d111a06eb", 32),
    "D": ("smcleish/Llama-3.2-non-recurrent-posttrained", "004e96ace360c1a37c08f4555227136e845087b3", 0),
}
BOS = 128000
app = modal.App("oversight-pilot")
hf_cache = modal.Volume.from_name("huginn-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("huginn-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.53.1", "safetensors", "huggingface_hub[hf_transfer]", "numpy")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def make_prompts(n2: int, n3: int, seed: int):
    """Four demonstrations 'a + b = c' and a query 'a + b = ' (trailing space = its own token); all sums < 1000."""
    rng = random.Random(seed)

    def prob(nd):
        while True:
            a, b = rng.randint(10 ** (nd - 1), 10 ** nd - 1), rng.randint(10 ** (nd - 1), 10 ** nd - 1)
            if a + b < 1000:
                return a, b

    groups = {}
    for nd, cnt in ((2, n2), (3, n3)):
        groups[nd] = []
        for _ in range(cnt):
            demos = [prob(nd) for _ in range(4)]
            a, b = prob(nd)
            text = "".join(f"{x} + {y} = {x + y}\n" for x, y in demos) + f"{a} + {b} = "
            groups[nd].append(dict(nd=nd, a=a, b=b, ans=a + b, prompt=text))
    items = []
    for j in range(max(n2, n3)):        # interleaved 2-digit / 3-digit, so any prefix is balanced
        for nd in (2, 3):
            if j < len(groups[nd]):
                items.append(dict(uid=len(items), **groups[nd][j]))
    return items


FAM_ORDER = ("sub", "add3", "mul1", "mul2")


def make_hard_candidates(n_per_family: int, seed: int):
    """Part 2 candidate pool (docs/oversight_hard.md): four families, four same-family demonstrations then the query,
    ending '= '. Every answer is below 1000 (one token). Queries are distinct within the pool, and no demonstration
    repeats its own query."""
    rng = random.Random(seed)

    def borrow(a, b):
        return any(int(x) < int(y) for x, y in zip(f"{a:03d}", f"{b:03d}"))

    def prob(fam):
        while True:
            if fam == "sub":
                a, b = rng.randint(100, 999), rng.randint(100, 999)
                if a > b and borrow(a, b):
                    return (a, b), a - b, f"{a} - {b} = "
            elif fam == "add3":
                a, b, c = (rng.randint(10, 99) for _ in range(3))
                return (a, b, c), a + b + c, f"{a} + {b} + {c} = "
            elif fam == "mul1":
                a, b = rng.randint(12, 99), rng.randint(3, 9)
                return (a, b), a * b, f"{a} * {b} = "
            else:
                a, b = rng.randint(11, 99), rng.randint(11, 99)
                if a * b < 1000:
                    return (a, b), a * b, f"{a} * {b} = "

    items, seen = [], set()
    for fam in FAM_ORDER:
        k = 0
        while k < n_per_family:
            ops, ans, q = prob(fam)
            if q in seen:
                continue
            seen.add(q)
            demos = []
            while len(demos) < 4:                    # a demonstration identical to the query is rejected and redrawn
                _, da, dq = prob(fam)
                if dq != q:
                    demos.append(f"{dq}{da}\n")
            demos = "".join(demos)
            assert ans < 1000
            items.append(dict(cid=len(items), fam=fam, ops=list(ops), ans=ans, prompt=demos + q))
            k += 1
    return items


def select_hard(items, res, n_each=8, thresh=0.9):
    """The fixed selection rule of docs/oversight_hard.md, Step 3. res[key] has p and top1 (n x depths), depths, ans.
    Uses r = 64 for looped models and D's single pass. Returns the selected items (uid 0..2*n_each-1) and the strata."""
    import numpy as np
    n = len(items); keys = sorted(res)
    P = np.zeros((n, len(keys))); C = np.zeros((n, len(keys)), bool); T = np.zeros((n, len(keys)), int)
    for j, k in enumerate(keys):
        col = 0 if k == "D" else list(res[k]["depths"]).index(64)
        P[:, j] = np.asarray(res[k]["p"], float)[:, col]
        T[:, j] = np.asarray(res[k]["top1"])[:, col]
        C[:, j] = T[:, j] == np.asarray(res[k]["ans"])
    assert np.isfinite(P).all(), "screen gave a non-finite probability"
    W = ~C.all(1); U = C.all(1) & (P.min(1) < thresh); E = C.all(1) & (P.min(1) >= thresh)

    def rr(mask, k):
        qs = {f: [i for i in range(n) if mask[i] and items[i]["fam"] == f] for f in FAM_ORDER}
        out = []
        while len(out) < k and any(qs.values()):
            for f in FAM_ORDER:
                if qs[f] and len(out) < k:
                    out.append(qs[f].pop(0))
        return out

    total = 2 * n_each
    w, u = rr(W, n_each), rr(U, n_each)
    if len(w) < n_each:
        u = rr(U, total - len(w))
    elif len(u) < n_each:
        w = rr(W, total - len(u))
    e = rr(E, total - len(w) - len(u))
    order = []
    for t in range(max(len(w), len(u))):
        order += [("W", w[t])] if t < len(w) else []
        order += [("U", u[t])] if t < len(u) else []
    order += [("E", i) for i in e]
    sel = []
    for uid, (st, i) in enumerate(order):
        it = dict(items[i], uid=uid, stratum=st,
                  screen={k: dict(p=float(P[i, j]), correct=bool(C[i, j]), top1=int(T[i, j])) for j, k in enumerate(keys)})
        sel.append(it)
    counts = dict(W=int(W.sum()), U=int(U.sum()), E=int(E.sum()), selected_W=len(w), selected_U=len(u), selected_E=len(e))
    return sel, counts


def verify_selection(prompts):
    """The hard run's prompts must be the screen's selection: check the files against the screen manifest, regenerate
    the candidate pool, and re-derive the selection from the saved screen results. Returns (items, manifest)."""
    raw = open(prompts, "rb").read()
    items = json.loads(raw)
    assert len(items) == 16 and [it["uid"] for it in items] == list(range(16)), "need the 16 selected prompts"
    sdir = os.path.dirname(os.path.abspath(prompts))
    man = json.load(open(os.path.join(sdir, "screen_manifest.json")))
    sha = lambda fn: hashlib.sha256(open(os.path.join(sdir, fn), "rb").read()).hexdigest()[:16]
    for fn, k in (("candidates.json", "candidates_sha256"), ("screen_results.json", "screen_results_sha256"),
                  ("hard_prompts.json", "hard_prompts_sha256")):
        assert sha(fn) == man[k], f"{fn} does not match the screen manifest"
    assert hashlib.sha256(raw).hexdigest()[:16] == man["hard_prompts_sha256"], "prompts file is not the screen's selection"
    cands = json.load(open(os.path.join(sdir, "candidates.json")))
    assert cands == make_hard_candidates(man["cfg"]["n_per_family"], man["cfg"]["cand_seed"]), "candidate pool differs"
    resel, _ = select_hard(cands, json.load(open(os.path.join(sdir, "screen_results.json"))))
    assert json.dumps(resel, sort_keys=True) == json.dumps(items, sort_keys=True), "selection does not reproduce"
    return items, man


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=480)
def screen(key: str, items_json: str, cfg: dict, run_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    T0 = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hf_id, rev, _ = MODELS[key]
    tok = AutoTokenizer.from_pretrained(hf_id, revision=rev)
    extra = dict(attn_implementation="sdpa") if key == "D" else dict(trust_remote_code=True)
    M = AutoModelForCausalLM.from_pretrained(hf_id, revision=rev, torch_dtype=getattr(torch, cfg["dtype"]),
                                             **extra).to("cuda").eval()
    try:
        hf_cache.commit()
    except Exception as ex:
        print(f"[{key}] cache commit skipped: {ex!r}", flush=True)
    o = f"/results/{run_id}/{key}"; os.makedirs(o, exist_ok=True)
    meta = run_screen(key, M, tok, json.loads(items_json), cfg, o, torch.device("cuda"), results_vol.commit, T0)
    meta.update(run_id=run_id, hf_id=hf_id, revision=rev)
    return meta


def run_screen(key, M, tok, items, cfg, o, dev, commit, T0):
    """Difficulty screen: unperturbed answer statistics only (no nudges, pushes or gradients). Looped models use their
    own forward (gate G1 of part 1: equal to the harness to 1e-15). Plain function so that it can be tested on CPU."""
    import numpy as np
    import torch

    Mdep = MODELS[key][2]; looped = key != "D"
    M.requires_grad_(False)
    DT = next(M.parameters()).dtype
    n_norms = patch_norms(M) if DT == torch.float64 else 0
    if not looped:
        assert M.config._attn_implementation == "sdpa", M.config._attn_implementation
    depths = sorted({Mdep, 32, 64}) if looped else [0]
    IDS, ANS = [], []
    for it in items:
        ids = [BOS] + tok.encode(it["prompt"], add_special_tokens=False)
        at = tok.encode(str(it["ans"]), add_special_tokens=False)
        assert len(at) == 1, (it, at)
        assert tok.decode(ids[-1:]) == " ", tok.decode(ids[-3:])
        IDS.append(ids); ANS.append(at[0])
    n, nD = len(items), len(depths)
    R = dict(p=np.full((n, nD), np.nan), logp=np.full((n, nD), np.nan), top1=np.full((n, nD), -1),
             ent=np.full((n, nD), np.nan), gap=np.full((n, nD), np.nan))
    d = M.config.n_embd if looped else M.config.hidden_size
    STD = float(M.config.init_values["std"]) if looped else None
    groups = {}
    for i, ids in enumerate(IDS):
        groups.setdefault(len(ids), []).append(i)
    B = cfg.get("screen_batch", 16)
    deadline = T0 + cfg["total_s"] - cfg["reserve_s"]
    status, err = "complete", None
    import inspect
    try:
        fwd_sha = hashlib.sha256(inspect.getsource(type(M).forward).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        fwd_sha = None
    logit_dtype = []
    try:
        with torch.no_grad():
            for L, idx in sorted(groups.items()):
                for s0 in range(0, len(idx), B):
                    if time.monotonic() > deadline:
                        raise TimeoutError("deadline")
                    bi = idx[s0:s0 + B]
                    ids = torch.tensor([IDS[i] for i in bi], device=dev)
                    if looped:
                        xs = []
                        for i in bi:
                            g = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 500_000 + items[i]["cid"])
                            x = torch.empty((1, L, d))
                            torch.nn.init.trunc_normal_(x, mean=0.0, std=STD, a=-3 * STD, b=3 * STD, generator=g)
                            xs.append(x)
                        x0 = torch.cat(xs).to(dev, DT)
                        Zs = [M(ids, input_states=x0, num_steps=r).logits[:, -1] for r in depths]
                    else:
                        Zs = [M(ids, use_cache=False, logits_to_keep=1).logits[:, -1]]
                    logit_dtype.append(str(Zs[0].dtype)); Zs = [Z.double() for Z in Zs]
                    ans = torch.tensor([ANS[i] for i in bi], device=dev)
                    for di, Z in enumerate(Zs):
                        lp = torch.log_softmax(Z, -1); la = lp.gather(1, ans[:, None])[:, 0]; tv, ti = Z.topk(2, dim=-1)
                        ent = -(lp.exp() * lp).sum(-1)
                        for j, i in enumerate(bi):
                            R["logp"][i, di] = float(la[j]); R["p"][i, di] = float(la[j].exp())
                            R["top1"][i, di] = int(ti[j, 0]); R["ent"][i, di] = float(ent[j])
                            R["gap"][i, di] = float(tv[j, 0] - tv[j, 1])
    except TimeoutError:
        status = "deadline: screen incomplete"
    except Exception:
        import traceback
        status, err = "error", traceback.format_exc()
        print(f"[{key}] ERROR\n{err}", flush=True)
    if status == "complete" and not np.isfinite(R["p"]).all():
        status = "gate: non-finite probability"
    np.savez_compressed(f"{o}/screen.npz", **R, ans=np.array(ANS), depths=np.array(depths), cid=np.array([it["cid"] for it in items]))
    fam = [it["fam"] for it in items]
    col = depths.index(64) if looped else 0
    acc = {f: float(np.mean([R["top1"][i, col] == ANS[i] for i in range(n) if fam[i] == f])) for f in FAM_ORDER if f in fam}
    meta = dict(key=key, status=status, error=err, runtime_s=time.monotonic() - T0, depths=depths, n=n, acc_by_family=acc,
                norms_patched_fp64=n_norms, param_dtype=str(DT), logit_dtypes=sorted(set(logit_dtype)),
                forward_source_sha256=fwd_sha, cfg=cfg, hf_id=MODELS[key][0], revision=MODELS[key][1],
                p=R["p"].tolist(), top1=R["top1"].tolist(), ans=ANS)
    with open(f"{o}/meta.json", "w") as f_:
        json.dump({k: v for k, v in meta.items() if k not in ("p", "top1")}, f_, indent=1)
    commit()
    if status == "error":
        raise RuntimeError(f"[{key}] {err}")
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=480)
def obs(key: str, items_json: str, cfg: dict, run_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    T0 = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    hf_id, rev, _ = MODELS[key]
    tok = AutoTokenizer.from_pretrained(hf_id, revision=rev)
    extra = dict(attn_implementation="sdpa") if key == "D" else dict(trust_remote_code=True)   # eager Llama softmaxes in fp32
    M = AutoModelForCausalLM.from_pretrained(hf_id, revision=rev, torch_dtype=getattr(torch, cfg["dtype"]),
                                             **extra).to("cuda").eval()
    try:
        hf_cache.commit()
    except Exception as ex:  # concurrent containers may race on the cache; the run does not depend on it
        print(f"[{key}] cache commit skipped: {ex!r}", flush=True)
    o = f"/results/{run_id}/{key}"; os.makedirs(o, exist_ok=True)
    meta = run_obs(key, M, tok, json.loads(items_json), cfg, o, torch.device("cuda"), results_vol.commit, T0)
    meta.update(run_id=run_id, hf_id=hf_id, revision=rev)
    return meta


def patch_norms(M):
    """Both RMSNorm classes (Raven's and transformers' Llama) cast to float32 internally. For float64 inputs compute the
    same formula, w * x / sqrt(mean(x^2) + eps), in float64; other dtypes keep the original code. Returns the count."""
    import torch
    n = 0
    for mod in M.modules():
        cls = type(mod)
        if cls.__name__ not in ("RMSNorm", "LlamaRMSNorm"):
            continue
        n += 1
        if getattr(cls, "_orig_forward", None) is None:
            cls._orig_forward = cls.forward

            def fwd(self, x):
                if x.dtype != torch.float64:
                    return type(self)._orig_forward(self, x)
                eps = self.eps if hasattr(self, "eps") else self.variance_epsilon
                return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
            cls.forward = fwd
    return n


def run_obs(key, M, tok, items, cfg, o, dev, commit, T0):
    """All measurements for one model (looped unless key == 'D'). Plain function so that it can be tested on CPU."""
    import numpy as np
    import torch
    import transformers

    Mdep = MODELS[key][2]
    looped = key != "D"
    M.requires_grad_(False)
    DT = next(M.parameters()).dtype
    n_norms = patch_norms(M) if DT == torch.float64 else 0
    versions = dict(torch=str(torch.__version__), transformers=str(transformers.__version__))
    n_all = len(items)
    n = min(n_all, cfg.get("max_items", n_all))
    peak = (lambda: torch.cuda.max_memory_allocated() / 2 ** 30) if dev.type == "cuda" else (lambda: 0.0)

    # ---- tokenization (identical rule for every model; ids saved for a cross-model check)
    IDS, ANS = [], []
    for it in items:
        ids = [BOS] + tok.encode(it["prompt"], add_special_tokens=False)
        at = tok.encode(str(it["ans"]), add_special_tokens=False)
        assert len(at) == 1, (it, at)
        assert tok.decode(ids[-1:]) == " ", tok.decode(ids[-3:])
        IDS.append(ids); ANS.append(at[0])
    d = M.config.hidden_size if not looped else M.config.n_embd

    # ---- model-specific pieces: site(ids) -> layer-4 output (1, L, d); run(ids, push, x0, r) -> logits at q
    if looped:
        cfgm = M.config
        assert cfgm.test_time_noise == 0 and cfgm.injection_type == "linear" and M.emb_scale == 1.0
        assert len(M.transformer.prelude) == 4 and len(M.transformer.core_block) == 6 and len(M.transformer.coda) == 4
        STD = float(cfgm.init_values["std"])
        DUM = torch.tensor(0, dtype=torch.long)

        def prep(ids):
            e = M.transformer.wte(ids)
            fc = M.rotary_emb(e, torch.arange(ids.shape[1], device=dev)[None])
            bi = torch.tensor(-1, dtype=torch.long)
            for blk in M.transformer.prelude:
                bi += 1
                e = blk(e, fc, bi, None, None)
            return e, fc

        def step(x, e, fc):
            x = M.transformer.adapter(torch.cat([x, e.expand(x.shape[0], -1, -1)], dim=-1))
            for blk in M.transformer.core_block:
                x = blk(x, fc, DUM, None, None)
            return x

        def read(x, fc):
            for blk in M.transformer.coda:
                x = blk(x, fc, DUM, None, None)
            return M.lm_head(M.transformer.ln_f(x[:, -1])).double()

        def start(B, L, seed):
            g = torch.Generator(device="cpu").manual_seed(seed)
            x = torch.empty((B, L, d))
            torch.nn.init.trunc_normal_(x, mean=0.0, std=STD, a=-3 * STD, b=3 * STD, generator=g)
            return x.to(dev, DT)
    else:
        assert M.config._attn_implementation == "sdpa", M.config._attn_implementation   # eager softmaxes in float32
        layer = M.model.layers[3]
        hook_state = {"delta": None, "cap": None}

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hook_state["cap"] = h.detach()
            if hook_state["delta"] is not None:
                h = h + hook_state["delta"]
            return ((h,) + tuple(out[1:])) if isinstance(out, tuple) else h

        layer.register_forward_hook(hook)

    # ---- site statistics over all prompts (non-BOS tokens): per-coordinate std s and RMS norm rho
    with torch.no_grad():
        acts = []
        for ids in IDS:
            t_ = torch.tensor([ids], device=dev)
            if looped:
                acts.append(prep(t_)[0][0, 1:])
            else:
                hook_state["delta"] = None
                M(t_, use_cache=False)
                acts.append(hook_state["cap"][0, 1:])
        A = torch.cat(acts).double()
        s_site = A.std(0).to(DT); rho = float(A.norm(dim=-1).pow(2).mean().sqrt())
        s_safe = s_site.clamp_min(1e-6 * float(s_site.max()))
        We = M.transformer.adapter.weight[:, d:] if looped else None      # adapter columns acting on the prelude output

    T, K = cfg["T"], cfg["K"]
    SNAP = [1, 2, 4, 8, 16, 32, 64]
    RA = sorted({Mdep, 32, 64}) if looped else []
    RC = sorted({Mdep, 32}) if looped else [0]
    SIG = cfg["sig"]; NDIR_A = cfg["ndir_a"]; NST = cfg["n_starts"]; NS = len(SIG)
    EPS = cfg["eps"]; NR = cfg["n_rand"]; NDIR_C = NR + 2; NE = len(EPS)
    i0, i1 = EPS.index(1e-3), EPS.index(3e-3)
    nA, nC, nR, nRC = (n if looped else 0), n, len(RA), len(RC)
    f64 = lambda *s: np.full(s, np.nan)
    R = dict(
        # A0 / A1: rows of the batch are [x0, x0 (duplicate), NST fresh starts]
        a0_logp=f64(nA, T + 1), a0_top1=np.full((nA, T + 1), -1), a0_ent=f64(nA, T + 1), a0_xn_all=f64(nA, T + 1),
        a0_xn_q=f64(nA, T + 1), a0_res_all=f64(nA, T), a0_res_q=f64(nA, T),
        a0_floor=f64(nA, T + 1), a0_floor_fish=f64(nA, T + 1), a1_dup_fish=f64(nA, T + 1),
        xq_snap=np.full((nA, len(SNAP), d), np.nan, np.float32),
        a1_kl=f64(nA, NST, T + 1), a1_fish=f64(nA, NST, T + 1), a1_agree=f64(nA, NST, T + 1), a1_dx_all=f64(nA, NST, T + 1),
        a1_dx_q=f64(nA, NST, T + 1),
        # A2: rows [x_r, x_r (duplicate), sigma_0 x dirs, sigma_1 x dirs]; the same NDIR_A directions at every sigma
        a2_fish=f64(nA, nR, NS, NDIR_A, K + 1), a2_kl=f64(nA, nR, NS, NDIR_A, K + 1),
        a2_dx_all=f64(nA, nR, NS, NDIR_A, K + 1), a2_dx_q=f64(nA, nR, NS, NDIR_A, K + 1),
        a2_dup_fish=f64(nA, nR, K + 1), a2_floor_fish=f64(nA, nR, K + 1),
        a2_base_res=f64(nA, nR), a2_base_res_q=f64(nA, nR), a2_xr_norm=f64(nA, nR),
        a2_base_drift=f64(nA, nR, K + 1), a2_base_kl=f64(nA, nR, K + 1),
        # C: rows [no push, no push (duplicate), NDIR_C dirs x NE doses]
        c_kl=f64(nC, nRC, NDIR_C, NE), c_fish=f64(nC, nRC, NDIR_C, NE),
        c_linerr=f64(nC, nRC, NDIR_C, NE), c_dlogp=f64(nC, nRC, NDIR_C, NE),
        c_dlogp_pred=f64(nC, nRC, NDIR_C, NE), c_flip=f64(nC, nRC, NDIR_C, NE),
        c_fdcheck=f64(nC, nRC, NDIR_C), c_gradcheck=f64(nC, nRC, NDIR_C), c_push_euclid=f64(nC, nRC, NDIR_C),
        c_adapter_gain=f64(nC, nRC, NDIR_C), c_gradnorm=f64(nC, nRC), c_base_logp=f64(nC, nRC),
        c_base_top1=np.full((nC, nRC), -1), c_floor=f64(nC, nRC), c_floor_fish=f64(nC, nRC), c_dup_fish=f64(nC, nRC),
        g1=f64(min(n, 2), 3, 2), g2=f64(min(n, 2), 2), doneA=np.zeros(n, bool), doneC=np.zeros(n, bool),
    )

    def mets(Z):
        """Z (B, V) logits, row 0 = reference: Fisher norm^2 of Z - Z[0] under pi_0, KL(pi_0 || pi), top-1, log-probs."""
        Zd = Z.double(); lp = torch.log_softmax(Zd, -1); p0 = lp[0].exp(); Dz = Zd - Zd[0:1]
        fish = (p0 * Dz ** 2).sum(-1) - (p0 * Dz).sum(-1) ** 2
        kl = (p0 * (lp[0:1] - lp)).sum(-1)
        return fish, kl, Zd.argmax(-1), lp, Dz, p0

    def hnorm(v, p0):
        return ((p0 * v ** 2).sum(-1) - (p0 * v).sum(-1) ** 2).clamp_min(0).sqrt()

    def ratio(a, b):
        """a / b with NaN where b is not positive (no silent regularization)."""
        return torch.where(b > 0, a / torch.where(b > 0, b, torch.ones_like(b)), torch.full_like(a, float("nan")))

    def save(fn, compressed=False):
        (np.savez_compressed if compressed else np.savez)(fn, **R, ans=np.array(ANS[:n]), s_site=s_site.cpu().numpy(),
                                                          rho=rho, RA=np.array(RA), RC=np.array(RC), SNAP=np.array(SNAP),
                                                          SIG=np.array(SIG), EPS=np.array(EPS))

    numerics = dict(param_dtype=str(DT), norms_patched_fp64=n_norms, tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
                    tf32_cudnn=bool(torch.backends.cudnn.allow_tf32), matmul_precision=torch.get_float32_matmul_precision(),
                    sdp_flash=bool(torch.backends.cuda.flash_sdp_enabled()), sdp_mem_eff=bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                    sdp_math=bool(torch.backends.cuda.math_sdp_enabled()),
                    attn_impl=str(getattr(M.config, "_attn_implementation", "sdpa (Raven code calls SDPA directly)")))
    npar = lambda m: int(sum(p_.numel() for p_ in m.parameters() if p_.dim() > 1))   # matmul weights (for FLOPs)
    if looped:
        tr = M.transformer
        param_counts = dict(prelude=npar(tr.prelude), core_block=npar(tr.core_block), coda=npar(tr.coda),
                            adapter=npar(tr.adapter), lm_head=npar(M.lm_head))
    else:
        param_counts = dict(layers=npar(M.model.layers), lm_head=npar(M.lm_head))
    with open(f"{o}/manifest.json", "w") as f_:
        json.dump(dict(cfg=cfg, key=key, model=MODELS[key], M=Mdep, d=d, rho=rho, RA=RA, RC=RC, ids=IDS, ans=ANS,
                       versions=versions, numerics=numerics, param_counts=param_counts), f_, indent=1)
    commit()
    print(f"[{key}] setup {time.monotonic() - T0:.0f}s, d {d}, rho {rho:.3f}, RA {RA}, RC {RC}, {versions}", flush=True)
    TM = []
    deadline = T0 + cfg["total_s"] - cfg["reserve_s"]

    class Stop(Exception):
        pass

    def check():
        if time.monotonic() > deadline:
            raise Stop()

    def one_prompt(i):
        ids = torch.tensor([IDS[i]], device=dev); L = ids.shape[1]; q = L - 1; ans = ANS[i]; uid = items[i]["uid"]
        with torch.no_grad():
            if looped:
                e, fc = prep(ids)
                x0 = start(1, L, cfg["seed"] + uid)
                # ---- G1: harness vs the model's own forward (first two items)
                if i < 2:
                    for gi, r in enumerate((1, 4, 32)):
                        out = M(ids, input_states=x0, num_steps=r)
                        x = x0
                        for _ in range(r):
                            x = step(x, e, fc)
                        z = read(x, fc)[0]; zr = out.logits[0, q].double()
                        R["g1"][i, gi] = [float((z - zr).abs().max() / zr.abs().max()),
                                          float((x - out.latent_states).norm() / out.latent_states.norm())]
                    assert np.nanmax(R["g1"][i]) < 1e-4, ("G1 failed", R["g1"][i])
                # ---- A0 (single row) and A1 (rows: x0, x0 duplicate, fresh starts), t = 0..T
                X1 = x0; XB = torch.cat([x0, x0, start(NST, L, cfg["seed"] + 100_000 + uid)])
                keep = {}
                for t in range(T + 1):
                    if t % 16 == 15:
                        check()
                    z1 = read(X1, fc)[0]; ZB = read(XB, fc)
                    fish, kl, top, lp, _, p0 = mets(ZB)
                    R["a0_floor"][i, t] = float((z1 - ZB[0]).abs().max() / z1.abs().max())
                    R["a0_floor_fish"][i, t] = float(hnorm(z1.double() - ZB[0].double(), p0) ** 2)
                    R["a1_dup_fish"][i, t] = float(fish[1])
                    lp1 = torch.log_softmax(z1.double(), -1)
                    R["a0_logp"][i, t] = float(lp1[ans]); R["a0_top1"][i, t] = int(z1.argmax())
                    R["a0_ent"][i, t] = float(-(lp1.exp() * lp1).sum())
                    R["a0_xn_all"][i, t] = float(X1.norm()); R["a0_xn_q"][i, t] = float(X1[0, q].norm())
                    R["a1_kl"][i, :, t] = kl[2:].cpu().numpy(); R["a1_fish"][i, :, t] = fish[2:].cpu().numpy()
                    R["a1_agree"][i, :, t] = (top[2:] == top[0]).float().cpu().numpy()
                    dX = XB[2:] - XB[0:1]
                    R["a1_dx_all"][i, :, t] = (dX.flatten(1).norm(dim=1) / XB[0].norm()).cpu().numpy()
                    R["a1_dx_q"][i, :, t] = (dX[:, q].norm(dim=-1) / XB[0, q].norm()).cpu().numpy()
                    if t in SNAP:
                        R["xq_snap"][i, SNAP.index(t)] = X1[0, q].cpu().numpy()
                    if t in RA:
                        keep[t] = XB[0:1].clone()
                    if t < T:
                        X1n = step(X1, e, fc)
                        R["a0_res_all"][i, t] = float((X1n - X1).norm() / X1.norm())
                        R["a0_res_q"][i, t] = float((X1n[0, q] - X1[0, q]).norm() / X1[0, q].norm())
                        X1 = X1n; XB = step(XB, e, fc)
                # ---- A2: per-position relative perturbations at depth r (one direction bank, reused for every sigma)
                for ri, r in enumerate(RA):
                    check()
                    xr = keep[r]
                    xr1 = step(xr, e, fc)
                    R["a2_base_res"][i, ri] = float((xr1 - xr).norm() / xr.norm())
                    R["a2_base_res_q"][i, ri] = float((xr1[0, q] - xr[0, q]).norm() / xr[0, q].norm())
                    R["a2_xr_norm"][i, ri] = float(xr.norm())
                    g = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 200_000 + 1000 * uid + r)   # (prompt, depth)
                    xi = torch.randn((NDIR_A, L, d), generator=g, dtype=torch.float64).to(dev, DT)
                    xi = xi / xi.norm(dim=-1, keepdim=True) * xr[0].norm(dim=-1, keepdim=True)[None]
                    sg = torch.tensor(SIG, device=dev, dtype=DT).repeat_interleave(NDIR_A)[:, None, None]
                    XP = torch.cat([xr, xr, xr + sg * xi.repeat(NS, 1, 1)])
                    X1r = xr
                    sg1 = sg[:, 0, 0].double()
                    z_r = None
                    for k in range(K + 1):
                        ZP = read(XP, fc)
                        fish, kl, _, lp, _, p0 = mets(ZP)
                        z1r = read(X1r, fc)[0]
                        R["a2_floor_fish"][i, ri, k] = float(hnorm(z1r.double() - ZP[0].double(), p0) ** 2)
                        R["a2_dup_fish"][i, ri, k] = float(fish[1])
                        if z_r is None:
                            z_r = lp[0]
                        R["a2_base_kl"][i, ri, k] = float((z_r.exp() * (z_r - lp[0])).sum())
                        R["a2_base_drift"][i, ri, k] = float((XP[0] - xr[0]).norm() / xr.norm())
                        dX = XP[2:] - XP[0:1]
                        dxa = dX.flatten(1).norm(dim=1) / XP[0].norm(); dxq = dX[:, q].norm(dim=-1) / XP[0, q].norm()
                        R["a2_fish"][i, ri, :, :, k] = (fish[2:] / sg1 ** 2).view(NS, NDIR_A).cpu().numpy()
                        R["a2_kl"][i, ri, :, :, k] = kl[2:].view(NS, NDIR_A).cpu().numpy()
                        R["a2_dx_all"][i, ri, :, :, k] = (dxa.double() / sg1).view(NS, NDIR_A).cpu().numpy()
                        R["a2_dx_q"][i, ri, :, :, k] = (dxq.double() / sg1).view(NS, NDIR_A).cpu().numpy()
                        if k < K:
                            XP = step(XP, e, fc); X1r = step(X1r, e, fc)
                R["doneA"][i] = True
            else:
                # ---- G2: hooked site = output_hidden_states[4]; zero push reproduces the logits
                if i < 2:
                    hook_state["delta"] = None
                    out = M(ids, use_cache=False, output_hidden_states=True)
                    R["g2"][i, 0] = float((hook_state["cap"] - out.hidden_states[4]).abs().max())
                    hook_state["delta"] = torch.zeros((1, L, d), device=dev, dtype=DT)
                    z0 = M(ids, use_cache=False).logits[0, q]
                    R["g2"][i, 1] = float((z0 - out.logits[0, q]).abs().max())
                    hook_state["delta"] = None
                    assert R["g2"][i, 0] == 0.0 and R["g2"][i, 1] <= 1e-6 * float(z0.abs().max()), ("G2 failed", R["g2"][i])
        # ---- Part C: steering at the layer-4 output, position q; directions at a fixed diagonal-Mahalanobis norm sqrt(d)
        gC = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 300_000 + uid)
        U = torch.randn((NR, d), generator=gC, dtype=torch.float64).to(dev, DT) * s_site
        for ci, r in enumerate(RC):
            check()
            if looped:
                def run_push(P, r=r):     # P (B, d) pushes at q of the prelude output; same start for every row
                    eb = e.expand(P.shape[0], -1, -1).clone(); eb[:, q] = eb[:, q] + P
                    x = x0.expand(P.shape[0], -1, -1)
                    for _ in range(r):
                        x = M.transformer.adapter(torch.cat([x, eb], dim=-1))
                        for blk in M.transformer.core_block:
                            x = blk(x, fc, DUM, None, None)
                    return read(x, fc)
            else:
                def run_push(P):
                    dl = torch.zeros((P.shape[0], L, d), device=dev, dtype=DT); dl[:, q] = P
                    hook_state["delta"] = dl
                    try:
                        return M(ids.expand(P.shape[0], -1), use_cache=False, logits_to_keep=1).logits[:, -1].double()
                    finally:
                        hook_state["delta"] = None
            with torch.enable_grad():
                pz = torch.zeros((1, d), device=dev, dtype=DT, requires_grad=True)
                lpa = torch.log_softmax(run_push(pz)[0].double(), -1)[ans]
                gr = torch.autograd.grad(lpa, pz)[0][0]
            with torch.no_grad():
                gnorm = float((gr * s_site).norm()); R["c_gradnorm"][i, ci] = gnorm
                ug = s_site ** 2 * gr
                Ud = torch.cat([U, ug[None], -ug[None]])
                mn = (Ud / s_safe).norm(dim=-1, keepdim=True)
                valid = (mn[:, 0] > 0) & torch.isfinite(mn[:, 0])
                Ud = torch.where(valid[:, None], Ud / torch.where(mn > 0, mn, torch.ones_like(mn)) * d ** 0.5,
                                 torch.zeros_like(Ud))
                R["c_push_euclid"][i, ci] = torch.where(valid, Ud.norm(dim=-1) / rho, torch.nan).cpu().numpy()
                if looped:
                    R["c_adapter_gain"][i, ci] = torch.where(valid, (Ud @ We.T).norm(dim=-1) / Ud.norm(dim=-1).clamp_min(1e-30),
                                                             torch.nan).cpu().numpy()
                epsv = torch.tensor(EPS, device=dev, dtype=DT)
                P = (Ud[:, None, :] * epsv[None, :, None]).reshape(-1, d)
                zero = torch.zeros((1, d), device=dev, dtype=DT)
                Z = run_push(torch.cat([zero, zero, P]))     # rows 0, 1 = no push (in batch), then the pushes
                z1 = run_push(zero)[0]                       # single-row run: reference computed differently
                fish, kl, top, lp, Dz, p0 = mets(Z)
                R["c_floor"][i, ci] = float((z1 - Z[0]).abs().max() / z1.abs().max())
                R["c_floor_fish"][i, ci] = float(hnorm(z1.double() - Z[0].double(), p0) ** 2)
                R["c_dup_fish"][i, ci] = float(fish[1])
                R["c_base_logp"][i, ci] = float(lp[0, ans]); base_top = int(top[0]); R["c_base_top1"][i, ci] = base_top
                fish, kl, top = fish[2:].view(NDIR_C, NE), kl[2:].view(NDIR_C, NE), top[2:].view(NDIR_C, NE)
                Dz = Dz[2:].view(NDIR_C, NE, -1); dlp = (lp[2:, ans] - lp[0, ans]).view(NDIR_C, NE)
                lin1 = Dz[:, i0] / EPS[i0]; lin3 = Dz[:, i1] / EPS[i1]
                R["c_fdcheck"][i, ci] = ratio(hnorm(lin3 - lin1, p0), hnorm(lin1, p0)).cpu().numpy()
                dd_fd = lin1[:, ans] - (p0 * lin1).sum(-1)                         # first-order d log pi(ans) / d eps
                dd_gr = (Ud.double() @ gr.double())                                # the same from the gradient
                scale = (gr.double() * s_site.double()).norm() * (Ud.double() / s_safe.double()).norm(dim=-1)
                R["c_gradcheck"][i, ci] = ratio((dd_fd - dd_gr).abs(), scale).cpu().numpy()
                pred = epsv.double()[None, :, None] * lin1[:, None, :]
                R["c_linerr"][i, ci] = ratio(hnorm(Dz - pred, p0), hnorm(Dz, p0)).cpu().numpy()
                R["c_dlogp_pred"][i, ci] = (epsv.double()[None, :] * dd_fd[:, None]).cpu().numpy()
                R["c_kl"][i, ci] = kl.cpu().numpy(); R["c_fish"][i, ci] = fish.cpu().numpy()
                R["c_dlogp"][i, ci] = dlp.cpu().numpy(); R["c_flip"][i, ci] = (top != base_top).float().cpu().numpy()
                bad = ~valid
                for arr in ("c_kl", "c_fish", "c_linerr", "c_dlogp", "c_dlogp_pred", "c_flip"):
                    R[arr][i, ci][bad.cpu().numpy()] = np.nan
                for arr in ("c_fdcheck", "c_gradcheck"):
                    R[arr][i, ci][bad.cpu().numpy()] = np.nan
        R["doneC"][i] = True

    status, err = "complete", None
    try:
        for i in range(n):
            est = max(TM) if TM else cfg["est_item_s"]
            if time.monotonic() + est > deadline:
                status = f"deadline: stopped before item {i}"
                print(f"[{key}] {status} (estimate {est:.0f}s)", flush=True)
                break
            tc = time.monotonic()
            one_prompt(i)
            TM.append(time.monotonic() - tc)
            print(f"[{key}] item {i + 1}/{n} {TM[-1]:.1f}s (total {time.monotonic() - T0:.0f}s, peak {peak():.1f} GiB)",
                  flush=True)
            save(f"{o}/partial.npz"); commit()
    except Stop:
        status = "deadline: stopped inside an item (its unfinished parts are marked not done)"
        print(f"[{key}] {status}", flush=True)
    except Exception:
        import traceback
        status, err = "error", traceback.format_exc()
        print(f"[{key}] ERROR\n{err}", flush=True)

    # ---- gates and summaries (G1/G2 raise inside an item; G4 thresholds are fixed in docs/oversight_pilot.md)
    def q_(a, p):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        return float(np.quantile(a, p)) if a.size else None
    fd = R["c_fdcheck"][R["doneC"]]; gc = R["c_gradcheck"][R["doneC"]]
    g4 = dict(fd_median=q_(fd, 0.5), fd_q90=q_(fd, 0.9), grad_median=q_(gc, 0.5), grad_q90=q_(gc, 0.9))
    g4["pass"] = bool(g4["fd_median"] is not None and g4["grad_median"] is not None and g4["fd_median"] <= 1e-2
                      and g4["fd_q90"] <= 5e-2 and g4["grad_median"] <= 1e-2 and g4["grad_q90"] <= 5e-2)
    gate_fail = []
    if status == "error":
        gate_fail.append("error")
    if not g4["pass"] and R["doneC"].any():
        gate_fail.append("G4")
    if looped and R["doneA"].any():
        lin = R["a2_fish"][R["doneA"]][:, :, 1, :, 1] / R["a2_fish"][R["doneA"]][:, :, 0, :, 1]   # paired, k = 1
        g4.update(a2_lin_k1_median=q_(lin, 0.5), a2_lin_k1_q10=q_(lin, 0.1), a2_lin_k1_q90=q_(lin, 0.9))
    floors = dict(a0_floor_max=q_(R["a0_floor"], 1.0), a0_floor_fish_max=q_(R["a0_floor_fish"], 1.0),
                  a1_dup_fish_max=q_(R["a1_dup_fish"], 1.0), a2_floor_fish_max=q_(R["a2_floor_fish"], 1.0),
                  a2_dup_fish_max=q_(R["a2_dup_fish"], 1.0), c_floor_max=q_(R["c_floor"], 1.0),
                  c_floor_fish_max=q_(R["c_floor_fish"], 1.0), c_dup_fish_max=q_(R["c_dup_fish"], 1.0))
    save(f"{o}/obs.npz", compressed=True)
    meta = dict(cfg=cfg, key=key, status=status, error=err, gate_fail=gate_fail, runtime_s=time.monotonic() - T0,
                n_doneA=int(R["doneA"].sum()), n_doneC=int(R["doneC"].sum()), n_requested=n, item_s=TM, peak_gib=peak(),
                versions=versions, numerics=numerics, g1_max=q_(R["g1"], 1.0) if looped else None,
                g2=R["g2"].tolist() if not looped else None, g4=g4, floors=floors)
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    commit()
    if status == "error":
        raise RuntimeError(f"[{key}] {err}")
    return meta


COMMON = dict(T=64, K=32, sig=[1e-3, 1e-2], ndir_a=4, n_starts=8, eps=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0],
              n_rand=4, seed=11_000_000, prompt_seed=2026_11_07, reserve_s=45, est_item_s=60, dtype="float64")
MODES = {
    "smoke": dict(COMMON, models=["L4", "D"], max_items=2, total_s=360),
    "run1": dict(COMMON, models=["L4", "L32", "D"], total_s=400),
    "run2": dict(COMMON, models=["L8", "L16"], total_s=400),
    # run2's L16 container lost its last prompt to a cold weight download (setup 100 s); rerun with warm cache
    "run2_L16": dict(COMMON, models=["L16"], total_s=400),
    # part 2 (docs/oversight_hard.md): difficulty screen on the candidate pool, then the protocol on the 16 selected
    "screen": dict(COMMON, models=["L4", "L32", "D"], n_per_family=64, cand_seed=2026_11_08, screen_batch=16, total_s=400),
    "hard": dict(COMMON, models=["L4", "L32", "D"], total_s=400, smoke_uids=[]),
}


@app.local_entrypoint()
def main(mode: str = "smoke", prompts: str = ""):
    cfg = dict(MODES[mode])
    here = os.path.dirname(os.path.abspath(__file__))
    cfg["code_sha256"] = hashlib.sha256(open(os.path.join(here, "obs_app.py"), "rb").read()).hexdigest()[:16]
    run_id = f"obs_{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    rdir = os.path.join(here, "..", "results", run_id)
    os.makedirs(rdir, exist_ok=True)
    if mode == "screen":
        items = make_hard_candidates(cfg["n_per_family"], cfg["cand_seed"])
        with open(os.path.join(rdir, "candidates.json"), "w") as f_:
            json.dump(items, f_, indent=1)
        print(run_id, len(items), "candidates", cfg)
        calls = [screen.spawn(k, json.dumps(items), cfg, run_id) for k in cfg["models"]]
        res, bad = {}, []
        for k, c in zip(cfg["models"], calls):
            try:
                m = c.get()
                print(k, m["status"], f"{m['runtime_s']:.0f}s", "accuracy by family", m["acc_by_family"])
                res[k] = m
                if m["status"] != "complete":
                    bad.append(f"{k}: status={m['status']}")
            except Exception as ex:
                print(k, "FAILED", repr(ex)); bad.append(f"{k}: {ex!r}")
        with open(os.path.join(rdir, "screen_results.json"), "w") as f_:
            json.dump(res, f_)
        if bad:
            print("SCREEN NOT CLEAN (no selection made):", *bad, sep="\n  ")
            raise SystemExit(1)
        sel, counts = select_hard(items, res)
        with open(os.path.join(rdir, "hard_prompts.json"), "w") as f_:
            json.dump(sel, f_, indent=1)
        sha = lambda fn: hashlib.sha256(open(os.path.join(rdir, fn), "rb").read()).hexdigest()[:16]
        man = dict(run_id=run_id, cfg=cfg, code_sha256=cfg["code_sha256"],
                   models={k: [res[k]["hf_id"], res[k]["revision"], res[k]["forward_source_sha256"], res[k]["logit_dtypes"]]
                           for k in res},
                   candidates_sha256=sha("candidates.json"), screen_results_sha256=sha("screen_results.json"),
                   selection=dict(rule="docs/oversight_hard.md Step 3", n_each=8, thresh=0.9, fam_order=list(FAM_ORDER),
                                  depth_looped=64), counts=counts, selected_cids=[it["cid"] for it in sel],
                   selected_strata=[it["stratum"] for it in sel], hard_prompts_sha256=sha("hard_prompts.json"))
        with open(os.path.join(rdir, "screen_manifest.json"), "w") as f_:
            json.dump(man, f_, indent=1)
        print("strata", counts)
        for it in sel:
            print(it["uid"], it["stratum"], it["fam"], repr(it["prompt"].splitlines()[-1]), it["ans"], it["screen"])
        print("SCREEN CLEAN", run_id, "->", os.path.join(rdir, "hard_prompts.json"))
        return
    if mode == "hard":
        assert prompts, "--prompts <hard_prompts.json from the screen> is required"
        items, man = verify_selection(prompts)
        cfg.update(prompts_file=os.path.abspath(prompts), prompts_sha256=man["hard_prompts_sha256"], screen_run_id=man["run_id"],
                   screen_code_sha256=man["code_sha256"])
        print("selection re-derived from", man["run_id"], "and verified")
    else:
        items = make_prompts(8, 8, cfg["prompt_seed"])
    with open(os.path.join(rdir, "prompts.json"), "w") as f_:
        json.dump(items, f_, indent=1)
    print(run_id, len(items), "prompts", cfg)
    calls = [obs.spawn(k, json.dumps(items), cfg, run_id) for k in cfg["models"]]
    bad = []
    for k, c in zip(cfg["models"], calls):
        try:
            m = c.get()
            print(k, json.dumps(m, indent=1))
            if m["status"] != "complete" or m["gate_fail"]:
                bad.append(f"{k}: status={m['status']} gate_fail={m['gate_fail']}")
        except Exception as ex:  # report each container separately
            print(k, "FAILED", repr(ex))
            bad.append(f"{k}: {ex!r}")
    if bad:
        print("RUN NOT CLEAN:", *bad, sep="\n  ")
        raise SystemExit(1)
    print("RUN CLEAN", run_id)
