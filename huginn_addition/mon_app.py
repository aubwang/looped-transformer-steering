"""Forward-only monitor collection; docs/oversight_monitor.md and docs/mon_spec.md (Part A).

Float32 model computation, float64 scalar diagnostics, streamed states and gated NPZ chunks.
The Raven loop is reproduced here because its iterate_one_step uses nonexistent freqs_cis.
"""
from __future__ import annotations

import hashlib
import inspect
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
FAMS = ["add3", "addsub", "muladd", "add4", "sub3", "add3w", "arc_easy", "arc_chal"]
N_SCREEN, N_MAIN_MAX, N_MAIN_MIN = 96, 1500, 800
SCREEN_MODELS = ["L4", "L32", "D"]  # the models the selection rule uses (as in the pilot screen)
T_STD = [1, 2, 3, 4, 6, 8, 12, 16, 24, 31, 32, 33]
T_DEEP = T_STD + [48, 63, 64]
T_PREV, R_POST, D_PREV = [8, 32], [4, 8, 16, 32], [8, 12, 16]
R_DISP = [2, 4, 8, 16, 32]  # whole-sequence displacement ||X_32 - X_r|| (and ||X_64 - X_r|| on the deep subset)
L_MAX = 160  # per-position step sizes are NaN-padded to this length
R_NPOS = [8, 16, 24, 32]  # loops at which per-position state norms ||X_t[p]|| are kept (floor for per-position rates)
SHARDS_MAIN = 3  # registered schedule: three shards per model (timing margin against the container deadline)
app = modal.App("oversight-monitor")
hf_cache = modal.Volume.from_name("huginn-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("huginn-results", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==4.53.1", "safetensors", "huggingface_hub[hf_transfer]",
                      "numpy", "pandas", "pyarrow")
         .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))


def arith_key(fam, ops):
    a, b, c = ops[:3]
    if fam in ("add3", "add4", "add3w"):
        return tuple(sorted(ops))
    if fam == "addsub":
        return tuple(sorted((a, b))), c
    if fam == "sub3":
        return a, tuple(sorted((b, c)))
    return tuple(ops)


def make_arith(fam, n_total, seed):
    rng = random.Random(seed)

    def draw():
        while True:
            if fam in ("add3", "add4", "add3w"):
                lo, hi = (100, 799) if fam == "add3w" else (10, 99)
                ops = [rng.randint(lo, hi) for _ in range(4 if fam == "add4" else 3)]
                ans = sum(ops); q = " + ".join(map(str, ops)) + " = "
            elif fam == "addsub":
                ops = [rng.randint(100, 899), rng.randint(100, 899), rng.randint(10, 999)]
                a, b, c = ops; ans = a + b - c; q = f"{a} + {b} - {c} = "
            elif fam == "muladd":
                ops = [rng.randint(12, 99), rng.randint(3, 9), rng.randint(10, 99)]
                a, b, c = ops; ans = a * b + c; q = f"{a} * {b} + {c} = "
            else:
                assert fam == "sub3"
                ops = [rng.randint(200, 999), rng.randint(10, 99), rng.randint(10, 99)]
                a, b, c = ops; ans = a - b - c; q = f"{a} - {b} - {c} = "
            if 1 <= ans <= 999:
                return ops, ans, q

    queries, keys = [], set()
    while len(queries) < n_total:
        p = draw(); k = arith_key(fam, p[0])
        if k not in keys:
            keys.add(k); queries.append(p)
    items = []
    for i, (ops, ans, q) in enumerate(queries):
        demos, seen = [], set()
        while len(demos) < 4:
            op, a, text = draw(); k = arith_key(fam, op)
            if k not in keys and k not in seen:
                seen.add(k); demos.append(text + str(a) + "\n")
        items.append(dict(pid=100000 * FAMS.index(fam) + i, fam=fam, i=i, kind="std", ops=ops,
                          ans=ans, prompt="".join(demos) + q, valid="num"))
    return items


def stable_hash(s):
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16)


def one_token(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) == 1, (text, ids)
    return ids[0]


def make_mcq(rows, tok, fam, n_total, seed, max_tokens=160):
    counts = dict(rows_total=len(rows)); seen, good = set(), []
    four = [r for r in rows if len(r["choices"]) == len(r["labels"]) == 4 and r["answerKey"] in r["labels"]]
    counts["four_option"] = len(four)
    for r in sorted(four, key=lambda r: str(r["id"])):
        if r["question"] not in seen:
            seen.add(r["question"]); good.append(r)
    counts["deduplicated"] = len(good)
    random.Random(seed).shuffle(good)

    def render(r):
        p = list(range(4)); random.Random(seed * 1000003 + stable_hash(r["id"])).shuffle(p)
        correct = r["labels"].index(r["answerKey"]); letter = "ABCD"[p.index(correct)]
        text = f'Question: {r["question"]}\n' + "".join(f'{l}. {r["choices"][j]}\n' for l, j in zip("ABCD", p)) + "Answer:"
        return text, letter, p

    pool = good[:200]; items, hist = [], {}; examined = 0
    for r in good[200:]:
        if len(items) >= n_total:
            break
        examined += 1
        rng = random.Random(seed * 1000003 + stable_hash(r["id"]))
        demos = rng.sample(pool, 2); q, ans, perm = render(r)
        prompt = "\n".join(render(d)[0] + " " + render(d)[1] for d in demos) + "\n" + q
        L = 1 + len(tok.encode(prompt, add_special_tokens=False))
        if L > max_tokens:
            continue
        one_token(tok, " " + ans)
        i = len(items)
        items.append(dict(pid=100000 * FAMS.index(fam) + i, fam=fam, i=i, kind="std", ans=ans,
                          prompt=prompt, valid="abcd", mc=dict(id=r["id"], permutation=perm,
                          demo_ids=[d["id"] for d in demos])))
        hist[str(L)] = hist.get(str(L), 0) + 1
    counts.update(examined=examined, kept=len(items), length_histogram=hist)
    return items, counts


@app.function(image=image, volumes={"/cache": hf_cache}, timeout=600)
def build_items(cfg):
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer
    import pandas as pd
    tok = AutoTokenizer.from_pretrained(*MODELS["L32"][:1], revision=MODELS["L32"][1])
    repo = "allenai/ai2_arc"; sha = HfApi().dataset_info(repo).sha
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=sha)
    items, counts = [], dict(dataset_commit_sha=sha)
    for fam in FAMS:
        seed = cfg["item_seed"] * 100 + FAMS.index(fam)
        if fam.startswith("arc"):
            name = "ARC-Easy" if fam == "arc_easy" else "ARC-Challenge"
            rows = []
            for fn in sorted(f for f in files if f.startswith(name + "/") and f.endswith(".parquet")):
                path = hf_hub_download(repo, fn, repo_type="dataset", revision=sha)
                for r in pd.read_parquet(path).to_dict("records"):
                    rows.append(dict(r, choices=list(r["choices"]["text"]), labels=list(r["choices"]["label"])))
            its, counts[fam] = make_mcq(rows, tok, fam, N_SCREEN + N_MAIN_MAX, seed)
        else:
            its = make_arith(fam, N_SCREEN + N_MAIN_MAX, seed); counts[fam] = dict(kept=len(its))
        items.extend(its)
    tokenize(tok, items)  # G3 before any GPU allocation
    hf_cache.commit()
    return dict(items=items, counts=counts)


def tokenize(tok, items):
    valid = {"num": [one_token(tok, str(i)) for i in range(1000)],
             "abcd": [one_token(tok, " " + c) for c in "ABCD"]}
    ids, ans = [], []
    for it in items:
        v = [BOS] + tok.encode(it["prompt"], add_special_tokens=False)
        if it["valid"] == "num":
            assert tok.decode(v[-1:]) == " "
        else:
            assert it["prompt"].endswith("Answer:") and tok.decode(v[-1:]) == ":" and len(v) <= 160
        ids.append(v); ans.append(one_token(tok, (" " if it["valid"] == "abcd" else "") + str(it["ans"])))
    return ids, ans, valid


def start(M, items, L, seed, dev):
    import torch
    std = float(M.config.init_values["std"]); xs = []
    for it in items:
        g = torch.Generator("cpu").manual_seed(seed + it["pid"])
        x = torch.empty((L, M.config.n_embd), dtype=torch.float32)
        torch.nn.init.trunc_normal_(x, std=std, a=-3 * std, b=3 * std, generator=g)
        xs.append(x)
    return torch.stack(xs).to(dev)


def harness(M, dev):
    import torch
    c = M.config
    assert c.test_time_noise == 0 and c.injection_type == "linear" and M.emb_scale == 1.0
    assert len(M.transformer.prelude) == 4 and len(M.transformer.core_block) == 6 and len(M.transformer.coda) == 4
    dum = torch.tensor(0, dtype=torch.long)

    def prep(ids):
        e = M.transformer.wte(ids); fc = M.rotary_emb(e, torch.arange(ids.shape[1], device=dev)[None])
        bi = torch.tensor(-1, dtype=torch.long)
        for blk in M.transformer.prelude:
            bi += 1; e = blk(e, fc, bi, None, None)
        return e, fc

    def step(x, e, fc):
        x = M.transformer.adapter(torch.cat([x, e], -1))
        for blk in M.transformer.core_block:
            x = blk(x, fc, dum, None, None)
        return x

    def read(x, fc):
        for blk in M.transformer.coda:
            x = blk(x, fc, dum, None, None)
        return x, M.lm_head(M.transformer.ln_f(x[:, -1])).double()
    return prep, step, read


def centered_error(a, b):
    a, b = a.double(), b.double()
    return float(((a - a.mean(-1, keepdim=True)) - (b - b.mean(-1, keepdim=True))).abs().max())


def summary(logits, ans, valid, letters, mc):
    import torch
    lp = logits.double().log_softmax(-1); top_lp, top_id = lp.topk(5, -1)
    opt = lp[:, letters].clone(); opt[~mc] = float("nan")
    return dict(top_id=top_id.to(torch.int32), top_lp=top_lp.float(), ent=(-(lp.exp() * lp).sum(-1)).float(),
                lp_ans=lp.gather(1, ans[:, None])[:, 0].float(),
                valid_lm=torch.where(mc, lp[:, valid["abcd"]].logsumexp(-1), lp[:, valid["num"]].logsumexp(-1)).float(), opt_lp=opt.float())


def norm(x):
    return x.double().flatten(1).norm(dim=1)


def pack_states(x, gates):
    import torch
    assert torch.isfinite(x).all() and float(x.abs().max()) < 6e4, "G4 state finite/range"
    return x.half()


def loop_batch(M, ids, its, ans, valid, letters, mc, cfg, dev, gates, first):
    import torch
    prep, step, read = harness(M, dev)
    T = T_DEEP if its[0]["kind"] == "deep" else T_STD; B, L = ids.shape
    x = start(M, its, L, cfg["seed"], dev); older = None
    e, fc = prep(ids); d = x.shape[-1]
    out = {k: torch.full((B, T[-1] + 1), float("nan"), dtype=torch.float64, device=dev)
           for k in ("d_full", "d_last", "e_full", "e_last", "n_full", "n_last", "cos_full", "cos_last")}
    out["n_full"][:, 0] = norm(x); out["n_last"][:, 0] = norm(x[:, -1])
    # d_pos[i, t, p] = ||X_t - X_{t-1}|| at position p; disp_full[i, j, k] = ||X_target - X_r||, r = R_DISP[j], target = (32, 64)[k]
    out["d_pos"] = torch.full((B, T[-1] + 1, L_MAX), float("nan"), dtype=torch.float32, device=dev)
    out["disp_full"] = torch.full((B, len(R_DISP), 2), float("nan"), dtype=torch.float64, device=dev)
    out["n_pos"] = torch.full((B, len(R_NPOS), L_MAX), float("nan"), dtype=torch.float32, device=dev)  # ||X_t[p]||, t = R_NPOS[j]
    kept = {}
    saved, prev, post, sums = [], [], [], []
    pre = pack_states(e[:, [-1, -2]], gates)
    alone_ref = None
    for t in range(1, T[-1] + 1):
        if time.monotonic() >= cfg.get("_deadline", float("inf")):
            raise TimeoutError("deadline")
        new = step(x, e, fc); delta = new.double() - x.double()
        out["d_pos"][:, t, :L] = delta.norm(dim=-1).float()
        for suffix, sl in (("full", slice(None)), ("last", -1)):
            out["d_" + suffix][:, t] = norm(delta[:, sl])
            out["n_" + suffix][:, t] = norm(new[:, sl])
            if older is not None:
                out["e_" + suffix][:, t] = norm(new[:, sl].double() - older[:, sl].double())
                u = delta[:, sl].flatten(1); v = (x[:, sl].double() - older[:, sl].double()).flatten(1)
                # Zero steps have no direction: choose cosine 0 (finite, torch cosine_similarity convention).
                out["cos_" + suffix][:, t] = (u * v).sum(1) / (u.norm(dim=1) * v.norm(dim=1)).clamp_min(1e-300)
        older, x = x, new
        if t in R_NPOS:
            out["n_pos"][:, R_NPOS.index(t), :L] = x.double().norm(dim=-1).float()
        if t in R_DISP:
            kept[t] = x
        if t in (32, 64):
            for j, r in enumerate(R_DISP):
                if r < t:
                    out["disp_full"][:, j, int(t == 64)] = norm(x.double() - kept[r].double())
        if t in T:
            saved.append(x[:, -1].clone()); coda, logits = read(x, fc)
            sums.append(summary(logits, ans, valid, letters, mc))
            if t in R_POST:
                post.append(pack_states(coda[:, -1], gates))
            if first and t in (4, 32):
                # Reference forward on the SAME batch and starts, so G1 measures harness agreement only.
                ref = M(ids, input_states=start(M, its, L, cfg["seed"], dev), num_steps=t).logits[:, -1]
                err = centered_error(logits, ref); gates.setdefault("G1", []).append(err)
                gates.setdefault("G1_items", []).append(B)
                assert err <= 1e-4, f"G1 {err}"
                if t == 32:
                    alone_ref = ref[:2].clone()
        if t in T_PREV:
            prev.append(pack_states(x[:, -2], gates))
    if first:  # batching alone: the model's own forward, batched versus one item at a time
        gates["batch_single_32"] = [centered_error(M(ids[j:j+1], input_states=start(M, its[j:j+1], L, cfg["seed"], dev), num_steps=32).logits[:, -1],
                                                  alone_ref[j:j+1]) for j in range(min(2, B))]
    states = torch.stack(saved, 1)
    out.update(x_last=pack_states(states, gates), x_last32=states, x_prev=torch.stack(prev, 1),
               post=torch.stack(post, 1), pre=pre)
    for k in sums[0]:
        out[k] = torch.stack([s[k] for s in sums], 1)
    return out, T


def d_batch(M, ids, ans, valid, letters, mc, gates):
    import torch
    assert len(M.model.layers) == 16 and M.config._attn_implementation == "sdpa"
    last, prev, sums = [], [], []; stats = {k: [] for k in ("dl_full", "dl_last", "nl_full", "nl_last")}
    previous = None

    def hook(mod, inp, value):
        nonlocal previous
        h = value[0] if isinstance(value, tuple) else value
        l = len(last); last.append(h[:, -1].clone())
        if l in D_PREV:
            prev.append(pack_states(h[:, -2], gates))
        for suffix, sl in (("full", slice(None)), ("last", -1)):
            stats["nl_" + suffix].append(norm(h[:, sl]))
            stats["dl_" + suffix].append(norm(h[:, sl].double() - previous[:, sl].double()) if previous is not None
                                         else torch.full((len(ids),), float("nan"), device=ids.device, dtype=torch.float64))
        sums.append(summary(M.lm_head(M.model.norm(h[:, -1])).double(), ans, valid, letters, mc))
        previous = h
    handles = [m.register_forward_hook(hook) for m in [M.model.embed_tokens, *M.model.layers]]
    try:
        true = M(ids, use_cache=False, logits_to_keep=1).logits[:, -1]
    finally:
        for h in handles:
            h.remove()
    lens = M.lm_head(M.model.norm(last[-1])).double(); err = centered_error(lens, true)
    gates.setdefault("G2", []).append(err); assert err <= 1e-4, f"G2 {err}"
    states = torch.stack(last, 1)
    out = dict(h_last=pack_states(states, gates), h_last32=states, h_prev=torch.stack(prev, 1))
    out.update({k: torch.stack(v, 1) for k, v in stats.items()})
    out.update({"lens_" + k: torch.stack([s[k] for s in sums], 1) for k in sums[0]})
    out.update({"ro_" + k: v for k, v in summary(true, ans, valid, letters, mc).items()})
    return out, None


def ordered_items(items, cfg):
    out = []
    for fam in FAMS:
        group = sorted((dict(it) for it in items if it["fam"] == fam), key=lambda it: it["i"])
        for it in group:  # rank in the run's whole set (i0 = first index of the set), so shards share one definition
            j = it["i"] - cfg.get("i0", 0)
            it["kind"] = "deep" if j < cfg["n_deep"] else "std"
            it["f32"] = j < cfg["n_f32"]
            out.append(it)
    return sorted(out, key=lambda it: (it["kind"] != "deep", it["i"], FAMS.index(it["fam"])))


def check_chunk(z, looped):
    import numpy as np
    for k, v in z.items():
        if v.dtype.kind != "f":
            continue
        allowed = np.zeros(v.shape, bool)
        if k == "d_pos":  # defined for loops t >= 1 at real positions only
            real = np.arange(v.shape[2])[None, :] < z["plen"][:, None]
            allowed = np.broadcast_to(~real[:, None, :], v.shape).copy(); allowed[:, 0] = True
            assert np.isnan(v[allowed]).all(), "G4 d_pos padding"
            total = np.sqrt(np.nansum(v[:, 1:].astype(np.float64) ** 2, axis=2)); ref = z["d_full"][:, 1:]
            assert np.all(np.abs(total - ref) <= 1e-4 * ref + 1e-30), "G4 d_pos against d_full"
        elif k == "n_pos":  # real positions only; consistent with the whole-sequence norms
            real = np.arange(v.shape[2])[None, :] < z["plen"][:, None]
            allowed = np.broadcast_to(~real[:, None, :], v.shape).copy()
            assert np.isnan(v[allowed]).all(), "G4 n_pos padding"
            total = np.sqrt(np.nansum(v.astype(np.float64) ** 2, axis=2)); ref = z["n_full"][:, R_NPOS]
            assert np.all(np.abs(total - ref) <= 1e-4 * ref + 1e-30), "G4 n_pos against n_full"
        elif k == "disp_full":  # target 32: r < 32; target 64: deep chunks only
            deep = z["d_full"].shape[1] == 65
            allowed[:, -1, 0] = True; allowed[:, :, 1] = not deep
            assert np.isnan(v[allowed]).all(), "G4 disp_full pattern"
        elif k in ("d_full", "d_last"):
            allowed[:, 0] = True
        elif k in ("e_full", "e_last", "cos_full", "cos_last"):
            allowed[:, :2] = True
        elif k in ("dl_full", "dl_last"):
            allowed[:, 0] = True
        elif k.endswith("opt_lp"):
            arith = z["pid"] // 100000 < 6
            allowed = np.broadcast_to(arith.reshape((-1,) + (1,) * (v.ndim-1)), v.shape)
            assert np.isnan(v[allowed]).all(), f"G4 arithmetic options {k}"
        assert np.isfinite(v[~allowed]).all() and not np.isinf(v).any(), f"G4 finite {k}"
    name = "x_last" if looped else "h_last"
    idx = {int(p): i for i, p in enumerate(z["pid"])}
    a = z[name][[idx[int(p)] for p in z["pid32"]]].astype(np.float64)
    b = z[name + "32"].astype(np.float64)
    if len(b):
        rel = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-300)
        err = float(rel.max()); assert err <= 1e-3, f"G4 rounding {err}"
        return err
    return None  # no float32 copies in this chunk: not checked (distinct from a perfect match)


def run_collect(key, M, tok, items, cfg, o, dev, commit, T0):
    import numpy as np
    import torch
    import transformers
    os.makedirs(o, exist_ok=True); M.eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    gates, timing, pending = {}, [], {"deep": [], "std": []}; done = 0; chunk = 0
    committed, chunks = [], []
    status, error = "complete", None
    deadline = T0 + cfg["limit_s"] - 40
    items = ordered_items(items, cfg); batches = []
    try:  # G3 and dtype failures must still leave a meta.json
        assert all(p.dtype == torch.float32 for p in M.parameters()), "dtype float32"
        IDS, ANS, valid_ids = tokenize(tok, items)
    except AssertionError as ex:
        status, error = "gate_fail", f"G3/dtype {ex}"; IDS, ANS, valid_ids = [[] for _ in items], [], {}

    def flush(kind):
        nonlocal chunk
        parts = pending[kind]
        if not parts:
            return
        z = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0] if k != "T"}
        if "T" in parts[0]:
            z["T"] = parts[0]["T"]
        pending[kind] = []  # a chunk that fails G4 is dropped, not retried
        err = check_chunk(z, key != "D")
        gates.setdefault("G4", []).extend([] if err is None else [err])
        gates.setdefault("G4_checked_states", []).append(int(len(z["pid32"])))
        name = f"c{chunk}_{kind}.npz"; np.savez(f"{o}/{name}", **z); chunk += 1; commit()
        committed.extend(int(p) for p in z["pid"]); chunks.append(name)

    # Round-robin homogeneous batches: each family visits its earliest remaining i/length bucket.
    # Item-level round-robin would force singleton batches when adjacent lengths differ.
    buckets = {}
    for i, it in enumerate(items):
        buckets.setdefault((it["kind"], it["fam"], len(IDS[i])), []).append(i)
    for kind in ("deep", "std") if status == "complete" else ():
        while any(v for (k, f, L), v in buckets.items() if k == kind):
            for fam in FAMS:
                keys = [k for k, v in buckets.items() if k[:2] == (kind, fam) and v]
                if not keys:
                    continue
                k = min(keys, key=lambda k: items[buckets[k][0]]["i"])
                cap = cfg["bs_mcq"] if fam.startswith("arc") else cfg["bs_arith"]
                batches.append(buckets[k][:cap]); buckets[k] = buckets[k][cap:]
    try:
        with torch.no_grad():
            for bi in batches:
                if time.monotonic() >= deadline:
                    status = "partial"; break
                t0 = time.monotonic(); its = [items[i] for i in bi]; kind = its[0]["kind"]
                ids = torch.tensor([IDS[i] for i in bi], device=dev)
                ans = torch.tensor([ANS[i] for i in bi], device=dev)
                letters = torch.tensor(valid_ids["abcd"], device=dev)
                mc = torch.tensor([it["valid"] == "abcd" for it in its], device=dev)
                valid = {k: torch.tensor(v, device=dev) for k, v in valid_ids.items()}
                if key == "D":
                    out, T = d_batch(M, ids, ans, valid, letters, mc, gates)
                else:
                    out, T = loop_batch(M, ids, its, ans, valid, letters, mc, dict(cfg, _deadline=deadline), dev, gates, done == 0)
                mask = torch.tensor([it["f32"] for it in its], device=dev)
                sname = "h_last32" if key == "D" else "x_last32"
                out[sname] = out[sname][mask]
                z = {k: v.cpu().numpy() for k, v in out.items()}
                z.update(pid=np.array([it["pid"] for it in its], dtype=np.int64),
                         pid32=np.array([it["pid"] for it in its if it["f32"]], dtype=np.int64),
                         ans_tok=np.array([ANS[i] for i in bi], dtype=np.int32),
                         tok_hash=np.array([int(hashlib.sha256(json.dumps(IDS[i]).encode()).hexdigest()[:15], 16) for i in bi], dtype=np.int64),
                         plen=np.full(len(bi), ids.shape[1], dtype=np.int32))
                if T is not None:
                    z["T"] = np.array(T, dtype=np.int64)
                pending[kind].append(z); done += len(bi)
                elapsed = time.monotonic() - t0
                # Attribute a mixed-family batch's elapsed time proportionally to its item counts.
                for fam in FAMS:
                    count = sum(it["fam"] == fam for it in its)
                    if count:
                        timing.append([fam, kind, ids.shape[1], count, T[-1] if T else 16, elapsed * count / len(bi)])
                if sum(len(p["pid"]) for p in pending[kind]) >= 384:
                    flush(kind)
    except TimeoutError:
        status = "partial"
    except AssertionError as ex:
        status, error = "gate_fail", str(ex)
    try:
        for kind in pending:
            flush(kind)
    except AssertionError as ex:
        status, error = "gate_fail", str(ex)
    meta = dict(status=status, error=error, items_requested=len(items), items_done=done, items_committed=len(committed),
                committed_pids=committed, chunks=chunks, model=key, tokenizer=getattr(tok, "name_or_path", None),
                hf_id=MODELS[key][0],
                revision=MODELS[key][1], dtype="float32", torch=str(torch.__version__), transformers=transformers.__version__,
                code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                forward_source_sha256=hashlib.sha256(inspect.getsource(type(M).forward).encode()).hexdigest(),
                gates=gates, T_STD=T_STD, T_DEEP=T_DEEP, T_PREV=T_PREV, R_POST=R_POST, D_PREV=D_PREV,
                R_DISP=R_DISP, R_NPOS=R_NPOS, L_MAX=L_MAX, cfg=cfg,
                # token ids of the valid answers (integers 0..999 in order; letters A-D): the analysis decodes predictions with them
                valid_ids={k: [int(t) for t in v] for k, v in valid_ids.items()},
                timing=dict(batches=timing, model_load_seconds=cfg.get("model_load_seconds", 0), total_seconds=time.monotonic()-T0))
    with open(f"{o}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    commit()
    return meta


def collect_impl(key, items_json, cfg, run_id, shard):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    T0 = time.monotonic(); hf_id, rev, _ = MODELS[key]
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(hf_id, revision=rev)
    extra = dict(attn_implementation="sdpa") if key == "D" else dict(trust_remote_code=True)
    M = AutoModelForCausalLM.from_pretrained(hf_id, revision=rev, torch_dtype=torch.float32, **extra).to("cuda").eval()
    try:
        hf_cache.commit()
    except Exception as ex:
        print(f"[{key}] cache commit skipped: {ex!r}", flush=True)
    cfg = dict(cfg, model_load_seconds=time.monotonic()-T0)
    its = json.loads(items_json); o = f"/results/{run_id}/{key}/s{shard}"
    meta = run_collect(key, M, tok, its, cfg, o, torch.device("cuda"), results_vol.commit, T0)
    return add_accuracy(meta, key, o, its)


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=480)
def collect(key: str, items_json: str, cfg: dict, run_id: str, shard: int = 0):
    return collect_impl(key, items_json, cfg, run_id, shard)


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=840)
def collect_long(key: str, items_json: str, cfg: dict, run_id: str, shard: int = 0):
    return collect_impl(key, items_json, cfg, run_id, shard)


def screen_report(master, results):
    acc, throughput = {}, []
    for key, res in results.items():
        acc[key] = res["accuracy"]
        for fam, kind, L, B, Tmax, seconds in res["timing"]["batches"]:
            throughput.append(dict(model=key, fam=fam, kind=kind, token_loops=L*B*Tmax, seconds=seconds))
    eligible = []
    for fam in FAMS:
        low = .4 if fam.startswith("arc") else .3
        enough = sum(it["fam"] == fam for it in master) >= N_SCREEN + N_MAIN_MIN
        if enough and all(low <= acc[k][fam][str(t)] <= .9 for k, t in (("D", 0), ("L32", 32), ("L4", 4))):
            eligible.append(fam)
    selected = [f for f in eligible if not f.startswith("arc")][:3] + [f for f in eligible if f.startswith("arc")][:1]
    return dict(accuracy=acc, eligible=eligible, selection=selected, arithmetic_only=not any(f.startswith("arc") for f in selected),
                redesign=len(eligible) < 2), throughput


def add_accuracy(meta, key, o, items):
    import numpy as np
    by_pid = {it["pid"]: it for it in items}; counts = {}
    for fn in sorted(os.listdir(o)):
        if not fn.endswith(".npz"):
            continue
        with np.load(os.path.join(o, fn)) as z:
            T = [0] if key == "D" else list(z["T"])
            pred = z["ro_top_id"][:, None, 0] if key == "D" else z["top_id"][:, :, 0]
            for fam in FAMS:
                mask = np.array([by_pid[int(p)]["fam"] == fam for p in z["pid"]])
                for j, t in enumerate(T):
                    if t not in T_STD and key != "D":
                        continue
                    pair = counts.setdefault(fam, {}).setdefault(str(t), [0, 0])
                    pair[0] += int((pred[mask, j] == z["ans_tok"][mask]).sum()); pair[1] += int(mask.sum())
    meta["accuracy"] = {f: {t: a/n for t, (a, n) in v.items() if n} for f, v in counts.items()}
    meta["accuracy_counts"] = counts; meta["out_dir"] = o
    return meta


@app.local_entrypoint()
def main(mode: str, items: str = "", spec: str = "", n_main: int = 0, fams: str = "", shards: int = 1):
    cfg = dict(seed=11000000, item_seed=20261109, bs_arith=64, bs_mcq=32,
               n_deep=16 if mode == "screen" else 256, n_f32=8 if mode == "screen" else 32,
               limit_s=460 if mode == "screen" else 820, i0=0 if mode == "screen" else N_SCREEN)
    assert mode in ("count", "screen", "main")
    run_id = f"mon_{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    rdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", run_id)); os.makedirs(rdir)
    if mode == "count":
        data = build_items.remote(cfg)
        raw = json.dumps(data, indent=1).encode(); open(f"{rdir}/items_master.json", "wb").write(raw)
        open(f"{rdir}/items_master.sha256", "w").write(hashlib.sha256(raw).hexdigest() + "\n")
        print(json.dumps(data["counts"], indent=1)); return
    assert items, "--items required"
    raw = open(items, "rb").read(); sha = hashlib.sha256(raw).hexdigest(); master = json.loads(raw)["items"]
    cfg.update(items_sha256=sha, run_id=run_id, mode=mode)
    if mode == "screen":
        pairs = [(k, 0) for k in SCREEN_MODELS]; timeout = 480; fn = collect
        # Families too small for the main collection cannot be selected, so they are not screened.
        big = [f for f in FAMS if sum(it["fam"] == f for it in master) >= N_SCREEN + N_MAIN_MIN]
        chosen = [it for it in master if it["i"] < N_SCREEN and it["fam"] in big]
        assert all(sum(it["fam"] == f for it in chosen) == N_SCREEN for f in big); print("screen families", big)
    else:
        import re
        manifest = json.load(open(os.path.join(os.path.dirname(items), "screen_manifest.json")))
        assert sha == manifest["items_sha256"] and not manifest["redesign"]
        fs = fams.split(","); assert fs == manifest["selection"]
        assert shards == SHARDS_MAIN, f"the registered schedule is --shards {SHARDS_MAIN}"
        assert N_MAIN_MIN <= n_main <= N_MAIN_MAX and spec
        assert all(re.fullmatch(r"(L4|L8|L16|L32|D):\d+", p) for p in spec.split(",")), "spec entries are model:shard"
        pairs = [(p.split(":")[0], int(p.split(":")[1])) for p in spec.split(",")]
        assert len(set(pairs)) == len(pairs) and all(k in MODELS and 0 <= s < shards for k, s in pairs)
        chosen = [it for it in master if it["fam"] in fs and N_SCREEN <= it["i"] < N_SCREEN+n_main]
        assert len(chosen) == len(fs)*n_main
        assert all(any((it["i"]-N_SCREEN) % shards == s for it in chosen) for _, s in pairs), "empty shard"
        # One frozen description of the main collection: the first launch writes it, later launches must match it.
        frozen = dict(families=fs, n_main=n_main, shards=shards, items_sha256=sha,
                      cfg={k: v for k, v in cfg.items() if k != "run_id"},
                      code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest())
        # The analysis is frozen with it: hashes of the analysis code and of the design documents at first launch.
        # A later change to those does not block collection (the data do not depend on them) but is reported
        # here and again by analyze_mon.py --manifest.
        here = os.path.dirname(os.path.abspath(__file__))
        analysis = {os.path.basename(q): hashlib.sha256(open(q, "rb").read()).hexdigest()
                    for q in (os.path.join(here, "analyze_mon.py"), os.path.join(here, "mon_probe.py"),
                              os.path.join(here, "..", "docs", "oversight_monitor.md"),
                              os.path.join(here, "..", "docs", "mon_spec.md"))}
        mpath = os.path.join(os.path.dirname(items), "main_manifest.json")
        if os.path.exists(mpath):
            old = json.load(open(mpath))
            assert {k: v for k, v in old.items() if k != "analysis_sha256"} == frozen, \
                "main_manifest.json differs: families, n, shards, items or code changed"
            changed = sorted(k for k in analysis if old.get("analysis_sha256", {}).get(k) != analysis[k])
            if changed:
                print("WARNING: changed since the manifest was frozen:", changed)
        else:
            json.dump(dict(frozen, analysis_sha256=analysis), open(mpath, "w"), indent=1)
        timeout = 840; fn = collect_long
    assert cfg["limit_s"] <= timeout-20 and len(pairs)*timeout <= 28*60
    calls = [(k, s, fn.spawn(k, json.dumps([it for it in chosen if mode == "screen" or (it["i"]-96) % shards == s]),
                             cfg, run_id, s)) for k, s in pairs]
    results, bad = {}, []
    for key, shard, call in calls:
        try:
            res = call.get(); results[key if mode == "screen" else f"{key}:{shard}"] = res
            print(key, shard, res["status"], res["timing"]["total_seconds"])
            if res["status"] != "complete":
                bad.append((key, shard))
        except Exception as ex:
            print(key, shard, "FAILED", repr(ex)); bad.append((key, shard))
    json.dump(results, open(f"{rdir}/containers.json", "w"), indent=1)
    if bad:
        raise SystemExit(1)
    if mode == "screen":
        report, throughput = screen_report(master, results)
        report.update(items_sha256=sha, cfg=cfg, code_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                      models={k: dict(hf_id=v["hf_id"], revision=v["revision"], forward_source_sha256=v["forward_source_sha256"])
                              for k, v in results.items()})
        json.dump(report, open(f"{rdir}/screen_manifest.json", "w"), indent=1)
        # Also place the manifest beside the master file for the main verifier.
        json.dump(report, open(os.path.join(os.path.dirname(items), "screen_manifest.json"), "w"), indent=1)
        groups = {}
        for r in throughput:
            p = groups.setdefault((r["model"], r["fam"], r["kind"]), [0, 0])
            p[0] += r["token_loops"]; p[1] += r["seconds"]
        table = [dict(model=k, fam=f, kind=c, token_loops_per_second=a/b) for (k, f, c), (a, b) in groups.items()]
        json.dump(table, open(f"{rdir}/throughput.json", "w"), indent=1); print(json.dumps(report, indent=1))
        if report["redesign"]:
            raise SystemExit(1)
