"""Experiment 1: per-iteration dynamics of Huginn-0125 on multi-digit addition.

Runs the recurrent core block manually for K steps from several random initial states
(sampled with the model's own `initialize_state`, seeded per prompt x seed at the prompt's true
length) and records, after every step, at each answer-digit prediction position:

  state geometry    ||s_k||, ||s_k - s_{k-1}||, cos(s_k, s_{k-1}) on the raw recurrent output
                    (NB: the core block ends in RMSNorm, so raw norm is architecturally pinned),
                    and ||ln_f(s_k) - ln_f(s_{k-1})|| (the state the coda actually reads)
  readout           teacher-forced conditional: logp(target), argmax-correct, entropy, KL(p_k||p_{k-1});
                    log-probs over the 10 digit tokens are saved for every step (offline readout analysis)
  cross-seed        per seed pair: L2 distance, cosine, total-variation distance of output distributions

End-of-run per trajectory (two late windows, so slow convergence vs steady state can be told apart):
  lag profiles      D(p) for p = 1..max_lag: relative, absolute and angular (1 - cos) distances
  phase alignment   per seed pair, residual distance under the best constant time shift tau vs tau = 0
                    (same fixed point: both ~0; same orbit, other phase: best << tau=0; different: both large)
  PCA               seeded, fit on all steps and on the late window
  compact traj      fixed Gaussian random projection (64-d) of every answer-position state, for reanalysis

Greedy generation check (no teacher forcing, with termination) at selected depths, seed 0.

Teacher forcing: the full "prompt + answer" string is fed; logits at the token *before* each answer
digit predict that digit. With zero test-time noise and fixed initial states, the trajectory for
num_steps = r is the prefix of the K-step run, so one run gives results for every r <= K.

Each chunk is checkpointed to the `huginn-results` volume under results/<run_id>/.

Usage (from repo root):
  modal run huginn_addition/modal_app.py --mode smoke
  modal run huginn_addition/modal_app.py --mode pilot
  modal volume get huginn-results <run_id> results/
"""

from __future__ import annotations

import json
import os
import sys
import time

import modal

REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"  # pinned huginn-0125 revision (remote code + weights)
MODEL_ID = "tomg-group-umd/huginn-0125"

app = modal.App("huginn-addition-dynamics")
hf_cache = modal.Volume.from_name("huginn-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("huginn-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "numpy",
        "pandas",
        "pyarrow",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 60 * 60
)
def run_experiment(problems_json: str, cfg: dict, run_id: str):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    outdir = f"/results/{run_id}"
    os.makedirs(outdir, exist_ok=True)
    seeds, K, max_lag, shift_max = cfg["seeds"], cfg["K"], cfg["max_lag"], cfg["shift_max"]
    greedy_depths = cfg["greedy_depths"]
    S = len(seeds)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=getattr(torch, cfg.get("dtype", "bfloat16")), trust_remote_code=True
    )
    model = model.cuda().eval()
    hf_cache.commit()
    mdtype = next(model.parameters()).dtype
    print("model dtype", mdtype, flush=True)
    print(f"loaded model in {time.time() - t0:.1f}s", flush=True)
    dev = torch.device("cuda")
    assert model.config.test_time_noise == 0
    ln_f = model.transformer.ln_f

    problems = json.loads(problems_json)
    NEWLINE_ID = tok.encode("\n", add_special_tokens=False)[0]  # right-pad filler (causal attn => never attended)
    DIGITS = [tok.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
    SP_DIGITS = [tok.encode(f" {i}", add_special_tokens=False)[0] for i in range(10)]
    assert len(set(DIGITS)) == 10 and len(set(SP_DIGITS)) == 10

    for p in problems:
        pre = tok.encode(p["prompt"], add_special_tokens=True)
        full = tok.encode(p["prompt"] + p["answer"], add_special_tokens=True)
        assert full[: len(pre)] == pre, (p["prompt"], pre, full)
        ans = full[len(pre):]
        assert len(ans) == len(p["digit_info"]), (p["answer"], ans)
        assert ans[0] == SP_DIGITS[p["digit_info"][0]["digit"]]
        assert all(ans[j] == DIGITS[p["digit_info"][j]["digit"]] for j in range(1, len(ans)))
        p["ids"], p["pre_len"] = full, len(pre)
        p["pred_pos"] = list(range(len(pre) - 1, len(full) - 1))
        p["targets"] = ans

    # fixed random projection for compact trajectories
    d = model.config.n_embd
    gproj = torch.Generator(device=dev).manual_seed(12345)
    R = torch.randn((d, cfg["proj_dim"]), generator=gproj, device=dev) / cfg["proj_dim"] ** 0.5

    def init_state(e_row, uid, seed):
        """Model's own initializer, seeded, at the row's true length."""
        with torch.random.fork_rng(devices=[dev]):
            torch.manual_seed(int(seed) * 1_000_003 + int(uid))
            return model.initialize_state(e_row[None])[0]

    def coda_logits(s, gather_idx):
        L = s.shape[1]
        freqs = model.freqs_cis[:, :L]
        x = ln_f(s)
        bi = torch.tensor(0, dtype=torch.long)
        for block in model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        x = ln_f(x)
        h = torch.gather(x, 1, gather_idx[..., None].expand(-1, -1, x.shape[-1]))
        return model.lm_head(h).float()

    def build_batch(chunk, rows_seeds):
        """rows_seeds: list of (chunk_index, seed). Returns padded ids, init states, index tensors."""
        L = max(len(chunk[i]["ids"]) for i, _ in rows_seeds)
        P = max(len(chunk[i]["targets"]) for i, _ in rows_seeds)
        N = len(rows_seeds)
        ids = torch.full((N, L), NEWLINE_ID, dtype=torch.long)
        gidx = torch.zeros((N, P), dtype=torch.long)
        tgt = torch.zeros((N, P), dtype=torch.long)
        valid = torch.zeros((N, P), dtype=torch.bool)
        for r, (i, _) in enumerate(rows_seeds):
            p = chunk[i]
            ids[r, : len(p["ids"])] = torch.tensor(p["ids"])
            n = len(p["targets"])
            gidx[r, :n] = torch.tensor(p["pred_pos"])
            gidx[r, n:] = p["pred_pos"][-1]
            tgt[r, :n] = torch.tensor(p["targets"])
            valid[r, :n] = True
        ids, gidx, tgt, valid = ids.to(dev), gidx.to(dev), tgt.to(dev), valid.to(dev)
        e, _ = model.embed_inputs(ids)
        s = model.initialize_state(e)  # placeholder values at padded positions
        for r, (i, seed) in enumerate(rows_seeds):
            n = len(chunk[i]["ids"])
            s[r, :n] = init_state(e[r, :n], chunk[i]["uid"], seed)
        return ids, e, s, gidx, tgt, valid

    # ------------------------------------------------------------------ validation (hard check)
    @torch.no_grad()
    def validate():
        lens = sorted({len(p["ids"]) for p in problems})
        picks = []
        for Lp in (lens[0], lens[len(lens) // 2], lens[-1]):
            picks.append(next(p for p in problems if len(p["ids"]) == Lp))
        chunk = picks
        ids, e, s0, gidx, tgt, valid = build_batch(chunk, [(i, seeds[0]) for i in range(len(chunk))])
        report = []
        for depth in (4, 16):
            s = s0.clone()
            for k in range(depth):
                s, _, _ = model.iterate_one_step(e, s, current_step=k)
            ours = coda_logits(s, gidx)
            for r, p in enumerate(chunk):
                n = len(p["ids"])
                ref_ids = torch.tensor([p["ids"]], device=dev)
                ref = model(ref_ids, input_states=s0[r : r + 1, :n].clone(), num_steps=depth).logits.float()
                m = len(p["targets"])
                refp = ref[0, torch.tensor(p["pred_pos"], device=dev)]
                o = ours[r, :m]
                assert torch.isfinite(o).all() and torch.isfinite(refp).all()
                report.append({
                    "depth": depth, "len": n, "max_abs": float((o - refp).abs().max()),
                    "argmax_agree": float((o.argmax(-1) == refp.argmax(-1)).float().mean()),
                })
        print("validation", report, flush=True)
        agree = np.mean([r["argmax_agree"] for r in report])
        worst = max(r["max_abs"] for r in report)
        if agree < 0.95 or worst > 2.0:
            raise RuntimeError(f"manual loop disagrees with model.forward: {report}")
        return report

    validation = validate()

    # ------------------------------------------------------------------ main loop
    iu = torch.triu_indices(S, S, 1)
    n_pairs = iu.shape[1]
    windows = {"mid": (K // 2, 3 * K // 4), "late": (3 * K // 4, K)}  # (lo, hi], steps

    def flat_rows(arrs, chunk, valid_cpu, rows_seeds, extra):
        """Flatten (N,P) arrays over valid entries into a dict of columns."""
        r_idx, p_idx = np.nonzero(valid_cpu)
        out = {
            "uid": np.array([chunk[rows_seeds[r][0]]["uid"] for r in r_idx]),
            "seed": np.array([rows_seeds[r][1] for r in r_idx]),
            "pos": p_idx,
        }
        for k_, v in extra.items():
            out[k_] = np.full(len(r_idx), v)
        for k_, v in arrs.items():
            out[k_] = v[r_idx, p_idx]
        return out

    mem_peak = 0
    for ci, c0 in enumerate(range(0, len(problems), cfg["chunk_prompts"])):
        tc = time.time()
        torch.cuda.reset_peak_memory_stats()
        chunk = problems[c0 : c0 + cfg["chunk_prompts"]]
        B = len(chunk)
        rows_seeds = [(i, sd) for i in range(B) for sd in seeds]  # prompt-major
        N = len(rows_seeds)
        step_cols, pair_cols = [], []
        with torch.no_grad():
            ids, e, s, gidx, tgt, valid = build_batch(chunk, rows_seeds)
            P = gidx.shape[1]
            valid_cpu = valid.cpu().numpy()
            gsel = gidx[..., None].expand(-1, -1, d)
            hist = torch.empty((K + 1, N, P, d), dtype=mdtype, device=dev)
            hist[0] = torch.gather(s, 1, gsel)
            digit_lp = np.zeros((K, N, P, 10), dtype=np.float16)
            dig_ids = torch.tensor([SP_DIGITS] + [DIGITS] * (P - 1), device=dev)  # (P,10)
            prev_logp, prev_ln = None, ln_f(hist[0]).float()
            for k in range(1, K + 1):
                s, _, _ = model.iterate_one_step(e, s, current_step=k - 1)
                cur = torch.gather(s, 1, gsel)
                hist[k] = cur
                curf, prevf = cur.float(), hist[k - 1].float()
                cur_ln = ln_f(cur).float()
                logits = coda_logits(s, gidx)
                logp = F.log_softmax(logits, -1)
                arrs = {
                    "norm": curf.norm(dim=-1),
                    "delta": (curf - prevf).norm(dim=-1),
                    "cos_prev": F.cosine_similarity(curf, prevf, dim=-1),
                    "delta_ln": (cur_ln - prev_ln).norm(dim=-1),
                    "logp_target": torch.gather(logp, -1, tgt[..., None])[..., 0],
                    "correct": logp.argmax(-1) == tgt,
                    "argmax": logp.argmax(-1),
                    "entropy": -(logp.exp() * logp).sum(-1),
                    "kl_prev": (logp.exp() * (logp - prev_logp)).sum(-1) if prev_logp is not None
                    else torch.full(logp.shape[:2], float("nan"), device=dev),
                }
                digit_lp[k - 1] = torch.gather(logp, -1, dig_ids[None].expand(N, -1, -1)).cpu().numpy()
                arrs = {k_: v.cpu().numpy() for k_, v in arrs.items()}
                step_cols.append(flat_rows(arrs, chunk, valid_cpu, rows_seeds, {"step": np.int16(k)}))
                if S > 1:
                    cs = curf.view(B, S, P, d)
                    pr = logp.exp().view(B, S, P, -1)
                    pd_, pc_, tv_ = [], [], []
                    for a_, b_ in zip(iu[0].tolist(), iu[1].tolist()):
                        pd_.append((cs[:, a_] - cs[:, b_]).norm(dim=-1))
                        pc_.append(F.cosine_similarity(cs[:, a_], cs[:, b_], dim=-1))
                        tv_.append(0.5 * (pr[:, a_] - pr[:, b_]).abs().sum(-1))
                    pd_, pc_, tv_ = (torch.stack(x, 1).cpu().numpy() for x in (pd_, pc_, tv_))  # (B,pairs,P)
                    vb = valid_cpu.reshape(B, S, P)[:, 0]
                    bi_, pi_ = np.nonzero(vb)
                    for q in range(n_pairs):
                        pair_cols.append({
                            "uid": np.array([chunk[b]["uid"] for b in bi_]), "pos": pi_,
                            "pair": np.full(len(bi_), f"{seeds[int(iu[0, q])]}-{seeds[int(iu[1, q])]}"),
                            "step": np.full(len(bi_), k, dtype=np.int16),
                            "dist": pd_[bi_, q, pi_], "cos": pc_[bi_, q, pi_], "tv": tv_[bi_, q, pi_],
                        })
                prev_logp, prev_ln = logp, cur_ln

            # ---------- lag profiles over two late windows
            lag_cols = []
            for wname, (lo, hi) in windows.items():
                ks = torch.arange(lo + 1, hi + 1, device=dev)
                ks = ks[ks - max_lag >= 1]
                a = hist[ks].float()
                for lag in range(1, max_lag + 1):
                    b = hist[ks - lag].float()
                    dist = (a - b).norm(dim=-1)
                    arrs = {
                        "D_rel": (dist / a.norm(dim=-1)).mean(0).cpu().numpy(),
                        "D_abs": dist.mean(0).cpu().numpy(),
                        "D_ang": (1 - F.cosine_similarity(a, b, dim=-1)).mean(0).cpu().numpy(),
                    }
                    lag_cols.append(flat_rows(arrs, chunk, valid_cpu, rows_seeds, {"window": wname, "lag": np.int16(lag)}))
                del a

            # ---------- phase-aligned cross-seed comparison (late region)
            phase_cols = []
            if S > 1:
                lo, hi = K // 2 + shift_max, K - shift_max
                ks = torch.arange(lo, hi + 1, device=dev)
                hb = hist.view(K + 1, B, S, P, d)
                for q in range(n_pairs):
                    a_, b_ = int(iu[0, q]), int(iu[1, q])
                    A = hb[ks, :, a_].float()
                    An = A.norm(dim=-1)
                    res = []
                    for tau in range(-shift_max, shift_max + 1):
                        Bt = hb[ks + tau, :, b_].float()
                        res.append(((A - Bt).norm(dim=-1) / An).mean(0))  # (B,P)
                    res = torch.stack(res, 0)  # (taus,B,P)
                    best, arg = res.min(0)
                    r0 = res[shift_max]
                    vb = valid_cpu.reshape(B, S, P)[:, 0]
                    bi_, pi_ = np.nonzero(vb)
                    best, arg, r0 = best.cpu().numpy(), arg.cpu().numpy() - shift_max, r0.cpu().numpy()
                    phase_cols.append({
                        "uid": np.array([chunk[b]["uid"] for b in bi_]), "pos": pi_,
                        "pair": np.full(len(bi_), f"{seeds[a_]}-{seeds[b_]}"),
                        "resid_tau0": r0[bi_, pi_], "resid_best": best[bi_, pi_], "tau_best": arg[bi_, pi_],
                    })

            # ---------- distance to own final state
            fin = hist[K].float()
            fin_cols = []
            for k in range(K + 1):
                hk = hist[k].float()
                arrs = {
                    "rel_dist_final": ((hk - fin).norm(dim=-1) / fin.norm(dim=-1)).cpu().numpy(),
                    "cos_final": F.cosine_similarity(hk, fin, dim=-1).cpu().numpy(),
                }
                fin_cols.append(flat_rows(arrs, chunk, valid_cpu, rows_seeds, {"step": np.int16(k)}))

            # ---------- seeded PCA (all steps, late window) per (prompt, pos), pooled over seeds
            pca_cols = []
            hb = hist.view(K + 1, B, S, P, d)
            for i, p in enumerate(chunk):
                for jp in range(len(p["targets"])):
                    X = hb[1:, i, :, jp].float().reshape(K * S, d)
                    for basis, sel in (("all", slice(0, K)), ("late", slice(K // 2, K))):
                        Xf = hb[1:, i, :, jp][sel].float().reshape(-1, d)
                        mu = Xf.mean(0, keepdim=True)
                        torch.manual_seed(0)
                        _, Sv, V = torch.pca_lowrank(Xf - mu, q=6, center=False)
                        tot = float(((Xf - mu) ** 2).sum()) + 1e-12
                        evr = (Sv[:3] ** 2 / tot).cpu().numpy()
                        Z = ((X - mu) @ V[:, :3]).view(K, S, 3).cpu().numpy()
                        kk, ss = np.meshgrid(np.arange(1, K + 1), np.arange(S), indexing="ij")
                        pca_cols.append({
                            "uid": np.full(K * S, p["uid"]), "pos": np.full(K * S, jp), "basis": np.full(K * S, basis),
                            "seed": np.array(seeds)[ss.ravel()], "step": kk.ravel().astype(np.int16),
                            "pc1": Z[..., 0].ravel(), "pc2": Z[..., 1].ravel(), "pc3": Z[..., 2].ravel(),
                            "evr1": np.full(K * S, evr[0]), "evr2": np.full(K * S, evr[1]), "evr3": np.full(K * S, evr[2]),
                        })

            # ---------- compact trajectories (random projection)
            proj = np.stack([(hist[k].float() @ R).to(torch.float16).cpu().numpy() for k in range(K + 1)])  # (K+1,N,P,pd)
            del hist, hb, fin
            torch.cuda.empty_cache()

        def cat(cols):
            return pd.DataFrame({k_: np.concatenate([c[k_] for c in cols]) for k_ in cols[0]}) if cols else pd.DataFrame()

        for name, cols in (("steps", step_cols), ("pairs", pair_cols), ("lags", lag_cols), ("phase", phase_cols),
                           ("to_final", fin_cols), ("pca", pca_cols)):
            df = cat(cols)
            if len(df):
                df.to_parquet(f"{outdir}/{name}_c{ci:03d}.parquet", index=False)
        np.savez_compressed(
            f"{outdir}/traj_c{ci:03d}.npz", proj=proj, digit_logp=digit_lp, valid=valid_cpu,
            uids=np.array([chunk[i]["uid"] for i, _ in rows_seeds]), seeds=np.array([sd for _, sd in rows_seeds]),
        )
        results_vol.commit()
        peak = torch.cuda.max_memory_allocated() / 2**30
        mem_peak = max(mem_peak, peak)
        print(f"chunk {ci + 1}: {c0 + B}/{len(problems)} prompts, chunk {time.time() - tc:.0f}s, "
              f"total {time.time() - t0:.0f}s, peak mem {peak:.1f} GiB", flush=True)

    # ------------------------------------------------------------------ greedy generation check
    greedy_rows = []
    with torch.no_grad():
        for c0 in range(0, len(problems), cfg["greedy_chunk"]):
            chunk = problems[c0 : c0 + cfg["greedy_chunk"]]
            for depth in greedy_depths:
                gen = [[] for _ in chunk]
                done = [False] * len(chunk)
                max_new = max(len(p["targets"]) for p in chunk) + 1
                for t in range(max_new):
                    seqs = [p["ids"][: p["pre_len"]] + g for p, g in zip(chunk, gen)]
                    L = max(len(x) for x in seqs)
                    ids = torch.full((len(chunk), L), NEWLINE_ID, dtype=torch.long, device=dev)
                    for r, x in enumerate(seqs):
                        ids[r, : len(x)] = torch.tensor(x, device=dev)
                    e, _ = model.embed_inputs(ids)
                    s = model.initialize_state(e)
                    for r, x in enumerate(seqs):
                        s[r, : len(x)] = init_state(e[r, : len(x)], chunk[r]["uid"] * 1000 + t, seeds[0])
                    for k in range(depth):
                        s, _, _ = model.iterate_one_step(e, s, current_step=k)
                    last = torch.tensor([len(x) - 1 for x in seqs], device=dev)[:, None]
                    nxt = coda_logits(s, last)[:, 0].argmax(-1).tolist()
                    for r in range(len(chunk)):
                        if not done[r]:
                            gen[r].append(nxt[r])
                            if nxt[r] == NEWLINE_ID or not tok.decode([nxt[r]]).strip().isdigit():
                                done[r] = True
                    if all(done):
                        break
                for p, g in zip(chunk, gen):
                    text = tok.decode(g)
                    pred = text.split("\n")[0].strip()
                    terminated = len(g) > len(p["targets"]) and g[len(p["targets"])] == NEWLINE_ID
                    greedy_rows.append({
                        "uid": p["uid"], "depth": depth, "pred": pred, "target": str(p["c"]),
                        "exact": pred == str(p["c"]), "exact_and_terminated": pred == str(p["c"]) and terminated,
                    })
        print(f"greedy done, total {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(greedy_rows).to_parquet(f"{outdir}/greedy.parquet", index=False)

    meta = {"validation": validation, "cfg": cfg, "revision": REVISION, "runtime_s": time.time() - t0,
            "peak_mem_gib": mem_peak, "windows": windows, "run_id": run_id}
    with open(f"{outdir}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"{outdir}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta


MODES = {
    "smoke": dict(per_cell=1, n_digits=(2, 3), seeds=[0, 1], K=24, chunk_prompts=8, max_lag=4, shift_max=2,
                  proj_dim=64, greedy_depths=[4, 16], greedy_chunk=32),
    # precision control: identical problems/seeds, run once per dtype
    "prec_bf16": dict(per_cell=1, n_digits=(2, 3, 4), seeds=[0, 1], K=128, chunk_prompts=21, max_lag=24, shift_max=16,
                      proj_dim=64, greedy_depths=[], greedy_chunk=64, dtype="bfloat16"),
    "prec_fp32": dict(per_cell=1, n_digits=(2, 3, 4), seeds=[0, 1], K=128, chunk_prompts=21, max_lag=24, shift_max=16,
                      proj_dim=64, greedy_depths=[], greedy_chunk=64, dtype="float32"),
    "pilot": dict(per_cell=6, n_digits=(2, 3, 4), seeds=[0, 1, 2, 3], K=256, chunk_prompts=24, max_lag=48,
                  shift_max=32, proj_dim=64, greedy_depths=[4, 8, 16, 32, 64, 128], greedy_chunk=64),
}


@app.local_entrypoint()
def main(mode: str = "smoke"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import sample_stratified, to_json

    cfg = dict(MODES[mode])
    problems = sample_stratified(n_digits_list=cfg.pop("n_digits"), per_cell=cfg.pop("per_cell"), seed=0)
    run_id = f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"{run_id}: {len(problems)} prompts, cfg={cfg}")
    meta = run_experiment.remote(to_json(problems), cfg, run_id)
    print(json.dumps(meta, indent=1))
    print(f"fetch with: modal volume get huginn-results {run_id} results/")
