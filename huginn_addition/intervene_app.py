"""Modal app with every GPU experiment of the Huginn-0125 looped-LM study (fp32 unless stated).

Each experiment is one @app.function. Its configs are an UPPERCASE dict near the bottom (mode name -> cfg), and the
local entrypoint `main` dispatches on --mode. Run from huginn_addition/:
  ../.venv/bin/modal run intervene_app.py --mode <mode> [--probs ../results/<items>.json] [--dirs <npz>] [--pids ...]
Outputs go to the Modal volume `huginn-results` under <mode>-<timestamp>/ (meta.json + arrays). Fetch files one at a
time (bulk `modal volume get` hits HTTP 429 behind the proxy). Each run, its inputs, its analysis script and its
accepted results are documented in huginn_addition/README.md (experiment log); the synthesis is WRITEUP.md.

INDEX  function [config dict: notable modes] -- what it does -- analysis / downstream scripts
Circle geometry and late-state dynamics
  collect      [COLLECT]                -- late states for the circle fit                  -- fit_circle.py
  intervene    [INTERVENE]              -- pushes along circle chords vs controls          -- analyze_interventions.py
  sweep        [SWEEP]                  -- angular sweep in the fitted plane               -- analyze_sweep.py
  gradients    [GRAD]                   -- local readout gradients (coda autograd)         -- analyze_gradients.py, analyze_centered.py
  jacobian     [JAC]                    -- own-position Jacobian, Arnoldi, push decay      -- analyze_jacobian.py
  carry_traj   [CARRY]                  -- carry-row trajectories and decoded phase        -- analyze_carry.py, analyze_carry_audit.py
Period-2 cycles ("blinking") and stabilization
  orbit        [ORBIT: orbit, orbit64, screen] -- long runs (fp32/fp64); demo-set screen   -- analyze_orbit.py, analyze_screen.py
  cycsrc       [CYCSRC]                 -- cycle source token, hidden fixed point, pinning -- analyze_cycsrc.py
  bifur        [BIFUR]                  -- prelude-output interpolation (flip bifurcation) -- analyze_bifur.py
  census       [CENSUS]                 -- prevalence on fresh prompts                     -- analyze_census.py
  ownblock     [OWNBLOCK]               -- own-block spectra (selected / random positions) -- analyze_ownblock.py
  ownspec      [OWNSPEC]                -- spectrum convergence, multi-start Arnoldi       -- analyze_ownspec.py
  freegen      [FREEGEN]                -- greedy generation across loop budgets           -- analyze_freegen.py
  relax        [RELAX]                  -- eigenvalue-guided source under-relaxation       -- analyze_relax.py
  relaxsweep   [SWEEPR]                 -- relaxation threshold sweep                      -- analyze_sweep_relax.py
  mcq          [MCQ: mcq_base, mcq_relax, ptr_screen, carry_chain]
                                        -- ARC stability / global relaxation; pointer and carry-chain screens
                                        -- analyze_mcq.py, analyze_pointer_screen.py, analyze_carry_chain.py
Uncertainty, settling, other cyclic concepts
  doubt        [DOUBT]                  -- is uncertainty a weak pull?                     -- analyze_doubt.py
  settle       [SETTLE]                 -- settling speed vs loops needed                  -- analyze_settle.py
  wkcollect    [WEEKDAY]                -- weekday pilot                                   -- analyze_weekday.py, analyze_weekday_start.py
  dgcollect    [DIGROT]                 -- digit-circle rotation pilot                     -- analyze_digit_rotation.py
  mblend       [MBLEND]                 -- month shape and opposite-month blends           -- analyze_month_blend.py
Steering and truth
  steer        [STEER]                  -- one-shot vs persistent (context) edits          -- analyze_steer.py
  truth        [TRUTH]                  -- truth probes (reading)                          -- analyze_truth.py
  truthsteer   [TSTEER: tsteer, tsteer_ans] -- probe-direction steering                    -- analyze_truthsteer.py
  respond      [RESPOND]                -- linear-response (resolvent) test                -- analyze_respond.py
  dstar_vjp    [DSTAR]                  -- 48-step resolvent pullbacks (d*)                -- make_dstar_pushes.py
  dstar_eval   [DSEVAL: dseval, tlever_eval] -- held-out True/False steering (+ off-TF KL) -- analyze_dstar.py, analyze_tlever.py, analyze_selector.py
  ctrl_eval    [CTRL: ctrl_formats, ctrl_cb, ctrl_dose] -- d* format / dose / pulse controls -- analyze_ctrl.py, analyze_ctrl_cb.py
Levers and the readout-reversal mechanism
  dglever      [DGLEVER: dglever_train/pred/eval, adlever_eval, ovl_eval]
                                        -- digit levers; cheap-lever and overlap-control evals
                                        -- analyze_dglever.py, analyze_adlever.py, analyze_overlap.py
  revmech      [REVMECH]                -- Arnoldi impulse responses from each push        -- analyze_revmech.py
  rtransfer    [RTRANSFER]              -- readout transfer operator Q^T J                 -- analyze_rtransfer.py
  blockdecomp  [BLOCKDECOMP]            -- exact one-step path decomposition               -- analyze_blockdecomp.py
  adaptcov     [ADAPTCOV]               -- adapter / one-step pullbacks (digits)           -- analyze_adaptcov.py, make_adlever_pushes.py
  truthcov     [TRUTHCOV]               -- one-step / adapter pullbacks + timings (truth)  -- make_tlever_pushes.py
(The first pilot, per-iteration dynamics on addition, lives in modal_app.py.)

Implementation notes: most functions check the manual loop against model.forward (H.validate). The pinned
iterate_one_step is decorated @torch.no_grad, so autograd (VJPs) goes through core_block_forward after an equivalence
check. Later entry branches record cfg["code_sha256"] in each run's meta.json (the early runs predate this), and the
dglever / dstar_eval branches also record cfg["push_sha256"] of the push file.

Original phases (collect / intervene), as first written:

Phase 1 `collect`: full 5280-d late states (mean over steps W0..K1, fp32) at every answer position for a fit set of
problems, plus the final digit readout. Saved to the `huginn-results` volume. Local `fit_circle.py` then fits the circle
plane in full state space (nuisance-adjusted) and builds control directions.

Phase 2 `intervene`: on held-out problems, at ONE answer position per prompt (pos >= 1, chosen in advance), add a
displacement to that position's recurrent state and measure the digit readout:
  readout-only : add to the converged state, read out through the coda immediately (no further recurrence)
  dynamic-mid  : add at step T_MID, keep iterating to K, read out at checkpoints (does the push survive the dynamics?)
  dynamic-final: add to the converged state at step K, iterate 32 more steps (does the state snap back?)
Directions (per row, by the row's true digit d; analysis keeps baseline-correct rows):
  circle±k  : U^T [cos θ_{d±k} − cos θ_d, sin θ_{d±k} − sin θ_d], θ_d = 2πd/10, U = fitted cos/sin coefficient vectors
  random_i  : fixed Gaussian direction, rescaled per row to ||circle+1||
  pca_i     : random combination of the top-50 PCs of fit-set states, rescaled to ||circle+1||
  scr_i     : one notch along a circle fitted with scrambled digit order π_i, rescaled to ||circle+1||
then scaled by alpha. Retention = <s_t − s_ref_t, Δ> / <Δ, Δ> (1 = push fully present, 0 = gone), with s_ref the
unperturbed trajectory at the same step.

Usage (repo root):
  modal run huginn_addition/intervene_app.py --mode collect_smoke | collect
  python -I huginn_addition/fit_circle.py results/<collect_run>          # writes directions.npz
  modal run huginn_addition/intervene_app.py --mode intervene_smoke --dirs results/<collect_run>/directions.npz
  modal run huginn_addition/intervene_app.py --mode intervene --dirs results/<collect_run>/directions.npz
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

import modal

REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
MODEL_ID = "tomg-group-umd/huginn-0125"

app = modal.App("huginn-circle-intervention")
hf_cache = modal.Volume.from_name("huginn-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("huginn-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "safetensors", "huggingface_hub[hf_transfer]", "numpy",
                 "pandas", "pyarrow")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


# ----------------------------------------------------------------------------- shared helpers (run in container)
class H:
    def __init__(self, dtype="float32", init_fp32=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=getattr(torch, dtype), trust_remote_code=True).cuda().eval()
        assert self.model.config.test_time_noise == 0
        self.dev = torch.device("cuda")
        self.ln_f = self.model.transformer.ln_f
        self.NL = self.tok.encode("\n", add_special_tokens=False)[0]
        self.DIG = [self.tok.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
        self.SPD = [self.tok.encode(f" {i}", add_special_tokens=False)[0] for i in range(10)]
        self.d = self.model.config.n_embd
        self.init_fp32 = init_fp32

    def prep(self, problems):
        for p in problems:
            pre = self.tok.encode(p["prompt"], add_special_tokens=True)
            full = self.tok.encode(p["prompt"] + p["answer"], add_special_tokens=True)
            assert full[: len(pre)] == pre
            ans = full[len(pre):]
            assert len(ans) == len(p["digit_info"])
            assert ans[0] == self.SPD[p["digit_info"][0]["digit"]]
            assert all(ans[j] == self.DIG[p["digit_info"][j]["digit"]] for j in range(1, len(ans)))
            p["ids"], p["pred_pos"] = full, list(range(len(pre) - 1, len(full) - 1))

    def init_state(self, e_row, uid, seed):
        torch = self.torch
        with torch.random.fork_rng(devices=[self.dev]):
            torch.manual_seed(int(seed) * 1_000_003 + int(uid))
            if self.init_fp32:
                return self.model.initialize_state(e_row[None].float())[0].to(e_row.dtype)
            return self.model.initialize_state(e_row[None])[0]

    def batch(self, plist, seeds_per_row):
        torch = self.torch
        L = max(len(p["ids"]) for p in plist)
        ids = torch.full((len(plist), L), self.NL, dtype=torch.long)
        for r, p in enumerate(plist):
            ids[r, : len(p["ids"])] = torch.tensor(p["ids"])
        ids = ids.to(self.dev)
        e, _ = self.model.embed_inputs(ids)
        s = self.model.initialize_state(e)
        for r, (p, sd) in enumerate(zip(plist, seeds_per_row)):
            n = len(p["ids"])
            s[r, :n] = self.init_state(e[r, :n], p["uid"], sd)
        return ids, e, s

    def step(self, e, s, k):
        s, _, _ = self.model.iterate_one_step(e, s, current_step=k)
        return s

    def coda_logits(self, s, gidx):
        torch = self.torch
        L = s.shape[1]
        freqs = self.model.freqs_cis[:, :L]
        x = self.ln_f(s)
        bi = torch.tensor(0, dtype=torch.long)
        for block in self.model.transformer.coda:
            bi -= 1
            x = block(x, freqs, bi, None, None)
        x = self.ln_f(x)
        h = torch.gather(x, 1, gidx[..., None].expand(-1, -1, x.shape[-1]))
        return self.model.lm_head(h).float()

    def validate(self, problems, tol=1e-3):
        """Padded mixed-length batch through the manual loop vs individual unpadded model.forward calls (hard check)."""
        torch = self.torch
        lens = sorted({len(p["ids"]) for p in problems})
        picks = [next(p for p in problems if len(p["ids"]) == Lp) for Lp in (lens[0], lens[len(lens) // 2], lens[-1])]
        with torch.no_grad():
            ids, e, s0 = self.batch(picks, [0] * len(picks))
            s = s0.clone()
            for k in range(16):
                s = self.step(e, s, k)
            P = max(len(p["pred_pos"]) for p in picks)
            g = torch.tensor([p["pred_pos"] + [p["pred_pos"][-1]] * (P - len(p["pred_pos"])) for p in picks], device=self.dev)
            ours = self.coda_logits(s, g)
            rep = []
            for r, p in enumerate(picks):
                n, m = len(p["ids"]), len(p["pred_pos"])
                ref = self.model(ids[r : r + 1, :n], input_states=s0[r : r + 1, :n].clone(), num_steps=16).logits.float()
                ref = ref[0, p["pred_pos"]]
                rep.append({"len": n, "max_abs": float((ours[r, :m] - ref).abs().max()),
                            "argmax_agree": float((ours[r, :m].argmax(-1) == ref.argmax(-1)).float().mean())})
        if any(r["argmax_agree"] < 1.0 or r["max_abs"] > tol for r in rep):
            raise RuntimeError(f"validation failed: {rep}")
        return rep


# ----------------------------------------------------------------------------- phase 1
@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def collect(problems_json: str, cfg: dict, run_id: str):
    import numpy as np
    import torch

    t0 = time.time()
    h = H("float32")
    hf_cache.commit()
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    K1, W0, seeds = cfg["K"], cfg["W0"], cfg["seeds"]
    states, logp, rows = [], [], []
    with torch.no_grad():
        for c0 in range(0, len(problems), cfg["chunk"]):
            chunk = problems[c0 : c0 + cfg["chunk"]]
            rs = [(p, sd) for p in chunk for sd in seeds]
            ids, e, s = h.batch([p for p, _ in rs], [sd for _, sd in rs])
            P = max(len(p["pred_pos"]) for p in chunk)
            gidx = torch.zeros((len(rs), P), dtype=torch.long, device=h.dev)
            for r, (p, _) in enumerate(rs):
                pp = p["pred_pos"] + [p["pred_pos"][-1]] * (P - len(p["pred_pos"]))
                gidx[r] = torch.tensor(pp, device=h.dev)
            gsel = gidx[..., None].expand(-1, -1, h.d)
            acc = torch.zeros((len(rs), P, h.d), device=h.dev)
            for k in range(K1):
                s = h.step(e, s, k)
                if k + 1 >= W0:
                    acc += torch.gather(s, 1, gsel).float()
            acc /= K1 - W0 + 1
            lg = torch.log_softmax(h.coda_logits(s, gidx), -1)
            dl = torch.stack([lg[:, 0][:, h.SPD]] + [lg[:, j][:, h.DIG] for j in range(1, P)], 1)  # (N,P,10)
            acc, dl = acc.cpu().numpy(), dl.cpu().numpy()
            for r, (p, sd) in enumerate(rs):
                for j in range(len(p["pred_pos"])):
                    states.append(acc[r, j].astype(np.float32))
                    logp.append(dl[r, j].astype(np.float32))
                    rows.append((p["uid"], sd, j))
            print(f"collect {c0 + len(chunk)}/{len(problems)} {time.time() - t0:.0f}s, peak "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    out = f"/results/{run_id}"
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(f"{out}/states.npz", states=np.stack(states), digit_logp=np.stack(logp),
                        rows=np.array(rows, dtype=np.int64))
    with open(f"{out}/problems.json", "w") as f:
        f.write(problems_json)
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.time() - t0, "run_id": run_id, "revision": REVISION}
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    results_vol.commit()
    return meta


# ----------------------------------------------------------------------------- phase 2
@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=3 * 3600)
def intervene(problems_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """Kinds: circle{k} (one chord d -> d+k on the fitted ellipse, own norm); radial (outward along f(d)U);
    random_i, pca_i (generic directions); scrT{k}_i (chord toward d+k on a scrambled-order plane: (f(pi(d+k)) - f(pi(d)))U_pi).
    Every non-circle kind is rescaled per row to the norm of its reference chord (`@circle{k}` suffix; default circle1,
    scrT{k} defaults to circle{k}). Conditions may set norm-preserving mode (state norm at the position kept fixed)."""
    import numpy as np
    import pandas as pd
    import torch

    t0 = time.time()
    h = H("float32")
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    D = np.load(io.BytesIO(dirs_npz))
    U = torch.tensor(D["U"], device=h.dev)                    # (2, d)
    Upinv = torch.linalg.pinv(U.T)                            # (2, d): plane coordinates of a state
    Uperm = torch.tensor(D["U_perm"], device=h.dev)           # (S, 2, d)
    perms = D["perms"]                                        # (S, 10)
    G = torch.tensor(D["random"], device=h.dev)
    PC = torch.tensor(D["pca_dirs"], device=h.dev)
    K, TM, TF = cfg["K"], cfg["T_MID"], cfg["T_FINAL_EXTRA"]
    th = lambda x: 2 * np.pi * np.asarray(x) / 10.0
    f_ = lambda x: np.stack([np.cos(th(x)), np.sin(th(x))], 1)

    def chord(Umat, a_pos, b_pos):
        return torch.tensor(f_(b_pos) - f_(a_pos), device=h.dev, dtype=torch.float32) @ Umat

    def delta(kind, digits):
        d = np.asarray(digits)
        base, _, ref = kind.partition("@")
        if base == "none":
            return torch.zeros((len(d), h.d), device=h.dev)
        if base.startswith("circle"):
            return chord(U, d, d + int(base[6:]))
        if base.startswith("scrT"):
            k_str, i_str = base[4:].split("_")
            k, i = int(k_str), int(i_str)
            ref = ref or f"circle{k}"
            v = chord(Uperm[i], perms[i][d % 10], perms[i][(d + k) % 10])
        elif base == "radial":
            v = torch.tensor(f_(d), device=h.dev, dtype=torch.float32) @ U
        elif base.startswith("random"):
            v = G[int(base.split("_")[1])][None].expand(len(d), -1)
        elif base.startswith("pca"):
            v = PC[int(base.split("_")[1])][None].expand(len(d), -1)
        else:
            raise ValueError(kind)
        rn = delta(ref or "circle1", digits).norm(dim=-1, keepdim=True)
        return v / v.norm(dim=-1, keepdim=True) * rn

    records = []
    all_ckpts = sorted(set(cfg["mid_checkpoints"]) | set(cfg["final_checkpoints"]) | {TM, K, K + TF})
    with torch.no_grad():
        for c0 in range(0, len(problems), cfg["chunk"]):
            tc = time.time()
            chunk = problems[c0 : c0 + cfg["chunk"]]
            N = len(chunk)
            ids, e, s = h.batch(chunk, [0] * N)
            ipos = [p["ipos"] for p in chunk]
            pos_tok = torch.tensor([p["pred_pos"][j] for p, j in zip(chunk, ipos)], device=h.dev)
            gidx = pos_tok[:, None]
            digits = [p["digit_info"][j]["digit"] for p, j in zip(chunk, ipos)]
            rix = torch.arange(N, device=h.dev)
            dig_t = torch.tensor(h.DIG, device=h.dev)

            def readout(st):
                lg = torch.log_softmax(h.coda_logits(st, gidx)[:, 0], -1)  # full vocab
                dl = lg[:, dig_t]
                return dl.cpu().numpy(), lg.argmax(-1).cpu().numpy(), torch.logsumexp(dl, -1).cpu().numpy()

            def rec(timing, kind, alpha, normp, step, ro, extra):
                dl, fam, dmass = ro
                for n, p in enumerate(chunk):
                    r = {"uid": p["uid"], "pid": p["pid"], "demo": p["demo"], "ipos": ipos[n], "digit": digits[n],
                         "timing": timing, "kind": kind, "alpha": alpha, "norm_preserve": normp, "step": step,
                         "full_argmax_tok": int(fam[n]), "full_argmax_is_digit": bool(int(fam[n]) in h.DIG),
                         "digit_argmax": int(dl[n].argmax()), "log_digit_mass": float(dmass[n])}
                    r.update({k_: (float(v[n]) if v is not None else np.nan) for k_, v in extra.items()})
                    r.update({f"lp{q}": float(dl[n, q]) for q in range(10)})
                    records.append(r)

            # ---- baseline trajectory with readouts at every checkpoint
            base_pos, base_ro = {}, {}
            prev_pos = s[rix, pos_tok].clone()
            for k in range(K + TF):
                s = h.step(e, s, k)
                cur = s[rix, pos_tok].clone()
                base_pos[k + 1] = cur
                if k + 1 == TM:
                    s_mid = s.clone()
                if k + 1 == K:
                    s_fin = s.clone()
                if k + 1 in all_ckpts:
                    base_ro[k + 1] = readout(s)
                    coords = (cur @ Upinv.T).cpu().numpy()
                    step_change = ((cur - prev_pos).norm(dim=-1) / cur.norm(dim=-1)).cpu().numpy()
                    rec("baseline", "none", 0.0, False, k + 1, base_ro[k + 1],
                        {"plane_c": coords[:, 0], "plane_s": coords[:, 1], "step_change": step_change})
                prev_pos = cur

            def push(s_start, kind, alpha, normp):
                dl = delta(kind, digits) * alpha
                st = s_start.clone()
                orig = st[rix, pos_tok].clone()
                new = orig + dl
                if normp:
                    new = new * (orig.norm(dim=-1, keepdim=True) / new.norm(dim=-1, keepdim=True))
                st[rix, pos_tok] = new
                return st, new - orig

            def dev_stats(st, step, applied):
                dev = st[rix, pos_tok] - base_pos[step]
                dd = (applied * applied).sum(-1)
                ok = dd > 0
                ret = torch.where(ok, (dev * applied).sum(-1) / dd.clamp_min(1e-30), torch.full_like(dd, float("nan")))
                rel = torch.where(ok, dev.norm(dim=-1) / dd.clamp_min(1e-30).sqrt(), torch.full_like(dd, float("nan")))
                coords = (st[rix, pos_tok] @ Upinv.T).cpu().numpy()
                return {"retention": ret.cpu().numpy(), "rel_dev": rel.cpu().numpy(), "abs_dev": dev.norm(dim=-1).cpu().numpy(),
                        "applied_norm": dd.sqrt().cpu().numpy(), "plane_c": coords[:, 0], "plane_s": coords[:, 1]}

            # ---- readout-only (converged state at K, no further recurrence)
            for kind in cfg["readout_kinds"]:
                for alpha in cfg["readout_alphas"]:
                    for normp in cfg["readout_normp"]:
                        st, applied = push(s_fin, kind, alpha, normp)
                        rec("readout", kind, alpha, normp, K, readout(st), dev_stats(st, K, applied))

            # ---- dynamic (push once, keep iterating)
            def dynamic(timing, s_start, t_start, t_end, kind, alpha, normp, checkpoints):
                st, applied = push(s_start, kind, alpha, normp)
                rec(timing, kind, alpha, normp, t_start, readout(st), dev_stats(st, t_start, applied))
                for k in range(t_start, t_end):
                    st = h.step(e, st, k)
                    if k + 1 in checkpoints:
                        rec(timing, kind, alpha, normp, k + 1, readout(st), dev_stats(st, k + 1, applied))

            for kind, alpha, normp in cfg["mid_conds"]:
                dynamic("mid", s_mid, TM, K, kind, alpha, normp, set(cfg["mid_checkpoints"]))
            for kind, alpha, normp in cfg["final_conds"]:
                dynamic("final", s_fin, K, K + TF, kind, alpha, normp, set(cfg["final_checkpoints"]))
            print(f"intervene {c0 + N}/{len(problems)}: chunk {time.time() - tc:.0f}s, total {time.time() - t0:.0f}s, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            out = f"/results/{run_id}"
            os.makedirs(out, exist_ok=True)
            pd.DataFrame(records).to_parquet(f"{out}/interventions.parquet", index=False)  # checkpoint per chunk
            results_vol.commit()

    with open(f"/results/{run_id}/problems.json", "w") as f:
        f.write(problems_json)
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.time() - t0, "run_id": run_id, "revision": REVISION}
    with open(f"/results/{run_id}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    results_vol.commit()
    return meta


# ----------------------------------------------------------------------------- phase 3: angular sweep (readout-only)
@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def sweep(problems_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """Readout-only pushes on the converged state (step K) at one answer position per prompt.
    angle: v(phi) = [cos phi, sin phi] U with phi = theta_d + phi_rel, scaled per row to size x ||circle+1 chord||.
    scr parts: each target-matched scrambled chord (shift ±1) applied in full, as its orthogonal projection onto the
    true circle plane span(U), and as the orthogonal residual (each at its natural size x alpha)."""
    import numpy as np
    import pandas as pd
    import torch

    t0 = time.time()
    h = H("float32")
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    D = np.load(io.BytesIO(dirs_npz))
    U = torch.tensor(D["U"], device=h.dev)
    Q, _ = torch.linalg.qr(U.T)                      # (d, 2) orthonormal basis of the circle plane
    Uperm = torch.tensor(D["U_perm"], device=h.dev)
    perms = D["perms"]
    K = cfg["K"]
    th = lambda x: 2 * np.pi * np.asarray(x) / 10.0
    f_ = lambda x: np.stack([np.cos(th(x)), np.sin(th(x))], 1)
    T = lambda a: torch.tensor(a, device=h.dev, dtype=torch.float32)
    records = []
    with torch.no_grad():
        for c0 in range(0, len(problems), cfg["chunk"]):
            tc = time.time()
            chunk = problems[c0 : c0 + cfg["chunk"]]
            N = len(chunk)
            ids, e, s = h.batch(chunk, [0] * N)
            ipos = [p["ipos"] for p in chunk]
            pos_tok = torch.tensor([p["pred_pos"][j] for p, j in zip(chunk, ipos)], device=h.dev)
            gidx = pos_tok[:, None]
            d = np.array([p["digit_info"][j]["digit"] for p, j in zip(chunk, ipos)])
            rix = torch.arange(N, device=h.dev)
            dig_t = torch.tensor(h.DIG, device=h.dev)
            for k in range(K):
                s = h.step(e, s, k)
            ref = (T(f_(d + 1) - f_(d)) @ U).norm(dim=-1, keepdim=True)  # ||circle+1 chord|| per row

            def readout(st):
                lg = torch.log_softmax(h.coda_logits(st, gidx)[:, 0], -1)
                dl = lg[:, dig_t]
                return dl.cpu().numpy(), lg.argmax(-1).cpu().numpy(), torch.logsumexp(dl, -1).cpu().numpy()

            def rec(kind, extra, ro, applied):
                dl, fam, dm = ro
                an = applied.norm(dim=-1).cpu().numpy()
                for n, p in enumerate(chunk):
                    r = {"uid": p["uid"], "pid": p["pid"], "demo": p["demo"], "ipos": ipos[n], "digit": int(d[n]), "kind": kind,
                         "full_argmax_tok": int(fam[n]), "full_argmax_is_digit": bool(int(fam[n]) in h.DIG),
                         "digit_argmax": int(dl[n].argmax()), "log_digit_mass": float(dm[n]), "applied_norm": float(an[n])}
                    r.update(extra)
                    r.update({f"lp{q}": float(dl[n, q]) for q in range(10)})
                    records.append(r)

            def apply(v):
                st = s.clone()
                st[rix, pos_tok] = st[rix, pos_tok] + v
                return st

            rec("baseline", {}, readout(s), torch.zeros((N, 1), device=h.dev))
            for size in cfg["sizes"]:
                for phi_rel in cfg["angles_deg"]:
                    phi = th(d) + np.deg2rad(phi_rel)
                    v = T(np.stack([np.cos(phi), np.sin(phi)], 1)) @ U
                    v = v / v.norm(dim=-1, keepdim=True) * ref * size
                    rec("angle", {"phi_rel": float(phi_rel), "size": float(size)}, readout(apply(v)), v)
            for i in range(len(perms)):
                for shift in (1, -1):
                    full = T(f_(perms[i][(d + shift) % 10]) - f_(perms[i][d % 10])) @ Uperm[i]
                    full = full / full.norm(dim=-1, keepdim=True) * ref  # same norm as in the intervention run
                    inpl = (full @ Q) @ Q.T
                    parts = {"full": full, "inplane": inpl, "residual": full - inpl}
                    for alpha in cfg["scr_alphas"]:
                        for part, v in parts.items():
                            rec("scr_part", {"scr": i, "shift": shift, "part": part, "size": float(alpha),
                                             "inplane_frac": float(((inpl.norm(dim=-1) / full.norm(dim=-1)) ** 2).mean())},
                                readout(apply(v * alpha)), v * alpha)
            print(f"sweep {c0 + N}/{len(problems)}: chunk {time.time() - tc:.0f}s, total {time.time() - t0:.0f}s", flush=True)
            out = f"/results/{run_id}"
            os.makedirs(out, exist_ok=True)
            pd.DataFrame(records).to_parquet(f"{out}/sweep.parquet", index=False)
            results_vol.commit()
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.time() - t0, "run_id": run_id, "revision": REVISION}
    with open(f"/results/{run_id}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"/results/{run_id}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta


# ----------------------------------------------------------------------------- phase 4: local readout gradients
@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def gradients(problems_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """At the converged state (step K), one answer position per prompt: exact gradients of the 10 digit log-probs w.r.t.
    that position's recurrent state (autograd through coda), projected onto the circle plane (orthonormal Q and circle
    coordinates U), plus their full-space norms; and central finite differences along the two orthonormal plane axes
    at several epsilons (validation / curvature check)."""
    import numpy as np
    import torch

    t0 = time.time()
    h = H("float32")
    for prm in h.model.parameters():
        prm.requires_grad_(False)
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    D = np.load(io.BytesIO(dirs_npz))
    U = torch.tensor(D["U"], device=h.dev)
    Q, _ = torch.linalg.qr(U.T)  # (d, 2)
    K, eps_list = cfg["K"], cfg["eps"]
    dig_t = torch.tensor(h.DIG, device=h.dev)
    out = {k: [] for k in ("uid", "pid", "ipos", "digit", "base_lp", "g_Q", "g_U", "g_norm", "fd_lp", "full_argmax_is_digit")}
    for c0 in range(0, len(problems), cfg["chunk"]):
        tc = time.time()
        chunk = problems[c0 : c0 + cfg["chunk"]]
        N = len(chunk)
        with torch.no_grad():
            ids, e, s = h.batch(chunk, [0] * N)
            for k in range(K):
                s = h.step(e, s, k)
        ipos = [p["ipos"] for p in chunk]
        pos_tok = torch.tensor([p["pred_pos"][j] for p, j in zip(chunk, ipos)], device=h.dev)
        gidx = pos_tok[:, None]
        rix = torch.arange(N, device=h.dev)
        s = s.detach()
        delta = torch.zeros((N, h.d), device=h.dev, requires_grad=True)
        with torch.enable_grad():
            st = s.index_put((rix, pos_tok), s[rix, pos_tok] + delta)
            full = torch.log_softmax(h.coda_logits(st, gidx)[:, 0], -1)
            lp = full[:, dig_t]
            grads = []
            for q in range(10):
                (g,) = torch.autograd.grad(lp[:, q].sum(), delta, retain_graph=q < 9)
                grads.append(g)
        G = torch.stack(grads, 1).detach()  # (N, 10, d)
        with torch.no_grad():
            fd = torch.zeros((N, len(eps_list), 2, 2, 10), device=h.dev)
            for ei, eps in enumerate(eps_list):
                for ax in range(2):
                    for si, sg in enumerate((1.0, -1.0)):
                        st2 = s.clone()
                        st2[rix, pos_tok] += sg * eps * Q[:, ax]
                        fd[:, ei, ax, si] = torch.log_softmax(h.coda_logits(st2, gidx)[:, 0], -1)[:, dig_t]
            out["uid"] += [p["uid"] for p in chunk]
            out["pid"] += [p["pid"] for p in chunk]
            out["ipos"] += ipos
            out["digit"] += [p["digit_info"][j]["digit"] for p, j in zip(chunk, ipos)]
            out["base_lp"].append(lp.detach().cpu().numpy())
            out["full_argmax_is_digit"] += [int(t) in h.DIG for t in full.argmax(-1).tolist()]
            out["g_Q"].append((G @ Q).cpu().numpy())
            out["g_U"].append((G @ U.T).cpu().numpy())
            out["g_norm"].append(G.norm(dim=-1).cpu().numpy())
            out["fd_lp"].append(fd.cpu().numpy())
        del G, grads, st, full, lp
        torch.cuda.empty_cache()
        print(f"gradients {c0 + N}/{len(problems)}: chunk {time.time() - tc:.0f}s, total {time.time() - t0:.0f}s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    res = {k: (np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v)) for k, v in out.items()}
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    np.savez_compressed(f"{o}/gradients.npz", eps=np.array(eps_list), Q_basis_from="qr(U.T)", **res)
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.time() - t0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"{o}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def jacobian(problems_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """Local dynamics of ONE answer position's recurrent state around the step-K state (fp32).

    Attention is causal and only position p is perturbed, so positions < p are unchanged and, to first order,
    delta_p(k+1) = J_pp(k) delta_p(k) exactly: J_pp (d x d) governs a single-position push. Matrix-vector products
    J v are central finite differences of one recurrent step: [step(s + eps v e_p) - step(s - eps v e_p)]_p / 2 eps.
      1. in-plane block B = Q^T J Q (Q orthonormal basis of the circle plane; column convention x' = B x), Gram of J Q
         (max gain over the plane), out-of-plane leak, at eps and a small eps (FD check); also at later checkpoints
         along the unperturbed trajectory (time variation); drift of the reference trajectory and of earlier positions
      2. Arnoldi (m Krylov vectors, random start) for the leading spectrum of J_pp at step K; overlap of the leading
         Ritz vectors with the circle plane; Ritz residual estimates
      3. nonlinear propagation of pushes applied at step K (circle+1, circle-1 chords, norm-matched random), sizes
         alpha * ||chord||, for T steps: in-plane (Q) coordinates and out-of-plane norm of s_pushed - s_ref each step
    """
    import numpy as np
    import torch

    t0 = time.time()
    h = H("float32")
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    D = np.load(io.BytesIO(dirs_npz))
    U = torch.tensor(D["U"], device=h.dev, dtype=torch.float32)
    Q, _ = torch.linalg.qr(U.T)  # (d, 2)
    rand0 = torch.tensor(D["random"][0], device=h.dev, dtype=torch.float32)
    K, T, m, eps, eps_small, alphas = cfg["K"], cfg["T"], cfg["m"], cfg["eps"], cfg["eps_small"], cfg["alphas"]
    dig_t = torch.tensor(h.DIG, device=h.dev)
    ck = [c for c in cfg["block_checkpoints"] if c <= T]
    keys = ("uid", "pid", "ipos", "digit", "carry_in", "base_correct", "drift", "prefix_drift", "ref_drift", "B", "B_small",
            "JQ_gram", "gainQ", "leakQ", "B_ck", "JQ_gram_ck", "ritz", "ritz_plane_overlap", "ritz_resid", "ritz_fd_resid",
            "prop_q", "prop_out", "push_q", "push_norm")
    out = {k: [] for k in keys}
    torch.manual_seed(cfg.get("arnoldi_seed", 0))
    with torch.no_grad():
        for c0 in range(0, len(problems), cfg["chunk"]):
            tc = time.time()
            chunk = problems[c0 : c0 + cfg["chunk"]]
            N = len(chunk)
            ids, e, s = h.batch(chunk, [0] * N)
            for k in range(K - 1):
                s = h.step(e, s, k)
            s_prev = s
            s = h.step(e, s, K - 1)  # state after K steps
            ipos = [p["ipos"] for p in chunk]
            pos_tok = torch.tensor([p["pred_pos"][j] for p, j in zip(chunk, ipos)], device=h.dev)
            rix = torch.arange(N, device=h.dev)
            sp = s[rix, pos_tok]
            drift = ((sp - s_prev[rix, pos_tok]).norm(dim=-1) / sp.norm(dim=-1)).cpu().numpy()
            # earlier positions (which force position p) still moving? mean relative last-step change over positions < p
            rel = ((s - s_prev).norm(dim=-1) / s.norm(dim=-1))  # (N, L)
            prefix_drift = np.array([float(rel[r_, : int(pos_tok[r_])].mean()) for r_ in range(N)])
            lp = torch.log_softmax(h.coda_logits(s, pos_tok[:, None])[:, 0], -1)
            digits = torch.tensor([p["digit_info"][j]["digit"] for p, j in zip(chunk, ipos)], device=h.dev)
            base_correct = (lp.argmax(-1) == dig_t[digits]).cpu().numpy()

            def jv(v, ep=eps, base=None, kk=K):  # v: (N, d) -> J_pp v (N, d) at state `base` (after kk steps)
                base = s if base is None else base
                a = base.clone(); a[rix, pos_tok] += ep * v
                b = base.clone(); b[rix, pos_tok] -= ep * v
                return (h.step(e, a, kk)[rix, pos_tok] - h.step(e, b, kk)[rix, pos_tok]) / (2 * ep)

            def plane_block(ep=eps, base=None, kk=K):
                JQ_ = torch.stack([jv(Q[:, i].expand(N, -1), ep, base, kk) for i in range(2)], 1)  # (N, 2, d): row i = J q_i
                Bm = (JQ_ @ Q).transpose(1, 2)  # Bm[n, a, b] = q_a^T J q_b  (column-vector convention: x' = Bm x)
                return JQ_, Bm, JQ_ @ JQ_.transpose(1, 2)  # Gram (J q_i).(J q_j): max plane gain = sqrt(max eig)

            # 1. in-plane block
            JQ, B, G2 = plane_block()
            _, Bs, _ = plane_block(eps_small)
            gain = JQ.norm(dim=-1)
            leak = (JQ - (JQ @ Q) @ Q.T).norm(dim=-1) / gain

            # 2. Arnoldi (fp64 basis)
            V = torch.zeros((N, m + 1, h.d), device=h.dev, dtype=torch.float64)
            Hm = torch.zeros((N, m + 1, m), device=h.dev, dtype=torch.float64)
            v0 = torch.randn((N, h.d), device=h.dev, dtype=torch.float64)
            V[:, 0] = v0 / v0.norm(dim=-1, keepdim=True)
            for jj in range(m):
                w = jv(V[:, jj].float()).double()
                for _ in range(2):  # classical Gram-Schmidt with reorthogonalisation
                    c = torch.einsum("nkd,nd->nk", V[:, : jj + 1], w)
                    Hm[:, : jj + 1, jj] += c
                    w = w - torch.einsum("nk,nkd->nd", c, V[:, : jj + 1])
                hn = w.norm(dim=-1)
                Hm[:, jj + 1, jj] = hn
                V[:, jj + 1] = w / hn.clamp_min(1e-10)[:, None]  # near-breakdown guard (rows with hn~0 are flagged by tiny residuals)
            Hc, Vc = Hm.cpu().numpy(), V[:, :m].cpu().numpy()
            Qn = Q.double().cpu().numpy()
            ritz, ov, rr, xtop = [], [], [], []
            for n_ in range(N):
                lam, Y = np.linalg.eig(Hc[n_, :m, :m])
                o = np.argsort(-np.abs(lam))
                lam, Y = lam[o], Y[:, o]
                ritz.append(lam)
                top = Y[:, :8]
                x = Vc[n_].T @ top  # (d, 8) complex Ritz vectors
                ov.append(np.linalg.norm(Qn.T @ x, axis=0) / np.linalg.norm(x, axis=0))
                rr.append(np.abs(Hc[n_, m, m - 1] * top[-1]) / np.linalg.norm(top, axis=0))
                xtop.append(x[:, :2] / np.linalg.norm(x[:, :2], axis=0))
            # fresh FD validation of the two leading Ritz pairs: ||J x - lambda x|| / ||x|| (complex x split into re/im)
            Xt = np.stack(xtop)  # (N, d, 2) complex, unit norm
            lam_t = np.stack([r_[:2] for r_ in ritz])  # (N, 2)
            fdr = np.zeros((N, 2, 2))
            for ei_, ep_ in enumerate((eps, eps_small)):
                for k_ in range(2):
                    xr = torch.tensor(Xt[:, :, k_].real, device=h.dev, dtype=torch.float32)
                    xi = torch.tensor(Xt[:, :, k_].imag, device=h.dev, dtype=torch.float32)
                    Jx = (jv(xr, ep_) + 1j * jv(xi, ep_)).cpu().numpy()
                    fdr[:, ei_, k_] = np.linalg.norm(Jx - lam_t[:, k_, None] * Xt[:, :, k_], axis=1)
            # 3. nonlinear propagation of pushes from step K
            th = 2 * np.pi * digits.double() / 10
            fcs = lambda a: torch.stack([torch.cos(a), torch.sin(a)], -1).float()
            chord = {k: fcs(th + 2 * np.pi * k / 10) - fcs(th) for k in (1, -1)}
            dirs = [chord[1] @ U, chord[-1] @ U]
            nref = dirs[0].norm(dim=-1, keepdim=True)
            dirs.append(rand0.expand(N, -1) * nref)
            ref = [s]
            for t in range(T):
                ref.append(h.step(e, ref[-1], K + t))
            ref_drift = torch.stack([(ref[t + 1][rix, pos_tok] - ref[t][rix, pos_tok]).norm(dim=-1) / ref[t][rix, pos_tok].norm(dim=-1)
                                     for t in range(T)], 1).cpu().numpy()
            Bck, Gck = [], []
            for c_ in ck:  # plane block along the reference trajectory (state after K + c_ steps)
                _, Bc, Gc = plane_block(eps, ref[c_], K + c_)
                Bck.append(Bc); Gck.append(Gc)
            pq = torch.zeros((N, len(dirs), len(alphas), T + 1, 2), device=h.dev)
            po = torch.zeros((N, len(dirs), len(alphas), T + 1), device=h.dev)
            for di, dv in enumerate(dirs):
                for ai, al in enumerate(alphas):
                    st = s.clone(); st[rix, pos_tok] += al * dv
                    for t in range(T + 1):
                        if t > 0:
                            st = h.step(e, st, K + t - 1)
                        dl = st[rix, pos_tok] - ref[t][rix, pos_tok]
                        pq[:, di, ai, t] = dl @ Q
                        po[:, di, ai, t] = (dl - (dl @ Q) @ Q.T).norm(dim=-1)
            out["uid"] += [p["uid"] for p in chunk]
            out["pid"] += [p["pid"] for p in chunk]
            out["ipos"] += ipos
            out["digit"] += digits.tolist()
            out["carry_in"] += [p["digit_info"][j]["carry_in"] for p, j in zip(chunk, ipos)]
            out["base_correct"].append(base_correct)
            out["drift"].append(drift)
            out["B"].append(B.cpu().numpy()); out["B_small"].append(Bs.cpu().numpy()); out["JQ_gram"].append(G2.cpu().numpy())
            out["prefix_drift"].append(prefix_drift); out["ref_drift"].append(ref_drift)
            out["B_ck"].append(torch.stack(Bck, 1).cpu().numpy()); out["JQ_gram_ck"].append(torch.stack(Gck, 1).cpu().numpy())
            out["ritz_fd_resid"].append(fdr)
            out["gainQ"].append(gain.cpu().numpy()); out["leakQ"].append(leak.cpu().numpy())
            out["ritz"].append(np.stack(ritz)); out["ritz_plane_overlap"].append(np.stack(ov)); out["ritz_resid"].append(np.stack(rr))
            out["prop_q"].append(pq.cpu().numpy()); out["prop_out"].append(po.cpu().numpy())
            out["push_q"].append(torch.stack([dv @ Q for dv in dirs], 1).cpu().numpy())
            out["push_norm"].append(torch.stack([dv.norm(dim=-1) for dv in dirs], 1).cpu().numpy())
            del V, Hm, ref
            torch.cuda.empty_cache()
            print(f"jacobian {c0 + N}/{len(problems)}: chunk {time.time() - tc:.0f}s, total {time.time() - t0:.0f}s, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    res = {k: (np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v)) for k, v in out.items()}
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    np.savez_compressed(f"{o}/jacobian.npz", alphas=np.array(alphas), block_checkpoints=np.array(ck), dir_names=np.array(["circle+1", "circle-1", "random_0"]), **res)
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.time() - t0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"{o}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def orbit(problems_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """Long unperturbed runs (fp32, K steps) for selected prompts: does a non-stationary trajectory converge slowly,
    drift, or cycle? Records per step the relative state change at every position; a fixed 64-d random projection of
    every position (every `stride_proj` steps, and every step over the last `tail` steps, for recurrence/period
    analysis); circle-plane coordinates (U s) at answer positions every step; digit readout every `stride_ro` steps."""
    import numpy as np
    import torch

    t0 = time.time()
    dt = cfg.get("dtype", "float32")
    h = H(dt, init_fp32=cfg.get("init_fp32", False))
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems, tol=cfg.get("val_tol", 1e-3))
    print("validation", val, flush=True)
    D = np.load(io.BytesIO(dirs_npz))
    tdt = getattr(torch, dt)
    U = torch.tensor(D["U"], device=h.dev).to(tdt)
    g = torch.Generator(device="cpu").manual_seed(11)
    Rp = (torch.randn((h.d, 64), generator=g) / np.sqrt(64)).to(h.dev).to(tdt)
    K, sp, tail, sro = cfg["K"], cfg["stride_proj"], cfg["tail"], cfg["stride_ro"]
    peps = cfg.get("perturb_eps", 0.0)
    res = {k: [] for k in ("uid", "pid", "L", "ids", "pred_pos", "drift", "proj_coarse", "proj_tail", "plane", "ro", "ro_full_is_digit",
                           "lag2_full", "pert_dist")}
    with torch.no_grad():
        for c0 in range(0, len(problems), cfg["chunk"]):
            tc = time.time()
            chunk = problems[c0 : c0 + cfg["chunk"]]
            N = len(chunk)
            ids, e, s = h.batch(chunk, [0] * N)
            Lmax = s.shape[1]
            P = max(len(p["pred_pos"]) for p in chunk)
            gidx = torch.tensor([p["pred_pos"] + [p["pred_pos"][-1]] * (P - len(p["pred_pos"])) for p in chunk], device=h.dev)
            gsel = gidx[..., None].expand(-1, -1, h.d)
            drift = torch.zeros((N, K, Lmax), device=h.dev)
            pc, pt, pl, ro, rof, l2, pdist = [], [], [], [], [], [], []
            s_m2, sp_ = None, None
            for k in range(K):
                s2 = h.step(e, s, k)
                drift[:, k] = (s2 - s).norm(dim=-1) / s2.norm(dim=-1)
                if k >= K - tail:
                    # full-state lag-2 residual ||s_k - s_{k-2}|| / ||s_k - s_{k-1}|| (period-2 test), per position
                    l2.append(((s2 - s_m2).norm(dim=-1) / (s2 - s).norm(dim=-1).clamp_min(1e-30)).float().cpu())
                    if peps > 0:
                        if sp_ is None:  # perturb every position by a random vector of relative size peps
                            nz = torch.randn(s.shape, device=h.dev, dtype=s.dtype, generator=torch.Generator(device=h.dev).manual_seed(5))
                            sp_ = s + peps * nz / nz.norm(dim=-1, keepdim=True) * s.norm(dim=-1, keepdim=True)
                        sp_ = h.step(e, sp_, k)
                        # distance to the reference trajectory allowing a phase swap of the 2-cycle
                        dd = torch.minimum((sp_ - s2).norm(dim=-1), (sp_ - s).norm(dim=-1)) / s2.norm(dim=-1)
                        pdist.append(dd.float().cpu())
                s_m2 = s
                s = s2
                pl.append((torch.gather(s, 1, gsel) @ U.T).float().cpu())
                if (k + 1) % sp == 0:
                    pc.append((s @ Rp).float().cpu())
                if k >= K - tail:
                    pt.append((s @ Rp).float().cpu())
                if (k + 1) % sro == 0 and (not cfg.get("ro_tail_only") or k >= K - tail):
                    lg = torch.log_softmax(h.coda_logits(s, gidx), -1)
                    ro.append(torch.stack([lg[:, 0][:, h.SPD]] + [lg[:, j][:, h.DIG] for j in range(1, P)], 1).cpu())
                    am = lg.argmax(-1)
                    isd = torch.stack([torch.isin(am[:, 0], torch.tensor(h.SPD, device=h.dev))] +
                                      [torch.isin(am[:, j], torch.tensor(h.DIG, device=h.dev)) for j in range(1, P)], 1)
                    rof.append(isd.cpu())
            for r, p in enumerate(chunk):
                n_, m_ = len(p["ids"]), len(p["pred_pos"])
                res["uid"].append(p["uid"]); res["pid"].append(p["pid"]); res["L"].append(n_)
                res["ids"].append(np.array(p["ids"])); res["pred_pos"].append(np.array(p["pred_pos"]))
                res["drift"].append(drift[r, :, :n_].float().cpu().numpy())
                res["proj_coarse"].append(torch.stack([x[r, :n_] for x in pc]).numpy())
                res["proj_tail"].append(torch.stack([x[r, :n_] for x in pt]).numpy())
                res["plane"].append(torch.stack([x[r, :m_] for x in pl]).numpy())
                res["ro"].append(torch.stack([x[r, :m_] for x in ro]).numpy())
                res["ro_full_is_digit"].append(torch.stack([x[r, :m_] for x in rof]).numpy())
                res["lag2_full"].append(torch.stack([x[r, :n_] for x in l2]).numpy())
                res["pert_dist"].append(torch.stack([x[r, :n_] for x in pdist]).numpy() if pdist else np.zeros((0, n_)))
            print(f"orbit {c0 + N}/{len(problems)}: chunk {time.time() - tc:.0f}s, total {time.time() - t0:.0f}s", flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    flat = {k: np.array(v) for k, v in res.items() if k in ("uid", "pid", "L")}
    # step index (1-based: state after k steps) of each saved sample
    flat["steps_coarse"] = np.arange(sp, K + 1, sp)
    flat["steps_tail"] = np.arange(K - tail + 1, K + 1)
    flat["steps_ro"] = np.arange(sro, K + 1, sro)
    if cfg.get("ro_tail_only"):
        flat["steps_ro"] = flat["steps_ro"][flat["steps_ro"] > K - tail]
    for k in ("ids", "pred_pos", "drift", "proj_coarse", "proj_tail", "plane", "ro", "ro_full_is_digit", "lag2_full", "pert_dist"):
        arr = np.empty(len(res[k]), dtype=object)
        arr[:] = res[k]
        flat[k] = arr
    np.savez_compressed(f"{o}/orbit.npz", allow_pickle=True, **flat)
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.time() - t0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"{o}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def carry_traj(problems_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """Early-recurrence trajectories at every answer position (fp32, steps 0..K; index t = state after t steps): circle-plane coordinates (U s), a fixed
    256-d random projection (for per-step decoders), relative state change, and the teacher-forced digit readout
    (10 digit log-probs + whether the full-vocab argmax is a digit) at every step. Used to test whether carry columns
    first represent the pre-carry digit (a+b mod 10) and later move one notch to the final digit."""
    import numpy as np
    import torch

    t0 = time.time()
    h = H("float32")
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    D = np.load(io.BytesIO(dirs_npz))
    U = torch.tensor(D["U"], device=h.dev, dtype=torch.float32)
    g = torch.Generator(device="cpu").manual_seed(12)
    Rp = (torch.randn((h.d, 256), generator=g) / np.sqrt(256)).to(h.dev)
    K = cfg["K"]
    rows, plane, proj, drift, lp, isd = [], [], [], [], [], []
    with torch.no_grad():
        for c0 in range(0, len(problems), cfg["chunk"]):
            tc = time.time()
            chunk = problems[c0 : c0 + cfg["chunk"]]
            N = len(chunk)
            ids, e, s = h.batch(chunk, [0] * N)
            P = max(len(p["pred_pos"]) for p in chunk)
            gidx = torch.tensor([p["pred_pos"] + [p["pred_pos"][-1]] * (P - len(p["pred_pos"])) for p in chunk], device=h.dev)
            gsel = gidx[..., None].expand(-1, -1, h.d)
            spd, dg = torch.tensor(h.SPD, device=h.dev), torch.tensor(h.DIG, device=h.dev)
            pl_, pr_, dr_, lp_, is_ = [], [], [], [], []
            prev = torch.gather(s, 1, gsel)
            for k in range(-1, K):  # k = -1: initial state (step 0 baseline); drift 0 there
                if k >= 0:
                    s = h.step(e, s, k)
                x = torch.gather(s, 1, gsel)
                pl_.append((x @ U.T).cpu()); pr_.append((x @ Rp).cpu())
                dr_.append(((x - prev).norm(dim=-1) / x.norm(dim=-1)).cpu())
                prev = x
                lg = torch.log_softmax(h.coda_logits(s, gidx), -1)
                lp_.append(torch.stack([lg[:, 0][:, spd]] + [lg[:, j][:, dg] for j in range(1, P)], 1).cpu())
                am = lg.argmax(-1)
                is_.append(torch.stack([torch.isin(am[:, 0], spd)] + [torch.isin(am[:, j], dg) for j in range(1, P)], 1).cpu())
            PL, PR, DR, LP, IS = (torch.stack(v, 2) for v in (pl_, pr_, dr_, lp_, is_))  # (N, P, K, ...)
            for r, p in enumerate(chunk):
                for j in range(len(p["pred_pos"])):
                    rows.append((p["uid"], j))
                    plane.append(PL[r, j].numpy()); proj.append(PR[r, j].numpy()); drift.append(DR[r, j].numpy())
                    lp.append(LP[r, j].numpy()); isd.append(IS[r, j].numpy())
            print(f"carry_traj {c0 + N}/{len(problems)}: chunk {time.time() - tc:.0f}s, total {time.time() - t0:.0f}s", flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    np.savez_compressed(f"{o}/carry_traj.npz", rows=np.array(rows), plane=np.stack(plane), proj=np.stack(proj),
                        drift=np.stack(drift), lp=np.stack(lp), full_is_digit=np.stack(isd))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.time() - t0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"{o}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def cycsrc(problems_json: str, cfg: dict, run_id: str):
    """Where does the period-2 cycle come from? (fp32; the loop map F is time-invariant: current_step only sets noise, off.)
    Per prompt: run K0 steps -> s_A, one more -> s_B (closure ||F(s_B) - s_A|| recorded). Moving positions = mean relative
    drift > thr over the last min(K0,128) warm-up steps; src = earliest moving position <= last answer prediction position.
    1. Hidden fixed point: damped iteration x <- x + beta (F(x) - x) from the cycle midpoint. For beta = 0.5 a real mode
       lambda is damped iff -3 < lambda < 1, so a flip mode (lambda < -1) can be stabilised, but convergence is NOT
       guaranteed (other modes) and the limit need not be "the" point between the phases: residuals, distance from the
       midpoint and readout are saved and must gate any interpretation. Undamped rollouts from x* and from x* + a small
       push along the cycle axis (signed axis projection per step: an alternating, growing sign = flip instability).
    2. Freeze test (after 1, so x* can be used): rows (q, mode, snap): mode 'prefix' pins positions <= q, 'single' pins only
       q; snap A/B = cycle phase snapshot (run starts from that phase), X = x* (run starts from s_A with pinned positions
       set to x*). Per step relative drift of every position, final lag-2 ratio, digit log-probs at the end.
    3. Spectra (Arnoldi, central FD, plus/minus evaluated in one batch): one-step J_qq at x* and two-step map at s_A, for
       selected (prompt, position) rows. Exact to first order for a single-position perturbation (causal attention; the
       prefix is fixed at x* / follows its reference cycle). Start vector = cycle axis (+ noise) where the position cycles.
       Saves the Hessenberg matrices (Ritz values at any Krylov size can be recomputed), the leading pair and the leading
       negative-real Ritz pair (validity mask), complex-aware FD residuals at two eps, FD derivative differences
       between eps, and span overlaps with the cycle axis. A missing negative mode is not evidence of absence."""
    import numpy as np
    import torch

    t0 = time.time()
    h = H("float32")
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    K0, Tf, thr, ndamp, beta, m = cfg["K0"], cfg["T_f"], cfg["thr"], cfg["n_damp"], cfg["beta"], cfg["m"]
    eps, eps_small, n_und, budget = cfg["eps"], cfg["eps_small"], cfg["n_undamped"], cfg["budget_s"]
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    R_ = {}  # everything saved
    skipped = []

    def obj(lst):
        a = np.empty(len(lst), dtype=object)
        a[:] = lst
        return a

    def save():
        np.savez_compressed(f"{o}/cycsrc.npz", allow_pickle=True, **R_)
        meta = {"validation": val, "cfg": cfg, "runtime_s": time.time() - t0, "run_id": run_id, "revision": REVISION,
                "skipped": skipped}
        with open(f"{o}/meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        with open(f"{o}/problems.json", "w") as f:
            f.write(problems_json)
        results_vol.commit()
        return meta

    nstep = [0]

    def F(e_, s_, k=K0):
        nstep[0] += s_.shape[0]
        return h.step(e_, s_, k)

    torch.manual_seed(0)
    with torch.no_grad():
        N = len(problems)
        ids, e, s = h.batch(problems, [0] * N)
        L = s.shape[1]
        P = max(len(p["pred_pos"]) for p in problems)
        gidx = torch.tensor([p["pred_pos"] + [p["pred_pos"][-1]] * (P - len(p["pred_pos"])) for p in problems], device=h.dev)
        spd, dg = torch.tensor(h.SPD, device=h.dev), torch.tensor(h.DIG, device=h.dev)

        def digit_lp(st, gi):  # (n, P, 10) digit log-probs (space-digit tokens for answer position 0)
            lg = torch.log_softmax(h.coda_logits(st, gi), -1)
            return torch.stack([lg[:, 0][:, spd]] + [lg[:, j][:, dg] for j in range(1, gi.shape[1])], 1).cpu().numpy()

        nh = min(K0, 128)
        drift_hist = torch.zeros((N, L), device=h.dev)
        for k in range(K0):
            s2 = F(e, s, k)
            if k >= K0 - nh:
                drift_hist += (s2 - s).norm(dim=-1) / s2.norm(dim=-1) / nh
            s = s2
        sA = s
        sB = F(e, sA)
        sA2 = F(e, sB)
        axis = sA - sB
        nrm = sA.norm(dim=-1)
        t_warm = time.time() - t0
        print(f"warm-up done {t_warm:.0f}s ({nstep[0]} row-steps)", flush=True)
        info = []
        for r, p in enumerate(problems):
            n_, pp = len(p["ids"]), p["pred_pos"]
            ipos_t, eq = pp[p["ipos"]], pp[0]
            op2 = eq - len(str(p["b"]))
            assert h.tok.decode([p["ids"][op2]]).strip() == str(p["b"])[0], (h.tok.decode([p["ids"][op2]]), p["b"])
            assert h.tok.decode([p["ids"][eq]]).strip() == "=", h.tok.decode([p["ids"][eq]])
            dh = drift_hist[r, :n_].cpu().numpy()
            mov = [int(i) for i in np.where(dh[: pp[-1] + 1] > thr)[0]]
            info.append(dict(uid=p["uid"], pid=p["pid"], demo=p["demo"], L=n_, ipos_t=ipos_t, eq=eq, op2=op2,
                             src=(mov[0] if mov else None), moving=mov))
        R_.update(info=obj([json.dumps(i) for i in info]), drift_hist=obj([drift_hist[r, :i["L"]].cpu().numpy() for r, i in enumerate(info)]),
                  axis_rel=obj([(axis[r, :i["L"]].norm(dim=-1) / nrm[r, :i["L"]]).cpu().numpy() for r, i in enumerate(info)]),
                  closure_rel=obj([((sA2 - sA)[r, :i["L"]].norm(dim=-1) / axis[r, :i["L"]].norm(dim=-1).clamp_min(1e-30)).cpu().numpy() for r, i in enumerate(info)]),
                  lp_A=digit_lp(sA, gidx), lp_B=digit_lp(sB, gidx), warm_s=t_warm)

        # 1. hidden fixed point
        x = 0.5 * (sA + sB)
        dcurve = []
        for t in range(ndamp):
            fx = F(e, x)
            if t % 10 == 0:
                dcurve.append(((fx - x).norm(dim=-1) / x.norm(dim=-1)).cpu().numpy())
            x = x + beta * (fx - x)
        fx = F(e, x)
        resid = ((fx - x).norm(dim=-1) / x.norm(dim=-1)).cpu().numpy()
        ax_unit = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        cyc_pos = (axis.norm(dim=-1) / nrm > 1e-4)  # positions that are actually cycling
        und = {}
        for nm_, push in (("plain", 0.0), ("axis_push", cfg["push_rel"])):
            y = x + push * ax_unit * nrm[..., None] * cyc_pos[..., None]
            dist, proj = [], []
            for t in range(n_und):
                y = F(e, y)
                dist.append(((y - x).norm(dim=-1) / x.norm(dim=-1)).cpu().numpy())
                proj.append((((y - x) * ax_unit).sum(-1) / nrm).cpu().numpy())
            und[nm_] = (np.stack(dist, 1), np.stack(proj, 1))
        R_.update(fp_resid=resid, fp_curve=np.stack(dcurve, 1),
                  fp_mid_off=((x - 0.5 * (sA + sB)).norm(dim=-1) / axis.norm(dim=-1).clamp_min(1e-30)).cpu().numpy(),
                  fp_dist_A=((x - sA).norm(dim=-1) / nrm).cpu().numpy(), lp_X=digit_lp(x, gidx),
                  und_plain_dist=und["plain"][0], und_plain_proj=und["plain"][1],
                  und_push_dist=und["axis_push"][0], und_push_proj=und["axis_push"][1])
        meta = save()
        print(f"fixed point done {time.time() - t0:.0f}s; resid max over real positions "
              f"{max(float(resid[r, :i['L']].max()) for r, i in enumerate(info)):.2e}", flush=True)

        # 2. freeze test
        frows = []
        for r, inf in enumerate(info):
            src, ip = inf["src"], inf["ipos_t"]
            if src is not None:
                mid = [q for q in inf["moving"] if src < q < ip]
                rows_ = [(-1, "prefix", "A"), (src - 1, "prefix", "A"), (src, "prefix", "A"), (src, "prefix", "B"), (src, "prefix", "X")]
                rows_ += [(q, "prefix", "A") for q in mid] + [(q, "single", "A") for q in mid]
                rows_ += [(ip - 1, "prefix", "A"), (ip - 1, "prefix", "B"), (ip - 1, "prefix", "X")]
            else:
                rows_ = [(-1, "prefix", "A"), (inf["op2"], "prefix", "A"), (inf["eq"], "prefix", "A"), (inf["op2"], "prefix", "X")]
            frows += [(r,) + q for q in dict.fromkeys(rows_)]
        print(f"freeze rows: {len(frows)}", flush=True)
        fr = {k: [] for k in ("row", "q", "mode", "snap", "drift", "lag2_end", "lp_end")}
        for c0 in range(0, len(frows), cfg["chunk"]):
            if time.time() - t0 > budget:
                skipped.append(f"freeze rows {c0}+"); break
            ch = frows[c0 : c0 + cfg["chunk"]]
            ri = torch.tensor([c[0] for c in ch], device=h.dev)
            mask = torch.zeros((len(ch), L, 1), device=h.dev, dtype=torch.bool)
            for j, (_, q, mode, _) in enumerate(ch):
                if q >= 0:
                    if mode == "prefix":
                        mask[j, : q + 1] = True
                    else:
                        mask[j, q] = True
            start = torch.stack([(sB if sn == "B" else sA)[r] for r, _, _, sn in ch])
            snap = torch.stack([(sB if sn == "B" else x if sn == "X" else sA)[r] for r, _, _, sn in ch])
            st = torch.where(mask, snap, start)
            ec = e[ri]
            dr = torch.zeros((len(ch), Tf, L), device=h.dev)
            prev2 = None
            for t in range(Tf):
                s2 = torch.where(mask, snap, F(ec, st))
                dr[:, t] = (s2 - st).norm(dim=-1) / s2.norm(dim=-1)
                if t == Tf - 1:
                    lag2 = (s2 - prev2).norm(dim=-1) / (s2 - st).norm(dim=-1).clamp_min(1e-30)
                prev2, st = st, s2
            lpe = digit_lp(st, gidx[ri])
            for j, (r, q, mode, sn) in enumerate(ch):
                n_ = info[r]["L"]
                fr["row"].append(r); fr["q"].append(q); fr["mode"].append(mode); fr["snap"].append(sn)
                fr["drift"].append(dr[j, :, :n_].cpu().numpy()); fr["lag2_end"].append(lag2[j, :n_].cpu().numpy())
                fr["lp_end"].append(lpe[j, : len(problems[r]["pred_pos"])])
            print(f"freeze {c0 + len(ch)}/{len(frows)} {time.time() - t0:.0f}s", flush=True)
        R_.update(f_row=np.array(fr["row"]), f_q=np.array(fr["q"]), f_mode=np.array(fr["mode"]), f_snap=np.array(fr["snap"]),
                  f_drift=obj(fr["drift"]), f_lag2_end=obj(fr["lag2_end"]), f_lp_end=obj(fr["lp_end"]))
        meta = save()

        # 3. spectra
        ctrl = set(cfg["spec_control_uids"])
        jrows = []
        for r, inf in enumerate(info):
            if inf["src"] is not None:
                jrows += [(r, inf["src"]), (r, inf["ipos_t"])]
            elif inf["uid"] in ctrl:
                jrows += [(r, inf["op2"]), (r, inf["eq"]), (r, inf["ipos_t"])]
        Rn = len(jrows)
        rr = torch.tensor([r for r, _ in jrows], device=h.dev)
        pq = torch.tensor([q for _, q in jrows], device=h.dev)
        ar = torch.arange(Rn, device=h.dev)
        e2 = torch.cat([e[rr], e[rr]])
        axq = axis[rr, pq].double()
        axq_unit = axq / axq.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        cyc_q = (axis[rr, pq].norm(dim=-1) / nrm[rr, pq] > 1e-4)
        R_["j_rows"] = np.array(jrows)
        R_["j_cyc"] = cyc_q.cpu().numpy()

        def make_jv(base, nsteps):
            b2 = torch.cat([base, base])
            pq2, ar2 = torch.cat([pq, pq]), torch.arange(2 * Rn, device=h.dev)

            def jv(v, ep=eps):  # v (Rn, d) -> derivative of F^nsteps at base, position-q block, applied to v
                st = b2.clone()
                st[ar2, pq2] += torch.cat([ep * v, -ep * v]).to(st.dtype)
                for t in range(nsteps):
                    st = F(e2, st, K0 + t)
                d_ = st[ar2, pq2]
                return (d_[:Rn] - d_[Rn:]) / (2 * ep)
            return jv

        def span_overlap(a, b, u):  # ||proj of unit u onto span(a, b)||
            Qm, Rm = torch.linalg.qr(torch.stack([a, b], -1))
            keep = (Rm.diagonal(dim1=-2, dim2=-1).abs() > 1e-8 * Rm[..., 0, 0].abs()[..., None]).double()
            c = torch.einsum("ndk,nd->nk", Qm, u) * keep
            return c.norm(dim=-1)

        for nm, base, nsteps, m in (("J1_xstar", x[rr], 1, cfg["m"]), ("J2_two_step_sA", sA[rr], 2, cfg["m_two_step"])):
            if time.time() - t0 > budget:
                skipped.append(nm); continue
            jv = make_jv(base, nsteps)
            V = torch.zeros((Rn, m + 1, h.d), device=h.dev, dtype=torch.float64)
            Hm = torch.zeros((Rn, m + 1, m), device=h.dev, dtype=torch.float64)
            v0 = torch.randn((Rn, h.d), device=h.dev, dtype=torch.float64)
            v0 = v0 / v0.norm(dim=-1, keepdim=True)
            v0 = torch.where(cyc_q[:, None], axq_unit + 0.3 * v0, v0)
            V[:, 0] = v0 / v0.norm(dim=-1, keepdim=True)
            for jj in range(m):
                w = jv(V[:, jj].float()).double()
                for _ in range(2):
                    c = torch.einsum("nkd,nd->nk", V[:, : jj + 1], w)
                    Hm[:, : jj + 1, jj] += c
                    w = w - torch.einsum("nk,nkd->nd", c, V[:, : jj + 1])
                hn = w.norm(dim=-1)
                Hm[:, jj + 1, jj] = hn
                V[:, jj + 1] = w / hn.clamp_min(1e-10)[:, None]
            orth_err = (torch.einsum("nkd,njd->nkj", V[:, :m], V[:, :m]) - torch.eye(m, device=h.dev, dtype=torch.float64)).abs().amax((1, 2))
            Hc = Hm.cpu().numpy()
            sel_lam = np.full((Rn, 2), np.nan + 0j)
            sel_valid = np.zeros((Rn, 2), bool)
            A_, B_ = torch.zeros((Rn, 2, h.d), device=h.dev, dtype=torch.float64), torch.zeros((Rn, 2, h.d), device=h.dev, dtype=torch.float64)
            ritz_all = []
            for n_ in range(Rn):
                lam, Y = np.linalg.eig(Hc[n_, :m, :m])
                o_ = np.argsort(-np.abs(lam))
                lam, Y = lam[o_], Y[:, o_]
                ritz_all.append(lam)
                negs = np.where((lam.real < 0) & (np.abs(lam.imag) < 1e-6 * np.abs(lam)))[0]
                for k_, idx in enumerate([0, int(negs[0]) if len(negs) else None]):
                    if idx is None:
                        continue
                    y = Y[:, idx]
                    if abs(lam[idx].imag) < 1e-6 * abs(lam[idx]):
                        y = y.real if np.linalg.norm(y.real) >= np.linalg.norm(y.imag) else y.imag
                        y = y.astype(complex)
                    a = V[n_, :m].T @ torch.tensor(y.real.copy(), device=h.dev)
                    b = V[n_, :m].T @ torch.tensor(y.imag.copy(), device=h.dev)
                    z_ = torch.sqrt(a.norm() ** 2 + b.norm() ** 2)
                    A_[n_, k_], B_[n_, k_] = a / z_, b / z_
                    sel_lam[n_, k_], sel_valid[n_, k_] = lam[idx], True
            # complex-aware FD residuals: J a = alpha a - beta b, J b = beta a + alpha b
            al = torch.tensor(sel_lam.real, device=h.dev).nan_to_num(0.0)
            be = torch.tensor(sel_lam.imag, device=h.dev).nan_to_num(0.0)
            fd_res = np.zeros((Rn, 2, 2))
            fd_diff = np.zeros((Rn, 2))
            Jstore = {}
            for ei, ep in enumerate((eps, eps_small)):
                for k_ in range(2):
                    Ja = jv(A_[:, k_].float(), ep).double()
                    Jb = jv(B_[:, k_].float(), ep).double()
                    Jstore[(ei, k_)] = (Ja, Jb)
                    ra = Ja - al[:, k_, None] * A_[:, k_] + be[:, k_, None] * B_[:, k_]
                    rb = Jb - be[:, k_, None] * A_[:, k_] - al[:, k_, None] * B_[:, k_]
                    fd_res[:, ei, k_] = torch.sqrt(ra.norm(dim=-1) ** 2 + rb.norm(dim=-1) ** 2).cpu().numpy()
            for k_ in range(2):
                (Ja0, Jb0), (Ja1, Jb1) = Jstore[(0, k_)], Jstore[(1, k_)]
                num = torch.sqrt((Ja0 - Ja1).norm(dim=-1) ** 2 + (Jb0 - Jb1).norm(dim=-1) ** 2)
                den = torch.sqrt(Ja0.norm(dim=-1) ** 2 + Jb0.norm(dim=-1) ** 2).clamp_min(1e-30)
                fd_diff[:, k_] = (num / den).cpu().numpy()
            ov = np.stack([span_overlap(A_[:, k_], B_[:, k_], axq_unit).cpu().numpy() for k_ in range(2)], 1)
            R_.update({f"{nm}__H": Hc, f"{nm}__ritz": obj(ritz_all), f"{nm}__sel_lam": sel_lam, f"{nm}__sel_valid": sel_valid,
                       f"{nm}__fd_resid": fd_res, f"{nm}__fd_eps_rel_diff": fd_diff, f"{nm}__axis_span_overlap": ov,
                       f"{nm}__orth_err": orth_err.cpu().numpy(), f"{nm}__sel_vec_re": A_.float().cpu().numpy(),
                       f"{nm}__sel_vec_im": B_.float().cpu().numpy()})
            del V, Hm
            torch.cuda.empty_cache()
            meta = save()
            print(f"{nm} done {time.time() - t0:.0f}s ({nstep[0]} row-steps total)", flush=True)
        R_["x_sel"] = torch.stack([x[r, q] for r, q in jrows]).cpu().numpy() if jrows else np.zeros(0)
        R_["sA_sel"] = torch.stack([sA[r, q] for r, q in jrows]).cpu().numpy() if jrows else np.zeros(0)
        R_["sB_sel"] = torch.stack([sB[r, q] for r, q in jrows]).cpu().numpy() if jrows else np.zeros(0)
        meta = save()
    meta["row_steps"] = nstep[0]
    return meta



@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def bifur(problems_json: str, cfg: dict, run_id: str):
    """Bifurcation along a continuous family of loop maps (fp32; loop map time-invariant, noise off).
    Two prompts of the same problem with different few-shot demos, token-aligned: uid0 stationary, uid1 cycling.
    E = prelude output (re-injected every loop; contextual). Family F_a(s) = step(E_a, s), E_a = (1-a) E_0 + a E_1 for the
    whole aligned tensor; a = 0 / 1 are exactly the original maps; intermediate a are not textual prompts.
    Priority order under a wall-clock deadline (partial results saved after every sub-stage):
      endpoints (guarded cycle axis) -> coarse grid [ordinary dynamics from ONE common init tensor (+ extension of
      unresolved rows); damped fixed points from the row's cycle midpoint, from interpolated endpoint fixed points
      (every other row), and by sequential continuation; source-block ('=') Arnoldi spectrum at the continuation fixed
      point; mode tracking by adjacent-alpha vector overlap anchored at the cycling endpoint; gated eigenpair validation]
      -> spectral crossing (eligible adjacent pairs only) -> refined grid (same) -> signed pushes (+ unpushed control,
      two magnitudes) on validated rows on both sides -> two-step multiplier on accepted refined cycles -> endpoint-start
      (basin-dependence) runs. Source labels refer to the '=' position only."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    byuid = {p["uid"]: p for p in problems}
    p0, p1 = byuid[cfg["uid0"]], byuid[cfg["uid1"]]
    assert (p0["a"], p0["b"], p0["pid"]) == (p1["a"], p1["b"], p1["pid"]), "different problems"
    assert len(p0["ids"]) == len(p1["ids"]) and p0["pred_pos"] == p1["pred_pos"], "prompts not aligned"
    pp = p0["pred_pos"]
    assert p0["ids"][pp[0]:] == p1["ids"][pp[0]:], "answer/eq tokens differ"
    q = pp[0]
    assert h.tok.decode([p0["ids"][q]]).strip() == "=", h.tok.decode([p0["ids"][q]])
    ipos = pp[p0["ipos"]]
    K, beta, m = cfg["K"], cfg["beta"], cfg["m"]
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    R_ = {"q": q, "ipos": ipos, "pred_pos": np.array(pp)}
    skipped, nstep = [], [0]
    elapsed = lambda: time.monotonic() - T0
    T_c = time.monotonic()  # compute clock starts after model load + validation
    late = lambda reserve=0.0: time.monotonic() - T_c > cfg["budget_s"] - reserve

    def save():
        np.savez_compressed(f"{o}/bifur.npz", allow_pickle=True, **R_)
        meta = {"validation": val, "cfg": cfg, "runtime_s": elapsed(), "run_id": run_id, "revision": REVISION,
                "skipped": skipped, "row_steps": nstep[0]}
        with open(f"{o}/meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        with open(f"{o}/problems.json", "w") as f:
            f.write(problems_json)
        results_vol.commit()
        return meta

    def put(tag, d):
        for k, v in d.items():
            R_[f"{tag}__{k}"] = v

    with torch.no_grad():
        ids, e01, s01 = h.batch([p0, p1], [0, 0])
        L = e01.shape[1]
        E0, E1 = e01[0].clone(), e01[1].clone()
        init = s01[0].clone()  # common initial state, reused for every alpha
        R_["E_diff_rel_per_pos"] = ((E1 - E0).norm(dim=-1) / E0.norm(dim=-1)).cpu().numpy()
        gidx1 = torch.tensor(pp, device=h.dev)[None]
        spd, dg = torch.tensor(h.SPD, device=h.dev), torch.tensor(h.DIG, device=h.dev)
        g = torch.Generator(device="cpu").manual_seed(21)
        Rdir = torch.randn((h.d, 2), generator=g).to(h.dev)
        Rdir = Rdir / Rdir.norm(dim=0, keepdim=True)
        garn = torch.Generator(device=h.dev).manual_seed(5)

        def Ea(al):
            al = torch.as_tensor(np.asarray(al, float), device=h.dev, dtype=E0.dtype).reshape(-1, 1, 1)
            return (1 - al) * E0[None] + al * E1[None]

        assert torch.equal(Ea([0.0])[0], E0) and torch.allclose(Ea([1.0])[0], E1, atol=0, rtol=1e-6)

        def F(E, s):
            nstep[0] += s.shape[0]
            return h.step(E, s, 0)

        def readout(s):
            lg = torch.log_softmax(h.coda_logits(s, gidx1.expand(s.shape[0], -1)), -1)
            dl = torch.stack([lg[:, 0][:, spd]] + [lg[:, j][:, dg] for j in range(1, len(pp))], 1)
            return dl.cpu().numpy(), lg.argmax(-1).cpu().numpy()

        def run(E, s, nsteps, axis_u):
            """Ordinary dynamics. drift (n, steps, L); lag-2 closure at q (n, steps); rolling 64-step tail at q
            (projections on axis_u and 2 random directions)."""
            n = s.shape[0]
            dr = torch.full((n, nsteps, L), float("nan"), device=h.dev)
            l2 = torch.full((n, nsteps), float("nan"), device=h.dev)
            tail, prev = [], None
            k_done = 0
            for k in range(nsteps):
                if late(cfg["reserve_s"]):
                    skipped.append(f"run truncated at {k}/{nsteps}")
                    break
                s2 = F(E, s)
                dr[:, k] = (s2 - s).norm(dim=-1) / s2.norm(dim=-1)
                if prev is not None:
                    l2[:, k] = (s2[:, q] - prev).norm(dim=-1) / (s2[:, q] - s[:, q]).norm(dim=-1).clamp_min(1e-30)
                prev = s[:, q]
                sq = s2[:, q]
                tail.append(torch.stack([(sq * axis_u).sum(-1), sq @ Rdir[:, 0], sq @ Rdir[:, 1]], -1))
                tail = tail[-64:]
                s = s2
                k_done = k + 1
            tp = torch.stack(tail, 1).cpu().numpy() if tail else np.zeros((n, 0, 3))
            return s, dr[:, :k_done], l2[:, :k_done], tp

        def classify(drq, l2q):
            """Source ('=') label from the last 64 steps: 0 stationary, 1 period-2, 2 unresolved (np arrays)."""
            n, T = drq.shape
            lab = np.full(n, 2)
            if T < 64:
                return lab, np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
            d1, d0 = drq[:, -32:].mean(1), drq[:, -64:-32].mean(1)
            cl = np.nanmax(l2q[:, -32:], 1)
            stat = (d1 < cfg["stat_thr"]) & ((d1 <= 1.05 * d0) | (d1 < cfg["floor_thr"]))
            p2 = (d1 >= cfg["stat_thr"]) & (np.abs(d1 / d0 - 1) < cfg["amp_tol"]) & (cl < cfg["closure_tol"])
            lab[stat] = 0
            lab[p2] = 1
            return lab, d1, d0, cl

        def damped(E, x, nmax, tol):
            """Damped iteration with per-row early stop on the PREFIX (positions <= q) residual.
            Returns x, residual per position (n, L), iterations used, residual curve every 10 iterations (max over prefix)."""
            n = x.shape[0]
            done = torch.zeros(n, dtype=torch.bool, device=h.dev)
            iters = torch.zeros(n, dtype=torch.long, device=h.dev)
            curve = []
            res = None
            for it in range(nmax):
                if late(cfg["reserve_s"]):
                    skipped.append(f"damped stopped at {it}")
                    break
                act = (~done).nonzero().flatten()
                if len(act) == 0:
                    break
                fx = F(E[act], x[act])
                r = (fx - x[act]).norm(dim=-1) / x[act].norm(dim=-1)
                if res is None:
                    res = torch.full((n, L), float("nan"), device=h.dev)
                res[act] = r
                x = x.clone()
                x[act] = x[act] + beta * (fx - x[act])
                iters[act] += 1
                done[act] = r[:, : q + 1].max(-1).values < tol
                if it % 10 == 0:
                    curve.append(res[:, : q + 1].max(-1).values.cpu().numpy())
            # final residual of the returned point
            fx = F(E, x)
            res = ((fx - x).norm(dim=-1) / x.norm(dim=-1))
            return x, res.cpu().numpy(), iters.cpu().numpy(), (np.stack(curve, 1) if curve else np.zeros((n, 0)))

        def arnoldi(jv, n, v0, mm):
            V = torch.zeros((n, mm + 1, h.d), device=h.dev, dtype=torch.float64)
            Hm = torch.zeros((n, mm + 1, mm), device=h.dev, dtype=torch.float64)
            V[:, 0] = v0 / v0.norm(dim=-1, keepdim=True)
            done = 0
            for jj in range(mm):
                if late(cfg["reserve_s"]):
                    skipped.append(f"arnoldi stopped at {jj}/{mm}")
                    break
                w = jv(V[:, jj].float()).double()
                for _ in range(2):
                    c = torch.einsum("nkd,nd->nk", V[:, : jj + 1], w)
                    Hm[:, : jj + 1, jj] += c
                    w = w - torch.einsum("nk,nkd->nd", c, V[:, : jj + 1])
                hn = w.norm(dim=-1)
                Hm[:, jj + 1, jj] = hn
                V[:, jj + 1] = w / hn.clamp_min(1e-10)[:, None]
                done = jj + 1
            orth = (torch.einsum("nkd,njd->nkj", V[:, :done], V[:, :done]) - torch.eye(done, device=h.dev, dtype=torch.float64)).abs().amax((1, 2))
            return V[:, :done], Hm[:, : done + 1, :done].cpu().numpy(), done, orth.cpu().numpy()

        def make_jv(E, base, nsteps=1):
            n = base.shape[0]
            ar2, q2 = torch.arange(2 * n, device=h.dev), torch.full((2 * n,), q, device=h.dev)
            E2, b2 = torch.cat([E, E]), torch.cat([base, base])

            def jv(v, ep=cfg["eps"]):
                st = b2.clone()
                st[ar2, q2] += torch.cat([ep * v, -ep * v]).to(st.dtype)
                for _ in range(nsteps):
                    st = F(E2, st)
                d_ = st[ar2, q2]
                return (d_[:n] - d_[n:]) / (2 * ep)
            return jv

        def real_candidates(V, Hc, i, done, sign):
            """Real Ritz pairs of row i with sign(lambda) == sign: list of (lambda, unit vector (d,) float64)."""
            lam, Y = np.linalg.eig(Hc[i, :done, :done])
            out = []
            for k in range(done):
                if np.sign(lam[k].real) == sign and abs(lam[k].imag) < 1e-6 * abs(lam[k]):
                    y = Y[:, k].real if np.linalg.norm(Y[:, k].real) >= np.linalg.norm(Y[:, k].imag) else Y[:, k].imag
                    xv = V[i].T @ torch.tensor(y.copy(), device=h.dev)
                    out.append((float(lam[k].real), xv / xv.norm()))
            return out, lam

        def validate_pair(jv, vecs, lams):
            """FD eigen-residual ||J v - lam v|| (unit v), Rayleigh v^T J v, at each eps; adjacent-eps derivative diffs."""
            n = vecs.shape[0]
            lt = torch.tensor(lams, device=h.dev, dtype=torch.float64)
            res, ray, Js = [], [], []
            for ep in cfg["eps_check"]:
                Jv = jv(vecs.float(), ep).double()
                Js.append(Jv)
                res.append((Jv - lt[:, None] * vecs).norm(dim=-1).cpu().numpy())
                ray.append((Jv * vecs).sum(-1).cpu().numpy())
            dd = [((Js[i] - Js[i + 1]).norm(dim=-1) / Js[i].norm(dim=-1).clamp_min(1e-30)).cpu().numpy() for i in range(len(Js) - 1)]
            return np.stack(res, 1), np.stack(ray, 1), np.stack(dd, 1)

        def spectrum_tracked(E, x, anchor_vec, order):
            """Arnoldi at fixed points x (rows); track the real-negative mode along `order` (row indices, starting next to
            the anchor) by max |overlap| with the previous tracked vector (anchor first). Gated validation."""
            n = x.shape[0]
            jv = make_jv(E, x)
            v0 = torch.randn((n, h.d), device=h.dev, dtype=torch.float64, generator=garn)
            v0 = anchor_vec.double()[None] + 0.3 * v0 / v0.norm(dim=-1, keepdim=True)
            V, Hc, done, orth = arnoldi(jv, n, v0, m)
            out = {"H": Hc, "krylov_dim": done, "orth_err": orth}
            if done < 4:
                return out, None
            lam_tr = np.full(n, np.nan); ov_prev = np.full(n, np.nan); ov_2nd = np.full(n, np.nan); lam_half = np.full(n, np.nan)
            vt = torch.zeros((n, h.d), device=h.dev, dtype=torch.float64)
            ritz = []
            prev = anchor_vec.double()
            for i in order:
                cands, lam = real_candidates(V, Hc, i, done, -1)
                ritz.append((i, lam))
                if not cands:
                    continue
                ovs = np.array([float((v @ prev).abs()) for _, v in cands])
                j = int(np.argmax(ovs))
                lam_tr[i], ov_prev[i] = cands[j][0], ovs[j]
                ov_2nd[i] = np.sort(ovs)[-2] if len(ovs) > 1 else 0.0
                v = cands[j][1]
                v = v if float(v @ prev) >= 0 else -v  # consistent orientation
                vt[i] = v
                prev = v
                half, _ = real_candidates(V[:, : done // 2], Hc[:, : done // 2 + 1, : done // 2], i, done // 2, -1)
                if half:
                    lam_half[i] = half[int(np.argmax([float((hv @ v).abs()) for _, hv in half]))][0]
            ritz = sorted(ritz)
            out.update(ritz=np.array([r[1][np.argsort(-np.abs(r[1]))][:16] for r in ritz], dtype=object), lam_track=lam_tr,
                       track_overlap_prev=ov_prev, track_overlap_2nd=ov_2nd, lam_half_track=lam_half,
                       lam_lead=np.array([r[1][np.argmax(np.abs(r[1]))] for r in ritz]))
            found = np.isfinite(lam_tr)
            res, ray, dd = validate_pair(jv, vt, np.nan_to_num(lam_tr))
            res[~found], ray[~found], dd[~found] = np.nan, np.nan, np.nan
            out.update(fd_resid=res, rayleigh=ray, fd_eps_rel_diff=dd, vec_track=vt.float().cpu().numpy())
            return out, vt

        def fixed_points(tag, E, al, xA, xB, interp_rows=None, cont_start=None):
            """Independent damped solves from the cycle midpoint (+ from interpolated endpoint fixed points on some rows),
            then sequential continuation along increasing alpha from cont_start. Returns continuation fixed points."""
            n = len(al)
            tol = cfg["fp_tol"]
            res = {}
            x_mid, r_mid, it_mid, cv_mid = damped(E, 0.5 * (xA + xB), cfg["n_damp"], tol)
            res.update(fp_mid_resid=r_mid, fp_mid_iters=it_mid, fp_mid_curve=cv_mid, fp_mid_q=x_mid[:, q].cpu().numpy())
            if interp_rows is not None and len(interp_rows) and not late(cfg["reserve_s"]):
                ii = torch.tensor(interp_rows, device=h.dev)
                a_ = torch.tensor(al[interp_rows], device=h.dev, dtype=E0.dtype)[:, None, None]
                x_i, r_i, it_i, _ = damped(E[ii], (1 - a_) * xstar_end[0][None] + a_ * xstar_end[1][None], cfg["n_damp"], tol)
                res.update(fp_interp_rows=np.array(interp_rows), fp_interp_resid=r_i, fp_interp_iters=it_i,
                           fp_interp_vs_mid_rel=((x_i - x_mid[ii]).norm(dim=-1) / x_mid[ii].norm(dim=-1)).cpu().numpy())
            # sequential continuation (increasing alpha), seeded from the previous accepted solution
            x_c = torch.zeros_like(x_mid)
            r_c = np.full((n, L), np.nan); it_c = np.zeros(n, int)
            prev = cont_start if cont_start is not None else x_mid[0:1]
            for i in range(n):
                if late(cfg["reserve_s"]):
                    skipped.append(f"{tag} continuation stopped at {i}")
                    break
                x1, r1, it1, _ = damped(E[i : i + 1], prev.clone(), cfg["n_damp_cont"], tol)
                x_c[i], r_c[i], it_c[i] = x1[0], r1[0], it1[0]
                if np.nanmax(r1[0, : q + 1]) < tol:
                    prev = x1
            res.update(fp_cont_resid=r_c, fp_cont_iters=it_c, fp_cont_q=x_c[:, q].cpu().numpy(),
                       fp_cont_vs_mid_rel=((x_c - x_mid).norm(dim=-1) / x_mid.norm(dim=-1)).cpu().numpy(),
                       fp_cont_step_rel=np.r_[np.nan, ((x_c[1:] - x_c[:-1]).norm(dim=-1) / x_c[1:].norm(dim=-1)).max(-1).values.cpu().numpy()],
                       fp_ok_prefix=np.nan_to_num(np.nanmax(r_c[:, : q + 1], 1), nan=1.0) < tol,
                       fp_ok_full=np.nan_to_num(np.nanmax(r_c, 1), nan=1.0) < tol)
            res["fp_cont_lp"], res["fp_cont_argmax"] = readout(x_c)
            return res, x_c

        def ordinary(tag, al, start, max_ext):
            E = Ea(al)
            n = len(al)
            s, dr, l2, tp = run(E, start.clone(), K, axis_ref)
            drq, l2q = dr[:, :, q].cpu().numpy(), l2.cpu().numpy()
            lab, d1, d0, cl = classify(drq, l2q)
            out = {"alpha": np.asarray(al, float), "label_initial": lab.copy(), "steps_done": np.full(n, dr.shape[1]),
                   "drift_q": drq, "drift_ans": dr[:, :, ipos].cpu().numpy(), "lag2_q": l2q, "tail": tp,
                   "drift_last32_all": dr[:, -32:].mean(1).cpu().numpy()}
            ext_idx = list(np.where(lab == 2)[0][:max_ext])
            lab_f, d1_f, d0_f, cl_f, tail_f = lab.copy(), d1.copy(), d0.copy(), cl.copy(), tp.copy()
            if ext_idx and not late(cfg["reserve_s"]):
                ii = torch.tensor(ext_idx, device=h.dev)
                s_e, dr_e, l2_e, tp_e = run(E[ii], s[ii].clone(), cfg["K_extend"], axis_ref)
                le, d1e, d0e, cle = classify(dr_e[:, :, q].cpu().numpy(), l2_e.cpu().numpy())
                for j, i in enumerate(ext_idx):
                    if dr_e.shape[1] >= 64:
                        lab_f[i], d1_f[i], d0_f[i], cl_f[i], tail_f[i] = le[j], d1e[j], d0e[j], cle[j], tp_e[j]
                out.update(ext_idx=np.array(ext_idx), ext_drift_q=dr_e[:, :, q].cpu().numpy(), ext_lag2_q=l2_e.cpu().numpy(),
                           ext_drift_last32_all=dr_e[:, -32:].mean(1).cpu().numpy())
                out["steps_done"][ext_idx] += dr_e.shape[1]
                s = s.clone(); s[ii] = s_e
            xA = s
            xB = F(E, xA)
            xA2 = F(E, xB)
            out.update(label_final=lab_f, d1_final=d1_f, d0_final=d0_f, closure_max_final=cl_f, tail_final=tail_f,
                       amp_rel=((xA - xB).norm(dim=-1) / (2 * xA.norm(dim=-1))).cpu().numpy(),
                       amp_abs_q=((xA - xB)[:, q].norm(dim=-1) / 2).cpu().numpy(),
                       closure_rel=((xA2 - xA).norm(dim=-1) / (xA - xB).norm(dim=-1).clamp_min(1e-30)).cpu().numpy(),
                       xA_q=xA[:, q].cpu().numpy(), xB_q=xB[:, q].cpu().numpy())
            out["lp_A"], out["argmax_A"] = readout(xA)
            out["lp_B"], out["argmax_B"] = readout(xB)
            put(tag, out)
            save()
            print(f"{tag} ordinary done {elapsed():.0f}s ({nstep[0]} row-steps); labels {lab_f.tolist()}", flush=True)
            return out, E, xA, xB

        # ---------------- endpoints (guarded reference axis)
        E_end = Ea([0.0, 1.0])
        axis_ref = torch.zeros(h.d, device=h.dev)
        s_end, dr_end, l2_end, _ = run(E_end, init[None].expand(2, -1, -1).clone(), K, axis_ref)
        xB_end = F(E_end, s_end)
        xA2_end = F(E_end, xB_end)
        amp_end = ((s_end - xB_end)[:, q].norm(dim=-1) / (2 * s_end[:, q].norm(dim=-1))).cpu().numpy()
        clo_end = ((xA2_end - s_end)[:, q].norm(dim=-1) / (s_end - xB_end)[:, q].norm(dim=-1).clamp_min(1e-30)).cpu().numpy()
        lab_end, *_ = classify(dr_end[:, :, q].cpu().numpy(), l2_end.cpu().numpy())
        axis_raw = s_end[1, q] - xB_end[1, q]
        R_.update(end_amp_rel_q=amp_end, end_closure_q=clo_end, end_label=lab_end, axis_ref=axis_raw.cpu().numpy())
        ok_axis = amp_end[1] > cfg["axis_min_amp"] and clo_end[1] < cfg["closure_tol"]
        R_["axis_ok"] = ok_axis
        print(f"endpoints {elapsed():.0f}s: amp_rel_q {amp_end}, closure {clo_end}, labels {lab_end}", flush=True)
        if not ok_axis and not cfg.get("smoke_skip_axis_guard", False):
            skipped.append("ABORT: cycling endpoint has no settled cycle (axis guard)")
            return save()
        axis_ref = axis_raw / axis_raw.norm()
        xstar_end, r_end, it_end, _ = damped(E_end, 0.5 * (s_end + xB_end), cfg["n_damp"], cfg["fp_tol"])
        R_.update(end_fp_resid=r_end, end_fp_iters=it_end)
        # anchor mode at the cycling endpoint
        sp_end, vt_end = spectrum_tracked(E_end[1:], xstar_end[1:], axis_ref, [0])
        put("end_spec", sp_end)
        anchor = vt_end[0] if vt_end is not None and np.isfinite(sp_end.get("lam_track", [np.nan])[0]) else axis_ref.double()
        save()
        print(f"endpoint spectrum {elapsed():.0f}s: lam {sp_end.get('lam_track')}", flush=True)

        def eligible(sp, fp):
            found = np.isfinite(sp.get("lam_track", np.array([np.nan])))
            if cfg.get("smoke_skip_axis_guard", False):  # smoke only: exercise downstream code paths
                return found
            if "fd_resid" not in sp:
                return np.zeros_like(found)
            r_mid = sp["fd_resid"][:, 1]  # eps = 0.1
            return (found & fp["fp_ok_prefix"] & (r_mid < cfg["eig_res_tol"]) & (sp["fd_eps_rel_diff"].max(1) < cfg["eig_res_tol"])
                    & (sp["track_overlap_prev"] > cfg["track_ov_min"]))

        if cfg.get("fixed_alphas"):  # follow-up mode: explicit refined grid, no coarse stage
            cross, reason = float(cfg["alpha_cross_prior"]), "prior_run"
            R_.update(alpha_cross_est=cross, refinement_reason=reason)
            ac = None
        # ---------------- coarse grid
        ac = np.linspace(0, 1, cfg["n_coarse"]) if not cfg.get("fixed_alphas") else None
        if ac is not None:
            co, Ec, xAc, xBc = ordinary("coarse", ac, init[None].expand(len(ac), -1, -1), cfg["max_extend_coarse"])
            fpc, xc = fixed_points("coarse", Ec, ac, xAc, xBc, interp_rows=list(range(0, len(ac), 2)), cont_start=xstar_end[0:1])
            put("coarse", fpc); save()
            spc, vtc = spectrum_tracked(Ec, xc, anchor, list(range(len(ac) - 1, -1, -1)))  # track from the cycling end
            put("coarse_spec", spc)
            elc = eligible(spc, fpc)
            R_["coarse_spec__eligible"] = elc
            save()
            print(f"coarse spectrum {elapsed():.0f}s: lam {np.round(spc.get('lam_track', []), 4).tolist()} eligible {elc.tolist()}", flush=True)

            # ---------------- crossing (spectral, eligible adjacent pairs only) and amplitude onset candidate
            lam = np.asarray(spc.get("lam_track", np.full(len(ac), np.nan)), float)
            brackets = [(i, ac[i] + (lam[i] + 1) / (lam[i] - lam[i + 1]) * (ac[i + 1] - ac[i])) for i in range(len(ac) - 1)
                        if elc[i] and elc[i + 1] and lam[i] > -1 >= lam[i + 1]]
            acc_cyc = np.where(co["label_final"] == 1)[0]
            onset = float(ac[acc_cyc[0]]) if len(acc_cyc) else np.nan
            R_.update(spectral_brackets=np.array(brackets) if brackets else np.zeros((0, 2)), amplitude_onset_candidate=onset)
            if brackets:
                cross, reason = brackets[0][1], "spectral"
            elif np.isfinite(onset):
                cross, reason = onset, "amplitude_onset_exploratory"
            else:
                cross, reason = None, "none"
            R_.update(alpha_cross_est=np.nan if cross is None else cross, refinement_reason=reason)
            print(f"crossing {cross} ({reason}); brackets {brackets}; onset {onset}", flush=True)

        if cross is not None and not late(cfg["reserve_s"]):
            if cfg.get("fixed_alphas"):
                af = np.array(cfg["fixed_alphas"], float)
            else:
                dlt = cfg["refine_halfwidth"] * (ac[1] - ac[0])
                af = np.unique(np.r_[np.linspace(max(0.0, cross - dlt), min(1.0, cross + dlt), cfg["n_refine"] - 1), cross])
            fo, Ef, xAf, xBf = ordinary("fine", af, init[None].expand(len(af), -1, -1), cfg["max_extend_fine"])
            if ac is not None:  # continuation start: accepted coarse fixed point at the largest coarse alpha below af[0]
                jlow = max([i for i in range(len(ac)) if ac[i] <= af[0] and fpc["fp_ok_prefix"][i]] or [0])
                cs = xc[jlow : jlow + 1]
            else:
                cs = None  # first fine row's own midpoint solution
            fpf, xf = fixed_points("fine", Ef, af, xAf, xBf, cont_start=cs)
            put("fine", fpf); save()
            # anchor for fine tracking: coarse tracked vector at the coarse alpha nearest the top of the fine grid
            if ac is not None:
                jhi = int(np.argmin(np.abs(ac - af[-1])))
                anc = vtc[jhi] if vtc is not None and np.isfinite(lam[jhi]) else anchor
            else:
                anc = anchor
            spf, vtf = spectrum_tracked(Ef, xf, anc, list(range(len(af) - 1, -1, -1)))
            put("fine_spec", spf)
            elf = eligible(spf, fpf)
            R_["fine_spec__eligible"] = elf
            save()
            lamf = np.asarray(spf.get("lam_track", np.full(len(af), np.nan)), float)
            print(f"fine spectrum {elapsed():.0f}s: alpha {np.round(af, 4).tolist()} lam {np.round(lamf, 4).tolist()} eligible {elf.tolist()}", flush=True)

            # signed pushes (+ unpushed control, 2 magnitudes) at validated rows on both sides of -1
            below = [i for i in np.argsort(lamf) if elf[i] and lamf[i] > -1][:2]  # closest to -1 from inside
            above = [i for i in np.argsort(-lamf) if elf[i] and lamf[i] <= -1][:2]  # closest to -1 from outside
            sel = sorted(below + above)
            if sel and vtf is not None and not late(cfg["reserve_s"] + 20):
                rows, sg, mag = [], [], []
                for i in sel:
                    for s_, m_ in ((0.0, 0.0), (1.0, cfg["push_rel"][0]), (-1.0, cfg["push_rel"][0]), (1.0, cfg["push_rel"][1]), (-1.0, cfg["push_rel"][1])):
                        rows.append(i); sg.append(s_); mag.append(m_)
                ii = torch.tensor(rows, device=h.dev)
                vv = vtf[ii].float()
                st = xf[ii].clone()
                nq = st[:, q].norm(dim=-1, keepdim=True)
                st[:, q] += torch.tensor(np.array(sg) * np.array(mag), device=h.dev, dtype=st.dtype)[:, None] * vv * nq
                proj, dist = [((st[:, q] - xf[ii][:, q]) * vv).sum(-1).cpu().numpy() / nq[:, 0].cpu().numpy()], []
                for k in range(cfg["K_push"]):
                    if late(cfg["reserve_s"]):
                        skipped.append("push truncated"); break
                    st = F(Ef[ii], st)
                    dl_ = st[:, q] - xf[ii][:, q]
                    proj.append(((dl_ * vv).sum(-1) / nq[:, 0]).cpu().numpy())
                    dist.append((dl_.norm(dim=-1) / nq[:, 0]).cpu().numpy())
                R_.update(push__row=np.array(rows), push__alpha=af[rows], push__lam=lamf[rows], push__sign=np.array(sg),
                          push__mag=np.array(mag), push__proj=np.stack(proj, 1), push__dist=np.stack(dist, 1) if dist else np.zeros((len(rows), 0)))
                save()
                print(f"push done {elapsed():.0f}s", flush=True)

            # two-step multiplier on accepted refined cycles (phase-A state from the ordinary run)
            cyc_rows = [i for i in np.where(fo["label_final"] == 1)[0]] if not cfg.get("smoke_skip_axis_guard", False) else list(range(len(af)))
            cyc_rows = sorted(cyc_rows, key=lambda i: af[i])[: cfg["n_twostep"]]
            if cyc_rows and not late(cfg["reserve_s"] + 20):
                ii = torch.tensor(cyc_rows, device=h.dev)
                jv2 = make_jv(Ef[ii], xAf[ii], nsteps=2)
                axq = (xAf[ii][:, q] - xBf[ii][:, q]).double()
                axq = axq / axq.norm(dim=-1, keepdim=True)
                v0 = axq + 0.3 * torch.randn(axq.shape, device=h.dev, dtype=torch.float64, generator=garn) / np.sqrt(h.d)
                V2, H2, d2, o2 = arnoldi(jv2, len(cyc_rows), v0, cfg["m_two_step"])
                mult, mov = np.full(len(cyc_rows), np.nan), np.full(len(cyc_rows), np.nan)
                vt2 = torch.zeros((len(cyc_rows), h.d), device=h.dev, dtype=torch.float64)
                for j in range(len(cyc_rows)):
                    cands, _ = real_candidates(V2, H2, j, d2, 1)
                    if cands:
                        ovs = [float((v @ axq[j]).abs()) for _, v in cands]
                        k = int(np.argmax(ovs))
                        mult[j], mov[j], vt2[j] = cands[k][0], ovs[k], cands[k][1]
                r2, ray2, dd2 = validate_pair(jv2, vt2, np.nan_to_num(mult))
                R_.update(twostep__rows=np.array(cyc_rows), twostep__alpha=af[cyc_rows], twostep__mult=mult, twostep__axis_ov=mov,
                          twostep__fd_resid=r2, twostep__fd_eps_rel_diff=dd2, twostep__H=H2, twostep__krylov_dim=d2,
                          twostep__closure=fo["closure_rel"][cyc_rows, q])
                save()
                print(f"two-step done {elapsed():.0f}s: alpha {af[cyc_rows]} mult {mult}", flush=True)

            # endpoint-start runs (basin dependence; exploratory): refined alphas started from the cycling endpoint state
            if not late(cfg["reserve_s"] + 30):
                s_h, dr_h, l2_h, tp_h = run(Ef, s_end[1][None].expand(len(af), -1, -1).clone(), cfg["K_hyst"], axis_ref)
                lh, d1h, d0h, clh = classify(dr_h[:, :, q].cpu().numpy(), l2_h.cpu().numpy())
                xBh = F(Ef, s_h)
                R_.update(hyst__label=lh, hyst__d1=d1h, hyst__closure_max=clh, hyst__steps=dr_h.shape[1],
                          hyst__amp_rel_q=((s_h - xBh)[:, q].norm(dim=-1) / (2 * s_h[:, q].norm(dim=-1))).cpu().numpy(),
                          hyst__xA_q=s_h[:, q].cpu().numpy(), hyst__tail=tp_h)
                save()
                print(f"endpoint-start runs done {elapsed():.0f}s: labels {lh.tolist()}", flush=True)
        meta = save()
    return meta



@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def doubt(problems_json: str, cfg: dict, run_id: str):
    """Is uncertainty a weak pull? (fp32; loop map time-invariant, noise off.) One primary answer-prediction position p per
    prompt (teacher-forced). Prompts carry `cell` (unsure/confident x correct/wrong, selected on init seed 0) and `multi`.
    Stage order (deadline-aware, partial saves): 1 dynamics -> 2 fixed points -> 3 pushes (primary) -> 4 spectrum.
    1. Fresh seeds (multi prompts: 1..S; others: 1), K loops; rows still moving at p are extended alone. Per step at p:
       drift, lag-2; digit log-probs + full-argmax-is-digit every step up to `dense_until`, then at checkpoints and the
       terminal step; p-state at checkpoints + terminal; distance to the prompt's seed-1 row (p and max over <= p) at
       checkpoints <= K. Chunks interleave cells and keep each prompt's seeds together.
    2. Fixed point from the seed-1 state: damped iteration; fp_valid = post-update prefix residual < fp_tol (rows that fail
       are still pushed but labelled; a damped limit may be unstable under the plain map).
    3. Pushes at x* (rel push_rel): control, duplicate control (numerical floor), 2 random at p, gap-shrinking direction
       at p (-grad of fixed top1-top2 logit gap through the coda), random at every position <= p (each position rel
       push_rel; initial prefix displacement recorded). Distance to the control at p and max over <= p every loop; digit
       log-probs at p every `gap_every` loops (gap differences = logit differences).
    4. Answer-block spectrum at x*: Arnoldi on J_pp (leading Ritz estimate; half/full comparison; complex-aware FD residual)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()  # deadline clock: after model load (includes validation)
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    late = lambda reserve=0.0: time.monotonic() - Tc > cfg["budget_s"] - reserve
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    R_, skipped, nstep = {}, [], [0]
    spd, dg = torch.tensor(h.SPD, device=h.dev), torch.tensor(h.DIG, device=h.dev)

    def save():
        np.savez_compressed(f"{o}/doubt.npz", allow_pickle=True, **R_)
        meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION,
                "skipped": skipped, "row_steps": nstep[0]}
        with open(f"{o}/meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        with open(f"{o}/problems.json", "w") as f:
            f.write(problems_json)
        results_vol.commit()
        return meta

    def obj(lst):
        a_ = np.empty(len(lst), dtype=object); a_[:] = lst
        return a_

    def F(e, s):
        nstep[0] += s.shape[0]
        return h.step(e, s, 0)

    def lp_at(s, pos, digit_first):
        lg = h.coda_logits(s, pos[:, None])[:, 0]
        ls = torch.log_softmax(lg, -1)
        dl = torch.where(digit_first[:, None], ls[:, spd], ls[:, dg])
        am = lg.argmax(-1)
        isd = torch.where(digit_first, torch.isin(am, spd), torch.isin(am, dg))
        return dl, isd

    pp = [p["pred_pos"][p["ipos"]] for p in problems]
    first = [p["ipos"] == 0 for p in problems]
    R_.update(uid=np.array([p["uid"] for p in problems]), pid=np.array([p["pid"] for p in problems]),
              cell=np.array([p["cell"] for p in problems]), multi=np.array([p["multi"] for p in problems]),
              ppos=np.array(pp), tok_len=np.array([len(p["ids"]) for p in problems]),
              truth=np.array([p["digit_info"][p["ipos"]]["digit"] for p in problems]))
    S, K, ck = cfg["n_seeds"], cfg["K"], cfg["checkpoints"]
    # interleave cells (round robin), keep each prompt's seeds together
    bycell = {}
    for i, p in enumerate(problems):
        bycell.setdefault(p["cell"], []).append(i)
    order = []
    while any(bycell.values()):
        for c in sorted(bycell):
            if bycell[c]:
                order.append(bycell[c].pop(0))
    rows = [(i, sd) for i in order for sd in (range(1, S + 1) if problems[i]["multi"] else [1])]
    chunks, cur = [], []
    for r in rows:
        if cur and r[0] != cur[-1][0] and len(cur) + (S if problems[r[0]]["multi"] else 1) > cfg["chunk"]:
            chunks.append(cur); cur = []
        cur.append(r)
    if cur:
        chunks.append(cur)
    with torch.no_grad():
        # ---------------- 1. dynamics
        D = {k: [] for k in ("prompt", "seed", "steps_done", "reason", "drift_p", "lag2_p", "ro_steps", "ro_lp", "ro_isd",
                             "ck_steps", "ck_state_p", "dist_p_ck", "dist_pre_ck")}
        final = {}
        for ci, ch in enumerate(chunks):
            if late(cfg["reserve_s"] + 120):
                skipped.append(f"dynamics chunk {ci}"); continue
            plist = [problems[i] for i, _ in ch]
            ids, e, s = h.batch(plist, [sd for _, sd in ch])
            n, L = s.shape[0], s.shape[1]
            pos = torch.tensor([pp[i] for i, _ in ch], device=h.dev)
            df = torch.tensor([first[i] for i, _ in ch], device=h.dev)
            ar = torch.arange(n, device=h.dev)
            ref = torch.tensor([next(j for j, (ii, _) in enumerate(ch) if ii == i) for i, _ in ch], device=h.dev)
            pre_mask = (torch.arange(L, device=h.dev)[None] <= pos[:, None]).float()
            drift, lag2, ro_t, ro_lp, ro_isd, ck_t, ck_s, dpk, dprek = [], [], [], [], [], [], [], [], []
            prev, k, reason = None, 0, "done"
            while k < K:
                if late(cfg["reserve_s"]):
                    reason = "deadline"; skipped.append(f"dynamics chunk {ci} truncated at {k}"); break
                s2 = F(e, s)
                drift.append(((s2[ar, pos] - s[ar, pos]).norm(dim=-1) / s2[ar, pos].norm(dim=-1)).cpu())
                lag2.append(((s2[ar, pos] - prev).norm(dim=-1) / (s2[ar, pos] - s[ar, pos]).norm(dim=-1).clamp_min(1e-30)).cpu()
                            if prev is not None else torch.full((n,), float("nan")))
                prev = s[ar, pos]
                s = s2
                k += 1
                if k <= cfg["dense_until"] or k in ck or k == K:
                    dl, isd = lp_at(s, pos, df)
                    ro_t.append(k); ro_lp.append(dl.cpu()); ro_isd.append(isd.cpu())
                if k in ck:
                    ck_t.append(k); ck_s.append(s[ar, pos].cpu())
                    dpk.append(((s[ar, pos] - s[ref, pos]).norm(dim=-1) / s[ref, pos].norm(dim=-1)).cpu())
                    dd = (s - s[ref]).norm(dim=-1) / s[ref].norm(dim=-1)
                    dprek.append((dd * pre_mask).max(-1).values.cpu())
            dr_ = torch.stack(drift, 1) if drift else torch.zeros((n, 0))
            steps = np.full(n, k)
            ext = {}
            mov = (dr_[:, -16:].mean(1) > cfg["stat_thr"]).nonzero().flatten().tolist() if dr_.shape[1] >= 16 else []
            if mov and reason == "done" and not late(cfg["reserve_s"] + 90):  # extend only rows still moving at p
                ii = torch.tensor(mov, device=h.dev)
                se, ee, pe, dfe = s[ii].clone(), e[ii], pos[ii], df[ii]
                are = torch.arange(len(mov), device=h.dev)
                dre, ext_ro_t, ext_ro, ext_isd = [], [], [], []
                kk = 0
                while kk < cfg["K_extend"]:
                    if late(cfg["reserve_s"] + 60):
                        skipped.append(f"extension chunk {ci} truncated at {kk}"); break
                    s2 = F(ee, se)
                    dre.append(((s2[are, pe] - se[are, pe]).norm(dim=-1) / s2[are, pe].norm(dim=-1)).cpu())
                    se = s2
                    kk += 1
                    if kk % 32 == 0 or kk == cfg["K_extend"]:
                        dl, isd = lp_at(se, pe, dfe)
                        ext_ro_t.append(K + kk); ext_ro.append(dl.cpu()); ext_isd.append(isd.cpu())
                s = s.clone(); s[ii] = se
                for jx, j in enumerate(mov):
                    steps[j] = K + kk
                    ext[j] = (torch.stack(dre, 1)[jx].numpy() if dre else np.zeros(0), ext_ro_t,
                              torch.stack(ext_ro, 1)[jx].numpy() if ext_ro else np.zeros((0, 10)),
                              torch.stack(ext_isd, 1)[jx].numpy() if ext_isd else np.zeros(0, bool))
            # terminal readout and state for every row
            dlT, isdT = lp_at(s, pos, df)
            for j, (i, sd) in enumerate(ch):
                D["prompt"].append(i); D["seed"].append(sd); D["steps_done"].append(int(steps[j])); D["reason"].append(reason)
                d_ = dr_[j].numpy(); rt = list(ro_t); rl = [x[j].numpy() for x in ro_lp]; ri = [bool(x[j]) for x in ro_isd]
                if j in ext:
                    d_ = np.r_[d_, ext[j][0]]; rt += ext[j][1]; rl += list(ext[j][2]); ri += list(ext[j][3])
                if not rt or rt[-1] != steps[j]:
                    rt.append(int(steps[j])); rl.append(dlT[j].cpu().numpy()); ri.append(bool(isdT[j]))
                D["drift_p"].append(d_); D["lag2_p"].append(torch.stack(lag2, 1)[j].numpy() if lag2 else np.zeros(0))
                D["ro_steps"].append(np.array(rt)); D["ro_lp"].append(np.stack(rl)); D["ro_isd"].append(np.array(ri))
                D["ck_steps"].append(np.array(ck_t + [int(steps[j])]))
                D["ck_state_p"].append(np.stack([x[j].numpy() for x in ck_s] + [s[j, pos[j]].cpu().numpy()]))
                D["dist_p_ck"].append(np.array([x[j].item() for x in dpk])); D["dist_pre_ck"].append(np.array([x[j].item() for x in dprek]))
                if sd == 1:
                    final[i] = (s[j].clone(), e[j].clone())
            print(f"dynamics chunk {ci} ({n} rows, extended {len(mov)}) done {time.monotonic() - Tc:.0f}s ({nstep[0]} row-steps)", flush=True)
        for k_, v in D.items():
            R_[f"dyn__{k_}"] = np.array(v) if k_ in ("prompt", "seed", "steps_done", "reason") else obj(v)
        R_["checkpoints"] = np.array(ck)
        save()
        if not final:
            skipped.append("no seed-1 states: stages 2-4 skipped")
            return save()

        # ---------------- 2. fixed points
        idx = sorted(final)
        Lm = max(final[i][0].shape[0] for i in idx)
        n = len(idx)
        X = torch.zeros((n, Lm, h.d), device=h.dev); E = torch.zeros((n, Lm, h.d), device=h.dev)
        for j, i in enumerate(idx):
            Li = final[i][0].shape[0]
            X[j, :Li], E[j, :Li] = final[i]
            if Li < Lm:  # filler rows after the prompt; causally never read by positions <= p
                X[j, Li:], E[j, Li:] = final[i][0][-1], final[i][1][-1]
        pos = torch.tensor([pp[i] for i in idx], device=h.dev)
        df = torch.tensor([first[i] for i in idx], device=h.dev)
        ar = torch.arange(n, device=h.dev)
        pre_mask = (torch.arange(Lm, device=h.dev)[None] <= pos[:, None])
        done = torch.zeros(n, dtype=torch.bool, device=h.dev)
        iters = torch.zeros(n, dtype=torch.long, device=h.dev)
        x = X.clone()
        for it in range(cfg["n_damp"]):
            if late(cfg["reserve_s"] + 90):
                skipped.append(f"damped stopped at {it}"); break
            act = (~done).nonzero().flatten()
            if len(act) == 0:
                break
            fx = F(E[act], x[act])
            r = (fx - x[act]).norm(dim=-1) / x[act].norm(dim=-1)
            x[act] = x[act] + cfg["beta"] * (fx - x[act])
            iters[act] += 1
            done[act] = (r * pre_mask[act]).max(-1).values < cfg["fp_tol"]
        fx = F(E, x)
        res = ((fx - x).norm(dim=-1) / x.norm(dim=-1))
        res_pre = (res * pre_mask).max(-1).values
        fp_valid = torch.isfinite(res_pre) & (res_pre < cfg["fp_tol"])
        lpx, isdx = lp_at(x, pos, df)
        R_.update(fp_prompt=np.array(idx), fp_resid_prefix=res_pre.cpu().numpy(), fp_valid=fp_valid.cpu().numpy(),
                  fp_iters=iters.cpu().numpy(), fp_lp=lpx.cpu().numpy(), fp_isd=isdx.cpu().numpy(),
                  fp_dist_from_seed1_p=((x[ar, pos] - X[ar, pos]).norm(dim=-1) / X[ar, pos].norm(dim=-1)).cpu().numpy())
        save()
        print(f"fixed points done {time.monotonic() - Tc:.0f}s; valid {int(fp_valid.sum())}/{n}", flush=True)

        # ---------------- 3. pushes (primary)
        if late(cfg["reserve_s"] + 45):
            skipped.append("pushes")
        else:
            with torch.enable_grad():
                xg = x.clone().requires_grad_(True)
                lg = h.coda_logits(xg, pos[:, None])[:, 0]
                tok = torch.where(df[:, None], spd[None].expand(n, -1), dg[None].expand(n, -1))
                zl = torch.gather(lg, 1, tok)
                t2 = zl.topk(2, -1).indices
                gap = zl.gather(1, t2[:, :1])[:, 0] - zl.gather(1, t2[:, 1:2])[:, 0]
                gr = torch.autograd.grad(gap.sum(), xg)[0][ar, pos]
            gn = gr.norm(dim=-1, keepdim=True)
            grad_ok = torch.isfinite(gn[:, 0]) & (gn[:, 0] > 0)
            gdir = torch.where(grad_ok[:, None], -gr / gn.clamp_min(1e-30), torch.zeros_like(gr))
            g = torch.Generator(device=h.dev).manual_seed(9)
            kinds = ["control", "control_dup", "rand1", "rand2", "gap", "prefix_rand"]
            nk = len(kinds)
            st = x.repeat_interleave(nk, 0).clone()
            Er = E.repeat_interleave(nk, 0)
            posr = pos.repeat_interleave(nk)
            dfr = df.repeat_interleave(nk)
            arr = torch.arange(n * nk, device=h.dev)
            nrm_p = x[ar, pos].norm(dim=-1).repeat_interleave(nk)
            pre_disp = np.zeros(n)
            for j in range(n):
                p_ = int(pos[j])
                for kk, kind in enumerate(kinds):
                    r_ = j * nk + kk
                    if kind in ("rand1", "rand2"):
                        v = torch.randn(h.d, device=h.dev, generator=g); st[r_, p_] += cfg["push_rel"] * nrm_p[r_] * v / v.norm()
                    elif kind == "gap":
                        st[r_, p_] += cfg["push_rel"] * nrm_p[r_] * gdir[j]
                    elif kind == "prefix_rand":
                        v = torch.randn((p_ + 1, h.d), device=h.dev, generator=g)
                        dv = cfg["push_rel"] * x[j, : p_ + 1].norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True)
                        st[r_, : p_ + 1] += dv
                        pre_disp[j] = float(dv.norm() / x[j, : p_ + 1].norm())
            pmask_r = pre_mask.repeat_interleave(nk, 0)
            ctrl_idx = torch.arange(n, device=h.dev).repeat_interleave(nk) * nk
            dist_p, dist_pre, lps = [], [], []

            def record(s_, t):
                c = s_[ctrl_idx]
                dist_p.append(((s_[arr, posr] - c[arr, posr]).norm(dim=-1) / nrm_p).cpu().numpy())
                dd = (s_ - c).norm(dim=-1) / c.norm(dim=-1)
                dist_pre.append((dd * pmask_r).max(-1).values.cpu().numpy())
                if t % cfg["gap_every"] == 0:
                    lps.append(lp_at(s_, posr, dfr)[0].cpu().numpy())

            record(st, 0)
            t_done = 0
            for t in range(1, cfg["K_push"] + 1):
                if late(cfg["reserve_s"] + 20):
                    skipped.append(f"push stopped at {t}"); break
                st = F(Er, st)
                record(st, t)
                t_done = t
            R_.update(push_kinds=np.array(kinds), push_dist_p=np.stack(dist_p, 1), push_dist_pre=np.stack(dist_pre, 1),
                      push_lp=np.stack(lps, 1), push_gap0=gap.detach().cpu().numpy(), push_gap_every=cfg["gap_every"],
                      push_top2=t2.cpu().numpy(), push_grad_ok=grad_ok.cpu().numpy(), push_prefix_rel_disp=pre_disp,
                      push_steps_done=t_done)
            save()
            print(f"pushes done {time.monotonic() - Tc:.0f}s", flush=True)

        # ---------------- 4. answer-block spectrum (optional)
        if late(cfg["reserve_s"] + 20):
            skipped.append("spectrum")
        else:
            ar2, p2, E2, x2 = torch.arange(2 * n, device=h.dev), torch.cat([pos, pos]), torch.cat([E, E]), torch.cat([x, x])

            def jv(v, ep=cfg["eps"]):
                st_ = x2.clone()
                st_[ar2, p2] += torch.cat([ep * v, -ep * v]).to(st_.dtype)
                d_ = F(E2, st_)[ar2, p2]
                return (d_[:n] - d_[n:]) / (2 * ep)

            m = cfg["m"]
            V = torch.zeros((n, m + 1, h.d), device=h.dev, dtype=torch.float64)
            Hm = torch.zeros((n, m + 1, m), device=h.dev, dtype=torch.float64)
            v0 = torch.randn((n, h.d), device=h.dev, dtype=torch.float64, generator=torch.Generator(device=h.dev).manual_seed(3))
            V[:, 0] = v0 / v0.norm(dim=-1, keepdim=True)
            dn = 0
            for jj in range(m):
                if late(cfg["reserve_s"] + 5):
                    skipped.append(f"arnoldi stopped at {jj}"); break
                w = jv(V[:, jj].float()).double()
                for _ in range(2):
                    c = torch.einsum("nkd,nd->nk", V[:, : jj + 1], w)
                    Hm[:, : jj + 1, jj] += c
                    w = w - torch.einsum("nk,nkd->nd", c, V[:, : jj + 1])
                hn = w.norm(dim=-1)
                Hm[:, jj + 1, jj] = hn
                V[:, jj + 1] = w / hn.clamp_min(1e-10)[:, None]
                dn = jj + 1
            if dn >= 4:
                Hc = Hm[:, : dn + 1, :dn].cpu().numpy()
                lead, top6, half = [], [], []
                A_ = torch.zeros((n, h.d), device=h.dev, dtype=torch.float64); B_ = torch.zeros_like(A_)
                for j in range(n):
                    lam, Y = np.linalg.eig(Hc[j, :dn, :dn])
                    o_ = np.argsort(-np.abs(lam)); lam, Y = lam[o_], Y[:, o_]
                    lead.append(lam[0]); top6.append(lam[:6])
                    half.append(np.abs(np.linalg.eigvals(Hc[j, : dn // 2, : dn // 2])).max())
                    a_ = V[j, :dn].T @ torch.tensor(Y[:, 0].real.copy(), device=h.dev)
                    b_ = V[j, :dn].T @ torch.tensor(Y[:, 0].imag.copy(), device=h.dev)
                    z_ = torch.sqrt(a_.norm() ** 2 + b_.norm() ** 2)
                    A_[j], B_[j] = a_ / z_, b_ / z_
                lead = np.array(lead)
                al = torch.tensor(lead.real, device=h.dev); be = torch.tensor(lead.imag, device=h.dev)
                fdr = np.full((n, 2), np.nan)
                for ei, ep in enumerate((cfg["eps"], cfg["eps_small"])):
                    if late(cfg["reserve_s"]):
                        skipped.append("spectrum FD check"); break
                    Ja, Jb = jv(A_.float(), ep).double(), jv(B_.float(), ep).double()
                    ra = Ja - al[:, None] * A_ + be[:, None] * B_
                    rb = Jb - be[:, None] * A_ - al[:, None] * B_
                    fdr[:, ei] = torch.sqrt(ra.norm(dim=-1) ** 2 + rb.norm(dim=-1) ** 2).cpu().numpy()
                R_.update(spec_lead=lead, spec_top6=np.array(top6), spec_half_lead_abs=np.array(half), spec_fd_resid=fdr,
                          spec_krylov=dn, spec_H=Hc)
            else:
                skipped.append(f"spectrum: krylov {dn} < 4")
            save()
            print(f"spectrum done {time.monotonic() - Tc:.0f}s", flush=True)
        meta = save()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def census(problems_json: str, cfg: dict, run_id: str):
    """Prevalence + convergence-mode census on fresh prompts (fp32; loop map time-invariant, noise off).
    Per prompt: K loops from seed-0 init. Per position: relative drift every loop; lag-2 / lag-1 ratio over the tail.
    Convergence-mode statistics: with d_k = s_{k+1} - s_k at a position, accumulate (float64) over loop windows W the
    Gram sums S00 = sum <d_k,d_k>, S11, S22, S01, S02, S12 over triples (d_k, d_{k+1}, d_{k+2}) inside W, plus the count
    and mean |d_k| / |s|. Offline: real fit lambda = S01/S00 (residual S11 - S01^2/S00); two-step fit
    d_{k+2} = a d_{k+1} + b d_k (complex pair = roots of z^2 - a z - b), residual from the same sums.
    NOTE these are trajectory (excited, observable) modes at each position, including upstream forcing; not own-block
    eigenvalues. Readout every loop over the last `tail` loops at all answer-prediction positions: digit log-probs and the
    full-vocabulary argmax id (teacher-forced)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    K, tail, W = cfg["K"], cfg["tail"], [tuple(w) for w in cfg["windows"]]
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    out = {k: [] for k in ("uid", "pid", "L", "ids", "pred_pos", "drift", "lag2_tail", "gram", "dnorm", "ro_lp", "ro_argmax", "steps")}
    skipped = []
    spd, dg = torch.tensor(h.SPD, device=h.dev), torch.tensor(h.DIG, device=h.dev)
    with torch.no_grad():
        for c0 in range(0, len(problems), cfg["chunk"]):
            if time.monotonic() - Tc > cfg["budget_s"] - 30:
                skipped.append(f"chunk {c0}"); continue
            tc = time.monotonic()
            chunk = problems[c0 : c0 + cfg["chunk"]]
            N = len(chunk)
            ids, e, s = h.batch(chunk, [0] * N)
            L = s.shape[1]
            P = max(len(p["pred_pos"]) for p in chunk)
            gidx = torch.tensor([p["pred_pos"] + [p["pred_pos"][-1]] * (P - len(p["pred_pos"])) for p in chunk], device=h.dev)
            drift = torch.zeros((N, K, L), device=h.dev)
            l2 = []
            G = torch.zeros((N, len(W), L, 7), device=h.dev, dtype=torch.float64)  # S00 S11 S22 S01 S02 S12 count
            DN = torch.zeros((N, len(W), L), device=h.dev, dtype=torch.float64)
            dm2 = dm1 = None
            ro_lp, ro_am = [], []
            k_done = 0
            for k in range(K):
                if time.monotonic() - Tc > cfg["budget_s"] - 20:
                    skipped.append(f"chunk {c0} truncated at {k}"); break
                s2 = h.step(e, s, k)
                d = s2 - s
                nrm = s2.norm(dim=-1)
                drift[:, k] = d.norm(dim=-1) / nrm
                if k >= K - tail and dm1 is not None:
                    l2.append(((d + dm1).norm(dim=-1) / d.norm(dim=-1).clamp_min(1e-30)).cpu())  # ||s_{k+1}-s_{k-1}|| / ||s_{k+1}-s_k||
                if dm2 is not None:
                    kk = k - 2  # triple (d_kk, d_kk+1, d_kk+2) = (dm2, dm1, d)
                    for wi, (a_, b_) in enumerate(W):
                        if a_ <= kk and kk + 2 < b_:
                            x0, x1, x2 = dm2.double(), dm1.double(), d.double()  # float64 products and reductions
                            dot = lambda x, y: (x * y).sum(-1)
                            G[:, wi, :, 0] += dot(x0, x0); G[:, wi, :, 1] += dot(x1, x1); G[:, wi, :, 2] += dot(x2, x2)
                            G[:, wi, :, 3] += dot(x0, x1); G[:, wi, :, 4] += dot(x0, x2); G[:, wi, :, 5] += dot(x1, x2)
                            G[:, wi, :, 6] += 1
                            DN[:, wi] += drift[:, kk].double()  # relative |d_kk| / |s_kk+1|, summed; divide by count
                            del x0, x1, x2
                dm2, dm1 = dm1, d
                s = s2
                k_done = k + 1
                if k >= K - tail:
                    lg = torch.log_softmax(h.coda_logits(s, gidx), -1)
                    ro_lp.append(torch.stack([lg[:, 0][:, spd]] + [lg[:, j][:, dg] for j in range(1, P)], 1).cpu())
                    ro_am.append(lg.argmax(-1).cpu())
            for r, p in enumerate(chunk):
                n_, m_ = len(p["ids"]), len(p["pred_pos"])
                out["uid"].append(p["uid"]); out["pid"].append(p["pid"]); out["L"].append(n_); out["steps"].append(k_done)
                out["ids"].append(np.array(p["ids"])); out["pred_pos"].append(np.array(p["pred_pos"]))
                out["drift"].append(drift[r, :k_done, :n_].cpu().numpy())
                out["lag2_tail"].append(torch.stack([x[r, :n_] for x in l2]).numpy() if l2 else np.zeros((0, n_)))
                out["gram"].append(G[r, :, :n_].cpu().numpy()); out["dnorm"].append(DN[r, :, :n_].cpu().numpy())
                out["ro_lp"].append(torch.stack([x[r, :m_] for x in ro_lp]).numpy() if ro_lp else np.zeros((0, m_, 10)))
                out["ro_argmax"].append(torch.stack([x[r, :m_] for x in ro_am]).numpy() if ro_am else np.zeros((0, m_), int))
            print(f"census {c0 + N}/{len(problems)}: chunk {time.monotonic() - tc:.0f}s, total {time.monotonic() - Tc:.0f}s", flush=True)
    flat = {k: np.array(v) for k, v in out.items() if k in ("uid", "pid", "L", "steps")}
    for k in ("ids", "pred_pos", "drift", "lag2_tail", "gram", "dnorm", "ro_lp", "ro_argmax"):
        a = np.empty(len(out[k]), dtype=object); a[:] = out[k]; flat[k] = a
    flat["windows"] = np.array(W)
    np.savez_compressed(f"{o}/census.npz", allow_pickle=True, **flat)
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped,
            "dig_tokens": h.DIG, "spd_tokens": h.SPD}
    with open(f"{o}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"{o}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta



@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def ownblock(problems_json: str, cfg: dict, run_id: str):
    """Pinned-prefix validation of census modes (fp32). Each row = (prompt, position p); the sequence is truncated after p
    (causal attention: later positions cannot affect positions <= p). Steps: K loops from the model's seed-0 init, damped
    iteration to a fixed point x* (prefix residual over positions <= p < fp_tol, saved); then the OWN-BLOCK dynamics of p
    with positions < p pinned to x*: symmetric +-perturbations at p along 2 random directions x 2 amplitudes, response
    r_t = (s+_p - s-_p)/2 for T loops -> per-(direction, amplitude) Gram sums over windows and |r_t| series; and Arnoldi on
    J_pp at x* (leading Ritz values, complex-aware FD residual of the leading pair)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()
    problems = json.loads(problems_json)
    for p in problems:  # truncate after the validation position (keep answer annotation untouched for prep)
        pass
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    R_, skipped = {}, []
    late = lambda r=0.0: time.monotonic() - Tc > cfg["budget_s"] - r

    def save():
        np.savez_compressed(f"{o}/ownblock.npz", allow_pickle=True, **R_)
        meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped}
        with open(f"{o}/meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        with open(f"{o}/problems.json", "w") as f:
            f.write(problems_json)
        results_vol.commit()
        return meta

    with torch.no_grad():
        n = len(problems)
        P_ = [int(p["val_pos"]) for p in problems]
        Lm = max(P_) + 1
        ids = torch.full((n, Lm), h.NL, dtype=torch.long)
        for r, p in enumerate(problems):
            ids[r, : P_[r] + 1] = torch.tensor(p["ids"][: P_[r] + 1])
        ids = ids.to(h.dev)
        e, _ = h.model.embed_inputs(ids)
        s = h.model.initialize_state(e)
        for r, p in enumerate(problems):  # same draw as the census run: full-length init for (uid, seed 0), then slice
            full = h.init_state(torch.zeros((len(p["ids"]), h.d), device=h.dev, dtype=e.dtype), p["uid"], 0)
            s[r, : P_[r] + 1] = full[: P_[r] + 1]
        pos = torch.tensor(P_, device=h.dev)
        ar = torch.arange(n, device=h.dev)
        pre = torch.arange(Lm, device=h.dev)[None] <= pos[:, None]
        k_done = 0
        for k in range(cfg["K"]):
            if late(150):
                skipped.append(f"initial run stopped at {k}"); break
            s = h.step(e, s, k)
            k_done = k + 1
        R_["K_done"] = k_done
        x = s.clone()
        done = torch.zeros(n, dtype=torch.bool, device=h.dev)
        for it in range(cfg["n_damp"]):
            act = (~done).nonzero().flatten()
            if len(act) == 0 or late(60):
                break
            fx = h.step(e[act], x[act], 0)
            rr = (fx - x[act]).norm(dim=-1) / x[act].norm(dim=-1)
            x[act] = x[act] + 0.5 * (fx - x[act])
            done[act] = (rr * pre[act]).max(-1).values < cfg["fp_tol"]
        fx = h.step(e, x, 0)
        res = (((fx - x).norm(dim=-1) / x.norm(dim=-1)) * pre).max(-1).values
        R_.update(uid=np.array([p["uid"] for p in problems]), pos=np.array(P_), fp_resid=res.cpu().numpy(),
                  fp_valid=(res < cfg["fp_tol"]).cpu().numpy(), drift_K=((s - x).norm(dim=-1)[ar, pos] / x[ar, pos].norm(dim=-1)).cpu().numpy())
        save()
        print(f"fixed points {time.monotonic() - Tc:.0f}s; valid {int((res < cfg['fp_tol']).sum())}/{n}", flush=True)
        # own-block perturbation response with prefix pinned
        g = torch.Generator(device=h.dev).manual_seed(17)
        nd, amps, T = 2, cfg["amps"], cfg["T"]
        W = [tuple(w) for w in cfg["windows"]]
        rows = [(r, di, ai) for r in range(n) for di in range(nd) for ai in range(len(amps))]
        dirs = torch.randn((n, nd, h.d), device=h.dev, generator=g)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        ri = torch.tensor([r for r, _, _ in rows], device=h.dev)
        nq = x[ar, pos].norm(dim=-1)
        dv = torch.stack([amps[ai] * nq[r] * dirs[r, di] for r, di, ai in rows])
        base = x[ri]
        sp, sm = base.clone(), base.clone()
        pr = pos[ri]; arr = torch.arange(len(rows), device=h.dev)
        sp[arr, pr] += dv; sm[arr, pr] -= dv
        pin = (torch.arange(Lm, device=h.dev)[None] < pr[:, None])[..., None]
        E2 = torch.cat([e[ri], e[ri]])
        G = torch.zeros((len(rows), len(W), 7), device=h.dev, dtype=torch.float64)
        norms = []
        hist = []
        r0 = ((sp[arr, pr] - sm[arr, pr]) / 2).double()
        norms.append((r0.norm(dim=-1) / nq[ri].double()).cpu().numpy())
        hist.append(r0)
        for t in range(1, T + 1):
            if late(30):
                skipped.append(f"push stopped at {t}"); break
            both = h.step(E2, torch.cat([sp, sm]), 0)
            sp, sm = both[: len(rows)], both[len(rows):]
            sp = torch.where(pin, base, sp); sm = torch.where(pin, base, sm)
            rt = ((sp[arr, pr] - sm[arr, pr]) / 2).double()
            norms.append((rt.norm(dim=-1) / nq[ri].double()).cpu().numpy())
            hist.append(rt)
            if len(hist) >= 4:  # differences d_k = r_k (response itself decays with the block's modes)
                x0, x1, x2 = hist[-3], hist[-2], hist[-1]
                kk = t - 2
                for wi, (a_, b_) in enumerate(W):
                    if a_ <= kk and kk + 2 < b_:
                        dot = lambda u, v: (u * v).sum(-1)
                        G[:, wi, 0] += dot(x0, x0); G[:, wi, 1] += dot(x1, x1); G[:, wi, 2] += dot(x2, x2)
                        G[:, wi, 3] += dot(x0, x1); G[:, wi, 4] += dot(x0, x2); G[:, wi, 5] += dot(x1, x2); G[:, wi, 6] += 1
                hist = hist[-3:]
        R_.update(push_rows=np.array(rows), push_amps=np.array(amps), push_windows=np.array(W), push_gram=G.cpu().numpy(),
                  push_norms=np.stack(norms, 1))
        save()
        print(f"pushes {time.monotonic() - Tc:.0f}s", flush=True)
        # Arnoldi on J_pp at x*
        if not late(40):
            m = cfg["m"]
            ar2 = torch.arange(2 * n, device=h.dev); p2 = torch.cat([pos, pos]); Ee = torch.cat([e, e]); xx = torch.cat([x, x])

            def jv(v, ep=cfg["eps"]):
                st = xx.clone()
                st[ar2, p2] += torch.cat([ep * v, -ep * v]).to(st.dtype)
                d_ = h.step(Ee, st, 0)[ar2, p2]
                return (d_[:n] - d_[n:]) / (2 * ep)

            V = torch.zeros((n, m + 1, h.d), device=h.dev, dtype=torch.float64)
            Hm = torch.zeros((n, m + 1, m), device=h.dev, dtype=torch.float64)
            v0 = torch.randn((n, h.d), device=h.dev, dtype=torch.float64, generator=torch.Generator(device=h.dev).manual_seed(4))
            V[:, 0] = v0 / v0.norm(dim=-1, keepdim=True)
            dn = 0
            for jj in range(m):
                if late(20):
                    skipped.append(f"arnoldi stopped {jj}"); break
                w = jv(V[:, jj].float()).double()
                for _ in range(2):
                    c = torch.einsum("nkd,nd->nk", V[:, : jj + 1], w)
                    Hm[:, : jj + 1, jj] += c
                    w = w - torch.einsum("nk,nkd->nd", c, V[:, : jj + 1])
                hn = w.norm(dim=-1)
                Hm[:, jj + 1, jj] = hn
                V[:, jj + 1] = w / hn.clamp_min(1e-10)[:, None]
                dn = jj + 1
            Hc = Hm[:, : dn + 1, :dn].cpu().numpy()
            R_.update(arn_H=Hc, arn_dim=dn)
            if dn >= 4:  # complex-aware FD validation of the leading Ritz pair at two step sizes
                lead = []
                A_ = torch.zeros((n, h.d), device=h.dev, dtype=torch.float64); B_ = torch.zeros_like(A_)
                for j in range(n):
                    lam, Y = np.linalg.eig(Hc[j, :dn, :dn])
                    k0 = int(np.argmax(np.abs(lam))); lead.append(lam[k0])
                    a_ = V[j, :dn].T @ torch.tensor(Y[:, k0].real.copy(), device=h.dev)
                    b_ = V[j, :dn].T @ torch.tensor(Y[:, k0].imag.copy(), device=h.dev)
                    z_ = torch.sqrt(a_.norm() ** 2 + b_.norm() ** 2); A_[j], B_[j] = a_ / z_, b_ / z_
                lead = np.array(lead)
                al = torch.tensor(lead.real, device=h.dev); be = torch.tensor(lead.imag, device=h.dev)
                fdr = np.full((n, 2), np.nan); Js = []
                for ei, ep in enumerate((cfg["eps"], cfg["eps"] / 2)):
                    if late(10):
                        skipped.append("ritz FD"); break
                    Ja, Jb = jv(A_.float(), ep).double(), jv(B_.float(), ep).double()
                    Js.append(torch.cat([Ja, Jb], -1))
                    ra = Ja - al[:, None] * A_ + be[:, None] * B_; rb = Jb - be[:, None] * A_ - al[:, None] * B_
                    fdr[:, ei] = torch.sqrt(ra.norm(dim=-1) ** 2 + rb.norm(dim=-1) ** 2).cpu().numpy()
                R_.update(arn_lead=lead, arn_lead_fd_resid=fdr,
                          arn_lead_eps_rel_diff=((Js[0] - Js[1]).norm(dim=-1) / Js[0].norm(dim=-1)).cpu().numpy() if len(Js) == 2 else np.full(n, np.nan))
            save()
            print(f"arnoldi {time.monotonic() - Tc:.0f}s", flush=True)
        meta = save()
    return meta



@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def ownspec(problems_json: str, cfg: dict, run_id: str):
    """Convergence check of the own-block spectrum J_pp at x* (fp32 model, fp64 Krylov basis). Same rows, init, K loops and
    damped fixed point as `ownblock` (truncate after p; seed-0 full-length init, sliced). Then S independent Arnoldi starts
    (seeds cfg["start_seeds"]) of size m each, batched over (row, start); the full Hessenberg H is saved per (row, start), so
    Ritz values at every Krylov size m' <= m are the eigenvalues of its leading m' x m' block (computed offline), with the
    Arnoldi residual estimate |h_{m'+1,m'} y_{m'}|. Explicit validation at the final m for each (row, start): the top
    cfg["n_top"] Ritz values by modulus plus up to cfg["n_neg"] real-negative-dominant candidates (most negative real part
    among Ritz values with Re < 0), complex-aware central-FD residual ||J[a b] - [a b][[al, be], [-be, al]]|| (unit-norm pair)
    at eps and eps/2."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    R_, skipped = {}, []
    late = lambda r=0.0: time.monotonic() - Tc > cfg["budget_s"] - r

    def save():
        np.savez_compressed(f"{o}/ownspec.npz", allow_pickle=True, **R_)
        meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped}
        with open(f"{o}/meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        with open(f"{o}/problems.json", "w") as f:
            f.write(problems_json)
        results_vol.commit()
        return meta

    with torch.no_grad():
        n = len(problems)
        P_ = [int(p["val_pos"]) for p in problems]
        Lm = max(P_) + 1
        ids = torch.full((n, Lm), h.NL, dtype=torch.long)
        for r, p in enumerate(problems):
            ids[r, : P_[r] + 1] = torch.tensor(p["ids"][: P_[r] + 1])
        ids = ids.to(h.dev)
        e, _ = h.model.embed_inputs(ids)
        s = h.model.initialize_state(e)
        for r, p in enumerate(problems):  # identical to ownblock: full-length init for (uid, seed 0), then slice
            full = h.init_state(torch.zeros((len(p["ids"]), h.d), device=h.dev, dtype=e.dtype), p["uid"], 0)
            s[r, : P_[r] + 1] = full[: P_[r] + 1]
        pos = torch.tensor(P_, device=h.dev)
        pre = torch.arange(Lm, device=h.dev)[None] <= pos[:, None]
        k_done = 0
        for k in range(cfg["K"]):
            if late(200):
                skipped.append(f"initial run stopped at {k}"); break
            s = h.step(e, s, k)
            k_done = k + 1
        R_["K_done"] = k_done
        x = s.clone()
        done = torch.zeros(n, dtype=torch.bool, device=h.dev)
        for it in range(cfg["n_damp"]):
            act = (~done).nonzero().flatten()
            if len(act) == 0 or late(150):
                break
            fx = h.step(e[act], x[act], 0)
            rr = (fx - x[act]).norm(dim=-1) / x[act].norm(dim=-1)
            x[act] = x[act] + 0.5 * (fx - x[act])
            done[act] = (rr * pre[act]).max(-1).values < cfg["fp_tol"]
        fx = h.step(e, x, 0)
        res = (((fx - x).norm(dim=-1) / x.norm(dim=-1)) * pre).max(-1).values
        R_.update(uid=np.array([p["uid"] for p in problems]), pos=np.array(P_), fp_resid=res.cpu().numpy(),
                  fp_valid=(res < cfg["fp_tol"]).cpu().numpy())
        save()
        print(f"fixed points {time.monotonic() - Tc:.0f}s; valid {int((res < cfg['fp_tol']).sum())}/{n}", flush=True)

        # batched Arnoldi over (start, row): batch index b = si * n + r
        S, m = len(cfg["start_seeds"]), cfg["m"]
        nb = S * n
        rb = torch.arange(nb, device=h.dev) % n
        Eb, Xb, Pb = e[rb], x[rb], pos[rb]

        def jv(v, ep, sel=None):  # v: (len(sel), d) float32; central FD of the own-block map at x* for batch rows sel
            sel = torch.arange(nb, device=h.dev) if sel is None else sel
            k_ = len(sel)
            ar2 = torch.arange(2 * k_, device=h.dev); p2 = torch.cat([Pb[sel], Pb[sel]])
            st = torch.cat([Xb[sel], Xb[sel]]).clone()
            st[ar2, p2] += torch.cat([ep * v, -ep * v]).to(st.dtype)
            E2 = torch.cat([Eb[sel], Eb[sel]])
            out = []
            for c0 in range(0, 2 * k_, cfg["chunk"]):  # chunk the forward pass
                c1 = min(2 * k_, c0 + cfg["chunk"])
                out.append(h.step(E2[c0:c1], st[c0:c1], 0)[torch.arange(c1 - c0, device=h.dev), p2[c0:c1]])
            d_ = torch.cat(out)
            return (d_[:k_] - d_[k_:]) / (2 * ep)

        V = torch.zeros((nb, m + 1, h.d), device=h.dev, dtype=torch.float64)
        Hm = torch.zeros((nb, m + 1, m), device=h.dev, dtype=torch.float64)
        v0 = torch.cat([torch.randn((n, h.d), device=h.dev, dtype=torch.float64,
                                    generator=torch.Generator(device=h.dev).manual_seed(int(sd))) for sd in cfg["start_seeds"]])
        V[:, 0] = v0 / v0.norm(dim=-1, keepdim=True)
        dn = 0
        for jj in range(m):
            if late(cfg["reserve_val_s"]):
                skipped.append(f"arnoldi stopped {jj}"); break
            w = jv(V[:, jj].float(), cfg["eps"]).double()
            for _ in range(2):  # CGS2
                c = torch.einsum("nkd,nd->nk", V[:, : jj + 1], w)
                Hm[:, : jj + 1, jj] += c
                w = w - torch.einsum("nk,nkd->nd", c, V[:, : jj + 1])
            hn = w.norm(dim=-1)
            Hm[:, jj + 1, jj] = hn
            V[:, jj + 1] = w / hn.clamp_min(1e-10)[:, None]
            dn = jj + 1
            if dn in cfg["save_at"]:
                R_.update(arn_H=Hm[:, : dn + 1, :dn].cpu().numpy().reshape(S, n, dn + 1, dn), arn_dim=dn)
                save()
                print(f"arnoldi m={dn} {time.monotonic() - Tc:.0f}s", flush=True)
        Hc = Hm[:, : dn + 1, :dn].cpu().numpy()
        R_.update(arn_H=Hc.reshape(S, n, dn + 1, dn), arn_dim=dn)
        save()
        # explicit FD validation of selected Ritz pairs at the final size
        if dn >= 4:
            nt, nn = cfg["n_top"], cfg["n_neg"]
            nv = nt + nn
            lamv = np.full((nb, nv), np.nan, dtype=complex)
            kind = np.zeros((nb, nv), dtype=int)  # 0 = top-modulus, 1 = negative candidate, -1 = empty
            A_ = torch.zeros((nb, nv, h.d), device=h.dev, dtype=torch.float64); B_ = torch.zeros_like(A_)
            for b in range(nb):
                lam, Y = np.linalg.eig(Hc[b, :dn, :dn])
                order = list(np.argsort(-np.abs(lam)))
                top = []
                for k0 in order:  # one representative per conjugate pair (Im >= 0)
                    if lam[k0].imag < -1e-12:
                        continue
                    top.append(k0)
                    if len(top) == nt:
                        break
                # near-real negative candidates (same criterion as the analysis: |Im| <= 0.05 |lam|), most negative first
                negs = [k0 for k0 in np.argsort(lam.real) if lam[k0].real < 0 and lam[k0].imag >= -1e-12
                        and abs(lam[k0].imag) <= 0.05 * abs(lam[k0]) and k0 not in top][:nn]
                for slot, k0 in enumerate(top + negs):
                    a_ = V[b, :dn].T @ torch.tensor(Y[:, k0].real.copy(), device=h.dev)
                    b_ = V[b, :dn].T @ torch.tensor(Y[:, k0].imag.copy(), device=h.dev)
                    z_ = torch.sqrt(a_.norm() ** 2 + b_.norm() ** 2)
                    A_[b, slot], B_[b, slot] = a_ / z_, b_ / z_
                    lamv[b, slot] = lam[k0]
                kind[b] = [0] * len(top) + [1] * len(negs) + [-1] * (nv - len(top) - len(negs))
            fdr = np.full((nb, nv, 2), np.nan); epsd = np.full((nb, nv), np.nan)
            al = torch.tensor(np.nan_to_num(lamv.real), device=h.dev); be = torch.tensor(np.nan_to_num(lamv.imag), device=h.dev)
            for slot in range(nv):
                Js = []
                for ei, ep in enumerate((cfg["eps"], cfg["eps"] / 2)):
                    if late(10):
                        skipped.append(f"ritz FD slot {slot} eps {ep}"); break
                    Ja, Jb = jv(A_[:, slot].float(), ep).double(), jv(B_[:, slot].float(), ep).double()
                    Js.append(torch.cat([Ja, Jb], -1))
                    a_, b_ = A_[:, slot], B_[:, slot]
                    ra = Ja - al[:, slot, None] * a_ + be[:, slot, None] * b_
                    rb_ = Jb - be[:, slot, None] * a_ - al[:, slot, None] * b_
                    fdr[:, slot, ei] = torch.sqrt(ra.norm(dim=-1) ** 2 + rb_.norm(dim=-1) ** 2).cpu().numpy()
                if len(Js) == 2:  # relative eps-consistency of the FD matvecs ||J_eps [a b] - J_eps/2 [a b]|| / ||J_eps [a b]||
                    epsd[:, slot] = ((Js[0] - Js[1]).norm(dim=-1) / Js[0].norm(dim=-1)).cpu().numpy()
            R_.update(val_lam=lamv.reshape(S, n, nv), val_kind=kind.reshape(S, n, nv), val_fd_resid=fdr.reshape(S, n, nv, 2),
                      val_eps_rel_diff=epsd.reshape(S, n, nv))
            save()
            print(f"validation {time.monotonic() - Tc:.0f}s", flush=True)
        meta = save()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def freegen(problems_json: str, cfg: dict, run_id: str):
    """Free-running greedy generation at adjacent loop budgets (fp32, noise off). Per prompt: forced prefix = prompt + true
    answer tokens before gen_from (identical to free-running there, see select_freegen.py); then generate answer positions
    gen_from .. n_ans (n_ans - gen_from + 1 tokens; a FIXED-HORIZON continuation probe: the emitted token at index n_ans,
    the reference termination position, is recorded but generation never stops early). Census reproduction gate: at the
    first generated position, the tail argmax is compared with the census tail argmax (p["census_tail_argmax"], 64 loops)
    and the per-prompt match is saved. The whole sequence is recomputed from
    the init state for every new token (no cache): K = max(budgets) loops, full-vocab logits at the last position read every
    loop over the last `tail` loops. Each loop budget b in cfg["budgets"] follows its own greedy branch: branches split
    whenever budgets disagree on the next token. Init: the census draw (init_state of a full-length zero row for
    (uid, seed 0), sliced); the longest forward is len(prompt + true answer) (predicting the termination token), so every
    position has a census init row."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    budgets = sorted(cfg["budgets"]); K = budgets[-1]; tail = cfg["tail"]
    assert K - tail + 1 <= budgets[0]
    skipped, log = [], []
    spd, dg = torch.tensor(h.SPD, device=h.dev), torch.tensor(h.DIG, device=h.dev)
    with torch.no_grad():
        inits = {}
        for p in problems:
            n = len(p["ids"])
            full = h.init_state(torch.zeros((n, h.d), device=h.dev), p["uid"], 0)
            ids_ = torch.tensor([p["ids"]], device=h.dev)
            e_, _ = h.model.embed_inputs(ids_)
            assert torch.equal(full.to(e_.dtype), h.init_state(e_[0], p["uid"], 0)), "init depends on embedding values"
            inits[p["uid"]] = full.to(e_.dtype)
        # branch = dict(r=problem index, toks=list, budgets=list, hist=list of per-step records)
        branches = []
        for r, p in enumerate(problems):
            Lp = p["pred_pos"][0] + 1  # prompt length
            branches.append(dict(r=r, toks=list(p["ids"][: Lp + p["gen_from"]]), budgets=list(budgets), hist=[], done=False))
        step, last_round_s, last_round_n, repro = 0, 0.0, 1, {}
        while any(not b["done"] for b in branches):
            act = [b for b in branches if not b["done"]]
            if time.monotonic() - Tc + last_round_s * len(act) / max(last_round_n, 1) > cfg["budget_s"]:
                skipped.append(f"stopped before gen step {step} with {len(act)} active branches"); break
            tc = time.monotonic()
            L = max(len(b["toks"]) for b in act)
            ids = torch.full((len(act), L), h.NL, dtype=torch.long)
            for i, b in enumerate(act):
                ids[i, : len(b["toks"])] = torch.tensor(b["toks"])
            ids = ids.to(h.dev)
            e, _ = h.model.embed_inputs(ids)
            s = h.model.initialize_state(e)
            for i, b in enumerate(act):
                n = len(b["toks"])
                s[i, :n] = inits[problems[b["r"]]["uid"]][:n]
            last = torch.tensor([[len(b["toks"]) - 1] for b in act], device=h.dev)
            am_t, top2_t, dl_t = [], [], []
            for k in range(K):
                s = h.step(e, s, k)
                if k + 1 > K - tail:
                    lg = torch.log_softmax(h.coda_logits(s, last)[:, 0], -1)
                    am_t.append(lg.argmax(-1).cpu())
                    top2_t.append(lg.topk(2, -1).values.cpu())
                    dl_t.append(torch.stack([lg[:, spd], lg[:, dg]], 1).cpu())  # (B, 2, 10): space-digit, bare-digit
            am_t, top2_t, dl_t = torch.stack(am_t, 1).numpy(), torch.stack(top2_t, 1).numpy(), torch.stack(dl_t, 1).numpy()
            new = []
            for i, b in enumerate(act):
                p = problems[b["r"]]
                gidx = len(b["toks"]) - (p["pred_pos"][0] + 1)  # answer index being predicted
                rec = dict(ans_idx=gidx, argmax_tail=am_t[i].tolist(), top2_tail=top2_t[i].round(4).tolist(),
                           digit_lp_tail=dl_t[i].round(3).tolist(), n_changes_tail=int((np.diff(am_t[i]) != 0).sum()))
                if step == 0 and p.get("census_tail_argmax") is not None and len(p["census_tail_argmax"]) == tail and K == 512:
                    ref = np.array(p["census_tail_argmax"])
                    repro[p["uid"]] = dict(match_all=bool(np.array_equal(ref, am_t[i][-len(ref):])),
                                           n_match=int((ref == am_t[i][-len(ref):]).sum()), n=len(ref))
                b["hist"].append(rec)
                nxt = {bd: int(am_t[i][bd - (K - tail) - 1]) for bd in b["budgets"]}
                groups = {}
                for bd, t in nxt.items():
                    groups.setdefault(t, []).append(bd)
                finished = gidx >= p["n_ans"]  # just predicted the reference termination position (fixed horizon)
                toks0, hist0 = list(b["toks"]), [dict(x) for x in b["hist"]]  # snapshot the parent before any child mutates it
                for j, (t, bds) in enumerate(sorted(groups.items())):
                    nb = b if j == 0 else dict(r=b["r"], hist=[dict(x) for x in hist0])
                    nb.update(budgets=bds, toks=toks0 + [t], done=finished)
                    if j > 0:
                        new.append(nb)
                if len(groups) > 1:
                    log.append(dict(step=step, uid=p["uid"], ans_idx=gidx, split={str(t): bds for t, bds in groups.items()}))
            branches += new
            last_round_s, last_round_n = time.monotonic() - tc, len(act)
            print(f"gen step {step}: {len(act)} branches, {time.monotonic() - tc:.0f}s, total {time.monotonic() - Tc:.0f}s, splits so far {len(log)}", flush=True)
            step += 1
    res = []
    for b in branches:
        p = problems[b["r"]]
        Lp = p["pred_pos"][0] + 1
        res.append(dict(uid=p["uid"], fg_role=p["fg_role"], budgets=b["budgets"], done=b["done"], answer_ids=b["toks"][Lp:],
                        answer_text=h.tok.decode(b["toks"][Lp:]), true_answer=p["answer"], gen_from=p["gen_from"], hist=b["hist"]))
    out = dict(branches=res, splits=log, census_repro={str(k): v for k, v in repro.items()})
    with open(f"{o}/freegen.json", "w") as f:
        json.dump(out, f)
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped,
            "n_branches": len(res), "n_splits": len(log),
            "census_repro_all_match": all(v["match_all"] for v in repro.values()), "n_repro": len(repro)}
    with open(f"{o}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"{o}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def wkcollect(problems_json: str, cfg: dict, run_id: str):
    """Weekday pilot collection (fp32, noise off, seed-0 init per uid as in the census). Per prompt: K loops; the state at
    the answer-prediction position q (last prompt token) at every loop in cfg["save_loops"] (trajectory); final states at
    q, the query's start-day token and the query's offset token; relative drift at every position every loop; readout at
    q at every saved loop: log-probs of the 7 space-prefixed day tokens, full-vocab argmax, and the total day-token
    probability mass. save_loops may include 0 (the initial state, before any loop). Completion masks: `done` per prompt,
    `ext_done` per extended prompt. Prompts whose max-over-positions of the mean drift over the last 8 loops exceeds
    stat_thr are extended to K_extend loops (same statistic recomputed there; final state and readout kept separately;
    not called fixed points)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()
    problems = json.loads(problems_json)
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    DT = [h.tok.encode(" " + d, add_special_tokens=False) for d in DAYS]
    assert all(len(t) == 1 for t in DT), DT
    DT = [t[0] for t in DT]
    for p in problems:
        pre = h.tok.encode(p["prompt"], add_special_tokens=True)
        full = h.tok.encode(p["prompt"] + p["answer"], add_special_tokens=True)
        assert full[: len(pre)] == pre and len(full) == len(pre) + 1 and full[-1] == DT[p["ans"]], p["uid"]
        p["ids"], p["pred_pos"] = full, [len(pre) - 1]
        dtok = DT[p["start"]]
        p["day_pos"] = max(i for i, t in enumerate(pre) if t == dtok)
        num = (" " + ["zero", "one", "two", "three", "four", "five", "six", "seven"][p["offset"]]) if p["template"] == "A" else f" {p['offset']}"
        ntok = h.tok.encode(num, add_special_tokens=False)
        assert len(ntok) == 1, num
        p["num_pos"] = max(i for i, t in enumerate(pre) if t == ntok[0])
        assert p["num_pos"] < p["day_pos"] if p["template"] == "A" else p["day_pos"] < p["num_pos"]
    val = h.validate(problems)
    print("validation", val, flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    K, SL = cfg["K"], sorted(cfg["save_loops"])
    assert SL[-1] == K and SL[0] >= 0
    dt = torch.tensor(DT, device=h.dev)
    n = len(problems)
    traj = np.zeros((n, len(SL), h.d), np.float32)
    fin = np.zeros((n, 3, h.d), np.float32)  # q, day_pos, num_pos at loop K
    ro_lp = np.zeros((n, len(SL), 7), np.float32); ro_am = np.zeros((n, len(SL)), np.int64); ro_mass = np.zeros((n, len(SL)), np.float32)
    drift = [None] * n
    done = np.zeros(n, bool)
    skipped = []
    with torch.no_grad():
        for c0 in range(0, n, cfg["chunk"]):
            if time.monotonic() - Tc > cfg["budget_s"] - 60:
                skipped.append(f"chunk {c0}"); continue
            tc = time.monotonic()
            chunk = problems[c0 : c0 + cfg["chunk"]]
            N = len(chunk)
            ids, e, s = h.batch(chunk, [0] * N)
            # drop the answer token: run on the prompt only (causal, so identical for positions <= q); keep census-style
            # full-length init draw sliced to the prompt
            Lq = max(p["pred_pos"][0] + 1 for p in chunk)
            e, s = e[:, :Lq], s[:, :Lq]
            ar = torch.arange(N, device=h.dev)
            q = torch.tensor([p["pred_pos"][0] for p in chunk], device=h.dev)
            dp = torch.tensor([p["day_pos"] for p in chunk], device=h.dev); np_ = torch.tensor([p["num_pos"] for p in chunk], device=h.dev)
            dr = torch.zeros((N, K, Lq), device=h.dev)
            si = 0

            def save_point(si, s):
                traj[c0 : c0 + N, si] = s[ar, q].cpu().numpy()
                lg = torch.log_softmax(h.coda_logits(s, q[:, None])[:, 0], -1)
                ro_lp[c0 : c0 + N, si] = lg[:, dt].cpu().numpy(); ro_am[c0 : c0 + N, si] = lg.argmax(-1).cpu().numpy()
                ro_mass[c0 : c0 + N, si] = lg[:, dt].exp().sum(-1).cpu().numpy()
            if SL[0] == 0:
                save_point(0, s); si = 1
            for k in range(K):
                s2 = h.step(e, s, k)
                dr[:, k] = (s2 - s).norm(dim=-1) / s2.norm(dim=-1)
                s = s2
                if k + 1 == SL[si]:
                    save_point(si, s)
                    si += 1
            fin[c0 : c0 + N] = torch.stack([s[ar, q], s[ar, dp], s[ar, np_]], 1).cpu().numpy()
            for r, p in enumerate(chunk):
                drift[c0 + r] = dr[r, :, : p["pred_pos"][0] + 1].cpu().numpy()
            done[c0 : c0 + N] = True
            print(f"chunk {c0 + N}/{n}: {time.monotonic() - tc:.0f}s, total {time.monotonic() - Tc:.0f}s", flush=True)
        # extension of unresolved prompts
        late8 = np.array([d[-8:].mean(0).max() if d is not None else np.nan for d in drift])
        ext = [i for i in range(n) if done[i] and late8[i] > cfg["stat_thr"]]
        ext_done = np.zeros(len(ext), bool)
        ext_fin = np.zeros((len(ext), h.d), np.float32); ext_lp = np.zeros((len(ext), 7), np.float32)
        ext_am = np.zeros(len(ext), np.int64); ext_late8 = np.full(len(ext), np.nan)
        if ext and time.monotonic() - Tc < cfg["budget_s"] - 30:
            sub = [problems[i] for i in ext]
            for c0 in range(0, len(sub), cfg["chunk"]):
                chunk = sub[c0 : c0 + cfg["chunk"]]
                N = len(chunk)
                ids, e, s = h.batch(chunk, [0] * N)
                Lq = max(p["pred_pos"][0] + 1 for p in chunk)
                e, s = e[:, :Lq], s[:, :Lq]
                ar = torch.arange(N, device=h.dev)
                q = torch.tensor([p["pred_pos"][0] for p in chunk], device=h.dev)
                pre = torch.arange(Lq, device=h.dev)[None] <= q[:, None]
                acc = torch.zeros((N, Lq), device=h.dev)
                for k in range(cfg["K_extend"]):
                    s2 = h.step(e, s, k)
                    if k >= cfg["K_extend"] - 8:
                        acc += ((s2 - s).norm(dim=-1) / s2.norm(dim=-1)) * pre / 8  # per-position mean, max below
                    s = s2
                acc = acc.max(-1).values
                lg = torch.log_softmax(h.coda_logits(s, q[:, None])[:, 0], -1)
                ext_fin[c0 : c0 + N] = s[ar, q].cpu().numpy(); ext_lp[c0 : c0 + N] = lg[:, dt].cpu().numpy()
                ext_am[c0 : c0 + N] = lg.argmax(-1).cpu().numpy(); ext_late8[c0 : c0 + N] = acc.cpu().numpy()
                ext_done[c0 : c0 + N] = True
            print(f"extended {len(ext)} prompts to {cfg['K_extend']}: total {time.monotonic() - Tc:.0f}s", flush=True)
        elif ext:
            skipped.append(f"extension of {len(ext)} prompts")
    dobj = np.empty(n, dtype=object); dobj[:] = drift
    np.savez_compressed(f"{o}/wk.npz", allow_pickle=True, uid=np.array([p["uid"] for p in problems]), save_loops=np.array(SL),
                        traj=traj, fin=fin, ro_lp=ro_lp, ro_am=ro_am, ro_mass=ro_mass, drift=dobj, late8=late8,
                        day_tokens=np.array(DT), q=np.array([p["pred_pos"][0] for p in problems]),
                        day_pos=np.array([p["day_pos"] for p in problems]), num_pos=np.array([p["num_pos"] for p in problems]),
                        done=done, ext_done=ext_done, ext_idx=np.array(ext, dtype=np.int64), ext_fin=ext_fin, ext_lp=ext_lp, ext_am=ext_am, ext_late8=ext_late8)
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped,
            "n_extended": len(ext), "n_ext_done": int(ext_done.sum()), "n_done": int(done.sum())}
    with open(f"{o}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"{o}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def relax(problems_json: str, cfg: dict, run_id: str):
    """Predictive source under-relaxation (fp32, noise off, census seed-0 init). Per prompt (source p0 = earliest
    period-2 position, downstream r):
    1. K0 warm-up loops (census init); the last two states s_K, s_{K-1}.
    2. Damped iteration x <- x + beta (F(x) - x) from (s_K + s_{K-1}) / 2 to a fixed point x* (relative residual over all
       positions < fp_tol); prompts without a valid x* are ineligible.
    3. Own-block Jacobian J_{p0 p0} at x* (single-step central FD at p0, eps; exact own block: one step from x* with only
       p0 perturbed, read at p0) by Arnoldi (m, CGS2, fp64 basis) from len(start_seeds) independent starts. Flip candidate
       = most negative EXACTLY-real Ritz value; plus the top-n_top Ritz values by modulus; all FD-validated (complex-aware
       residual at eps and eps/2). ELIGIBLE iff valid x*, a real flip candidate in every start, all < val_tol, start
       spread < start_tol, and lam + max(FD residual, spread) < -1. lam = mean over starts. The validated top modes are
       saved so the analysis can check that every condition's alpha keeps them inside the unit circle (|1-a+a mu| < 1).
       PREDICTION (saved before any relaxation): alpha* = 2 / (1 - lam); alpha(L) = (1 - L) / (1 - lam).
    4. Conditions, each from s_K for T loops; relaxation at position c: s' = F(s); s'[c] = (1 - a) s[c] + a s'[c]:
       sham (c = p0, a = 1), src_m0.6 (c = p0, a = alpha(-0.6)), src_m0.85 (alpha(-0.85)), src_half (alpha at
       L = (-1 + lam) / 2, only if mu = -1 - lam >= mu_min), down_m0.6 (c = r, a = alpha(-0.6)).
       Saved per condition: relative drift per loop at p0 and r, max-over-positions drift per loop, lag-2 closure at p0
       over the last 32 loops."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    late = lambda r=0.0: time.monotonic() - Tc > cfg["budget_s"] - r
    R_, skipped = {}, []

    def save():
        np.savez_compressed(f"{o}/relax.npz", allow_pickle=True, **R_)
        meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped}
        with open(f"{o}/meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        with open(f"{o}/problems.json", "w") as f:
            f.write(problems_json)
        results_vol.commit()
        return meta

    with torch.no_grad():
        n = len(problems)
        ids, e, s = h.batch(problems, [0] * n)
        L = s.shape[1]
        lens = torch.tensor([len(p["ids"]) for p in problems], device=h.dev)
        valid_pos = torch.arange(L, device=h.dev)[None] < lens[:, None]
        ar = torch.arange(n, device=h.dev)
        p0 = torch.tensor([p["p0"] for p in problems], device=h.dev)
        rr = torch.tensor([p["r"] for p in problems], device=h.dev)
        prev = None
        for k in range(cfg["K0"]):
            prev = s
            s = h.step(e, s, k)
        sK, sK1 = s.clone(), prev.clone()
        rel = lambda a, b: ((a - b).norm(dim=-1) / a.norm(dim=-1))
        R_["warm_drift_p0"] = rel(sK, sK1)[ar, p0].cpu().numpy()
        print(f"warm-up {time.monotonic() - Tc:.0f}s", flush=True)
        # fixed point by damped iteration
        x = (sK + sK1) / 2
        done = torch.zeros(n, dtype=torch.bool, device=h.dev)
        for it in range(cfg["n_damp"]):
            act = (~done).nonzero().flatten()
            if len(act) == 0 or late(300):
                break
            fx = h.step(e[act], x[act], 0)
            res = (rel(fx, x[act]) * valid_pos[act]).max(-1).values
            x[act] = x[act] + cfg["beta"] * (fx - x[act])
            done[act] = res < cfg["fp_tol"]
        fx = h.step(e, x, 0)
        fres = (rel(fx, x) * valid_pos).max(-1).values
        fp_ok = (fres < cfg["fp_tol"]).cpu().numpy()
        R_.update(fp_resid=fres.cpu().numpy(), fp_ok=fp_ok, uid=np.array([p["uid"] for p in problems]),
                  p0=p0.cpu().numpy(), r=rr.cpu().numpy(), dist_cycle_to_fp=rel(sK, x)[ar, p0].cpu().numpy())
        save()
        print(f"fixed points {time.monotonic() - Tc:.0f}s; valid {int(fp_ok.sum())}/{n}", flush=True)
        # Arnoldi on J_{p0 p0} at x*: S independent starts batched (row b = si * n + j), m steps each
        m, S = cfg["m"], len(cfg["start_seeds"])
        nb = S * n
        jb = torch.arange(nb, device=h.dev) % n
        Pb = p0[jb]; Eb = e[jb]; Xb = x[jb]
        arb2 = torch.arange(2 * nb, device=h.dev); P2 = torch.cat([Pb, Pb]); E2 = torch.cat([Eb, Eb]); X2 = torch.cat([Xb, Xb])

        def jv(v, ep):
            st = X2.clone()
            st[arb2, P2] += torch.cat([ep * v, -ep * v]).to(st.dtype)
            outs = []
            for c0_ in range(0, 2 * nb, cfg["chunk"]):
                c1_ = min(2 * nb, c0_ + cfg["chunk"])
                outs.append(h.step(E2[c0_:c1_], st[c0_:c1_], 0)[torch.arange(c1_ - c0_, device=h.dev), P2[c0_:c1_]])
            d_ = torch.cat(outs)
            return (d_[:nb] - d_[nb:]) / (2 * ep)

        V = torch.zeros((nb, m + 1, h.d), device=h.dev, dtype=torch.float64)
        Hm = torch.zeros((nb, m + 1, m), device=h.dev, dtype=torch.float64)
        v0 = torch.cat([torch.randn((n, h.d), device=h.dev, dtype=torch.float64, generator=torch.Generator(device=h.dev).manual_seed(int(sd)))
                        for sd in cfg["start_seeds"]])
        V[:, 0] = v0 / v0.norm(dim=-1, keepdim=True)
        for jj in range(m):
            w = jv(V[:, jj].float(), cfg["eps"]).double()
            for _ in range(2):
                c = torch.einsum("nkd,nd->nk", V[:, : jj + 1], w)
                Hm[:, : jj + 1, jj] += c
                w = w - torch.einsum("nk,nkd->nd", c, V[:, : jj + 1])
            hn = w.norm(dim=-1)
            Hm[:, jj + 1, jj] = hn
            V[:, jj + 1] = w / hn.clamp_min(1e-10)[:, None]
        Hc = Hm.cpu().numpy()
        # slots per (start, row): 0 = flip candidate (most negative EXACTLY-real Ritz value), 1..n_top = top-modulus Ritz
        # values (one per conjugate pair); all validated by complex-aware central FD at eps and eps/2
        nv = 1 + cfg["n_top"]
        lamv = np.full((nb, nv), np.nan, dtype=complex)
        A_ = torch.zeros((nb, nv, h.d), device=h.dev, dtype=torch.float64); B_ = torch.zeros_like(A_)
        for b in range(nb):
            lam, Y = np.linalg.eig(Hc[b, :m, :m])
            reals = [k0 for k0 in np.argsort(lam.real) if abs(lam[k0].imag) < 1e-10 and lam[k0].real < 0]
            top = []
            for k0 in np.argsort(-np.abs(lam)):
                if lam[k0].imag < -1e-12 or (reals and k0 == reals[0]):
                    continue
                top.append(k0)
                if len(top) == cfg["n_top"]:
                    break
            for slot, k0 in enumerate(([reals[0]] if reals else [None]) + top):
                if k0 is None:
                    continue
                a_ = V[b, :m].T @ torch.tensor(Y[:, k0].real.copy(), device=h.dev)
                b_ = V[b, :m].T @ torch.tensor(Y[:, k0].imag.copy(), device=h.dev)
                z_ = torch.sqrt(a_.norm() ** 2 + b_.norm() ** 2); A_[b, slot], B_[b, slot] = a_ / z_, b_ / z_
                lamv[b, slot] = lam[k0]
        fdr = np.full((nb, nv, 2), np.nan)
        al = torch.tensor(np.nan_to_num(lamv.real), device=h.dev); be = torch.tensor(np.nan_to_num(lamv.imag), device=h.dev)
        for slot in range(nv):
            for ei, ep in enumerate((cfg["eps"], cfg["eps"] / 2)):
                Ja, Jb = jv(A_[:, slot].float(), ep).double(), jv(B_[:, slot].float(), ep).double()
                a_, b_ = A_[:, slot], B_[:, slot]
                ra = Ja - al[:, slot, None] * a_ + be[:, slot, None] * b_; rb_ = Jb - be[:, slot, None] * a_ - al[:, slot, None] * b_
                fdr[:, slot, ei] = torch.sqrt(ra.norm(dim=-1) ** 2 + rb_.norm(dim=-1) ** 2).cpu().numpy()
        lamv = lamv.reshape(S, n, nv); fdr = fdr.reshape(S, n, nv, 2)
        flip = lamv[:, :, 0].real  # (S, n)
        flip_res = np.nan_to_num(fdr[:, :, 0].max(-1), nan=9.0)
        lam_r = np.nanmean(flip, 0)
        spread = np.nanmax(flip, 0) - np.nanmin(flip, 0)
        unc = np.maximum(np.max(flip_res, 0), spread)  # crude uncertainty: worst FD residual or start disagreement
        # ELIGIBLE: valid x*, an exactly-real flip candidate in every start, all FD-validated (< val_tol at both eps), starts
        # agree within start_tol, and the candidate lies past -1 by more than its uncertainty (lam + unc < -1)
        eligible = (fp_ok & np.all(np.isfinite(flip), 0) & np.all(flip_res < cfg["val_tol"], 0) & (spread < cfg["start_tol"])
                    & (lam_r + unc < -1))
        if cfg.get("smoke_force_eligible"):  # smoke only: exercise the condition code path
            lam_r = np.where(np.isfinite(lam_r) & (lam_r < -1), lam_r, -1.05); eligible = np.ones(n, bool)
        alpha = lambda Lt: (1 - Lt) / (1 - lam_r)
        mu = -1 - lam_r
        R_.update(arn_H=Hc.reshape(S, n, m + 1, m), val_lam=lamv, val_fd_resid=fdr, flip_lam=lam_r, flip_spread=spread, flip_unc=unc,
                  eligible=eligible, alpha_star=2 / (1 - lam_r), mu=mu)
        save()
        print(f"spectra {time.monotonic() - Tc:.0f}s; eligible {int(eligible.sum())}/{n}; lam {np.round(lam_r, 3).tolist()}", flush=True)
        # conditions
        conds = []  # (row, name, position, alpha, predicted multiplier)
        for j in np.where(eligible)[0]:
            conds.append((j, "sham", int(p0[j]), 1.0, float(lam_r[j])))
            conds.append((j, "src_m0.6", int(p0[j]), float(alpha(-0.6)[j]), -0.6))
            conds.append((j, "src_m0.85", int(p0[j]), float(alpha(-0.85)[j]), -0.85))
            if mu[j] >= cfg["mu_min"]:
                Lh = (-1 + lam_r[j]) / 2
                conds.append((j, "src_half", int(p0[j]), float((1 - Lh) / (1 - lam_r[j])), float(Lh)))
            conds.append((j, "down_m0.6", int(rr[j]), float(alpha(-0.6)[j]), float("nan")))
        nc = len(conds)
        ci = torch.tensor([c[0] for c in conds], device=h.dev, dtype=torch.long)
        cpos = torch.tensor([c[2] for c in conds], device=h.dev, dtype=torch.long)
        calpha = torch.tensor([c[3] for c in conds], device=h.dev, dtype=torch.float32)
        T = cfg["T"]
        drift_p0 = np.zeros((nc, T), np.float32); drift_r = np.zeros((nc, T), np.float32); drift_max = np.zeros((nc, T), np.float32)
        tail_pos = np.zeros((nc, L), np.float32)  # per-position mean drift over the last 32 loops (localise residual activity)
        l2 = np.full(nc, np.nan)
        done_c = np.zeros(nc, bool)
        for c0 in range(0, nc, cfg["chunk"]):
            if late(15):
                skipped.append(f"conditions from {c0}"); break
            sl = slice(c0, min(nc, c0 + cfg["chunk"]))
            idx = ci[sl]; pos = cpos[sl]; a = calpha[sl][:, None]
            Ec, sc = e[idx], sK[idx].clone()
            arc = torch.arange(len(idx), device=h.dev)
            vp = valid_pos[idx]
            hist = []
            tail_acc = torch.zeros((len(idx), L), device=h.dev)
            aborted = False
            for t in range(T):
                if late(2):
                    aborted = True; skipped.append(f"conditions chunk {c0} aborted at loop {t}"); break
                s2 = h.step(Ec, sc, 0)
                s2[arc, pos] = (1 - a) * sc[arc, pos] + a * s2[arc, pos]
                d = rel(s2, sc)
                drift_p0[sl, t] = d[arc, p0[idx]].cpu().numpy(); drift_r[sl, t] = d[arc, rr[idx]].cpu().numpy()
                drift_max[sl, t] = (d * vp).max(-1).values.cpu().numpy()
                if t >= T - 34:
                    hist.append(s2[arc, p0[idx]].clone())
                if t >= T - 32:
                    tail_acc += d * vp / 32
                sc = s2
            if aborted:
                break
            tail_pos[sl] = tail_acc.cpu().numpy()
            # lag-2 closure at p0 over the last 32 loops: median ||s_{t+1} - s_{t-1}|| / ||s_{t+1} - s_t||
            H_ = torch.stack(hist)  # (34, b, d)
            num = (H_[2:] - H_[:-2]).norm(dim=-1); den = (H_[2:] - H_[1:-1]).norm(dim=-1).clamp_min(1e-30)
            l2[sl] = (num / den).median(0).values.cpu().numpy()
            done_c[sl] = True
            print(f"conditions {sl.stop}/{nc} {time.monotonic() - Tc:.0f}s", flush=True)
        R_.update(cond_row=np.array([c[0] for c in conds]), cond_name=np.array([c[1] for c in conds]), cond_pos=np.array([c[2] for c in conds]),
                  cond_alpha=np.array([c[3] for c in conds]), cond_pred_mult=np.array([c[4] for c in conds]), cond_done=done_c,
                  drift_p0=drift_p0, drift_r=drift_r, drift_max=drift_max, lag2_p0=l2, tail_pos=tail_pos)
        meta = save()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def dgcollect(problems_json: str, cfg: dict, run_id: str):
    """Digit-circle rotation pilot collection (fp32, noise off, census seed-0 init per uid). Teacher-forced prompt + tens
    answer digit; q = position of the tens answer token (predicts the units digit U = (a + b) mod 10). Per prompt: K loops;
    the state at q at EVERY loop 0..K (fp32); final states (fp32) at q, at the first operand's units-digit
    token (a) and at the second operand's units-digit token (b); readout at q every `ro_every` loops (and at K): log-probs
    of the 10 bare digit tokens, full-vocab argmax, digit mass; relative drift at every position every loop. Completion
    mask `done`."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()
    problems = json.loads(problems_json)
    h.prep(problems)
    for p in problems:
        ids = p["ids"]; q = p["pred_pos"][-1]
        assert ids[q + 1] == h.DIG[p["ans"]] and q == len(ids) - 2
        p["a_pos"], p["b_pos"] = q - 5, q - 2  # tokens: ..., a_u, " +", " b_t", b_u, " =", " T" (q)
        assert ids[p["a_pos"]] == h.DIG[p["start"]] and ids[p["b_pos"]] == h.DIG[p["offset"]], p["uid"]
    val = h.validate(problems)
    print("validation", val, flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    K = cfg["K"]
    RO = sorted(set(list(range(0, K + 1, cfg["ro_every"])) + [K]))
    dg = torch.tensor(h.DIG, device=h.dev)
    n = len(problems)
    traj = np.zeros((n, K + 1, h.d), np.float32)
    fin = np.zeros((n, 3, h.d), np.float32)
    ro_lp = np.zeros((n, len(RO), 10), np.float32); ro_am = np.zeros((n, len(RO)), np.int64); ro_mass = np.zeros((n, len(RO)), np.float32)
    drift = np.zeros((n, K, max(len(p["ids"]) for p in problems) - 1), np.float32)
    done = np.zeros(n, bool)
    skipped = []
    with torch.no_grad():
        for c0 in range(0, n, cfg["chunk"]):
            if time.monotonic() - Tc > cfg["budget_s"] - 60:
                skipped.append(f"chunk {c0}"); continue
            tc = time.monotonic()
            chunk = problems[c0 : c0 + cfg["chunk"]]
            N = len(chunk)
            ids, e, s = h.batch(chunk, [0] * N)
            Lq = max(p["pred_pos"][-1] + 1 for p in chunk)  # drop the units answer token (causal: identical for <= q)
            e, s = e[:, :Lq], s[:, :Lq]
            ar = torch.arange(N, device=h.dev)
            q = torch.tensor([p["pred_pos"][-1] for p in chunk], device=h.dev)
            ap = torch.tensor([p["a_pos"] for p in chunk], device=h.dev); bp = torch.tensor([p["b_pos"] for p in chunk], device=h.dev)

            def readout(k, s):
                i = RO.index(k)
                lg = torch.log_softmax(h.coda_logits(s, q[:, None])[:, 0], -1)
                ro_lp[c0 : c0 + N, i] = lg[:, dg].cpu().numpy(); ro_am[c0 : c0 + N, i] = lg.argmax(-1).cpu().numpy()
                ro_mass[c0 : c0 + N, i] = lg[:, dg].exp().sum(-1).cpu().numpy()
            traj[c0 : c0 + N, 0] = s[ar, q].cpu().numpy()
            readout(0, s)
            for k in range(K):
                s2 = h.step(e, s, k)
                drift[c0 : c0 + N, k, :Lq] = ((s2 - s).norm(dim=-1) / s2.norm(dim=-1)).cpu().numpy()
                s = s2
                traj[c0 : c0 + N, k + 1] = s[ar, q].cpu().numpy()
                if k + 1 in RO:
                    readout(k + 1, s)
            fin[c0 : c0 + N] = torch.stack([s[ar, q], s[ar, ap], s[ar, bp]], 1).cpu().numpy()
            done[c0 : c0 + N] = True
            print(f"chunk {c0 + N}/{n}: {time.monotonic() - tc:.0f}s, total {time.monotonic() - Tc:.0f}s", flush=True)
    np.savez_compressed(f"{o}/dg.npz", allow_pickle=True, uid=np.array([p["uid"] for p in problems]), traj=traj, fin=fin,
                        ro_loops=np.array(RO), ro_lp=ro_lp, ro_am=ro_am, ro_mass=ro_mass, drift=drift, done=done,
                        dig_tokens=np.array(h.DIG), q=np.array([p["pred_pos"][-1] for p in problems]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped,
            "n_done": int(done.sum())}
    with open(f"{o}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"{o}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def relaxsweep(problems_json: str, cfg: dict, run_id: str):
    """Threshold sweep of source under-relaxation (fp32, noise off, census seed-0 init). Uses the PREDICTED threshold
    alpha* saved by the earlier `relax` run (p["alpha_star_pred"]; not recomputed). Per prompt: K0 warm-up loops (identical
    to `relax`, deterministic), then for each multiple r in cfg["r_mults"] relax the source p0 with a = r alpha* for T
    loops from s_K: s' = F(s); s'[p0] = (1 - a) s[p0] + a s'[p0]. Predicted relaxed flip multiplier 1 - 2r (independent of
    lam). Saved per condition: relative drift at p0 every loop, per-position mean drift over the last 32 loops, lag-2
    closure at p0 over the last 32 loops, completion mask."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    late = lambda r=0.0: time.monotonic() - Tc > cfg["budget_s"] - r
    skipped = []
    with torch.no_grad():
        n = len(problems)
        ids, e, s = h.batch(problems, [0] * n)
        L = s.shape[1]
        lens = torch.tensor([len(p["ids"]) for p in problems], device=h.dev)
        valid_pos = torch.arange(L, device=h.dev)[None] < lens[:, None]
        p0 = torch.tensor([p["p0"] for p in problems], device=h.dev)
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        sK = s.clone()
        print(f"warm-up {time.monotonic() - Tc:.0f}s", flush=True)
        rel = lambda a, b: ((a - b).norm(dim=-1) / a.norm(dim=-1))
        astar = np.array([p["alpha_star_pred"] for p in problems])
        conds = [(j, r) for j in range(n) for r in cfg["r_mults"]]
        nc, T = len(conds), cfg["T"]
        ci = torch.tensor([c[0] for c in conds], device=h.dev)
        calpha = torch.tensor([c[1] * astar[c[0]] for c in conds], device=h.dev, dtype=torch.float32)
        drift_p0 = np.zeros((nc, T), np.float32); tail_pos = np.zeros((nc, L), np.float32); l2 = np.full(nc, np.nan)
        done = np.zeros(nc, bool)
        for c0 in range(0, nc, cfg["chunk"]):
            if late(10):
                skipped.append(f"conditions from {c0}"); break
            sl = slice(c0, min(nc, c0 + cfg["chunk"]))
            idx = ci[sl]; a = calpha[sl][:, None]; pp = p0[idx]
            Ec, sc = e[idx], sK[idx].clone()
            arc = torch.arange(len(idx), device=h.dev); vp = valid_pos[idx]
            hist, tail_acc, aborted = [], torch.zeros((len(idx), L), device=h.dev), False
            for t in range(T):
                if late(2):
                    aborted = True; skipped.append(f"chunk {c0} aborted at loop {t}"); break
                s2 = h.step(Ec, sc, 0)
                s2[arc, pp] = (1 - a) * sc[arc, pp] + a * s2[arc, pp]
                d = rel(s2, sc)
                drift_p0[sl, t] = d[arc, pp].cpu().numpy()
                if t >= T - 34:
                    hist.append(s2[arc, pp].clone())
                if t >= T - 32:
                    tail_acc += d * vp / 32
                sc = s2
            if aborted:
                break
            H_ = torch.stack(hist)
            num = (H_[2:] - H_[:-2]).norm(dim=-1); den = (H_[2:] - H_[1:-1]).norm(dim=-1).clamp_min(1e-30)
            l2[sl] = (num / den).median(0).values.cpu().numpy()
            tail_pos[sl] = tail_acc.cpu().numpy()
            done[sl] = True
            print(f"conditions {sl.stop}/{nc} {time.monotonic() - Tc:.0f}s", flush=True)
    np.savez_compressed(f"{o}/sweep.npz", uid=np.array([p["uid"] for p in problems]), p0=p0.cpu().numpy(), alpha_star=astar,
                        cond_row=np.array([c[0] for c in conds]), cond_r=np.array([c[1] for c in conds]),
                        cond_alpha=calpha.cpu().numpy(), drift_p0=drift_p0, tail_pos=tail_pos, lag2_p0=l2, done=done)
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped}
    with open(f"{o}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(f"{o}/problems.json", "w") as f:
        f.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=2 * 3600)
def settle(problems_json: str, cfg: dict, run_id: str):
    """Settling-speed experiment (fp32, noise off, census seed-0 init). Per prompt (teacher-forced prompt + answer): K
    loops; at every loop and every answer-prediction position: full-vocab argmax, top-1 log-prob, KL(p_t || p_{t-1})
    (full vocabulary; t >= 2), relative state drift; plus the max-over-positions relative drift. Then a damped fixed point
    x* from s_K (relative residual over all positions < fp_tol) and, at each answer-prediction position p, the own-block
    Jacobian J_pp at x* by Arnoldi (m, one start, CGS2, fp64 basis; single-step central FD at p): saved Hessenberg, leading
    Ritz value (largest modulus) with complex-aware FD residual at eps and eps/2."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    print("validation", val, flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    late = lambda r=0.0: time.monotonic() - Tc > cfg["budget_s"] - r
    R_, skipped = {}, []

    def save():
        np.savez_compressed(f"{o}/settle.npz", allow_pickle=True, **R_)
        meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped}
        with open(f"{o}/meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        with open(f"{o}/problems.json", "w") as f:
            f.write(problems_json)
        results_vol.commit()
        return meta

    K = cfg["K"]
    with torch.no_grad():
        n = len(problems)
        ids, e, s = h.batch(problems, [0] * n)
        L = s.shape[1]
        lens = torch.tensor([len(p["ids"]) for p in problems], device=h.dev)
        valid_pos = torch.arange(L, device=h.dev)[None] < lens[:, None]
        P = max(len(p["pred_pos"]) for p in problems)
        gidx = torch.tensor([p["pred_pos"] + [p["pred_pos"][-1]] * (P - len(p["pred_pos"])) for p in problems], device=h.dev)
        am = np.zeros((n, K, P), np.int64); top1 = np.zeros((n, K, P), np.float32); kl = np.full((n, K, P), np.nan, np.float32)
        dpos = np.zeros((n, K, P), np.float32); dmax = np.zeros((n, K), np.float32)
        prev = None
        k_done = 0
        for k in range(K):
            if late(cfg["reserve_arnoldi_s"] + 60):
                skipped.append(f"collection stopped at loop {k}"); break
            s2 = h.step(e, s, k)
            d = (s2 - s).norm(dim=-1) / s2.norm(dim=-1)
            dpos[:, k] = torch.gather(d, 1, gidx).cpu().numpy(); dmax[:, k] = (d * valid_pos).max(-1).values.cpu().numpy()
            k_done = k + 1
            s = s2
            lg = torch.log_softmax(h.coda_logits(s, gidx), -1)  # (n, P, V)
            am[:, k] = lg.argmax(-1).cpu().numpy(); top1[:, k] = lg.max(-1).values.cpu().numpy()
            if prev is not None:
                kl[:, k] = (lg.exp() * (lg - prev)).sum(-1).cpu().numpy()
            prev = lg
        del prev, lg
        R_.update(k_done=k_done, uid=np.array([p["uid"] for p in problems]), n_pos=np.array([len(p["pred_pos"]) for p in problems]),
                  argmax=am, top1_lp=top1, kl=kl, drift_pos=dpos, drift_max=dmax,
                  truth=np.array([[p["ids"][q + 1] for q in p["pred_pos"]] + [-1] * (P - len(p["pred_pos"])) for p in problems]))
        save()
        print(f"collection {time.monotonic() - Tc:.0f}s", flush=True)
        # damped fixed point from s_K
        x = s.clone()
        done = torch.zeros(n, dtype=torch.bool, device=h.dev)
        rel = lambda a, b: ((a - b).norm(dim=-1) / a.norm(dim=-1))
        for it in range(cfg["n_damp"]):
            act = (~done).nonzero().flatten()
            if len(act) == 0 or late(cfg["reserve_arnoldi_s"]):
                break
            fx = h.step(e[act], x[act], 0)
            res = (rel(fx, x[act]) * valid_pos[act]).max(-1).values
            x[act] = x[act] + cfg["beta"] * (fx - x[act])
            done[act] = res < cfg["fp_tol"]
        fx = h.step(e, x, 0)
        fres = (rel(fx, x) * valid_pos).max(-1).values
        R_.update(fp_resid=fres.cpu().numpy(), fp_ok=(fres < cfg["fp_tol"]).cpu().numpy())
        save()
        print(f"fixed points {time.monotonic() - Tc:.0f}s; valid {int((fres < cfg['fp_tol']).sum())}/{n}", flush=True)
        # Arnoldi on J_pp at every answer-prediction position
        rows = [(i, j, q) for i, p in enumerate(problems) for j, q in enumerate(p["pred_pos"])]
        nr, m = len(rows), cfg["m"]
        ri = torch.tensor([r[0] for r in rows], device=h.dev); rp = torch.tensor([r[2] for r in rows], device=h.dev)
        ar2 = torch.arange(2 * nr, device=h.dev); P2 = torch.cat([rp, rp]); I2 = torch.cat([ri, ri])

        def jv(v, ep):
            st = x[I2].clone()
            st[ar2, P2] += torch.cat([ep * v, -ep * v]).to(st.dtype)
            outs = []
            for c0 in range(0, 2 * nr, cfg["chunk"]):
                c1 = min(2 * nr, c0 + cfg["chunk"])
                outs.append(h.step(e[I2[c0:c1]], st[c0:c1], 0)[torch.arange(c1 - c0, device=h.dev), P2[c0:c1]])
            d_ = torch.cat(outs)
            return (d_[:nr] - d_[nr:]) / (2 * ep)

        V = torch.zeros((nr, m + 1, h.d), device=h.dev, dtype=torch.float64)
        Hm = torch.zeros((nr, m + 1, m), device=h.dev, dtype=torch.float64)
        v0 = torch.randn((nr, h.d), device=h.dev, dtype=torch.float64, generator=torch.Generator(device=h.dev).manual_seed(4))
        V[:, 0] = v0 / v0.norm(dim=-1, keepdim=True)
        dn = 0
        for jj in range(m):
            if late(20):
                skipped.append(f"arnoldi stopped at {jj}"); break
            w = jv(V[:, jj].float(), cfg["eps"]).double()
            for _ in range(2):
                c = torch.einsum("nkd,nd->nk", V[:, : jj + 1], w)
                Hm[:, : jj + 1, jj] += c
                w = w - torch.einsum("nk,nkd->nd", c, V[:, : jj + 1])
            hn = w.norm(dim=-1)
            Hm[:, jj + 1, jj] = hn
            V[:, jj + 1] = w / hn.clamp_min(1e-10)[:, None]
            dn = jj + 1
        Hc = Hm[:, : dn + 1, :dn].cpu().numpy()
        R_.update(arn_rows=np.array(rows), arn_H=Hc, arn_dim=dn)
        save()
        if dn >= 4 and not late(10):
            lead = np.zeros(nr, dtype=complex)
            A_ = torch.zeros((nr, h.d), device=h.dev, dtype=torch.float64); B_ = torch.zeros_like(A_)
            for b in range(nr):
                lam, Y = np.linalg.eig(Hc[b, :dn, :dn])
                k0 = int(np.argmax(np.abs(lam) - 1e-9 * (lam.imag < 0)))
                if lam[k0].imag < 0:
                    k0 = int(np.argmin(np.abs(lam - np.conj(lam[k0]))))
                lead[b] = lam[k0]
                a_ = V[b, :dn].T @ torch.tensor(Y[:, k0].real.copy(), device=h.dev)
                b_ = V[b, :dn].T @ torch.tensor(Y[:, k0].imag.copy(), device=h.dev)
                z_ = torch.sqrt(a_.norm() ** 2 + b_.norm() ** 2); A_[b], B_[b] = a_ / z_, b_ / z_
            al = torch.tensor(lead.real, device=h.dev); be = torch.tensor(lead.imag, device=h.dev)
            fdr = np.full((nr, 2), np.nan)
            for ei, ep in enumerate((cfg["eps"], cfg["eps"] / 2)):
                Ja, Jb = jv(A_.float(), ep).double(), jv(B_.float(), ep).double()
                ra = Ja - al[:, None] * A_ + be[:, None] * B_; rb_ = Jb - be[:, None] * A_ - al[:, None] * B_
                fdr[:, ei] = torch.sqrt(ra.norm(dim=-1) ** 2 + rb_.norm(dim=-1) ** 2).cpu().numpy()
            R_.update(lead=lead, lead_fd_resid=fdr)
        meta = save()
        print(f"done {time.monotonic() - Tc:.0f}s", flush=True)
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=660)
def mcq(problems_json: str, cfg: dict, run_id: str):
    """Budget robustness on zero-shot multiple-choice prompts (fp32, noise off, seed-0 init per uid as in the census).
    Loop map with optional GLOBAL relaxation: s <- (1 - alpha) s + alpha F(s) at every position (alpha = 1 is the
    unmodified model; relaxation leaves every fixed point unchanged). K loops. Readout at the answer position (last prompt
    token) at loops in the schedule (every loop for 1..dense, every `sparse` loops after, every loop in the last `tail`):
    log-probs of " A".." D", full-vocab argmax, letter mass. Relative drift at every position every loop; lag-2 closure
    ||s_t - s_{t-2}|| / ||s_t - s_{t-1}|| at every position for the last 32 loops."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    Tc = time.monotonic()
    problems = json.loads(problems_json)
    for p in problems:
        p["ids"] = h.tok.encode(p["prompt"], add_special_tokens=True)
        p["pred_pos"] = [len(p["ids"]) - 1]
    val = h.validate(problems)
    print("validation", val, flush=True)
    LET = [h.tok.encode(c, add_special_tokens=False) for c in cfg.get("choices", [" A", " B", " C", " D"])]
    assert all(len(t) == 1 for t in LET)
    LET = torch.tensor([t[0] for t in LET], device=h.dev)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    K, alpha = cfg["K"], float(cfg["alpha"])
    sched = sorted(set(list(range(1, cfg["dense"] + 1)) + list(range(cfg["dense"], K + 1, cfg["sparse"])) + list(range(K - cfg["tail"] + 1, K + 1))))
    n = len(problems)
    Lmax = max(len(p["ids"]) for p in problems)
    ro_lp = np.zeros((n, len(sched), len(LET)), np.float32); ro_am = np.zeros((n, len(sched)), np.int64); ro_mass = np.zeros((n, len(sched)), np.float32)
    drift = np.zeros((n, K, Lmax), np.float32); clos = np.full((n, 32, Lmax), np.nan, np.float32)
    done = np.zeros(n, bool); skipped = []
    with torch.no_grad():
        for c0 in range(0, n, cfg["chunk"]):
            if time.monotonic() - T0 > cfg["total_s"] - 60:  # end-to-end deadline (includes model load), save reserve
                skipped.append(f"chunk {c0}"); continue
            tc = time.monotonic()
            chunk = problems[c0 : c0 + cfg["chunk"]]
            N = len(chunk)
            ids, e, s = h.batch(chunk, [0] * N)
            L = s.shape[1]
            q = torch.tensor([p["pred_pos"][0] for p in chunk], device=h.dev)
            prev2 = None
            aborted = False
            for k in range(K):
                if time.monotonic() - T0 > cfg["total_s"] - 30:
                    aborted = True; skipped.append(f"chunk {c0} aborted at loop {k}"); break
                f = h.step(e, s, k)
                s2 = f if alpha == 1.0 else (1 - alpha) * s + alpha * f
                dd = s2 - s
                drift[c0 : c0 + N, k, :L] = (dd.norm(dim=-1) / s2.norm(dim=-1)).cpu().numpy()
                if k >= K - 32 and prev2 is not None:
                    clos[c0 : c0 + N, k - (K - 32), :L] = ((s2 - prev2).norm(dim=-1) / dd.norm(dim=-1).clamp_min(1e-30)).cpu().numpy()
                prev2 = s
                s = s2
                if k + 1 in sched:
                    i = sched.index(k + 1)
                    lg = torch.log_softmax(h.coda_logits(s, q[:, None])[:, 0], -1)
                    ro_lp[c0 : c0 + N, i] = lg[:, LET].cpu().numpy(); ro_am[c0 : c0 + N, i] = lg.argmax(-1).cpu().numpy()
                    ro_mass[c0 : c0 + N, i] = lg[:, LET].exp().sum(-1).cpu().numpy()
            if aborted:
                break
            done[c0 : c0 + N] = True
            print(f"chunk {c0 + N}/{n}: {time.monotonic() - tc:.0f}s, total {time.monotonic() - Tc:.0f}s", flush=True)
    np.savez_compressed(f"{o}/mcq.npz", uid=np.array([p["uid"] for p in problems]), sched=np.array(sched), ro_lp=ro_lp, ro_am=ro_am,
                        ro_mass=ro_mass, drift=drift, closure=clos, done=done, letter_tokens=LET.cpu().numpy(),
                        lens=np.array([len(p["ids"]) for p in problems]), gold=np.array([p["gold"] for p in problems]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped,
            "n_done": int(done.sum())}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/problems.json", "w") as f_:
        f_.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=660)
def mblend(spec_json: str, cfg: dict, run_id: str):
    """Month shape + context-blend experiment (fp32, noise off). Rows = the 24 pure prompts followed by the blend items
    (E_a = (1 - a) E(m0) + a E(m1), whole prelude-output tensor; prompts of a template have identical length). Every row of
    a template starts from ONE common init state (init_state of a zero row of the template's length, uid 95000 + template
    index, seed 0), so along a blend path only the context changes. K loops. Saved: the state at the answer position q
    (last token) at checkpoint loops; the 12 month-token log-probs, full-vocab argmax and month mass at q every loop;
    max-over-positions mean drift over the last 8 loops. End-to-end deadline total_s (includes model load)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    spec = json.loads(spec_json)
    pure, items, MN = spec["pure"], spec["items"], spec["months"]
    for p in pure:
        p["ids"] = h.tok.encode(p["prompt"], add_special_tokens=True)
        p["pred_pos"] = [len(p["ids"]) - 1]
    val = h.validate(pure)
    print("validation", val, flush=True)
    MT = [h.tok.encode(" " + m, add_special_tokens=False) for m in MN]
    assert all(len(t) == 1 for t in MT)
    MT = torch.tensor([t[0] for t in MT], device=h.dev)
    tmpl = sorted({p["template"] for p in pure})
    by = {(p["template"], p["month"]): p for p in pure}
    for t in tmpl:
        assert len({len(p["ids"]) for p in pure if p["template"] == t}) == 1
        ref = by[(t, 0)]["ids"]
        for m in range(1, 12):  # token-wise alignment: prompts differ only at the query-month position
            diff = [i for i, (x, y) in enumerate(zip(ref, by[(t, m)]["ids"])) if x != y]
            assert len(diff) == 1 and diff[0] == len(ref) - 5, (t, m, diff)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    K, CK = cfg["K"], sorted(cfg["checkpoints"])
    with torch.no_grad():
        E, S0 = {}, {}
        for ti, t in enumerate(tmpl):
            ps = [by[(t, m)] for m in range(12)]
            ids = torch.tensor([p["ids"] for p in ps], device=h.dev)
            e, _ = h.model.embed_inputs(ids)
            E[t] = e  # (12, L, d)
            S0[t] = h.init_state(torch.zeros((ids.shape[1], h.d), device=h.dev, dtype=e.dtype), 95000 + ti, 0)
        rows = [dict(kind="pure", template=p["template"], m0=p["month"], m1=p["month"], a=0.0) for p in pure] + items
        n = len(rows)
        states = np.zeros((n, len(CK), h.d), np.float32)
        ro_lp = np.zeros((n, K, 12), np.float32); ro_am = np.zeros((n, K), np.int64); ro_mass = np.zeros((n, K), np.float32)
        late8 = np.full(n, np.nan, np.float32); done = np.zeros(n, bool); skipped = []
        for t in tmpl:
            idx = [i for i, r in enumerate(rows) if r["template"] == t]
            for c0 in range(0, len(idx), cfg["chunk"]):
                if time.monotonic() - T0 > cfg["total_s"] - 45:
                    skipped.append(f"{t} chunk {c0}"); continue
                sel = idx[c0 : c0 + cfg["chunk"]]
                a = torch.tensor([rows[i]["a"] for i in sel], device=h.dev, dtype=E[t].dtype)[:, None, None]
                e = (1 - a) * E[t][[rows[i]["m0"] for i in sel]] + a * E[t][[rows[i]["m1"] for i in sel]]
                s = S0[t][None].expand(len(sel), -1, -1).clone()
                q = e.shape[1] - 1
                acc = torch.zeros(len(sel), e.shape[1], device=h.dev)
                aborted = False
                for k in range(K):
                    if time.monotonic() - T0 > cfg["total_s"] - 20:
                        aborted = True; skipped.append(f"{t} chunk {c0} aborted at {k}"); break
                    s2 = h.step(e, s, k)
                    if k >= K - 8:
                        acc += ((s2 - s).norm(dim=-1) / s2.norm(dim=-1)) / 8
                    s = s2
                    lg = torch.log_softmax(h.coda_logits(s, torch.full((len(sel), 1), q, device=h.dev))[:, 0], -1)
                    ro_lp[sel, k] = lg[:, MT].cpu().numpy(); ro_am[sel, k] = lg.argmax(-1).cpu().numpy()
                    ro_mass[sel, k] = lg[:, MT].exp().sum(-1).cpu().numpy()
                    if k + 1 in CK:
                        states[sel, CK.index(k + 1)] = s[:, q].cpu().numpy()
                if aborted:
                    break
                late8[sel] = acc.max(-1).values.cpu().numpy()
                done[sel] = True
                print(f"{t} {c0 + len(sel)}/{len(idx)} total {time.monotonic() - T0:.0f}s", flush=True)
    np.savez_compressed(f"{o}/mblend.npz", states=states, checkpoints=np.array(CK), ro_lp=ro_lp, ro_am=ro_am, ro_mass=ro_mass,
                        late8=late8, done=done, month_tokens=MT.cpu().numpy())
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped,
            "n_done": int(done.sum()), "n_rows": n}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/rows.json", "w") as f_:
        json.dump(dict(rows=rows, pure=[{k: v for k, v in p.items() if k not in ("ids", "pred_pos")} for p in pure], months=MN), f_)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=540)
def steer(problems_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """One-shot state push vs context (E) edit at the digit answer position (fp32, noise off, census seed-0 init, same
    truncation as dgcollect). Per prompt (answer U at q): K0 warm-up loops -> s0. Direction d_s = C[U+1] - C[U] (adjusted
    one-notch chord in state space, from make_steer_dirs.py; fit on held-out cells) or a norm-matched Gaussian direction
    (seeded per prompt), times scale. ONE-SHOT: s[q] += d_s once, then T loops with the original E. CONTEXT: e[q] += d_E for
    every subsequent loop, with W_e d_E = W_s d_s (adapter = Linear on cat([state, E]); least squares, residual saved), so
    the context edit equals adding d_s to the state INPUT of every loop. SHAM: no change. Readout (bare digits at q) at
    t = 0 (before any further loop) and after t in cfg["ro_t"] loops; state displacement at q vs sham at the end."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    problems = json.loads(problems_json)
    h.prep(problems)
    val = h.validate(problems)
    C = torch.tensor(np.load(io.BytesIO(dirs_npz))["C"], device=h.dev, dtype=torch.float64)
    W = h.model.transformer.adapter.weight.double()
    d = h.d
    assert W.shape == (d, 2 * d)
    Ws, We = W[:, :d], W[:, d:]
    dig = torch.tensor(h.DIG, device=h.dev)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    n = len(problems)
    conds = [("sham", "none", 0.0)] + [(m, dk, sc) for sc in cfg["scales"] for m in ("oneshot", "context") for dk in ("chord", "random")]
    RO = [0] + sorted(cfg["ro_t"]); T = RO[-1]
    lp = np.zeros((n, len(conds), len(RO), 10), np.float32); am = np.zeros((n, len(conds), len(RO)), np.int64)
    disp = np.zeros((n, len(conds)), np.float32); along = np.zeros((n, len(conds)), np.float32)
    dnorm = np.zeros((n, len(conds)), np.float32); eres = np.full((n, len(conds)), np.nan, np.float32)
    with torch.no_grad():
        ids, e, s = h.batch(problems, [0] * n)
        Lq = max(p["pred_pos"][-1] + 1 for p in problems)
        e, s = e[:, :Lq], s[:, :Lq]
        q = torch.tensor([p["pred_pos"][-1] for p in problems], device=h.dev)
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        s0 = s.clone()
        print(f"warm-up {time.monotonic() - T0:.0f}s", flush=True)
        rows = [(i, c) for i in range(n) for c in range(len(conds))]
        dS = torch.zeros((len(rows), d), device=h.dev, dtype=torch.float64); dE = torch.zeros_like(dS)
        for r, (i, c) in enumerate(rows):
            m, dk, sc = conds[c]
            if m == "sham":
                continue
            U = problems[i]["ans"]
            ch = C[(U + 1) % 10] - C[U]
            if dk == "random":
                g = torch.Generator(device=h.dev).manual_seed(1000 + i)
                v = torch.randn(d, device=h.dev, dtype=torch.float64, generator=g)
                ch = v / v.norm() * ch.norm()
            dv = sc * ch
            dnorm[i, c] = float(dv.norm())
            if m == "oneshot":
                dS[r] = dv
            else:
                dE[r] = dv  # replaced by the E-space solution below
        ctx_rows = [r for r, (i, c) in enumerate(rows) if conds[c][0] == "context"]
        if ctx_rows:  # one least-squares solve with multiple right-hand sides
            Dn = torch.stack([dE[r] for r in ctx_rows], 1)  # d_s for context rows
            Rhs = Ws @ Dn
            Sol = torch.linalg.lstsq(We, Rhs).solution
            res_ = (We @ Sol - Rhs).norm(dim=0) / Rhs.norm(dim=0)
            res32 = (We.float() @ Sol.float() - Rhs.float()).norm(dim=0) / Rhs.float().norm(dim=0)
            for jj, r in enumerate(ctx_rows):
                i_, c_ = rows[r]
                dE[r] = Sol[:, jj]; eres[i_, c_] = float(max(res_[jj], res32[jj]))
        ri = torch.tensor([r[0] for r in rows], device=h.dev)
        FIN = torch.zeros((len(rows), d), dtype=torch.float64)
        DR = torch.zeros(len(rows), dtype=torch.float64); L2 = torch.zeros(len(rows), dtype=torch.float64)
        for c0 in range(0, len(rows), cfg["chunk"]):
            if time.monotonic() - T0 > cfg["total_s"] - 40:
                raise RuntimeError("deadline")  # partial results are not saved by design (small run)
            sl = slice(c0, min(len(rows), c0 + cfg["chunk"]))
            idx = ri[sl]; N = len(idx); ar = torch.arange(N, device=h.dev); qq = q[idx]
            ss = s0[idx].clone(); ee = e[idx].clone()
            ss[ar, qq] += dS[sl].to(ss.dtype)
            hist = []
            ee[ar, qq] += dE[sl].to(ee.dtype)

            def ro(ti, st):
                lg = torch.log_softmax(h.coda_logits(st, qq[:, None])[:, 0], -1)
                for j, r in enumerate(range(sl.start, sl.stop)):
                    i, c = rows[r]
                    lp[i, c, ti] = lg[j, dig].cpu().numpy(); am[i, c, ti] = int(lg[j].argmax())
            ro(0, ss)
            for t in range(1, T + 1):
                if time.monotonic() - T0 > cfg["total_s"] - 25:
                    raise RuntimeError("deadline")
                ss = h.step(ee, ss, cfg["K0"] + t - 1)
                if t >= T - 2:
                    hist.append(ss[ar, qq].double())
                if t in RO:
                    ro(RO.index(t), ss)
            FIN[sl] = ss[ar, qq].double().cpu()
            h0, h1, h2 = hist[-3], hist[-2], hist[-1]
            DR[sl] = ((h2 - h1).norm(dim=-1) / h2.norm(dim=-1)).cpu()
            L2[sl] = ((h2 - h0).norm(dim=-1) / (h2 - h1).norm(dim=-1).clamp_min(1e-30)).cpu()
            print(f"conditions {sl.stop}/{len(rows)} {time.monotonic() - T0:.0f}s", flush=True)
        for r, (i, c) in enumerate(rows):
            sham = FIN[i * len(conds)]
            dx = FIN[r] - sham
            disp[i, c] = float(dx.norm())
            U = problems[i]["ans"]
            ch = (C[(U + 1) % 10] - C[U]).cpu()
            along[i, c] = float(dx @ ch / ch.norm())
    late_drift = np.zeros((n, len(conds)), np.float32); lag2 = np.zeros((n, len(conds)), np.float32)
    for r, (i, c) in enumerate(rows):
        late_drift[i, c] = float(DR[r]); lag2[i, c] = float(L2[r])
    np.savez_compressed(f"{o}/steer.npz", late_drift=late_drift, lag2_closure=lag2, lp=lp, am=am, disp=disp, along=along, dnorm=dnorm, e_resid=eres, ro_t=np.array(RO),
                        cond=np.array([f"{m}|{dk}|{sc}" for m, dk, sc in conds]), uid=np.array([p["uid"] for p in problems]),
                        ans=np.array([p["ans"] for p in problems]), dig_tokens=np.array(h.DIG))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/problems.json", "w") as f_:
        f_.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=540)
def truth(problems_json: str, cfg: dict, run_id: str):
    """Truth experiment stage 1 (fp32, noise off). Each statement in two formats: A (bare, unlabeled prefix; state at the
    statement-final '.') and B (few-shot True/False judgment; states at the statement-final '.' and at the answer position
    = last token; readout every loop of log p(' True'), log p(' False'), full-vocab argmax, and their mass). Seed-0 init
    per sequence (uid for A, uid + 500000 for B). K loops; states saved at checkpoint loops; late-8 mean drift at the saved
    positions. End-to-end deadline total_s."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    rows = json.loads(problems_json)
    dot = h.tok.encode(".", add_special_tokens=False)
    assert len(dot) == 1
    dot = dot[0]
    TF = [h.tok.encode(w, add_special_tokens=False) for w in (" True", " False")]
    assert all(len(t) == 1 for t in TF)
    TF = torch.tensor([t[0] for t in TF], device=h.dev)
    seqs = []
    for i, r in enumerate(rows):
        for fmt in ("A", "B"):
            ids = h.tok.encode(r[f"prompt_{fmt}"], add_special_tokens=True)
            dpos = max(j for j, t in enumerate(ids) if t == dot)
            if fmt == "A":
                assert dpos == len(ids) - 1
            else:  # the selected '.' must be the statement's terminal period, followed exactly by "\nTrue or false?"
                assert h.tok.decode(ids[dpos + 1:]) == "\nTrue or false?", h.tok.decode(ids[dpos:])
                assert h.tok.decode(ids[:dpos + 1]).endswith(r["statement"]), r["statement"]
                for w, tid in zip((" True", " False"), TF.tolist()):
                    assert h.tok.encode(r["prompt_B"] + w, add_special_tokens=True) == ids + [tid]
            seqs.append(dict(i=i, fmt=fmt, uid=r["uid"] + (0 if fmt == "A" else 500000), ids=ids, pred_pos=[len(ids) - 1], dot=dpos))
    val = h.validate([s for s in seqs if s["fmt"] == "B"][:40])
    print("validation", val, flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    K, CK = cfg["K"], sorted(cfg["checkpoints"])
    n = len(rows)
    stA = np.zeros((n, len(CK), h.d), np.float32); stBd = np.zeros_like(stA); stBa = np.zeros_like(stA)
    lpB = np.zeros((n, K, 2), np.float32); amB = np.zeros((n, K), np.int64)
    late = np.full((n, 3), np.nan, np.float32); done = np.zeros((n, 2), bool); skipped = []
    with torch.no_grad():
        for fmt in ("A", "B"):
            sub = [s for s in seqs if s["fmt"] == fmt]
            for c0 in range(0, len(sub), cfg["chunk"]):
                if time.monotonic() - T0 > cfg["total_s"] - 40:
                    skipped.append(f"{fmt} chunk {c0}"); continue
                ch = sub[c0 : c0 + cfg["chunk"]]
                N = len(ch)
                ids, e, s = h.batch(ch, [0] * N)
                ar = torch.arange(N, device=h.dev)
                dp = torch.tensor([x["dot"] for x in ch], device=h.dev); qp = torch.tensor([x["pred_pos"][0] for x in ch], device=h.dev)
                accd = torch.zeros(N, device=h.dev); acca = torch.zeros(N, device=h.dev)
                ii = [x["i"] for x in ch]
                aborted = False
                for k in range(K):
                    if time.monotonic() - T0 > cfg["total_s"] - 20:
                        aborted = True; skipped.append(f"{fmt} chunk {c0} aborted at {k}"); break
                    s2 = h.step(e, s, k)
                    if k >= K - 8:
                        dd = (s2 - s).norm(dim=-1) / s2.norm(dim=-1)
                        accd += dd[ar, dp] / 8; acca += dd[ar, qp] / 8
                    s = s2
                    if fmt == "B":
                        lg = torch.log_softmax(h.coda_logits(s, qp[:, None])[:, 0], -1)
                        lpB[ii, k] = lg[:, TF].cpu().numpy(); amB[ii, k] = lg.argmax(-1).cpu().numpy()
                    if k + 1 in CK:
                        ci = CK.index(k + 1)
                        if fmt == "A":
                            stA[ii, ci] = s[ar, dp].cpu().numpy()
                        else:
                            stBd[ii, ci] = s[ar, dp].cpu().numpy(); stBa[ii, ci] = s[ar, qp].cpu().numpy()
                if aborted:
                    break
                if fmt == "A":
                    late[ii, 0] = accd.cpu().numpy()
                else:
                    late[ii, 1] = accd.cpu().numpy(); late[ii, 2] = acca.cpu().numpy()
                done[ii, 0 if fmt == "A" else 1] = True
                print(f"{fmt} {c0 + N}/{len(sub)} total {time.monotonic() - T0:.0f}s", flush=True)
    np.savez_compressed(f"{o}/truth.npz", stA=stA, stBd=stBd, stBa=stBa, lpB=lpB, amB=amB, late=late, done=done,
                        checkpoints=np.array(CK), tf_tokens=TF.cpu().numpy())
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "skipped": skipped,
            "n_done": done.sum(0).tolist()}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/problems.json", "w") as f_:
        f_.write(problems_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=510)
def truthsteer(items_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """Truth stage 2: steering at the statement-final '.' (B_dot) of the judgment format (fp32, noise off, seed-0 init
    uid + 500000 as in stage 1). Warm-up K0 loops; then per condition from identical cloned states:
    SHAM; ONESHOT (state[B_dot] += delta once, then Tr + Trec loops with the original context); CONTEXT (e[B_dot] += dE for
    Tr loops with W_e dE = W_s delta, then REMOVED for Trec loops without resetting the state). delta = sign * dose * scale_i
    * u_i with u_i the item's fold-specific B_dot truth direction (unit), scale_i the training class-mean distance; sign +1
    = toward TRUE, -1 = toward FALSE; RANDOM = norm-matched Gaussian direction (seeded per item and mode).
    Readout: judgment logit difference log p(' True') - log p(' False') at the answer position at offsets cfg["ro"]
    (0 = right after the one-shot push, before any further loop)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    Z = np.load(io.BytesIO(dirs_npz))
    U = torch.tensor(Z["dirB"], device=h.dev, dtype=torch.float64); SC = torch.tensor(Z["scale"], device=h.dev, dtype=torch.float64)
    dot = h.tok.encode(".", add_special_tokens=False)[0]
    TF = torch.tensor([h.tok.encode(w, add_special_tokens=False)[0] for w in (" True", " False")], device=h.dev)
    seqs = []
    for k, r in enumerate(items):
        ids = h.tok.encode(r["prompt_B"], add_special_tokens=True)
        dpos = max(j for j, t in enumerate(ids) if t == dot)
        assert h.tok.decode(ids[dpos + 1:]) == "\nTrue or false?"
        seqs.append(dict(uid=r["uid"] + 500000, ids=ids, pred_pos=[len(ids) - 1], dot=dpos))
    val = h.validate(seqs[:20])
    W = h.model.transformer.adapter.weight.double(); d = h.d
    Ws, We = W[:, :d], W[:, d:]
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    conds = cfg["conds"]  # list of [name, mode, dirkind, sign, dose]
    RO = sorted(cfg["ro"]); Tr, Trec = cfg["Tr"], cfg["Trec"]; T = Tr + Trec
    n, C = len(items), len(conds)
    ld = np.full((n, C, len(RO)), np.nan, np.float32); eres = np.full((n, C), np.nan, np.float32)
    with torch.no_grad():
        ids, e, s = h.batch(seqs, [0] * n)
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        s0 = s.clone()
        print(f"warm-up {time.monotonic() - T0:.0f}s", flush=True)
        dp = torch.tensor([x["dot"] for x in seqs], device=h.dev); qp = torch.tensor([x["pred_pos"][0] for x in seqs], device=h.dev)
        if cfg.get("site", "dot") == "ans":  # intervene at the answer position instead of the statement-final '.'
            dp = qp.clone()
        rows = [(i, c) for i in range(n) for c in range(C)]
        dS = torch.zeros((len(rows), d), device=h.dev, dtype=torch.float64); ctx = torch.zeros(len(rows), dtype=torch.bool)
        for r, (i, c) in enumerate(rows):
            name, mode, dk, sign, dose = conds[c]
            if mode == "sham":
                continue
            if dk == "B":
                v = U[i]
            else:
                g = torch.Generator(device=h.dev).manual_seed(5000 + 37 * i + (1 if mode == "context" else 0))
                v = torch.randn(d, device=h.dev, dtype=torch.float64, generator=g); v = v / v.norm()
            dS[r] = sign * dose * SC[i] * v
            ctx[r] = mode == "context"
        cr = [r for r in range(len(rows)) if ctx[r]]
        dE = torch.zeros_like(dS)
        if cr:
            Rhs = Ws @ dS[cr].T
            Sol = torch.linalg.lstsq(We, Rhs).solution
            res_ = (We @ Sol - Rhs).norm(dim=0) / Rhs.norm(dim=0)
            for jj, r in enumerate(cr):
                dE[r] = Sol[:, jj]; eres[rows[r][0], rows[r][1]] = float(res_[jj])
        ri = torch.tensor([r[0] for r in rows], device=h.dev)
        for c0 in range(0, len(rows), cfg["chunk"]):
            if time.monotonic() - T0 > cfg["total_s"] - 25:
                raise RuntimeError("deadline")
            sl = slice(c0, min(len(rows), c0 + cfg["chunk"]))
            idx = ri[sl]; N = len(idx); ar = torch.arange(N, device=h.dev); dpp = dp[idx]; qq = qp[idx]
            ss = s0[idx].clone(); e_on = e[idx].clone(); e_off = e[idx]
            oneshot = ~ctx[sl].to(h.dev)
            ss[ar[oneshot], dpp[oneshot]] += dS[sl][oneshot].to(ss.dtype)
            e_on[ar, dpp] += dE[sl].to(e_on.dtype)

            def ro(t, st):
                lg = torch.log_softmax(h.coda_logits(st, qq[:, None])[:, 0], -1)
                v_ = (lg[:, TF[0]] - lg[:, TF[1]]).cpu().numpy()
                for j, r in enumerate(range(sl.start, sl.stop)):
                    i, c = rows[r]
                    ld[i, c, RO.index(t)] = v_[j]
            ro(0, ss)
            for t in range(1, T + 1):
                if time.monotonic() - T0 > cfg["total_s"] - 15:
                    raise RuntimeError("deadline")
                ss = h.step(e_on if t <= Tr else e_off, ss, cfg["K0"] + t - 1)
                if t in RO:
                    ro(t, ss)
            print(f"conditions {sl.stop}/{len(rows)} {time.monotonic() - T0:.0f}s", flush=True)
    np.savez_compressed(f"{o}/truthsteer.npz", ld=ld, ro=np.array(RO), e_resid=eres, truth=np.array([it["truth"] for it in items]),
                        dataset=np.array([it["dataset"] for it in items]), unit=np.array([it["unit"] for it in items]),
                        conds=np.array([json.dumps(c) for c in conds]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=420)
def respond(items_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """Linear-response (resolvent) test of steerability at the answer position q (the last token) (fp32, noise off).
    task 'digits': steer_prompts (dg records), seed-0 init + truncation as dgcollect, K0 = 96, d = C[U+1] - C[U],
      readout R = log p(U+1) - log p(U) (bare digits).
    task 'truth': truthsteer items (judgment format), init uid + 500000, K0 = 64, d = sign_opp * scale * u (answer-site
      fold-specific direction, toward the opposite label), R = sign_opp * (log p(' True') - log p(' False')).
    At x* = state after K0 loops: g = dR/dx_q by autograd through the coda; S0 = g . d (immediate coupling);
    J v = own-block central FD at q (eps on the unit vector, scaled by |v|; exact own block since q is last);
    delta_lin = sum_{k=1..N} J^k d = (I - J)^-1 J d (Neumann series; tail ratio saved); S1 = g . J d; S_inf = g . delta_lin.
    Observed forced responses: s_{t+1} = F(s_t + dose * D) (D = d at q; equals the adapter-matched context edit) for T loops
    from x*, R(s_T) - R(sham_T) for doses in cfg["doses"]; state displacement at q. One-shot immediate: R(x* + d) - R(x*)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    Z = np.load(io.BytesIO(dirs_npz))
    task = cfg["task"]
    if task == "digits":
        h.prep(items)
        val = h.validate(items[:20])
        ids, e, s = h.batch(items, [0] * len(items))
        Lq = max(p["pred_pos"][-1] + 1 for p in items)
        e, s = e[:, :Lq], s[:, :Lq]
        q = torch.tensor([p["pred_pos"][-1] for p in items], device=h.dev)
        assert bool((q == Lq - 1).all())
        C = torch.tensor(Z["C"], device=h.dev, dtype=torch.float64)
        U = [p["ans"] for p in items]
        D = torch.stack([C[(u + 1) % 10] - C[u] for u in U])
        tokA = torch.tensor([h.DIG[(u + 1) % 10] for u in U], device=h.dev); tokB = torch.tensor([h.DIG[u] for u in U], device=h.dev)
        sgn = torch.ones(len(items), device=h.dev)
    else:
        seqs = []
        for r in items:
            ids_ = h.tok.encode(r["prompt_B"], add_special_tokens=True)
            seqs.append(dict(uid=r["uid"] + 500000, ids=ids_, pred_pos=[len(ids_) - 1]))
        val = h.validate(seqs[:20])
        ids, e, s = h.batch(seqs, [0] * len(seqs))
        lens = torch.tensor([len(x["ids"]) for x in seqs], device=h.dev)
        q = lens - 1
        TF = [h.tok.encode(w, add_special_tokens=False)[0] for w in (" True", " False")]
        tokA = torch.full((len(items),), TF[0], device=h.dev); tokB = torch.full((len(items),), TF[1], device=h.dev)
        sgn = torch.tensor([-1.0 if r["truth"] == 1 else 1.0 for r in items], device=h.dev)
        U_ = torch.tensor(Z["dirB"], device=h.dev, dtype=torch.float64); SC = torch.tensor(Z["scale"], device=h.dev, dtype=torch.float64)
        D = sgn.double()[:, None] * SC[:, None] * U_
    n = len(items)
    ar = torch.arange(n, device=h.dev)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)

    def R(st):
        lg = torch.log_softmax(h.coda_logits(st, q[:, None])[:, 0], -1)
        return sgn * (lg[ar, tokA] - lg[ar, tokB])

    with torch.no_grad():
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        x = s.clone()
        fx = h.step(e, x, cfg["K0"])
        drift = ((fx - x)[ar, q].norm(dim=-1) / x[ar, q].norm(dim=-1)).cpu().numpy()
    print(f"warm-up {time.monotonic() - T0:.0f}s", flush=True)
    # readout gradient at q
    xq = x[ar, q].clone().requires_grad_(True)
    st = x.clone()
    st = st.index_put((ar, q), xq)
    with torch.enable_grad():
        Rv = R(st)
        g = torch.autograd.grad(Rv.sum(), xq)[0].double()
    with torch.no_grad():
        R0 = R(x).double()

        def jv(v):  # own-block J at q applied to v (n, d) float64
            nv = v.norm(dim=-1).clamp_min(1e-30)
            u = (v / nv[:, None]).float()
            ep = cfg["eps"]
            sp, sm = x.clone(), x.clone()
            sp[ar, q] += ep * u; sm[ar, q] -= ep * u
            both = h.step(torch.cat([e, e]), torch.cat([sp, sm]), cfg["K0"])
            dlt = (both[:n][ar, q] - both[n:][ar, q]).double() / (2 * ep)
            return dlt * nv[:, None]
        Jd = jv(D)
        ep0 = cfg["eps"]; cfg["eps"] = ep0 / 2; Jd_half = jv(D); cfg["eps"] = ep0  # FD step-size sensitivity
        fd_rel = ((Jd - Jd_half).norm(dim=-1) / Jd.norm(dim=-1)).cpu().numpy()
        v = Jd.clone(); acc = Jd.clone(); tail = None; acc48 = None
        for k in range(cfg["N"] - 1):
            if time.monotonic() - T0 > cfg["total_s"] - 120:
                raise RuntimeError("deadline")
            v = jv(v); acc += v
            if k + 2 == cfg["T"]:
                acc48 = acc.clone()
        tail = (v.norm(dim=-1) / acc.norm(dim=-1)).cpu().numpy()
        S0 = (g * D).sum(-1); S1 = (g * Jd).sum(-1); Sinf = (g * acc).sum(-1); ST = (g * acc48).sum(-1)
        print(f"neumann {time.monotonic() - T0:.0f}s", flush=True)
        # observed forced responses and sham
        obs, disp = {}, {}
        sham = x.clone()
        forced = {dz: x.clone() for dz in cfg["doses"]}
        for t in range(cfg["T"]):
            if time.monotonic() - T0 > cfg["total_s"] - 30:
                raise RuntimeError("deadline")
            sham = h.step(e, sham, cfg["K0"] + 1 + t)
            for dz in cfg["doses"]:
                inp = forced[dz].clone(); inp[ar, q] += (dz * D).to(inp.dtype)
                forced[dz] = h.step(e, inp, cfg["K0"] + 1 + t)
        Rs = R(sham).double()
        pre = torch.arange(sham.shape[1], device=h.dev)[None] <= q[:, None]
        sham_drift_q = ((sham - x)[ar, q].norm(dim=-1) / x[ar, q].norm(dim=-1)).cpu().numpy()
        sham_drift_prefix = (((sham - x).norm(dim=-1) / x.norm(dim=-1)) * pre).max(-1).values.cpu().numpy()
        for dz in cfg["doses"]:
            obs[dz] = ((R(forced[dz]).double() - Rs) / dz).cpu().numpy()
            disp[dz] = ((forced[dz] - sham)[ar, q].double().norm(dim=-1) / dz).cpu().numpy()
        os_ = x.clone(); os_[ar, q] += D.to(os_.dtype)
        oneshot_imm = (R(os_).double() - R0).cpu().numpy()
    np.savez_compressed(f"{o}/respond.npz", S0=S0.cpu().numpy(), S1=S1.cpu().numpy(), Sinf=Sinf.cpu().numpy(), ST=ST.cpu().numpy(),
                        fd_rel_eps_half=fd_rel, sham_drift_q=sham_drift_q, sham_drift_prefix=sham_drift_prefix,
                        unit=np.array([str(r.get("unit", r.get("uid"))) for r in items]),
                        dnorm=D.norm(dim=-1).cpu().numpy(), Jd_norm=Jd.norm(dim=-1).cpu().numpy(), dlin_norm=acc.norm(dim=-1).cpu().numpy(),
                        g_norm=g.norm(dim=-1).cpu().numpy(), cos_g_d=((g * D).sum(-1) / g.norm(dim=-1) / D.norm(dim=-1)).cpu().numpy(),
                        cos_g_dlin=((g * acc).sum(-1) / g.norm(dim=-1) / acc.norm(dim=-1)).cpu().numpy(),
                        cos_d_dlin=((D * acc).sum(-1) / D.norm(dim=-1) / acc.norm(dim=-1)).cpu().numpy(),
                        neumann_tail=tail, drift=drift, oneshot_imm=oneshot_imm,
                        doses=np.array(cfg["doses"]), obs=np.stack([obs[dz] for dz in cfg["doses"]], 1), disp=np.stack([disp[dz] for dz in cfg["doses"]], 1))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/items.json", "w") as f_:
        f_.write(items_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=480)
def dstar_vjp(items_json: str, cfg: dict, run_id: str):
    """Resolvent-optimal steering directions (truth judgment format, answer position q = last token; fp32, noise off,
    seed-0 init uid + 500000, K0 warm-up loops -> x*). Readout R = log p(' True') - log p(' False') (label-free).
    g = dR/dx_q (autograd through the coda); a = A^T g with A = sum_{k=1..T} J^k (J = own-block Jacobian at q, exact since q
    is last), computed by T vector-Jacobian products w <- J^T w through one recurrent step with autograd (a = sum of w_k).
    Saved per item: g, a (float32), |x*_q|, x* drift. a maximises the predicted finite-horizon response g . A d over unit d."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    seqs = []
    for r in items:
        ids_ = h.tok.encode(r["prompt_B"], add_special_tokens=True)
        seqs.append(dict(uid=r["uid"] + 500000, ids=ids_, pred_pos=[len(ids_) - 1]))
    val = h.validate(seqs[:20])
    TF = [h.tok.encode(w, add_special_tokens=False)[0] for w in (" True", " False")]
    n = len(items)
    G = np.zeros((n, h.d), np.float32); A = np.zeros((n, h.d), np.float32); drift = np.zeros(n, np.float32); xn = np.zeros(n, np.float32)
    ADJ = np.full(n, np.nan, np.float32)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    for c0 in range(0, n, cfg["chunk"]):
        ch = list(range(c0, min(n, c0 + cfg["chunk"])))
        N = len(ch)
        with torch.no_grad():
            ids, e, s = h.batch([seqs[i] for i in ch], [0] * N)
            for k in range(cfg["K0"]):
                s = h.step(e, s, k)
            x = s.clone()
            q = torch.tensor([seqs[i]["pred_pos"][0] for i in ch], device=h.dev)
            ar = torch.arange(N, device=h.dev)
            fx = h.step(e, x, cfg["K0"])
            drift[ch] = ((fx - x)[ar, q].norm(dim=-1) / x[ar, q].norm(dim=-1)).cpu().numpy()
            xn[ch] = x[ar, q].norm(dim=-1).cpu().numpy()
        with torch.enable_grad():
            xq = x[ar, q].clone().requires_grad_(True)
            st = x.clone().index_put((ar, q), xq)
            lg = torch.log_softmax(h.coda_logits(st, q[:, None])[:, 0], -1)
            Rv = lg[:, TF[0]] - lg[:, TF[1]]
            g = torch.autograd.grad(Rv.sum(), xq)[0].detach()
        with torch.no_grad():
            fc = h.model.freqs_cis[:, : e.shape[1]]
            ref = h.step(e, x, cfg["K0"])
            alt = h.model.core_block_forward(x, e, fc, None, None, torch.tensor(0, dtype=torch.long), current_step=cfg["K0"])[0]
            eq_err = float(((ref - alt).norm() / ref.norm()).item())
            assert eq_err < 1e-5, eq_err
        w = g.clone(); acc = torch.zeros_like(g)
        for k in range(cfg["T"]):
            if time.monotonic() - T0 > cfg["total_s"]:
                raise RuntimeError("deadline")
            with torch.enable_grad():
                xq = x[ar, q].clone().requires_grad_(True)
                st = x.clone().index_put((ar, q), xq)
                # iterate_one_step is decorated @torch.no_grad in the pinned model code; call core_block_forward directly
                # (identical computation: freqs_cis slice, adapter, core blocks; no mask/cache), which allows autograd
                fc = h.model.freqs_cis[:, : e.shape[1]]
                out = h.model.core_block_forward(st, e, fc, None, None, torch.tensor(0, dtype=torch.long), current_step=cfg["K0"])[0][ar, q]
                w = torch.autograd.grad((out * w).sum(), xq)[0].detach()
            acc += w
            if k == 0:  # adjoint check: (J^T g) . r  vs  g . (J r) with J r by central FD (eps on unit r)
                with torch.no_grad():
                    r = torch.randn(g.shape, device=h.dev, generator=torch.Generator(device=h.dev).manual_seed(9)); r = r / r.norm(dim=-1, keepdim=True)
                    ep = 0.1; sp, sm = x.clone(), x.clone(); sp[ar, q] += ep * r; sm[ar, q] -= ep * r
                    jr = (h.step(e, sp, cfg["K0"])[ar, q] - h.step(e, sm, cfg["K0"])[ar, q]) / (2 * ep)
                    lhs = (w * r).sum(-1); rhs = (g * jr).sum(-1)
                    ADJ[ch] = ((lhs - rhs).abs() / (lhs.abs() + rhs.abs()).clamp_min(1e-12)).cpu().numpy()
        G[ch] = g.cpu().numpy(); A[ch] = acc.cpu().numpy()
        print(f"chunk {ch[-1] + 1}/{n} {time.monotonic() - T0:.0f}s; |w_T|/|a| median {float((w.norm(dim=-1) / acc.norm(dim=-1)).median()):.2e}", flush=True)
    np.savez_compressed(f"{o}/dstar.npz", G=G, A=A, drift=drift, xnorm=xn, adjoint_rel_err=ADJ, uid=np.array([r["uid"] for r in items]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/items.json", "w") as f_:
        f_.write(items_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=540)
def dstar_eval(items_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """Held-out evaluation of steering directions under persistent forcing (truth judgment format, answer position q = last
    token; fp32; seed-0 init uid + 500000; K0 warm-up loops -> x*). Rows = items x conditions; condition c applies the push
    vector P[i, c] (given; zero for sham) to the state INPUT at q every loop: s <- F(s + P) (equivalent to the adapter-
    matched context edit). Readout R = log p(' True') - log p(' False') and the True+False mass at offsets cfg["ro"]."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    PZ = np.load(io.BytesIO(dirs_npz))
    Pv = torch.tensor(PZ["P"], device=h.dev, dtype=torch.float32)  # (n, C, d)
    names = [str(x) for x in PZ["names"]]
    seqs = []
    for r in items:
        ids_ = h.tok.encode(r["prompt_B"], add_special_tokens=True)
        seqs.append(dict(uid=r["uid"] + 500000, ids=ids_, pred_pos=[len(ids_) - 1]))
    val = h.validate(seqs[:20])
    TF = [h.tok.encode(w, add_special_tokens=False)[0] for w in (" True", " False")]
    n, C = Pv.shape[0], Pv.shape[1]
    RO = sorted(cfg["ro"]); T = RO[-1]
    Rv = np.full((n, C, len(RO)), np.nan, np.float32); M = np.full((n, C, len(RO)), np.nan, np.float32)
    KL = np.full((n, C, len(RO)), np.nan, np.float32)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    with torch.no_grad():
        ids, e, s = h.batch(seqs, [0] * n)
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        x = s.clone()
        q = torch.tensor([x_["pred_pos"][0] for x_ in seqs], device=h.dev)
        print(f"warm-up {time.monotonic() - T0:.0f}s", flush=True)

        def offtf(lg):  # log-probs renormalised over all tokens except ' True' / ' False'
            lg = lg.clone(); lg[:, TF] = float("-inf")
            return lg - torch.logsumexp(lg, -1, keepdim=True)
        if cfg.get("offkl"):
            REF = offtf(torch.log_softmax(h.coda_logits(x, q[:, None])[:, 0], -1))  # reference: unpushed x*
        rows = [(i, c) for i in range(n) for c in range(C)]
        for c0 in range(0, len(rows), cfg["chunk"]):
            if time.monotonic() - T0 > cfg["total_s"] - 25:
                raise RuntimeError("deadline")
            rr = rows[c0 : c0 + cfg["chunk"]]
            ii = torch.tensor([r[0] for r in rr], device=h.dev); cc = torch.tensor([r[1] for r in rr], device=h.dev)
            N = len(rr); ar = torch.arange(N, device=h.dev); qq = q[ii]
            st = x[ii].clone(); ee = e[ii]; push = Pv[ii, cc]

            def ro(t, stt):
                lg = torch.log_softmax(h.coda_logits(stt, qq[:, None])[:, 0], -1)
                rv = (lg[:, TF[0]] - lg[:, TF[1]]).cpu().numpy(); mv = (lg[:, TF[0]].exp() + lg[:, TF[1]].exp()).cpu().numpy()
                if cfg.get("offkl"):
                    lr = REF[ii]; lp = offtf(lg); pr = lr.exp()
                    kv = torch.where(pr > 0, pr * (lr - lp), torch.zeros_like(pr)).sum(-1).cpu().numpy()
                for j, (i, c) in enumerate(rr):
                    Rv[i, c, RO.index(t)] = rv[j]; M[i, c, RO.index(t)] = mv[j]
                    if cfg.get("offkl"):
                        KL[i, c, RO.index(t)] = kv[j]
            if 0 in RO:
                ro(0, st)
            for t in range(1, T + 1):
                if time.monotonic() - T0 > cfg["total_s"] - 15:
                    raise RuntimeError("deadline")
                inp = st.clone(); inp[ar, qq] += push
                st = h.step(ee, inp, cfg["K0"] + t - 1)
                if t in RO:
                    ro(t, st)
            print(f"rows {c0 + N}/{len(rows)} {time.monotonic() - T0:.0f}s", flush=True)
    np.savez_compressed(f"{o}/dstar_eval.npz", R=Rv, mass=M, offkl=KL, ro=np.array(RO), names=np.array(names), uid=np.array([r["uid"] for r in items]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/items.json", "w") as f_:
        f_.write(items_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=540)
def ctrl_eval(rows_json: str, dirs_npz: bytes, cfg: dict, run_id: str):
    """Control experiment for the frozen population d* (answer-token lever vs semantic truth lever); fp32, noise off,
    seed-0 init per (uid, format) (uid + format offset), K0 warm-up loops. Answer position q = last token. Conditions
    (cfg["conds"]: [name, dir, sign, norm, mode]): dir in {dstar, rand}; mode 'force' (push added to the state input at q
    every loop), 'pulse' (added once at offset 0), 'remove' (forced for cfg["Tr"] loops then off). Readout at offsets
    cfg["ro"]: log-probs of ' True', ' False', ' Yes', ' No', ' A', ' B' at q."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    rows = [r for r in json.loads(rows_json) if r["fmt"] in cfg["formats"]]
    Z = np.load(io.BytesIO(dirs_npz))
    u_dstar = torch.tensor(Z["u_dstar"], device=h.dev, dtype=torch.float32)
    FOFF = {"tf": 500000, "yn": 600000, "rev": 700000, "ab": 800000, "fq": 900000, "ba": 1000000}
    seqs = []
    for r in rows:
        ids_ = h.tok.encode(r["prompt"], add_special_tokens=True)
        seqs.append(dict(uid=r["uid"] + FOFF[r["fmt"]], ids=ids_, pred_pos=[len(ids_) - 1]))
    val = h.validate(seqs[:12])
    TOK = [h.tok.encode(w, add_special_tokens=False) for w in (" True", " False", " Yes", " No", " A", " B")]
    assert all(len(t) == 1 for t in TOK)
    TOK = torch.tensor([t[0] for t in TOK], device=h.dev)
    conds = cfg["conds"]; RO = sorted(cfg["ro"]); T = RO[-1]
    n, C = len(rows), len(conds)
    LP = np.full((n, C, len(RO), 6), np.nan, np.float32); DONE = np.zeros((n, C), bool)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    with torch.no_grad():
        ids, e, s = h.batch(seqs, [0] * n)
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        x = s.clone()
        q = torch.tensor([x_["pred_pos"][0] for x_ in seqs], device=h.dev)
        print(f"warm-up {time.monotonic() - T0:.0f}s", flush=True)
        g_ = torch.Generator(device=h.dev).manual_seed(77)
        u_rand = torch.randn(h.d, device=h.dev, generator=g_); u_rand = u_rand / u_rand.norm()
        allrows = [(i, c) for i in range(n) for c in range(C)]
        for c0 in range(0, len(allrows), cfg["chunk"]):
            if time.monotonic() - T0 > cfg["total_s"] - 25:
                print("deadline: stopping before chunk", c0, flush=True)
                break
            rr = allrows[c0 : c0 + cfg["chunk"]]
            ii = torch.tensor([r[0] for r in rr], device=h.dev); N = len(rr); ar = torch.arange(N, device=h.dev); qq = q[ii]
            push = torch.zeros((N, h.d), device=h.dev); mode = []
            for j, (i, c) in enumerate(rr):
                name, dk, sg, nm, md = conds[c]
                if dk != "none":
                    push[j] = sg * nm * (u_dstar if dk == "dstar" else u_rand)
                mode.append(md)
            pulse = torch.tensor([m == "pulse" for m in mode], device=h.dev)
            remove = torch.tensor([m == "remove" for m in mode], device=h.dev)
            st = x[ii].clone(); ee = e[ii]
            st[ar[pulse], qq[pulse]] += push[pulse]

            def ro(t, stt):
                lg = torch.log_softmax(h.coda_logits(stt, qq[:, None])[:, 0], -1)[:, TOK].cpu().numpy()
                for j, (i, c) in enumerate(rr):
                    LP[i, c, RO.index(t)] = lg[j]
            if 0 in RO:
                ro(0, st)
            aborted = False
            for t in range(1, T + 1):
                if time.monotonic() - T0 > cfg["total_s"] - 10:
                    aborted = True; print("deadline inside chunk", c0, flush=True); break
                active = ~pulse & (~remove | torch.tensor(t <= cfg["Tr"], device=h.dev))
                inp = st.clone(); inp[ar[active], qq[active]] += push[active]
                st = h.step(ee, inp, cfg["K0"] + t - 1)
                if t in RO:
                    ro(t, st)
            if aborted:
                break
            for (i, c) in rr:
                DONE[i, c] = True
            print(f"rows {c0 + N}/{len(allrows)} {time.monotonic() - T0:.0f}s", flush=True)
    np.savez_compressed(f"{o}/ctrl.npz", LP=LP, done=DONE, ro=np.array(RO), conds=np.array([json.dumps(c_) for c_ in conds]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION,
            "complete": bool(DONE.all())}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/rows.json", "w") as f_:
        json.dump(rows, f_)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=540)
def dglever(items_json: str, pushes_npz: bytes, cfg: dict, run_id: str):
    """Digit lever experiment (fp32, noise off; dg-style prompts, seed-0 init + truncation as dgcollect, K0 warm-up -> x*;
    q = last token = the tens answer token predicting the units digit). Centered digit logits z_k = log p(k) - mean_j log p(j)
    over the 10 bare digit tokens. Modes:
      'train': G = dz_k/dx_q (10 per item, autograd through the coda) and a_k = A^T G_k with A = sum_{t=1..T} J^t (T VJPs,
               batched over the 10 targets; core_block_forward because iterate_one_step is @torch.no_grad).
      'pred' : G (as above) and A P for every push P in the npz (JVP Neumann sum by central FD at q).
      'eval' : observed persistent forcing s <- F(s + P) for T loops for every push P (+ sham); z at the end.
    Pushes: npz 'P' (C x d) applied identically to every item (population directions)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    h.prep(items)
    val = h.validate(items[:10])
    n = len(items)
    with torch.no_grad():
        ids, e, s = h.batch(items, [0] * n)
        Lq = max(p["pred_pos"][-1] + 1 for p in items)
        e, s = e[:, :Lq], s[:, :Lq]
        q = torch.tensor([p["pred_pos"][-1] for p in items], device=h.dev)
        assert bool((q == Lq - 1).all())
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        x = s.clone()
    ar = torch.arange(n, device=h.dev)
    dig = torch.tensor(h.DIG, device=h.dev)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    out = {}

    def zfun(st, qq):
        lg = torch.log_softmax(h.coda_logits(st, qq[:, None])[:, 0], -1)[:, dig]
        return lg - lg.mean(-1, keepdim=True)

    def readout_grads():
        Gs = []
        for k in range(10):
            with torch.enable_grad():
                xq = x[ar, q].clone().requires_grad_(True)
                st = x.clone().index_put((ar, q), xq)
                gk = torch.autograd.grad(zfun(st, q)[:, k].sum(), xq)[0].detach()
            Gs.append(gk)
        return torch.stack(Gs, 1)  # (n, 10, d)
    fc = h.model.freqs_cis[:, : e.shape[1]]
    if cfg["mode"] in ("train", "pred"):
        G = readout_grads(); out["G"] = G.cpu().numpy()
    if cfg["mode"] == "train":
        # VJP chains batched over (item, target): rows r = i*10 + k
        rows_i = torch.arange(n, device=h.dev).repeat_interleave(10)
        w = G.reshape(n * 10, -1).clone(); acc = torch.zeros_like(w)
        xr, er, qr = x[rows_i], e[rows_i], q[rows_i]; arr = torch.arange(n * 10, device=h.dev)
        for t in range(cfg["T"]):
            if time.monotonic() - T0 > cfg["total_s"]:
                raise RuntimeError("deadline")
            outs = []
            for c0 in range(0, n * 10, cfg["chunk"]):
                sl = slice(c0, min(n * 10, c0 + cfg["chunk"])); a_ = arr[sl] - c0
                with torch.enable_grad():
                    xq = xr[sl][a_, qr[sl]].clone().requires_grad_(True)
                    st = xr[sl].clone().index_put((a_, qr[sl]), xq)
                    yq = h.model.core_block_forward(st, er[sl], fc, None, None, torch.tensor(0, dtype=torch.long), current_step=cfg["K0"])[0][a_, qr[sl]]
                    outs.append(torch.autograd.grad((yq * w[sl]).sum(), xq)[0].detach())
            w = torch.cat(outs); acc += w
        out["AtG"] = acc.reshape(n, 10, -1).cpu().numpy()
        out["tail"] = (w.norm(dim=-1) / acc.norm(dim=-1)).reshape(n, 10).cpu().numpy()
    if cfg["mode"] in ("pred", "eval"):
        Pz = torch.tensor(np.load(io.BytesIO(pushes_npz))["P"], device=h.dev, dtype=torch.float32)  # (C, d)
        C = Pz.shape[0]
        assert C == cfg.get("n_push", 10 if cfg["mode"] == "pred" else 32), (cfg["mode"], C)
    if cfg["mode"] == "pred":
        rows = [(i, c) for i in range(n) for c in range(C)]
        ri = torch.tensor([r[0] for r in rows], device=h.dev); ci = torch.tensor([r[1] for r in rows], device=h.dev)
        D = Pz[ci].double()
        ep = cfg["eps"]

        def jv(v):
            res = []
            for c0 in range(0, len(rows), cfg["chunk"]):
                sl = slice(c0, min(len(rows), c0 + cfg["chunk"])); ii = ri[sl]; N = len(ii); a_ = torch.arange(N, device=h.dev)
                vv = v[sl]; nv = vv.norm(dim=-1).clamp_min(1e-30); u = (vv / nv[:, None]).float()
                sp, sm = x[ii].clone(), x[ii].clone(); sp[a_, q[ii]] += ep * u; sm[a_, q[ii]] -= ep * u
                with torch.no_grad():
                    both = h.step(torch.cat([e[ii], e[ii]]), torch.cat([sp, sm]), cfg["K0"])
                res.append(((both[:N][a_, q[ii]] - both[N:][a_, q[ii]]).double() / (2 * ep)) * nv[:, None])
            return torch.cat(res)
        v = jv(D); accp = v.clone()
        for t in range(cfg["T"] - 1):
            if time.monotonic() - T0 > cfg["total_s"]:
                raise RuntimeError("deadline")
            v = jv(v); accp += v
        out["AP"] = accp.reshape(n, C, -1).float().cpu().numpy()
        out["tail"] = (v.norm(dim=-1) / accp.norm(dim=-1)).reshape(n, C).cpu().numpy()
    if cfg["mode"] == "eval":
        Z = np.full((n, C + 1, 10), np.nan, np.float32)  # index 0 = sham
        DL = np.full((n, C + 1), np.nan, np.float32); DISP = np.full((n, C + 1), np.nan, np.float32)
        rows = [(i, c) for i in range(n) for c in range(C + 1)]
        for c0 in range(0, len(rows), cfg["chunk"]):
            if time.monotonic() - T0 > cfg["total_s"] - 20:
                raise RuntimeError("deadline")
            rr = rows[c0 : c0 + cfg["chunk"]]; N = len(rr)
            ii = torch.tensor([r[0] for r in rr], device=h.dev); a_ = torch.arange(N, device=h.dev)
            push = torch.stack([torch.zeros(h.d, device=h.dev) if c == 0 else Pz[c - 1] for _, c in rr])
            st = x[ii].clone()
            with torch.no_grad():
                for t in range(cfg["T"]):
                    if time.monotonic() - T0 > cfg["total_s"] - 10:
                        raise RuntimeError("deadline")
                    inp = st.clone(); inp[a_, q[ii]] += push
                    prev = st
                    st = h.step(e[ii], inp, cfg["K0"] + t)
                zz = zfun(st, q[ii]).cpu().numpy()
                dlast = ((st - prev).norm(dim=-1) / st.norm(dim=-1)).max(-1).values.cpu().numpy()
                disp = ((st - x[ii])[a_, q[ii]].norm(dim=-1) / x[ii][a_, q[ii]].norm(dim=-1)).cpu().numpy()
            for j, (i, c) in enumerate(rr):
                Z[i, c] = zz[j]; DL[i, c] = dlast[j]; DISP[i, c] = disp[j]
            print(f"eval rows {c0 + N}/{len(rows)} {time.monotonic() - T0:.0f}s", flush=True)
        out["Z"] = Z; out["last_step_max_drift"] = DL; out["endpoint_displacement_q"] = DISP
    np.savez_compressed(f"{o}/dglever.npz", uid=np.array([p["uid"] for p in items]), ans=np.array([p["ans"] for p in items]), **out)
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/items.json", "w") as f_:
        f_.write(items_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=540)
def revmech(items_json: str, pushes_npz: bytes, cfg: dict, run_id: str):
    """Reversal mechanism (digits; fp32, noise off; dgcollect-style init/truncation, K0 warm-up -> x*, q = last token).
    For every (item, push P) row: Arnoldi on the own-block J_qq STARTED FROM P (unit vector), m steps, central-FD J v at q
    (eps on unit vectors), CGS2 in fp64. Saved: H ((m+1) x m), |P|, and the projections V^T g_k of the 10 centered digit-logit
    readout gradients g_k (autograd through the coda) onto the Krylov basis ((m+1) x 10). Offline, the readout impulse
    response c_t = |P| (V^T g)^T H^t e1 is exact for t < m (no truncation beyond the FD/Arnoldi numerics), and its modal
    decomposition follows from eig(H[:m, :m])."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    h.prep(items)
    val = h.validate(items[:10])
    Z = np.load(io.BytesIO(pushes_npz)); Pz = torch.tensor(Z["P"][cfg["push_idx"]], device=h.dev, dtype=torch.float64)
    names = [str(Z["names"][i]) for i in cfg["push_idx"]]
    n, C, m = len(items), Pz.shape[0], cfg["m"]
    dig = torch.tensor(h.DIG, device=h.dev)
    with torch.no_grad():
        ids, e, s = h.batch(items, [0] * n)
        Lq = max(p["pred_pos"][-1] + 1 for p in items)
        e, s = e[:, :Lq], s[:, :Lq]
        q = torch.tensor([p["pred_pos"][-1] for p in items], device=h.dev)
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        x = s.clone()
    ar = torch.arange(n, device=h.dev)
    Gs = []
    for k in range(10):
        with torch.enable_grad():
            xq = x[ar, q].clone().requires_grad_(True)
            st = x.clone().index_put((ar, q), xq)
            lg = torch.log_softmax(h.coda_logits(st, q[:, None])[:, 0], -1)[:, dig]
            z = lg - lg.mean(-1, keepdim=True)
            Gs.append(torch.autograd.grad(z[:, k].sum(), xq)[0].detach().double())
    G = torch.stack(Gs, 1)  # (n, 10, d)
    rows = [(i, c) for i in range(n) for c in range(C)]
    HH = np.zeros((len(rows), m + 1, m)); VG = np.zeros((len(rows), m + 1, 10)); PN = np.zeros(len(rows))
    ORTH = np.zeros(len(rows)); EPS = np.zeros(len(rows))
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    ep = cfg["eps"]
    for c0 in range(0, len(rows), cfg["chunk"]):
        rr = rows[c0 : c0 + cfg["chunk"]]; N = len(rr)
        ii = torch.tensor([r[0] for r in rr], device=h.dev); cc = torch.tensor([r[1] for r in rr], device=h.dev)
        a_ = torch.arange(N, device=h.dev); qq = q[ii]; xx = x[ii]; ee = e[ii]
        P0 = Pz[cc]; PN[c0 : c0 + N] = P0.norm(dim=-1).cpu().numpy()
        V = torch.zeros((N, m + 1, h.d), device=h.dev, dtype=torch.float64); Hm = torch.zeros((N, m + 1, m), device=h.dev, dtype=torch.float64)
        V[:, 0] = P0 / P0.norm(dim=-1, keepdim=True)
        for j in range(m):
            if time.monotonic() - T0 > cfg["total_s"]:
                raise RuntimeError("deadline")
            u = V[:, j].float()
            sp, sm = xx.clone(), xx.clone(); sp[a_, qq] += ep * u; sm[a_, qq] -= ep * u
            with torch.no_grad():
                both = h.step(torch.cat([ee, ee]), torch.cat([sp, sm]), cfg["K0"])
            w = ((both[:N][a_, qq] - both[N:][a_, qq]).double() / (2 * ep))
            for _ in range(2):
                cf = torch.einsum("nkd,nd->nk", V[:, : j + 1], w)
                Hm[:, : j + 1, j] += cf
                w = w - torch.einsum("nk,nkd->nd", cf, V[:, : j + 1])
            hn = w.norm(dim=-1); Hm[:, j + 1, j] = hn; V[:, j + 1] = w / hn.clamp_min(1e-30)[:, None]
        HH[c0 : c0 + N] = Hm.cpu().numpy()
        Vo = V[:, : m + 1]; gram = torch.einsum("nkd,nld->nkl", Vo, Vo)
        ORTH[c0 : c0 + N] = (gram - torch.eye(m + 1, device=h.dev, dtype=torch.float64)).abs().amax(dim=(1, 2)).cpu().numpy()
        with torch.no_grad():  # FD step-size check of J P at eps vs eps/2
            u = V[:, 0].float(); outs = []
            for ep_ in (ep, ep / 2):
                sp, sm = xx.clone(), xx.clone(); sp[a_, qq] += ep_ * u; sm[a_, qq] -= ep_ * u
                both = h.step(torch.cat([ee, ee]), torch.cat([sp, sm]), cfg["K0"])
                outs.append((both[:N][a_, qq] - both[N:][a_, qq]).double() / (2 * ep_))
            EPS[c0 : c0 + N] = ((outs[0] - outs[1]).norm(dim=-1) / outs[0].norm(dim=-1)).cpu().numpy()
        VG[c0 : c0 + N] = torch.einsum("nkd,nrd->nkr", V, G[ii]).cpu().numpy()
        del V
        print(f"rows {c0 + N}/{len(rows)} {time.monotonic() - T0:.0f}s", flush=True)
    np.savez_compressed(f"{o}/revmech.npz", H=HH, VG=VG, PN=PN, orth_err=ORTH, fd_eps_rel=EPS, row_item=np.array([r[0] for r in rows]), row_push=np.array([r[1] for r in rows]),
                        push_names=np.array(names), uid=np.array([p["uid"] for p in items]), ans=np.array([p["ans"] for p in items]),
                        G_norm=G.norm(dim=-1).cpu().numpy())
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/items.json", "w") as f_:
        f_.write(items_json)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=420)
def rtransfer(items_json: str, cfg: dict, run_id: str):
    """Local readout transfer operator (digits; fp32; dgcollect-style x* after K0 loops; q = last token). Per item: the 10
    centered digit-logit gradients g_k (autograd through the coda); J^T g_k for each k (one VJP through core_block_forward,
    since iterate_one_step is @torch.no_grad); and J g_k (central FD JVP at q) as a cross-check of the adjoint. Saved:
    G (n,10,d), JTG (n,10,d), JG (n,10,d)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    h.prep(items)
    val = h.validate(items[:10])
    n = len(items)
    dig = torch.tensor(h.DIG, device=h.dev)
    with torch.no_grad():
        ids, e, s = h.batch(items, [0] * n)
        Lq = max(p["pred_pos"][-1] + 1 for p in items)
        e, s = e[:, :Lq], s[:, :Lq]
        q = torch.tensor([p["pred_pos"][-1] for p in items], device=h.dev)
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        x = s.clone()
    ar = torch.arange(n, device=h.dev)
    fc = h.model.freqs_cis[:, : e.shape[1]]
    G, JTG, JG = [], [], []
    for k in range(10):
        with torch.enable_grad():
            xq = x[ar, q].clone().requires_grad_(True)
            st = x.clone().index_put((ar, q), xq)
            lg = torch.log_softmax(h.coda_logits(st, q[:, None])[:, 0], -1)[:, dig]
            gk = torch.autograd.grad((lg - lg.mean(-1, keepdim=True))[:, k].sum(), xq)[0].detach()
        with torch.enable_grad():
            xq = x[ar, q].clone().requires_grad_(True)
            st = x.clone().index_put((ar, q), xq)
            yq = h.model.core_block_forward(st, e, fc, None, None, torch.tensor(0, dtype=torch.long), current_step=cfg["K0"])[0][ar, q]
            jt = torch.autograd.grad((yq * gk).sum(), xq)[0].detach()
        with torch.no_grad():
            nv = gk.norm(dim=-1, keepdim=True); u = gk / nv; ep = cfg["eps"]
            sp, sm = x.clone(), x.clone(); sp[ar, q] += ep * u; sm[ar, q] -= ep * u
            jg = (h.step(e, sp, cfg["K0"])[ar, q] - h.step(e, sm, cfg["K0"])[ar, q]) / (2 * ep) * nv
        G.append(gk); JTG.append(jt); JG.append(jg)
    G, JTG, JG = (torch.stack(v, 1).cpu().numpy() for v in (G, JTG, JG))
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    np.savez_compressed(f"{o}/rtransfer.npz", G=G, JTG=JTG, JG=JG, uid=np.array([p["uid"] for p in items]), ans=np.array([p["ans"] for p in items]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=420)
def blockdecomp(items_json: str, pushes_npz: bytes, cfg: dict, run_id: str):
    """Sublayer decomposition of the one-step readout response (digits; fp32; x* after K0 loops; q = last token; own-block
    perturbation at q only). The core block is adapter -> 4 sandwich layers (a = attn(N1 x); y = N2(a + x);
    m = mlp(N3 y); x' = N4(m + y)). Linearising, the output tangent at q is EXACTLY sum_s R_s dw_s, where dw_s is the total
    tangent of write s (adapter output, attn_l, mlp_l at q) and R_s is the residual/norm-only chain from write s to the
    output. Readout contribution of write s to readout k: mu_{k,s} . dw_s, with mu_{k,s} = R_s^T g_k (autograd through a
    replica whose sublayer inputs/outputs are detached). dw_s by central FD (eps on the unit push, scaled by |P|).
    Attention split (first order): query path (query from the perturbed attention input, keys/values from the base) vs
    self key/value path (query base, k/v perturbed, which differ only at row q). Query path attributed to source positions
    j: dalpha_{h,j} * (u_h . (v_{h,j} - y_h)), u = W_o^T mu, y_h = attention output of head h; aggregated by token category.
    Gates saved: replica vs core_block_forward; sum of contributions vs FD readout change vs VJP (J^T g . P); Q+KV vs attn
    write; per-position sum vs query-path contribution."""
    import math
    import sys as _sys

    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    h.prep(items)
    val = h.validate(items[:10])
    Z = np.load(io.BytesIO(pushes_npz)); PZ = torch.tensor(Z["P"], device=h.dev, dtype=torch.float32)
    names = [str(x_) for x_ in Z["names"]]
    n, NP_ = len(items), PZ.shape[0]
    dig = torch.tensor(h.DIG, device=h.dev)
    M = h.model; core = list(M.transformer.core_block); NLy = len(core)
    rot = _sys.modules[type(core[0].attn).__module__].apply_rotary_emb_complex_like
    CATS = ["context", "symbols", "op_samecol", "op_lowercol", "op_othercol", "ans_prev", "self"]
    cat = np.zeros((n, max(len(p_["ids"]) for p_ in items)), dtype=np.int64) - 1
    for i, p_ in enumerate(items):
        ids_ = p_["ids"]; qq = p_["pred_pos"][-1]; npre = p_["pred_pos"][0] + 1
        nl = max(j for j in range(qq) if ids_[j] == h.NL)
        tcol = p_["digit_info"][-1]["col"]
        assert tcol == 0
        cat[i, : nl + 1] = 0
        txt = [h.tok.decode([t]).strip() for t in ids_[nl + 1: npre]]
        plus = txt.index("+"); eq = txt.index("=")
        for side, (lo, hi) in enumerate(((0, plus), (plus + 1, eq))):
            dpos = [nl + 1 + j for j in range(lo, hi) if txt[j].isdigit()]
            assert "".join(txt[j - nl - 1] for j in dpos) == str(p_["a"] if side == 0 else p_["b"]), (txt, p_["a"], p_["b"])
            for r_, j in enumerate(reversed(dpos)):
                cat[i, j] = 2 if r_ == tcol else (3 if r_ == tcol - 1 else 4)
        for j in range(nl + 1, npre):
            if cat[i, j] < 0:
                cat[i, j] = 1
        cat[i, npre:qq] = 5; cat[i, qq] = 6
        assert (cat[i, : qq + 1] >= 0).all()
    with torch.no_grad():
        ids, e, s = h.batch(items, [0] * n)
        Lq = max(p_["pred_pos"][-1] + 1 for p_ in items)
        e, s = e[:, :Lq], s[:, :Lq]
        q = torch.tensor([p_["pred_pos"][-1] for p_ in items], device=h.dev)
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        x = s.clone()
    catT = torch.tensor(cat[:, :Lq], device=h.dev)
    ar = torch.arange(n, device=h.dev)
    fc = M.freqs_cis[:, :Lq]
    bi = torch.tensor(0, dtype=torch.long)
    K0 = cfg["K0"]

    def attn_mix(at, nq, nkv):
        B, S, E = nq.shape
        qh = at.Wqkv(nq).split(at.chunks, dim=2)[0].view(B, S, at.n_head, at.head_dim)
        _, kh, vh = at.Wqkv(nkv).split(at.chunks, dim=2)
        kh = kh.view(B, S, at.n_kv_heads, at.head_dim); vh = vh.view(B, S, at.n_kv_heads, at.head_dim)
        if at.config.qk_bias:
            qb, kb = at.qk_bias.split(1, dim=0); qh, kh = (qh + qb).to(qh.dtype), (kh + kb).to(qh.dtype)
        qh, kh = rot(qh, kh, freqs_cis=fc)
        qh, kh, vh = qh.transpose(1, 2), kh.transpose(1, 2), vh.transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=True)
        return at.proj(y.transpose(1, 2).reshape(B, S, E)), qh, kh, vh

    def replica(st):
        """Full forward with records: writes at q (adapter, attn_l, mlp_l) and the attention inputs N1 x (all positions)."""
        x_ = M.transformer.adapter(torch.cat([st, e], dim=-1))
        W = [x_[ar, q]]; N1 = []
        for blk in core:
            n1 = blk.norm_1(x_); N1.append(n1)
            a = blk.attn(n1, fc, bi, None, None); W.append(a[ar, q])
            y_ = blk.norm_2(a + x_)
            m_ = blk.mlp(blk.norm_3(y_)); W.append(m_[ar, q])
            x_ = blk.norm_4(m_ + y_)
        return x_, W, N1

    def adjoint(st, gk):
        Zs = [torch.zeros(n, Lq, h.d, device=h.dev, requires_grad=True) for _ in range(1 + 2 * NLy)]
        x_ = M.transformer.adapter(torch.cat([st, e], dim=-1)).detach() + Zs[0]
        for l_, blk in enumerate(core):
            a = blk.attn(blk.norm_1(x_.detach()), fc, bi, None, None).detach() + Zs[1 + 2 * l_]
            y_ = blk.norm_2(a + x_)
            m_ = blk.mlp(blk.norm_3(y_.detach())).detach() + Zs[2 + 2 * l_]
            x_ = blk.norm_4(m_ + y_)
        gr = torch.autograd.grad((x_[ar, q] * gk).sum(), Zs)
        return torch.stack([g_[ar, q] for g_ in gr], 1)  # (n, 9, d)

    with torch.no_grad():
        ref = M.core_block_forward(x, e, fc, None, None, torch.tensor(0, dtype=torch.long), current_step=K0)[0]
        xb, Wb, N1b = replica(x)
        rep_err = float(((xb - ref).norm() / ref.norm()).item())
        mix_err = max(float(((attn_mix(blk.attn, N1b[l_], N1b[l_])[0] - blk.attn(N1b[l_], fc, bi, None, None)).norm()
                             / blk.attn(N1b[l_], fc, bi, None, None).norm()).item()) for l_, blk in enumerate(core))
        step_err = float(((h.step(e, x, K0) - ref).norm() / ref.norm()).item())
        # base attention quantities at row q per layer: alpha (n,H,S), v (n,H,S,hd) (kv heads expanded), y (n,H,hd)
        BASE = []
        for l_, blk in enumerate(core):
            _, qh, kh, vh = attn_mix(blk.attn, N1b[l_], N1b[l_])
            rep_ = blk.attn.n_head // blk.attn.n_kv_heads
            kh = kh.repeat_interleave(rep_, 1); vh = vh.repeat_interleave(rep_, 1)
            BASE.append((kh, vh))
        attn_cat = np.zeros((n, NLy, len(CATS)))
    G = []; MU = []
    for k in range(10):
        with torch.enable_grad():
            xq = x[ar, q].clone().requires_grad_(True)
            st = x.clone().index_put((ar, q), xq)
            lg = torch.log_softmax(h.coda_logits(st, q[:, None])[:, 0], -1)[:, dig]
            gk = torch.autograd.grad((lg - lg.mean(-1, keepdim=True))[:, k].sum(), xq)[0].detach()
            MU.append(adjoint(x, gk))
        G.append(gk)
    G = torch.stack(G, 1); MU = torch.stack(MU, 1)  # (n,10,d), (n,10,9,d)
    JTG = []
    for k in range(10):
        with torch.enable_grad():
            xq = x[ar, q].clone().requires_grad_(True)
            st = x.clone().index_put((ar, q), xq)
            yq = M.core_block_forward(st, e, fc, None, None, torch.tensor(0, dtype=torch.long), current_step=K0)[0][ar, q]
            JTG.append(torch.autograd.grad((yq * G[:, k]).sum(), xq)[0].detach())
    JTG = torch.stack(JTG, 1)
    NC = len(CATS); NS = 1 + 2 * NLy
    comp = np.zeros((n, NP_, 10, NS)); qpath = np.zeros((n, NP_, 10, NLy)); kvpath = np.zeros((n, NP_, 10, NLy))
    src = np.zeros((n, NP_, 10, NLy, NC)); c1fd = np.zeros((n, NP_, 10)); c1vjp = np.zeros((n, NP_, 10)); c0 = np.zeros((n, NP_, 10))
    dwn = np.zeros((n, NP_, NS)); split_err = np.zeros((n, NP_, NLy)); src_err = np.zeros((n, NP_, 10, NLy))
    ep = cfg["eps"]
    ALB = []; nonfinite = 0
    with torch.no_grad():
        for l_, blk in enumerate(core):  # where row q attends (base), by category, head-averaged
            _, qh, _, _ = attn_mix(blk.attn, N1b[l_], N1b[l_]); kh, vh = BASE[l_]
            sc = torch.einsum("bhd,bhsd->bhs", qh[ar, :, q], kh) / math.sqrt(blk.attn.head_dim)
            sc = sc.masked_fill(torch.arange(Lq, device=h.dev)[None, None] > q[:, None, None], float("-inf"))
            ALB.append(sc.softmax(-1)); al = ALB[-1].mean(1)
            for c_ in range(NC):
                attn_cat[:, l_, c_] = (al * (catT == c_)).sum(-1).cpu().numpy()
        for pi in range(NP_):
            if time.monotonic() - T0 > cfg["total_s"]:
                break
            P = PZ[pi]; pn = P.norm(); u = P / pn
            outs = []
            for sg in (1.0, -1.0):
                xs = x.clone(); xs[ar, q] += sg * ep * u
                xo, W, N1 = replica(xs)
                rec = dict(out=xo[ar, q], W=W, Q=[], KV=[], AL=[])
                for l_, blk in enumerate(core):
                    oq, qh, _, _ = attn_mix(blk.attn, N1[l_], N1b[l_])
                    okv = attn_mix(blk.attn, N1b[l_], N1[l_])[0]
                    rec["Q"].append(oq[ar, q]); rec["KV"].append(okv[ar, q])
                    kh, _ = BASE[l_]
                    sc = torch.einsum("bhd,bhsd->bhs", qh[ar, :, q], kh) / math.sqrt(blk.attn.head_dim)
                    sc = sc.masked_fill(torch.arange(Lq, device=h.dev)[None, None] > q[:, None, None], float("-inf"))
                    rec["AL"].append(sc.softmax(-1))
                outs.append(rec)
            Dp, Dm = outs
            fd = lambda a_, b_: (a_ - b_) / (2 * ep) * pn
            dW = torch.stack([fd(Dp["W"][s_], Dm["W"][s_]) for s_ in range(NS)], 1)  # (n,NS,d)
            dout = fd(Dp["out"], Dm["out"])
            comp[:, pi] = torch.einsum("bksd,bsd->bks", MU, dW).cpu().numpy()
            c1fd[:, pi] = torch.einsum("bkd,bd->bk", G, dout).cpu().numpy()
            c1vjp[:, pi] = torch.einsum("bkd,d->bk", JTG, P).cpu().numpy()
            c0[:, pi] = torch.einsum("bkd,d->bk", G, P).cpu().numpy()
            dwn[:, pi] = dW.norm(dim=-1).cpu().numpy()
            for l_, blk in enumerate(core):
                dQ = fd(Dp["Q"][l_], Dm["Q"][l_]); dKV = fd(Dp["KV"][l_], Dm["KV"][l_]); dA = dW[:, 1 + 2 * l_]
                split_err[:, pi, l_] = ((dQ + dKV - dA).norm(dim=-1) / dA.norm(dim=-1)).cpu().numpy()
                mu = MU[:, :, 1 + 2 * l_]  # (n,10,d)
                qpath[:, pi, :, l_] = torch.einsum("bkd,bd->bk", mu, dQ).cpu().numpy()
                kvpath[:, pi, :, l_] = torch.einsum("bkd,bd->bk", mu, dKV).cpu().numpy()
                dal = fd(Dp["AL"][l_], Dm["AL"][l_])  # (n,H,S)
                at = blk.attn; _, vh = BASE[l_]
                yh = torch.einsum("bhs,bhsd->bhd", ALB[l_], vh)  # base head outputs at row q
                uu = torch.einsum("bkd,de->bke", mu, at.proj.weight).view(n, 10, at.n_head, at.head_dim)
                cj = torch.einsum("bhs,bkhs->bks", dal, torch.einsum("bkhd,bhsd->bkhs", uu, vh) - torch.einsum("bkhd,bhd->bkh", uu, yh)[..., None])
                valid = torch.arange(Lq, device=h.dev)[None, None] <= q[:, None, None]
                nonfinite += int((~torch.isfinite(cj) & valid).sum().item())
                cj = cj.masked_fill(~valid, 0.0)
                cjn = cj.cpu().numpy(); src_err[:, pi, :, l_] = cjn.sum(-1) - qpath[:, pi, :, l_]
                for c_ in range(NC):
                    src[:, pi, :, l_, c_] = (cjn * (cat[:, None, :Lq] == c_)).sum(-1)
                src_err[:, pi, :, l_] = src[:, pi, :, l_].sum(-1) - qpath[:, pi, :, l_]  # categorized sum vs query path
    done = int(pi + 1 if time.monotonic() - T0 <= cfg["total_s"] else pi)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    np.savez_compressed(f"{o}/blockdecomp.npz", comp=comp, qpath=qpath, kvpath=kvpath, src=src, c1fd=c1fd, c1vjp=c1vjp, c0=c0,
                        dwn=dwn, split_err=split_err, src_err=src_err, attn_cat=attn_cat, names=np.array(names), cats=np.array(CATS),
                        cat=cat, n_done_push=done, uid=np.array([p_["uid"] for p_ in items]), ans=np.array([p_["ans"] for p_ in items]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION,
            "replica_rel_err": rep_err, "attn_mix_rel_err": mix_err, "step_vs_core_rel_err": step_err, "n_done_push": done,
            "nonfinite_source_terms": nonfinite}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=420)
def adaptcov(items_json: str, cfg: dict, run_id: str):
    """Pulled-back readout covectors at x* (digits; fp32; K0 loops; q = last token). Per item and digit k (centered digit
    logit z_k): g_k = dz_k/dx_q (coda); mu_k = residual/norm-only adjoint at the adapter output (as in `blockdecomp`);
    a_k = W_s^T mu_k (adapter pullback; W_s = state columns of the adapter); w_k = W_s^T g_k (no norm chain);
    b_k = J^T g_k (full one-step pullback, VJP through core_block_forward). Saved: G, A, W, B (n, 10, d)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    h.prep(items)
    val = h.validate(items[:10])
    n = len(items); M = h.model; core = list(M.transformer.core_block)
    dig = torch.tensor(h.DIG, device=h.dev)
    with torch.no_grad():
        ids, e, s = h.batch(items, [0] * n)
        Lq = max(p_["pred_pos"][-1] + 1 for p_ in items)
        e, s = e[:, :Lq], s[:, :Lq]
        q = torch.tensor([p_["pred_pos"][-1] for p_ in items], device=h.dev)
        for k in range(cfg["K0"]):
            s = h.step(e, s, k)
        x = s.clone()
    ar = torch.arange(n, device=h.dev); fc = M.freqs_cis[:, :Lq]; bi = torch.tensor(0, dtype=torch.long)
    Ws = M.transformer.adapter.weight[:, : h.d]  # (d_out, d_state)
    assert M.transformer.adapter.weight.shape[1] == 2 * h.d

    def mu_adapter(gk):
        Z0 = torch.zeros(n, Lq, h.d, device=h.dev, requires_grad=True)
        x_ = M.transformer.adapter(torch.cat([x, e], dim=-1)).detach() + Z0
        for blk in core:
            a = blk.attn(blk.norm_1(x_.detach()), fc, bi, None, None).detach()
            y_ = blk.norm_2(a + x_)
            m_ = blk.mlp(blk.norm_3(y_.detach())).detach()
            x_ = blk.norm_4(m_ + y_)
        return torch.autograd.grad((x_[ar, q] * gk).sum(), Z0)[0][ar, q]
    G, A, W, B = [], [], [], []
    for k in range(10):
        with torch.enable_grad():
            xq = x[ar, q].clone().requires_grad_(True)
            st = x.clone().index_put((ar, q), xq)
            lg = torch.log_softmax(h.coda_logits(st, q[:, None])[:, 0], -1)[:, dig]
            gk = torch.autograd.grad((lg - lg.mean(-1, keepdim=True))[:, k].sum(), xq)[0].detach()
            mu = mu_adapter(gk)
            xq = x[ar, q].clone().requires_grad_(True)
            st = x.clone().index_put((ar, q), xq)
            yq = M.core_block_forward(st, e, fc, None, None, torch.tensor(0, dtype=torch.long), current_step=cfg["K0"])[0][ar, q]
            bk = torch.autograd.grad((yq * gk).sum(), xq)[0].detach()
        with torch.no_grad():
            G.append(gk); A.append(mu @ Ws); W.append(gk @ Ws); B.append(bk)
    G, A, W, B = (torch.stack(v, 1).cpu().numpy() for v in (G, A, W, B))
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    np.savez_compressed(f"{o}/adaptcov.npz", G=G, A=A, W=W, B=B, uid=np.array([p_["uid"] for p_ in items]),
                        split=np.array([p_.get("split", "") for p_ in items]), ans=np.array([p_["ans"] for p_ in items]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    results_vol.commit()
    return meta


@app.function(image=image, gpu="A100-40GB", volumes={"/cache": hf_cache, "/results": results_vol}, timeout=420)
def truthcov(items_json: str, cfg: dict, run_id: str):
    """Cheap truth-lever covectors (judgment format, q = last token; fp32; seed-0 init uid + 500000; K0 loops -> x*, exactly
    as dstar_vjp). Readout R = log p(' True') - log p(' False'). Per item: g = dR/dx_q; b = J^T g (one VJP through
    core_block_forward); w = W_s^T g (adapter state columns). Adjoint check of b against a central-FD J r (random unit r).
    Timed with cuda synchronisation per chunk: warm-up loops, g, one VJP (the 48-step resolvent costs 48 such VJPs)."""
    import numpy as np
    import torch

    T0 = time.monotonic()
    h = H("float32")
    items = json.loads(items_json)
    seqs = []
    for r in items:
        ids_ = h.tok.encode(r["prompt_B"], add_special_tokens=True)
        seqs.append(dict(uid=r["uid"] + 500000, ids=ids_, pred_pos=[len(ids_) - 1]))
    val = h.validate(seqs[:20])
    TF = [h.tok.encode(w_, add_special_tokens=False)[0] for w_ in (" True", " False")]
    n = len(items); Ws = h.model.transformer.adapter.weight[:, : h.d]
    assert h.model.transformer.adapter.weight.shape[1] == 2 * h.d
    G = np.zeros((n, h.d), np.float32); B = np.zeros((n, h.d), np.float32); W = np.zeros((n, h.d), np.float32)
    ADJ = np.full(n, np.nan, np.float32); TM = []

    def tick():
        torch.cuda.synchronize(); return time.monotonic()
    for c0 in range(0, n, cfg["chunk"]):
        ch = list(range(c0, min(n, c0 + cfg["chunk"]))); N = len(ch)
        t0 = tick()
        with torch.no_grad():
            ids, e, s = h.batch([seqs[i] for i in ch], [0] * N)
            for k in range(cfg["K0"]):
                s = h.step(e, s, k)
            x = s.clone()
        q = torch.tensor([seqs[i]["pred_pos"][0] for i in ch], device=h.dev); ar = torch.arange(N, device=h.dev)
        fc = h.model.freqs_cis[:, : e.shape[1]]
        t1 = tick()
        with torch.enable_grad():
            xq = x[ar, q].clone().requires_grad_(True)
            st = x.clone().index_put((ar, q), xq)
            lg = torch.log_softmax(h.coda_logits(st, q[:, None])[:, 0], -1)
            g = torch.autograd.grad((lg[:, TF[0]] - lg[:, TF[1]]).sum(), xq)[0].detach()
        t2 = tick()
        with torch.enable_grad():
            xq = x[ar, q].clone().requires_grad_(True)
            st = x.clone().index_put((ar, q), xq)
            out = h.model.core_block_forward(st, e, fc, None, None, torch.tensor(0, dtype=torch.long), current_step=cfg["K0"])[0][ar, q]
            b = torch.autograd.grad((out * g).sum(), xq)[0].detach()
        t3 = tick()
        with torch.no_grad():
            w = g @ Ws
            t4 = tick()
            r = torch.randn(g.shape, device=h.dev, generator=torch.Generator(device=h.dev).manual_seed(9)); r = r / r.norm(dim=-1, keepdim=True)
            ep = 0.1; sp, sm = x.clone(), x.clone(); sp[ar, q] += ep * r; sm[ar, q] -= ep * r
            jr = (h.step(e, sp, cfg["K0"])[ar, q] - h.step(e, sm, cfg["K0"])[ar, q]) / (2 * ep)
            lhs = (b * r).sum(-1); rhs = (g * jr).sum(-1)
            ADJ[ch] = ((lhs - rhs).abs() / (lhs.abs() + rhs.abs()).clamp_min(1e-12)).cpu().numpy()
        G[ch] = g.cpu().numpy(); B[ch] = b.cpu().numpy(); W[ch] = w.cpu().numpy()
        TM.append(dict(n=N, warmup_s=t1 - t0, g_s=t2 - t1, vjp_s=t3 - t2, w_s=t4 - t3))
        print(TM[-1], flush=True)
    o = f"/results/{run_id}"
    os.makedirs(o, exist_ok=True)
    np.savez_compressed(f"{o}/truthcov.npz", G=G, B=B, W=W, adjoint_rel_err=ADJ, uid=np.array([r["uid"] for r in items]))
    meta = {"validation": val, "cfg": cfg, "runtime_s": time.monotonic() - T0, "run_id": run_id, "revision": REVISION, "timings": TM}
    with open(f"{o}/meta.json", "w") as f_:
        json.dump(meta, f_, indent=1)
    with open(f"{o}/items.json", "w") as f_:
        f_.write(items_json)
    results_vol.commit()
    return meta


# ----------------------------------------------------------------------------- configs / entrypoint
COLLECT = {
    "collect_smoke": dict(per_cell=1, n_digits=(2, 3), seed=0, K=24, W0=17, seeds=[0, 1], chunk=8),
    "collect": dict(per_cell=6, n_digits=(2, 3, 4), seed=0, K=96, W0=65, seeds=[0, 1], chunk=32),
}
INTERVENE = {
    "intervene_smoke": dict(per_cell=1, n_digits=(2, 3), seed=1, K=24, T_MID=8, T_FINAL_EXTRA=4, chunk=8,
                            readout_kinds=["circle1", "radial", "random_0", "scrT1_0"], readout_alphas=[1.0], readout_normp=[False, True],
                            mid_conds=[["none", 0.0, False], ["circle1", 1.0, False]], mid_checkpoints=[9, 16, 24],
                            final_conds=[["none", 0.0, False], ["circle1", 1.0, False], ["circle1", 1.0, True]],
                            final_checkpoints=[25, 28]),
    # primary: circle1 / circle-1 at alpha 1 vs norm-matched controls; endpoint = paired change in target-vs-source log odds
    "intervene": dict(per_cell=4, n_digits=(2, 3, 4), seed=1, K=96, T_MID=24, T_FINAL_EXTRA=32, chunk=56,
                      readout_kinds=["circle1", "circle-1", "circle2", "radial", "radial@circle-1",
                                     "random_0", "random_1", "random_2", "random_0@circle-1",
                                     "pca_0", "pca_1", "pca_2", "pca_0@circle-1",
                                     "scrT1_0", "scrT1_1", "scrT1_2", "scrT-1_0", "scrT-1_1"],
                      readout_alphas=[0.5, 1.0, 2.0], readout_normp=[False, True],
                      mid_conds=[["none", 0.0, False], ["circle1", 1.0, False], ["circle-1", 1.0, False],
                                 ["radial", 1.0, False], ["scrT1_0", 1.0, False]],
                      mid_checkpoints=[25, 28, 32, 40, 56, 96],
                      final_conds=[["none", 0.0, False], ["circle1", 1.0, False], ["circle-1", 1.0, False],
                                   ["radial", 1.0, False], ["pca_0", 1.0, False], ["random_0", 1.0, False],
                                   ["scrT1_0", 1.0, False], ["circle1", 1.0, True], ["circle-1", 1.0, True]],
                      final_checkpoints=[97, 98, 100, 104, 112, 128]),
}


SWEEP = {
    "sweep_smoke": dict(per_cell=1, n_digits=(2, 3), seed=1, K=24, chunk=8, sizes=[1.0], angles_deg=[0, 90, 180, 270],
                        scr_alphas=[1.0]),
    "sweep": dict(per_cell=4, n_digits=(2, 3, 4), seed=1, K=96, chunk=56, sizes=[1.0, 2.0],
                  angles_deg=list(range(0, 360, 15)), scr_alphas=[1.0, 2.0]),
}


GRAD = {
    "grad_smoke": dict(per_cell=1, n_digits=(2, 3), seed=1, K=24, chunk=8, eps=[0.25, 2.0]),
    "grad": dict(per_cell=4, n_digits=(2, 3, 4), seed=1, K=96, chunk=28, eps=[0.25, 2.0]),
}


JAC = {
    "jac_smoke": dict(per_cell=1, n_digits=(2, 3), seed=1, K=24, T=8, m=8, eps=0.5, eps_small=0.05, alphas=[0.1, 1.0], chunk=8, block_checkpoints=[4, 8]),
    "jac": dict(per_cell=4, n_digits=(2, 3, 4), seed=1, K=96, T=32, m=60, eps=0.5, eps_small=0.05, alphas=[0.1, 1.0], chunk=56, block_checkpoints=[8, 32]),
}


ORBIT = {  # prompts come from --probs (a problems.json) filtered by --pids
    # screen of demo-set variants (make_demo_variants.py): all prompts in --probs, readout every loop over the last 64
    "screen": dict(K=512, stride_proj=64, tail=64, stride_ro=1, ro_tail_only=True, chunk=30, all_prompts=True),
    "orbit_smoke": dict(K=64, stride_proj=8, tail=32, stride_ro=1, chunk=8),
    "orbit": dict(K=1024, stride_proj=8, tail=256, stride_ro=1, chunk=24),
    # fp64 replication of cycling prompts from the same (fp32-generated) initial state, + perturbation test over the tail
    "orbit64_smoke": dict(K=16, stride_proj=8, tail=8, stride_ro=1, chunk=4, dtype="float64", init_fp32=True, perturb_eps=1e-3, val_tol=1e-3),
    "orbit64": dict(K=1024, stride_proj=8, tail=256, stride_ro=1, chunk=6, dtype="float64", init_fp32=True, perturb_eps=1e-3, val_tol=1e-3),
}
CYCSRC = {  # prompts from --probs filtered by --pids (all prompts of those problems)
    "cycsrc_smoke": dict(K0=24, T_f=6, thr=1e-4, n_damp=8, beta=0.5, m=4, m_two_step=4, eps=0.25, eps_small=0.05, n_undamped=4, push_rel=1e-3,
                         chunk=64, budget_s=600, spec_control_uids=[52]),
    # matched controls: 24A (uid 48) and 49B (uid 99) are the same problems as cycling 24B / 49A with different demos
    # run with --pids 23,24,26,44,49,50 (4 cycling problems incl. matched demos + 2 stationary control problems)
    "cycsrc": dict(K0=512, T_f=64, thr=1e-4, n_damp=200, beta=0.5, m=40, m_two_step=32, eps=0.25, eps_small=0.05, n_undamped=48,
                   push_rel=1e-3, chunk=64, budget_s=480, spec_control_uids=[48, 99, 52, 88]),
}
BIFUR = {  # --probs results/orbit-20261104-193804/problems.json --pids 49   (49B uid 99 stationary -> 49A uid 98 cycling)
    "bifur_smoke": dict(uid0=99, uid1=98, K=64, beta=0.5, n_damp=12, n_damp_cont=6, fp_tol=1e-5, m=6, m_two_step=4, eps=0.1,
                        eps_check=[0.25, 0.1, 0.05], stat_thr=2e-5, floor_thr=1e-5, amp_tol=0.02, closure_tol=1e-2,
                        axis_min_amp=1e-3, eig_res_tol=1e-2, track_ov_min=0.9, max_extend_coarse=2, max_extend_fine=1,
                        K_extend=64, n_coarse=4, n_refine=3, refine_halfwidth=1.5, push_rel=[1e-3, 1e-4], K_push=6,
                        n_twostep=1, K_hyst=8, budget_s=900, reserve_s=15, smoke_skip_axis_guard=True),
    # follow-up: zoom on the '=' flip crossing (coarse run bifur-20261105-050715: tracked eigenvalue -0.979 at a=0.667,
    # -1.010 at a=0.733; linear estimate a* ~ 0.712); more damped iterations so continuation fixed points converge
    "bifur_zoom": dict(uid0=99, uid1=98, K=512, beta=0.5, n_damp=300, n_damp_cont=300, fp_tol=1e-5, m=40, m_two_step=24, eps=0.1,
                       eps_check=[0.25, 0.1, 0.05], stat_thr=2e-5, floor_thr=1e-5, amp_tol=0.02, closure_tol=1e-2,
                       axis_min_amp=1e-3, eig_res_tol=1e-2, track_ov_min=0.9, max_extend_coarse=0, max_extend_fine=6,
                       K_extend=1024, n_coarse=0, n_refine=0, refine_halfwidth=0, push_rel=[1e-3, 1e-4], K_push=128,
                       n_twostep=4, K_hyst=256, budget_s=420, reserve_s=25, alpha_cross_prior=0.712,
                       fixed_alphas=[0.64, 0.67, 0.69, 0.70, 0.705, 0.71, 0.715, 0.72, 0.73, 0.75, 0.78]),
    # replication on the other matched pair: 24A (uid 48, stationary) -> 24B (uid 49, cycling); --pids 24
    "bifur24": dict(uid0=48, uid1=49, K=512, beta=0.5, n_damp=300, n_damp_cont=300, fp_tol=1e-5, m=40, m_two_step=24, eps=0.1,
                    eps_check=[0.25, 0.1, 0.05], stat_thr=2e-5, floor_thr=1e-5, amp_tol=0.02, closure_tol=1e-2,
                    axis_min_amp=1e-3, eig_res_tol=1e-2, track_ov_min=0.9, max_extend_coarse=6, max_extend_fine=4,
                    K_extend=512, n_coarse=16, n_refine=8, refine_halfwidth=1.5, push_rel=[1e-3, 1e-4], K_push=128,
                    n_twostep=3, K_hyst=256, budget_s=420, reserve_s=25),
    # loop-parity flicker scan in the 49 family where the 3rd answer digit's argmax switches 3 -> 4 (between a=0.75 and 0.78)
    "bifur49_flicker": dict(uid0=99, uid1=98, K=512, beta=0.5, n_damp=300, n_damp_cont=300, fp_tol=1e-5, m=16, m_two_step=8,
                            eps=0.1, eps_check=[0.25, 0.1, 0.05], stat_thr=2e-5, floor_thr=1e-5, amp_tol=0.02,
                            closure_tol=1e-2, axis_min_amp=1e-3, eig_res_tol=1e-2, track_ov_min=0.9, max_extend_coarse=0,
                            max_extend_fine=0, K_extend=0, n_coarse=0, n_refine=0, refine_halfwidth=0, push_rel=[1e-3, 1e-4],
                            K_push=16, n_twostep=0, K_hyst=64, budget_s=400, reserve_s=25, alpha_cross_prior=0.71,
                            fixed_alphas=[round(0.74 + 0.0025 * i, 4) for i in range(25)]),
    # finer flicker scan: the 3rd-digit argmax switch lies between a=0.7575 and 0.7600 (bifur49_flicker run)
    "bifur49_flicker2": dict(uid0=99, uid1=98, K=512, beta=0.5, n_damp=100, n_damp_cont=100, fp_tol=1e-5, m=8, m_two_step=8,
                             eps=0.1, eps_check=[0.25, 0.1, 0.05], stat_thr=2e-5, floor_thr=1e-5, amp_tol=0.02,
                             closure_tol=1e-2, axis_min_amp=1e-3, eig_res_tol=1e-2, track_ov_min=0.9, max_extend_coarse=0,
                             max_extend_fine=0, K_extend=0, n_coarse=0, n_refine=0, refine_halfwidth=0, push_rel=[1e-3, 1e-4],
                             K_push=8, n_twostep=0, K_hyst=64, budget_s=400, reserve_s=25, alpha_cross_prior=0.71,
                             fixed_alphas=[round(0.7575 + 0.0001 * i, 4) for i in range(26)]),
    "bifur": dict(uid0=99, uid1=98, K=512, beta=0.5, n_damp=200, n_damp_cont=120, fp_tol=1e-5, m=40, m_two_step=24, eps=0.1,
                  eps_check=[0.25, 0.1, 0.05], stat_thr=2e-5, floor_thr=1e-5, amp_tol=0.02, closure_tol=1e-2,
                  axis_min_amp=1e-3, eig_res_tol=1e-2, track_ov_min=0.9, max_extend_coarse=6, max_extend_fine=4,
                  K_extend=512, n_coarse=16, n_refine=8, refine_halfwidth=1.5, push_rel=[1e-3, 1e-4], K_push=128,
                  n_twostep=3, K_hyst=256, budget_s=420, reserve_s=25),
}
DOUBT = {  # --probs results/doubt_selection.json (from select_doubt.py); all entries used
    "doubt_smoke": dict(n_seeds=2, K=16, K_extend=8, checkpoints=[4, 8, 16], dense_until=6, stat_thr=2e-5, chunk=16, n_damp=10,
                        beta=0.5, fp_tol=1e-5, m=4, eps=0.1, eps_small=0.05, push_rel=1e-3, K_push=6, gap_every=2, budget_s=600,
                        reserve_s=15, max_prompts=4),
    "doubt": dict(n_seeds=8, K=96, K_extend=128, checkpoints=[8, 16, 32, 64, 96], dense_until=32, stat_thr=2e-5, chunk=64,
                  n_damp=200, beta=0.5, fp_tol=1e-5, m=32, eps=0.1, eps_small=0.05, push_rel=1e-3, K_push=64, gap_every=2,
                  budget_s=480, reserve_s=25),
}
_CW = [[8, 16], [16, 24], [24, 32], [32, 48], [48, 64], [64, 96], [96, 128]] + [[a, a + 32] for a in range(128, 512, 32)]
CENSUS = {  # --probs results/census_prompts_<shard>.json (make_census_prompts.py)
    "census_smoke": dict(K=40, tail=8, windows=[[8, 16], [16, 24], [24, 40]], chunk=4, budget_s=600, max_prompts=4),
    "census": dict(K=512, tail=64, windows=_CW, chunk=50, budget_s=480),
}
OWNBLOCK = {  # --probs results/ownblock_selection.json (census positions to validate)
    "ownblock_smoke": dict(K=16, n_damp=10, fp_tol=1e-5, amps=[1e-3, 1e-4], T=8, windows=[[1, 4], [4, 8]], m=4, eps=0.1, budget_s=600, max_prompts=4),
    # unbiased own-block spectrum by token type (random stationary positions); short pushes, Arnoldi is the main output
    "ownblock_random": dict(K=256, n_damp=300, fp_tol=1e-5, amps=[1e-3, 1e-4], T=8, windows=[[1, 4], [4, 8]], m=24, eps=0.1,
                            budget_s=480),
    "ownblock": dict(K=256, n_damp=300, fp_tol=1e-5, amps=[1e-3, 1e-4], T=48, windows=[[1, 8], [8, 16], [16, 32], [32, 48]],
                     m=24, eps=0.1, budget_s=480),
}
OWNSPEC = {  # --probs results/ownblock_random_selection.json (same 40 random positions as ownblock_random)
    "ownspec_smoke": dict(K=16, n_damp=10, fp_tol=1e-5, start_seeds=[4, 5], m=6, save_at=[4], eps=0.1, n_top=2, n_neg=1,
                          chunk=80, budget_s=600, reserve_val_s=60, max_prompts=4),
    # Krylov sizes 16/24/32/48/64 are leading blocks of H; start seed 4 reproduces ownblock_random's start vector
    "ownspec": dict(K=256, n_damp=300, fp_tol=1e-5, start_seeds=[4, 5, 6], m=64, save_at=[24, 32, 48], eps=0.1, n_top=4, n_neg=3,
                    chunk=80, budget_s=480, reserve_val_s=90),
}
FREEGEN = {  # --probs results/freegen_selection.json (select_freegen.py)
    "freegen_smoke": dict(budgets=[15, 16], tail=4, budget_s=600, max_prompts=2),
    "freegen": dict(budgets=[509, 510, 511, 512], tail=64, budget_s=480),
}
WEEKDAY = {  # --probs results/weekday_prompts.json (make_weekday_prompts.py); all 98 prompts
    "wk_smoke": dict(K=8, save_loops=[0, 1, 2, 4, 8], K_extend=12, stat_thr=2e-5, chunk=4, budget_s=600, max_prompts=4),
    "wk": dict(K=96, save_loops=list(range(97)), K_extend=192, stat_thr=2e-5, chunk=49, budget_s=480),
}
RELAX = {  # --probs results/relax_selection.json (select_relax.py)
    "relax_smoke": dict(K0=16, beta=0.5, n_damp=10, fp_tol=1e-5, m=6, start_seeds=[4, 5], n_top=2, start_tol=0.01, eps=0.1,
                        val_tol=0.02, mu_min=0.02, T=36, chunk=4,
                        budget_s=600, max_prompts=3, smoke_force_eligible=True),
    "relax": dict(K0=320, beta=0.5, n_damp=200, fp_tol=1e-5, m=40, start_seeds=[4, 5], n_top=4, start_tol=0.01, eps=0.1,
                  val_tol=0.02, mu_min=0.02, T=160, chunk=64,
                  budget_s=480),
}
DIGROT = {  # --probs results/digit_rotation_prompts.json (make_digit_rotation_prompts.py); all 200 prompts
    "dg_smoke": dict(K=8, ro_every=4, chunk=4, budget_s=600, max_prompts=4),
    "dg": dict(K=96, ro_every=4, chunk=100, budget_s=480),
}
SWEEPR = {  # --probs results/sweep_selection.json (select_sweep.py; predictions saved by the relax run)
    "rsweep_smoke": dict(K0=16, r_mults=[0.98, 1.02], T=40, chunk=4, budget_s=600, max_prompts=2),
    "rsweep": dict(K0=320, r_mults=[0.96, 0.98, 0.99, 1.01, 1.02, 1.04], T=192, chunk=78, budget_s=480),
}
SETTLE = {  # --probs results/settle_prompts.json (make_settle_prompts.py)
    "settle_smoke": dict(K=12, beta=0.5, n_damp=10, fp_tol=1e-5, m=6, eps=0.1, chunk=128, reserve_arnoldi_s=60, budget_s=600,
                         max_prompts=3),
    "settle": dict(K=128, beta=0.5, n_damp=200, fp_tol=1e-5, m=24, eps=0.1, chunk=128, reserve_arnoldi_s=170, budget_s=480),
}
MCQ = {  # --probs results/mcq_prompts.json (make_mcq_prompts.py); same prompts and init for both conditions
    "mcq_smoke": dict(K=12, alpha=1.0, dense=4, sparse=4, tail=4, chunk=4, total_s=540, max_prompts=4),
    "mcq_base": dict(K=256, alpha=1.0, dense=64, sparse=4, tail=64, chunk=50, total_s=540),
    "mcq_relax": dict(K=256, alpha=0.85, dense=64, sparse=4, tail=64, chunk=50, total_s=540),
    # pointer-chasing behavioural screen (results/pointer_prompts.json): readout over " 0".." 9", every loop to 128
    # carry-chain procedure experiment, stage 1 (results/carry_chain_prompts.json): bare-digit readout every loop to 96
    "carry_chain": dict(K=96, alpha=1.0, dense=96, sparse=4, tail=8, chunk=50, total_s=480, choices=[str(d) for d in range(10)]),
    "ptr_screen": dict(K=96, alpha=1.0, dense=96, sparse=4, tail=8, chunk=40, total_s=480,
                       choices=[f" {d}" for d in range(10)]),
}
MBLEND = {  # --probs results/month_prompts.json (make_month_prompts.py)
    "mblend_smoke": dict(K=6, checkpoints=[1, 2, 4, 6], chunk=200, total_s=400, max_items=6),
    "mblend": dict(K=96, checkpoints=[1, 2, 4, 8, 16, 32, 64, 96], chunk=60, total_s=540),
}
STEER = {  # --probs results/steer_prompts.json --dirs results/steer_dirs.npz (make_steer_dirs.py)
    "steer_smoke": dict(K0=8, scales=[1.0], ro_t=[1, 4], chunk=20, total_s=400, max_prompts=4),
    "steer": dict(K0=96, scales=[1.0, 2.0], ro_t=[1, 2, 4, 8, 16, 32, 47, 48], chunk=150, total_s=480),
}
TRUTH = {  # --probs results/truth_prompts.json (make_truth_prompts.py)
    "truth_smoke": dict(K=6, checkpoints=[1, 2, 4, 6], chunk=8, total_s=400, max_prompts=8),
    "truth": dict(K=64, checkpoints=[1, 2, 4, 8, 16, 32, 64], chunk=120, total_s=440),
}
_TS_CONDS = [["sham", "sham", "-", 0, 0.0],
             ["oneshot+", "oneshot", "B", 1, 1.0], ["oneshot-", "oneshot", "B", -1, 1.0],
             ["context+", "context", "B", 1, 1.0], ["context-", "context", "B", -1, 1.0],
             ["context+half", "context", "B", 1, 0.5], ["context-half", "context", "B", -1, 0.5],
             ["oneshot_rand", "oneshot", "R", 1, 1.0], ["context_rand", "context", "R", 1, 1.0]]
TSTEER = {  # --probs results/truthsteer_items.json --dirs results/truthsteer_dirs.npz
    "tsteer_smoke": dict(K0=4, Tr=3, Trec=3, ro=[0, 1, 3, 4, 6], conds=_TS_CONDS, chunk=40, total_s=400, max_items=4),
    # same items/controls at the ANSWER position with answer-site fold-specific directions (--dirs results/truthsteer_dirs_ans.npz)
    "tsteer_ans": dict(K0=64, Tr=24, Trec=24, ro=[0, 1, 2, 4, 8, 16, 24, 25, 26, 28, 32, 40, 48], conds=_TS_CONDS, chunk=180,
                       total_s=440, site="ans"),
    "tsteer": dict(K0=64, Tr=24, Trec=24, ro=[0, 1, 2, 4, 8, 16, 24, 25, 26, 28, 32, 40, 48], conds=_TS_CONDS, chunk=180, total_s=440),
}
RESPOND = {  # digits: --probs results/steer_prompts.json --dirs results/steer_dirs.npz; truth: --probs results/truthsteer_items.json --dirs results/truthsteer_dirs_ans.npz
    "respond_smoke": dict(task="truth", K0=4, N=4, T=4, doses=[0.1, 1.0], eps=0.1, total_s=360, max_items=4),
    "respond_smoke_digits": dict(task="digits", K0=4, N=4, T=4, doses=[0.1, 1.0], eps=0.1, total_s=360, max_items=4),
    "respond_digits": dict(task="digits", K0=96, N=64, T=48, doses=[0.1, 1.0], eps=0.1, total_s=360),
    "respond_truth": dict(task="truth", K0=64, N=64, T=48, doses=[0.1, 1.0], eps=0.1, total_s=360),
}
DSTAR = {  # --probs results/dstar_items.json
    "dstar_smoke": dict(K0=4, T=4, chunk=4, total_s=300, max_items=4),
    "dstar_vjp": dict(K0=64, T=48, chunk=40, total_s=400),
}
DSEVAL = {  # --probs results/dstar_eval_items.json --dirs results/dstar_pushes.npz (built by make_dstar_pushes.py)
    "dseval_smoke": dict(K0=4, ro=[0, 1, 4], chunk=40, total_s=300),
    "dseval": dict(K0=64, ro=[0, 1, 2, 4, 8, 16, 24, 32, 48], chunk=200, total_s=440),
    # cheap truth levers: --probs results/dstar_eval_items.json --dirs results/tlever_pushes_{1,2}.npz
    "tlever_smoke": dict(K0=4, ro=[0, 1, 2], chunk=220, total_s=300, offkl=True),  # production chunk size (memory check)
    "tlever_eval": dict(K0=64, ro=[0, 1, 2, 4, 8, 16, 24], chunk=220, total_s=470, offkl=True),
}
_CTRL_BASE = [["sham", "none", 0, 0.0, "force"], ["dstar+", "dstar", 1, 12.5, "force"], ["dstar-", "dstar", -1, 12.5, "force"],
              ["rand+", "rand", 1, 12.5, "force"]]
_CTRL_TF = [["sham", "none", 0, 0.0, "force"], ["dstar+_1.5625", "dstar", 1, 1.5625, "force"], ["dstar+_3.125", "dstar", 1, 3.125, "force"],
            ["dstar+_6.25", "dstar", 1, 6.25, "force"], ["pulse+", "dstar", 1, 12.5, "pulse"], ["remove+", "dstar", 1, 12.5, "remove"]]
CTRL = {  # --probs results/ctrl_rows.json --dirs results/ctrl_dirs.npz
    "ctrl_smoke": dict(formats=["tf", "rev"], conds=_CTRL_BASE, K0=4, Tr=2, ro=[0, 1, 4], chunk=40, total_s=300),
    "ctrl_formats": dict(formats=["tf", "yn", "rev", "ab"], conds=_CTRL_BASE, K0=64, Tr=24, ro=[0, 1, 2, 4, 8, 16, 24, 32, 48], chunk=160, total_s=440),
    # counterbalanced mappings (truth vs affirmation / first-option): fq, ba
    "ctrl_cb": dict(formats=["fq", "ba"], conds=_CTRL_BASE, K0=64, Tr=24, ro=[0, 1, 2, 4, 8, 16, 24, 32, 48], chunk=120, total_s=440),
    "ctrl_dose": dict(formats=["tf"], conds=_CTRL_TF, K0=64, Tr=24, ro=[0, 1, 2, 4, 8, 16, 24, 25, 26, 28, 32, 48], chunk=180, total_s=440),
}
DGLEVER = {  # train: --probs results/dglever_train.json (no --dirs); pred: --probs results/dglever_pred_items.json
    # --dirs results/dglever_pushes_L.npz (10 pushes, asserted); eval: --probs results/dglever_eval.json --dirs results/dglever_pushes.npz (32)
    "dglever_smoke": dict(mode="train", K0=4, T=3, chunk=40, total_s=300, max_items=2),
    "dglever_train": dict(mode="train", K0=96, T=48, chunk=100, total_s=420),
    "dglever_pred": dict(mode="pred", K0=96, T=48, eps=0.1, chunk=200, total_s=420),
    "dglever_eval": dict(mode="eval", K0=96, T=48, chunk=220, total_s=420),
    # adapter-pullback levers: --probs results/dglever_eval.json --dirs results/adlever_pushes.npz (L, A, W, B: 40 pushes)
    "adlever_eval": dict(mode="eval", K0=96, T=48, chunk=205, total_s=480, n_push=40),
    # overlap-matched controls: --probs results/dglever_eval.json --dirs results/ovl_pushes_{1,2}.npz (30 pushes each)
    "ovl_eval": dict(mode="eval", K0=96, T=48, chunk=210, total_s=480, n_push=30),
}
REVMECH = {  # --probs results/dglever_eval.json --dirs results/dglever_pushes.npz (push_idx: L0..L9, G0..G9, R0)
    "revmech_smoke": dict(K0=4, m=6, eps=0.1, chunk=40, total_s=300, push_idx=list(range(20)), max_items=2),
    "revmech": dict(K0=96, m=50, eps=0.1, chunk=105, total_s=440, push_idx=list(range(20)) + [30]),
}
BLOCKDECOMP = {  # --probs results/dglever_eval.json --dirs results/dglever_pushes.npz (all 32 pushes)
    "blockdecomp_smoke": dict(K0=4, eps=0.1, total_s=200, max_items=3),
    "blockdecomp": dict(K0=96, eps=0.1, total_s=330),
}
ADAPTCOV = {  # --probs results/dglever_pred_items.json (eval + train, with 'split')
    "adaptcov_smoke": dict(K0=4, max_items=3),
    "adaptcov": dict(K0=96),
}
TRUTHCOV = {  # --probs results/dstar_items.json (train + eval, as dstar_vjp)
    "truthcov_smoke": dict(K0=4, chunk=4, max_items=4),
    "truthcov": dict(K0=64, chunk=40),
}
RTRANSFER = {  # --probs results/dglever_eval.json
    "rtransfer": dict(K0=96, eps=0.1),
}
CARRY = {
    "carry_smoke": dict(per_cell=1, n_digits=(2, 3), seed=2, K=16, chunk=8),
    "carry": dict(per_cell=4, n_digits=(2, 3, 4), seed=2, K=64, chunk=56),
}


@app.local_entrypoint()
def main(mode: str = "collect_smoke", dirs: str = "", probs: str = "", pids: str = ""):
    import random

    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import sample_stratified, to_json

    run_id = f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    if mode in CENSUS or mode in OWNBLOCK or mode in OWNSPEC:  # (OWNBLOCK includes ownblock_random)
        cfg = dict(CENSUS[mode] if mode in CENSUS else OWNBLOCK[mode] if mode in OWNBLOCK else OWNSPEC[mode])
        pj = json.load(open(probs))
        if cfg.get("max_prompts"):
            pj = pj[: cfg.pop("max_prompts")]
        key = (lambda p: (p["uid"], p["val_pos"])) if mode not in CENSUS else (lambda p: p["uid"])
        assert len({key(p) for p in pj}) == len(pj)
        for p in pj:
            for k in ("ids", "pred_pos"):
                p.pop(k, None)
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "prompts", cfg)
        meta = (census if mode in CENSUS else ownblock if mode in OWNBLOCK else ownspec).remote(json.dumps(pj), cfg, run_id)
        print(json.dumps(meta, indent=1))
        return
    if mode in WEEKDAY:
        cfg = dict(WEEKDAY[mode])
        pj = json.load(open(probs))
        if cfg.get("max_prompts"):
            pj = pj[: cfg.pop("max_prompts")]
        assert len({p["uid"] for p in pj}) == len(pj)
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "prompts", cfg)
        print(json.dumps(wkcollect.remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in TRUTHCOV:
        cfg = dict(TRUTHCOV[mode])
        pj = json.load(open(probs))
        if cfg.get("max_items"):
            pj = pj[: cfg.pop("max_items")]
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "items", cfg)
        print(json.dumps(truthcov.remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in ADAPTCOV:
        cfg = dict(ADAPTCOV[mode])
        pj = json.load(open(probs))
        if cfg.get("max_items"):
            pj = pj[: cfg.pop("max_items")]
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "items", cfg)
        print(json.dumps(adaptcov.remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in BLOCKDECOMP:
        cfg = dict(BLOCKDECOMP[mode])
        pj = json.load(open(probs))
        if cfg.get("max_items"):
            pj = pj[: cfg.pop("max_items")]
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "items", cfg)
        print(json.dumps(blockdecomp.remote(json.dumps(pj), open(dirs, "rb").read(), cfg, run_id), indent=1))
        return
    if mode in RTRANSFER:
        cfg = dict(RTRANSFER[mode])
        pj = json.load(open(probs))
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "items", cfg)
        print(json.dumps(rtransfer.remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in REVMECH:
        cfg = dict(REVMECH[mode])
        pj = json.load(open(probs))
        if cfg.get("max_items"):
            pj = pj[: cfg.pop("max_items")]
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "items", cfg)
        print(json.dumps(revmech.remote(json.dumps(pj), open(dirs, "rb").read(), cfg, run_id), indent=1))
        return
    if mode in DGLEVER:
        cfg = dict(DGLEVER[mode])
        pj = json.load(open(probs))
        if cfg.get("max_items"):
            pj = pj[: cfg.pop("max_items")]
        blob = open(dirs, "rb").read() if dirs else b""
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        cfg["push_sha256"] = hashlib.sha256(blob).hexdigest(); cfg["push_file"] = dirs
        print(run_id, len(pj), "items", cfg)
        print(json.dumps(dglever.remote(json.dumps(pj), blob, cfg, run_id), indent=1))
        return
    if mode in CTRL:
        cfg = dict(CTRL[mode])
        pj = json.load(open(probs))
        blob = open(dirs, "rb").read()
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "rows", {k_: v for k_, v in cfg.items() if k_ != "conds"})
        print(json.dumps(ctrl_eval.remote(json.dumps(pj), blob, cfg, run_id), indent=1))
        return
    if mode in DSEVAL:
        cfg = dict(DSEVAL[mode])
        pj = json.load(open(probs))
        blob = open(dirs, "rb").read()
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        cfg["push_sha256"] = hashlib.sha256(blob).hexdigest(); cfg["push_file"] = dirs
        print(run_id, len(pj), "items", cfg)
        print(json.dumps(dstar_eval.remote(json.dumps(pj), blob, cfg, run_id), indent=1))
        return
    if mode in DSTAR:
        cfg = dict(DSTAR[mode])
        pj = json.load(open(probs))
        if cfg.get("max_items"):
            pj = pj[: cfg.pop("max_items")]
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "items", cfg)
        print(json.dumps(dstar_vjp.remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in RESPOND:
        cfg = dict(RESPOND[mode])
        pj = json.load(open(probs))
        blob = open(dirs, "rb").read()
        if cfg.get("max_items"):
            k = cfg.pop("max_items"); pj = pj[:k]
            Z = dict(np.load(io.BytesIO(blob)))
            if "dirB" in Z:
                buf = io.BytesIO(); np.savez(buf, **{kk: v[:k] for kk, v in Z.items()}); blob = buf.getvalue()
        if cfg["task"] == "digits":
            for p in pj:
                for k_ in ("ids", "pred_pos"):
                    p.pop(k_, None)
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "items", cfg)
        print(json.dumps(respond.remote(json.dumps(pj), blob, cfg, run_id), indent=1))
        return
    if mode in TSTEER:
        cfg = dict(TSTEER[mode])
        pj = json.load(open(probs))
        blob = open(dirs, "rb").read()
        if cfg.get("max_items"):
            k = cfg.pop("max_items")
            pj = pj[:k]
            Z = dict(np.load(io.BytesIO(blob)))
            buf = io.BytesIO(); np.savez(buf, **{kk: v[:k] for kk, v in Z.items()}); blob = buf.getvalue()
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "items", {k_: v for k_, v in cfg.items() if k_ != "conds"})
        print(json.dumps(truthsteer.remote(json.dumps(pj), blob, cfg, run_id), indent=1))
        return
    if mode in TRUTH:
        cfg = dict(TRUTH[mode])
        pj = json.load(open(probs))
        if cfg.get("max_prompts"):
            pj = pj[:4] + pj[-4:]
            cfg.pop("max_prompts")
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "statements", cfg)
        print(json.dumps(truth.remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in STEER:
        cfg = dict(STEER[mode])
        pj = json.load(open(probs))
        if cfg.get("max_prompts"):
            pj = pj[: cfg.pop("max_prompts")]
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "prompts", cfg)
        print(json.dumps(steer.remote(json.dumps(pj), open(dirs, "rb").read(), cfg, run_id), indent=1))
        return
    if mode in MBLEND:
        cfg = dict(MBLEND[mode])
        spec = json.load(open(probs))
        if cfg.get("max_items"):
            spec["items"] = spec["items"][: cfg.pop("max_items")]
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(spec["pure"]), "pure", len(spec["items"]), "items", cfg)
        print(json.dumps(mblend.remote(json.dumps(spec), cfg, run_id), indent=1))
        return
    if mode in MCQ:
        cfg = dict(MCQ[mode])
        pj = json.load(open(probs))
        if cfg.get("max_prompts"):
            pj = pj[: cfg.pop("max_prompts")]
        assert len({p["uid"] for p in pj}) == len(pj)
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "prompts", cfg)
        print(json.dumps(mcq.remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in SETTLE:
        cfg = dict(SETTLE[mode])
        pj = json.load(open(probs))
        if cfg.get("max_prompts"):
            pj = pj[: cfg.pop("max_prompts")]
        assert len({p["uid"] for p in pj}) == len(pj)
        for p in pj:
            for k in ("ids", "pred_pos"):
                p.pop(k, None)
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "prompts", cfg)
        print(json.dumps(settle.remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in DIGROT:
        cfg = dict(DIGROT[mode])
        pj = json.load(open(probs))
        if cfg.get("max_prompts"):
            pj = pj[: cfg.pop("max_prompts")]
        assert len({p["uid"] for p in pj}) == len(pj)
        for p in pj:
            for k in ("ids", "pred_pos"):
                p.pop(k, None)
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "prompts", cfg)
        print(json.dumps(dgcollect.remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in RELAX or mode in SWEEPR:
        cfg = dict(RELAX[mode] if mode in RELAX else SWEEPR[mode])
        pj = json.load(open(probs))
        if cfg.get("max_prompts"):
            pj = pj[: cfg.pop("max_prompts")]
        assert len({p["uid"] for p in pj}) == len(pj)
        for p in pj:
            for k in ("ids", "pred_pos"):
                p.pop(k, None)
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "prompts", cfg)
        print(json.dumps((relax if mode in RELAX else relaxsweep).remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in FREEGEN:
        cfg = dict(FREEGEN[mode])
        pj = json.load(open(probs))
        if cfg.get("max_prompts"):
            pj = pj[: cfg.pop("max_prompts")]
        assert len({p["uid"] for p in pj}) == len(pj)
        for p in pj:
            for k in ("ids", "pred_pos"):
                p.pop(k, None)
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "prompts", cfg)
        print(json.dumps(freegen.remote(json.dumps(pj), cfg, run_id), indent=1))
        return
    if mode in DOUBT:
        cfg = dict(DOUBT[mode])
        pj = json.load(open(probs))
        if cfg.get("max_prompts"):
            pj = [q for q in pj if q["multi"]][:2] + [q for q in pj if not q["multi"]][: cfg["max_prompts"] - 2]
        assert len({p["uid"] for p in pj}) == len(pj) and len({p["pid"] for p in pj}) == len(pj)
        for p in pj:
            for k in ("ids", "pred_pos"):
                p.pop(k, None)
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "prompts", cfg)
        meta = doubt.remote(json.dumps(pj), cfg, run_id)
        print(json.dumps(meta, indent=1))
        return
    if mode in ORBIT or mode in CYCSRC or mode in BIFUR:
        cfg = dict(ORBIT[mode] if mode in ORBIT else CYCSRC[mode] if mode in CYCSRC else BIFUR[mode])
        if cfg.pop("all_prompts", False):
            pj = json.load(open(probs))
            for p in pj:
                p.setdefault("ipos", 1)
            assert len({p["uid"] for p in pj}) == len(pj), "duplicate uids"
        else:
            want = {int(x) for x in pids.split(",")}
            pj = [p for p in json.load(open(probs)) if p["pid"] in want]
            assert {p["pid"] for p in pj} == want, f"missing pids {want - {p['pid'] for p in pj}}"
            assert len({p["uid"] for p in pj}) == len(pj), "duplicate uids"
            for w in want:
                assert sorted(p["demo"] for p in pj if p["pid"] == w) == ["A", "B"], f"pid {w} lacks both demos"
        for p in pj:
            for k in ("ids", "pred_pos"):
                p.pop(k, None)
        blob = open(dirs, "rb").read() if dirs else b""
        import hashlib
        cfg["code_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
        print(run_id, len(pj), "prompts", cfg)
        meta = (orbit.remote(json.dumps(pj), blob, cfg, run_id) if mode in ORBIT else cycsrc.remote(json.dumps(pj), cfg, run_id)
                if mode in CYCSRC else bifur.remote(json.dumps(pj), cfg, run_id))
        print(json.dumps(meta, indent=1))
        return
    if mode in COLLECT:
        cfg = dict(COLLECT[mode])
        probs = sample_stratified(n_digits_list=cfg.pop("n_digits"), per_cell=cfg.pop("per_cell"), seed=cfg.pop("seed"))
        print(run_id, len(probs), "prompts", cfg)
        meta = collect.remote(to_json(probs), cfg, run_id)
    else:
        cfg = dict(INTERVENE[mode] if mode in INTERVENE else SWEEP[mode] if mode in SWEEP else GRAD[mode] if mode in GRAD
                   else JAC[mode] if mode in JAC else CARRY[mode])
        fit = sample_stratified(n_digits_list=(2, 3, 4), per_cell=6, seed=0)
        fit_pairs = {(p.a, p.b) for p in fit}
        probs = sample_stratified(n_digits_list=cfg.pop("n_digits"), per_cell=cfg.pop("per_cell"), seed=cfg.pop("seed"))
        probs = [p for p in probs if (p.a, p.b) not in fit_pairs]
        rng = random.Random(7)
        pj = json.loads(to_json(probs))
        ipos_by_pid = {}
        for p in pj:  # same intervention position for both demo sets of a problem
            if p["pid"] not in ipos_by_pid:
                ipos_by_pid[p["pid"]] = rng.randrange(1, len(p["digit_info"]))
            p["ipos"] = ipos_by_pid[p["pid"]]
        with open(dirs, "rb") as f:
            blob = f.read()
        print(run_id, len(pj), "prompts", {k: v for k, v in cfg.items() if "kinds" not in k})
        fn = (intervene if mode in INTERVENE else sweep if mode in SWEEP else gradients if mode in GRAD
              else jacobian if mode in JAC else carry_traj)
        meta = fn.remote(json.dumps(pj), blob, cfg, run_id)
    print(json.dumps(meta, indent=1))
    print(f"fetch: modal volume ls huginn-results {run_id}")
