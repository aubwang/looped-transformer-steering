"""CPU dry run of obs_app.run_obs on tiny random-weight copies of both architectures (no GPU, no download of weights).

Checks that every code path runs, that gate G1 (loop harness vs the model's own forward) and gate G2 (hook site and
zero push) pass, and that the output arrays are finite where they should be. It says nothing about the real models.
Needs: torch (CPU), transformers==4.53.1, modal (import only), numpy; the pinned Raven code and the tokenizer in
<tmp>/tinyL and <tmp>/tok (raven_config_minimal.py, raven_modeling_minimal.py, config.json; tokenizer files).
Usage: python -I obs_local_test.py <tmp>
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_app  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM  # noqa: E402

tmp = sys.argv[1]
torch.manual_seed(0)
cdir = os.path.join(tmp, "tinyL")
c = json.load(open(os.path.join(cdir, "config.json")))
c.update(n_embd=64, n_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=128)
json.dump(c, open(os.path.join(cdir, "config.json"), "w"), indent=1)
tok = AutoTokenizer.from_pretrained(os.path.join(tmp, "tok"))
cfgL = AutoConfig.from_pretrained(cdir, trust_remote_code=True)
ML = AutoModelForCausalLM.from_config(cfgL, trust_remote_code=True, torch_dtype=torch.float32).eval()
MD = LlamaForCausalLM(LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                                  num_key_value_heads=2, vocab_size=128256, rms_norm_eps=1e-5,
                                  attn_implementation="sdpa")).eval()
for m in (ML, MD):
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.08)
FAILS = []


def ok(c, msg):
    print(("  ok   " if c else "  FAIL ") + msg)
    if not c:
        FAILS.append(msg)


# the float64 norm patch must compute the same function as the original float32 code
xin = torch.randn(3, 7, 64, dtype=torch.float64) * 5
ref = {id(m): m(xin.float()).double() for m in (ML.transformer.ln_f, MD.model.norm)}
ML.double(); MD.double()
npatch = obs_app.patch_norms(ML) + obs_app.patch_norms(MD)
for m in (ML.transformer.ln_f, MD.model.norm):
    err = float(((m(xin) - ref[id(m)]).abs().max() / ref[id(m)].abs().max()))
    ok(err < 1e-6, f"{type(m).__name__} float64 patch vs the float32 original: observed {err:.1e}, threshold 1e-6 "
                   f"({npatch} norms patched in total)")
    m.weight.data.normal_(1, 0.1)
    xf = m(xin.float()).double(); x6 = m(xin)
    ok(float((x6 - xf).abs().max() / xf.abs().max()) < 1e-6, f"{type(m).__name__} patch with a non-unit weight")
    torch.nn.init.ones_(m.weight)
print("tiny models built (float64); Raven class", type(ML).__name__)

cfg = dict(obs_app.MODES["smoke"])
cfg.update(max_items=2, total_s=10 ** 6)
items = obs_app.make_prompts(8, 8, cfg["prompt_seed"])
assert all(it["ans"] < 1000 for it in items) and [it["nd"] for it in items[:2]] == [2, 3]


for key, M in (("L4", ML), ("D", MD)):
    o = os.path.join(tmp, "out", key); os.makedirs(o, exist_ok=True)
    t0 = time.monotonic()
    meta = obs_app.run_obs(key, M, tok, items, cfg, o, torch.device("cpu"), lambda: None, t0)
    z = np.load(os.path.join(o, "obs.npz"))
    print(key, f"{time.monotonic() - t0:.0f}s", {k: meta[k] for k in ("status", "gate_fail", "g1_max", "g2", "g4", "floors")})
    ok(meta["status"] == "complete" and not meta["gate_fail"], f"{key} status complete, no failed gate")
    ok(np.nanmax(z["c_dup_fish"]) <= 1e-20, f"{key} Part C duplicate reference rows agree")
    ok(np.isfinite(z["c_gradcheck"]).all() and np.nanmax(z["c_gradcheck"]) < 1e-2, f"{key} gradient vs finite difference")
    ok(np.isfinite(z["c_linerr"]).all() and np.nanmax(np.abs(z["c_push_euclid"])) > 0, f"{key} Part C arrays finite")
    fl = {k: v for k, v in meta["floors"].items() if "fish" in k and v is not None}
    ok(meta["numerics"]["param_dtype"] == "torch.float64" and all(v < 1e-18 for v in fl.values()),
       f"{key} float64 throughout: Fisher floors {max(fl.values()):.1e} (an fp32 cast would give ~1e-12)")
    if key == "L4":
        for k in ("a0_logp", "a0_res_all", "a1_kl", "a2_fish", "a2_dx_all", "a2_base_kl", "g1"):
            ok(np.isfinite(z[k]).all(), f"L4 {k} finite")
        ok(np.allclose(z["a2_dx_all"][..., 0], 1.0, rtol=1e-3), "L4 A2 state response is 1 at k = 0")
        ok(np.nanmax(z["a1_dup_fish"]) <= 1e-20 and np.nanmax(z["a2_dup_fish"]) <= 1e-20, "L4 duplicate rows agree (A1, A2)")
        r0 = z["a2_fish"][:, :, 1, :, 0] / z["a2_fish"][:, :, 0, :, 0]   # same direction at both sigmas, readout only
        ok(np.nanmedian(np.abs(r0 - 1)) < 0.1, f"L4 A2 directions paired across sigma (median |ratio - 1| = {np.nanmedian(np.abs(r0 - 1)):.3g})")

# a run stopped by the time budget must still save its arrays and report the stop
cfg2 = dict(cfg, total_s=1.0, reserve_s=0.0, est_item_s=0.0)
o = os.path.join(tmp, "out", "stop"); os.makedirs(o, exist_ok=True)
meta = obs_app.run_obs("L4", ML, tok, items, cfg2, o, torch.device("cpu"), lambda: None, time.monotonic())
z = np.load(os.path.join(o, "obs.npz"))
ok(meta["status"].startswith("deadline") and not z["doneA"].any() and os.path.exists(os.path.join(o, "meta.json")),
   f"deadline stop saves and reports ({meta['status']})")
# ---- analysis script: end-to-end on the tiny outputs, then rule checks on synthetic inputs
import analyze_obs as an  # noqa: E402
an.main(os.path.join(tmp, "out"))
ok(sorted(["L32", "L16", "L4", "L8"], key=lambda k: an.MDEP[k]) == ["L4", "L8", "L16", "L32"], "A ratios oriented L4 / L32")
zL = np.load(os.path.join(tmp, "out", "L4", "obs.npz")); zL = {k: zL[k] for k in zL.files}
pc = an.part_c(zL, {})
r0 = int(zL["RC"][0]); i, j, ti = np.argwhere(np.isfinite(pc[r0]["logE"]))[0]
EPS = np.array(zL["EPS"]); kl = zL["c_kl"][i, 0, j]; a = next(a for a in range(1, len(EPS) - 1) if kl[a] <= an.KL_TARGETS[ti] <= kl[a + 1])
w = (np.log(an.KL_TARGETS[ti]) - np.log(kl[a])) / (np.log(kl[a + 1]) - np.log(kl[a]))
eps_t = np.exp((1 - w) * np.log(EPS[a]) + w * np.log(EPS[a + 1]))
E_ref = r0 * (eps_t * zL["c_push_euclid"][i, 0, j] * float(zL["rho"])) ** 2
ok(abs(np.exp(pc[r0]["logE"][i, j, ti]) / E_ref - 1) < 1e-9, "Part C energy is r * ||eps u||^2 with ||u|| = c_push_euclid * rho")
n, nd, nt = 14, 6, len(an.KL_TARGETS)
base = dict(done=np.ones(n, bool), logE=np.zeros((n, nd, nt)), logE_adapter=np.zeros((n, nd, nt)))
cd = dict(base, err=np.ones((n, nd, nt)))
ca = dict(base, err=2 * np.ones((n, nd, nt)))
ca["err"][0, :2, 0] = [1.0, 9.0]; ca["err"][0, 2:4, 0] = np.nan          # prompt 0: two common dirs, errors {1, 9}
cd["err"][1, 1:4, 0] = np.nan                                            # prompt 1: one common random dir -> excluded
ca["err"][2, 5, 0] = np.nan                                              # prompt 2: one gradient sign missing -> excluded
cc = an.compare_C(ca, cd, nd)[f"random KL={an.KL_TARGETS[0]:g}"]
ok(abs(cc["per_prompt"][0] - np.log(5.0)) < 1e-12, "Part C per-prompt value is the log of the median error (5), not of the geometric midpoint (3)")
ok(cc["per_prompt"][1] is None and cc["n_common_dirs"][1] == 1, "Part C needs >= 2 common random directions")
ok(an.compare_C(ca, cd, nd)[f"gradient KL={an.KL_TARGETS[0]:g}"]["per_prompt"][2] is None, "Part C needs both gradient signs")
ok(cc["log_linerr_ratio"]["n"] == 13 and abs(cc["log_linerr_ratio"]["median"] - np.log(2)) < 1e-12, "Part C median over prompts")
ok(an.compare_C(ca, cd, nd, an.SMOKE_UIDS)[f"random KL={an.KL_TARGETS[0]:g}"]["log_linerr_ratio"]["n"] == 12, "smoke prompts excluded on request")
# ---- part 2 (docs/oversight_hard.md): candidate pool, screen, selection rule, protocol on hard prompts
cands = obs_app.make_hard_candidates(64, 2026_11_08)
ok(len(cands) == 256 and len({c["prompt"].splitlines()[-1] for c in cands}) == 256 and max(c["ans"] for c in cands) < 1000,
   "candidate pool: 256 distinct queries, answers < 1000")
ok(all(eval(c["prompt"].splitlines()[-1].rstrip(" =").replace(" ", "")) == c["ans"] for c in cands), "candidate answers correct")
ok(all(eval(line.split("=")[0]) == int(line.split("=")[1]) for c in cands for line in c["prompt"].splitlines()[:4]),
   "demonstration answers correct")
sub = [c for fam in obs_app.FAM_ORDER for c in cands if c["fam"] == fam][::21][:12]          # a few of each family
for key, Mx in (("L4", ML), ("D", MD)):
    o = os.path.join(tmp, "out", "screen_" + key); os.makedirs(o, exist_ok=True)
    cfg_s = dict(obs_app.MODES["screen"], total_s=10 ** 6)
    ms = obs_app.run_screen(key, Mx, tok, sub, cfg_s, o, torch.device("cpu"), lambda: None, time.monotonic())
    P = np.array(ms["p"])
    ok(ms["status"] == "complete" and np.isfinite(P).all() and P.shape == (len(sub), 3 if key == "L4" else 1),
       f"screen {key}: complete, finite p, depths {ms['depths']}")
    if key == "L4":   # batched model forward with per-candidate starts equals a single-candidate call
        i0 = 0; ids = torch.tensor([[obs_app.BOS] + tok.encode(sub[i0]["prompt"], add_special_tokens=False)])
        g = torch.Generator(device="cpu").manual_seed(cfg_s["seed"] + 500_000 + sub[i0]["cid"])
        STD = float(ML.config.init_values["std"]); x = torch.empty((1, ids.shape[1], 64))
        torch.nn.init.trunc_normal_(x, mean=0.0, std=STD, a=-3 * STD, b=3 * STD, generator=g)
        with torch.no_grad():
            lp = torch.log_softmax(ML(ids, input_states=x.double(), num_steps=64).logits[0, -1].double(), -1)
        ans_t = tok.encode(str(sub[i0]["ans"]), add_special_tokens=False)[0]
        ok(abs(float(lp[ans_t].exp()) - P[i0, ms["depths"].index(64)]) < 1e-9, "screen batch result equals a single call")


def fake_res(pattern):
    """pattern[i] = (correct_L4, correct_L32, correct_D, min p) -> screen results for select_hard"""
    n = len(pattern); res = {}
    for j, k in enumerate(("D", "L32", "L4")):
        jj = {"L4": 0, "L32": 1, "D": 2}[k]
        p = [[q[3]] * (1 if k == "D" else 3) for q in pattern]
        top1 = [[(5 if q[jj] else 6)] * (1 if k == "D" else 3) for q in pattern]
        res[k] = dict(p=p, top1=top1, ans=[5] * n, depths=[0] if k == "D" else [4, 32, 64])
    return res


fams = [c["fam"] for c in cands]
pat = [(1, 1, 1, 0.99)] * 256
for i in range(0, 256, 5):
    pat[i] = (1, 0, 1, 0.4)                  # W
for i in range(1, 256, 7):
    pat[i] = (1, 1, 1, 0.5)                  # U
sel, cnt = obs_app.select_hard(cands, fake_res(pat))
ok([it["stratum"] for it in sel] == ["W", "U"] * 8 and [it["uid"] for it in sel] == list(range(16)), "selection: 8 W + 8 U, alternating")
ok([it["fam"] for it in sel if it["stratum"] == "W"][:4] == list(obs_app.FAM_ORDER), "selection: round-robin over families")
pat2 = [(1, 1, 1, 0.99)] * 256
for i in (3, 70):
    pat2[i] = (0, 0, 0, 0.1)                 # only 2 W
for i in range(2, 256, 9):
    pat2[i] = (1, 1, 1, 0.8)
sel2, cnt2 = obs_app.select_hard(cands, fake_res(pat2))
ok(cnt2["selected_W"] == 2 and cnt2["selected_U"] == 14 and len(sel2) == 16, f"selection fills from U when W is short ({cnt2})")
pat3 = [(1, 1, 1, 0.99)] * 256
pat3[10] = (1, 1, 0, 0.3)
sel3, cnt3 = obs_app.select_hard(cands, fake_res(pat3))
ok(cnt3["selected_W"] == 1 and cnt3["selected_E"] == 15 and len(sel3) == 16, "selection fills from E when W and U are short")

# the full protocol on two hard prompts of different families and lengths, then the stratified analysis
hard = [dict(sel[0], uid=0), dict(next(it for it in sel if it["stratum"] == "U" and it["fam"] == "add3"), uid=1)]
hd = os.path.join(tmp, "out_hard"); os.makedirs(hd, exist_ok=True)
json.dump(hard, open(os.path.join(hd, "prompts.json"), "w"))
cfg_h = dict(obs_app.MODES["hard"], total_s=10 ** 6)
for key, Mx in (("L4", ML), ("D", MD)):
    o = os.path.join(hd, key); os.makedirs(o, exist_ok=True)
    mh = obs_app.run_obs(key, Mx, tok, hard, cfg_h, o, torch.device("cpu"), lambda: None, time.monotonic())
    ok(mh["status"] == "complete" and not mh["gate_fail"], f"hard prompts ({hard[0]['fam']}, {hard[1]['fam']}; lengths "
       f"{[len(tok.encode(h['prompt'])) for h in hard]}): {key} complete")
an.main(hd)
Sh = json.load(open(os.path.join(hd, "summary.json")))
ok(list(Sh["subsets"]) == ["all", "stratum_U", "stratum_W"] and "L4 t64" in Sh["correctness"], "hard analysis: strata, no smoke subset")
# ---- fixes after Astra's pre-run review
ok(all(c["prompt"].splitlines()[-1] not in [ln.split("=")[0] + "= " for ln in c["prompt"].splitlines()[:4]] for c in cands),
   "no demonstration repeats its query")
row = an.correct_row(np.array([5, -1, 7]), np.array([5, 5, 5]), np.array([True, False, True]), np.log([0.9, 0.5, 0.2]),
                     [dict(uid=0, stratum="W", screen={"D": dict(correct=False, top1=7)}),
                      dict(uid=1, stratum="W", screen={"D": dict(correct=False, top1=9)}),
                      dict(uid=2, stratum="U", screen={"D": dict(correct=False, top1=7)})], ["U", "W"], "D")
ok(row["all"] == 0.5 and row["per_prompt"] == [1, None, 0] and row["n_measured"] == 2 and row["W"] == 1.0
   and row["correctness_agrees_with_screen"] == 1 and row["same_greedy_token_as_screen"] == 1,
   f"unmeasured prompts are None and leave the denominators ({row['all']}, {row['per_prompt']})")
fa = dict(A=np.ones(4), Aabs=np.ones(4), done=np.ones(4, bool), lowres=np.array([True, True, False, False]),
          valid_A=np.ones(4, bool), curve=np.ones((4, 32)))
cA = an.compare_A({64: fa}, {64: dict(fa, lowres=np.ones(4, bool))}, 64, 64, exclude=(2, 3))
ok(cA["excluded"]["not_lowres_a"] == 0 and cA["relative"]["n"] == 2, "exclusion counts refer to the requested subset")
# provenance: a fake screen directory, a verified selection, and a tampered prompts file
sd = os.path.join(tmp, "out_screen_prov"); os.makedirs(sd, exist_ok=True)
import hashlib  # noqa: E402
json.dump(cands, open(os.path.join(sd, "candidates.json"), "w"), indent=1)
json.dump(fake_res(pat), open(os.path.join(sd, "screen_results.json"), "w"))
selp, _ = obs_app.select_hard(cands, json.load(open(os.path.join(sd, "screen_results.json"))))
json.dump(selp, open(os.path.join(sd, "hard_prompts.json"), "w"), indent=1)
shaf = lambda fn: hashlib.sha256(open(os.path.join(sd, fn), "rb").read()).hexdigest()[:16]
json.dump(dict(run_id="fake", cfg=dict(n_per_family=64, cand_seed=2026_11_08), code_sha256="x",
               candidates_sha256=shaf("candidates.json"), screen_results_sha256=shaf("screen_results.json"),
               hard_prompts_sha256=shaf("hard_prompts.json")), open(os.path.join(sd, "screen_manifest.json"), "w"))
itv, _ = obs_app.verify_selection(os.path.join(sd, "hard_prompts.json"))
ok(len(itv) == 16, "selection re-derived from the screen artifacts")
bad = json.load(open(os.path.join(sd, "hard_prompts.json"))); bad[3]["prompt"] = bad[3]["prompt"].replace("=", "= ", 1)
json.dump(bad, open(os.path.join(sd, "tampered.json"), "w"))
try:
    obs_app.verify_selection(os.path.join(sd, "tampered.json")); ok(False, "tampered prompts rejected")
except AssertionError as ex:
    ok(True, f"tampered prompts rejected ({ex})")
print("LOCAL TEST PASSED" if not FAILS else f"LOCAL TEST FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
