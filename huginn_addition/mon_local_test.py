"""Offline CPU checks for mon_app (docs/mon_spec.md A7); no Modal calls or downloads.
Usage: python -I huginn_addition/mon_local_test.py <scratch directory>.
Uses the cached Raven source and tokenizer fixture also used by obs_local_test.py.
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mon_app as mon


def main(tmp):
    os.makedirs(tmp, exist_ok=True)
    # Reuse offline fixtures, never modify the reference fixtures or existing repo files.
    source = os.environ.get("MON_TEST_FIXTURES", "/tmp/obs_test")
    for name in ("tinyL", "tok"):
        shutil.copytree(os.path.join(source, name), os.path.join(tmp, name), dirs_exist_ok=True)
    cdir = os.path.join(tmp, "tinyL")
    c = json.load(open(os.path.join(cdir, "config.json")))
    c.update(n_embd=32, n_heads=2, num_key_value_heads=1, head_dim=16, intermediate_size=64)
    json.dump(c, open(os.path.join(cdir, "config.json"), "w"))
    torch.manual_seed(17)
    tok = AutoTokenizer.from_pretrained(os.path.join(tmp, "tok"), local_files_only=True)
    ML = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(cdir, trust_remote_code=True),
                                         trust_remote_code=True, torch_dtype=torch.float32).eval()
    MD = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=16,
                                    num_attention_heads=2, num_key_value_heads=1, vocab_size=128256,
                                    rms_norm_eps=1e-5, attn_implementation="sdpa")).eval()
    with torch.no_grad():
        for M in (ML, MD):
            for p in M.parameters():
                if p.ndim > 1:
                    p.normal_(0, .05)
    with torch.no_grad():
        MD.model.norm.weight.copy_(torch.linspace(.7, 1.3, 32))
    arith = []
    for fam in mon.FAMS[:6]:
        its = mon.make_arith(fam, 25, 99)
        assert its == mon.make_arith(fam, 25, 99)
        keys = {mon.arith_key(fam, it["ops"]) for it in its}; assert len(keys) == 25
        for it in its:
            assert eval(it["prompt"].splitlines()[-1].split("=")[0]) == it["ans"]
            dkeys = []
            for line in it["prompt"].splitlines()[:4]:
                expr, ans = line.split("="); assert eval(expr) == int(ans)
                import re
                ops = list(map(int, re.findall(r"\d+", expr))); k = mon.arith_key(fam, ops)
                assert k not in keys; dkeys.append(k)
            assert len(set(dkeys)) == 4
        arith.append(its[0])
    rows = [dict(id=str(i), question=f"Pick a number {i}?", choices=["one", "two", "three", "four"],
                 labels=list("ABCD"), answerKey="A") for i in range(1200)]
    mc, counts = mon.make_mcq(rows, tok, "arc_easy", 800, 71)
    assert len(mc) == 800 and max(1+len(tok.encode(it["prompt"], add_special_tokens=False)) for it in mc) <= 160
    assert max(abs(sum(it["ans"] == c for it in mc)/len(mc)-.25) for c in "ABCD") < .06
    assert (mc, counts) == mon.make_mcq(rows, tok, "arc_easy", 800, 71)
    assert all(it["mc"]["id"] not in it["mc"]["demo_ids"] for it in mc)
    pool = {p for it in mc for p in it["mc"]["demo_ids"]}
    assert not pool.intersection(it["mc"]["id"] for it in mc)
    noisy = rows + [rows[0], dict(rows[1], id="bad", choices=["x"])]
    _, nc = mon.make_mcq(noisy, tok, "arc_easy", 3, 71)
    assert nc["four_option"] == 1201 and nc["deduplicated"] == 1200
    assert not mon.make_mcq(rows, tok, "arc_easy", 3, 71, max_tokens=2)[0]
    print("item rules, MC permutations, disjointness and token limits passed", flush=True)
    dev = torch.device("cpu")
    its = [dict(arith[0], pid=10), dict(arith[0], pid=11)]
    x = mon.start(ML, its, 9, 42, dev)
    assert torch.equal(x[1], mon.start(ML, its[1:], 9, 42, dev)[0])
    assert torch.equal(x[0], mon.start(ML, its[::-1], 9, 42, dev)[1])
    assert not torch.equal(x[0], x[1])
    cfg = dict(seed=42, n_deep=2, n_f32=3, bs_arith=2, bs_mcq=2, limit_s=100000)
    # Duplicate prompt lengths exercise a real multi-item G1 batch and singleton diagnostic.
    test_items = [dict(arith[0], pid=0, i=0), dict(arith[0], pid=1, i=1),
                  dict(arith[0], pid=2, i=2), mc[0]]
    for key, M in (("L4", ML), ("D", MD)):
        o = os.path.join(tmp, key)
        meta = mon.run_collect(key, M, tok, test_items, cfg, o, dev, lambda: None, time.monotonic())
        assert meta["status"] == "complete", meta
        assert meta["items_done"] == len(test_items)
        assert max(meta["gates"]["G4"]) <= 1e-3
        assert max(meta["gates"]["G1"] if key == "L4" else meta["gates"]["G2"]) <= 1e-4
        assert meta["items_committed"] == len(test_items) and sorted(meta["committed_pids"]) == sorted(it["pid"] for it in test_items)
        saved = json.load(open(os.path.join(o, "meta.json")))["valid_ids"]  # what the analysis decodes predictions with
        assert len(saved["num"]) == 1000 and len(saved["abcd"]) == 4 and len(set(saved["num"] + saved["abcd"])) == 1004
        assert all(tok.decode([saved["num"][i]]) == str(i) for i in (0, 7, 123, 999))
        assert [tok.decode([t]) for t in saved["abcd"]] == [" A", " B", " C", " D"]
        if key == "L4":  # the gate batch really holds two items, and both singleton comparisons ran
            assert meta["timing"]["batches"][0][:5] == ["add3", "deep", meta["timing"]["batches"][0][2], 2, 64]
            assert meta["gates"]["G1_items"] == [2, 2] and len(meta["gates"]["batch_single_32"]) == 2
            assert max(meta["gates"]["batch_single_32"]) <= 1e-4
        for fn in glob.glob(o + "/*.npz"):
            z = dict(np.load(fn)); n = len(z["pid"]); m = len(z["pid32"]); d = 32
            assert z["pid"].dtype == z["pid32"].dtype == np.int64
            if "T" in z:
                assert z["T"].dtype == np.int64
            assert z["ans_tok"].dtype == z["plen"].dtype == np.int32 and z["tok_hash"].dtype == np.int64
            loop = key != "D"; nt = len(z["T"]) if loop else 17
            expected = {}
            if loop:
                expected.update(x_last=((n, nt, d), "float16"), x_prev=((n, 2, d), "float16"),
                                pre=((n, 2, d), "float16"), post=((n, 4, d), "float16"),
                                x_last32=((m, nt, d), "float32"),
                                d_pos=((n, int(z["T"][-1])+1, mon.L_MAX), "float32"), disp_full=((n, 5, 2), "float64"),
                                n_pos=((n, len(mon.R_NPOS), mon.L_MAX), "float32"))
                for k in ("d_full", "d_last", "e_full", "e_last", "n_full", "n_last", "cos_full", "cos_last"):
                    expected[k] = ((n, int(z["T"][-1])+1), "float64")
            else:
                expected.update(h_last=((n, 17, d), "float16"), h_prev=((n, 3, d), "float16"), h_last32=((m, 17, d), "float32"))
                for k in ("dl_full", "dl_last", "nl_full", "nl_last"):
                    expected[k] = ((n, 17), "float64")
            for prefix, shape in (("", (n, nt)),) if loop else (("lens_", (n, 17)), ("ro_", (n,))):
                for k, suffix, dt in (("top_id", (5,), "int32"), ("top_lp", (5,), "float32"),
                                      ("ent", (), "float32"), ("lp_ans", (), "float32"),
                                      ("valid_lm", (), "float32"), ("opt_lp", (4,), "float32")):
                    expected[prefix+k] = (shape+suffix, dt)
            assert set(z) == set(expected) | {"pid", "pid32", "ans_tok", "plen", "tok_hash"} | ({"T"} if loop else set())
            for k, (shape, dt) in expected.items():
                assert z[k].shape == shape and str(z[k].dtype) == dt, (k, z[k].shape, z[k].dtype)
            mon.check_chunk(z, loop)
            if loop:
                for col in range(1, nt):
                    t, p = int(z["T"][col]), int(z["T"][col-1])
                    if t == p + 1:
                        delta = z["x_last32"][:, col].astype(np.float64)-z["x_last32"][:, col-1].astype(np.float64)
                        indices = [list(z["pid"]).index(pid) for pid in z["pid32"]]
                        assert np.allclose(np.linalg.norm(delta, axis=-1), z["d_last"][indices, t], atol=1e-12)

            top = z["top_id"][:, :, 0] if loop else z["ro_top_id"][:, None, 0]
            labels = top == z["ans_tok"][:, None]
            assert labels.dtype == bool
            if not loop:
                continue
            ordered = {it["pid"]: it for it in test_items}
            batch = [ordered[int(p)] for p in z["pid"]]
            IDS, ANS, _ = mon.tokenize(tok, batch)
            # Each chunk here has identical lengths (the MC family has its own chunk kind).
            for j, it in enumerate(batch):
                ids = torch.tensor([IDS[j]])
                prep, step, read = mon.harness(M, dev)
                with torch.no_grad():
                    e, fc = prep(ids); x = mon.start(M, [it], len(IDS[j]), cfg["seed"], dev); hist = {0: x}
                    for t in range(1, int(z["T"][-1])+1):
                        new = step(x, e, fc)
                        direct = float((new.double()-x.double()).norm())
                        per_pos = (new.double()-x.double())[0].norm(dim=-1).numpy(); x = new; hist[t] = x
                        assert abs(direct-z["d_full"][j, t]) < 1e-6
                        # Independent values for the other backward-looking scalars and the displacement fields.
                        assert np.allclose(per_pos, z["d_pos"][j, t, :len(IDS[j])], rtol=1e-4, atol=1e-7)
                        assert np.isnan(z["d_pos"][j, t, len(IDS[j]):]).all() and z["plen"][j] == len(IDS[j])
                        assert abs(float(x.double().norm())-z["n_full"][j, t]) < 1e-6
                        if t in mon.R_NPOS:
                            row = z["n_pos"][j, mon.R_NPOS.index(t)]
                            assert np.allclose(x[0].double().norm(dim=-1).numpy(), row[:len(IDS[j])], rtol=1e-6, atol=1e-7)
                            assert np.isnan(row[len(IDS[j]):]).all()
                        assert abs(float((x[0, -1].double()-hist[t-1][0, -1].double()).norm())-z["d_last"][j, t]) < 1e-6
                        if t >= 2:
                            u = (hist[t].double()-hist[t-1].double()).flatten(); v = (hist[t-1].double()-hist[t-2].double()).flatten()
                            assert abs(float((hist[t].double()-hist[t-2].double()).norm())-z["e_full"][j, t]) < 1e-6
                            assert abs(float(u @ v / (u.norm()*v.norm()).clamp_min(1e-300))-z["cos_full"][j, t]) < 1e-5
                        if t in (32, 64):
                            for jr, r in enumerate(mon.R_DISP):
                                if r < t:
                                    assert abs(float((x.double()-hist[r].double()).norm())-z["disp_full"][j, jr, int(t == 64)]) < 1e-6
                        if t in z["T"]:
                            col = list(z["T"]).index(t)
                            k = list(z["pid32"]).index(it["pid"])
                            assert np.allclose(x[0, -1].numpy(), z["x_last32"][k, col], atol=1e-6)
                            ref = M(ids, input_states=mon.start(M, [it], len(IDS[j]), cfg["seed"], dev), num_steps=t).logits[:, -1]
                            lp = ref.double().log_softmax(-1)
                            assert abs(float(lp[0, ANS[j]]) - z["lp_ans"][j, col]) < 2e-6
                            assert int(ref.argmax(-1)) == z["top_id"][j, col, 0]
        print(key, "G1/G2/G4, shapes/dtypes, direct trajectory and own-forward readouts passed", flush=True)
    # A wrong final hook (post-norm) must fail G2, even though output shapes are valid.
    norm_forward = MD.model.norm.forward
    def wrong_lens(x):
        return norm_forward(norm_forward(x)) if x.ndim == 2 else norm_forward(x)
    MD.model.norm.forward = wrong_lens
    try:
        its = [arith[0]]; ids, ans, valid = mon.tokenize(tok, its)
        with torch.no_grad():
            try:
                mon.d_batch(MD, torch.tensor(ids), torch.tensor(ans), {k: torch.tensor(v) for k,v in valid.items()},
                            torch.tensor(valid["abcd"]), torch.tensor([False]), {})
            except AssertionError:
                pass
            else:
                raise AssertionError("G2 accepted post-norm state")
    finally:
        MD.model.norm.forward = norm_forward
    # Stop after a completed batch, retaining a valid pending chunk.
    ticks = 0
    original = mon.time.monotonic
    def clock():
        nonlocal ticks
        ticks += 1
        return 0 if ticks <= 3 else 1000
    mon.time.monotonic = clock
    try:
        meta = mon.run_collect("D", MD, tok, test_items, dict(cfg, limit_s=100), os.path.join(tmp, "partial"),
                               dev, lambda: None, 0)
    finally:
        mon.time.monotonic = original
    assert meta["status"] == "partial" and 0 < meta["items_done"] < len(test_items), meta
    for fn in glob.glob(os.path.join(tmp, "partial", "*.npz")):
        mon.check_chunk(dict(np.load(fn)), False)
    # Looped model: the deadline passes in the MIDDLE of the second batch. That batch is discarded,
    # the first batch is flushed and committed, and the status is "partial".
    steps = 0; real_harness = mon.harness
    def counting_harness(M, dev):
        prep, step, read = real_harness(M, dev)
        def counted(x, e, fc):
            nonlocal steps
            steps += 1
            return step(x, e, fc)
        return prep, counted, read
    mon.harness = counting_harness; mon.time.monotonic = lambda: 1000 if steps >= 64 + 5 else 0
    try:
        meta = mon.run_collect("L4", ML, tok, test_items, dict(cfg, limit_s=100), os.path.join(tmp, "partial_loop"),
                               dev, lambda: None, 0)
    finally:
        mon.harness = real_harness; mon.time.monotonic = original
    assert meta["status"] == "partial" and meta["items_done"] == meta["items_committed"] == 2, meta
    (fn,) = glob.glob(os.path.join(tmp, "partial_loop", "*.npz"))
    z = dict(np.load(fn)); mon.check_chunk(z, True); assert sorted(z["pid"]) == sorted(meta["committed_pids"])
    # A G1 failure (harness disagrees with the model's own forward) is recorded and nothing is saved.
    def bad_harness(M, dev):
        prep, step, read = real_harness(M, dev)
        return prep, step, lambda x, fc: (lambda c, l: (c, l + torch.linspace(0, 1, l.shape[-1])))(*read(x, fc))
    mon.harness = bad_harness
    try:
        meta = mon.run_collect("L4", ML, tok, test_items, cfg, os.path.join(tmp, "g1_fail"), dev, lambda: None, time.monotonic())
    finally:
        mon.harness = real_harness
    assert meta["status"] == "gate_fail" and meta["error"].startswith("G1") and meta["items_committed"] == 0, meta
    assert not glob.glob(os.path.join(tmp, "g1_fail", "*.npz")) and os.path.exists(os.path.join(tmp, "g1_fail", "meta.json"))
    # A finite chunk must still fail if its saved f32/f16 pair disagrees.
    chunk = dict(np.load(glob.glob(os.path.join(tmp, "D", "*.npz"))[0]))
    broken = {k: v.copy() for k, v in chunk.items()}
    broken["h_last32"] *= 2
    try:
        mon.check_chunk(broken, False)
    except AssertionError:
        pass
    else:
        raise AssertionError("G4 accepted bad f16 rounding")
    broken = {k: v.copy() for k, v in chunk.items()}; broken["lens_ent"][0, 0] = np.nan
    try:
        mon.check_chunk(broken, False)
    except AssertionError:
        pass
    else:
        raise AssertionError("G4 accepted undefined entropy")
    # Gate rejects corrupted finite diagnostics and oversized state values.
    try:
        mon.pack_states(torch.tensor([float("inf")]), {})
    except AssertionError:
        pass
    else:
        raise AssertionError("G4 accepted overflow")
    try:
        mon.pack_states(torch.tensor([60000.]), {})
    except AssertionError:
        pass
    else:
        raise AssertionError("G4 accepted state at 6e4")
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
