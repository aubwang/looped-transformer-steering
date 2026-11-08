"""CPU-only weight statistics for the read-space analysis (docs/freeloop.pdf; ctrlprobe runs). No model forward pass.
Reads tensors of Huginn-0125 (pinned revision, HF cache volume) with safetensors and computes, for the adapter's state
block W_s = adapter.weight[:, :d] (s_{k+1} = core(W_s s_k + W_e e)):
  tau(W_s) = tr(W_s) / d; the eigenvalues of its symmetric part;
  self-gains <u, W_s u> for (a) Gaussian unit u; (b) unembedding proxies u_v = normalize(w_f * E_v) for a
  random sample of tokens (E = lm_head rows, w_f = ln_f weight; the coda blocks are ignored: a first-order proxy for the
  directions the coda reads); (c) random combinations normalize(w_f * (E^T xi)), xi Gaussian centered.
Also saves all 1-D (norm) weights and the list of tensor names. Output: volume huginn-results, weights-<timestamp>/.
Run from huginn_addition/:  <venv>/bin/modal run weights_app.py
"""
from __future__ import annotations

import json
import os
import time

import modal

import intervene_app as iv
from intervene_app import MODEL_ID, REVISION, hf_cache, results_vol

app = modal.App("huginn-weights")
image = iv.image.add_local_python_source("intervene_app")


@app.function(image=image, cpu=8, memory=32768, volumes={"/cache": hf_cache, "/results": results_vol}, timeout=900)
def weights(run_id: str, n_tok: int = 8192, n_rand: int = 512, seed: int = 0):
    import glob

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    T0 = time.monotonic()
    path = snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=["*.safetensors", "*.json"])
    files = sorted(glob.glob(f"{path}/*.safetensors"))
    where = {}
    for f in files:
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                where[k] = f

    def get(k):
        with safe_open(where[k], "pt") as fh:
            return fh.get_tensor(k).float()

    names = sorted(where)
    ad = get("transformer.adapter.weight"); d = ad.shape[0]
    assert ad.shape[1] == 2 * d
    Ws = ad[:, :d].double()
    lnf = get("transformer.ln_f.weight").double()
    Ek = "lm_head.weight" if "lm_head.weight" in where else "transformer.wte.weight"
    E = get(Ek)
    g = torch.Generator().manual_seed(seed)
    unit = lambda X: X / X.norm(dim=-1, keepdim=True)
    sg = lambda U: ((U @ Ws.T) * U).sum(-1)   # <u, W_s u> row-wise
    tau = float(torch.trace(Ws) / d)
    evs = torch.linalg.eigvalsh((Ws + Ws.T) / 2).numpy()
    Z = unit(torch.randn((n_rand, d), generator=g, dtype=torch.float64))
    tok = torch.randperm(E.shape[0], generator=g)[:n_tok]
    Ue = unit(lnf * E[tok].double())
    xi = torch.randn((n_rand, E.shape[0]), generator=g, dtype=torch.float64); xi -= xi.mean(-1, keepdim=True)
    Uc = unit(lnf * (xi @ E.double()))
    res = dict(d=d, tau_Ws=tau, E_key=Ek, n_tok=n_tok, n_rand=n_rand,
               self_gain=dict(random=sg(Z).numpy(), unembed=sg(Ue).numpy(), combo=sg(Uc).numpy()))
    def ndim(k):
        with safe_open(where[k], "pt") as fh:
            return len(fh.get_slice(k).get_shape())
    norms = {k: get(k).numpy() for k in names if ndim(k) == 1}
    o = f"/results/{run_id}"; os.makedirs(o, exist_ok=True)
    np.savez_compressed(f"{o}/weights.npz", sym_eigs=evs, sg_random=res["self_gain"]["random"], sg_unembed=res["self_gain"]["unembed"],
                        sg_combo=res["self_gain"]["combo"], tok=tok.numpy(), Ws_diag=torch.diagonal(Ws).numpy(), lnf=lnf.numpy(),
                        **{"norm__" + k.replace(".", "_"): v for k, v in norms.items()})
    summ = dict(d=d, tau_Ws=tau, E_key=Ek, names=names, runtime_s=time.monotonic() - T0,
                med=dict(random=float(np.median(res["self_gain"]["random"])), unembed=float(np.median(res["self_gain"]["unembed"])),
                         combo=float(np.median(res["self_gain"]["combo"]))),
                sym_eig_range=[float(evs.min()), float(evs.max())])
    with open(f"{o}/summary.json", "w") as f_:
        json.dump(summ, f_, indent=1)
    results_vol.commit()
    return {k: v for k, v in summ.items() if k != "names"}


@app.local_entrypoint()
def main():
    run_id = f"weights-{time.strftime('%Y%m%d-%H%M%S')}"
    print(run_id)
    print(json.dumps(weights.remote(run_id), indent=1))
