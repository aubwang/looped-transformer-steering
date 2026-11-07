"""CPU-only Modal helper: download public benchmark files from the Hugging Face hub and return them (no GPU).
Usage: modal run huginn_addition/fetch_data_app.py --repo allenai/ai2_arc --pattern ARC-Challenge/test --out results/raw_arc
"""
import os

import modal

app = modal.App("huginn-fetch-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_transfer]")


@app.function(image=image, timeout=600)
def fetch(repo: str, pattern: str):
    from huggingface_hub import hf_hub_download, list_repo_files
    files = [f for f in list_repo_files(repo, repo_type="dataset") if pattern in f]
    out = {}
    for f in files:
        out[f] = open(hf_hub_download(repo, f, repo_type="dataset"), "rb").read()
    return out


@app.local_entrypoint()
def main(repo: str, pattern: str, out: str):
    os.makedirs(out, exist_ok=True)
    got = fetch.remote(repo, pattern)
    for f, b in got.items():
        p = os.path.join(out, f.replace("/", "__"))
        open(p, "wb").write(b)
        print(p, len(b))
