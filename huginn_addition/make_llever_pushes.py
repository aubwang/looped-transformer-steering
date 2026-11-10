"""Freeze unit population directions from train/fit only. Usage: python -I make_llever_pushes.py RUN."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def nz(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    assert np.all(n > 0), 'zero direction'
    return v/n


def build(z):
    assert set(z['split']) <= {'train','fit'}, 'eval leakage'
    L = nz(nz(z['AtG']).mean(0)); G = nz(nz(z['G']).mean(0))
    fit = z['split']=='fit'; tr = z['split']=='train'
    assert fit.sum()==100 and tr.sum()==40
    assert np.array_equal(z['gradient_pid'], z['pid'][tr])
    C = np.stack([z['xq'][fit & (z['ans']==70+k)].mean(0) for k in range(10)])
    C = nz(C-C.mean(0)); R = nz(np.random.default_rng(55).normal(size=(2, L.shape[-1])))
    names = [f'{f}{k}' for f in 'LGC' for k in range(10)]+['R0','R1']
    return dict(P=np.vstack([L,G,C,R]).astype(np.float64), names=np.array(names), m_ref=np.median(z['norms'][tr]))


def main(run):
    root = Path(run); meta = json.loads((root/'meta.json').read_text())
    assert meta['status']=='complete'
    with np.load(root/'llever.npz') as z: push = build(z)
    dst = Path(__file__).resolve().parent.parent/'results'; path = dst/'llever_pushes.npz'
    np.savez(path, **push, train_code_sha256=meta['code_sha256'], T=meta['cfg']['T'])
    (dst/'llever_pushes.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    L,G,C = push['P'][:10],push['P'][10:20],push['P'][20:30]
    cos = lambda a,b: float(np.mean(np.sum(a*b,-1)))
    print(json.dumps(dict(path=str(path), cos_L_G=cos(L,G), cos_L_C=cos(L,C), cos_G_C=cos(G,C),
                         L_gram_offdiag_mean=float((L@L.T)[~np.eye(10,dtype=bool)].mean()), m_ref=float(push['m_ref']))))


if __name__=='__main__': main(sys.argv[1])
