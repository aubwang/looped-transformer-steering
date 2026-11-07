"""ARC-Challenge zero-shot multiple-choice prompts for the budget-robustness experiment.
Filter: exactly 4 options labelled A-D (1-4 relabelled to A-D); prompt <= max_tokens tokens. Random sample of n (seed 0).
Format: "Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:"; the answer is read as the next token among " A".." D".
Usage: python -I huginn_addition/make_mcq_prompts.py results/raw_arc/ARC-Challenge__test-00000-of-00001.parquet results/mcq_prompts.json 100 120
"""
import json
import sys

import numpy as np
import pandas as pd
from tokenizers import Tokenizer


def main(src, out, n, max_tokens):
    tk = Tokenizer.from_file("/tmp/huginn_tok/tokenizer.json")
    df = pd.read_parquet(src)
    rows = []
    for _, r in df.iterrows():
        labels = list(r["choices"]["label"]); texts = list(r["choices"]["text"])
        if len(labels) != 4:
            continue
        if labels == ["1", "2", "3", "4"]:
            labels = ["A", "B", "C", "D"]; key = "ABCD"[int(r["answerKey"]) - 1]
        else:
            key = r["answerKey"]
        if labels != ["A", "B", "C", "D"] or key not in "ABCD":
            continue
        prompt = f"Question: {r['question']}\n" + "".join(f"{l}. {t}\n" for l, t in zip(labels, texts)) + "Answer:"
        L = len(tk.encode(prompt).ids)
        if L <= max_tokens:
            rows.append(dict(qid=r["id"], prompt=prompt, gold="ABCD".index(key), n_tokens=L))
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
    sel = [dict(rows[i], uid=60000 + k) for k, i in enumerate(pick)]
    json.dump(sel, open(out, "w"), indent=1)
    print(len(rows), "eligible;", len(sel), "selected; tokens median", int(np.median([s["n_tokens"] for s in sel])), "max", max(s["n_tokens"] for s in sel))
    print(sel[0]["prompt"])
    for l in " A", " B", " C", " D":
        print(repr(l), tk.encode(l, add_special_tokens=False).ids)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
