"""Addition prompt generation with per-column carry annotations.

Answer digits are indexed most-significant first (the order the model emits them).

Column types (classic carry-lookahead terms), for column i with operand digits a_i, b_i:
  G  generate:  a_i + b_i >= 10   (emits a carry regardless of carry-in)
  P  propagate: a_i + b_i == 9    (emits a carry iff a carry comes in)
  K  kill:      a_i + b_i <= 8    (never emits a carry)

For each answer digit (column c, counted from the right, 0 = units) we record:
  - carry_in:  whether a carry actually flows into column c
  - chain_in:  length of the contiguous run of carry-*producing* columns directly below c
               (counts independent generates too: 55+55 -> 2)
  - dep_len:   how many lower columns must be inspected to know the carry into c:
               dep(0) = 0; dep(c) = 1 + (dep(c-1) if column c-1 is P else 0).
               This is the true causal dependency length (only P columns pass information through).
  - ctype:     G/P/K type of column c itself ("-" for the overflow column)

The first emitted token also decides the answer length (overflow); that decision is the
carry into column n (n = operand digits), recorded as problem-level `overflow` / `overflow_dep`.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

DEMO_SETS = {
    "A": "17 + 25 = 42\n348 + 571 = 919\n6054 + 2389 = 8443\n",
    "B": "63 + 29 = 92\n705 + 186 = 891\n2471 + 3958 = 6429\n",
}


@dataclass
class Problem:
    pid: int             # problem id (shared across demo sets)
    uid: int             # unique prompt id (problem x demo set)
    family: str          # stratification family: "carries" or "dep"
    demo: str
    a: int
    b: int
    c: int
    n_digits: int
    n_carries: int
    max_chain: int
    max_dep: int
    overflow: int
    overflow_dep: int
    prompt: str          # text up to and including " ="
    answer: str          # " " + str(c)
    digit_info: list     # per answer digit (MSB first)


def column_types(a: int, b: int, width: int) -> list[str]:
    out = []
    for col in range(width):
        s = (a // 10**col) % 10 + (b // 10**col) % 10
        out.append("G" if s >= 10 else ("P" if s == 9 else "K"))
    return out


def carry_profile(a: int, b: int):
    """Per column 0..len(c)-1: carry_in, chain_in, dep_len; plus carry_out list."""
    c = a + b
    width = len(str(c))
    n = max(len(str(a)), len(str(b)))
    ctypes = column_types(a, b, n)
    carry_in, chain_in, dep, carry_out = [], [], [], []
    carry, run = 0, 0
    for col in range(width):
        carry_in.append(carry)
        chain_in.append(run if carry else 0)
        if col == 0:
            dep.append(0)
        else:
            dep.append(1 + (dep[col - 1] if ctypes[col - 1] == "P" else 0))
        da, db = (a // 10**col) % 10, (b // 10**col) % 10
        carry = 1 if da + db + carry >= 10 else 0
        carry_out.append(carry)
        run = run + 1 if carry else 0
    return carry_in, chain_in, dep, carry_out, ctypes


def make_problem(pid: int, uid: int, family: str, demo: str, a: int, b: int) -> Problem:
    c = a + b
    n = max(len(str(a)), len(str(b)))
    carry_in, chain_in, dep, carry_out, ctypes = carry_profile(a, b)
    width = len(str(c))
    # dependency for the overflow decision = carry into column n
    ov_dep = 1 + (dep[n - 1] if ctypes[n - 1] == "P" else 0)
    digits = []
    for j, ch in enumerate(str(c)):
        col = width - 1 - j
        digits.append({
            "col": col, "carry_in": int(carry_in[col]), "chain_in": chain_in[col], "dep_len": dep[col],
            "ctype": ctypes[col] if col < n else "-", "digit": int(ch),
        })
    return Problem(
        pid=pid, uid=uid, family=family, demo=demo, a=a, b=b, c=c, n_digits=n,
        n_carries=sum(carry_out[:n]), max_chain=max(chain_in + [0]), max_dep=max(dep + [ov_dep]),
        overflow=int(width > n), overflow_dep=ov_dep,
        prompt=f"{DEMO_SETS[demo]}{a} + {b} =", answer=f" {c}", digit_info=digits,
    )


def _max_dep(a: int, b: int) -> int:
    n = max(len(str(a)), len(str(b)))
    _, _, dep, _, ctypes = carry_profile(a, b)
    return max(dep + [1 + (dep[n - 1] if ctypes[n - 1] == "P" else 0)])


def sample_stratified(n_digits_list=(2, 3, 4), per_cell=6, seed=0, demos=("A", "B")) -> list[Problem]:
    """Two stratification families over equal-digit operands:
      carries: cells = (n_digits, number of carries in 0..n)
      dep:     cells = (n_digits, max dependency length in 1..n+1)  -- forces long propagate chains
    Each problem is rendered once per demo set (same pid, different uid).
    """
    rng = random.Random(seed)
    pairs, seen = [], set()

    def draw(n, pred, fam):
        lo, hi = 10 ** (n - 1), 10**n - 1
        got, tries = 0, 0
        while got < per_cell and tries < 500_000:
            tries += 1
            a = rng.randint(lo, hi)
            # bias toward propagate columns for the dep family so long chains are reachable
            if fam == "dep" and rng.random() < 0.7:
                b_digits = []
                for col in range(n):
                    da = (a // 10**col) % 10
                    r = rng.random()
                    db = (9 - da) if r < 0.6 else rng.randint(0, 9)
                    b_digits.append(db)
                b = sum(d * 10**i for i, d in enumerate(b_digits))
                if b < lo:
                    continue
            else:
                b = rng.randint(lo, hi)
            if (a, b) in seen or not pred(a, b):
                continue
            seen.add((a, b))
            pairs.append((fam, a, b))
            got += 1

    for n in n_digits_list:
        for target in range(0, n + 1):
            draw(n, lambda a, b, t=target: sum(carry_profile(a, b)[3][:n]) == t, "carries")
        for target in range(1, n + 2):
            draw(n, lambda a, b, t=target: _max_dep(a, b) == t, "dep")

    problems, uid = [], 0
    for pid, (fam, a, b) in enumerate(pairs):
        for demo in demos:
            problems.append(make_problem(pid, uid, fam, demo, a, b))
            uid += 1
    return problems


def to_json(problems: list[Problem]) -> str:
    return json.dumps([asdict(p) for p in problems])


if __name__ == "__main__":
    from collections import Counter

    ps = sample_stratified()
    print(len(ps), "prompts")
    print(Counter((p.family, p.n_digits, p.n_carries if p.family == "carries" else p.max_dep) for p in ps if p.demo == "A"))
    for a, b in [(55, 55), (199, 1), (36, 59), (4999, 5001), (1234, 4321)]:
        p = make_problem(0, 0, "x", "A", a, b)
        print(a, "+", b, "=", p.c, "max_dep", p.max_dep, "ov_dep", p.overflow_dep,
              [(d["col"], d["ctype"], d["carry_in"], d["chain_in"], d["dep_len"]) for d in p.digit_info])
