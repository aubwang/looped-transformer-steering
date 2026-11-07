"""Pointer-chasing behavioural screen: iterated lookup in a random 10-cycle permutation f of the digits 0-9.
Each prompt: its own table "f(0)=a, f(1)=b, ..., f(9)=j." (listed in digit order), three worked demos with depths 1, 2, 3
(distinct starts, answers shown), then the query f^n(x) = ? with depth n in 1..6. f is a single 10-cycle, so the
intermediate values x, f(x), ..., f^n(x) are all distinct. Answers balanced: for each depth, 20 prompts with answers
cycling over 0..9 twice. Each depth level adds 3 tokens ("f", "(", ")"); answer = next token " d".
Usage: python -I huginn_addition/make_pointer_prompts.py results/pointer_prompts.json
"""
import json
import random
import sys


def cycle_perm(rng):
    order = list(range(10)); rng.shuffle(order)
    f = [0] * 10
    for i in range(10):
        f[order[i]] = order[(i + 1) % 10]
    return f


def apply(f, x, n):
    for _ in range(n):
        x = f[x]
    return x


def expr(x, n):
    return "f(" * n + str(x) + ")" * n


def main(out):
    rng = random.Random(20261106)
    rows, uid = [], 70000
    for n in range(1, 7):
        for k in range(20):
            target = k % 10
            while True:
                f = cycle_perm(rng)
                x = [v for v in range(10) if apply(f, v, n) == target][0]
                starts = [v for v in range(10) if v != x]
                rng.shuffle(starts)
                demos = [(starts[i], d) for i, d in enumerate((1, 2, 3))]
                break
            table = ", ".join(f"f({i})={f[i]}" for i in range(10)) + ".\n"
            demo = "".join(f"{expr(s, d)} = {apply(f, s, d)}\n" for s, d in demos)
            prompt = table + demo + f"{expr(x, n)} ="
            rows.append(dict(uid=uid, depth=n, start=x, answer_digit=target, gold=target,
                             intermediates=[apply(f, x, j) for j in range(n + 1)], f=f, prompt=prompt))
            uid += 1
    json.dump(rows, open(out, "w"), indent=1)
    print(len(rows), "prompts"); print(rows[45]["prompt"], "->", rows[45]["answer_digit"], rows[45]["intermediates"])


if __name__ == "__main__":
    main(sys.argv[1])
