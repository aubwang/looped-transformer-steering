# Late-shift screen: is there any task where the loop still changes the answer after loop 8? (Draft 1.1, 2026-11-10)

Status: design fixed before any run. Code `huginn_addition/lateshift_app.py`, analysis
`huginn_addition/analyze_lateshift.py`, local test `huginn_addition/lateshift_local_test.py`.
Budget: at most 10 GPU-minutes in total (one L32 container, one D container, A100-40GB).

## 1. Why

The matched-pair monitor experiment (`docs/monitor_results.pdf`) found no resolved difference between the looped
model L32 and its ordinary partner D. A limit of that experiment: on its problems the looped model's answer is
settled within about four loops, so the loop is nearly idle from loop 8 on. "Does looping help monitoring" is a
fair question only on tasks where the loop keeps doing work that reaches the output. This screen looks for such
tasks. If none qualifies, the monitoring question is dropped for this model pair.

## 2. Models

- L32 = `smcleish/Recurrent-Llama-3.2-train-recurrence-32` @ `a5f6f126e9d9302445346844f1d7d29d111a06eb`.
- D = `smcleish/Llama-3.2-non-recurrent-posttrained` @ `004e96ace360c1a37c08f4555227136e845087b3`.

Float32, TF32 off, the loop harness of `mon_app.py` (`harness`, `start`), random start drawn per problem from the
model's training start distribution with seed `11000000 + pid` (as in the monitor experiment).

## 3. What is measured

For every problem, at the last prompt token, after t loops for t in `T = [1,2,3,4,6,8,12,16,24,31,32,33]`:
top-5 token ids and log-probabilities over the whole vocabulary, log-probability of the correct token, entropy,
and the relative step size of the last-position state `||x_t − x_{t−1}|| / ||x_t||` (every loop 1..33). For D:
the same readout quantities from its single forward pass.

The *answer at loop t* is the whole-vocabulary argmax token. A problem is *correct at t* iff that token is the
answer token.

## 4. Task families

Every problem's answer is exactly one token. Two conventions, both asserted at build time:
- *number answers*: the prompt ends with a space that is its own token (`decode(ids[-1]) == " "`), the answer
  token is the bare decimal string (`"0"`..`"999"`);
- *word answers*: the prompt ends without a trailing space, the answer token has a leading space (`" red"`).

Each prompt is `k` worked examples of the same family (never sharing a problem with any query) followed by the
query. 80 queries per family; `pid = 100000 * (100 + family_index) + i`. All randomness from
`random.Random(20261110 * 100 + family_index)`; queries within a family pairwise distinct.

| # | Family | k | Query format (answer after the arrow) | Chance |
|---|---|---|---|---|
| 0 | `add3` (reference: known to settle early) | 4 | `mon_app.make_arith("add3")` unchanged | 0 |
| 1 | `add6` | 4 | `a + b + c + d + e + f = ` → sum; each 10..99 | 0 |
| 2 | `chain6` | 4 | `a - b + c + d - e - f = `; a in 100..500, others 10..99, random signs with at least two of each, every prefix value in 1..999 | 0 |
| 3 | `mul2add` | 4 | `a * b + c * d = `; a, c in 11..40, b, d in 3..9, value ≤ 999 | 0 |
| 4 | `paren` | 4 | `(a + b) * c - d = `; a, b in 10..60, c in 2..8, d in 10..99, value in 1..999 | 0 |
| 5 | `var4` | 3 | four assignment lines in dependency order, then `Value of <last>: `; see below | 0 |
| 6 | `var4s` | 3 | as `var4`, the four lines in a random non-identity order | 0 |
| 7 | `ptr3` | 2 | eight lines `f(u) = v` (shuffled), then `f(f(f(u0))) = ` | 1/8 |
| 8 | `swap` | 2 | three people, three coloured balls, four swaps, `Now <person> has the` → ` red` | 1/3 |
| 9 | `acc5` | 3 | `Start with 12. Add 30. Subtract 17. Multiply by 2. Add 5. Subtract 20. Result: ` | 0 |
| 10 | `count` | 3 | `List: 3 7 1 7 7 2 9 7 4 7 0 5\nHow many times does 7 appear? Answer: ` | 1/6 |
| 11 | `sort2` | 3 | `Numbers: 34 81 17 56 62 49\nSecond largest: ` | 1/6 |
| 12 | `bool` | 4 | `not ( True and False ) or ( False or not True ) is` → ` True` / ` False` | 1/2 |
| 13 | `story2` | 2 | templated two-step word problem, `Question: …\nAnswer: ` | 0 |
| 14 | `story4` | 2 | templated four-step word problem, same format | 0 |
| 15 | `gsm` | 2 | GSM8K test questions, direct answer, same format | 0 |
| 16 | `svamp` | 2 | SVAMP questions, direct answer, same format | 0 |
| 17 | `arc_chal` | 2 | ARC-Challenge, `mon_app.make_mcq` unchanged | 1/4 |

Details.
- Worked examples are separated from each other and from the query by one blank line for the multi-line families
  (5–8, 10, 11, 13–16) and by a single newline for the one-line families (1–4, 9, 12), as in `mon_app.make_arith`.
- `var4`: four distinct random lowercase letters `p, q, r, s`; lines `p = n0`, `q = p + n1`, `r = q - n2`,
  `s = r + n3` with each operator + or − at random, n0 in 10..99, increments in 10..99, every value in 1..999;
  last line `Value of s: ` (worked examples end `Value of s: 61`).
- `ptr3`: eight distinct numbers from 10..99 arranged in one random 8-cycle `f`; lines shuffled; the query starts
  at a random element; the answer is the third iterate. The correct token is one of the eight numbers.
- `swap`: names Alice, Bob, Carol; colours a random 3-subset and assignment of {red, blue, green, yellow, black,
  white}; `Alice has the red ball, Bob has the blue ball, and Carol has the green ball. Alice and Bob swap balls.
  …` (four swaps of random distinct pairs); `Now Carol has the` (worked examples end `Now Carol has the red
  ball.`). The asked person is uniform; answers approximately balanced over the three initial owners.
- `acc5`: five operations from Add n / Subtract n (n in 10..99) / Multiply by m (m in 2..3, at most two of them),
  every intermediate value in 1..999.
- `count`: twelve digits 0..9; the target digit appears c times, c uniform on 1..6.
- `sort2`: six distinct numbers in 10..99.
- `bool`: a random expression tree with exactly five operators from {and, or, not} (at least one `not`, at least
  one parenthesised group), balanced so that half the queries are True.
- `story2`/`story4`: `<Name> has n0 <things>. <Name> buys n1 more. <Name> gives n2 to a friend. … How many
  <things> does <Name> have now?` Step templates (buys/finds/receives: add; gives away/loses/sells: subtract; for
  `story4` also at most one `doubles the number`); several names and things; no pronouns; every value in 1..999.
- `gsm`: `openai/gsm8k`, config `main`, test split; keep questions whose final answer (after `####`, commas
  removed) is an integer in 0..999 and whose question is at most 110 tokens; shuffle with the family seed; the two
  worked examples are the two shortest eligible *train* questions; whole prompt at most 320 tokens.
- `svamp`: `ChilleD/SVAMP`, all splits pooled; integer answers in 0..999; question = body + question; worked
  examples the two shortest; same format and limits.
- If a dataset cannot be fetched, that family is dropped and the fact recorded; the screen proceeds.
- The build step records the dataset commit hashes, per-family token-length histograms and the total number of
  token-loops `sum_i len_i * 33`, and writes `items.json` with its SHA-256. If the projected L32 time at 14,000
  token-loops per second exceeds 330 s, n is lowered uniformly (80 → 72 → 64) until it fits.

## 5. Batching

Prompts in a batch are right-padded to a common length with the BOS id; the readout for problem i is taken at
its own last real position `len_i − 1`. Attention is causal, so padding to the right cannot change earlier
positions. Random starts are drawn at each problem's own length and zero-padded. Batches are formed within a
family in length order (batch size 32), families visited round-robin so a deadline stop leaves the families
equally covered.

Gates (recorded; a failed gate marks the run `gate_fail`):
- G1: for the first batch of each of four families of different lengths, the padded-batch logits at
  r = 4 and r = 32 agree with the model's own `forward(ids_i, input_states=start_i, num_steps=r)` on the single
  unpadded problem, centred maximum absolute difference at most 1e-3 (two problems per batch).
- G2: the same for D against its own unpadded single-problem forward.
- G3: answer-token conventions above; every prompt at most 320 tokens.
- G4: all saved arrays finite.

A family counts in the analysis if at least 48 of its problems completed in both models.

## 6. Decision rule (fixed now)

For family f let c(a, b) be the fraction of problems whose L32 answer at loop a differs from its answer at loop b.

- *usable*: `chance + 0.10 ≤ accuracy ≤ 0.90` for L32 at 32 loops and for D.
- *not flicker*: `c(31, 32) ≤ c(8, 32) / 3`.
- **QUALIFIES** if `c(8, 32) ≥ 0.15`, usable, not flicker.
- **BORDERLINE** if `0.10 ≤ c(8, 32) < 0.15`, usable, not flicker.
- **CHANGES BUT UNUSABLE** if `c(8, 32) ≥ 0.10` and one of the other two conditions fails.
- **SETTLED** otherwise.

The largest of eighteen noisy rates overstates the truth, so a QUALIFIES or BORDERLINE family is not accepted on
this run alone: it is re-collected on 300 fresh problems (same generator, next indices; about 1–3 GPU-minutes per
family) and accepted only if `c(8, 32) ≥ 0.10` with the lower end of a 95% Wilson interval above 0.07, usable and
not flicker on the fresh problems.

If no family is accepted, the monitoring question is dropped for this model pair.

Reported for every family, without a decision attached: accuracy at every loop count and for D; c(a, b) with 95%
Wilson intervals for (4,32), (8,32), (16,32), (24,32), (8,16), (31,32), (32,33); right/wrong transitions between
loops 8 and 32 with an exact McNemar p-value; top-1 minus top-2 log-probability margins at loops 8 and 32 for
changed and unchanged problems; the distribution of the settling loop (the first loop count in T after which
the answer no longer changes up to 32); median relative step size of the last-position state by loop.

## 7. What this cannot show

Eighty problems per family give a standard error of about 0.04 at a rate of 0.15. The families are small,
synthetic or short-prompt subsets; direct answers without written working. A SETTLED verdict says the answer
token is stable under these formats, not that the state has stopped moving, and not that no task exists.

## 8. Changes

Draft 1.1, before any GPU run, after the build step and the pre-run review
(`results/lateshift_items-20261110-170859/astra_prerun_review.md`):

- The build rule reduced n to 64 per family. The time projection now uses the actual right-padded batches
  (4.65M token-loops, 332 s at 14,000 per second); the limit is 345 s. Internal deadlines: 450 s for L32 and
  90 s for D from function start, 60 s reserved for container start, at most 600 GPU-seconds in total.
- Batches of 16 (four round-robin rounds), so a deadline stop leaves every family at 16, 32, 48 or 64 problems.
- Every query has its own worked examples (drawn after a fixed pool of 380 queries per family), as in the monitor
  experiment, instead of one shared set per family. `chain6` uses random signs. D is read from its final hidden
  state at each problem's last position (gate G2 checks this against its own forward).
- Gates G1/G2 compare the most padded and the unpadded problem of the first batch of the three variable-length
  families with the longest first batches and of one uniform-length family; the run fails its gate if no
  right-padded problem was compared.
- An incomplete screen cannot give the negative conclusion: if any planned family has fewer than 48 paired
  problems and no family is a candidate, the result is "screen incomplete".
- Confirmation is a separate run, not part of the 10 GPU-minutes: at most two candidate families (largest
  c(8, 32) first, ties by table order), 300 fresh problems each (indices 64..363), at most 5 GPU-minutes in total.
  Acceptance is a replication on fresh problems under the written rule, not a multiplicity-corrected claim.
- Reading the verdicts: thresholds are eligibility rules for this pair. At n = 64 a family with a true rate of
  0.10 reaches BORDERLINE or better less than half the time, so SETTLED is a stopping decision under this budget,
  not evidence against every such family. Chance levels are guessing baselines, not protection against shortcuts.
