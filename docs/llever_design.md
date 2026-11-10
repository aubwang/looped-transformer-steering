# Steering-lever replication on Recurrent-Llama L32 (Draft 1.3, 2026-11-10)

Status: design fixed before any run. Code `huginn_addition/llever_app.py`, push builder
`huginn_addition/make_llever_pushes.py`, analysis `huginn_addition/analyze_llever.py`, local test
`huginn_addition/llever_local_test.py`. Budget: about 20 GPU-minutes, hard cap 30, A100-40GB.

## 1. Why

The strongest result of the Huginn study (`huginn_addition/WRITEUP.md`, runs `dglever_*-20261106`): a persistent
push `P` added to the loop state at the answer position every loop changes the readout as the *linear response*
of the loop predicts,

    Δz ≈ G · A · P,     A = Σ_{t=1..T} J^t,

with `J` the Jacobian of one loop at the settled state and `G` the gradient of the readout. Directions built from
the transposed sum, `L_k ∝ Aᵀ G_k` ("loop-aware levers"), raised the target answer by +1.78 at push norm 3, while
the immediate readout gradient `G_k` did nothing (−0.05); the item's own operator predicted the 10 × 10 response
matrix with median correlation 0.978, and the prediction without the loop (`G · P`) had correlation −0.59.

That was one model. This run repeats it on the looped model of the matched-pair work, with the same analysis.
If the linear prediction fails here, the Huginn result is model-specific.

## 2. Model, precision, state

L32 = `smcleish/Recurrent-Llama-3.2-train-recurrence-32` @ `a5f6f126e9d9302445346844f1d7d29d111a06eb`.
One loop is `x ← core_block(adapter([x, e]))`, `e` = prelude output (harness of `mon_app.py`); readout
`logits = lm_head(ln_f(coda(x)))`.

Float64 throughout (weights cast to float64, both RMSNorm classes patched as in `obs_app.patch_norms`), TF32 off.
Gate G1 compares the harness with the model's own forward (r = 4, 32) to 1e-6 centred logits in float64.

Base state: start drawn as in the monitor experiment (seed `11000000 + pid`, float32 draw cast to float64),
`K0 = 64` loops, `x* = x_64`. In the monitor data the last-position relative step of L32 is 5e-5 at loop 32 and
at the float32 floor (3e-6) by loop 64; the per-loop decay is about 0.8. `T = 48` forced loops follow.

Only the last position `q` is pushed and differentiated. Attention is causal and the sequence ends at `q`, so a
change at `q` reaches no other position: the d = 2048 state at `q` is a closed system given the others.

## 3. Problems

Ten answer classes that are single tokens: the sums 70..79. Class `u` = answer `70 + u`.
Two contexts (different worked examples), as in the Huginn design:
- context A: `x = 20 + a`, `y = 70 + u − x`; context B: `x = 40 + a`, `y = 70 + u − x`; `a` in 1..9.
- prompt = four fixed worked examples of the context (two-digit additions with sums outside 70..79, operands
  disjoint from every query) + `x + y = ` (trailing space is its own token; answer token `"7u"`).
All prompts have the same token length (asserted).

Cells `(ctx, a, u)`: 2 × 9 × 10 = 180. For each `(u, ctx)` the nine values of `a` are shuffled with seed
20261110 and split 2 eval / 2 train / 5 fit. So 40 eval, 40 train, 100 fit problems, all distinct.
Order within eval and within train: index `j = rep·20 + u·2 + ctx` (so `j xor 1` is the same class in the other
context and `(j + 2) mod 40` is a different class).

Readout: `z_k = logit(token 7k) − mean_j logit(token 7j)`, k = 0..9 (centred logits over the ten class tokens).

## 4. Runs

Every run first computes `x*` for its problems and records: last-step relative drift at `q`, `||x*_q||`, the
answer at loops 32 and 64 (whole-vocabulary argmax), `z`.

- **train** (40 train + 100 fit problems). For the train problems: `G` (40 × 10 × d), the gradient of `z_k` with
  respect to `x_q` at `x*` (autograd through the coda); `AtG` = `Aᵀ G_k` by T vector–Jacobian products through
  one loop with respect to `x_q` (other positions held at `x*`); `tail = ||last term|| / ||sum||`. For train and
  fit problems: `x*_q` (for the centroid directions) and `||x*_q||`.
- **push file** (local, `make_llever_pushes.py`), frozen on train/fit data only, unit vectors:
  `L_k = normalise(mean_i normalise(AtG_ik))`, `G_k = normalise(mean_i normalise(G_ik))`,
  `C_k = normalise(c_k − mean_j c_j)` with `c_k` the mean `x*_q` of fit problems of class k, `R_0, R_1` seeded
  random unit vectors. Order `L0..L9, G0..G9, C0..C9, R0, R1`. Also stores `m_ref` = median `||x*_q||` over train
  problems, and the SHA-256 of the file is recorded by later runs.
- **pred** (40 eval + 40 train problems; pushes `L0..L9` as unit vectors). `G` as above and `A L_k` by T
  Jacobian–vector products (central finite differences in float64, step `1e-6 · ||x*_q||`, direction
  renormalised each step as in the Huginn code); `tail`.
- **eval** (40 eval problems, two shards of 20). Rows: one sham (no push) + 32 pushes × 3 doses, dose
  `ρ · m_ref` with ρ in {0.01, 0.03, 0.09}. Each row: T loops of `x ← F(x + P at q)`; `z` read after
  t = 1, 2, 4, 8, 16, 48 forced loops; last-step relative drift; endpoint displacement at `q` relative to
  `||x*_q||`. Response `Δz = z(push) − z(sham)` at the same t.

**smoke**: 2 train, 2 eval problems, K0 = 64, T = 48, all three stages end to end with 4 pushes; reports G1,
tails, the adjoint check and rows per second. Rule fixed now: if the smoke median tail exceeds 1e-4, T is raised
to 96 for all stages; if the projected total exceeds 26 GPU-minutes, the C and R pushes are run at ρ = 0.03 only.

## 5. Analysis (fixed now; primary dose ρ = 0.03, primary time t = T, the end of the forced run)

Gates: all arrays finite; G1; Neumann tails median < 1e-3 and 95th percentile < 1e-2 (train and pred); adjoint
consistency on train problems, `(Aᵀ G_k)·L_m` against `G_k·(A L_m)`, median relative error < 0.05; at least 36 of
40 eval problems answered correctly by the sham row at the end; sham last-step drift median < 1e-6.

Per eval problem j, bootstrap over the 40 problems (5,000 resamples, 95% intervals):

- **P1, efficacy.** Target effect of family F in {L, G, C}: mean over k of `Δz_k` under push `F_k`.
  Differences L − G and L − C.
- **P3, linear prediction.** Observed 10 × 10 matrix `Δz[k, m]` (push `L_k`, class m) against
  - FULL `ρ m_ref · G_j (A_j L_k)`, the problem's own operator;
  - AVERAGED: `A` averaged over the 40 train problems;
  - SWAPPED: the operator of eval problem `(j + 2) mod 40` (other class);
  - SWAPPED_SAME_CLASS: the operator of `j xor 1` (same class, other context);
  - IDENTITY `ρ m_ref · G_j L_k` (no loop).
  Score: Pearson correlation of the flattened matrices per problem; also least-squares slope and normalised error.
- **Time course** (new, descriptive): `Δz` target effect of L and G pushes at t = 1, 2, 4, 8, 16, 48, and the
  fraction of (problem, k) pairs whose L-push target effect is negative at t = 1 and positive at t = T (the
  "first-step reversal" seen in Huginn).
- **Dose check**: the same P1 and P3 at ρ = 0.01 and 0.09; ratio of target effects across doses against the dose
  ratio.

Criteria:
- R1: interval of L − G target effect entirely above 0, and interval of the L target effect itself above 0.
- R2: median FULL correlation ≥ 0.90 and interval of FULL − IDENTITY entirely above 0.
- R3 (secondary): intervals of FULL − SWAPPED and FULL − AVERAGED entirely above 0.

Verdict (only if the gates pass; otherwise INCONCLUSIVE):
- **HUGINN PATTERN REPLICATES**: R1 and R2.
- **RESPONSE SHAPE PREDICTED, NO DEMONSTRATED LEVER ADVANTAGE**: R2 but not R1.
- **PATTERN NOT REPLICATED**: R2 fails.

What the criteria do and do not test. R2 is the Huginn pattern (the loop-aware prediction matches the shape of
the response and beats the loop-free one). It is a correlation, so it does not test the size of the response:
quantitative agreement is read from the reported slope and normalised error (Huginn: 0.95 and 0.22). R2 can fail
while the linear response is exact, if the loop-free prediction is equally good (for example `J` close to a
positive multiple of the identity); that case is recognised by a high IDENTITY correlation and reported as such,
not as a failure of linear response. Failing R1 means no advantage of L over G was demonstrated, not that G
steers.

## 6. Limits

One task (two-digit sums 70..79), population directions, one position, a state reached after 64 loops rather
than the 32 the model was trained around (answers at 32 and 64 are compared). R1 compares directions of equal
norm, not equal overlap with L (the Huginn overlap control is not repeated). The linear prediction uses the
Jacobian at `x*` with the other positions frozen, while the forced runs let them evolve; base drift at loop 64
and the sham displacement over the forced run are reported, but they do not bound that error.

## 7. Changes

Draft 1.1 (before any run): the primary time is the end of the forced run, t = T (48, or 96 if the smoke rule
raises T), since the linear prediction `G · A · P` is computed for that T only; with T = 96 the readout at 96 is
added to the time course. Rows are processed in batches of 128 (base states 32).

Draft 1.2 (before any run, after the pre-run review `results/llever_prerun_review.md`): verdict labels renamed
and their scope stated (they certify the Huginn correlation and relative-efficacy pattern, not quantitative
linear response); R1 additionally requires L's own effect to be positive. Container limits: function timeout
450 s, internal deadlines 300 s (train), 340 s (pred), 390 s (each eval shard); the smoke run times production-sized
batches (128 rows) and reports peak memory, and a stage whose projection does not fit its deadline is not
launched as is. The rotary phases are computed in float64 (the model's own code computes them in float32); both
sides of G1 use the patch.

Draft 1.3 (after the smoke run `llever_smoke-20261110-171809`, before any production run): the smoke gates passed
(G1 6e-14, adjoint 8e-10, tails 3e-16, so T stays 48); measured cost 3.6-3.8 ms per row-loop in batches of 128,
peak memory 19.6 GB. One eval shard of 20 problems projected to 355 s, above its limit, so the evaluation runs as
four shards of ten problems (about 180 s each, internal deadline 250 s) with the full dose design kept for all
push families. Projected total about 16.5 GPU-minutes; ceiling by internal deadlines 27.3.

## 8. Exploratory follow-up E1: what repetition adds (written before its run, after the main results)

Suggested by the results review (`results/followup_results_review.md`). The main run showed that the mean
target effect is present after one forced loop while the whole response matrix still changes by a quarter
afterwards. E1 asks how much of the final response the *prediction* captures with one pass through the block,
and with t passes.

- Run: the prediction stage again (same problems, same push file, T = 48), additionally saving the partial
  operator sums `A_t L_k = (J + … + J^t) L_k` for t = 1, 2, 4, 8, 16 (`--partial 1,2,4,8,16`). About 3 GPU-minutes.
  The full sum must reproduce the first prediction run.
- Analysis (`huginn_addition/analyze_llever_partial.py`), eval problems, doses 0.01, 0.03 and 0.09:
  each `ρ·m_ref·G_j·(A_t L_k)` against (a) the observed response at the end of the forced run and (b) the observed
  response after the same number t of forced loops; per-problem correlation, slope and normalised error;
  bootstrap interval of the mean paired difference in error between t passes and the full sum.
- Reading, fixed now: "what repetition adds to the prediction" is the normalised error of the one-pass
  predictor against the endpoint, compared with the full sum's, at ρ = 0.01 and 0.03. (b) is a check that the
  linear response also holds at intermediate times. No verdict label; this is exploratory and does not change
  the registered verdict.
