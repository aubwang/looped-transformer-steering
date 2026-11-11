# A real-text loop-parity flicker in Huginn: search design (Draft 1.2, 2026-11-11)

Status: design fixed before any run; Draft 1.1 applies the pre-run review (Section 8). Code `huginn_addition/flicker_app.py`, analysis
`huginn_addition/analyze_flicker.py`, local test `huginn_addition/flicker_local_test.py`.
Budget: every run at most 8 GPU-minutes on one A100-40GB; expected total about 12 including test runs.

## 1. Question

About 13% of fresh addition prompts settle onto a period-two orbit of Huginn's loop
(`huginn_addition/WRITEUP.md`, Section 2.3). In every real-text prompt tested so far the answer was the same in
both phases: the census (26 oscillating prompts) and the screen (60 prompts, 18 with late motion) read the output
only at the positions that predict an answer digit, where the model is confident. The only observed change of the
top token with loop parity was at a blended, non-textual input (`bifur49_flicker`).

**Is there a sequence of real tokens after which Huginn's most probable next token differs between consecutive
loop counts, alternating with period two, from one fixed start state?**

The earlier runs could not see such a sequence unless it ended at an answer position. Every position of a prompt
is a next-token prediction, and positions downstream of an oscillating source inherit the oscillation. At many
of them (the digits of an operand, for example) the next token is unpredictable, so the two leading candidates are
nearly tied. This search reads the output at every position.

## 2. Model, precision, start

Huginn-0125 at the pinned revision, float32, noise off, harness `H` of `huginn_addition/intervene_app.py`
(`prep`, `validate`, `batch`, `step`, `coda_logits`), exactly as the census. Start state: the census draw,
`init_state(·, uid, seed 0)`. `K = 512` loops. "Budget `b`" means the state after `b` loops; the readout at
budget `b` is the model's logits computed from that state through the coda.

## 3. Prompts (fixed now)

The 44 real-text prompts in which the earlier runs found late motion:

- the 26 census prompts of `results/relax_selection.json` (uids 20009 … 20110);
- the 18 screen prompts with late motion above 1e-4 in `results/screen-20261105-114648/analysis/summary.json`
  (rows with `blinking` true; 15 settled period two, 3 unresolved), taken from `results/screen_prompts.json`.

Each is the prompt plus its true answer, teacher-forced, as in the census. They were selected for oscillating, so
nothing below estimates how common a flicker is in general text.

## 4. Stage 1: scan

One call. For each prompt, 512 loops from the census start. Saved per prompt and position:

- relative drift `|s_{k+1} − s_k| / |s_{k+1}|` at every loop, and the lag-2 ratio
  `|s_{k+1} − s_{k−1}| / |s_{k+1} − s_k|` over the last 64 loops (as the census);
- at each of the last 64 budgets (449 … 512) and at **every position**: the three most probable tokens over the
  full vocabulary and their log-probabilities.

Gates (no hits are reported if any fails): the harness validation of `H.validate`; all 512 loops completed for
every prompt; **reproduction**: for the census prompts, the top token at the answer positions over the 64 tail
budgets equals the one saved by the census run (`census.npz` on the results volume, read inside the container);
for the screen prompts, the most probable *digit* at each answer position at budget 512 (argmax over the ten digit
tokens, as the screen computed it) equals `answer_argmax_last` of the screen summary. The scan must contain
exactly the 44 prompts of Section 3, with `K = 512` and 64 tail budgets; anything else gets no classification.

### Classification (fixed now), per prompt and position `p`

With `a_b` the top token at budget `b`, `b = 449 … 512`:

- **FLICKER**: `a` takes exactly two values and `a_b ≠ a_{b+1}` for all 63 consecutive pairs; and the position is
  on a settled period-two orbit by the screen's criterion: mean relative drift over the last 64 loops above
  1e-4, median lag-2 ratio over the last 32 loops below 0.01, and the mean drift of the last 32 loops within 2%
  of that of the 32 before.
- **UNRESOLVED**: `a` is not constant over the tail but the position is not FLICKER. Reported, never counted as a
  hit.
- **NONE**: `a` constant.

For a FLICKER position: the two tokens, the smallest lead (top log-probability minus the second) over the 64
budgets, the median lead and the top probability in each phase, the true next token and whether either phase
token equals it, and the role of the position:

- *final answer*: `p` is one of the prompt's `pred_pos` (the next token is a digit of the answer to the last sum);
- *demo answer*: the next token is a digit of the answer in one of the worked examples;
- *other*: everything else (operand digits, `+`, `=`, line breaks, the token after the complete answer).

Tier A = final answer or demo answer. Tier B = other. Many Tier B positions are guesses (the model cannot know
the next operand digit); some are predictable (a `+` or a line break). The role and the true next token are
reported with every hit. A Tier B flicker still answers the question of Section 1 and is reported as such.

## 5. Stage 2: confirmation

Hits to confirm (rule fixed now): sort FLICKER positions by tier (A first), then by decreasing smallest lead;
take the first four with distinct prompts. For each, the token sequence is cut after position `p` (attention is
causal, so the states at positions up to `p` are unchanged) and the start state is the first `p + 1` rows of the
census draw.

- **C1, the model's own forward.** For budgets 509, 510, 511, 512: `model(ids, input_states=start,
  num_steps=b)`, top three tokens and log-probabilities at the last position. Pass: the top token equals the
  scan's at each of the four budgets, and alternates.
- **C2, the harness on the cut sequence, and other starts.** Start seeds 0, 1, 2, 3, 4 in one batch (harness,
  512 loops, last 8 budgets, 505 … 512). Seed 0 is the census start: it gives a third computation of the same
  quantity (scan: 44 padded prompts; C1: the model's forward on one sequence; here: the harness on five).
  For seeds 1 to 4: whether the top token alternates strictly over the 8 budgets, and which token is on top at
  budget 512. Descriptive: says whether the flicker and its phase depend on the start.
- **C3, continuation.** Greedy continuation by 6 tokens at budget 511 and at budget 512, each following its own
  branch, the whole sequence recomputed from the start for every new token (as `freegen`). Start rows beyond the
  census draw come from `init_state(·, uid, seed 1000)`. Descriptive: the two texts.
- **C4, double precision (separate call, only for hits that pass C1).** The model in float64 with the float32
  start cast up, 512 loops, last 8 budgets (505 … 512) at the last position. The pinned model code casts to
  float32 inside its normalisation and its rotary position step, and the shared harness casts the logits to
  float32; for this call those three places are replaced by the same formulas in float64 (the rotary constants
  keep their float32 values), and the run records that the replacement was active. Pass: strict alternation
  between the same two tokens over the 8 budgets, with a finite positive lead at each. The pair must agree with
  the scan; the phase need not (a different rounding history may land on the other phase).

**Numerical guard (fixed now).** For a hit, the *pair gap* at a budget is the log-probability of its first token
minus that of its second. Let `δ` be the largest difference, over budgets 509 … 512, between the pair gaps of
any two of the three float32 computations (scan, C1, C2 seed 0). A hit is **confirmed** if C1 and C4 pass and
its smallest lead over the 64 scan budgets exceeds `max(1e-4, 10 δ)`. A hit that passes C1 and C4 but not the
guard is **numerically unresolved**: reported, not counted. The guard is empirical, not an error bound. C2
(seeds 1 to 4) and C3 are reported with every hit.

## 6. What a result means

- At least one confirmed hit: there is a sequence of real tokens for which Huginn's most probable next token
  depends on the parity of the loop count, from the tested start. What was measured, precisely: in float32 on
  the full prompt the top token alternates over budgets 449 to 512; the model's own forward on the cut sequence
  reproduces budgets 509 to 512; in double precision the same pair alternates over budgets 505 to 512. If C2
  shows other starts land in the other phase or do not flicker, that is part of the statement.
- FLICKER positions in the scan but none confirmed, or the confirmation incomplete: reported as candidates
  only. This is neither a positive result nor "no flicker".
- No FLICKER position: none among these 44 prompts at any position. Not evidence that none exists.
- Not claimed in either case: how common such sequences are; persistence beyond 512 loops; behaviour with the
  model's default random start and stopping rule; any effect on a task's accuracy.

## 7. Cost control

Scan: one chunk of 44 prompts; function timeout 480 s; computation stops at the earlier of 340 s after the model
is loaded and 45 s before the timeout counted from function entry (an incomplete prompt gets no classification). The census ran 50 prompts × 512 loops in 274 s including loading.
Confirmation: timeout 420 s, computation stops at 290 s, C1 before C2 before C3, partial results returned. C4:
timeout 420 s, computation stops at 270 s. The margins leave room for loading the model inside the timeout. Each production
call requires a passing test call of the same code (same file hash): for the scan, all 44 prompts with 64 loops,
which measures the time per loop and the peak memory at the production batch shape, checks that the census
references can be read, and must project the production run inside its allowance; for the confirmations, one
made-up hit with 64 loops. The test calls license no classification. A model forward in C1 is started only if
1.5 times the measured duration of one fits before the deadline. Completed parts are returned if a later part
fails.

## 8. Changes

Draft 1.1 (after the pre-run review `results/flicker_prerun_review.md`, before any run). (1) C4 as first written
was not a double-precision computation: the model and the harness cast to float32 in three places. Replaced by
the patched double path above, and the numerical guard was added. (2) The screen reproduction gate compared a
full-vocabulary top token with a digit-restricted one; it now compares digit with digit. (3) Section 6 now says
which budgets each check covers, that the phase may differ across precisions, and names the outcome "candidates,
none confirmed". (4) The analysis is bound to the fixed experiment: exactly the 44 prompts, `K = 512`, 64 tail
budgets, and a confirmation is accepted only for the hits file of the scan it belongs to, with the fixed
budgets. (5) Deadlines count from function entry, C1 calls are admitted by measured duration, production
requires a passing test call of the same code. C2 gained seed 0 as a third float32 computation. The compute
allowances were lowered (scan 340 s, confirmation 290 s, C4 270 s) to leave room for loading.

Draft 1.2 (after the results). In the confirmation run the continuation C3 reached the compute deadline after 4
of its 6 tokens. A continuation-only call (`--mode continue`: C3 alone, 9 new tokens, hits 20178 and 20146,
same start rule) was added afterwards to obtain complete lines. It is descriptive and changes no label.

Clarification after the results check (`results/flicker_results_review.md`), not a change of rule. Section 6's
second outcome ("the confirmation incomplete") refers to the parts that define the label: C1, C4 and the
numerical guard. Those completed for all three hits, so the label CONFIRMED of Section 5 applies. The
confirmation call's overall status is nevertheless "not ok": its flags are `C3 incomplete round=4` and
`incomplete`, and inspection shows the incompleteness is confined to the descriptive continuation C3 (4 of 6
tokens). The four census reference files were downloaded afterwards to `results/census-20261105-*/census.npz`;
their hashes equal those recorded by the scan and the reproduction comparison was repeated locally (26 of 26
equal).
