# An answer that flips with loop parity: search design (Draft 1.1, 2026-11-11)

Status: design fixed before any run of this search; Draft 1.1 applies the pre-run review (Section 8). It reuses, unchanged, the scan, the classification, the
confirmation and the numerical guard of `docs/flicker_design.md` (Draft 1.2, "flicker 1" below) and its code
(`huginn_addition/flicker_app.py`, `analyze_flicker.py`). Only what differs is written here.
Budget: every call at most 8 GPU-minutes on one A100-40GB (function timeouts 480 s and 420 s); expected total
about 27 including the test calls. The total is an expectation, not an enforced limit; reruns count towards it.

## 1. Question

Flicker 1 found three real-token prefixes whose most probable next token alternates with the parity of the loop
count. All three were guesses about an operand. None of the 551 answer positions flipped.

**Is there an addition prompt in which Huginn's most probable token for a digit of the answer alternates with
the parity of the loop count, and in particular alternates between the correct digit and a wrong one?**

What flicker 1's scan says about the odds (computed from its saved file before this design): 235 answer
positions sit on a settled period-two orbit; the lead of the top digit over the runner-up has median 3.2 in
log-probability and the difference of that lead between the two phases (the *swing*) has median 0.03; the
closest position came within 0.018 of a tie. A flip needs an oscillating answer position where two digits are
nearly tied. The search therefore (a) uses worked-example blocks in which the oscillation starts inside the
examples, so that it does not depend on the final sum, and (b) screens many final sums cheaply for a near-tie
before spending 512 loops on them.

## 2. Prompts (fixed now)

Four contexts: the three worked-example lines of the flicker 1 prompts 20059, 20145, 20144 and 20178. These are
the four prompts with the largest median swing at their final-answer positions among those whose first settled
period-two position lies in the example lines (0.132, 0.077, 0.064, 0.059); in each, every position of the last
line was on the orbit.

For each context, 100 new final sums `a + b =`: 50 with three-digit and 50 with four-digit operands, operands
uniform over numbers of that length, drawn with `random.Random(20261111 + context uid)`, rejecting any pair
`(a, b)` that occurs as a final sum in `results/census_prompts_all.json`, `results/screen_prompts.json` or
earlier in this file. Problems are built with `data.make_problem` as in the census (same fields; `uid` 30000 …
30399 in order context by context; `source = "flicker2"`; `context_uid`). File:
`results/flicker2_screen_prompts.json` (400 prompts), each prompt teacher-forced with its true answer.

Whether the oscillation really carries over to new final sums is not assumed: it is measured in Stage B.

## 3. Stage S: screen (32 loops)

All 400 prompts, `K = 32`, readout at the last 8 budgets (25 … 32), census start (uid, seed 0), two calls of 200
prompts. No classification is made from this stage. Screen statistic of a prompt: over its final-answer
positions, the smallest value of the lead (top log-probability minus second) averaged over the 8 budgets.

Selection for Stage B (fixed now): the 85 prompts with the smallest screen statistic (ties by uid), in that
order. The statistic is a cheap proxy; if leads at 32 loops do not predict leads at 512, the search loses yield,
not validity.

## 4. Stage B: scan (512 loops)

88 prompts: the three flicker 1 hits as positive controls (20178, 20147, 20146, from
`results/flicker_prompts.json`, unchanged) followed by the 85 selected, in two calls of 44 (the batch size of
flicker 1's scan). Everything else as flicker 1's scan: `K = 512`, top three tokens at every position over
budgets 449 … 512, drift and lag-2.

Gates (if any fails the search is invalid or incomplete, which is not the outcome "no answer hit"): harness
validation; both calls complete with no error flag and with the same model, revision, code and configuration;
the prompt file is the one whose hash the run recorded, its 85 selected rows are exactly the screen's selection
recomputed from the screen files, and the decoded tokens of every row spell its prompt and answer; exactly these
88 prompts in this order; `K = 512`, 64 tail budgets; array shapes; finite values; role parse consistent with
`pred_pos`; **positive controls**: positions 40, 41 and 44 of the three control prompts are FLICKER with the same
token pairs as in `results/flicker_scan-20261111-022450/flicker_summary.json`.

Classification of every position: FLICKER / UNRESOLVED / NONE exactly as flicker 1 (strict alternation over all
63 pairs between two tokens, on a settled period-two orbit). Reported:

- **Answer hits**: FLICKER at a final-answer position of a new prompt. The top token is taken over the whole
  vocabulary, so a token of the pair need not be a digit; whether each is a digit token is recorded. Sub-label
  **RIGHT/WRONG DIGIT** if both are digit tokens and exactly one is the true digit at that position;
  **RIGHT/OTHER** if one is the true digit and the other is not a digit token; **WRONG/WRONG** if neither is the
  true digit. For each: the two tokens, the true digit, which parity is right, smallest lead, median lead and
  top probability per phase, and whether the top token at every earlier answer position of that prompt is the
  true digit at all 64 budgets. That last flag describes the scan only; it does not show that the model
  would write those digits by itself, which is what the continuation of Section 5 tests.
- FLICKER at example-answer positions: the example text is shared by all prompts of a context (the start states
  are not, since each prompt has its own draw), so these are counted once per context and position, with the
  token pairs seen and the number of prompts showing each.
- FLICKER at other positions (operand guesses and the like): counted, not pursued.
- Carry-over: per context, among the selected new prompts, the share whose final-answer positions are all on a
  settled period-two orbit, and the distribution of the swing over the settled final-answer positions. Here the
  *pair gap* of a position at a budget is the log-probability of its top token at budget 512 minus that of its
  runner-up at budget 512 (both read at that budget; undefined if either is outside the saved top three), and
  the *swing* is the absolute difference between the mean pair gap at even and at odd budgets.
- Near misses: the ten settled final-answer positions that are not hits with the smallest ratio of the smaller
  absolute mean pair gap (even or odd) to the swing.

## 5. Confirmation

Hits to confirm (rule fixed now): answer hits, RIGHT/WRONG DIGIT first, then RIGHT/OTHER, then WRONG/WRONG,
within each by decreasing smallest lead, first four with distinct prompts. C1 (model's own forward at budgets 509 … 512), C2 (harness on the cut
sequence, seeds 0 … 4, budgets 505 … 512), C4 (patched double precision, 505 … 512) and the numerical guard
`smallest lead > max(1e-4, 10 δ)` exactly as flicker 1, with labels CONFIRMED / NUMERICALLY UNRESOLVED /
CONFIRMATION INCOMPLETE / NOT CONFIRMED. C4 requires the same pair of tokens; its phase is free. The confirmation
call runs C1 and C2 only (`--parts C1,C2`).

Continuation (descriptive, separate call, `--n-new 7`, for the first two CONFIRMED hits in selection order; the
file of hits for this call is written by the confirmation analysis, not before): the sequence is cut after the
`=` of the final sum (the first answer position), and continued greedily by 7 tokens at 511 and at 512 loops per
token, each branch following its own tokens and using its own budget for every token; start rows beyond the
prompt's draw come from seed 1000. Reporting rule (fixed now): the *answer line* is the decoded text before the
first line-feed character, wherever in a token it falls; the raw text and token ids are kept. Status of a
branch: TRUNCATED if there is no line feed within the 7 tokens; MALFORMED if the answer line, stripped of
spaces, is not a string of digits; otherwise CORRECT if it equals `a + b` and INCORRECT if not. This is a
comparison under this stated policy, not the model's default decoding.

## 6. What a result means

- A CONFIRMED RIGHT/WRONG DIGIT answer hit: there is an addition prompt for which, with the true earlier digits
  supplied, Huginn's most probable next token at an answer-digit position is the correct digit at one parity of
  the loop count and a wrong digit at the other, from the tested start; with the same budget windows as flicker
  1 (scan 449 … 512, forward 509 … 512, double precision 505 … 512, pair with phase free). Only if both
  continuation branches end (status CORRECT for one, INCORRECT for the other) may it be said that the model
  writes the right sum at one parity and a wrong one at the other, under the stated continuation policy.
- A CONFIRMED RIGHT/OTHER hit: the correct digit against a token that is not a digit; stated in those words.
- Only WRONG/WRONG answer hits confirmed: the top token at an answer position flips between two tokens neither of
  which is the true digit. Reported as such.
- Answer hits in the scan, none confirmed or confirmation incomplete: candidates only.
- No answer hit, all gates passing: none among these 85 prompts. Not evidence that none exists; the near-miss
  table is reported.
- A gate fails or a call is incomplete: the search is invalid or incomplete. No statement about hits either way.
- Never claimed: how common this is. The prompts are selected three times over (contexts that oscillate, sums
  screened for near-ties, then hits). No statement about the model's accuracy in general, about the default
  random start and stopping rule, or beyond 512 loops.

## 7. Cost control

Screen: two calls of 200 prompts, timeout 480 s each, admitted only if the projection from the latest passing
scan test call (time per loop scaled by 200/44) fits. Scan: two calls of 44, as flicker 1 (about 4.5 minutes
each). Confirmation about 4 minutes, double precision about 1, continuation about 4. Every production call
requires a passing test call of the same code hash, as in flicker 1.

## 8. Changes

Draft 1.1 (after the pre-run review `results/flicker2_prerun_review.md`, before any run). (1) The analysis is
bound to its inputs: prompt-file hashes, the screen prompts against their deterministic builder, the Stage B
rows against the recomputed screen selection, decoded tokens against prompt text, and one model, revision, code
and configuration across the calls of a stage. (2) The continuation hits are written after confirmation, for at
most two CONFIRMED hits; its reporting rule (answer line, four statuses) is fixed above. (3) Sub-labels now say
whether the two tokens are digits; the sentence implying that teacher-forced digits are what the model would
write was removed; gate failure is its own outcome. (4) The per-call budget is stated as it is enforced (8
minutes, by the function timeouts); the screen is admitted against its own compute allowance. (5) If a call of
a stage fails, the stage stops at once and is invalid; nothing is patched chunk by chunk. A rerun is a complete
new run of that stage with the same prompt file, code and configuration, and its time counts. Descriptive
statistics (carry-over, swing, near misses) were given exact definitions.

Results check (`results/flicker2_results_review.md`, ACCEPT WITH CHANGES; no number differed). Recorded here so
that the outcome is stated with its scope: four answer hits, all RIGHT/WRONG DIGIT and CONFIRMED, among 358
settled final-answer positions (363 final-answer positions in the 85 selected prompts; 84 of 85 prompts have
every answer position settled). Also in the scan: six FLICKER at other positions of the new prompts (nine with
the three controls) and one UNRESOLVED position, uid 30381 position 42, varying between `0` and `4`. The
continuation gives CORRECT at 511 loops and INCORRECT at 512 for uid 30010 (`816 + 254`), and INCORRECT at both
for uid 30049 (`924 + 675`). The continuation call reused the passing C1/C2 test call of the same code hash,
which did not exercise C3. GPU time 20.05 minutes over ten calls, largest 4.14 minutes.
