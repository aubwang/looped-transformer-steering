ACCEPT WITH CHANGES

Required changes to claims

1. Scope “accuracy depends on loop parity” to the demonstrated example and policy. For the tested start and worked-example context, continuing `816 + 254 =` greedily gives ` 1070` at 511 loops per token and ` 1060` at 512: CORRECT versus INCORRECT, with both lines terminated. That existence sentence is licensed; general accuracy, default initialization/stopping, arbitrary odd/even budgets, and persistence beyond 512 are not.
2. Describe selection explicitly: oscillating contexts, near-tie screening, then hit selection. The descriptive yield is **4/358 settled final-answer positions** (about 1.12%) among 363 final-answer positions in 85 selected prompts; 84/85 prompts have every answer position settled. This is a selected-sample yield, not a population rate or an accuracy estimate.
3. Retain start dependence. For 30010, seed 1 predicts the true digit `7` at all eight budgets despite an oscillating pair gap; seeds 2–4 alternate. For 30049, all four alternate, but seeds 3–4 have odd/even gaps approximately −0.178/+0.260 instead of −0.224/+0.318: the observed orbit is not unique across starts. All four alternate for 30386 (seeds 1 and 4 reverse phase) and 30032 (seed 2 reverses phase). These are finite-window observations, not proofs about asymptotic attractors.
4. Keep 30049's correct/wrong-digit result explicitly teacher-forced. Continuing `924 + 675 =` with its own digits gives ` 1609` at 511 and ` 1509` at 512, both INCORRECT against 1599. The earlier-digits-right scan flag does not guarantee a correct free continuation.
5. Report the omitted **one UNRESOLVED position**, uid 30381 / position 42, an “other” position varying between `0` and `4`. Six “other” FLICKER positions refers to the new prompts only; including the three controls there are nine. No claimed numerical value differs beyond rounding.

Recomputed numbers (positions zero-based; leads and delta in log-probability units)

| Item | Recomputed result |
|---|---|
| Screen | 400 prompts, 32 loops; minimum 0.0453517437; 9 below 0.1, 55 below 0.3 |
| Selection | Exact smallest 85, ties by uid; maximum 0.4986214638 |
| Contexts 20059 / 20145 / 20144 / 20178 | Selected 27 / 25 / 12 / 21; all-answer-settled 26 / 25 / 12 / 21 |
| Full scan, including controls | 4,447 positions: 13 FLICKER, 1 UNRESOLVED, 4,433 NONE |
| New-prompt FLICKER | 4 final-answer, 0 example-answer, 6 other |
| Runtime | 10 calls; 1,202.779913 s = 20.0463 min; largest 248.418398 s = 4.1403 min |

| uid / position | even / odd token | Earlier digits right | Minimum lead | delta | Guard threshold | Minimum C4 lead |
|---|---|---|---:|---:|---:|---:|
| 30049 / 45 | `5` / `6` | true | 0.223957062 | 0.0000953674 | 0.000953674 | 0.224012984 |
| 30010 / 46 | `6` / `7` | true | 0.031177521 | 0.0000209808 | 0.000209808 | 0.031245316 |
| 30386 / 49 | `8` / `9` | false | 0.029673576 | 0.0000610352 | 0.000610352 | 0.029703130 |
| 30032 / 46 | `0` / `1` | false | 0.026605606 | 0.0001258850 | 0.001258850 | 0.026577440 |

All four are RIGHT/WRONG DIGIT and CONFIRMED: C1 and C4 pass, and minimum scan lead exceeds max(1e-4, 10 delta). True sums and digit positions match all four claims. C1 reproduces the scan phase at 509–512; scan alternation covers 449–512; C2 and patched C4 cover 505–512. C4 requires the same pair with phase free (these four also happen to match phase).

Recommended public statement

In a targeted search of addition prompts, Huginn showed four answer-digit positions whose most probable token alternated between the correct digit and a wrong digit over loop budgets 449–512 from a fixed start; its own forward and a patched double-precision computation confirmed the pairs over shorter windows. For one tested prompt ending `816 + 254 =`, greedy continuation produced the correct sum 1070 at 511 loops per token and the incorrect sum 1060 at 512. This establishes an example under the tested policy, not a general accuracy or prevalence result; the behavior depends on the start state.

Notes

- Independent CPU calculations used raw NPZ/confirmation JSON for selection, classifications, denominator, leads, delta, guards, seeds and continuation statuses. Re-running the inspected analysis in memory reproduces all four saved summaries exactly. All screen, scan, confirmation and continuation binding gates pass; the three controls reproduce their original raw token pairs. Cached-tokenizer decoding verifies saved screen/scan strings and continuation text.
- Execution follows the planned order and selection: screen, scan, C1/C2, C4 after C1 passes, then continuation of 30049 and 30010 in confirmation-selection order. `--parts C1,C2` and continuation `--n-new 7` are recorded. All four continuation strings contain a newline after the answer and then `33`; Section 5 excludes that suffix. The correctness-comparison flag is true only for 30010.
- No substantive design departure is evident. All ten calls completed with empty flags; fresh preceding smokes and production calls share the current app hash `4a7270a6…`. Screen and scan smoke projections are 89.33 and 249.86 seconds including load, passing both admission limits. No recorded retry or stopping-rule exception occurred. Continuation reused the matching C1/C2 smoke as permitted by the launcher; that smoke did not exercise C3, a coverage limitation rather than an incomplete production result. Metadata corroborates execution order, not the historical time at which the design text was fixed.
- No evidence implicates batch composition: all four scan hits survive single-sequence C1 and five-sequence C2 seed 0, with pair-gap discrepancies far below their leads. This supports these hits, not arbitrary-batch invariance. The guard is empirical, not a formal error bound. Patched C4 records double normalization, rotary arithmetic and logits, with float32-drawn starts and retained float32 rotary constants cast up.
- No Modal, GPU or network calls, commits, or saved-summary changes. Only this review was written; the local test source was inspected without running its file-writing fixtures.
