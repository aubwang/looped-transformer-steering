# Exploratory additions to the monitor experiment (NOT registered)

Written 2026-11-10, after the registered analysis (`docs/oversight_monitor.md`, Draft 1.3) had been run on three
seeds and read, and after the independent results review (`results/mon_report-20261110/astra_review.md`). Everything
below was chosen with the registered outcomes already known. It uses only data already collected. Nothing here
changes a registered number, and the frozen files are not edited. Each item states what it is for and what would
count as which answer, written before the item is run.

## E1. Learning curve (does the probe improve with more training problems?)

Why: the registered probes read d = 2,048 coordinates and are trained on about 1,200 problems per fold. They reach
AUROC 0.69, below the output-confidence baseline (0.74), and add about 0.005 to it. One possible reason is that
1,200 problems are too few to fit 2,048 coefficients; another is that the state has little more to give to a linear
reader. The learning curve distinguishes "still rising at the largest size" from "already flat".

What: for each family, for L32 (state x_32, labels at 32 loops) and D (state h_12), own labels, the registered fixed
site, seed 0, the same five outer folds as the registered 2×2 table (so the held-out problems are the same at every
size). The training part of each outer fold is cut down to nested subsets of 1/8, 1/4, 1/2, 3/4 and all of its
problems, drawn within the two classes; three independent nested draws (the full size is one fit). At each size the
whole registered fitting procedure is run on the subset only: standardization, the four inner folds, the choice of
penalty. Fitted at each size: the probe; the confidence baseline (four features); confidence + probe.

Reported per family and as the four-family mean: AUROC of the three fits against the number of training problems;
the added value (confidence + probe minus confidence); L32 minus D at each size; and the last-doubling gain,
AUROC(all) − AUROC(1/2), for the probe and for the added value, with an interval from 2,000 resamples of the held-out
problems within folds, the same resample at both sizes (conditional on the fitted probes, like the registered
intervals; subset draws are averaged first).

Check built in: at full size the probe and confidence AUROCs must reproduce the registered seed-0 values
(`m1a` `table/L32/L32`, `table/D/D`; `m1c` `fixed/*/confidence`, `fixed/*/confidence+probe`) to 1e-9.

Reading, fixed in advance:
- last-doubling gain of the probe ≥ 0.02 in the family mean with an interval excluding 0: the probes are still
  sample-limited; the registered comparison is a comparison at this sample size, and a larger collection could move it.
- gain within ±0.01 with an interval inside ±0.02: the curve is flat at this size; sample size is not the main limit
  for a linear reader of this state.
- anything else: not resolved.
- A curve cannot say what happens far beyond the largest size.

Clarification added 2026-11-10 after the run (found by the write-up review, `results/mon_report-20261110/
astra_review_writeup.md`): the text above says every fit uses the outer folds of the registered 2×2 table, and also
that the full-size values must reproduce both `m1a` and `m1c`. These two requirements conflict, because `m1c` uses
each model's own-label folds. The code (`huginn_addition/mon_learning_curve.py`) follows the reproduction
requirement: the probe alone uses the shared `m1a` folds; confidence and confidence + probe use the `m1c` own-label
folds. Each fit keeps its folds at every size. The reading rule is unaffected.

## E2. Tables from numbers that already exist (no fitting)

- Per-seed values of the headline contrasts and how often each site is chosen (the design asks for the spread over
  seeds; the compact summary shows the median).
- The frozen-threshold quantities of the budget transfer that the compact summary left out: fraction rejected,
  error among accepted answers, fraction of labels changed.
- The calibrated forecast against the constant forecast with the fraction of items for which a forecast exists and
  the versions that allow for storage rounding.
- Error among accepted answers at 80% coverage for the own-state probes.

## E3. Small computations on the saved held-out scores (no fitting)

- Joint table of right/wrong for L32 and D per family, with the association between the two models' errors; the
  rank correlation between the probe scores read from the two models' states, and between probe score and confidence.
  Purpose: describe how much of the predictable part is shared between models. It cannot identify a cause.
- Deep-budget comparison on the same problems: the probe fitted at 32 loops, evaluated at 32 loops on the 256
  deep-subset problems only, next to the same probe applied at 48, 63, 64 loops on those problems. Purpose: separate
  "the problems are different" from "the state moved". The probes fitted at 48–64 loops were trained on 256
  problems instead of 1,500, so they are not compared.

## E4. Position −2, paired (added 2026-11-10 after E1–E3 had been run)

Why, and what was already known: while writing up, the registered secondary rows for the token position before the
last one (`m1s` `L32/prev32`, `D/prev12`) were read for the first time. Family-mean AUROC is 0.636, 0.630, 0.636 for
L32 and 0.599, 0.609, 0.615 for D over the three seeds. The design registered these as single rows with no contrast,
so no interval for the difference exists. This item was therefore chosen *after seeing a difference*, among roughly a
dozen secondary rows; it can describe the difference, not confirm it.

What: from the saved held-out scores only (no fitting), L32 `prev32` minus D `prev12` (the matched sites: what each
model's last four layers receive), per family and for the four-family mean, per seed; also D's other saved sites at
this position (`prev8`, `prev16`) so that the choice of D's site is visible. Interval: 2,000 resamples of the problems
(the same resample for both models; each model's AUROC is the mean over its own five folds). The point estimates must
reproduce the registered AUROCs.

Reading, fixed before running: reported as an observation whatever the interval. If the family-mean interval excludes
zero in all three seeds and the difference survives against D's best saved site at this position, it is named as the
one place where the looped model's state was the more readable, and as a candidate for a registered test on new data;
otherwise as not resolved.
