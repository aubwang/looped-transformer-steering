# Independent review of the monitor write-up

**Verdict: ACCEPT WITH CHANGES.** The E4 saved-score calculations reproduce, its stated exploratory decision rule is satisfied, and the principal registered findings remain supportable. No refitting is requested. The manuscript nevertheless contains numerical inaccuracies and stronger claims than its evidence supports, including several qualifications explicitly requested in the earlier review that have not actually been incorporated. The header's assertion that all required changes were applied is premature.

Scope: docs/monitor_results.typ (line numbers below refer to the reviewed source), its generated JSON inputs, the frozen design/specification, relevant analysis/collection code, and the earlier astra_review.md. I used the specified Python with -I and single-threaded numerical libraries. All statistical computations used saved scores or JSON summaries; no probes were fitted, no GPU or Modal work was performed, and no hidden-state arrays were loaded. A local cached tokenizer was used only to inspect representative prompt suffixes. Only this report was written. I reviewed the source, not the PDF's visual layout.

## Required changes

1. **E4: disclose the actual pairing, labels and token sites.** Locations: Section 4, lines 315–319, and Section 7.4, lines 591–594. The problems match, but the outer folds do not. Replace the methodological description with:

   > At position −2 we read the token “ =” in the arithmetic prompts and “Answer” in ARC-Easy, immediately before the final space or colon respectively. Both models have already received the problem text. L32/prev32 and D/prev12 are the inputs to their respective last four layers at this position. Each probe predicts its own model's final-answer error, using 32 loops for L32. The same 1,500 problems per family are evaluated, but each model uses its own label-stratified five-fold assignment. At each fixed site the penalty is selected using four inner folds. For the exploratory difference we resample 1,500 problems with replacement within each family, use identical problem multiplicities for both models, average each model's AUROC over its own five folds, and then average the four family differences equally. The 95% percentile intervals use 2,000 draws and hold all fitted probes fixed. Unlike the registered intervals, this exploratory bootstrap does not stratify by outer-fold intersections.

   The existing brief description is reasonable but incomplete; it must not leave “paired” open to interpretation as shared outer fits. The “last-position state of L32 versus intermediate D” explanation would be factually wrong here: both are position −2, and both matched sites precede four final layers.

2. **E4: qualify readability and distinguish matched-layer from all-layer claims.** Locations: summary lines 93–95; Section 7.4 lines 608–620; Section 8 lines 633–635. Replace the first two E4 interpretation bullets with:

   > Against D at the matched layer 12, the four-family mean difference is 0.0374, 0.0214 and 0.0214 in seeds 0, 1 and 2. Each conditional interval excludes zero. All nine family-mean comparisons with D's saved layers 8, 12 and 16 have positive intervals; across those comparisons the point differences range from 0.0208 to 0.0507. Against layer 12 the point difference is positive in every family and seed, although only three of the twelve individual-family intervals exclude zero. This satisfies E4's exploratory rule and makes the position a candidate for a registered test on new data. It does not establish that L32's representation is intrinsically more informative: the probes predict different models' errors. The comparison is conditional on the fitted probes, selected after inspecting the secondary rows, and the seeds reuse the same problems. Only three D layers and two L32 loop budgets were saved here; no nested best-site comparison at position −2 was performed.

   Replace or follow the speculative third bullet with:

   > The models have different error sets and accuracy (four-family means 0.599 for L32 and 0.574 for D). AUROC is not mechanically raised by a lower error prevalence, but different error types and class counts can affect predictability and fitted-probe performance. L32's position −2 probes at 8 loops already reach 0.6329, 0.6324 and 0.6347, close to its 32-loop values 0.6364, 0.6300 and 0.6365. Those 8-loop probes use 8-loop labels, so this is not a fixed-label test of state convergence. It nonetheless supplies no evidence that loops 9–32 create this advantage. Repeated computation, the particular token position, checkpoint differences and task-specific features remain possible explanations.

   In the summary and Section 8 use “a post-hoc advantage for predicting each model's own errors at position −2” in place of an unqualified advantage in state readability. The phrase “the interval excludes zero in each seed, and against each of D's saved layers” is **true for the family means** and need not be retracted.

3. **Remove losslessness/safety claims and correct the budget numbers.** Locations: summary lines 84–85; Section 5 lines 354–373; Section 7.3 lines 581–582; Section 8 lines 629–635. Suggested summary replacement:

   > At the tested budgets from 4 to 32 loops, ranking performance changes little relative to native probes. On the matched deep subset, applying the same 32-loop probe at 48, 63 or 64 loops also changes AUROC little. These observations do not establish lossless transfer or general budget safety. At 1 loop ranking degrades relative to a native probe; at 2 loops the ranking difference is unresolved, while the frozen threshold rejects substantially more answers.

   Replace Section 5's first bullet with:

   > Across tested targets 4–32 and the registered source/model combinations, the absolute median native-minus-transfer gap is at most 0.00668 for the family mean and 0.01831 within a family. These are descriptive small differences, not equivalence results. L32's frozen threshold rejects about 20%, but accepted-answer error is 0.381 at 4 loops versus 0.355 at 32. Its right/wrong labels agree with the 32-loop labels on about 91% of problems at 4 loops and 98% at 8; these percentages do not measure exact answer-token agreement.

   In the deep-budget paragraph and Section 7.3 replace the causal conclusion about the native probe with:

   > Native deep-budget probes have much smaller training sets (about 205 versus 1,200 problems). This confounds their comparison with transferred probes. On matched deep problems, the same 32-loop probe changes AUROC by at most 0.00182 across the tested deep budgets. This supports stability of that probe but does not prove that training-set size alone caused the native probe's lower AUROC.

   Use these restricted observations in Section 8 instead of “Transfer holds everywhere … observed for 4 to 64.” Change “answers changed” to “correctness labels changed” wherever it describes label_changed, including Section 7.3 lines 578 and 580. Remove the implication at lines 634–635 that a lack of an observed warning establishes absence of silent failures generally.

4. **Keep the main null distinct from equivalence and a literally flat table.** Locations: Section 3 lines 215–216; Section 8 lines 626–628; Section 4 line 278; uncertainty statement lines 646–647. Suggested replacements:

   > No overall advantage was resolved. The diagonal interval envelope is [−0.017, 0.025] and the best-site envelope is [−0.025, 0.018]; neither establishes equivalence. The positive interaction, 0.017 [0.005, 0.030], remains evidence of a small own-state advantage.

   > The closest registered outcome is small added value over confidence in both models, with no resolved overall diagonal or column advantage. The table is not literally flat.

   > No difference between the models in added value was resolved.

   > These intervals omit variability from fitting and from model training; they do not quantify the full uncertainty.

   The current claim that a difference of 0.02 “in either direction is compatible with them” is not true for each quoted envelope: −0.02 lies outside the diagonal envelope and +0.02 outside the best-site envelope. “Within about ±0.02” revives the equivalence-like interpretation rejected in the earlier review. Omitted uncertainty sources do not prove a strict numerical ordering of interval widths, so “the true uncertainty is larger” is too categorical.

5. **Separate path failure, displacement failure and score conservatism.** Locations: summary lines 86–88; Section 6 lines 406–415; Section 8 lines 657–658. Suggested replacements:

   > The raw estimated-ratio forecast had no observed score violations among forecastable held-out problems, but was very loose: the displayed family averages of median bound/movement ratios range from about 183 to 2,640. Calibration produced no demonstrated general advantage over a constant forecast for stable decisions.

   > The predicted bound fails for the sum of later step lengths in about 42%–100% of forecastable problems across the displayed model/budget summaries. Failure for net whole-state displacement is a different statistic, ranging from about 4% to 91%. These observations invalidate extrapolation using the estimated ratio, not the existence of any contraction constant. No observed score violation is an empirical finding, not a certificate.

   > Using whole-state movement and Cauchy–Schwarz can make a score bound conservative. A bound specific to the last row and probe direction is worth investigating, but these summaries do not identify how much conservatism comes from each inequality, the estimated contraction ratio, or storage rounding.

   “The whole-state forecast usually fails” followed only by the path statistic conflates two different tests; at r=16 several net-displacement failure rates are only 4%–6%. The attribution of the entire factor of hundreds to exactly two inequalities is not established.

6. **Complete the calibration and precision qualifications requested previously.** Locations: Section 6 lines 417–450 and captions at 402–403 and 429–431. Replace the calibration explanation with:

   > Calibration is performed separately within each outer fold: half of its held-out problems determine a multiplier using the ceil((n_cal+1)×0.95)-th ordered realized/predicted movement ratio among eligible forecasts, and the other half evaluate it. The constant comparator is likewise calibrated per fold. Coverage is marginal among forecastable evaluation problems, not a 95% guarantee conditional on a decision being declared stable. The rejection threshold is the 80th percentile of the held-out source scores themselves: it uses those test features, though not their labels.

   Replace “declares no more decisions stable” with “does not show a general advantage in the fraction declared stable; it is sometimes slightly higher and often lower.” L32/8 is a direct counterexample to the literal existing sentence: 0.929 versus 0.928. In the raw-forecast caption distinguish denominators: violation rates condition on having a forecast, whereas fraction_forecast_stable uses all problems. In the calibrated table explain that false_stability is the changed-decision fraction among declared-stable items.

   Replace the rounding paragraph's causal assertion with:

   > At 16 loops the conservative rounding allowance exceeds the measured movement, by family-averaged median factors about 224, 65, 19 and 4.7 for L4, L8, L16 and L32. An allowance is an upper bound, not measured rounding error. On the 32-per-family float32-copy subset, the corresponding family-averaged median relative movement changes are approximately 0.596, 0.296, 0.163 and 0.0255. For example, at L4/16 allowing for rounding changes calibrated dynamics/constant coverage from 0.957/0.956 to 1.000/0.996 and stable fractions from 0.881/0.997 to 0.868/0.982. The full adjusted tables show no general dynamics advantage.

   This supplies the missing direct-copy evidence and an actual allowance-adjusted comparison rather than only referring to an external table.

7. **Correct the learning-curve narrative and disclose its fold convention.** Locations: Section 7.1 lines 481–485, 522–535; summary lines 89–91. The E1 reading “not resolved” at lines 519–521 is correct and should remain. Replace the detailed gain claim with:

   > The last-doubling gains are 0.01408 [0.00916, 0.01892] for L32 and 0.01876 [0.01249, 0.02514] for D. Both are positive in every family, but neither reaches E1's 0.02 point-estimate threshold. Earlier doubling gains are 0.0152 and 0.0195 for L32, and 0.0342 and 0.0225 for D. At full size the added values are 0.00440 and 0.00454. The added-value last-doubling gains are positive point estimates, 0.00227 and 0.00330, but their intervals include zero. Thus the increase in added value is unresolved; the curve does not establish that more data would have no benefit.

   Replace “each doubling … 0.015 to 0.02,” the single full-size “0.004,” and “the trend … gives no sign of it” accordingly. “Confidence does not need the problems” should become “The family-mean confidence AUROC is within 0.014 of its full-size value using about 150 training problems.” The summary's “about 0.015 per doubling” should specify the last doubling and preferably say “about 0.014–0.019.”

   Add: “To reproduce each registered fit, the state-only curve uses M1-a's shared joint-label folds, while confidence and confidence+state use M1-c's own-label folds; each procedure retains its folds across sizes.” This is the implemented convention in mon_learning_curve.py and curve metadata. E1's literal instruction that all fits use M1-a folds conflicts with its simultaneous demand to reproduce M1-c, which uses different folds. Disclose this clarification rather than claiming every procedural detail of E1 was followed verbatim; the stated E1 reading rule is unaffected.

8. **Remove unsupported explanations from controls and the permutation diagnostic.** Locations: Section 4 lines 280–283 and 308–312; Section 6 caption lines 461–462; Section 7.1 line 528. Suggested replacements:

   > These particular problem-derived controls reach about 0.60, below the late-state probe. They do not exhaust problem difficulty, so they cannot show that the late probe is reading something other than difficulty-related features.

   > D's seed-0 shuffled-label interval excludes 0.5 from below. The same permutation is reused across split seeds, and the conditional score bootstrap does not repeat permutation, fitting and selection. The result is an anomalous diagnostic observation, not a full-pipeline null test; its mechanism has not been established here.

   > The added-value point estimate is negative, possibly because additional inputs increase fitting variability; that explanation was not isolated experimentally.

   A class-imbalance explanation is not demonstrated by these data. In particular, AUROC is computed within folds and is unchanged by a foldwise intercept shift, so the standard pooled-fold imbalance intuition is insufficient. Do not replace this with a new diagnostic experiment during this read-only revision.

9. **Correct basic mathematical definitions and delimit the contraction argument.** Locations: summary line 78; Section 2 lines 151–152; Section 6 lines 378–388. Use:

   > AUROC ranges from 0 to 1; 0.5 means chance ranking and 1 perfect ranking. It is unchanged by a strictly increasing transformation of the score.

   > Suppose 0 ≤ q < 1 and the stated Lipschitz inequality holds for all consecutive trajectory pairs needed through r+h, with that trajectory remaining in the region where the inequality applies.

   The geometric-series formula and probe-score inequality themselves are correct: the sum starts at q d_r, the intercept cancels, and a row norm is bounded by the whole matrix's Frobenius norm. Keep them. Replace “An ordinary model offers nothing similar” with “Repeated application of the same map gives this geometric form; distinct feedforward layers would instead require bounds for their respective maps.” Replace “last eight ratios” with “up to the last eight available ratios,” and mention the registered relative-step floor of 10^−6. No eight ratios exist at r=4.

10. **Finish the reporting conventions and missing-interval count.** Locations: Section 2 lines 166–173; Section 5 lines 367–368; Section 9 lines 668 and 678–679. Add:

    > Family means and contrasts are formed within each seed before taking seed medians. Consequently a displayed mean or contrast need not equal the corresponding arithmetic operation on separately displayed medians. The interval envelope is descriptive, not a new confidence level or a multiple-comparison adjustment.

    Replace the count with:

    > There are 135 unavailable per-family audit interval entries, all for add3 at deep targets, plus 135 propagated unavailable family-mean entries. These count reporting entries, not distinct fits. Point estimates remain available; any class-losing bootstrap draw makes that entire interval unavailable under the registered rule.

    Once the revisions are made, the introductory/footer review statements may say the earlier wording recommendations were addressed; before that they should say the earlier review motivated the qualifications, rather than assert complete compliance.

11. **Define the remaining technical vocabulary for the intended reader.** Locations: Section 1 line 114; Section 2 lines 161–162; Section 3 line 214; Section 4 lines 302–304; Section 7.2 table headings, lines 540–554. Insert brief first-use definitions:

    > A checkpoint is one saved set of trained model parameters. An input embedding is the numerical vector assigned to an input token before the transforming layers. Entropy is −Σ p_j log p_j, a measure of how spread out the output probabilities are. A principal component is a direction of maximal remaining variance in the training states. Rank correlation here is Spearman correlation, the ordinary correlation of score ranks, with ties averaged. For the joint-error table, correlation is Pearson correlation of the two binary error indicators; if the four cells are a,b,c,d in the displayed order, the odds ratio is ad/(bc).

    Explain “decision” in Section 6 as the monitor's accept/reject flag, not the model's predicted answer. AUROC, probe, budget, label, seeds, and nested fitting already have useful definitions. These additions address the significant remaining unexplained terms without requiring machine-learning background.

## E4 independent computation and fairness audit

I did not import either analysis implementation. Point AUROCs were computed by explicit positive–negative score comparisons within each saved outer fold, giving ties half credit, then averaging five folds equally. After sorting by problem ID, I checked all four sites in every family and seed against the registered M1s AUROCs, and all 45 differences (36 family and 9 family-mean rows) against prev.json. **Maximum point discrepancy: 0 at floating-point precision.**

For an independent bootstrap I generated 2,000 multinomial count vectors of size 1,500 separately within each family, using numpy default_rng(SeedSequence([90211, seed, family_index])). Each count vector was shared by all four sites. Weighted positive–negative pair counts were computed using sorted equal-score groups; I averaged each site's five fold AUROCs, took differences, and averaged the four families on each bootstrap draw. Percentiles were 2.5 and 97.5. This reproduces the exploratory scheme with a different random stream and a different AUROC implementation. All intervals were finite. Maximum endpoint discrepancy across all rows was 0.00552; among the nine family means it was 0.00143. That is approximate Monte Carlo agreement, not exact endpoint reproduction. All nine independent family-mean lower endpoints were also positive.

The resampling scheme is coherent as a conditional, problem-paired descriptive bootstrap despite different model folds. It allows fold and class counts to vary; it is not the registered fold-intersection-stratified bootstrap. It does not capture fitting uncertainty, model-training variability, or selection after inspecting results. Three seeds on the same problems do not supply independent experimental replications. Lower limits near 0.001–0.002 should not be promoted into robust confirmatory evidence.

**Problems and labels.** Every E4 site has the same 1,500 IDs in each family. All E4 L32 labels equal the saved M1-a L32-own labels; all D-site labels equal D-own labels, after ID alignment, for all three seeds. The code independently explains their origin: analyze_mon.labels compares L32's final readout top token at budget 32, or D's true final readout top token, with ans_tok. The position −2 state is not used to redefine the label. Error counts L32/D are 312/386, 821/884, 730/769 and 546/519 in add3/add4/add3w/ARC-Easy. Thus the target events are different, even though the problems match.

**Folds.** The models do not share outer-fold assignments. The numbers of IDs with equal fold indices, out of 1,500, are respectively 303/285/270/326 for seed 0, 325/303/278/316 for seed 1, and 313/278/290/303 for seed 2 (family order as above). All three D sites share D's folds. This follows the registered own-label stratification for M1s; only M1-a explicitly uses shared joint-label folds. Different folds do not invalidate the paired saved-score comparison, but cannot be represented as identical train/test splits.

**Sites and penalties.** Collection code saves x_prev at loops 8 and 32 and h_prev at D layers 8, 12 and 16; no L32 post-coda state is saved at −2. Both matched sites are immediately before four final layers. This matches the registration but is not a match in total computation: L32 has 4 + 32×6 layers before this site, versus 12 for D. Both use the same nine penalties 10^−6 through 10^2 and training-only standardization, selected separately at each fixed site by four inner-fold AUROCs, with ties favoring the larger penalty. E4 performs no new penalty or site selection. Actual chosen penalties differ, as expected; there is no asymmetric tuning procedure. Only the number of saved sites differs. The saved states are float16; direct float32-copy checks do not cover position −2.

The prompt suffixes are “ = ” for the three arithmetic families and “Answer:” for ARC-Easy. The local tokenizer decodes representative final pairs to [“ =”, “ ”] and [“Answer”, “:”], respectively. These positions are the same for both models, not missing arithmetic operands or missing science choices. The manuscript's shorthand is therefore substantively right but should name the tokens. Causal attention prevents −2 from incorporating the final separator token; both models are subject to that same restriction.

**E4 rule.** The matched-layer family-mean intervals exclude zero in all three seeds, as do comparisons against every fixed saved D layer. The strongest D family-mean site is layer 8 in seeds 0 and 2, and layer 12 in seed 1; its lower limits are 0.01416, 0.00117 and 0.00145. Thus the written rule is satisfied when “best saved site” means the strongest of these separately reported fixed-layer family means. No jointly nested site selector or per-family adaptively selected D comparator was estimated; do not describe the result as defeating such a comparator. Selection/multiplicity qualifications are already substantially present in Section 7.4 and should remain.

The stronger defensible statement is a repeatable positive **conditional family-mean own-error AUROC difference for these saved probes**, against all three saved fixed D layers. The weaker but misleading statement would be that the sign is unresolved in the stored conditional intervals: it is not. What remains unresolved is intrinsic state informativeness, architecture causality, robustness to refitting, and confirmation on new problems/checkpoints. The matched-layer per-family signs are all positive, but individual-family intervals are not uniformly unresolved: seed 0 add3w and ARC-Easy and seed 2 ARC-Easy exclude zero. Against D/prev8, seed 0 add3 even has a small negative point difference (−0.00142).

## Compliance with the earlier review

| Earlier required item | Status in this draft | Remaining work |
|---|---|---|
| 1. Conditional envelopes, family-before-seed aggregation, seed spread and site counts | Mostly met: Section 2, Appendix A and Section 3 site counts | Explicit order of family/contrast aggregation missing; change 10 |
| 2. Retain interaction/column/family effects; avoid equivalence; small increments; L4 positive secondary | Tables, interaction and L4 retained | ±0.02/flat language remains in Sections 3 and 8; change 4 |
| 3. Permutation median, all seeds, seed-0 anomaly, no claimed full-pipeline null pass | Numbers and anomaly retained | Unsupported explanation substituted for a demonstrated null mechanism; change 8 |
| 4. Threshold outcomes, restricted transfer, deep training confound, missing intervals | Threshold table and deep sample sizes included | Harmless/nothing-lost wording, 0.016 maximum, causal deep explanation, and 135-entry scope need changes 3 and 10 |
| 5. T1 path/displacement distinction, estimated-q caveat, conditional empirical score result, constant/availability/rounding | Much improved, both violation columns and constant comparator present | Path used to justify broad state-failure prose; marginal/transductive qualifications and float32-copy results missing; changes 5–6 |
| 6. No identified difficulty mechanism; descriptive transfer; no resolved dynamics increment; sample limitation a hypothesis | Section 7.2 and main dynamics discussion mostly meet this | Control-based inference and fitting-noise explanations still overreach; E1 trend language needs changes 7–8 |
| 7. Approximate 2e−5 provenance; stale U annotation; standardized PC fitting | Met at lines 666–671 | None; preserve frozen files |

The summary box is appropriately limited to one checkpoint pair and honest errors, but its “harmless,” “never wrong,” and “no better” formulations outrun those qualifications. Section 8 similarly reintroduces a flat-table/equivalence reading and universal budget language. Changes 3–6 make those passages consistent with the body.

The Section 6 derivation is algebraically sound under the stated contraction hypothesis with the explicit domain/nonnegative-q qualification in change 9. The implementation substitutes an estimated ratio; therefore the resulting numerical forecast is not the theorem's guaranteed bound. The missing assumption is not a missing factor in the geometric series.

## Hand-written numeric spot-checks

Source abbreviations: T = docs/monitor_results_gen/tables.json, identified by table/row/family (or key/field); C = docs/monitor_results_gen/curve/curve.json; E = docs/monitor_results_gen/explore/explore.json; P = docs/monitor_results_gen/explore/prev.json; R = results/mon_analysis-20261110/seed{0,1,2}/results.json, under analyses/analysis/seed/family/estimates/key. “Pass, approximate” means appropriate rounding or explicitly approximate prose, not exact equality. These are hand-written claims, not merely checks that generated table cells print themselves.

| # | Claim and source line | Checked against / observed value | Verdict |
|---:|---|---|---|
| 1 | Summary, 83: added value about 0.003–0.006 | T added/fixed L32,D confidence/added/family_mean: 0.00338, 0.00575 | Pass |
| 2 | Section 2, 147: nine penalties 10^−6…10^2 | Frozen spec and mon_probe.LAMS; nine powers | Pass |
| 3 | Section 3, 216: ±0.02 both compatible with the intervals | T table/diagonal and best/best_difference: [−0.017,0.025], [−0.025,0.018] | Fail as stated; change 4 |
| 4 | Section 3, 223: average own-state advantage about 0.009 | T table/interaction: 0.017; half ≈0.0085 | Pass |
| 5 | Section 3, 229–230: observed family diagonal differences below 0.03 | R m1a family diagonal seed medians ≈−0.0162, −0.0212, +0.0138, +0.0206 | Pass for point differences, not interval limits |
| 6 | Section 3, 234: 60 outer fits per model | 4 families ×3 seeds ×5 folds; T sites totals sum to 60 each | Pass |
| 7 | Section 4, 268: confidence 0.74 versus probe 0.69 | T added fixed confidence ≈0.743 each; probes 0.689/0.686 | Pass |
| 8 | Section 4, 275–276: 0.005 means one more pair in 200 | AUROC definition: 1/200=0.005, counting ties half | Pass as net expected pair-ordering improvement |
| 9 | Section 4, 280: output identity about 0.54 | T added/output_identity: 0.543/0.541 | Pass |
| 10 | Section 4, 281: privileged features about 0.60 | T added/privileged_ground_truth: 0.606/0.593 | Pass |
| 11 | Section 4, 281: early probe about 0.61 | T added/early: 0.621/0.602 | Pass as rough summary; preferably give both |
| 12 | Section 4, 311: one of six shuffled-label intervals excludes 0.5 below | R m1s family_mean/permutation: D seed 0 [0.457941,0.487190], other five include 0.5 | Pass |
| 13 | Section 4, 323: cross-family loss about 0.10–0.12 | T cross_mean single-probe minus-own: −0.1173/−0.0999 | Pass; these drops differ from ensemble target means |
| 14 | Section 4, 324: target cells near 0.47–0.71 | T cross source-target cells; upper example D add4→add3 ≈0.708 | Pass, approximate |
| 15 | Section 5, 355: mean native-transfer differences at most 0.007 | R m3a all registered sources, targets 4–32, seed-median family means: max abs 0.00667693 | Pass |
| 16 | Section 5, 356: per-family differences at most 0.016 | Same scan: L8 8→4/add4 =0.01830895; L32 32→4/add3 =0.01616038 | Fail; change 3 |
| 17 | Section 5, 357: accepted error as at 32 | T threshold L32/32→4 vs →32 accepted_error: 0.381 vs 0.355 | Fail literally; change 3 |
| 18 | Section 5, 358: agreement 91% at 4, 98% at 8 | T threshold label_changed: 0.087,0.022 | Pass for correctness labels only |
| 19 | Section 5, 359–361: 1-loop error 72%, changed labels 43%, gap 0.04, rejection 12% | T threshold L32/32→1: 0.718,0.427,0.043,0.120 | Pass |
| 20 | Section 5, 362: 2-loop rejection 31% versus 20% | T threshold L32/32→2 vs →32: 0.310,0.200 | Pass, rounded |
| 21 | Section 5, 365: about 205 versus 1,200 training problems | 256×4/5≈204.8; 1,500×4/5=1,200 | Pass |
| 22 | Section 5, 366: native/transfer at 64 =0.633/0.689 | T budget32 L32/native64 and L32/32→64 | Pass |
| 23 | Section 5, 367 and Section 9, 668: 135 unavailable intervals | Prior audit and R unavailable per-family entries; another 135 propagated family-mean entries | Incomplete scope; change 10 |
| 24 | Section 6, 407: path failures 42%–100% | T forecast/path_violation extrema 0.419 and 1.000 | Pass for path, not net displacement |
| 25 | Section 6, 413: bounds 180–2,600 times movement | T forecast/score_bound_over_realized.median extrema 182.604 and 2639.893 | Pass as approximate range; use about 183–2,640 |
| 26 | Section 6, 411–412: zero observed score violations | T forecast/score_violation: all displayed rows 0; R corresponding entries | Pass conditional on forecast availability |
| 27 | Section 6, 434–436: roughly 95% coverage and never more stable decisions than constant | T forecast_main scaled_rule coverage ≈0.950–0.959; L32/8 stable 0.929 vs constant 0.928 | Coverage approximately passes; universal comparison fails |
| 28 | Section 6, 465: dynamics-only 0.60–0.63 | T dynamics: 0.602,0.602,0.633,0.618 | Pass |
| 29 | Section 7.1, 484: full-size reproduces exactly | C reproduction max_abs_difference=0; its 24 references checked against R m1a/m1c | Pass |
| 30 | Section 7.1, 519–521: gains between E1 flat/sample-limited thresholds | C derived/family_mean last_doubling/probe =0.01408083,0.01875652 | Pass: neither ≤0.01 nor ≥0.02 |
| 31 | Section 7.1, 522: both intervals exclude zero, positive in every family | C last_doubling/probe: family means [0.00916,0.01892], [0.01249,0.02514]; eight family estimates 0.00868–0.02278 | Pass |
| 32 | Section 7.1, 523: each doubling gains about 0.015–0.02 | C family_mean probe: L32 gains 0.01518,0.01952,0.01408; D 0.03420,0.02254,0.01876 | Fail; change 7 |
| 33 | Section 7.1, 524: another last-doubling gain would put probes near 0.70 | C full AUROC + last gain: 0.70173/0.69950 | Arithmetic passes, conditional extrapolation only |
| 34 | Section 7.1, 525–526: confidence within 0.014 at 150 problems | C family_mean confidence full-minus-0.125: 0.013863/0.013649 | Pass for family mean |
| 35 | Section 7.1, 527–529: negative small-n added value; full-size adds 0.004; last added-value gain interval contains zero | C added at 0.125/0.25 negative for both; full 0.004400/0.004536; gain intervals [−0.000768,0.005272]/[−0.000429,0.007102] | Sign/interval claims pass; D rounds to 0.005, so give both |
| 36 | Section 7.2, 557–558: several hundred discordant problems per family | E joint counts b+c: 386,593,495,415 | Pass |
| 37 | Section 7.2, 560–561: cross-state rank correlations 0.80–0.94 sums, 0.69–0.70 ARC | E scorecorr seed medians: sums 0.798799–0.938301; ARC 0.692212/0.700882 | Pass, rounded |
| 38 | Section 7.2, 562–563: own/confidence correlation 0.30–0.48 sums, 0.71–0.80 ARC | E scorecorr seed medians: sums 0.297506–0.483744; ARC 0.708556/0.801753 | Pass, rounded |
| 39 | Section 7.2, 569: other model's error gives AUROC 0.60–0.70 | E joint binary-score AUROC =½[P(other wrong\|own wrong)+P(other right\|own right)], both directions: 0.59708–0.70046 | Pass, approximate |
| 40 | Section 7.4, 608: difference 0.02–0.04 | P matched layer12: 0.021403–0.037434; all saved layers: 0.020823–0.050663 | Pass only for matched layer; qualify |
| 41 | Section 7.4, 609: positive interval each seed and each D layer | P all nine family_mean rows; independently recomputed below | Pass |
| 42 | Section 7.4, 610: single families individually uncertain | P layer12: three of twelve intervals exclude zero | Fail if universal; say “most individual-family intervals include zero” |
| 43 | Section 7.4, 614–615: position −2 about 0.63 versus last about 0.69 | P L32 means 0.6300–0.6365; T fixed L32 ≈0.689 | Pass, approximate; does not prove no combined-position benefit |

## Optional suggestions

- Replace “a monitor would not choose to read here” (615) with “this position alone performs worse with this fitting procedure.” Joint use of the two positions was not tested.
- In the main confidence comparison, explain that confidence+probe is a jointly fitted regression on confidence features and state coordinates, not a second fit on a precomputed scalar probe score. This avoids a natural non-ML reader's misinterpretation.
- Say confidence is “available from the output without a state probe” rather than “costs nothing”: fitting and output-distribution summaries still involve computation.
- Label figures' bound/movement summaries “mean of family medians,” and keep distinct the median across seeds, mean across families, and within-family median over problems.
- Replace “lead and not a result” with “exploratory result, not confirmation.” The E4 result is numerically real as a conditional descriptive calculation; its interpretation is what remains limited.
- A future registered E4 test should use a crossed labels-by-states design at position −2, matched folds, broader site coverage, and independent problems/checkpoints. These are future-study suggestions, not requests for more computation in this revision.

## Full independent E4 numerical audit

The tables below contain the independently computed L32 and D AUROCs and differences, the stored intervals, and the independent-bootstrap intervals. AUROCs and differences agree with the stored point values exactly; displayed precision is six decimals. The last table is the compact family-mean result, for convenience.

### Individual families

| Seed | Family | D site | L32 AUROC | D AUROC | L32 − D | Stored 95% interval | Independent 95% interval |
|---:|---|---|---:|---:|---:|---|---|
| 0 | add3 | D/prev12 | 0.635681 | 0.599685 | 0.035996 | [-0.010021, 0.080478] | [-0.007150, 0.079468] |
| 0 | add3 | D/prev8 | 0.635681 | 0.637099 | -0.001418 | [-0.047926, 0.044090] | [-0.042907, 0.041685] |
| 0 | add3 | D/prev16 | 0.635681 | 0.556856 | 0.078825 | [0.032069, 0.124051] | [0.033342, 0.123002] |
| 0 | add4 | D/prev12 | 0.576147 | 0.544363 | 0.031784 | [-0.008814, 0.073023] | [-0.009078, 0.071891] |
| 0 | add4 | D/prev8 | 0.576147 | 0.563383 | 0.012764 | [-0.027438, 0.055207] | [-0.027874, 0.051638] |
| 0 | add4 | D/prev16 | 0.576147 | 0.553249 | 0.022899 | [-0.017095, 0.062406] | [-0.017142, 0.062861] |
| 0 | add3w | D/prev12 | 0.669107 | 0.629293 | 0.039814 | [0.004564, 0.076059] | [0.003301, 0.074903] |
| 0 | add3w | D/prev8 | 0.669107 | 0.618333 | 0.050774 | [0.015173, 0.086040] | [0.014212, 0.086893] |
| 0 | add3w | D/prev16 | 0.669107 | 0.620199 | 0.048908 | [0.012106, 0.083897] | [0.012654, 0.084943] |
| 0 | arc_easy | D/prev12 | 0.664657 | 0.622516 | 0.042141 | [0.004122, 0.078986] | [0.005406, 0.078971] |
| 0 | arc_easy | D/prev8 | 0.664657 | 0.593890 | 0.070768 | [0.031484, 0.109380] | [0.033974, 0.107125] |
| 0 | arc_easy | D/prev16 | 0.664657 | 0.612638 | 0.052019 | [0.012869, 0.090665] | [0.014245, 0.088788] |
| 1 | add3 | D/prev12 | 0.632368 | 0.609727 | 0.022642 | [-0.024113, 0.066269] | [-0.021620, 0.068767] |
| 1 | add3 | D/prev8 | 0.632368 | 0.604343 | 0.028026 | [-0.017709, 0.073471] | [-0.017179, 0.073467] |
| 1 | add3 | D/prev16 | 0.632368 | 0.569174 | 0.063194 | [0.016637, 0.107483] | [0.021147, 0.112056] |
| 1 | add4 | D/prev12 | 0.573409 | 0.567912 | 0.005497 | [-0.033874, 0.043890] | [-0.033206, 0.047528] |
| 1 | add4 | D/prev8 | 0.573409 | 0.572882 | 0.000527 | [-0.039232, 0.037734] | [-0.037776, 0.043251] |
| 1 | add4 | D/prev16 | 0.573409 | 0.568751 | 0.004657 | [-0.036813, 0.043129] | [-0.034113, 0.047554] |
| 1 | add3w | D/prev12 | 0.643213 | 0.619273 | 0.023940 | [-0.011872, 0.058259] | [-0.010493, 0.058684] |
| 1 | add3w | D/prev8 | 0.643213 | 0.626298 | 0.016914 | [-0.017169, 0.052690] | [-0.018304, 0.052140] |
| 1 | add3w | D/prev16 | 0.643213 | 0.614753 | 0.028460 | [-0.006945, 0.062973] | [-0.006435, 0.064822] |
| 1 | arc_easy | D/prev12 | 0.671102 | 0.637449 | 0.033652 | [-0.002826, 0.068442] | [-0.004429, 0.072062] |
| 1 | arc_easy | D/prev8 | 0.671102 | 0.616845 | 0.054257 | [0.018482, 0.089565] | [0.017309, 0.091172] |
| 1 | arc_easy | D/prev16 | 0.671102 | 0.631798 | 0.039303 | [0.001158, 0.075351] | [0.001008, 0.076229] |
| 2 | add3 | D/prev12 | 0.635550 | 0.626766 | 0.008785 | [-0.037410, 0.052097] | [-0.035142, 0.053289] |
| 2 | add3 | D/prev8 | 0.635550 | 0.632598 | 0.002952 | [-0.042808, 0.046082] | [-0.039353, 0.047956] |
| 2 | add3 | D/prev16 | 0.635550 | 0.570489 | 0.065061 | [0.018464, 0.109529] | [0.019126, 0.110706] |
| 2 | add4 | D/prev12 | 0.587943 | 0.578285 | 0.009658 | [-0.031340, 0.049622] | [-0.027315, 0.047137] |
| 2 | add4 | D/prev8 | 0.587943 | 0.585757 | 0.002186 | [-0.038725, 0.045310] | [-0.035673, 0.042316] |
| 2 | add4 | D/prev16 | 0.587943 | 0.571984 | 0.015959 | [-0.025002, 0.055408] | [-0.022517, 0.055359] |
| 2 | add3w | D/prev12 | 0.651254 | 0.629955 | 0.021299 | [-0.013266, 0.054819] | [-0.015525, 0.057950] |
| 2 | add3w | D/prev8 | 0.651254 | 0.635868 | 0.015387 | [-0.022417, 0.050124] | [-0.019633, 0.051844] |
| 2 | add3w | D/prev16 | 0.651254 | 0.614266 | 0.036988 | [0.002015, 0.073668] | [0.000596, 0.073780] |
| 2 | arc_easy | D/prev12 | 0.671080 | 0.625211 | 0.045869 | [0.009466, 0.083424] | [0.009234, 0.081936] |
| 2 | arc_easy | D/prev8 | 0.671080 | 0.608312 | 0.062768 | [0.028031, 0.098899] | [0.026751, 0.098844] |
| 2 | arc_easy | D/prev16 | 0.671080 | 0.618945 | 0.052135 | [0.016811, 0.089975] | [0.016492, 0.088350] |

### Family means

| Seed | Family | D site | L32 AUROC | D AUROC | L32 − D | Stored 95% interval | Independent 95% interval |
|---:|---|---|---:|---:|---:|---|---|
| 0 | family_mean | D/prev12 | 0.636398 | 0.598964 | 0.037434 | [0.017276, 0.056561] | [0.018179, 0.057110] |
| 0 | family_mean | D/prev8 | 0.636398 | 0.603176 | 0.033222 | [0.014165, 0.052347] | [0.013109, 0.052439] |
| 0 | family_mean | D/prev16 | 0.636398 | 0.585736 | 0.050663 | [0.030446, 0.070664] | [0.031090, 0.069661] |
| 1 | family_mean | D/prev12 | 0.630023 | 0.608590 | 0.021433 | [0.001169, 0.040324] | [0.002596, 0.041015] |
| 1 | family_mean | D/prev8 | 0.630023 | 0.605092 | 0.024931 | [0.005236, 0.044482] | [0.005700, 0.045112] |
| 1 | family_mean | D/prev16 | 0.630023 | 0.596119 | 0.033904 | [0.013778, 0.052917] | [0.014381, 0.054076] |
| 2 | family_mean | D/prev12 | 0.636457 | 0.615054 | 0.021403 | [0.001994, 0.040863] | [0.001890, 0.039779] |
| 2 | family_mean | D/prev8 | 0.636457 | 0.615634 | 0.020823 | [0.001452, 0.040721] | [0.001741, 0.039644] |
| 2 | family_mean | D/prev16 | 0.636457 | 0.593921 | 0.042536 | [0.023634, 0.062701] | [0.023675, 0.061684] |

