# Independent results review — 2026-11-10

Scope: read-only review of the registered design, implementation contract, frozen analysis, stored results and held-out scores. No probe refitting, GPU use or network access. Only this review and /tmp scratch files are written. The draft conclusions were included in the request, so this is not a literally blinded review; section A was written from the design and results before undertaking the point-by-point critique in C.

## A. Initial reading, recorded before the draft-conclusion critique

Numbers here use the supplied descriptive convention: median across split seeds and envelope of their conditional bootstrap intervals. These envelopes are not newly constructed 95% confidence intervals for the median.

- **M1-a:** No resolved overall advantage for either checkpoint. Own-state AUROCs are L32 0.687 and D 0.685; diagonal contrast 0.002 [−0.017, 0.025], column effect −0.003 [−0.010, 0.005], row effect 0.005 [−0.012, 0.025]. But the table is not completely flat: interaction 0.017 [0.005, 0.030], especially ARC-Easy 0.047 [0.018, 0.076]. The nearest registered outcome is “small added value over confidence in both,” qualified by a small own-state advantage. Neither registered directional column-effect interpretation is established. This is a comparison of two checkpoints, not an architecture intervention.
- **M1-b:** Best-site probes remain similar: 0.686 versus 0.687, difference −0.002 [−0.025, 0.018]. Site search does not establish a loop advantage. Conditional precision is useful but does not establish equivalence or include model-training variability.
- **M1-c:** Confidence baselines reach about 0.743; fixed-site additions are L32 0.003 [−0.001, 0.009], D 0.006 [−0.000, 0.012]. Flexible-baseline additions are 0.005 [−0.001, 0.010] and 0.001 [−0.006, 0.005]. Small trainable incremental value is the supported reading, not absence of information. Some individual seed intervals exclude zero.
- **Secondary M1:** Own-training-budget probes for L4/L8/L16/L32 average 0.680/0.672/0.664/0.689; these are different checkpoint/label comparisons, not a training-loop dose response. Position −2 performs worse (roughly 0.59–0.64). Concordant-item probes are stronger (0.763/0.765), with difference −0.001 [−0.017, 0.010], but this is an explicitly selected population. Mean-difference and PCA sensitivities do not establish a reliable architecture difference. Assess 80%-coverage error separately from ranking.
- **Cross-family M1:** Transfer is substantially weaker on average (approximately 0.57–0.59); single-probe target-versus-own differences are roughly −0.10 to −0.12. This is evidence against generalizing these family-specific monitors, not a clean estimate isolating distribution shift from target difficulty. Source-target heterogeneity matters.
- **M3-a:** Strong descriptive evidence for similar ranking performance at 4–32 loops; no universal guarantee of losslessness. L32 32→4 has native-minus-transfer 0.005 [−0.006, 0.013], while 32→1 has a resolved deficit of 0.043 [0.013, 0.065] and 42.7% changed labels. At 32→2 the deficit is 0.004 [−0.015, 0.024], despite 23.5% changed labels. Deep-budget comparisons use a smaller evaluation set and native training set. “Transfer holds everywhere” is therefore not supported literally; neither registered forecast-success/failure outcome is established at 1–2, outside T1’s evaluated forecast budgets. AUROC alone cannot settle frozen-threshold portability.
- **T1:** The raw estimated-ratio forecast is often too small for whole-state movement but extremely conservative for probe-score movement, with no observed score violations on forecastable items. For L32 at 4, whole-state exact violation is 0.832 versus score violation 0, with score bound/movement about 686. These facts do not verify contraction. Calibrated forecasts have near-nominal marginal coverage on forecastable items, but the registered comparison is against a calibrated constant, and practical usefulness requires that comparison, forecast availability and stable-decision rates. The forecast is not a certificate, and no 95% guarantee applies conditional on “stable.”
- **M4:** Looped dynamics-only AUROCs are roughly 0.60–0.63. No robust positive average increment over confidence is established; some point estimates are negative. This does not support the registered outcome that dynamics add output-independent error information. D’s stronger descriptive trajectory baseline is not an architectural comparison with matched features.

## B. Verification and registration audit

### Independent numerical checks

I used `/home/owner/work/.t3/projects/analysis/.venv/bin/python -I`, single-threaded numpy, without importing either analysis module or fitting any model. My AUROC implementation explicitly counted positive–negative score comparisons, awarding half credit for ties, separately in each saved outer fold; I then averaged the five AUROCs equally. Positive means **wrong answer**. I did not pool fold scores into one AUROC.

**All 240 checked AUROCs agree exactly at floating-point precision (maximum absolute discrepancy 0), hence at the displayed three decimals.** These comprise 48 M1-a cells (four cells × four families × three seeds) and 192 M1-c cells (fixed/best × L32/D × confidence/confidence+probe/flex/flex+probe × four families × three seeds). Equal-family means also agree to <1e−14. The appendix below gives all M1-a recomputations and the ordinary-confidence M1-c recomputations. The flexible versions were checked as well.

I reconstructed the L32/D budget-32 labels directly from raw `pid`, `top_id`/`ro_top_id` and `ans_tok` arrays, loading no hidden-state arrays. All 12,000 model–item labels matched the saved M1-a labels for each seed. All four M1-a cells share exactly the same outer folds. Fixed and best confidence-only score arrays (including flexible confidence) are exactly identical for each model/family/seed, as expected: there is no state-site choice in a confidence-only regression.

I independently reconstructed rule A from L32's raw `d_full`, `n_full`, and `disp_full` arrays for all four families at r=4,8,16. Forecast availability, path violations and exact whole-state displacement violations match `t1` to <1e−12. For example, all L32 r=4 path forecasts fail; exact displacement failures are 1.000/1.000/1.000/0.328 across add3/add4/add3w/ARC-Easy. At r=8 the path failures are 0.9993/0.9993/1.000/0.9933, but exact displacement failures are 0.3613/0.2493/0.5340/0.0300. Thus high path violations are real and must not be substituted for displacement violations.

I also reproduced bootstrap **class-loss counts**, independently of the project's bootstrap/AUROC functions, for add3 L32 native64, 32→64 transfer, and their contrast. Using the saved fold strata/intersections and registered 2,000 draws:

| Seed | Native undefined draws | Transfer undefined draws | Contrast undefined draws |
|---|---:|---:|---:|
| 0 | 1 | 87 | 77 |
| 1 | 1 | 102 | 60 |
| 2 | 0 | 4 | 2 |

These exactly match `undefined_bootstrap_draws` / `transfer_undefined_bootstrap_draws`. I did **not** independently regenerate every bootstrap endpoint, all fitted weights, or the entire collection; the agreement claims above concern stored-score point estimates, selected class-loss diagnostics and raw trajectory statistics.

### Summary and seed handling

`summarize_mon.py` reproduces the existing `summary.txt` byte-for-byte. Replacing its cell formatter with an independently written implementation also reproduces all **7,448 displayed cells**, including nested-key lookup, seed medians, interval envelopes and missing markers. I inspected the row-to-key mappings: M1-a row/column orientation, fixed/best and ordinary/flexible M1-c keys, M3 transfer versus native interval keys, cross-family source/target orientation and T1 nested fields are correct.

Important reporting details:

- The family-mean column is **median over seeds of the per-seed equal-family mean**, not necessarily the arithmetic mean of the four displayed family medians. Likewise median contrasts need not equal contrasts of displayed medians. For example, add3 displayed LL=0.713 and DD=0.721 do not imply its displayed diagonal contrast, which is −0.016. Compute contrasts before taking seed medians, as the script correctly does.
- Design §5 requires: “Repeated for three fixed split seeds; the spread over seeds and how often each site is chosen are reported.” §5 also says seeds indicate split variability separately from conditional bootstrap uncertainty. **No arithmetic mean, pooled-seed bootstrap, median or envelope is prescribed as the inferential combination of seeds.** The median/envelope is an added descriptive presentation, not a replacement registered estimator or a 95% interval for the median. There is no registered alternative pooling rule to restore.
- The compact view alone omits explicit seed spread and site frequencies. `report.md` includes per-seed values, min–max and site-selection counts, satisfying that requirement if retained and clearly linked. Include a short seed-range table in any main write-up. M1-a diagonal estimates by seed are 0.006906, 0.002035, 0.000266; best-site differences are approximately −0.00203, −0.00024, −0.00769. Do not treat the seeds as independent model replications.
- M1-b chooses L32 x32/post32 46/14 times across the 60 outer fits; D chooses h7/h12/h13/h14/h15/h16 1/10/21/14/11/3 times. Site search is performed inside training folds, not selected using the reported test scores.
- The summary suppresses several registered quantities for brevity: M1 accepted-answer errors, M3 `fraction_rejected` and `accepted_error`, and T1 `fraction_forecast_stable_with_allowance`, among others. They exist in the JSON/appendix. Include those needed for substantive conclusions rather than treating the compact summary as exhaustive.

### Audit and deviations

All four current file hashes (`analyze_mon.py`, `mon_probe.py`, `oversight_monitor.md`, `mon_spec.md`) match `main_manifest.json`. All three result files report `analysis_matches_manifest=true`, full 1,500-item coverage for every model/family, no skipped shards, no missing optional arrays, outer=5, inner=4, boot=2000, smoke=false, and the intended seeds. My inspection found training-only standardization, inner site/penalty selection, training-only spline knots, shared joint M1-a folds, paired fold-intersection bootstraps, fixed source weights/thresholds and backward-looking dynamics implemented as registered.

There are **zero recorded outer failures, zero inner failures and zero analysis skips** in every seed. Audit row counts per seed are M1-a 36, M1-b 16, M1-c 156, M1s 208, M1x 28, M3-a 1,308, T1 64, M4 80, f16 20, dyn 16. These are reporting entries, not counts of independent fits; cached/reused fits and contrasts can repeat fit information.

The **135 unavailable interval entries** in the compact audit are correct: 39/48/48 by seed, all M3-a/add3/deep targets 48/63/64. Summed undefined-draw counts are 660/828/237, counting repeated reporting entries rather than unique random draws. Another 39/48/48 unavailable intervals appear in the propagated `family_mean` rows; those are deliberately excluded from the compact per-family audit. So 135 is not the number of null interval fields across the entire JSON collection.

This handling follows registration: original held-out folds contain both classes, so the analyses are retained; **any** bootstrap draw losing a class invalidates the entire interval, without dropping the draw. Native add3 L32 at 64 has 209 correct and 47 wrong items. Seed 0 transfer fold class counts are (46,10), (44,11), (43,14), (36,3), (40,9); a fold with only three wrong items readily loses that class on resampling. Native folds have 9–10 errors but even one failed draw suffices. Point estimates remain valid descriptive estimates; corresponding family-mean intervals remain unavailable rather than averaging three available families. `?` accurately marks the missing interval, not a missing point estimate. Do not silently introduce a class-stratified bootstrap to rescue these registered intervals; any such sensitivity must be separately labelled exploratory.

Running the seeds in separate local/cloud CPU processes is not a deviation from a registered statistical rule: changing `jobs` from 6 to 16 changes scheduling, not the estimator. These are CPU analyses, not extra GPU collection launches. The stored partial cloud seed-0 comparison contains 30,126 comparable numeric leaves; maximum discrepancy is **2.0397393e−5**, at `t1/0/add3/estimates/L4/8/score_bound_over_realized/p90`. Thus “approximately 2e−5” is accurate, whereas a literal “≤2e−5” is slightly too strong. I did not independently verify remote environment histories or package installations; their matching versions are supplied execution provenance, supported by the stored cross-check, not proven by this review.

Two implementation-description qualifications deserve explicit disclosure rather than silently rewriting the frozen analysis:

1. `settings.baseline` calls U “unpenalized,” but the code correctly applies the registered U ridge of 1e−6 after whitening. This is stale metadata wording, not evidence that the actual fits omitted ridge.
2. The PCA sensitivity needs precise description. `_pca_path` forms unwhitened principal-component scores, then calls `fit_path`, which standardizes each component before ridge fitting. Consequently the effective penalty is on standardized PC coefficients, equivalent to scaling the PC scores to unit variance for fitting. The contract explicitly says to call `fit_logistic` after projection, so this is not a departure from that implementation contract; it does qualify the design’s shorthand “not whitened.” Do not describe this as ridge on unscaled PC scores. It does not affect the primary M1 results.

I found no established primary-analysis implementation deviation requiring refitting. Hash matching establishes consistency with the frozen files, not the absence of every possible bug. Collection gate outcomes and completeness are recorded, but I did not rerun the collection gates or exhaustively verify all 3.3 GB of arrays.

### Checks of unusual results

**Permutation diagnostic.** The draft’s D=0.47 is seed 0, not the seed median:

| Model | Seed 0 | Seed 1 | Seed 2 | Median; interval envelope |
|---|---|---|---|---|
| L32 | 0.51034 [0.49454,0.52561] | 0.49914 [0.48308,0.51464] | 0.51459 [0.49832,0.53078] | 0.51034 [0.48308,0.53078] |
| D | 0.47239 [0.45794,0.48719] | 0.50936 [0.49403,0.52489] | 0.49351 [0.47822,0.50867] | 0.49351 [0.45794,0.52489] |

D seed 0 is **not within its reported conditional 95% interval around chance**: 0.5 is excluded. It is nevertheless not evidence of leakage or an AUROC sign bug. The same fixed permutation is used for each split seed; the seeds are correlated, overlapping CV training sets make predictions dependent, and the stored-score bootstrap does not simulate repeating permutation and fitting. The 0.037 split-seed range is visible instability under this diagnostic. The all-seed envelope contains 0.5, but is not a formal full-pipeline null test either. Report the anomaly; a repeated-permutation/refit distribution would be needed to quantify whether it is unusually extreme under the whole procedure. No diagnostic-driven rerun or gate is registered.

**Privileged ground truth and output identity.** Their mean AUROCs (L32/D) are 0.606/0.593 and 0.543/0.541. These are plausible. Privileged features encode operands, true-answer digits/carries (ARC: correct letter and prompt length), not the event “this model’s prediction equals the answer.” They are not an oracle correctness feature or an upper bound on difficulty confounding. Output identity uses the saved full valid-token table, with an unknown-token indicator, and does not construct its vocabulary from held-out gold answers. `lp_ans` is not among the four confidence features. I found no label leakage in these feature definitions; I did not refit these controls.

**Native deep-budget probes below transferred probes.** Native48/63/64 train on only 256 deep items, approximately 205 per outer training fold; source32 probes train on approximately 1,200 items. Both are evaluated on the 256 deep items, but the native versus source fold assignments also differ. At L32 64 the mean native/transfer AUROCs are 0.633/0.689, native-minus-transfer −0.051 with unavailable family-mean interval; add3 is 0.589/0.710 with contrast −0.124. This is not a fair comparison of budgets at matched training sample size, and certainly does not show that more recurrence damages monitor information. The transferred predictions exclude each target item’s source training fold, so I found no direct target-item training leakage in this comparison.

**Path violations near one.** Independently reproduced above. A geometric bound derived with a true contraction q would bound the sum of future steps as well as net displacement. Here q is an estimate from earlier ratios, so its failure only refutes extrapolation with that estimated q. A larger valid contraction constant could still exist. Path length is also greater than net displacement, particularly when steps reverse or cancel. Do not equate `path_violation` with `exact_violation`, and do not infer “the model is not contractive” from either.

**Identical fixed/best confidence rows.** Correct: same confidence features, labels and own-label folds; best-site selection applies only to fits containing a state. The arrays were verified identical, not merely equal after rounding.

**Secondary positive result worth retaining.** L4 at its own budget has added confidence value 0.00485 with envelope **[0.000300, 0.011004]**, above zero in every seed. It is secondary and small, with no multiplicity correction, but a blanket statement that no model’s state ever adds anything would erase it. At 80% retained coverage, M1-a own-state accepted error rates by family are L32 0.162/0.525/0.428/0.302 versus original error rates 0.208/0.547/0.487/0.364; D 0.206/0.553/0.460/0.293 versus 0.257/0.589/0.513/0.346. Monitoring has some practical discrimination even though its incremental value over confidence is small.

## C. Critique of the eight draft conclusions

1. **Needs a caveat; numerical headline supported.** The reported 0.687/0.685 and contrasts are correct under the stated display convention. Replace “tight null” and “neither helps nor hurts … to within ±0.02–0.025” with: **“No overall advantage was resolved for this checkpoint pair; conditional interval envelopes for the four-family diagonal and best-site differences were [−0.017,0.025] and [−0.025,0.018].”** These are conditional on fitted probes, not equivalence tests or uncertainty over model training. Per-family intervals are roughly 0.09–0.10 wide, but not uniformly bounded by ±0.05 around zero: diagonal endpoints reach −0.067 and +0.073, and best-site endpoints reach −0.070. The unrounded best-site lower endpoint is −0.025409, so even a literal ±0.025 containment claim would fail for that envelope. The positive interaction belongs beside the null diagonal, not in a footnote.

2. **Supported as “small added value”; overstated as zero or literally flat.** Ordinary fixed-confidence additions are L32 0.00338 [−0.00126,0.00904], D 0.00575 [−0.00024,0.01210]; flexible additions differ (0.00459 and 0.00108), and best-site ordinary additions are about 0.00185/0.00491. Quote the actual mode. Some per-seed intervals exclude zero: L32 fixed/flex seed 0 [0.00080,0.00963]; D fixed/confidence seed 1 [0.00085,0.01069]. The registered “table flat, small added value” reading is a reasonable broad description of the absence of an overall column advantage, but the interaction prevents an unqualified “flat.” Confidence here is a trained four-feature output baseline, not a direct verbal self-report by the model. Avoid “probes add nothing.”

3. **Numbers supported; mechanism overstated.** Cross-state AUROC 0.680 versus own-state 0.687 for L32 labels is correct. The interaction 0.017 [0.005,0.030] sums the two own-minus-other advantages; the average own-state advantage is **half** this (about 0.0085). It is not specifically L32’s own-state advantage. ARC’s 0.047 contributes about 0.0118 to an equal-family mean of 0.017, hence “mostly ARC” is reasonable descriptively, but no formal interaction-heterogeneity test was registered. **Shared item-level predictability** is supported; “most of what the probe reads is item difficulty” is not identified. AUROC excesses are not an additive decomposition of information or variance, and both models’ states can encode other shared structure.

   As an exploratory descriptive check I computed the joint error table directly from saved labels:

   | Family | Both correct | L32 correct, D wrong | L32 wrong, D correct | Both wrong |
   |---|---:|---:|---:|---:|
   | add3 | 958 | 230 | 156 | 156 |
   | add4 | 351 | 328 | 265 | 556 |
   | add3w | 503 | 267 | 228 | 502 |
   | arc_easy | 760 | 194 | 221 | 325 |

   These counts explain the concordant subset sizes 1,114/907/1,005/1,085 and confirm substantial label overlap, while leaving the source of cross-state predictability unresolved.

4. **Overstated; the literal registered “holds everywhere” conclusion is wrong.** For saved targets 4–32, median four-family native-minus-transfer differences across all own-M/32 source combinations have absolute value at most 0.00668. That is useful evidence for good ranking portability. Per-family medians exceed 0.01 (L8 8→4/add4 0.01831, L32 32→4/add3 0.01616), and intervals do not establish ±0.01 equivalence. At 1 loop L32 has a resolved transfer deficit 0.043 [0.013,0.065]. At 2 loops many labels change, but a transfer-specific deficit is not clearly established: L32 0.004 [−0.015,0.024], and native performance also falls. Do not conflate falling absolute AUROC with failure relative to a native monitor.

   Frozen-threshold behavior is also material. L32 rejects 0.120 at 1, 0.311 at 2, 0.196 at 4, 0.207 at 8, and 0.200 at 32; accepted-answer error is respectively 0.706/0.457/0.381/0.353/0.355. L8 32→1 rejects 0.415 despite a near-zero median native-transfer AUROC gap. Therefore “budget changes are benign” is too broad even when rankings transfer. L32 labels change by 8.7% at 4 and 2.2% at 8 relative to 32; “the answer settles very early” needs a budget and tolerance. T1 forecasts were evaluated at 4/8/16, not at 1/2, so neither “forecast flags failure” nor “silent failure despite forecast” is established at the failing budget. Deep-budget results need the sample-size caveat in B.

5. **Wrong as a theory-validation claim; empirical findings need substantial qualification.** Replace “the theory’s bound is valid for scores” with **“The raw estimated-ratio forecast had no observed score violations on forecastable held-out items, but was loose.”** Zero observed violations is not a certificate, and the whole-state forecast failures do not disprove the existence of a contraction constant. Use exact displacement rather than path violation for the state comparison.

   Across model/budget family means, raw score bound/movement ratios span approximately **183–2,640**, not just hundreds. The primary calibrated `scaled_rule` has mean coverage about 0.950–0.959 on forecastable items. Its median-movement ratios span **3.04–9.22**; L16 at r=16 is the exception to “3–5.” Each family-mean ratio averages family medians, not all item ratios pooled.

   The registered comparison does not show a general practical improvement over `constant`:

   | Model / r | Scaled-rule stable fraction | Constant stable fraction |
   |---|---:|---:|
   | L32 / 4 | 0.773 | 0.773 |
   | L32 / 8 | 0.929 | 0.928 |
   | L32 / 16 | 0.927 | 0.981 |
   | L4 / 16 | 0.881 | 0.997 |
   | L8 / 16 | 0.977 | 0.995 |
   | L16 / 16 | 0.925 | 0.994 |

   Forecast availability at r=16 is only 0.884/0.984/0.939/0.945 for L4/L8/L16/L32, whereas the constant predictor is available everywhere. Better relative bound tightness on a selected forecastable subset is not necessarily better decision coverage. State-displacement forecasts may improve in relative tightness without improving monitor decisions; report these as separate outcomes.

   Quantization matters for late tiny movements: family-mean median allowance/movement at r=16 is 223.8/65.3/19.4/4.69; on the 32-item-per-family float32 subset, median relative movement changes average about 0.596/0.296/0.163/0.0255. Absolute probe-score rounding is small, but relative motion can be dominated by storage. Include both allowance-adjusted coverage and stability when claiming usability. Calibration uses held-out calibration halves, while thresholds use the same held-out features transductively. Nominal coverage is marginal on forecastable items, not conditional on being declared stable, and empirical coverage need not equal exactly 0.95 in every family. “Empirically calibrated and sometimes tighter, without a demonstrated general advantage over a constant forecast for stable decisions” is acceptable; an unqualified “usable theory bound” is not.

6. **Supported with wording change.** Dynamics-only means are L4 0.602, L8 0.602, L16 0.633, L32 0.618. Added values are −0.00292 [−0.00711,0.00218], −0.00016 [−0.00739,0.00586], −0.00052 [−0.00550,0.00428], −0.00443 [−0.00935,0.00149]. Say “no robust positive incremental AUROC was resolved,” not “add nothing.” L32 seed 0 is negative with interval [−0.00935,−0.00043], so even a uniformly zero-effect description misses variation. D dynamics reach 0.718, but its different trajectory/features preclude an architecture inference.

7. **Supported, with the registered comparison caveat.** Mean ensemble target AUROCs are L32 **0.5731**, D **0.5879**. Mean single-probe target-minus-own values are **−0.1173** and **−0.0999**. These are different, correctly defined quantities: do not compute the drop by simply subtracting ensemble target AUROC from own held-out AUROC. Report source-target cells; transfer to arithmetic from related arithmetic is sometimes much better (D add4→add3 0.708), while some cells approach chance. There is no interval for the grand `mean_over_targets` drop in the current output. Do not present its seed spread as an inferential CI or infer a resolved architecture difference from the two grand means.

8. **Necessary caveat, but sample limitation is a hypothesis rather than a demonstrated mechanism.** The full probes have d=2,048 and about 1,200 outer-training items, with inner fitting on roughly 900; large d relative to n is a plausible limitation. It does not prove performance would rise materially with more data. The PCA-128 probes reach only 0.670/0.671 versus full fixed probes around 0.689/0.686; this does not establish a learning curve or rule out sample limitation. Use: **“These results concern regularized linear probes trainable with this data and selection procedure; they do not establish that the state lacks further error information. Limited sample size is one possible explanation.”** Also retain one checkpoint pair, selected short-prompt families, honest mistakes, unequal model compute and conditional/no-refitting uncertainty. Those limitations are at least as important as d/n.

## D. Ranked exploratory additions using already collected data

All items below are **exploratory, not registered**. CPU estimates are rough core-time, informed by the stored seed-0 worker times (M1-a ~18 CPU-min, M1-b ~50 CPU-min, M1-c ~214 CPU-min), and depend on BLAS/hardware. No refitting was performed for this review. Long proposals would need a separate authorized execution setting; they are not instructions to exceed this review’s five-minute/two-core limits.

1. **Matched-population deep-budget and frozen-threshold audit — necessary before claiming broad budget safety.** Restrict stored source32 scores to exactly the deep IDs and frozen source folds, compare with the already stored transferred48/63/64 scores on those same items, and tabulate threshold coverage/accepted error by budget and family. This separates evaluation-population changes from state transfer; it cannot match native training sizes without fitting. Point checks cost seconds; paired score bootstraps roughly 1–5 CPU-min per compact panel. Present the existing threshold results even if no new bootstrap is run.
2. **Common-availability T1 comparison with constant, including quantization — necessary before claiming useful dynamics-based forecasting.** Use existing result summaries immediately; for paired common-item comparisons, reconstruct the small per-item predictions with the original saved fits if available. `scores.npz` does not save T1 bounds/weights, so do not claim it alone supports that paired analysis. Alternatively independently analyze state-displacement forecasts from the small trajectory arrays, which needs no probes. Report availability, coverage, stable fraction, false stability and absolute bound scale, with/without rounding allowance. Summary extraction is seconds; state calibration and bootstrap panels about 1–5 CPU-min; reconstructing missing score fits is extra work outside this review. A learning-free constant comparator is essential to the prose now.
3. **Joint errors and difficulty-adjusted association — necessary if retaining the proposed difficulty explanation.** The raw joint counts are already provided in C3 (seconds). Add correctness association and cross-state score associations; stratify by prespecified coarse, observable problem properties and confidence, or fit a small cross-fitted incremental model on existing scores. Roughly <1 CPU-min for tables and a few CPU-min for small-score regressions/bootstraps. This can strengthen evidence for shared difficulty but cannot causally identify “most”; controlling for another model’s errors is selected/privileged conditioning and must be labelled accordingly. This addition is optional if that mechanistic wording is removed.
4. **Learning curves — useful, not necessary if caveat 8 is stated as a possibility.** Refit nested fixed-site probes and confidence+probe at, for example, 25/50/75/100% of outer-training data with common held-out items and training-only selection at each size. Include confidence and a low-dimensional sensitivity, rather than just full-state AUROC. A minimal one-seed fixed-site panel should take tens of CPU-minutes; a three-seed augmented panel roughly 1–4 CPU-hours. Reduced-n fits test whether current performance is still improving with sample size; they cannot certify extrapolated information at arbitrarily large n. Do not manufacture a learning curve by subsampling only the test scores.
5. **Practical-equivalence sensitivity — optional and explicitly post hoc.** Choose a scientifically meaningful margin *before running this extra calculation*, while admitting the outcomes were already seen; this is not a preregistered equivalence test. Report each seed’s conditional 90% interval/TOST decision, plus a sensitivity over justified margins. Existing 95% envelopes already do not fit inside ±0.02; the unrounded best-site 95% envelope also extends just beyond −0.025 (to −0.025409). A 90% TOST interval is a different calculation; none includes refitting uncertainty. Recomputing a small set of paired bootstrap intervals from held-out scores costs roughly 1–5 CPU-min per seed/panel. Do not turn this into an unconditional “looping is equivalent” claim.
6. **Repeated full-pipeline permutation diagnostic — optional, needed only for a formal claim that D’s seed-0 anomaly is typical null noise.** Keep the registered permutation result; add perhaps 20 independent permutations through the same nested best-site pipeline, declared exploratory. Existing best-site worker timing suggests approximately 15–25 CPU-hours for a three-seed, all-family panel, so this is substantially more expensive than a label shuffle of frozen scores. The latter would not reproduce training/selection dependence and would not answer the same question. Reporting the observed split sensitivity without a formal null p-value is acceptable.

No additional model collection is required to report the restricted descriptive conclusions. I would not accept: “looping neither helps nor hurts” as an architecture-level or equivalence conclusion; “no information beyond confidence”; “most signal is difficulty” as identified; “transfer holds everywhere / budget changes are benign”; “the contraction assumption fails” based solely on these estimated-ratio failures; or “the theory bound is valid and calibrated dynamics are useful” without the availability/constant/quantization qualifications.

## Numerical appendix: independently reconstructed AUROCs

M1-a column order: L32 labels/L32 states (LL), L32 labels/D states (LD), D labels/L32 states (DL), D labels/D states (DD). Each entry averages the five within-fold pairwise-comparison AUROCs. All match the JSON exactly.

| Seed | Family | LL | LD | DL | DD |
|---|---|---:|---:|---:|---:|
| 0 | add3 | 0.716147 | 0.712085 | 0.706306 | 0.714601 |
| 0 | add4 | 0.604676 | 0.610058 | 0.591222 | 0.612986 |
| 0 | add3w | 0.690918 | 0.688232 | 0.677844 | 0.677114 |
| 0 | arc_easy | 0.738843 | 0.703106 | 0.705086 | 0.718259 |
| 1 | add3 | 0.713233 | 0.716960 | 0.712199 | 0.732119 |
| 1 | add4 | 0.589913 | 0.607588 | 0.597563 | 0.611066 |
| 1 | add3w | 0.694876 | 0.695081 | 0.682677 | 0.687024 |
| 1 | arc_easy | 0.750688 | 0.711224 | 0.703102 | 0.710363 |
| 2 | add3 | 0.705020 | 0.714717 | 0.702249 | 0.721185 |
| 2 | add4 | 0.593561 | 0.595353 | 0.602333 | 0.620428 |
| 2 | add3w | 0.705017 | 0.706769 | 0.678338 | 0.676168 |
| 2 | arc_easy | 0.736545 | 0.704908 | 0.705515 | 0.721298 |

M1-c ordinary-confidence checks. C=confidence; J=confidence+probe. “Fixed” and “best” refer to the probe sites. Confidence-only values are identical for those modes. Flexible-confidence and flexible+probe AUROCs were additionally checked in both modes (96 further checks).

| Seed | Family | L32 C | L32 fixed J | L32 best J | D C | D fixed J | D best J |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | add3 | 0.804247 | 0.808862 | 0.800336 | 0.791947 | 0.802865 | 0.802669 |
| 0 | add4 | 0.653154 | 0.653937 | 0.651967 | 0.657565 | 0.663228 | 0.662499 |
| 0 | add3w | 0.753229 | 0.761502 | 0.761502 | 0.738798 | 0.740076 | 0.741520 |
| 0 | arc_easy | 0.766062 | 0.769991 | 0.769991 | 0.783428 | 0.783712 | 0.784675 |
| 1 | add3 | 0.804459 | 0.812325 | 0.808886 | 0.792035 | 0.804927 | 0.794149 |
| 1 | add4 | 0.646627 | 0.639961 | 0.638571 | 0.661061 | 0.667813 | 0.662307 |
| 1 | add3w | 0.754047 | 0.766483 | 0.765202 | 0.737490 | 0.738887 | 0.744377 |
| 1 | arc_easy | 0.763333 | 0.763213 | 0.763213 | 0.783973 | 0.785946 | 0.783952 |
| 2 | add3 | 0.801612 | 0.809523 | 0.807972 | 0.788452 | 0.806409 | 0.796456 |
| 2 | add4 | 0.649195 | 0.642415 | 0.643360 | 0.660047 | 0.660283 | 0.673693 |
| 2 | add3w | 0.754118 | 0.762151 | 0.762151 | 0.739270 | 0.746564 | 0.751026 |
| 2 | arc_easy | 0.766398 | 0.770691 | 0.770691 | 0.783386 | 0.782399 | 0.781995 |

## Verdict and concrete corrections

**ACCEPT WITH CHANGES.** The primary stored-score computations are reproducible and the registered missing-interval rule is respected. Changes are needed to the interpretation and presentation; no primary refit is warranted by this review.

1. **`results/mon_report-20261110/summary.txt` and write-up:** label brackets as envelopes of per-seed conditional 95% intervals, not a 95% interval for a seed median. Explain that `family_mean` is aggregated before taking seed medians. Retain `report.md`’s seed ranges and `chosen`/site-selection counts. `summarize_mon.py:cell` is numerically correct and need not be changed to another seed-pooling rule.
2. **M1 interpretation, `seed*/results.json → analyses.m1a.<seed>.<family>.estimates`:** retain `column`, `interaction` and per-family `diagonal` beside the overall null; avoid equivalence/causal architecture wording. `m1b/.../best_difference` has unrounded envelope [−0.025409,0.017924]. `m1c/.../fixed|best/<model>/confidence|flex/added` supports small, not universally zero, increments. Include the secondary `m1s/.../L4/4/added` result.
3. **Permutation, `m1s/.../D/permutation`:** replace an unqualified D=0.47 with median 0.493515, show all seeds, and disclose seed-0 interval [0.457941,0.487190]. Do not assert the anomaly has passed a full-pipeline null test.
4. **Budget conclusions, `m3a/.../<model>/<source>-><target>`:** report `fraction_rejected`, `accepted_error`, `label_changed` and `native_minus_transfer`. Restrict near-lossless ranking language to the supported budgets and precision. Explain 256-item deep native training versus 1,500-item source training; preserve `transfer_interval`, `interval`, and unavailable reasons. Clarify that the 135 audit entries exclude another 135 propagated family-mean interval entries.
5. **T1, `t1/.../<model>/<r>`:** distinguish `path_violation` from `exact_violation`, estimated-q failure from contraction failure, and empirical `score_violation=0` from a certificate. Compare `score_conformal.scaled_rule` to `constant` using `forecast_fraction`, `coverage`, `fraction_forecast_stable`, `false_stability` and allowance-adjusted quantities. Replace universal 3–5× with the actual 3.04–9.22 mean-of-family-medians range; disclose `rounding_subset.relative_movement_change` and `allowance_over_movement` at late budgets.
6. **Mechanism, transfer and dynamics:** `m1a` off-diagonals do not identify “most signal is difficulty”; `m1x/.../mean_over_targets/*_minus_own_family` is descriptive single-probe target-minus-own performance; `m4/.../added` supports no resolved positive increment, not zero information. Describe d/n sample limitation as a plausible explanation rather than a measured learning-curve result.
7. **Execution/method wording:** describe the cloud seed-0 agreement as approximately 2e−5 (observed maximum 2.0397393e−5). Correct the stale description of U in `analyze_mon.py:settings.baseline` through an explicit reporting annotation, preserving the frozen analysis provenance. Clarify standardized-PC fitting in `mon_probe.py:_pca_path`/`fit_path`; do not claim ridge on unscaled PCA scores. These are documentation corrections, not requests to alter the frozen results.

No existing repository files other than this newly created review were changed. Scratch verification scripts and outputs were confined to /tmp. Remote execution provenance, full refitting uncertainty, all bootstrap endpoints and all raw-state arrays were not independently revalidated.
