# ACCEPT WITH CHANGES

The two operational verdicts survive independent recomputation. The late-shift screen has no accepted family under the written eligibility rule; the lever experiment meets R1 and R2 (and secondary R3) at all three doses. The principal numerical tables reproduce at their displayed precision. The write-up needs substantive interpretive changes, especially its claim that the response is complete after one loop. No new GPU experiment is needed to correct the report.

Review scope: I read the designs/amendments, pre-run reviews, collection/analysis/push-building code and supplied results. I independently computed the statistics from NPZ arrays using NumPy/SciPy, without importing or executing either production analyzer. Bootstrap intervals below use 5,000 problem resamples with seed 0, matching the specified analysis convention; correlations, slopes and errors were computed directly from matrix products and centered sums. Only this report was added to the project; scripts/output were placed in /tmp. No Modal calls, GPU jobs, project edits or commits.

## Findings, ordered by severity

1. **High — “complete after the first forced loop” and “the loop's part is one application” are contradicted by the full response.** The mean target effects are correctly quoted, but a near-equal mean is not an equal response matrix. At the primary dose, median across problems of `||ΔZ(t)−ΔZ(48)||F / ||ΔZ(48)||F` is **0.251045 at t=1**, 0.094351 at t=2, 0.016041 at t=4, 0.000482899 at t=8, and 9.5193e-7 at t=16. The t=1 mean itself exceeds the endpoint by 3.51%. A tiny 48th Neumann term does not show that J² and subsequent terms are negligible; neither does it measure the leading eigenvalue of the full Jacobian. Replace the entire first “What differs” bullet with:

   > **The mean target effect emerges quickly, but later loops still change the response.** The mean lever effect is 0.171 after one forced loop and 0.165 after 48. However, the full response matrix at one loop differs from its endpoint by a median 25.1% in relative Frobenius norm; this falls to 1.60% after four loops and 0.0483% after eight. The final Neumann-term ratios are below 1e-12 for the sampled directions. These results establish rapid settling of this local forced response, not equivalence to a single block application or a spectrum-wide contraction rate.

   The Huginn reference's 0.78 is a **median largest observed Ritz modulus**, with convergence qualifications, not a universal “largest factor.” If retained, use: “A separate Huginn analysis reported a median largest observed Ritz modulus of 0.78; no matched spectral comparison was performed here.” The six-layer block interpretation is architecturally correct, but reducing the observed replication to just one pass is not. Conversely this is a local intervention experiment around a state already reached after 64 loops, not evidence that recurrence improves reasoning or monitoring.

2. **High — immediate effects are linearized predictions, not observed finite-push effects; post hoc labeling points to the wrong items.** The requested products reproduce: lever −0.00713525, negative in 219/400 pairs (54.75%); gradient +0.96079019. There is no stored t=0 nonlinear readout under the finite push. Replace the second bullet with:

   > **The immediate readout direction has very little measured target effect after a loop.** The post hoc linearized immediate effect, ρ m_ref G_jk·P_k, averages +0.96079 for gradient pushes and −0.00714 for lever pushes (negative in 54.75% of pairs). These are derivative predictions, not measured finite-push t=0 effects. After one forced loop the measured mean target effects are −0.000734 and +0.171187 respectively. IDENTITY has weak positive median correlation 0.114 with the endpoint response, versus −0.587 in Huginn. This supports a related readout-versus-transmitted-direction pattern, without establishing an identical mechanism.

   “Erased” overstates a nearly zero **mean target** effect, which does not bound every class response or each individual pair. “Uncorrelated” should be “weakly positively correlated”; no null-correlation inference was specified. Replace the section's parenthesis “the last two items computed after seeing the results” with: “Descriptive interpretation; time courses and predictor comparisons were prespecified, while the immediate derivative summaries and the response-matrix time-course comparison were computed post hoc.” The design did prespecify t=1-to-end reversal; the immediate-to-one-loop comparison was not that registered statistic.

3. **Medium — the monitoring stopping decision is valid, but its scientific scope and narrow mul2add failure need stating together.** All confirmation requirements except D's accuracy pass for mul2add: 52/300 changes, Wilson lower bound 13.4701%, L32 accuracy 33/300, and 1/300 changes from 31 to 32. D is 22/300: eight successes short of the fixed 30/300 floor. Its descriptive Wilson interval is 4.8924–10.8531%, crossing 10%; rejection is a point-estimate eligibility decision, not proof its population accuracy is below 10%. Replace the result paragraph's ending with:

   > Neither family was accepted, so we stop this monitoring comparison under the fixed rule and budget. Mul2add passed every confirmation condition except D's accuracy: 22/300 (7.3%) versus the required 30/300 (10%). L32 achieved 33/300 (11.0%). This narrow eligibility failure does not establish that no suitable task exists. A monitoring experiment at roughly 11% accuracy could study discrimination of rare correct answers, but would have few correct cases and would not by itself demonstrate useful late computation; 47 of its 52 changes were wrong-to-wrong.

   Do not relax the floor retrospectively. “Only there would ... be a fair question” should become “The fixed screen used these thresholds to decide whether to pursue the monitoring comparison.” This is an operational choice, not a necessary scientific condition for every monitoring experiment.

4. **Medium — descriptive late-shift bullets overstate stability and absence of benefit.** “Stable from loop 16 on everywhere” is false literally and endpoint comparisons do not exclude intervening changes. The largest screen c(16,32) is 4/64 = 6.25%, c(31,32) 1/64 = 1.5625%. The fixed not-flicker condition is met by all screen families; both candidates pass. Replace the first three bullets' leading claims with:

   > Late changes occur in several families. Both candidates pass the fixed not-flicker criterion. In the screen, 16→32 changes range up to 6.25% and 31→32 changes up to 1.5625%; these endpoint comparisons do not establish uninterrupted stability after loop 16.

   > Most changed answers remain incorrect: 116/133 screen changes are wrong-to-wrong, with 10 wrong-to-right and 7 right-to-wrong. This is a predominance, not a statement that all changes occur on unsolved problems.

   > No family shows a statistically resolved net accuracy gain from loops 8 to 32 in this sample. The largest observed screen gain is count, 30/64 to 33/64, with four wrong-to-right versus one right-to-wrong transition (exact McNemar p=0.375). This does not establish equivalence or exclude small benefits.

   Confirmation also has unresolved positive gains: mul2add +1 percentage point (p=.375); story2 +1.333 points (p=.423950). The aggregate screen improvement is 3/1152, not zero.

5. **Medium — the margin sentence mixes run/time summaries and implies universality.** All quoted magnitudes are plausible approximations, but 0.03 versus 0.8 for story2 corresponds to the **confirmation at loop 8**, whereas the clearest 0.06 versus 0.3 mul2add pair is **confirmation at loop 32**. The screen story2 loop-32 unchanged median is 0.736855, conventionally 0.7 to one decimal. Replace with a fully specified comparison:

   > In confirmation, median top-1 minus top-2 log-probability margins at loop 32 are 0.05984 versus 0.29951 nats for changed versus unchanged mul2add problems, and 0.04117 versus 0.84389 for story2. Thus changed answers tend to have smaller margins in these families; this does not say that every changed answer is a near-tie.

6. **Medium — cross-model claims need comparison controls/caveats.** Huginn's table entries are correctly quoted from its summary JSON. The protocols differ in task (digit task versus sums 70–79), sample size (20 versus 40 evaluation problems), precision (float32 versus float64), directions and dose (absolute norm 3 versus 0.746585 at the primary dose). The table can summarize two experiments but cannot isolate model differences. Add immediately before it:

   > These are descriptive results from different tasks, sample sizes, precisions and push doses, not a controlled comparison of models. Intervals on FULL-minus rows bootstrap the mean paired per-problem correlation difference; displayed standalone correlations are medians.

   Replace “The operator is more problem-specific” with: “Own-problem operators outperform the averaged and selected swapped operators in this experiment. The observed penalties are larger than in the Huginn run, but the different protocols prevent attributing that difference to the model alone.” Replace “Smaller effects per unit of push” with: “At the tested doses the mean effect per absolute push norm is 0.2215 here versus 0.5923 for Huginn; state scales, tasks, readouts and dose nonlinearity differ, so this is not an intrinsic steering-efficiency comparison.” Alternatively delete the cross-model efficiency claim. The current body explicitly says sizes are not directly comparable, contradicting its own heading.

7. **Medium — section 4 should distinguish the operational result, local mechanism, and incomplete model search.** Replace all three bullets with:

   > We stop the monitoring comparison for this pair under the fixed eligibility rule and budget. On the original monitor tasks, answers change on 3.5% of problems between loops 8 and 32; no candidate passed every confirmation criterion. This does not imply that the hidden state is idle or that no suitable task exists.

   > The prespecified lever-advantage and loop-aware-prediction pattern holds in both tested looped models under their respective protocols. On L32 the primary-dose median prediction error is 7.67% with correlation 0.997 and slope 1.009. This supports local linear response around the settled state, not a demonstrated benefit of recurrence for task solving or a proof of an identical mechanism.

   > The documented search has not established a second fully verified downloadable matched pair. Release of the ordinary OLMo or TinyLlama partner by the authors would supply one route; verifying other candidate pairs remains another.

   Section 3's “The only fully matched downloadable pair remains...” should likewise be “The Llama pair is the only fully verified downloadable matched pair established by this search.” `docs/pairs_check.md` explicitly bounds its search and lists an unverified released pair. “Needs the authors to release” is not the only possible route. I checked this against the supplied audit, not a new live repository census.

8. **Low — correct the G1 maximum and clarify the test's scope.** Change “harness against the model's own forward 0 to 6e-14” to “harness against the patched model forward: centered-logit maximum errors 0 to 9.60e-14.” The largest observed error is **9.592326932761353e-14** in evaluation. Train and pred both have exactly [0,0], plausibly: the harness and model forward execute the same operations with matching batch shape, inputs and precision patches; deterministic kernels can agree bit-for-bit. Code calls `M(... input_states=xb.clone(), num_steps=t)` separately at r=4 and 32, rather than comparing the harness to itself. G1 is nonvacuous, but shared norm/rotary patches are not independently validated by agreement between the two patched paths. Smoke's different shapes give up to 6.0396e-14; these are mutually consistent. This is source/metadata validation, not a new execution of trained weights.

9. **Low — registration chronology is misstated.** Draft 1.3 explicitly follows the smoke run. Replace the opening claim with: “The scientific decision rules and amendments were documented before the production runs. Lever Draft 1.3 followed the smoke run and changed evaluation sharding in response to measured runtime; it retained the full dose design and T=48 under the earlier smoke rule.” This does not invalidate the verdict. It avoids claiming that every amendment preceded every run. No evidence found of outcome-dependent changes to the scientific thresholds in the supplied record; document labels alone are not an independent timestamped registration archive.

10. **Low — baseline equality and relative-step range need numerical precision.** Existing-data pooled L32 accuracy is 3593/6000 = 59.8833% at 8 and 3591/6000 = 59.8500% at 32: both display as 59.9%, but are not identical, nor are all family accuracies identical. Replace “with the same accuracy” with “with pooled accuracy 59.88% and 59.85%, respectively.” The baseline 6,000, 3.5%, and 1.8–5.2% family range agree with the supplied existing-data summary. Replace the screen relative-step range with “family-median relative steps range from 0.0132 to 0.0955 at loop 8 and from 1.22e-5 to 2.39e-3 at loop 32.” The current values are acceptable orders of magnitude, but 2e-3 is not a literal upper bound.

## Checks that do not require changes

- All screen table counts, rounded percentages, Wilson intervals and verdicts reproduce. All 18 families have 64 paired observations. Confirmation contains exactly the two candidates, each with 300 paired observations, indices 64–363, matched to the master manifest, with no screen-index reuse. Master SHA-256 is df7e737cdd431b8590fc0822e70947defcfccd61e8c17851543d5a8756818901, matching metadata. The finite-array check excludes the intentional rel_step index-zero sentinel.
- Screen G1: 16 records, max 2.52246039e-4, six right-padded comparisons; G2: eight records, max 6.68916126e-5, three right-padded. Families: add3, acc5, story4, arc_chal. Confirmation G1: eight records, max 9.14849422e-5, two right-padded; G2: four records, max 5.66424686e-5, one right-padded. Both available confirmation families are covered. All four containers complete; G3/G4 recorded true. Gates are comfortably below 1e-3.
- Eval merged arrays match concatenation of the four saved shards exactly, preserving shared row/time specifications. All shards complete. Eval and pred share the exact push bytes, SHA-256 f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f, also matching the root push artifact. All 32 direction vectors reconstruct exactly from the supplied train/fit arrays and seeded random generator (max error 0); m_ref reconstructs exactly. Train/fit and eval PIDs, prompts and `(ctx,a,u)` cells are disjoint. Pred evaluation ordering matches eval; pred training ordering matches training gradient PIDs. T=48 in smoke, train, pred, every eval shard, push artifact and final saved time. K0=64 throughout. All saved lever floating arrays are finite. Answers at 32 and 64 agree throughout the saved stages.
- Primary R1: L's own and L−G intervals strictly positive; R2: median FULL correlation >.90 and FULL−IDENTITY interval positive. Secondary R3 passes. These also pass at both secondary doses. “HUGINN PATTERN REPLICATES” is therefore justified **as the explicitly limited registered label**.
- Timing reproduces: late-shift function seconds sum to 550.8375 = 9.1806 minutes (screen alone 321.3069 = 5.3551); lever smoke/train/pred/four eval shards sum to 720.7023 = 12.0117 minutes. These are recorded function runtimes, not an independent billing audit.
- The 64-problem screen misses a true 10% change rate more than half the time before usability/flicker exclusions; the existing limitation is appropriate. Neither late-shift rule failure nor nonsignificance is an equivalence result.

## Most valuable next step

First, use the saved response matrices to report the time-resolved full-matrix discrepancy (already computed above), rather than deciding recurrence's role from a scalar mean. If one small follow-up is warranted, freeze a matched one-step linear predictor `G J L` versus `G (J+…+J^48) L` on the existing held-out problems and directions, at the existing low and primary doses; compare both against the saved endpoint responses with paired problem-level intervals. Only the J L calculation needs new model work, and none was run in this review. This directly tests how much repeated application contributes beyond passage through the six-layer block, resolves the central mechanistic ambiguity cheaply, and is more informative than relaxing the monitoring floor or starting another broad task screen.

## Reproduced values

The independent numerical transcript follows. For late-shift rows, all entries before Wilson intervals are **counts**: divide by 64 for screen rates and 300 for confirmation rates. Transition vectors are `[wrong→right, right→wrong, right→right, wrong→wrong total, wrong→wrong changed]`. Wilson intervals are percentages. Primary-dose lever intervals are problem-bootstrap intervals of means; all predictor scores are medians across problems. Small last-digit adjoint differences reflect matrix-product summation order, not a discrepancy.

Additional derived quantities: screen aggregate changes 133 = 116 WW + 10 WR + 7 RW; confirmation mul2add 52 = 47 WW + 4 WR + 1 RW. L dose ratios 0.3343164460 and 2.9030755015. The Huginn column agrees with summary values L=1.7768999255, G=−0.0460268962, C=−0.0463415611; FULL correlation/slope/error=0.9779327602/0.9494735211/0.2213817601; AVERAGED=0.9467317891, SWAPPED=0.9382945999, SAME=0.9306384457, IDENTITY=−0.5868565004; FULL-minus intervals IDENTITY [1.4770720247,1.5872411435], AVERAGED [0.0222561655,0.0400353202], SWAPPED [0.0303256826,0.0625351983], SAME [0.0339895877,0.0563233716]. These Huginn numbers were checked against the requested summary JSON, not independently regenerated from Huginn raw arrays.

```text

RUN lateshift-20261110-171533 indices 0 63
family | changes 8,16,31→32 counts | correct 8,32,D counts | WR,RW,RR,WW(changed) | Wilson % | verdict
add3 [0, 1, 0] [51, 51, 48] [0, 0, 51, 13, 0] [0.       5.662406] SETTLED McNemar 1
add6 [4, 1, 1] [8, 8, 6] [0, 0, 8, 56, 4] [ 2.45712  14.997485] SETTLED McNemar 1
chain6 [5, 0, 0] [3, 3, 1] [0, 0, 3, 61, 5] [ 3.383118 17.019537] SETTLED McNemar 1
mul2add [13, 2, 0] [7, 8, 7] [1, 0, 7, 56, 12] [12.273418 31.713636] QUALIFIES McNemar 1.0
margin 8 0.0583953857421875 0.2654705047607422
margin 32 0.0660552978515625 0.2728080749511719
paren [11, 3, 1] [2, 1, 0] [0, 1, 1, 62, 10] [ 9.877742 28.213212] CHANGES BUT UNUSABLE McNemar 1.0
var4 [28, 1, 0] [0, 0, 1] [0, 0, 0, 64, 28] [32.29401  55.913791] CHANGES BUT UNUSABLE McNemar 1
var4s [26, 4, 0] [1, 0, 0] [0, 1, 0, 63, 25] [29.456909 52.854792] CHANGES BUT UNUSABLE McNemar 1.0
ptr3 [9, 0, 1] [9, 8, 10] [0, 1, 8, 55, 8] [ 7.578573 24.616282] CHANGES BUT UNUSABLE McNemar 1.0
swap [5, 0, 0] [14, 16, 18] [2, 0, 14, 48, 3] [ 3.383118 17.019537] SETTLED McNemar 0.5
acc5 [5, 1, 0] [0, 0, 1] [0, 0, 0, 64, 5] [ 3.383118 17.019537] SETTLED McNemar 1
count [6, 0, 0] [30, 33, 24] [4, 1, 29, 30, 1] [ 4.367825 18.98288 ] SETTLED McNemar 0.375
sort2 [0, 0, 0] [8, 8, 12] [0, 0, 8, 56, 0] [0.       5.662406] SETTLED McNemar 1
bool [1, 0, 0] [42, 43, 40] [1, 0, 42, 21, 0] [0.276354 8.334102] SETTLED McNemar 1.0
story2 [7, 1, 0] [34, 33, 24] [1, 2, 32, 29, 4] [ 5.400113 20.898641] BORDERLINE McNemar 1.0
margin 8 0.05194664001464844 0.710861325263977
margin 32 0.032337188720703125 0.7368545532226562
story4 [6, 2, 0] [0, 0, 0] [0, 0, 0, 64, 6] [ 4.367825 18.98288 ] SETTLED McNemar 1
gsm [2, 0, 0] [11, 11, 3] [0, 0, 11, 53, 2] [ 0.861214 10.697292] SETTLED McNemar 1
svamp [2, 0, 0] [42, 43, 30] [1, 0, 42, 21, 1] [ 0.861214 10.697292] SETTLED McNemar 1.0
arc_chal [3, 0, 0] [35, 34, 27] [0, 1, 34, 29, 2] [ 1.606902 12.899654] SETTLED McNemar 1.0
relative step ranges [(8, np.float32(0.01316754), np.float32(0.09553526)), (32, np.float32(1.2221741e-05), np.float32(0.0023853425))]
gate L32 complete 16 0.00025224603931306433 padded 6 families {'acc5', 'arc_chal', 'add3', 'story4'} seconds 286.551175064
gate D complete 8 6.689161264006316e-05 padded 3 families {'acc5', 'arc_chal', 'add3', 'story4'} seconds 34.75572953299999

RUN lateshift-20261110-172550 indices 64 363
family | changes 8,16,31→32 counts | correct 8,32,D counts | WR,RW,RR,WW(changed) | Wilson % | verdict
mul2add [52, 17, 1] [30, 33, 22] [4, 1, 29, 266, 47] [13.47006  22.022614] NOT ACCEPTED McNemar 0.375
margin 8 0.07523155212402344 0.30628395080566406
margin 32 0.05984306335449219 0.2995142936706543
story2 [22, 4, 0] [135, 139, 110] [9, 5, 130, 156, 8] [ 4.892399 10.853134] NOT ACCEPTED McNemar 0.4239501953125
margin 8 0.033438682556152344 0.7904367446899414
margin 32 0.04117107391357422 0.8438949584960938
relative step ranges [(8, np.float32(0.023874354), np.float32(0.027227787)), (32, np.float32(2.2337172e-05), np.float32(0.0001524844))]
gate L32 complete 8 9.148494219957115e-05 padded 2 families {'story2', 'mul2add'} seconds 110.29414794899999
gate D complete 4 5.664246859415911e-05 padded 1 families {'story2', 'mul2add'} seconds 16.720671330000002

LEVER m_ref 24.886178125323305 norm 0.7465853437596991
push max reconstruction error 0.0 m_ref error 0.0 disjoint True
dose 0.01 L 0.05529122608634536 [0.051425274877418894, 0.05911832057808874] course [0.05711677 0.05453324 0.05539954 0.05529273 0.05529172 0.05529123]
dose 0.01 G -0.0003316177833876699 [-0.0007205523150066341, 7.87344960741637e-05] course [-0.00024398 -0.00045133 -0.00034362 -0.00033107 -0.00033187 -0.00033162]
dose 0.01 C -7.68676299778237e-05 [-0.0005897973356774875, 0.0004188699774174756] course [ 6.10461678e-06 -1.64139363e-05 -7.96665545e-05 -7.66168785e-05
 -7.68780729e-05 -7.68676300e-05]
diff effects [0.051663969672648406, 0.05949439612460021] [0.051490856472697896, 0.05929669329840642]
FULL corr/slope/error 0.9996625685237598 1.0036451886599604 0.02641585903230411
AVERAGED corr/slope/error 0.937276926463034 1.0029436643186087 0.37121827185808476
SWAPPED corr/slope/error 0.9224219103434145 0.9185751740454686 0.4056140990436008
SAME corr/slope/error 0.8149364219645434 0.8198322112738511 0.6742197215293251
IDENTITY corr/slope/error 0.1170574392833022 0.2945372245945347 1.0260939390121266
FULL-minus AVERAGED [0.058326357958716435, 0.07264711175785146]
FULL-minus SWAPPED [0.06780027078593531, 0.10039821053140004]
FULL-minus SAME [0.16633191524602153, 0.22594914311093953]
FULL-minus IDENTITY [0.7906126083503554, 0.9126480639314279]
dose 0.03 L 0.16538589931717174 [0.15369802394776338, 0.17716627910913196] course [0.171187   0.16316198 0.16577546 0.1653912  0.16538744 0.1653859 ]
dose 0.03 G -0.000993335581060104 [-0.0021683905528712896, 0.00023659640140384814] course [-0.00073361 -0.00134956 -0.00102884 -0.00099168 -0.00099411 -0.00099334]
dose 0.03 C -0.000227991308717912 [-0.0017709358168529039, 0.0012659462304124265] course [ 2.07755126e-05 -4.65459040e-05 -2.35566343e-04 -2.27230552e-04
 -2.28024258e-04 -2.27991309e-04]
diff effects [0.15447594921485513, 0.17836999741071172] [0.15368679631758186, 0.17786749655446432]
FULL corr/slope/error 0.9970775476468046 1.0094794726973446 0.07674304585284891
AVERAGED corr/slope/error 0.9365169568307502 1.0024557752734724 0.36679853497911746
SWAPPED corr/slope/error 0.9197770536676056 0.9337294087394103 0.40914820618945286
SAME corr/slope/error 0.81769293177348 0.8220979941880864 0.6834421727174242
IDENTITY corr/slope/error 0.11433972537703341 0.2909231038837471 1.0242367960851095
FULL-minus AVERAGED [0.05667951460310523, 0.07192385008747132]
FULL-minus SWAPPED [0.06782908586769278, 0.10075766747422497]
FULL-minus SAME [0.16401433859230616, 0.22365266276924928]
FULL-minus IDENTITY [0.7874009510426395, 0.9058036119267053]
immediate linear L -0.0071352469311410345 negative 0.5475
immediate linear G 0.9607901869831791 negative 0.0
matrix first vs end norm difference 0.25104532592556045
reversal 0.0075
dose 0.09 L 0.4801277525994127 [0.44347394471017504, 0.5187852665969848] course [0.50307197 0.47683513 0.48216076 0.48011565 0.48013299 0.48012775]
dose 0.09 G -0.002923879738652899 [-0.00642745914257524, 0.000774450618501172] course [-0.0021372  -0.0039697  -0.00301958 -0.0029186  -0.00292628 -0.00292388]
dose 0.09 C -0.000651282276062539 [-0.005304018218415827, 0.0038618890653093233] course [ 7.47917099e-05 -1.14483112e-04 -6.70036121e-04 -6.49230419e-04
 -6.51395151e-04 -6.51282276e-04]
diff effects [0.4456537334133023, 0.5218817793664188] [0.44253336361091034, 0.5200170670252785]
FULL corr/slope/error 0.976420791926006 1.007449915379838 0.21676697130916633
AVERAGED corr/slope/error 0.9230530855780985 1.0070076900949458 0.4083191990847276
SWAPPED corr/slope/error 0.9047028642936555 0.9350892157751165 0.4453767742860878
SAME corr/slope/error 0.8137357272747328 0.8237944630539629 0.6708599417963703
IDENTITY corr/slope/error 0.10975611867528184 0.29368299748306836 1.0251214520920682
FULL-minus AVERAGED [0.04835124970396156, 0.06603561243166732]
FULL-minus SWAPPED [0.06480791041918897, 0.09706452694633204]
FULL-minus SAME [0.1515038425219899, 0.21009067691421185]
FULL-minus IDENTITY [0.7650128029573282, 0.8764641811743165]
adjoint 6.829615539633369e-10
tail train 4.196806222969278e-17 9.459312759007682e-14 4.649425174745314e-13 above 1e-12 0
tail pred 3.030825803344061e-17 2.538453674088786e-13 9.92876788305754e-13 above 1e-12 0
sham 40 4.299645218530939e-09 answer32=64 True finite True
META llever_smoke-20261110-171809 complete {'T': 48, 'restrict_cr': False, 'chunk': 128, 'K0': 64} {'G1': [1.0658141036401503e-14, 1.0658141036401503e-14, 6.039613253960852e-14, 4.618527782440651e-14, 8.881784197001252e-15, 0.0]} 35.578062666 None
META llever_train-20261110-171953 complete {'T': 48, 'restrict_cr': False, 'chunk': 128, 'K0': 64} {'G1': [0.0, 0.0]} 106.878883171 None
META llever_pred-20261110-172208 complete {'T': 48, 'restrict_cr': False, 'chunk': 128, 'K0': 64, 'train_run': '../results/llever_train-20261110-171953'} {'G1': [0.0, 0.0]} 161.37689605 f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f
META llever_eval-20261110-172520 complete {'T': 48, 'restrict_cr': False, 'chunk': 128, 'K0': 64, 'train_run': '../results/llever_train-20261110-171953', 'pred_run': '../results/llever_pred-20261110-172208'} {'G1': [6.039613253960852e-14, 7.815970093361102e-14, 6.750155989720952e-14, 9.592326932761353e-14, 5.684341886080802e-14, 6.394884621840902e-14, 6.394884621840902e-14, 5.684341886080802e-14]} 101.743815099 f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f
SHARDS [('complete', 48, 'f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f', 101.743815099), ('complete', 48, 'f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f', 102.306664187), ('complete', 48, 'f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f', 101.751418757), ('complete', 48, 'f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f', 111.066568428)]
SHA llever_pushes.npz f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f
SHA llever_pred-20261110-172208/pushes.npz f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f
SHA llever_eval-20261110-172520/pushes.npz f021c80a02d3b0d5b8f21bb9fdb7a42fe65eb7d59ea71109ae5526d44cb8ea3f
```
