#set document(title: "Is a looped model easier to monitor? Results of the matched-pair monitor experiment")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#show table: set par(justify: false)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => [#v(0.6em) #it #v(0.2em)]
#show figure.caption: set text(size: 9.5pt)

#let boxed(title, color, body) = block(
  fill: color, inset: 9pt, radius: 3pt, width: 100%, breakable: true,
  [*#title.* #body],
)
#let plain(body) = boxed("In words", rgb("#fbf3e6"), body)
#let caveat(body) = boxed("Careful", rgb("#fdeeee"), body)
#let explore(body) = boxed("Exploratory, not registered", rgb("#eaf1fb"), body)

// ---- Numbers. Every number in a table below is read from generated files, never typed.
// tables.json: tables_mon.py (median over the three seeds, envelope of the seeds' 95% intervals).
// explore.json: mon_explore_scores.py. curve.json: mon_learning_curve.py.
#let cells = json("monitor_results_gen/tables.json")
#let (Q, R) = {
  let q = (:)
  let r = (:)
  for c in cells {
    r.insert(c.table + "|" + c.row_label + "|" + c.column_label, c.formatted)
    if "key" in c and c.key != none and c.field != none {
      q.insert(c.table + "|" + c.family + "|" + c.key + "|" + c.field, c.formatted)
    }
  }
  (q, r)
}
#let q(t, fam, key, field) = Q.at(t + "|" + fam + "|" + key + "|" + field)
#let r(t, row, col) = R.at(t + "|" + row + "|" + col)
#let pt(s) = s.split(" [").at(0)
#let ex = json("monitor_results_gen/explore/explore.json")
#let lc = json("monitor_results_gen/curve/curve.json")
#let pv = json("monitor_results_gen/explore/prev.json")
#let pvr(seed, fam, site) = pv.rows.filter(x => x.seed == seed and x.family == fam and x.d_site == site).at(0)
#let f3(x) = {
  let a = calc.round(calc.abs(x), digits: 3)
  let s = str(a)
  if not s.contains(".") { s = s + "." }
  s = s + "0" * (3 - s.split(".").at(1).len())
  if x < 0 and a != 0 { "−" + s } else { s }
}
#let fiv(x, iv) = if iv.at(0) == none { f3(x) + " [n/a]" } else { f3(x) + " [" + f3(iv.at(0)) + ", " + f3(iv.at(1)) + "]" }
#let med(a) = a.sorted().at(calc.floor(a.len() / 2))
#let fams = ("add3", "add4", "add3w", "arc_easy")
#let famname = (add3: "add3", add4: "add4", add3w: "add3w", arc_easy: "ARC-Easy", family_mean: "Mean of the four")
#let allfams = fams + ("family_mean",)
#let tb(cols, ..args) = align(center, block(breakable: false, text(size: 8.8pt, table(
  columns: cols, inset: (x: 5pt, y: 3.2pt),
  stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
  ..args))))
#let hd(..a) = table.header(..a.pos().map(x => strong(x)))

#align(center)[
  #text(16pt, weight: "bold")[Is a looped model easier to monitor?]
  #v(0.1em)
  #text(12pt)[Results of the matched-pair monitor experiment]
  #v(0.2em)
  #text(9.5pt)[Draft 1, 2026-11-10. Design fixed before the data: `docs/oversight_monitor.md` (Draft 1.3). Three seeds.
  Results recomputed independently by GPT-6 Astra (xhigh): all 240 recomputed values matched; its required changes
  to the wording, and the second review's changes, are applied here (
  `results/mon_report-20261110/astra_review.md`; `results/mon_report-20261110/astra_review_writeup.md`).]
]

#v(0.4em)

#plain[
A language model can be built so that a block of its layers is applied over and over (a _looped_ model) instead of
passing once through a longer stack of different layers. We asked whether this makes the model easier to supervise
in one specific sense: *can a simple reader of the model's internal state tell when the model's answer is going to
be wrong?* We compared a looped model (L32) with an ordinary model (D) trained on the same data from the same
starting weights, on 6,000 problems in four task families, with the analysis fixed in advance.

+ *No overall advantage was resolved for this pair of models.* Reading each model's own state, the quality of the
  error prediction (a number from 0 to 1; 0.5 means chance ranking and 1 perfect ranking) was #pt(r("table", "errors of L32, read from L32's state", "family_mean"))
  for the looped model and #pt(r("table", "errors of D, read from D's state", "family_mean")) for the ordinary one,
  with difference #r("table", "diagonal contrast", "family_mean"); single families differ in sign and are individually uncertain.
+ *The state reader does worse than simply looking at how confident the model's output is*
  (#pt(r("added", "fixed/L32/probe", "family_mean")) against #pt(r("added", "fixed/L32/confidence", "family_mean"))), and adding it
  to the confidence improves the prediction only slightly (by about 0.003 to 0.006), in both models.
+ *A reader fitted at 32 loops ranks errors about as well when the model is run with 4 to 32 loops* (and, on a
  smaller set of problems, up to 64). This is an observation at the budgets tested, not a guarantee. At 1 loop the
  ranking is worse, and at 2 loops the fixed rejection threshold rejects many more answers.
+ *The loop's own dynamics did not give a useful warning signal here.* A forecast of "this reading will not change
  if the model keeps looping", derived from a contraction argument, was not violated on any problem where it could
  be made, but it was hundreds of times too cautious; after calibration it showed no general advantage over a
  constant.
+ Follow-up checks chosen after seeing the results (marked as exploratory): the state reader is still improving
  slowly with more training problems (by 0.014 to 0.019 over the last doubling), so point 2 is a statement about
  readers trained on about 1,200 problems; and a reader of the _other_ model's state predicts a model's errors
  almost as well as a reader of its own.
+ One exploratory result points the other way: at the token position _before_ the last one, the looped model's
  errors were predicted better from its state than the ordinary model's errors from its state, by 0.02 to 0.04 in
  all three seeds (Section 7.4). It was noticed after the fact. The two models make different errors, so this does
  not show that the looped state holds more information, and it needs a registered test on new data.

This is one pair of trained models and honest mistakes on short problems. It is not evidence about looping as such,
and not about a model that conceals something.
]

= The objects

*Models.* A prompt is a sequence of tokens (word pieces). A _checkpoint_ is one saved set of trained model
parameters. The _input embedding_ is the vector assigned to each token before any layer acts.
A model turns the prompt into a matrix with one row of $d = 2048$
numbers per token and transforms that matrix by a sequence of _layers_ (fixed nonlinear maps); from the row at the
last token it finally computes a probability for every possible next token. Its _answer_ is the most probable token.

- The ordinary model *D* applies 16 different layers once each. Write $h_l$ for the last token's row after layer $l$.
- A _looped_ model applies 4 layers, then one block of 6 layers $r$ times, then 4 final layers. Write $F$ for
  the block, a map from states to states, and $X_t$ for the whole matrix after $t$ applications:
  $ X_(t+1) = F(X_t), quad t = 0, 1, dots, r - 1, $
  with $x_t$ the row of $X_t$ at the last token. The number $r$ of loops is chosen when the model is run; we call it
  the _budget_. The four looped models *L4, L8, L16, L32* were trained with a random number of loops of mean 4, 8,
  16, 32. All five models come from the same starting weights and the same training data (the published
  Recurrent-Llama-3.2 checkpoints, about one billion parameters).

The comparison of interest is *L32 run for 32 loops against D*. In both, what we read is what the model's last four
layers receive: $x_32$ for L32 and $h_12$ for D.

*Problems.* Four families, 1,500 problems each, every answer a single token. Three are sums
(add3: $a + b + c$ with two-digit numbers; add4: four two-digit numbers; add3w: three three-digit numbers); the
fourth, ARC-Easy, is multiple-choice school science questions (answer A, B, C or D). The families were picked by a
rule fixed beforehand from a small screening run: the first families in a fixed list on which the models were
right between 30% and 90% of the time, so that there are both right and wrong answers to tell apart.

#figure(
  tb(6, hd([Fraction of answers correct], ..allfams.map(f => famname.at(f))),
    ..("L4", "L8", "L16", "L32", "D").map(m => (
      [#m#if m != "D" [ (at its training mean)]],
      ..allfams.map(f => r("accuracy", m + " accuracy (fraction correct)", f)))).flatten()),
  caption: [How often each model is right. The models are wrong often enough for errors to be predicted, and the
  two models of the main comparison are about equally accurate, though not on the same problems (Section 7).],
)

*The label.* For a model, a budget and a problem, the label is $y = 1$ if the answer is *wrong* and $y = 0$ if it
is right. "Monitoring" here means predicting $y$ without knowing the true answer.

= The monitor and how it is scored

*Linear probe.* A monitor is a fixed vector $m in RR^d$ and a number $b$; its _score_ for a state $x$ is
$ s(x) = m dot x + b. $
A high score is meant to say "this answer is probably wrong". The vector is found from training problems
$(x_i, y_i)$ by _penalized logistic regression_: after shifting and scaling each coordinate to mean 0 and variance 1
on the training problems (giving $z_i$), minimize over $w, b$
$ 1/n sum_(i=1)^n log(1 + e^(-(2 y_i - 1)(w dot z_i + b))) + lambda/2 norm(w)^2 . $
The first term is small when the score is positive on wrong answers and negative on right ones; the second keeps
$w$ small, which matters because there are more coordinates (2,048) than is comfortable for about 1,200 training
problems. The strength $lambda$ is chosen from nine values $10^(-6), dots, 10^2$ as described below.

*Scoring a monitor: AUROC.* Take a wrong answer and a right answer at random. The _AUROC_ of a score is
$ "AUROC" = P(s("wrong") > s("right")) + 1/2 P(s("wrong") = s("right")). $
AUROC ranges from 0 to 1; 0.5 means chance ranking and 1 perfect ranking. It is unchanged by a strictly
increasing transformation of the score.

*Honest scoring.* A monitor is always scored on problems it was not fitted on. The 1,500 problems of a family are
split into five parts; the monitor is fitted on four and scores the fifth, in turn. Inside the four training parts,
a second split into four chooses $lambda$ (and, where there is a choice, which layer to read) by the same
principle. The AUROC is computed in each fifth and averaged, then averaged over the four families with equal weight.

*What the model itself already says: confidence.* The model's output distribution already carries information about
whether it is right. The _confidence baseline_ is a monitor fitted in the same way on four numbers computed from
the output: the log-probability of the answer, the gap to the second candidate, the entropy of the distribution, and
the log of the total probability on valid answers. Entropy is $-sum_j p_j log p_j$, a measure of how spread out
the output probabilities are. It is a trained four-number monitor, not a single raw number. Confidence+probe is
one regression fitted jointly on these four numbers and the state coordinates.
The *added value* of a probe is
$ "AUROC"("confidence and probe together") - "AUROC"("confidence alone"). $

*Uncertainty, and how numbers are written.* The whole analysis is run three times with three different random
splits (_seeds_). Inside each run, a 95% interval comes from resampling the held-out problems 2,000 times with
the fitted monitors kept fixed. In this document
$ 0.687 thick [0.671, 0.701] $
means: the *median of the three seeds' values*, and in brackets the *smallest lower end and largest upper end of
the three seeds' intervals*. The bracket is a conservative summary of three conditional intervals. It is not a
confidence interval for the median, it does not include the variability from refitting the monitors, and it says
nothing about other trained models. Family means and contrasts are formed within each seed before taking seed
medians, so they need not equal arithmetic operations on separately displayed medians. The interval envelope is
descriptive, not a new confidence level or a multiple-comparison adjustment. Per-seed values are in Appendix A.

*What was fixed in advance.* The design, the analysis code, the problem list and the choice of families were frozen
(by file hashes) before the main data existed, after two external design reviews; the three analysis runs report
that they used exactly the frozen files. The quantities in Sections 3 to 6 are the registered ones. Section 7
contains analyses chosen afterwards and is marked as such.

= Looped against ordinary: the two-by-two table

"L32 is easier to monitor than D" could be true for two different reasons: L32's _errors_ might be of a more
predictable kind, or L32's _state_ might be more informative. To separate them, each model's errors are predicted
from each model's state (the same problems and the same splits in all four cells):

#figure(
  tb(3, hd([AUROC, mean of the four families], [read from L32's state $x_32$], [read from D's state $h_12$]),
    [errors of L32], r("table", "errors of L32, read from L32's state", "family_mean"), r("table", "errors of L32, read from D's state", "family_mean"),
    [errors of D], r("table", "errors of D, read from L32's state", "family_mean"), r("table", "errors of D, read from D's state", "family_mean")),
  caption: [The registered two-by-two table. Diagonal cells: each model monitored from its own state.],
)

Write $A_("LL"), A_("LD"), A_("DL"), A_("DD")$ for the four cells (first letter: whose errors; second: whose state).

#figure(
  tb((2.1fr, 1fr, 1fr, 1fr, 1fr, 1fr), hd([Contrast], ..allfams.map(f => famname.at(f))),
    ..(
      ("diagonal contrast", [own state: $A_("LL") - A_("DD")$]),
      ("row contrast", [whose errors: $(A_("LL") + A_("LD"))/2 - (A_("DL") + A_("DD"))/2$]),
      ("column contrast", [whose state: $(A_("LL") + A_("DL"))/2 - (A_("LD") + A_("DD"))/2$]),
      ("interaction contrast", [own-state advantage: $A_("LL") - A_("LD") - A_("DL") + A_("DD")$]),
    ).map(x => (x.at(1), ..allfams.map(f => r("table", x.at(0), f)))).flatten()),
  caption: [Contrasts of the table, per family and for the mean. Positive diagonal, row or column contrasts favour L32.],
)

#figure(placement: auto, image("monitor_results_gen/figs/fig_table.pdf", width: 92%),
  caption: [The four cells in each family. Colour: whose state is read (blue L32, orange D). Bars: envelope of the
  three seeds' intervals. The dashed line is chance.])

What the table says:

- *No overall advantage was resolved for this checkpoint pair.* The own-state difference is
  #r("table", "diagonal contrast", "family_mean"). With the best layer chosen inside the fitting procedure instead of the fixed
  layer (L32: $x_32$ or the output of its final layers; D: its input embedding or the output of any of its 16 layers), it is
  #r("best", "best_difference", "family_mean"). These brackets are conditional on the fitted monitors. They are not a
  test of equivalence: a true difference of 0.02 in favour of L32 is compatible with the first bracket, and one of
  0.02 in favour of D with the second.
- *The families do not agree in sign.* L32's own-state monitor is behind in add3 and add4 and ahead in add3w and
  ARC-Easy, each within its own uncertainty. The mean near zero is an average of small opposite differences, not
  four zeros.
- *Neither the errors nor the state of one model is clearly more useful* (row and column contrasts).
- *Each model's errors are read slightly better from its own state than from the other model's*: the interaction is
  #r("table", "interaction contrast", "family_mean"), clearest in ARC-Easy. This number adds two own-minus-other
  advantages, so the average advantage of reading the model's own state is half of it, about 0.009. What is striking
  is how small it is: *D's state predicts L32's errors almost as well as L32's own state does*, and conversely.
  Section 7 returns to what this does and does not mean.

#caveat[
The design allowed "not resolved at this sample size" as an outcome, and that is the outcome for the main
comparison. The planning simulation (design, Section 7) put the resolvable difference at about 0.03 in a single
family; the observed differences are below that. Because the families disagree in sign, more problems per family
would mainly sharpen the family-by-family picture.
]

*Which layer was chosen* (60 fits per model: 4 families, 3 seeds, 5 splits). For L32:
$x_32$ #r("sites", "L32/x32", "total") times, the output of the final layers #r("sites", "L32/post32", "total") times.
For D: layer 7: #r("sites", "D/h7", "total"); 12: #r("sites", "D/h12", "total"); 13: #r("sites", "D/h13", "total");
14: #r("sites", "D/h14", "total"); 15: #r("sites", "D/h15", "total"); 16: #r("sites", "D/h16", "total"). Choosing the
layer by data did not help either model (best-site AUROC #pt(r("best", "best/L32", "family_mean")) and
#pt(r("best", "best/D", "family_mean")), the same as the fixed layers).

*In operational terms.* If the 20% of answers with the highest score are set aside, the error rate among the
remaining answers falls from #r("coverage", "L32 overall_error", "family_mean") to #r("coverage", "L32 accepted_error", "family_mean")
for L32 and from #r("coverage", "D overall_error", "family_mean") to #r("coverage", "D accepted_error", "family_mean") for D
(means of the four families). An AUROC of 0.69 is a modest monitor.

= Does the probe add anything to the model's own confidence?

#figure(
  tb(3, hd([AUROC, mean of the four families], [L32], [D]),
    [confidence alone (four numbers from the output)], r("added", "fixed/L32/confidence", "family_mean"), r("added", "fixed/D/confidence", "family_mean"),
    [probe alone], r("added", "fixed/L32/probe", "family_mean"), r("added", "fixed/D/probe", "family_mean"),
    [confidence and probe together], r("added", "fixed/L32/confidence+probe", "family_mean"), r("added", "fixed/D/confidence+probe", "family_mean"),
    [*added value* of the probe], r("added", "fixed/L32/confidence/added", "family_mean"), r("added", "fixed/D/confidence/added", "family_mean"),
    table.hline(stroke: 0.3pt),
    [flexible confidence alone (curved, not only straight-line, use of the four numbers)], r("added", "fixed/L32/flex", "family_mean"), r("added", "fixed/D/flex", "family_mean"),
    [flexible confidence and probe together], r("added", "fixed/L32/flex+probe", "family_mean"), r("added", "fixed/D/flex+probe", "family_mean"),
    [*added value* over flexible confidence], r("added", "fixed/L32/flex/added", "family_mean"), r("added", "fixed/D/flex/added", "family_mean"),
    table.hline(stroke: 0.3pt),
    [control: which token was predicted], r("added", "L32/output_identity", "family_mean"), r("added", "D/output_identity", "family_mean"),
    [control: the problem itself (uses the true answer)], r("added", "L32/privileged_ground_truth", "family_mean"), r("added", "D/privileged_ground_truth", "family_mean"),
    [control: probe on an early layer], r("added", "L32/early", "family_mean"), r("added", "D/early", "family_mean")),
  caption: [Probe, confidence, both, and three controls. Fixed layers ($x_32$, $h_12$).],
)

- *Confidence is the better monitor in both models* (0.74 against 0.69), and is available from the output without a state probe.
- *The probe has a small added value.* Over the confidence baseline it is
  #r("added", "fixed/L32/confidence/added", "family_mean") for L32 and #r("added", "fixed/D/confidence/added", "family_mean") for D;
  over the flexible baseline #r("added", "fixed/L32/flex/added", "family_mean") and #r("added", "fixed/D/flex/added", "family_mean").
  Several single-seed intervals lie just above zero (Appendix A), so "the probe adds nothing" would be wrong;
  "a few thousandths" is right. An improvement of 0.005 in AUROC means that the combined monitor orders one more
  pair in 200 correctly.
- *No difference between the models in added value was resolved:* L32 minus D is #r("added", "L32 minus D added (confidence)", "family_mean")
  against the confidence baseline and #r("added", "L32 minus D added (flexible confidence)", "family_mean") against the flexible one.
- *Controls.* The identity of the predicted token alone gives 0.54; features of the problem that use the true
  answer (digits, carries, the correct letter) give about 0.60; a probe on an early layer about 0.61. These controls are below the late-state probe,
  but do not exhaust problem difficulty. They cannot show that
  the late probe reads something other than difficulty-related features.

#figure(placement: auto, image("monitor_results_gen/figs/fig_confidence.pdf", width: 92%),
  caption: [The same quantities in each family. Circles L32, squares D; grey: the three controls. In ARC-Easy
  confidence and probe are both good and close; in the sums the probe is clearly behind confidence.])

*The other looped models (registered secondary analysis).* Each looped model at its own training mean:

#figure(
  tb(5, hd([AUROC, mean of the four families], [L4 at 4 loops], [L8 at 8], [L16 at 16], [L32 at 32]),
    [probe], ..("L4/4", "L8/8", "L16/16", "L32/32").map(k => r("loopcounts", k + "/probe", "family_mean")),
    [confidence], ..("L4/4", "L8/8", "L16/16", "L32/32").map(k => r("loopcounts", k + "/confidence", "family_mean")),
    [added value], ..("L4/4", "L8/8", "L16/16", "L32/32").map(k => r("loopcounts", k + "/added", "family_mean"))),
  caption: [The four looped models are alike. The added value for L4 is positive in every seed, and still small.],
)

*Robustness checks (registered).* A _principal component_ is a direction of maximal remaining variance in the
training states.

#figure(
  tb(4, hd([AUROC, mean of the four families], [L32], [D], [L32 minus D]),
    [only problems where both models are right or both wrong], r("robust", "concordant/L32", "family_mean"), r("robust", "concordant/D", "family_mean"), r("robust", "concordant/diagonal_difference", "family_mean"),
    [simplest probe: difference of the two class means], r("robust", "L32/meandiff", "family_mean"), r("robust", "D/meandiff", "family_mean"), r("robust", "meandiff/diagonal_difference", "family_mean"),
    [probe on the 128 directions of largest variance], r("robust", "L32/pca_logistic", "family_mean"), r("robust", "D/pca_logistic", "family_mean"), r("robust", "pca_logistic/diagonal_difference", "family_mean"),
    [labels shuffled once (should be near 0.5)], r("robust", "L32/permutation", "family_mean"), r("robust", "D/permutation", "family_mean"), [],
  ),
  caption: [Other populations and other probes give the same picture. On the both-right-or-both-wrong problems the
  two labels coincide and the AUROC is higher; this is a selected set of problems, shown as a check only.],
)

#caveat[
The shuffled-label check is a single shuffle per family, a diagnostic and not a test. Per seed it gives, for L32,
#r("seeds", "m1s L32/permutation", "seed 0"), #r("seeds", "m1s L32/permutation", "seed 1"), #r("seeds", "m1s L32/permutation", "seed 2"),
and for D, #r("seeds", "m1s D/permutation", "seed 0"), #r("seeds", "m1s D/permutation", "seed 1"), #r("seeds", "m1s D/permutation", "seed 2").
D's seed-0 interval excludes 0.5 from below. The same permutation is reused across split seeds. The conditional
score bootstrap does not repeat permutation, fitting and selection. This is an anomalous diagnostic observation,
not a full-pipeline null test; its mechanism has not been established here.
]

#let p3(site, key) = (0, 1, 2).map(s => f3(pvr(s, "family_mean", "D/prev12").at(key))).join(", ")
*The position before the last (registered secondary rows).* The same probes were also fitted on the state at the
second-to-last token (position −2): “ =” before the final space in the sums, “Answer” before the colon in
ARC-Easy. Both models have received the problem text. The matched states precede four final layers. Each probe
predicts its own model's final-answer error (32-loop labels for L32). The same 1,500 problems per family are used,
but each model has its own label-stratified five folds, with four inner folds selecting the penalty at each site.
Both are weaker than at the last position, and they are not equal: for L32 (state after 32 loops) the AUROC
in the three seeds is #p3("", "L32"), for D (layer 12) #p3("", "D") (means of the four families). The design
registered these as single rows without a contrast between the models. Section 7.4 looks at the difference.

*Across families.* A probe fitted on one family and applied unchanged to another does much worse: the mean AUROC
over the other three families is #r("cross_mean", "L32_auroc", "family_mean") for L32 and #r("cross_mean", "D_auroc", "family_mean")
for D, about 0.10 to 0.12 below the same probes on their own family. The cells are very uneven (from near 0.47 to
0.71; Appendix B). The probes are family-specific in both models; no interval is available for these means.

= Changing the number of loops

A looped model can be run with any number of loops. A monitor fitted at 32 loops might or might not keep working at
another budget. The registered test: fit the probe on $x_32$ with the labels at 32 loops; freeze its vector, its
scaling and the threshold above which it rejects 20% of answers; then apply it to $x_r$ and compare with the labels
at budget $r$ (the model's answers at budget $r$). "Native" is a probe fitted directly at budget $r$.

#figure(
  tb(8, hd([Budget $r$], [model's error rate], [correctness labels changed vs. 32], [AUROC, frozen probe from 32], [AUROC, native probe], [native minus frozen], [fraction rejected], [error among accepted]),
    ..("1", "2", "4", "8", "16", "32").map(t => (
      t,
      q("threshold", "family_mean", "L32/32->" + t, "overall_error"),
      q("threshold", "family_mean", "L32/32->" + t, "label_changed"),
      pt(q("budget32", "family_mean", "L32/32->" + t, "transfer_auroc")),
      pt(q("budget32", "family_mean", "L32/native" + t, "auroc")),
      q("budget32", "family_mean", "L32/32->" + t + "/native_minus_transfer", "difference"),
      q("threshold", "family_mean", "L32/32->" + t, "fraction_rejected"),
      q("threshold", "family_mean", "L32/32->" + t, "accepted_error"))).flatten()),
  caption: [L32, probe fitted at 32 loops and applied at other budgets; means of the four families. "Correctness labels
  changed": fraction of problems whose right/wrong label differs from the label at 32 loops.],
)

#figure(placement: auto, image("monitor_results_gen/figs/fig_budget.pdf", width: 92%),
  caption: [The same for every saved budget. Blue: the probe fitted at 32 loops, applied at $r$. Grey: a probe
  fitted at $r$. Triangles (48, 63, 64 loops) use a subset of 256 problems per family, on which the native probe has
  far fewer training problems; see the text.])

- *Across tested targets 4–32, ranking changes little relative to native probes.* Over the registered sources,
  the gap between a native probe and the frozen one is at most 0.007 for the family mean and 0.018 within a family.
  These are descriptive small differences, not equivalence results. L32's frozen threshold rejects about 20%;
  accepted-answer error is #q("threshold", "family_mean", "L32/32->4", "accepted_error") at 4 loops versus
  #q("threshold", "family_mean", "L32/32->32", "accepted_error") at 32. Correctness labels agree with the 32-loop
  labels on about 91% of problems at 4 loops and 98% at 8; this is not exact answer-token agreement.
- *At 1 loop ranking degrades; at 2 loops the ranking difference is unresolved.* At 1 loop, L32 is wrong 72% of
  the time, 43% of the labels differ, the frozen probe loses 0.04 against a native one and rejects only 12%.
  At 2 loops it rejects 31% instead of 20%. AUROC cannot detect threshold effects from shifted scores.
- *Beyond 32 loops,* native probes have about 205 training problems versus 1,200 for transferred probes.
  This confounds the comparison (0.633 versus 0.689 at 64 loops). On matched deep problems the same 32-loop probe
  changes AUROC by at most #f3(calc.max(..ex.deep.filter(x => x.source == 32).map(x => calc.abs(x.difference)))). This supports stability of that probe, but does not prove that
  training-set size alone caused the native probe's lower AUROC (Section 7.3).
  Some deep add3 intervals are unavailable; Section 9 gives the counts.

#caveat[
These observations do not establish lossless transfer or general budget safety. The forecast below was evaluated
at 4, 8 and 16 loops only, so it was not tested as a warning about 1 or 2 loops.
]

= Can the loop's own dynamics forecast stability?

*The idea.* Suppose $0 <= q < 1$ and the block $F$ satisfies $norm(F(a) - F(b)) <= q norm(a - b)$
for every consecutive trajectory pair needed through $r+h$, with the trajectory remaining in the region where
this inequality applies. Write $d_t = norm(X_t - X_(t-1))$ for the size of step $t$ (Euclidean norm over the whole matrix). Then
$d_(t+1) <= q d_t$, and summing a geometric series,
$ norm(X_(r+h) - X_r) <= sum_(t=r+1)^(r+h) d_t <= q d_r (1 - q^h)/(1 - q) =: B . $
For a probe with vector $m$ (in the original coordinates) the score then moves by at most $norm(m) B$, because
$abs(m dot (x_(r+h) - x_r)) <= norm(m) norm(x_(r+h) - x_r) <= norm(m) norm(X_(r+h) - X_r)$. So after $r$ loops one
could say, for each problem separately: "the monitor's reading will not move by more than this if the model keeps
looping". Repeated application of the same map gives this geometric form; distinct feedforward layers would instead
require bounds for their respective maps.

*The forecast.* The number $q$ is unknown. The registered rule estimates it by $hat(q)$, the largest of up to the last
eight available ratios $d_t \/ d_(t-1)$ seen up to loop $r$, and gives no forecast if $hat(q) >= 1$ or the relative step
$d_r / norm(X_r)$ is below $10^(-6)$. It is a forecast, not a
guarantee: later ratios can exceed all earlier ones. It was tested at $r = 4, 8, 16$ against what actually
happened by loop 32.

#figure(
  tb(7, hd([Model, $r$], [sum of later steps exceeds $B$], [$norm(X_32 - X_r)$ exceeds $B$], [last row's movement exceeds $B$], [score movement exceeds $norm(m) B$], [score bound ÷ actual movement (mean of family medians)], [forecast "decision stable"]),
    ..("L4", "L8", "L16", "L32").map(m => ("4", "8", "16").map(t => (
      m + ", " + t,
      r("forecast", m + "/" + t, "path_violation"),
      r("forecast", m + "/" + t, "exact_violation"),
      r("forecast", m + "/" + t, "last_position_violation"),
      r("forecast", m + "/" + t, "score_violation"),
      str(calc.round(float(r("forecast", m + "/" + t, "score_bound_over_realized.median")))),
      r("forecast", m + "/" + t, "fraction_forecast_stable")))).flatten()),
  caption: [Uncalibrated forecast, means of the four families. Violation rates condition on having a forecast;
  the stable fraction uses all problems. Bound/movement is a mean of within-family medians. "Decision" means the
  monitor's accept/reject flag; stable means the score is further from the rejection threshold than the bound.],
)

- *The path and net-displacement failures are different.* The sum of later step lengths exceeds $B$ in about
  42%–100% of forecastable problems across the displayed summaries. Net whole-state displacement exceeds it in
  about 4%–91%. These observations invalidate extrapolation using the estimated ratio, not the existence of any
  contraction constant.
- *No score violation was observed among forecastable held-out problems.* This is an empirical finding, not a
  certificate. The displayed means of family medians of bound/movement range from about 183 to 2,640.
  Whole-state movement and Cauchy–Schwarz can make a score bound conservative, but these summaries do not
  separate their contributions from those of the estimated ratio or storage rounding.

*Calibrated version (registered).* Multiply the forecast by one constant, chosen on half of the held-out problems
so that about 95% of them move by less than the calibrated forecast, and test on the other half. Precisely:
separately within each outer fold, half of its held-out problems determine the multiplier as the $ceil((n_"cal"+1) times 0.95)$-th ordered realized/predicted movement ratio among eligible
forecasts, where $n_"cal"$ counts eligible calibration problems. The registered comparison is with the
*constant forecast*: the same number for every problem in the fold, calibrated in the same way, which uses no
dynamics at all.
Coverage is marginal among forecastable evaluation problems, not a 95% guarantee conditional on declaring a
decision stable. The rejection threshold is the 80th percentile of the held-out source scores themselves:
it uses those test features, though not their labels.

#figure(
  tb(9, hd([Model, $r$], [has a forecast], [coverage: dynamics], [coverage: constant], [bound ÷ actual: dynamics], [bound ÷ actual: constant], [declared stable: dynamics], [declared stable: constant], [wrongly declared stable: dynamics / constant]),
    ..("L4", "L8", "L16", "L32").map(m => ("4", "8", "16").map(t => {
      let g(c, f) = q("forecast_main", "family_mean", m + "/" + t, "score_conformal." + c + "." + f)
      (m + ", " + t, g("scaled_rule", "forecast_fraction"), g("scaled_rule", "coverage"), g("constant", "coverage"),
       g("scaled_rule", "bound_over_realized.median"), g("constant", "bound_over_realized.median"),
       g("scaled_rule", "fraction_forecast_stable"), g("constant", "fraction_forecast_stable"),
       g("scaled_rule", "false_stability") + " / " + g("constant", "false_stability"))})).flatten()),
  caption: [Calibrated forecasts of the score movement from loop $r$ to loop 32, means of the four families.
  "Dynamics": the contraction forecast times a calibrated constant. "Constant": one number for all problems.
  Problems without a forecast count as not stable. "Wrongly declared stable" is the changed-decision fraction
  among declared-stable items; bound/movement columns are means of family medians.],
)

- Both versions reach roughly 95% marginal coverage, and both almost never declare stable a decision that then changes.
- *The dynamics-based forecast is sometimes tighter for the typical problem* (bound ÷ actual movement), *but it
  does not show a general advantage in the fraction declared stable: it is sometimes slightly higher and often lower*, partly because some problems have no
  forecast at all ($hat(q) >= 1$ or a step too small to measure).
- Summary in the wording agreed with the reviewer: the forecast is _empirically calibrated and sometimes tighter,
  without a demonstrated general advantage over a constant forecast for stable decisions_.

#caveat[
The states were stored with 16-bit precision. At 16 loops the conservative rounding allowance exceeds measured
movement, by mean-of-family-median factors
#q("forecast_main", "family_mean", "L4/16", "allowance_over_movement.median"),
#q("forecast_main", "family_mean", "L8/16", "allowance_over_movement.median"),
#q("forecast_main", "family_mean", "L16/16", "allowance_over_movement.median") and
#q("forecast_main", "family_mean", "L32/16", "allowance_over_movement.median") for L4, L8, L16 and L32.
An allowance is an upper bound, not measured rounding error. On the 32-per-family float32-copy subset, the
corresponding mean-of-family-median relative movement changes are
#q("forecast_main", "family_mean", "L4/16", "rounding_subset.relative_movement_change.median"),
#q("forecast_main", "family_mean", "L8/16", "rounding_subset.relative_movement_change.median"),
#q("forecast_main", "family_mean", "L16/16", "rounding_subset.relative_movement_change.median") and
#q("forecast_main", "family_mean", "L32/16", "rounding_subset.relative_movement_change.median").
For L4/16, allowing for rounding changes calibrated dynamics/constant coverage from
#q("forecast_main", "family_mean", "L4/16", "score_conformal.scaled_rule.coverage") /
#q("forecast_main", "family_mean", "L4/16", "score_conformal.constant.coverage") to
#q("forecast_main", "family_mean", "L4/16", "score_conformal.scaled_rule.coverage_with_allowance") /
#q("forecast_main", "family_mean", "L4/16", "score_conformal.constant.coverage_with_allowance"),
and stable fractions from
#q("forecast_main", "family_mean", "L4/16", "score_conformal.scaled_rule.fraction_forecast_stable") /
#q("forecast_main", "family_mean", "L4/16", "score_conformal.constant.fraction_forecast_stable") to
#q("forecast_main", "family_mean", "L4/16", "score_conformal.scaled_rule.fraction_forecast_stable_with_allowance") /
#q("forecast_main", "family_mean", "L4/16", "score_conformal.constant.fraction_forecast_stable_with_allowance").
The full adjusted tables show no general dynamics advantage.
]

*Dynamical signals as error predictors (registered M4).* Label-free signals of the loop (how large the last steps
are, whether consecutive steps reverse direction, whether and when the answer changed during the loops) were used
as a monitor, alone and added to confidence:

#figure(
  tb(5, hd([AUROC, mean of the four families], [L4 at 4], [L8 at 8], [L16 at 16], [L32 at 32]),
    [dynamics alone], ..("L4/4", "L8/8", "L16/16", "L32/32").map(k => r("dynamics", k + "/dynamics", "family_mean")),
    [added to confidence], ..("L4/4", "L8/8", "L16/16", "L32/32").map(k => r("dynamics", k + "/added", "family_mean"))),
  caption: [Dynamics features. No positive added value was resolved; for L32 the point estimate is slightly
  negative. Extra inputs may increase fitting variability, but that explanation was not isolated experimentally.],
)

Alone, the dynamics predict errors a little (0.60 to 0.63), less than the probe and much less than confidence; added
to confidence they bring nothing that could be resolved. For D, the analogous features along its layers give
#r("dynamics", "D/16/dynamics", "family_mean"), but these are different features (they include how the answer read
off intermediate layers changes), so the number is not comparable and is not read as a difference between
architectures.

= Follow-up analyses chosen after seeing the results <explore>

#explore[
Everything in this section was chosen after the registered results were known, on the reviewer's suggestion. The
plan and the reading rules were written down before running (`docs/mon_exploratory.md`), but that is not the same
as registration. Only data already collected were used.
]

== Is the probe limited by the number of training problems?

The probe has 2,048 coefficients and about 1,200 training problems per fit. To see whether more problems would
help, the registered procedure was repeated (seed 0) with the training part of each split cut down to
1/8, 1/4, 1/2, 3/4 of its problems (three random nested subsets each) and scored on the same held-out problems. At
full size the procedure reproduces the registered seed-0 values exactly (largest difference
#lc.reproduction.max_abs_difference). To reproduce each registered fit, the state-only curve uses M1-a's shared
joint-label folds, while confidence and confidence+state use M1-c's own-label folds; each retains its folds across
sizes. This clarifies E1's conflicting demands to use M1-a folds throughout and reproduce M1-c; its reading rule
is unaffected.

#let lcrow(model, fit) = ("0.125", "0.25", "0.5", "0.75", "1.0").map(p => f3(lc.curves.family_mean.at(model).at(fit).at(p).auroc))
#let lcadd(model) = ("0.125", "0.25", "0.5", "0.75", "1.0").map(p => f3(lc.derived.family_mean.at(model).added.at(p).auroc))
#figure(
  tb(6, hd([AUROC, mean of the four families], ..("0.125", "0.25", "0.5", "0.75", "1.0").map(p => [#calc.round(lc.curves.family_mean.L32.probe.at(p).training_items) problems])),
    [L32 probe], ..lcrow("L32", "probe"),
    [D probe], ..lcrow("D", "probe"),
    [L32 confidence], ..lcrow("L32", "confidence"),
    [D confidence], ..lcrow("D", "confidence"),
    [L32 added value of the probe], ..lcadd("L32"),
    [D added value of the probe], ..lcadd("D")),
  caption: [Learning curve: AUROC against the number of training problems (seed 0; subsets averaged).],
)

#figure(placement: auto, image("monitor_results_gen/curve/fig_curve.pdf", width: 92%),
  caption: [Learning curves per family. Solid: probe; dotted: confidence; dashed: both. Blue L32, orange D. Bars:
  range over the three random subsets.])

#let ld(model, what) = {
  let v = lc.derived.family_mean.at(model).last_doubling.at(what)
  fiv(v.estimate, v.interval)
}
#let lcd(p) = f3(lc.derived.family_mean.L32_minus_D.probe.at(p))
The quantity fixed in advance was the *gain from the last doubling*: AUROC with all training problems minus AUROC
with half of them (mean of the four families, interval from resampling the held-out problems).

#figure(
  tb(3, hd([Gain from the last doubling (600 to 1,200 problems)], [L32], [D]),
    [probe], ld("L32", "probe"), ld("D", "probe"),
    [added value of the probe over confidence], ld("L32", "added"), ld("D", "added")),
  caption: [Last-doubling gains, mean of the four families, seed 0.],
)

- *By the rule written down beforehand, the answer is "not resolved".* The rule called the probes sample-limited
  if the gain was at least 0.02 with an interval excluding 0, and flat if it was within ±0.01 with an interval inside
  ±0.02. The gains fall between the two thresholds.
- *Descriptively, the probe is still improving.* Both last-doubling intervals exclude zero and all families have
  positive gains, but neither family-mean point estimate reaches E1's 0.02 threshold. Earlier doubling gains
  are 0.015 and 0.020 for L32, and 0.034 and 0.023 for D. One more doubling at the last rate would put the probes
  near 0.70, still below confidence (0.74); a curve says nothing beyond its last point.
- The family-mean confidence AUROC is within 0.014 of its full-size value using about 150 training problems.
- *Added value moves from negative to slightly positive.* At small sizes, extra inputs may increase fitting
  variability, but that explanation was not isolated experimentally. At full size the added values are
  #f3(lc.derived.family_mean.L32.added.at("1.0").auroc) and
  #f3(lc.derived.family_mean.D.added.at("1.0").auroc). Last-doubling added-value gains are positive point estimates
  (shown above), but their intervals include zero: the increase is unresolved.
- An observation that was not planned: the looped model's probe is ahead when training problems are few, and the
  gap closes as they increase (L32 minus D: #lcd("0.125"), #lcd("0.25"), #lcd("0.5"), #lcd("0.75"), #lcd("1.0") at
  the five sizes). No interval was computed for this and it is one seed; it is recorded, not claimed.

So Section 4 is a statement about linear probes trained on about 1,200 problems. Whether the small added value
would grow with more problems is open; the curve does not establish that more data would have no benefit.

== How much of the predictable part is shared between the two models?

Here correlation is Pearson correlation of the two binary error indicators. If the four cells in the displayed
order are $a,b,c,d$, the odds ratio is $a d / (b c)$. Rank correlation below is Spearman correlation: ordinary
correlation of score ranks, with ties averaged.

#figure(
  tb(7, hd([Family], [both right], [only L32 right], [only D right], [both wrong], [correlation of the two models' errors], [odds ratio]),
    ..ex.joint.map(j => (famname.at(j.family), ..j.counts.map(str), fiv(j.phi, j.phi_bootstrap.interval),
      str(calc.round(j.odds_ratio, digits: 1)) + " [" + str(calc.round(j.odds_ratio_bootstrap.interval.at(0), digits: 1)) + ", " + str(calc.round(j.odds_ratio_bootstrap.interval.at(1), digits: 1)) + "]")).flatten()),
  caption: [Right and wrong answers of L32 and D on the same 1,500 problems per family, with the correlation between
  the two error indicators (intervals from resampling problems).],
)

#let sc(comp, fam) = f3(med(ex.scorecorr.filter(x => x.comparison == comp and x.family == fam).map(x => x.spearman)))
#figure(
  tb(5, hd([Rank correlation between two scores (median of three seeds)], ..fams.map(f => famname.at(f))),
    [L32's errors: probe on L32's state vs. probe on D's state], ..fams.map(f => sc("L32 labels: states", f)),
    [D's errors: probe on L32's state vs. probe on D's state], ..fams.map(f => sc("D labels: states", f)),
    [L32: own-state probe vs. its confidence], ..fams.map(f => sc("L32 own vs confidence", f)),
    [D: own-state probe vs. its confidence], ..fams.map(f => sc("D own vs confidence", f))),
  caption: [How similar the held-out scores are, problem by problem.],
)

- The two models' errors are positively but far from perfectly related: in every family several hundred problems
  are solved by exactly one of them.
- In the three sum families, the probe that predicts a model's errors gives nearly the same ranking of the problems
  whether it reads L32's state or D's (rank correlation 0.80 to 0.94). In ARC-Easy the two readings agree less
  (0.69 to 0.70), which is also where the own-state advantage of Section 3 is largest.
- In the sums, the probe and the confidence are only moderately related (0.30 to 0.48) and the probe is the weaker
  of the two; in ARC-Easy they are much more strongly related (0.71 to 0.80).

What this supports: much of what a linear probe predicts about one model's errors is _equally available from the
other model's state_. That is compatible with both models' states encoding the same features of the problem that
make it hard for both, but the data cannot identify what the probe reads, and "the probe mostly reads problem
difficulty" is not established. As a description only: knowing whether the _other_ model got the problem wrong
predicts a model's own error with an AUROC of 0.60 to 0.70 (one bit, and a quantity no real monitor has).

== Beyond 32 loops, on the same problems

#let deepmax(sel) = calc.max(..ex.deep.filter(sel).map(x => calc.abs(x.difference)))
#let deeplab(sel) = calc.max(..ex.deep.filter(sel).map(x => x.label_changed))
For the 256 problems per family that were run to 64 loops, take the probe fitted at 32 loops and compare its AUROC
at 32 loops with its AUROC at 48, 63 and 64 loops _on these same problems_. Over all four looped models, families,
seeds and the three deep budgets, the largest change in AUROC is #f3(deepmax(x => x.source == 32)) and the largest
fraction whose correctness labels changed is #f3(deeplab(x => x.source == 32)). For probes fitted at each model's own training mean
(4, 8 or 16 loops) and applied at 48 to 64 loops, the largest change is #f3(deepmax(x => x.source != 32)), with at most
#f3(deeplab(x => x.source != 32)) of the correctness labels changed. The native deep-budget probe's small training set is a confound,
not a demonstrated cause of its lower AUROC in Section 5. These are 256 problems per
family, so differences of a few hundredths in a single family cannot be excluded.

== The position before the last, paired by problem

#explore[
This item was added last, _after_ noticing in the registered secondary rows that the two models differ at this
position. A difference found this way, among about a dozen secondary rows, is an exploratory result, not a confirmation.
]

At position −2 the sites and own-error labels are those described in Section 4. L32/prev32 and D/prev12 both
precede four final layers. The models use the same 1,500 problems per family but their own label-stratified five
folds, with four inner folds selecting the penalty at each fixed site. No probes are refitted here.
For each exploratory draw, we resample 1,500 problems with replacement within each family, using identical problem
multiplicities for both models. We average each model's AUROC over its own five folds, then average the four
family differences equally. The 95% percentile intervals use #pv.draws draws and hold fitted probes fixed.
Unlike the registered intervals, this bootstrap does not stratify by outer-fold intersections.
D is shown at all its saved layers at this position; the point values reproduce the registered rows exactly.

#let pd(s, fam, site) = { let x = pvr(s, fam, site); fiv(x.difference, x.interval) }
#figure(
  tb(4, hd([L32 (after 32 loops) minus D, AUROC], [seed 0], [seed 1], [seed 2]),
    [mean of the four families, D at layer 12 (matched site)], ..(0, 1, 2).map(s => pd(s, "family_mean", "D/prev12")),
    [mean of the four families, D at layer 8], ..(0, 1, 2).map(s => pd(s, "family_mean", "D/prev8")),
    [mean of the four families, D at layer 16], ..(0, 1, 2).map(s => pd(s, "family_mean", "D/prev16")),
    table.hline(stroke: 0.3pt),
    ..fams.map(f => ([#famname.at(f), D at layer 12], ..(0, 1, 2).map(s => pd(s, f, "D/prev12")))).flatten()),
  caption: [Second-to-last token position: looped minus ordinary. The three seeds use the same problems and
  differ only in the random splits, so they are not three independent confirmations.],
)

- *There is a post hoc advantage for predicting each model's own errors at position −2.* Against matched layer
  12, the family-mean differences range from 0.021 to 0.037; all three conditional intervals exclude zero.
  Across all three saved D layers, the nine family-mean intervals are positive and point differences range from
  0.021 to 0.051. Against layer 12 every family/seed difference is positive, but only three of twelve
  individual-family intervals exclude zero.
- This satisfies E4's exploratory rule and makes the position a candidate for a registered test on new data.
  It does not show that the looped state is intrinsically more informative: the probes predict different errors.
  The comparison holds probes fixed, was selected after inspecting secondary rows, and reuses the same problems
  across seeds. Only three D layers and two L32 loop budgets were saved; no nested best-site comparison was made.
  This position alone performs worse with this fitting procedure; joint use of positions was not tested.
- The models have different error sets and accuracy (family means
  #r("accuracy", "L32 accuracy (fraction correct)", "family_mean") and
  #r("accuracy", "D accuracy (fraction correct)", "family_mean")). Lower error prevalence does not mechanically
  raise AUROC, but error types and class counts can affect predictability and fitting. L32's 8-loop probes already
  perform nearly as well as its 32-loop probes. They use 8-loop labels, so this is not a fixed-label convergence
  test; it nevertheless gives no evidence that loops 9–32 create the advantage. Repeated computation, token
  position, checkpoint differences and task-specific features remain possible explanations.
- A hypothesis for a next experiment, not something these data show: the state at the last position is what the
  answer is computed from, and both models are trained to make it serve the answer. At earlier positions the two
  architectures are freer to differ. The main comparison of this study looks at the one position where the two
  models are most constrained to agree.

= What this does and does not show

Against the outcomes the design listed in advance (its Section 12):

- "Two-by-two table flat, small added value over confidence in both: state probes add little to asking the model,
  in either architecture." *This is the closest listed outcome.* No overall advantage of either model was resolved,
  and the added value is a few thousandths in both. The table is not literally flat: each model's errors are read
  slightly better from its own state (Section 3), and the families differ in sign.
- "Transfer holds everywhere: budget changes are benign; the forecast adds little." *Observed in part.* Ranking
  changes little at the tested budgets from 4 to 32 loops, and for the same probe up to 64 on the deep subset. At
  1 loop the ranking is worse, and at 2 loops the frozen threshold rejects more. This does not establish that
  budget changes are safe in general. The forecast showed no general advantage over a constant.
- The outcomes that would have pointed to a looping-specific benefit (the looped state more informative; dynamics
  adding to confidence; a forecast that flags failing transfer) were *not observed* in the registered main
  comparisons. The one exploratory result in that direction is at the second-to-last position (Section 7.4); it
  concerns each model's own errors and is not a confirmation.

Limits, in decreasing order of importance:

+ *One pair of trained models.* A difference or its absence is a fact about these two checkpoints. They also
  differ in ways other than looping (the looped model has 14 layers with six reused, D has 16), and a second
  training run of either could differ by more than what is measured here.
+ *Honest mistakes.* A probe that predicts ordinary errors on arithmetic says nothing about a model that conceals
  something. The oversight question that motivates this work is only touched at its easy end.
+ *Linear probes on about 1,200 problems.* See the learning curve: the comparison is between probes at this
  training size. A richer reader, or far more problems, could separate the models.
+ *Conditional uncertainty.* The intervals hold the fitted probes fixed and the three seeds share the same
  problems. These intervals omit variability from fitting and model training; they do not quantify the full uncertainty.
+ *Selected short problems,* four families, and the last token position for all main results. At the position
  before it the picture is not the same (Section 7.4).
+ *Storage precision* limits what can be said about very late, very small movements of the state (Section 6).

*What would make the question sharper.* (i) A second matched pair, to know how large checkpoint-to-checkpoint
variation is; without it no result of this kind can be attributed to looping. A registered test at position −2
should cross both models' errors with both models' states, use shared folds, save all layers and use new problems
and a second pair of models. (ii) Tasks where the answer does not settle within four loops, since on these
problems the loop is nearly idle after loop 8. (iii) For the forecast, a bound specific to the last row and probe
direction is worth investigating. The summaries do not identify how much conservatism comes from each inequality,
the estimated ratio or storage rounding.

= How the numbers were produced

- *Data.* One forward pass of each model per problem, no gradients, no interventions; states at the last token
  position for the looped models after 1, 2, 3, 4, 6, 8, 12, 16, 24, 31, 32, 33 loops (and 48, 63, 64 for 256
  problems per family), every layer for D, plus the output summaries and step sizes. Eight collection runs on a cloud GPU.
- *Analysis.* The frozen script `huginn_addition/analyze_mon.py`, three seeds, 2,000 resamples. Seed 0 ran locally,
  seeds 1 and 2 on cloud CPUs; a partial rerun of seed 0 in the cloud agreed with the local run to approximately
  $2 times 10^(-5)$ over the 30,126 numbers compared. No fit failed to converge and nothing was skipped.
  There are 135 unavailable per-family audit interval entries, all for add3 at deep targets, plus 135 propagated
  unavailable family-mean entries. These count reporting entries, not distinct fits. Point estimates remain
  available; any class-losing bootstrap draw makes the entire interval unavailable under the registered rule.
- *A stale label.* The result files describe the confidence features as "unpenalized" in a settings field. That
  wording predates the last design amendment: a fixed small penalty ($10^(-6)$) was applied to them, as the design
  states. The principal-component check standardizes the component scores before fitting, like every other input.
- *Tables and figures.* Generated by `huginn_addition/tables_mon.py`, `plot_mon.py`, `mon_explore_scores.py`,
  `mon_learning_curve.py`; every number in a table of this document is read from their output files at compile
  time. All per-family tables: `docs/monitor_results_gen/preview.pdf`. Per-seed values of everything:
  `results/mon_report-20261110/report.md`.
- *Review.* GPT-6 Astra reviewed the design, the code and the frozen package before the main data, and the results
  afterwards (it recomputed 240 AUROCs from the saved scores, rebuilt the labels, the splits and the forecast rule
  independently; all matched). Its verdict on the results was "accept with changes"; the changes concerned wording
  and presentation and are incorporated, together with the second review's changes in
  `results/mon_report-20261110/astra_review_writeup.md`.

#pagebreak()
#set heading(numbering: "A.")
#counter(heading).update(0)
#import "monitor_results_gen/tables.typ": tab_seeds, tab_cross, tab_budget_L32_families

= Per-seed values of the main quantities

Family means; each seed with its own 95% interval; last column: smallest and largest of the three values. Row
names are those of the result files: `m1a` two-by-two table (`table/R/C`: errors of R read from the state of C)
and its contrasts; `m1b` best-layer difference; `m1c` added value over confidence (`flex`: flexible baseline;
`False/False`, `False/True`: L32 minus D in added value against the ordinary and the flexible baseline);
`m4` dynamics added to confidence; `m1s` shuffled labels.

#text(size: 8pt, tab_seeds)

= Probes applied to another family

Rows: the family the probe was fitted on (first four rows L32, last four D). Columns: the family it is applied to
(own family: held-out AUROC).

#text(size: 8pt, tab_cross)

= Native minus frozen probe, L32, by family and budget

Probe fitted at 32 loops and applied at the budget in the first column, compared with a probe fitted at that
budget. † : the 256-problem subset, where the native probe has a much smaller training set (Sections 5 and 7).

#text(size: 7.5pt, tab_budget_L32_families)
