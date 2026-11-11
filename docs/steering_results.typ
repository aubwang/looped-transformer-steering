#set document(title: "Pushing a looped model through its loop")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#show table: set par(justify: false)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => [#v(0.6em) #it #v(0.2em)]
#show figure.caption: set text(size: 9.5pt)
#set math.equation(numbering: "(1)")
// References to equations print as "(3)".
#show ref: it => {
  let el = it.element
  if el != none and el.func() == math.equation {
    link(el.location(), numbering("(1)", ..counter(math.equation).at(el.location())))
  } else { it }
}

#let boxed(title, color, body) = block(
  fill: color, inset: 9pt, radius: 3pt, width: 100%, breakable: true,
  [*#title.* #body],
)
#let plain(body) = boxed("In words", rgb("#fbf3e6"), body)
#let caveat(body) = boxed("Careful", rgb("#fdeeee"), body)
#let explore(body) = boxed("Computed after seeing the main results", rgb("#eaf1fb"), body)
#let explore-tight(body) = block(fill: rgb("#eaf1fb"), inset: 9pt, radius: 3pt, width: 100%, breakable: false,
  [*Computed after seeing the main results.* #body])

// ---- Numbers. The empirical summaries of the second model, of the screen and of the comparison tables are read
// from steering_gen/numbers.json, written by huginn_addition/steering_report.py from the saved run outputs.
#let D = json("steering_gen/numbers.json")
#let N = D.N
#let H = D.H
#let RS = D.RS
#let cells = json("monitor_results_gen/tables.json")
#let R = {
  let r = (:)
  for c in cells { r.insert(c.table + "|" + c.row_label + "|" + c.column_label, c.formatted) }
  r
}
#let mon(t, row, col) = R.at(t + "|" + row + "|" + col)
#let pt(s) = s.split(" [").at(0)
#let tb(cols, ..args) = align(center, block(breakable: false, text(size: 9pt, table(
  columns: cols, inset: (x: 5pt, y: 3.4pt),
  stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
  ..args))))
#let hd(..a) = table.header(..a.pos().map(x => strong(x)))

#align(center)[
  #text(16pt, weight: "bold")[Pushing a looped model through its loop]
  #v(0.1em)
  #text(12pt)[Local score levers on two models, a self-gain diagnostic on three, and an error-detection
  comparison on one matched pair]
  #v(0.2em)
  #text(9.5pt)[Draft 5, 2026-11-11. How the experiments were registered, reviewed and computed: @provenance.]
]

#v(0.4em)

#plain[
A _looped_ language model applies the same block of computations again and again to an internal state, and reads
its answer off that state at the end. We asked how the scores it gives to candidate answers respond when a small
fixed vector is added to the state at every loop, and which vector moves a chosen score most. "Steering" below
means moving scores in this sense, not changing answers: on the second model, across the #N.flip_pairs pairs of a
held-out problem (#N.n_eval) and a candidate other than its answer (9 of 10), no push at any tested dose made
the pushed candidate the answer, and at the largest lever dose #N.flip_other_09 of the #N.flip_pairs pushes, on
three problems, changed the answer to something else.

+ *The effect of a sustained push follows the loop's own linear response, at the doses tested.* Differentiating
  one loop at the settled state gives a matrix $J$; to first order the change of the answer scores is
  $G (J + J^2 + dots.c + J^T) P$ for a push $P$, with $G$ the derivative of the readout. That formula is standard
  calculus (@objects), not a finding. The finding is how far it reaches: on the second model it matches the
  measured change with median per-problem correlation #N.corr_FULL_03, slope #N.slope_FULL_03 and relative error
  #N.err_FULL_03 at the main dose, a push of 3% of the length of the state
  (first model: #H.corr_full, #H.slope_full, #H.err_full).
+ *The direction the readout is most sensitive to has little effect.* The tested push along the gradient of the
  readout changes the target score by #N.eff_G_03 on average on the second model and #H.eff_G on the first. The
  direction built from the loop's response (the _lever_) changes it by #N.eff_L_03 and #H.eff_L. To first order
  this is the same fact as the lever being nearly orthogonal to the gradient and having almost no immediate
  effect (Proposition 3): one number, read three ways.
+ *Immediate and one-loop responses differ strongly, in both models* (@fig-course). Gradient pushes have a large
  immediate response by the derivative and a small response of the target score one loop later; lever pushes have
  a small or negative immediate response and a positive one a loop later. The one-loop values are derivative
  quantities on the first model and measurements on the second. On the second model the mean target effect is
  already close to its final value after one loop, although the full table of responses keeps changing.
+ *Repetition matters for a few loops.* On the second model, at the smallest dose (1% of the length of the
  state, where the error from non-linearity is smallest), a prediction using a single pass through the block has
  the right shape (correlation #N.e1_corr_01_t1) but a relative error of #N.e1_err_01_t1 against the final
  response\; with four passes the error is #N.e1_err_01_t4, and at the sampled counts of eight and more it agrees
  with that of the full sum (#N.e1_err_01_t48) to the displayed precision. At the 3% main dose the three figures
  are #N.e1_err_03_t1, #N.e1_err_03_t4 and #N.e1_err_03_t48.
+ *Whether the loop suppresses the directions the output reads, relative to random ones, depends on the model*
  (@three). A
  cheaper diagnostic asks how much of a push along a direction comes back along that same direction after one
  loop, for directions the output reads and for random directions. In the first model a random direction keeps
  about a quarter of itself and a read direction almost nothing. In the second, measured for this draft, the
  sampled random directions, read directions and task gradients all keep at most a few hundredths of themselves
  on the sums, and no deficit of the read directions is detected. The interval for the read directions rules out
  a deficit larger than 0.002, a hundred times smaller than the first model's. This does not explain the steering
  response: the diagnostic measures the component along the original direction, not the length of what a push
  becomes (on that, from saved outputs: a random push shrinks to about a fifth of its length in one loop of the
  second model, a lever grows by a factor #N.jl_median). In a third looped model of another design, Ouro-1.4B, read directions
  keep slightly more than random ones.
+ *A separate question, whether looping makes errors easier to detect, resolved no overall advantage* in the
  registered comparison for one matched pair of models (@monitoring), and none of #N.n_fam further task families
  screened for late changes of the answer passed every criterion.

The steering results are local statements about small pushes around a settled state, on two models of one
design, one task family each. The pushes shift scores on problems the model already answers correctly, by a
small fraction of the lead of the correct answer. They do not show that looping helps a model reason, and the
two experiments differ in task, dose and numerical precision, so the columns of the comparison table are two
experiments side by side, not a controlled comparison of models.
]

= The objects <objects>

*Conventions.* Vectors in $RR^d$ are columns. $u^top$ is the transpose, $chevron.l u, v chevron.r = u^top v$ the usual
inner product, $norm(u) = sqrt(chevron.l u\, u chevron.r)$ the length. $I$ is the identity matrix and $J^s$ means
$J$ multiplied by itself $s$ times (the superscript $top$ always means transpose; $T$ will be a number of loops).
The trace $"tr"(J)$ is the sum of the diagonal entries of $J$; it equals the sum of its eigenvalues.

*The model, as far as needed.* The model is given a text, the _prompt_, split into positions. Its computation is
a composition of vector-valued maps, the _layers_. Each layer depends on fixed numerical parameters, the
_weights_, chosen beforehand by an optimisation that makes the model predict the next piece of a large body of
text (_training_; a _pretrained_ model is one that has been through it). Everything below concerns the last position, where the model is
about to produce its answer.

*State and loop.* The model holds a vector $x in RR^d$ at the last position ($d = 5280$ in the first model, $2048$
in the second) and vectors at the earlier positions, collected in $y$. One loop updates all of them. Earlier
positions cannot depend on the last one, but the last one depends on them: its update is
$x_(t+1) = F(x_t\; y_t)$. For the local analysis we hold the earlier states fixed at the values $y^*$ they have
reached, write $F(x) = F(x\; y^*)$, and idealise the state $x^*$ reached at the last position as an exact fixed
point, $F(x^*) = x^*$. On the problems used here the iteration does settle ($x_(t+1) approx x_t$ after enough
loops). Because the earlier positions cannot depend on the last one, their trajectory $y_t$ is the same with and
without a push at the last position. Holding them at $y^*$ in the analysis therefore neglects only whatever
movement they still have of their own after the loops that precede the push, not any effect of the push on them.

*Readout.* The model's text vocabulary is a finite list of _tokens_ (pieces of text; a few are reserved control
symbols, the others are called ordinary below). From the state at the last position the model computes one real
number $ell_j$ for each token $j$, its _score_ (in the field's vocabulary, a logit). It assigns token $j$ the
probability $p_j = exp(ell_j) \/ sum_i exp(ell_i)$ and its answer is the token with the largest score. Here the answer is one of ten candidates (ten digits
in the first model, the sums 70 to 79 in the second), each a single token. The readout $z : RR^d -> RR^(10)$
gives the ten candidates' scores, centred: $z_k = ell_k - 1/10 sum_j ell_j$ with the sum over the ten candidates,
so $z_k (x)$ says how much candidate $k$ stands out.

*Derivatives.* Assume $F$ and $z$ are twice continuously differentiable near $x^*$ (in these two models the
layers are built from sums, products, quotients with positive denominators, exponentials and other smooth
functions, so this is not an idealisation). Two matrices describe the
neighbourhood of $x^*$:
$ J = D F(x^*) in RR^(d times d), quad G = D z(x^*) in RR^(10 times d). $
Row $k$ of $G$, written $G_k$, is a row vector; $G_k^top$ is the gradient of score $k$ as a column. For $epsilon > 0$ and
$G_k != 0$, among unit vectors $u$ the first-order immediate change $epsilon G_k u$ of
$z_k (x^* + epsilon u) - z_k (x^*)$ is largest for $u = G_k^top \/ norm(G_k)$.

*A sustained push.* Fix $P in RR^d$ and add it to the state before every loop:
$ x_0 = x^*, quad x_(t+1) = F(x_t + P). $ <forced>

*Proposition 1 (linear response).* Let $delta_t = x_t - x^*$ under @forced and
$A_T = J + J^2 + dots.c + J^T$. For each fixed whole number $T >= 1$, as $P -> 0$,
$ delta_T = A_T P + O_T (norm(P)^2), quad z(x_T) - z(x^*) = G A_T P + O_T (norm(P)^2), $ <resp>
where the constants in the remainders may depend on $T$. If every eigenvalue of $J$ has modulus below 1, then
$A_T -> (I - J)^(-1) J$ as $T -> infinity$.

_Proof._ We show $delta_t = A_t P + O_t (norm(P)^2)$ for each fixed $t >= 0$, with $A_0 = 0$. It holds at
$t = 0$, where $delta_0 = 0$. If it holds at $t$, then $delta_t = O_t (norm(P))$, and Taylor's formula for $F$ at
$x^*$ gives
$delta_(t+1) = F(x^* + delta_t + P) - F(x^*) = J delta_t + J P + O(norm(delta_t + P)^2) = (J A_t + J) P + O_(t+1) (norm(P)^2)$.
So $A_(t+1) = J A_t + J$, hence $A_T = J + dots.c + J^T$. The sum starts at $J$, not at the identity: the push is added before the loop and the
state is read after it. Taylor's formula for $z$ gives the second statement. For the limit,
$(I - J) A_T = J - J^(T+1)$; the eigenvalue condition implies $J^(T+1) -> 0$ and that $I - J$ is invertible. The
limit concerns the matrices $A_T$ only; nothing is claimed about the remainders uniformly in $T$. #h(1fr) $square$

*Proposition 2 (the lever).* Fix $k$ and a length $r > 0$, and write $v = A_T^top G_k^top$. Among pushes with
$norm(P) = r$, the predicted change $G_k A_T P$ of score $k$ in @resp is largest for
$ P = r thin v \/ norm(v) quad "if" v != 0, $ <lever>
where it equals $r norm(v)$; if $v = 0$ it is zero for every push.

_Proof._ $G_k A_T P = (A_T^top G_k^top)^top P = chevron.l v, P chevron.r$. If $v = 0$ this is zero for every $P$. If
$v != 0$, by the Cauchy–Schwarz inequality $chevron.l v, P chevron.r <= norm(v) norm(P)$, with equality among
pushes of length $r$ only at $P = r v \/ norm(v)$.
#h(1fr) $square$

*Immediate against later effects.* For a push $P$ and a score $k$ put $c_s = G_k J^s P$ for $s = 0, 1, 2, dots$.
Then $c_0 = G_k P$ is the first-order change that would be read at once, with no loop; $c_1 = G_k J P$ is the first-order
change read after one loop; and the predicted change after $T$ loops in @resp is $c_1 + c_2 + dots.c + c_T$. The
gradient direction $P = r G_k^top \/ norm(G_k)$ maximises $c_0$. Its $c_1$ equals $r G_k J G_k^top \/ norm(G_k)$, a
number determined by how $J$ acts on that one direction; nothing forces it to be large, or positive, even when $J$
is far from zero. The lever @lever maximises $c_1 + dots.c + c_T$; that optimisation does not separately constrain
$c_0$.

*Proposition 3 (one number, three readings).* On one problem, let $G_k != 0$ and $v = A_T^top G_k^top != 0$, and
write $u = G_k^top \/ norm(G_k)$ for the unit gradient and $ell = v \/ norm(v)$ for the unit lever. Then
$ (G_k A_T u) / (G_k A_T ell) = chevron.l ell, u chevron.r = (G_k ell) / (G_k u). $ <three-readings>
In words: the predicted effect of the gradient push after $T$ loops, as a fraction of the lever's\; the cosine of
the angle between lever and gradient\; and the immediate effect $c_0$ of the lever, as a fraction of the
gradient's, are the same number.

_Proof._ $G_k A_T u = chevron.l v, u chevron.r = norm(v) chevron.l ell, u chevron.r$ and
$G_k A_T ell = chevron.l v, ell chevron.r = norm(v)$. Also $G_k ell = norm(G_k) chevron.l u, ell chevron.r$ and
$G_k u = norm(G_k)$. #h(1fr) $square$

So, to first order and on a single problem, "the gradient push does little after $T$ loops", "lever and gradient
are nearly orthogonal" and "the lever does almost nothing at once" are one statement, namely that
$G_k A_T G_k^top$ is small against $norm(G_k) norm(A_T^top G_k^top)$. The experiments use directions averaged over
other problems, for which @three-readings holds only approximately; the three measured ratios are compared in
@both.

*Self-gain.* For a unit vector $u$ put $alpha(u) = chevron.l u, J u chevron.r = chevron.l u, J^top u chevron.r$: of a push of length $epsilon$ along $u$,
the component along $u$ after one loop is $epsilon alpha(u)$ to first order. For the gradient direction
$u = G_k^top \/ norm(G_k)$ of the same problem (when $G_k != 0$) this is exactly the ratio of the two effects above,
$alpha(u) = c_1 \/ c_0$; for a direction averaged over other problems, such as $U_k$ below, it is not. If $u$ is
drawn uniformly from the unit sphere, the average of $u u^top$ is $I \/ d$, so the average of $alpha(u)$ is
$ tau = "tr"(J) \/ d, $ <tau>
the mean of the eigenvalues of $J$. The difference $alpha(u) - tau$ from this mean over random directions is
called the _excess_ of $u$ (@three). All self-gains are governed by the symmetric part of $J$,
$alpha(u) = chevron.l u, 1/2 (J + J^top) u chevron.r$; a small $alpha$ says nothing about the length of $J u$.

*What is standard and what is measured.* Propositions 1 and 2 are first-order calculus: Taylor's formula and the
Cauchy–Schwarz inequality. They hold for any twice differentiable map at a fixed point, looped language model or
not. At zero push, the lever $A_T^top G_k^top$ is the gradient of the score read after $T$ forced loops with respect
to the repeated push, which is what differentiating backwards through the $T$ loops computes. Under the
eigenvalue condition of Proposition 1, $I - J$ is invertible, so by the implicit function theorem the equation
$x = F(x + P)$ has, for small $P$, a unique solution $x(P)$ near $x^*$, differentiable in $P$, with $x(0) = x^*$
and $D_P x(0) = (I - J)^(-1) J$. The limit of the lever is $J^top (I - J^top)^(-1) G_k^top$, the gradient at
$P = 0$ of the score at that equilibrium. Differentiating
through such an implicit equation is the principle by which models defined through a fixed point are trained
("deep equilibrium" models; Bai, Kolter and Koltun 2019); the extra factor $J$ comes from adding the push before
the loop. Earlier work found by a later search is described in @related. So that a
lever exists and that @resp holds for small enough pushes are not findings. What had to be measured is: (i) how
large a push may be before @resp fails, when the directions are averages over other problems and the earlier
positions still move slightly; (ii) how $J$ acts on the directions the readout is sensitive to, that is, $c_1$ against
$c_0$ and $alpha$ against $tau$, which calculus leaves open; (iii) how many terms of $A_T$ matter.

*The protocol.* Directions are built on one set of problems (_training_ problems) and tested on another
(_held-out_ problems). For each candidate $k$ three unit vectors are fixed in advance:

- the lever $L_k$: the average, over training problems, of the unit vectors @lever, rescaled to length one;
- the gradient push $U_k$: the same average of the unit gradients $G_k^top \/ norm(G_k)$;
- the centroid push $C_k$: the unit vector from the average settled state to the average settled state of
  problems whose answer is $k$.

An average of per-problem best directions need not be the best single direction for the average problem; these
are fixed test directions, not optimal ones. On each held-out problem the push $P = r F_k$, for a family
$F in {L, U, C}$ and a length $r$, is applied as in @forced, and the change of the ten scores is recorded against a
run without push. Three things are reported.

- _Target effect:_ the change of score $k$ under push $r F_k$, averaged over $k$ and over problems.
- _Prediction:_ for each problem, the measured $10 times 10$ table (push $r L_k$, score $m$) against the table
  $r G A_T L_k$ computed with that problem's own $J$ and $G$ (FULL); with $A_T$ averaged over training problems
  (AVERAGED); with another problem's $A_T$ (SWAPPED); and with $A_T$ replaced by the identity, i.e. no loop
  (IDENTITY). The score is the correlation between predicted and measured tables, per problem; for FULL also the
  slope of measured on predicted and the relative error $norm("predicted" - "measured") \/ norm("measured")$, with
  the tables read as vectors of 100 numbers. The median over problems is reported.
- _Correlation_ is the usual sample correlation coefficient of the two tables read as 100 pairs of numbers, and
  _slope_ the least-squares slope of measured on predicted, with intercept.
- _Intervals_ are 95% bootstrap intervals over held-out problems (the 2.5% and 97.5% points of the statistic
  over 5,000 samples of the problems drawn with replacement); for a difference of two predictions they are
  intervals for the mean, over problems, of the difference of the two correlations.

= The first model: Huginn-0125 <huginn>

Huginn-0125 is a looped model trained as such from the start. The task is two-digit addition; the ten candidates
are the possible units digits of the sum; pushes have length $r = 3$; 20 held-out problems; 48 forced loops;
computed with about 7 significant digits (single precision). These results are from the reviewed write-up
`huginn_addition/WRITEUP.md` (Sections 3 to 5), where the methods, checks and limits are given in full.

- *Levers have a larger mean target effect than the tested gradient and centroid pushes.* Target effect #H.eff_L for the lever (difference to
  the gradient push #H.L_minus_G), #H.eff_G for the gradient push and #H.eff_C for the centroid push; a random
  direction moves a score by about #H.random_abs in absolute value. 189 of the 200 lever effects are positive.
- *The loop's linear response predicts the whole table.* FULL correlation #H.corr_full, slope #H.slope_full,
  relative error #H.err_full. Without the loop the prediction is _anti_-correlated with what happens
  (#H.corr_identity).
- *Immediate and one-loop responses of the target score differ.* By the derivatives (a separate analysis of the
  same pushes; medians), the gradient push has $c_0 =$ #H.c0_G and $c_1 =$ #H.c1_G\; the lever has $c_0 =$ #H.c0_L
  and $c_1 =$ #H.c1_L. Of 157 lever effects that start negative and end positive, 154 have changed sign after a
  single application of $J$. The reported length ratio $norm(J P) \/ norm(P)$ for the gradient pushes is 1.17, so
  the loss of the target response does not come from the whole perturbation shrinking.
- *One-loop sensitivity enters mainly from outside the span of the gradients.* The ten centred scores sum to
  zero, so their gradients span a subspace of dimension at most 9 (here 9). Let $Q in RR^(d times 9)$ have
  orthonormal columns spanning it, and let $norm(dot)$ of a matrix be the square root of the sum of its squared
  entries. Then $norm(Q^top J Q) \/ norm(Q^top J) =$ #H.rho_span. For comparison, a 9-dimensional subspace
  in no special position in $RR^(5280)$ would give about $sqrt(9 \/ 5280) approx 0.04$, and a subspace carried
  into itself by $J^top$ would give the value 1; so #H.rho_span is not small for geometric reasons alone, but it is well below that.
  The levers lie mostly outside the span ($norm(Q^top P) \/ norm(P) =$ #H.inspan_L, against #H.inspan_G for
  the gradient pushes) and have three times the one-loop gain onto it ($norm(Q^top J P) \/ norm(P) =$ #H.gain_L
  against #H.gain_G).
- *One pass already gives a good lever.* The one-loop version of @lever, $J^top G_k^top$, reaches #H.cheap_onestep,
  that is #H.cheap_onestep_share of the full lever's effect. In this model the state enters the block through a
  fixed matrix $W in RR^(d times d)$ (the block is applied to $W x$ plus a term that does not depend on $x$); the direction $W^top G_k^top$, which costs nothing beyond the gradient, reaches #H.cheap_adapter
  (#H.cheap_adapter_share). Random directions with the same inner product with the full lever reproduce
  #H.ovl_a_ctrl_share to #H.ovl_b_ctrl_share of these effects; the levers exceed them by #H.ovl_b (one-loop) and
  #H.ovl_a (through $W$; each interval is at the 97.5% level, so that the two hold jointly at 95%, and is conditional on
  the two random directions drawn per candidate).
- *The same construction works on a second readout.* For true/false statements, the lever moved the
  true-against-false score $ell_"True" - ell_"False"$ by a median #H.dstar at push length 12.5, against 0.93 for the gradient and #H.probe for
  a direction that _separates_ true from false statements in the state (the difference of the two class averages).
  In the reviewed wording this is a lever on the readout; it was not shown to change what the model represents as
  true.

#caveat[
In the first model the prediction was validated at small doses; at the full dose of an earlier experiment the
response was larger than linear. The statements about one loop (the sign change, the feeding from outside the
gradients' span) are statements about derivatives at the settled state, not measurements of what flows through the
network. A minority of prompts in that model do not settle at all but alternate between two states; they are
outside the steering experiments.
]

= The second model: Recurrent-Llama-3.2 L32 <llama>

L32 is an ordinary pretrained model that was rebuilt afterwards so that six of its layers form a loop, and trained
further with about 32 loops (McLeish et al., arXiv 2511.07384). It is the looped member of the matched pair of
@monitoring. It differs from Huginn in origin, size and training, which is the point of repeating the experiment.

*Protocol.* The main design, the decision rule and the checks were written down before the runs
(`docs/llever_design.md`) and reviewed with the code by GPT-6 Astra before launch; the results were recomputed
afterwards. Task: two-digit additions whose sum is one of 70, …, 79, each a single token of this model\; every prompt
puts one of two fixed lists of four solved sums in front of the unsolved one; #N.n_eval held-out problems, 40 training problems for $L$ and $U$, 100
further problems for $C$. The settled state is taken after 64 loops, then 48 forced loops. Push lengths are 1%, 3%
and 9% of the typical length of the state (#N.m_ref), i.e. $r =$ #N.norm_01, #N.norm_03 and #N.norm_09\; 3% is the
main dose. Computed with about 16 significant digits (double precision).

*Checks, all passed.* The released program runs all loops inside one routine that cannot be differentiated loop
by loop and rounds some intermediate quantities to 7 digits. Our code therefore calls the model's layers
directly, one loop at a time, with those roundings replaced by 16-digit arithmetic. Run without a push, the two
agree (both with the 16-digit replacements) to a largest score difference of #N.g1_max\; this checks our
loop-by-loop code, not the replacements. The numbers $G_k A_T L_m$ computed two ways (differentiating backwards for $A_T^top G_k^top$, and
finite differences forwards for $A_T L_m$) agree to a median relative difference of #N.adjoint. The largest ratio
of the last term of a sum $A_T$ to the sum, over the directions used, is #N.tail_max. All #N.sham_correct held-out
problems are answered correctly without a push, at 32 and at 64 loops, and the median relative movement of the
unpushed state in its last loop is #N.sham_drift.

*Verdict under the rule fixed in advance:* the pattern of @huginn replicates, at all three doses. The rule
required (R1) a lever effect that is positive and larger than the gradient push's, and (R2) a FULL correlation of
at least 0.90 that exceeds IDENTITY; the intervals for the lever effect, for lever minus gradient push, and for
FULL minus IDENTITY all had to lie above zero. The label certifies this pattern, not more.

#figure(
  placement: auto,
  tb((auto, auto, auto),
    hd([Quantity], [Huginn-0125], [Recurrent-Llama L32]),
    [Push length $r$], [3], [#N.norm_03 (3% of the state)],
    [Target effect, lever $L$], [#H.eff_L], [#N.eff_L_03 #N.eff_L_ci_03],
    [Target effect, gradient push $U$], [#H.eff_G], [#N.eff_G_03 #N.eff_G_ci_03],
    [Target effect, centroid push $C$], [#H.eff_C], [#N.eff_C_03 #N.eff_C_ci_03],
    [$L$ minus $U$], [#H.L_minus_G], [#N.L_minus_G_03],
    table.hline(stroke: 0.3pt),
    [FULL: correlation / slope / error], [#H.corr_full / #H.slope_full / #H.err_full], [#N.corr_FULL_03 / #N.slope_FULL_03 / #N.err_FULL_03],
    [AVERAGED: correlation], [#H.corr_averaged], [#N.corr_AVERAGED_03],
    [SWAPPED, other answer: correlation], [#H.corr_swapped], [#N.corr_SWAPPED_03],
    [SWAPPED, same answer: correlation], [#H.corr_swapped_same_digit], [#N.corr_SWAPPED_SAME_CLASS_03],
    [IDENTITY (no loop): correlation], [#H.corr_identity], [#N.corr_IDENTITY_03],
    [FULL minus IDENTITY], [#H.full_minus_identity], [#N.FULL_minus_IDENTITY_03],
    [FULL minus AVERAGED], [#H.full_minus_averaged], [#N.FULL_minus_AVERAGED_03],
    [FULL minus SWAPPED, other answer], [#H.full_minus_swapped], [#N.FULL_minus_SWAPPED_03],
    table.hline(stroke: 0.3pt),
    [Immediate derivative effect $c_0$, $U$ / $L$], [#H.c0_G / #H.c0_L], [#N.imm_G_03 / #N.imm_L_03],
    [After one loop, $U$ / $L$], [#H.c1_G / #H.c1_L (derivative)], [#N.course_G_03_t1 / #N.course_L_03_t1 (measured)],
  ),
  caption: [The two experiments side by side, main dose of each. Tasks, doses and precision differ, so differences
  between the columns cannot be attributed to the models. Correlations, slope and error are medians over held-out
  problems (20 and #N.n_eval). Brackets are 95% bootstrap intervals over problems: for target effects and $L$ minus
  $U$, of the mean; for FULL minus another prediction, of the mean per-problem difference of correlations (not the
  difference of the displayed medians). The last two rows of the Huginn column are derivative summaries from a
  separate analysis of the same pushes (medians over problems of medians over candidates). In the L32 column the
  immediate row is a derivative prediction averaged over problems and candidates, and the one-loop row the
  corresponding measured mean; the immediate values were computed after the main results, the measured time course
  was planned.],
) <tab-main>

#figure(
  placement: auto,
  image("steering_gen/figs/fig_pred.pdf", width: 92%),
  caption: [Second model, main dose: every entry of the measured response tables (#N.n_eval problems, ten lever
  pushes, ten candidate scores) against its prediction. Left: with the problem's own loop operator, $r G A_T L_k$.
  Right: with the loop left out, $r G L_k$. The line marks equality.],
) <fig-pred>

*The prediction is quantitative.* At the main dose the measured table is the predicted one up to a relative error
of #N.err_FULL_03 (@fig-pred). Effects are close to proportional to the dose over a ninefold range: #N.eff_L_01,
#N.eff_L_03 and #N.eff_L_09 at 1%, 3% and 9%, ratios #N.dose_ratio_low and #N.dose_ratio_high against $1\/3$ and
$3$. The error of the prediction grows with the dose, as a first-order statement should: #N.err_FULL_01,
#N.err_FULL_03, #N.err_FULL_09 (correlations #N.corr_FULL_01, #N.corr_FULL_03, #N.corr_FULL_09).

*Each problem's own operator predicts best.* Averaging $A_T$ over training problems, or borrowing it from another
problem, lowers the correlation (@tab-main), most for a problem with the same answer in the other set of worked
examples (#N.corr_SWAPPED_SAME_CLASS_03). The losses are larger than in the first experiment, but the differing
protocols prevent attributing that to the model.

#figure(
  placement: auto,
  image("steering_gen/figs/fig_course.pdf", width: 68%),
  caption: [Second model, main dose: mean change of the target score against the number of forced loops before
  reading (the sampled loop counts are equally spaced on the horizontal axis). Filled points are measured. The
  hollow points at 0 are the immediate effects $c_0 = r G_k F_k$ predicted by the derivative, i.e. the change that
  reading $x^* + P$ without any loop would show; no such reading was taken.],
) <fig-course>

#explore-tight[
*Immediate and one-loop responses of the target score differ, as in the first model* (@fig-course). The gradient
push has immediate effect $c_0 =$ #N.imm_G_03 by the derivative, and a measured mean effect of #N.course_G_03_t1
after one loop. The lever has $c_0 =$ #N.imm_L_03 (negative for #N.imm_L_neg_03 of the problem–candidate pairs) and a
measured #N.course_L_03_t1 after one loop. These are means of the target score; they do not say that every part
of a gradient push's effect has vanished. The first model's $c_0$ for the lever was clearly negative and its
loop-free prediction anti-correlated with the response; here $c_0$ is about zero and the loop-free prediction
weakly positively correlated (#N.corr_IDENTITY_03). So a related picture holds in both, that what the readout is
most sensitive to is not what the loop passes on to the target score; an identical mechanism is not established.
]

== What repetition adds <passes>

#explore[
*Measured.* The mean target effect is already close to its final value after one loop (#N.course_L_03_t1, against
#N.course_L_03_t48 after 48). The whole table is not: after one forced loop it differs from its final value by a
median relative #N.disc_03_t1 (correlation #N.corr_t1_end_03), after two loops by #N.disc_03_t2, after four by
#N.disc_03_t4, after eight by #N.disc_03_t8.

*Predicted.* A follow-up run, specified after the main results and before it was launched
(`docs/llever_design.md`, Section 8), computed the partial sums $A_t = J + dots.c + J^t$ and compared the predictions
$r G A_t L_k$ with the final measured response (@fig-passes and the table below). It reproduced the first
prediction run #N.e1_reproduces.

#tb((auto, auto, auto, auto, auto),
  hd([Passes $t$], [Error, 1% dose], [Correlation, 1% dose], [Error, 3% dose], [Correlation, 3% dose]),
  [1], [#N.e1_err_01_t1], [#N.e1_corr_01_t1], [#N.e1_err_03_t1], [#N.e1_corr_03_t1],
  [2], [#N.e1_err_01_t2], [#N.e1_corr_01_t2], [#N.e1_err_03_t2], [#N.e1_corr_03_t2],
  [4], [#N.e1_err_01_t4], [#N.e1_corr_01_t4], [#N.e1_err_03_t4], [#N.e1_corr_03_t4],
  [8], [#N.e1_err_01_t8], [#N.e1_corr_01_t8], [#N.e1_err_03_t8], [#N.e1_corr_03_t8],
  [48 (full sum)], [#N.e1_err_01_t48], [#N.e1_corr_01_t48], [#N.e1_err_03_t48], [#N.e1_corr_03_t48],
)

One pass gives the right shape and has about 25% relative error against the final response; the mean excess of
its error over the full sum's is #N.e1_excess_01_t1 percentage points at the 1% dose and #N.e1_excess_03_t1 at 3%
(95% intervals over problems). Four passes leave #N.e1_excess_01_t4 and #N.e1_excess_03_t4 points. The same $t$-pass
prediction also matches the response _measured after $t$ forced loops_ (error #N.e1_matched_01_t1,
#N.e1_matched_01_t2, #N.e1_matched_01_t4 for $t = 1, 2, 4$ at the 1% dose), so @resp holds along the way and not
only at the end.
]

#figure(
  placement: auto,
  image("steering_gen/figs/fig_passes.pdf", width: 68%),
  caption: [Second model: relative error of the prediction $r G (J + dots.c + J^t) L_k$ against the final measured
  response, by the number $t$ of passes through the block used in the prediction (median over #N.n_eval problems;
  the sampled values of $t$ are equally spaced on the horizontal axis).],
) <fig-passes>

In this model, at this position, a sustained push along these directions is therefore felt through about four
applications of the block, and its effect has settled by eight. This describes the forced response along the
directions tested; it is not a measurement of the eigenvalues of $J$, and no matching measurement was made on the
first model, where the one-loop lever reached #H.cheap_onestep_share of the full lever's effect.

= Is the readout special? A diagnostic on three models <three>

The two steered models have the same design: a few layers that run once, a block that is looped and receives the
output of those first layers again at every loop, and a few final layers that read the state. Huginn was trained
that way from the start and L32 rebuilt into it. That the gradient push loses its effect on the target score
within one loop in both could reflect something particular to the readout, or something the loop does to most
directions. The self-gain of @objects gives a partial test: whether sampled directions that the output reads
differ, on this one scalar, from the mean $tau$ over random directions, @tau. It does not by itself explain the
steering response.

*What is measured.* On each problem, at the state used by its protocol (approximately settled for Huginn and L32,
the last loop for Ouro), with $J$ the derivative of one loop there:

- 16 _random directions_ (uniform unit vectors); their mean self-gain estimates $tau$;
- 8 _read directions_: for each, draw independent centred normal coefficients on the ordinary tokens, subtract
  their mean over those tokens, differentiate the resulting linear combination of scores $ell_j$ with respect to
  the state, and normalise the gradient. This sampling favours directions the output reads strongly; it is not
  uniform over the directions the output reads and can miss weakly read ones;
- the _task readouts_: the normalised gradients of the scores of the task (ten for sums and digits, one for
  true/false statements).

Write $hat(tau)_i$ for the mean of the 16 random self-gains of problem $i$; it estimates $tau$ there. For a
direction $u$ of problem $i$ the table uses $alpha(u)$; its _excess_ $alpha(u) - hat(tau)_i$; the size
$norm(J^top u)$ of the sensitivity carried back through one loop; and, when that is not zero, the _self-cosine_
$alpha(u) \/ norm(J^top u)$, the cosine of the angle between $u$ and $J^top u$. In the table, $hat(tau)$ is the
median over problems of $hat(tau)_i$; a median excess is the median over problems of the within-problem median
excess (so it is not the difference of two entries of the table); self-gains, sizes and self-cosines are
medians over all sampled directions of all problems. Intervals are 95% bootstrap intervals that resample
problems.

*Sources.* Huginn and Ouro-1.4B were measured earlier in this project (`docs/freeloop.pdf`, Sections 6 and 7,
both with rules fixed in advance). Ouro-1.4B is a looped model of another design: its whole stack of 24 layers is
applied four times, the input is not fed in again, and the output is read linearly from the state after every
loop; it has no settled state, and $J$ is taken at its last loop. L32 was measured for this draft with the same
families (`docs/lread_design.md`; final design and reading rule fixed before the main run, code reviewed by
GPT-6 Astra before launch): on the #N.n_eval held-out sums of @llama, at the same settled states as the steering runs, and
on the 40 true/false prompts used for the other two models.

#figure(
  placement: auto,
  tb((1.75in, auto, auto, auto, auto, auto),
    align: (x, y) => if x == 0 { left } else { center },
    hd([], ..D.rs_cols.map(c => c.name)),
    [Problems], ..D.rs_cols.map(c => c.n),
    [Random: $hat(tau)$], ..D.rs_cols.map(c => c.tau),
    [#h(1em) 95% interval], ..D.rs_cols.map(c => c.tau_ci),
    [Read: median self-gain], ..D.rs_cols.map(c => c.c3_alpha),
    [Read: median excess $alpha - hat(tau)_i$], ..D.rs_cols.map(c => c.c3_excess),
    [#h(1em) 95% interval], ..D.rs_cols.map(c => c.c3_ci),
    [Read below all 16 random], ..D.rs_cols.map(c => c.c3_below),
    [Task: median self-gain], ..D.rs_cols.map(c => c.task_alpha),
    [Task: median excess $alpha - hat(tau)_i$], ..D.rs_cols.map(c => c.task_excess),
    table.hline(stroke: 0.3pt),
    [$norm(J^top u)$: random / read / task], ..D.rs_cols.map(c => [#c.probe_norm / #c.c3_norm / #c.task_norm]),
    [Self-cosine: random / read / task], ..D.rs_cols.map(c => [#c.probe_cos / #c.c3_cos / #c.task_cos]),
    table.hline(stroke: 0.3pt),
    [Deficit of read directions (summary)], ..D.rs_cols.map(c => c.reading),
  ),
  caption: [Self-gain of random directions ("random"), of sampled read directions ("read") and of the task
  readouts ("task"), in three looped models. The Huginn and L32 columns are at a settled state of a looped block read through further layers; the
  Ouro column is at the last loop of a model that does not settle and reads its state linearly. "Below all 16"
  is the share of read directions whose self-gain is below that of every random direction of the same problem.
  The last row is a qualitative summary; each run had its own rule fixed in advance, and in each a deficit needs
  the interval of the excess below zero. "Not detected" is not a finding of equality; "positive excess" (interval
  above zero) was a registered label for L32 and is applied to Ouro descriptively. The L32 columns are printed
  with four decimals because their values are near zero.],
) <tab-three>

*Three different pictures* (@tab-three).

- _Huginn._ A random direction keeps about a quarter of itself after one loop ($hat(tau) approx 0.24$ to $0.26$),
  a sampled read direction almost nothing: the read directions have lower self-gain than every random direction
  of their problem. This is consistent with the small one-loop response to the gradient push in @huginn but does
  not establish its mechanism; the steering experiment uses gradients averaged over training problems, this
  diagnostic each problem's own.
- _L32._ The mean self-gain of random directions is #RS.tau_A #RS.tau_A_ci on the sums: about a thousandth,
  against a quarter in Huginn. All #RS.n_probe_values individual values lie between #RS.probes_min_A and
  #RS.probes_max_A. The excess of the read directions is #RS.c3_A #RS.c3_A_ci, so no deficit is detected. Exact
  equality was not tested, but the interval rules out a deficit larger than #RS.c3_A_lo, a hundred times smaller
  than Huginn's. The task gradients' self-gains are also small in size (individual values between #RS.task_min_A and
  #RS.task_max_A), with a positive median excess, #RS.task_A #RS.task_A_ci. Under the rule fixed in advance: no
  deficit detected, with $hat(tau)$ near zero. On the true/false prompts the read directions have a small
  _positive_ excess, #RS.c3_B #RS.c3_B_ci. So on L32 the sampled random directions, like the readout's, have only
  a small component along themselves after one loop, and the small one-loop response to the gradient push
  (@tab-main) is not, on this diagnostic, something that sets the readout apart. This does not measure the length
  of what a push becomes, and it does not show that all directions have a small self-gain.
- _Ouro-1.4B._ The sampled read directions keep slightly more than random ones (a positive excess).

A mean self-gain near zero does not mean that every direction has a small self-gain: $tau$ is the mean of the
eigenvalues of $J$, and eigenvalues of opposite sign can cancel. The statements above are about uniformly random
directions and the sampled read directions. Three further families of directions, whose reporting was fixed in
advance, agree with this on L32: differences of the scores of two random ordinary tokens (median excess
#RS.C1_A on the sums, #RS.C1_B on the true/false prompts), of two of the 64 highest-scoring tokens (#RS.C2_A,
#RS.C2_B), and the read directions with the signs of their coordinates flipped at random (#RS.C3flip_A,
#RS.C3flip_B). All are within about a hundredth of zero.

#explore[
*What one loop of L32 does to the length of a push* (from saved outputs of the steering runs and of this run).
A self-gain near zero does not mean that one loop lets nothing through.

- _Random directions shrink and turn._ For a uniformly random unit vector $u$, the mean of $norm(J u)^2$ equals
  the mean of $norm(J^top u)^2$ (both are $"tr"(J^top J) \/ d$). The 16 random directions therefore also give the
  root-mean-square length of a random unit push after one loop: #RS.rms_A (between #RS.rms_A_min and #RS.rms_A_max
  over the #N.n_eval problems). What is left is almost orthogonal to what went in: the component along $u$ is $alpha(u)$, within 0.02 of
  zero for every random direction, against a length of about 0.2.
- _Levers grow._ For the ten unit levers, $norm(J L_k)$ has median #N.jl_median over problems and candidates
  (quartiles #N.jl_q25 and #N.jl_q75, range #N.jl_min to #N.jl_max): about six times the root-mean-square value.
- _After 48 forced loops_ at the main dose, the state has moved by #N.disp_L_03 times the length of the push for
  levers, against #N.disp_G_03 for gradient pushes, #N.disp_C_03 for centroid pushes and #N.disp_R_03 for the two
  random pushes of the design (medians). The random pushes change a score by a median #N.random_abs_03 in absolute
  value.

So on L32 a random push is shrunk to about a fifth in one loop and turned away from itself\; sustained gradient,
centroid and random pushes all displace the state by about a fifth of their length\; and the levers are
directions that one loop lengthens. In Huginn, by the same identity and the sizes of $J^top u$ in @tab-three, a
random direction keeps about two thirds of its length, and a quarter of itself, and it is the read directions
that are turned away. These are descriptions of the two derivative matrices on the directions sampled; they do not
identify a mechanism.
]

#caveat[
The three columns are three protocols. The run on L32 passed its checks: re-implemented loop against the
model's own computation to #RS.g1_max\; the two ways of computing $chevron.l u, J r chevron.r$ agree to
#RS.adjoint_max (largest family median)\; on the sums, the states and task gradients are those of the steering
runs to #RS.state_identity and #RS.identity, and all #RS.correct_A are answered correctly. One threshold of the
design, on how much the state still moves at loop 64, was corrected after a test run on three problems and
before the main run: the median relative movement per loop had to be below $10^(-6)$ and now has to be below
$10^(-5)$, because the saved steering runs showed that those same states move by about $2 times 10^(-6)$
(`docs/lread_design.md`, Section 8). The measured median is #RS.drift_A. On the true/false prompts L32 prefers the labelled answer on
#RS.correct_B, and #RS.dropped_B prompt whose state still moves by more than $10^(-3)$ per loop does not change the
label when dropped. Further differences between the columns: Huginn and Ouro were computed with about 7 significant digits, L32
with 16\; the state is taken after 96 loops for Huginn's digits and after 64 for its true/false prompts and for
L32, and at the fourth and last loop for Ouro, where the state still changes by 17% per loop\; $d$ is 5280 for
Huginn and 2048 for the other two, and the spread of random self-gains shrinks like $d^(-1\/2)$\; on the
true/false prompts Ouro answers all 40 correctly, L32 prefers the labelled answer on #RS.correct_B, and Huginn
is biased towards "False" (55% correct without a push). Why the three models differ is not addressed: design,
training and the kind of state at which $J$ is taken all differ.
]

= What holds in both models, and what was not shown <both>

*Holds in both, each under its own protocol.* Both models have the same design (@three), so this is agreement
within one design. The small one-loop target response of the third item coexists with different patterns of
self-gain in the two models (@three); the mechanism of neither steering experiment is identified.

+ @resp predicts the response to a small sustained push along the levers, problem by problem, far better than
  the readout's own derivative does.
+ The tested gradient push has little mean effect on its target score; the lever has a clear one; the two are
  nearly orthogonal; and the lever has almost no immediate effect. By Proposition 3 these are, to first order,
  one fact. On the second model the three ratios of @three-readings, formed from means over problems and
  candidates at the main dose, are #N.ratio_eff_03 (measured effect of the gradient push over the lever's),
  #N.cos_L_G (mean of $chevron.l L_k, U_k chevron.r$) and #N.ratio_c0_03 (immediate derivative effect of the lever
  over the gradient push's).
+ The response of the target score at once and after one loop differ strongly for both kinds of push. (One-loop
  values: derivatives on the first model, measurements on the second.) For the gradient push this is the
  statement that its self-gain is near zero (@three).

*Not shown.*

- That this is how looped models behave in general, or that the loop singles out the readout. The diagnostic of
  @three gives three different pictures on three models; on the second model's sums no deficit of the read
  directions was detected, while small positive excesses occur for the task gradients and, on the true/false
  prompts, for the read directions.
- Control of the answer. On the second model the pushed candidate never became the answer: 0 of #N.flip_pairs
  problem–candidate pairs for the lever at each of the three doses, and at the largest dose #N.flip_other_09 of
  #N.flip_pairs answers changed to something other than the pushed candidate (gradient and centroid pushes: no
  change at any dose). A
  lever effect of #N.eff_L_03 is small against the lead of the correct candidate over the next one, which has
  median #N.margin_median (#N.margin_min to #N.margin_max) on these problems. Nor did the pushed candidate lead
  among the ten at any of the saved earlier loop counts (#N.saved_times). On the first model, an earlier
  experiment with a larger sustained push of another kind (not a lever) changed 8 of 50 answers to the intended
  digit and 12 of 50 to another digit (`WRITEUP.md`, Section 3.1).
- That the mechanism is the same. In the first model the first loop _reverses_ the sign of the lever's effect; in
  the second the immediate effect is near zero.
- That one model is easier to steer. The effect per unit of push length is #N.per_unit_norm here and
  #H.per_unit_norm in the first experiment, but state sizes, tasks and readouts differ.
- Control at arbitrary doses, at other positions of the text, or over what the model writes after the pushed
  position. These are local predictions.
- That steering changes what the model "believes". In the first model, when a separate reader of the state chose
  the sign of the push, the steered answers reproduced that reader's verdict exactly and added no accuracy
  (`WRITEUP.md`, Section 5.3).

*No consequence for detecting errors is claimed.* One might ask whether a reader of the state built on levers
would predict errors better than the readers of @monitoring. Nothing here supports that. A lever is a
sensitivity (it says which added vector moves a score); that does not by itself make it a direction along which
the states of right and wrong answers differ. A reader $x |-> chevron.l w, x chevron.r + b$ with $w$ in the span of
the levers belongs to the class of functions used in @monitoring, where $w$ ranged over all of $RR^d$ and was
fitted with a penalty on its size, the strength of the penalty being chosen on problems held back from the fit;
those fitted readers scored below a baseline fitted to four summary numbers of the model's output probabilities. Restricting $w$ to the lever span would change what the fit prefers and could improve or worsen its
predictions on new problems. It was not tested.

= Does looping help detect errors? No advantage resolved for one pair <monitoring>

The second question of the project was whether a looped model is easier to supervise than an ordinary one, in
the specific sense that a simple reader of its state can tell when its answer will be wrong. A search of the
public model repositories established one fully verified matched pair that can be downloaded: L32 and a model D
trained on the same data from the same starting weights without a loop. It found the looped members of two more
pairs but not their ordinary partners, and one further pair whose matching is unverified (`docs/pairs_check.md`).

*The registered comparison* (`docs/monitor_results.pdf`; 6,000 problems, four task families, analysis fixed in
advance). A _reader_ is a function $x |-> chevron.l w, x chevron.r + b$ of the state, with $w$ and $b$ fitted on
about 1,200 problems whose outcome is known. Its score is the probability that it ranks a randomly chosen wrong
answer above a randomly chosen right one, ties counting one half (0.5 is chance, 1 perfect), averaged over the four
families with equal weight. Reading each model's own state, the score was
#pt(mon("table", "errors of L32, read from L32's state", "family_mean")) for L32 and
#pt(mon("table", "errors of D, read from D's state", "family_mean")) for D, difference
#mon("table", "diagonal contrast", "family_mean") (medians over three fitted splits of the data; the bracket runs from the lowest lower end to the highest upper
end of their three 95% intervals; the difference is formed within each split). No overall advantage was resolved. In both
models the reader scored lower than, and added only a few thousandths to, a _confidence baseline_: a reader
fitted in the same way to four numbers computed from the output probabilities $p_j$ (the logarithm of the
probability of the answer, its gap to the second candidate, the entropy $-sum_j p_j log p_j$, and the logarithm
of the total probability on valid answers). That baseline scores #pt(mon("added", "fixed/L32/confidence", "family_mean"))
for L32 and #pt(mon("added", "fixed/D/confidence", "family_mean")) for D; adding the state to it gains
#mon("added", "fixed/L32/confidence/added", "family_mean") and #mon("added", "fixed/D/confidence/added", "family_mean").
One exploratory result pointed the other way: at the second-to-last position of the text, a reader of L32's
state predicted L32's errors better than a reader of D's state predicted D's, by 0.02 to 0.04 over the three
splits; the two models make different errors there and this was not tested on fresh problems
(`docs/monitor_results.pdf`, Section 7.4).
This is a statement about such readers at the last position on one pair
of trained models; the intervals omit the uncertainty of fitting and of training, and the two models also differ
in how their layers are arranged, so it does not isolate looping as a cause.

*A search for tasks where the loop acts late.* On those problems the looped model's answer changes between loops 8
and 32 on only #N.base_change of problems (#N.base_lo to #N.base_hi by family), with accuracy #N.base_acc8 and
#N.base_acc32. This motivated a search for tasks with more late changes of the answer; a stable answer alone does
not establish that the evolving state holds no further information about errors. The screen, with thresholds
fixed in advance (`docs/lateshift_screen.md`), covered #N.n_fam families with single-token answers (longer
arithmetic, chains of assignments, following pointers, swapping objects, logical expressions, word problems,
science questions), #N.n_per problems each, on both models. At screening, a family qualified with answer changes
between loops 8 and 32 on at least 15% of problems and was borderline from 10%; either could advance. Both models
had to be between chance plus 10 percentage points and 90% correct, and changes between loops 31 and 32 could be
at most a third as frequent as between 8 and 32. On 300 fresh problems, acceptance required at least 10% changes
with the lower end of a 95% interval (Wilson's interval for a proportion) above 7%, and the same conditions on accuracy and on adjacent loops.

#figure(
  placement: auto,
  image("steering_gen/figs/fig_screen.pdf", width: 78%),
  caption: [The screen: one point per task family (#N.n_per problems each). The largest rates of late change occur
  in families with low accuracy. The shaded rectangle is an illustration of the first-screen thresholds for L32
  when guessing is negligible; it is not the full rule, which also involves D's accuracy, chance levels and the
  borderline band. `mul2add` ($a times b + c times d$) and `story2` (a two-step word problem) were re-collected.
  The other labelled families: `add3`, sums of three-digit numbers\; `paren`, $(a + b) times c - d$\; `var4`, four
  lines assigning values to letters, each from the one before, then a question about the last\; `var4s`, the same
  with the lines shuffled\; `ptr3`, eight lines $f(u) = v$ and the question $f(f(f(u_0)))$.],
) <fig-screen>

*Result: no family passed every criterion.* Two candidates were re-collected on 300 fresh problems:

#tb((auto, auto, auto, auto, auto),
  hd([Family], [Changed 8→32], [L32 correct at 32], [D correct], [Outcome]),
  ..D.confirm_rows.map(r => (
    raw(r.fam), r.change, r.acc32, r.accD,
    if r.fam == "mul2add" { [not accepted: D below 10% (#r.kD of #r.n)] } else { [not accepted: changes below 10%] },
  )).flatten(),
)

`mul2add` ($a times b + c times d$) failed narrowly, on the accuracy of D only, and its late changes are real. But
both models get about nine in ten of these problems wrong, and #D.confirm_rows.at(0).ww of its
#D.confirm_rows.at(0).kch changes go from one wrong answer to another. Across the screen, #N.changes_ww of the
#N.changes_total answer changes between loops 8 and 32 are wrong-to-wrong, #N.changes_wr wrong-to-right and
#N.changes_rw right-to-wrong; no family shows a resolved gain in accuracy from 8 to 32 loops.

#caveat[
This is a decision to stop under a fixed rule and a small budget, for one pair of models. With #N.n_per problems a
family whose true rate of late changes is 10% is missed more than half the time. It does not show that the state
is idle after loop 8 (it keeps moving, see `docs/followup_results.md`), that no suitable task exists, or anything
about looped models in general. The registered comparison concerns honest mistakes on short problems, not a model
that conceals something.
]

= Earlier work <related>

A search for earlier work was made on 2026-11-11, after all the results above. The three memos, with every query
and source, are in `results/novelty-20261111/`. It was a targeted search of about 160 queries and not a systematic
review, so "not found" below is not proof of absence. The description of Wang and Reid was checked against the
text of the paper; the other descriptions follow the memos, and for them only the abstracts were checked here.

*The mathematics.* Bai, Kolter and Koltun (2019, arXiv 1909.01377) differentiate the output of a model defined by
a fixed point with respect to its input and its weights, through $(I - J)^(-1)$. Geng, Zhang, Bai, Wang and Lin
(2021, arXiv 2111.05177) study the finite sum over the loops and its limit as cheaper substitutes for that
derivative when such a model is trained. Propositions 1 and 2 are these derivatives for the particular input
"a vector added to the state before every loop". Proposition 3 is bookkeeping on two unit vectors.

*Pushes built from a Jacobian, in two of these models.* Wang and Reid (2026, arXiv 2609.01924) work with
Huginn-0125 and a larger Ouro model. They take the readout direction $u_t$ of a token $t$, pull it back through a
matrix $J_v$ to $d = J_v^top u_t \/ norm(J_v^top u_t)$, and set or remove the component of the state along $d$.
Their $J_v$ is a derivative between two depths of the model averaged over a collection of texts and positions,
not the derivative of one loop at a settled state. In Ouro an intervention held over all remaining loops succeeded
on 38% of their cases against 21% to 29% for one loop, the later loops partly restoring the original answer; in
Huginn only the interventions close to the end mattered. So intervening along a direction pulled back through a
Jacobian in a looped language model, and later loops undoing an early intervention, are not new here. That
paper contains no sum of powers of $J$, no prediction of the size of an effect, and no comparison of a pulled-back
direction with the readout gradient. Its method comes from Gurnee and others (2026, arXiv 2607.15495), who in
ordinary models compare how later layers respond to readout-related directions and to random ones.

*The same idea without a loop.* In ordinary (not looped) models, Wang, Zhao, Cao, Yao and Shu (2026, arXiv
2605.17231) choose the direction of a push by pulling a direction of the output back through the Jacobian of the
layers in between, and Olivieri and Pérez Rodríguez (2026, arXiv 2605.25225) predict the effect of small
insertions from first-order sensitivities. That a direction which predicts an answer need not be a good direction
to push is also known: Marks and Tegmark (2023, arXiv 2310.06824) for ordinary models, and Huang and Qi (2026,
arXiv 2609.21662) for models that reason in vectors, where a push has weak effect although the information can be
read.

*Self-gain.* The number $alpha(u) = chevron.l u, J u chevron.r$ is the Rayleigh quotient of $J$ at $u$, and
estimating $"tr"(J)$ from its values on random $u$ is Hutchinson's estimator (1989); neither is new. Bai, Koltun
and Kolter (2021, arXiv 2106.14342) apply the estimator, in the squared form $"tr"(J J^top)$, to the Jacobian of
a fixed-point model. That one loop can lengthen a vector at a stable settled state was reported for small looped
models by Buehlmaier (2026, arXiv 2607.10681). Differences between Huginn and Ouro in how content passes from loop
to loop are a main result of Wang and Reid.

*What the search did not find.* Three measurements: the problem-by-problem prediction of a sustained push of
finite size from $J + dots.c + J^T$ in a trained looped language model (@huginn, @llama); the lever being nearly
orthogonal to the readout gradient, with a gradient push losing its effect (@both); and the signed comparison of
$alpha$ on read directions with $tau$ on three models (@three). These are measurements on particular models. No
method in this document is new.

= Limits

In decreasing order of importance for the steering results:

+ *Two models of one design, one position, one addition task family each in the central comparison.* Both tasks
  have ten candidate answers. The comparison across three models in @three uses a cheaper diagnostic, not the
  steering experiment, and its protocols differ between models.
+ *Scores, not answers.* The effects are shifts of centred scores on problems already answered correctly; the
  tested doses did not make a different, pushed candidate become the second model's answer. #N.flip_other_09
  lever pushes at the largest dose, on three problems, changed the answer to something else. Answers are those
  read at the end of the forced run\; at the saved earlier loop counts the pushed candidate never led among the
  ten candidates either.
+ *Local.* Everything is a first-order statement around a settled state. L32's doses are 1% to 9% of the length of
  its state, and the prediction error grows from #N.err_FULL_01 to #N.err_FULL_09 over that range; Huginn's digit
  dose is length 3. The separately quoted true/false experiment used length 12.5, about 16% of that state, with
  the linear response validated at smaller doses.
+ *Predictions are tested along the levers.* The problems are held out, but the directions are built from the
  same family of operators $A_T$ that makes the prediction; this is not a test on arbitrary directions. In an
  earlier pilot with a different kind of push on L32 (a vector added to the output of one layer;
  `docs/oversight_results.pdf`, Part C), the straight-line prediction for random directions had a median relative
  error of 0.14 for pushes that changed the output distribution slightly and 0.42 for moderate ones.
+ *Directions fitted on few problems.* The levers are averages over 20 (first model) and 40 (second model)
  training problems, and the intervals do not include the uncertainty of that fit.
+ *Equal length, not equal inner product with the lever.* Pushes are compared at the same length. In the first
  model, random directions with the same inner product with the full lever reproduced most of the cheaper levers' effects (intervals given two
  random draws per candidate); that control was not repeated on the second model.
+ *The settled state of the second model is taken after 64 loops,* twice what it was trained around; its answers
  at 32 and 64 loops agree on all problems. The prediction uses $J$ at that state with the earlier positions held
  fixed. A push at the last position cannot change the earlier positions, so what this neglects is only their
  own remaining movement after loop 64, which the small movement of the unpushed state indicates but does not
  bound.
+ *Parts computed after the main results* are in blue boxes: the immediate effects $c_0$ on the second model, the
  comparison of whole tables over time, and the follow-up on repetition. The counts of changed answers, the leads of the correct
  answer, the three ratios of Proposition 3 and the lengths of pushes after one loop were also computed
  afterwards, from the saved outputs.

= How the numbers were produced <provenance>

*Registration and review.* First model: `huginn_addition/WRITEUP.md`. Second model: `docs/llever_design.md`; the
main design and decision rule were fixed before the runs, a later draft changed only how the evaluation was split
across machines, and its Section 8 is a follow-up specified after the main results. Diagnostic of @three on L32:
`docs/lread_design.md`. GPT-6 Astra reviewed designs and code before the runs and recomputed the results
afterwards (`results/llever_prerun_review.md`, `results/followup_results_review.md`,
`results/steering_results_review.md`, `results/lread_prerun_review.md`, `results/steering_results_review3.md`).
Gemini 4 Carbon read Drafts 2 and 3 independently (`results/steering_results_review2.md`,
`results/steering_results_review4.md`); Draft 3 added the diagnostic on three models, the statement of what is
standard calculus and the counts of changed answers, and Draft 4 added Proposition 3, corrected the description
of the confidence baseline and of the role of the earlier positions, and added the lengths of pushes after one
loop. The required changes of each review are applied. Draft 5 added @related after the search for earlier work
(carried out by three GPT-6.1 Sol agents); Drafts 4 and 5 have not been reviewed.

The empirical summaries are read from saved outputs; constants of the protocols and the four quotations listed
below are written in the document. Second model and screen: `huginn_addition/steering_report.py` reads the outputs
of the runs `llever_train-20261110-171953`, `llever_pred-20261110-172208`, `llever_pred-20261110-174655`,
`llever_eval-20261110-172520`, `lateshift-20261110-171533` and `lateshift-20261110-172550` under `results/` and
writes `docs/steering_gen/numbers.json` and the figures. The table of @three is built by the same script from
the saved summaries of the runs `ctrlprobe-20261107-144949`, `ctrlprobe_digits-20261107-144951`,
`ouro-20261107-150835` and `lread-run-20261110-221504`. First model: the values in the tables and in @huginn come
from `docs/steering_gen/huginn_facts.json`, which records for each the result file and field it was read from; the
counts 189 of 200 and 154 of 157, the ratio 1.17 and the value 0.93 are quoted from `huginn_addition/WRITEUP.md`
(Sections 3.4, 3.5 and 4.1). Monitoring: `docs/monitor_results_gen/tables.json`.

Computing time for the runs of 2026-11-10 (recorded running times on one A100 graphics processor, summed over
machines; not a billing record): screen #N.gpu_screen_all minutes including its test run and the confirmation;
steering replication #N.gpu_lever_main minutes including its test run, and #N.gpu_lever_e1 minutes for the follow-up
on repetition. The diagnostic of @three on L32: run `lread-run-20261110-221504`, #RS.gpu_min minutes, and
#RS.gpu_min_all minutes including its two test runs.
