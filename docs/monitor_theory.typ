#set document(title: "How far can the monitor's reading still move? Step-size forecasts for looped models")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#show table: set par(justify: false)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => [#v(0.6em) #it #v(0.2em)]
#set math.equation(numbering: "(1)", supplement: none)
#show math.equation.where(block: true): it => {
  if it.has("label") or it.numbering == none { it } else {
    counter(math.equation).update(v => v - 1)
    math.equation(block: true, numbering: none, it.body)
  }
}

#let boxed(title, color, body) = block(
  fill: color, inset: 9pt, radius: 3pt, width: 100%, breakable: true,
  [*#title.* #body],
)
#let thm(title, body) = boxed(title, rgb("#eef3fb"), body)
#let plain(body) = boxed("In words", rgb("#fbf3e6"), body)
#let caveat(body) = boxed("Careful", rgb("#fdeeee"), body)
#let proof(body) = [_Proof._ #body #h(1fr) $square$]
#let flag = math.op("flag")
#let wrong = math.op("wrong")
#let Err = math.op("Err")
#let AUC = math.op("AUROC")
#let ip(a, b) = $lr(chevron.l #a, #b chevron.r)$

#align(center)[
  #text(16pt, weight: "bold")[How far can the monitor's reading still move?]
  #v(0.2em)
  #text(10.5pt)[Step-size forecasts for looped models: what is guaranteed, why the obvious rule is biased, and what
  token order does]
  #v(0.2em)
  #text(9.5pt)[Draft 1, 2026-11-09. Written after the 96-problem screen and before any main-collection data.
  Companion to `docs/oversight_monitor.md` (experiment design). Every statement is checked numerically by
  `huginn_addition/monitor_theory_check.py`.]
]

#v(0.4em)

#plain[
A looped model applies one map $F$ to its internal state again and again, and can be stopped after any number of
loops. A _monitor_ reads a number off the state and raises a flag when that number is above a threshold. We would
like the flag not to depend on when we happened to stop. This note asks: after $r$ loops, how far can the state, and
therefore the monitor's reading, still move in later loops, judged only from the sizes of the steps taken so far?

- *Theorem 1 (classical).* If each loop shrinks distances by a factor $q < 1$, the remaining movement is at most the
  last step times $q\/(1-q)$. The factor $q$ is not known.
- *Theorem 2 (the obvious rule has a sign error).* Estimating $q$ by the shrinkage seen so far and extrapolating
  gives, for symmetric linear maps, a _lower_ bound on the remaining movement, never an upper bound. Fast-decaying
  components die first, so the observed shrinkage keeps getting worse. The screen showed exactly this.
- *Theorem 3 (two-step bound).* Comparing states two loops apart gives a bound that is never worse in the pure
  cases and up to $(1+q)\/(1-q)$ times smaller when consecutive steps reverse direction, as they do late in the run
  of the 32-loop model. For symmetric linear maps the true movement is sandwiched between the two-step
  extrapolation and the two-step bound.
- *Theorem 4 (token order).* Position $p$ reads only positions up to $p$. So the loop's derivative is block
  triangular, the first $p$ positions together settle at a rate that can only get slower as $p$ grows, and the
  derivative is not a normal matrix as soon as some position actually depends on an earlier one. A single position inherits the slow rate of an earlier one only if it is
  actually coupled to it; then the last position, where the monitor reads, is as slow as the whole state.
- *Theorems 5 and 6 (what can be claimed from data).* A multiplier calibrated on held-out problems turns any
  forecast into a bound with 95% coverage; the forecast's quality then shows only in how small the bounds are. A
  monitor's error at another budget is at most its error at the fitted budget plus three measurable terms.

The mathematics is elementary. Its use is to say which measured quantities matter, and which way the errors go.
]

= Setting and notation

A prompt has $L$ tokens. The _state_ is a list $X = (x_1, dots, x_L)$ with one vector $x_p in RR^d$ per token
position ($d = 2048$ here). We treat $X$ as one vector in $RR^(L d)$ with the Euclidean norm,
$norm(X)^2 = sum_p norm(x_p)^2$. For a fixed prompt the model defines a map $F : RR^(L d) -> RR^(L d)$, one _loop_.
Starting from a random $X_0$,
$ X_t = F(X_(t-1)), quad t = 1, 2, dots $
Stopping after $r$ loops (_budget_ $r$) means reading the answer off $X_r$. We write

$ delta_t = X_t - X_(t-1), quad d_t = norm(delta_t), quad e_t = norm(X_t - X_(t-2)) = norm(delta_t + delta_(t-1)). $

So $d_t$ is the size of step $t$ and $e_t$ the distance covered by two consecutive steps. If the two steps point the
same way, $e_t = d_t + d_(t-1)$; if they point in opposite directions, $e_t = abs(d_t - d_(t-1))$.

A _monitor_ is a vector $w in RR^d$, a number $b$ and a threshold $tau$. Its _score_ at budget $t$ is
$s_t = ip(w, x_L^((t))) + b$, read at the last position, and it raises a flag when $s_t > tau$. By the
Cauchy–Schwarz inequality, for any two budgets,

$ abs(s_t - s_r) = abs(ip(w, x_L^((t)) - x_L^((r)))) <= norm(w) norm(x_L^((t)) - x_L^((r))) <= norm(w) norm(X_t - X_r). $ <cs>

So a bound $beta$ on how far the state moves between budgets $r$ and $t$ bounds how far the score moves, and if
$abs(s_r - tau) > norm(w) beta$ the flag at budget $t$ equals the flag at budget $r$. Everything below is about
finding such a $beta$ at budget $r$, without running the later loops.

Two quantities must be kept apart. For $h >= 1$,
$ underbrace(norm(X_(r+h) - X_r), "displacement") <= underbrace(d_(r+1) + dots + d_(r+h), "path length"). $
The monitor cares about the displacement. A state that oscillates has a long path and a small displacement.

= The classical bound <sec-banach>

#thm("Theorem 1 (a-posteriori bound; Banach 1922)")[
Let $C subset RR^(L d)$ be closed with $F(C) subset C$, and suppose $norm(F(X) - F(Y)) <= q norm(X - Y)$ for all
$X, Y in C$, with $q < 1$. If $X_(r-1) in C$ then for every $h >= 1$:
+ $d_(r+k) <= q^k d_r$ for $k >= 0$;
+ $norm(X_(r+h) - X_r) <= d_(r+1) + dots + d_(r+h) <= d_r (q + q^2 + dots + q^h) = d_r dot q(1-q^h)/(1-q)$;
+ $X_t$ converges to a limit $X_oo$ with $F(X_oo) = X_oo$ and $norm(X_oo - X_r) <= d_r dot q\/(1-q)$.
]

#proof[
$d_(t+1) = norm(F(X_t) - F(X_(t-1))) <= q d_t$ for $t >= r$, which gives 1 by induction. The first inequality in 2
is the triangle inequality and the second is 1 summed over $k$. By 2 the sequence is Cauchy; $C$ is closed, so the
limit lies in $C$, and $F$ is continuous, so the limit is a fixed point. Let $h -> oo$ in 2 to get 3.
]

We write $B(q) = d_r (q + dots + q^h)$ for this bound. It needs the number $q$, which nobody hands us. The steps
$d_1, dots, d_r$ are observed, so the tempting move is to estimate $q$ from the ratios $d_t \/ d_(t-1)$. The
experiment's registered rule did this: take $hat(q)$ to be the largest of the last eight observed ratios and
forecast $B(hat(q))$.

= Why extrapolating the observed ratio under-predicts <sec-convex>

Near a fixed point, $F(X) approx X_oo + A (X - X_oo)$ where $A$ is the derivative of $F$ at $X_oo$. So to first
order the steps obey $delta_(t+1) = A delta_t$. We study that linear recursion exactly.

Recall the spectral theorem: a real symmetric matrix $A$ has real eigenvalues $lambda_1, dots, lambda_n$ and an
orthonormal basis of eigenvectors $u_1, dots, u_n$. If $delta = sum_i a_i u_i$ then $A^k delta = sum_i lambda_i^k
a_i u_i$ and $norm(A^k delta)^2 = sum_i lambda_i^(2k) a_i^2$. Each eigen-direction is a _mode_ that is multiplied by
its own $lambda_i$ every loop.

#thm("Theorem 2 (ratios rise; the extrapolation is a lower bound)")[
Let $A$ be real symmetric and $delta_(t+1) = A delta_t$, with $d_t = norm(delta_t) > 0$ for the indices used.
+ $d_t^2 <= d_(t-1) d_(t+1)$. Equivalently the ratios $d_(t+1) \/ d_t$ never decrease as $t$ grows.
+ Hence the largest ratio seen up to loop $r$ is the last one, $hat(q) = d_r \/ d_(r-1)$, and for every $h >= 1$
  $ d_(r+1) + dots + d_(r+h) >= d_r (hat(q) + hat(q)^2 + dots + hat(q)^h) = B(hat(q)). $
  Equality holds only if $delta_(r-1)$ is a combination of eigenvectors whose eigenvalues all have the same
  absolute value.
+ If in addition all eigenvalues are $>= 0$, the displacement obeys the same inequality:
  $norm(X_(r+h) - X_r) >= B(hat(q))$.
]

#proof[
_1._ Write $delta_(t-1) = sum_i a_i u_i$. Then $d_(t-1)^2 = sum_i a_i^2$, $d_t^2 = sum_i lambda_i^2 a_i^2$ and
$d_(t+1)^2 = sum_i lambda_i^4 a_i^2$. By Cauchy–Schwarz,
$ d_t^4 = (sum_i a_i dot lambda_i^2 a_i)^2 <= (sum_i a_i^2)(sum_i lambda_i^4 a_i^2) = d_(t-1)^2 d_(t+1)^2. $

_2._ By 1 the ratios $rho_t = d_(t+1)\/d_t$ are nondecreasing, so $d_(r+k) = d_r rho_r rho_(r+1) dots.c
rho_(r+k-1) >= d_r rho_(r-1)^k = d_r hat(q)^k$. Sum over $k$. Equality for all $k$ forces equality in
Cauchy–Schwarz, that is $lambda_i^2 a_i$ proportional to $a_i$: all $lambda_i^2$ with $a_i != 0$ coincide.

_3._ Write $delta_r = sum_i b_i u_i$, so $b_i = lambda_i a_i$ with $a_i$ the coefficients of $delta_(r-1)$, and let
$I = {i : b_i != 0}$; for $i in I$ we have $lambda_i > 0$. Put $g(lambda) = lambda + lambda^2 + dots + lambda^h$.
Then $X_(r+h) - X_r = sum_(k=1)^h A^k delta_r = sum_(i in I) g(lambda_i) b_i u_i$, so
$ norm(X_(r+h) - X_r)^2 = sum_(i in I) g(lambda_i)^2 b_i^2, quad d_r^2 = sum_(i in I) b_i^2, quad
  d_(r-1)^2 = sum_i a_i^2 >= sum_(i in I) lambda_i^(-2) b_i^2. $
Let $Y$ be the random variable that takes the value $lambda_i^(-2)$ with probability $b_i^2 \/ d_r^2$, $i in I$.
The last display says $EE[Y] <= d_(r-1)^2 \/ d_r^2 = hat(q)^(-2)$. Define $phi(y) = g(y^(-1\/2))^2 =
sum_(j,k=1)^h y^(-(j+k)\/2)$ for $y > 0$: a sum of convex decreasing functions, so convex and decreasing. By
Jensen's inequality and monotonicity,
$ norm(X_(r+h) - X_r)^2 / d_r^2 = EE[phi(Y)] >= phi(EE[Y]) >= phi(hat(q)^(-2)) = g(hat(q))^2. $
]

#plain[
With more than one decay rate present, the fast modes disappear first and the slow ones are left, so the measured
shrinkage factor creeps upward towards the slowest rate. A forecast that freezes the factor at what has been seen so
far must fall short. The error is not noise that averages out: it always has the same sign. A probabilist will
recognise part 1: the numbers $d_t^2$ are the moments $integral x^t dif mu(x)$ of the positive measure $mu$ that puts
mass $a_i^2$ at the point $lambda_i^2$, and moments of a positive measure are log-convex (Lyapunov's inequality).
]

*What the screen showed.* On 96 problems from each of seven task families, the ratio $d_t\/d_(t-1)$ of the 32-loop
model rose in 86% of consecutive steps between loops 3 and 16 (medians 0.43 at loop 4, 0.58 at 8, 0.80 at 16, 0.83
at 32). The path length to loop 32 exceeded the registered forecast $B(hat(q))$ for 100% of problems at $r = 4$
and at $r = 8$, and for 39 to 48% at $r = 16$. Theorem 2 says this is what the rule does, not a quirk of the
sample.

#caveat[
Theorem 2 needs symmetry. It extends to _normal_ matrices ($A A^top = A^top A$; these have an orthonormal basis of
complex eigenvectors) with $abs(lambda_i)$ in place of $lambda_i$ in parts 1 and 2; part 3 then fails, because
rotation makes the displacement shorter than the path. For matrices that are not normal even part 1 can fail: for
$A = mat(a, 0; 1, a)$ and $delta_1 = (1, 0)$ the second coordinate of $delta_t$ is $(t-1) a^(t-2)$, whose ratios
$a t\/(t-1)$ _decrease_ towards $a$. Section 6 shows that the loop's derivative is not normal as soon as one position influences a later one. So Theorem 2
explains the sign that was observed; it does not prove the sign must always be that one. Either way the rule
carries no guarantee.
]

= Looking two steps back <sec-two>

Late in the run of the 32-loop model, consecutive steps point in nearly opposite directions (median cosine of the
angle between them: 0.35 at loop 8, $-0.11$ at loop 16, $-0.58$ at loop 32). The state zig-zags towards its limit.
Then the path is long and the displacement short, and any bound built from $d_r$ alone pays for the zig-zag.

#thm("Theorem 3 (two-step bound)")[
Under the hypotheses of Theorem 1, for every $m >= 1$,
$ norm(X_(r+2m) - X_r) <= e_r (q^2 + q^4 + dots + q^(2m)) = e_r dot q^2 (1 - q^(2m))/(1 - q^2) =: B_2(q). $
Compared with the one-step bound $B(q)$ for the same horizon $h = 2m$,
$ B_2(q) = B(q) dot e_r/d_r dot q/(1+q). $
]

#proof[
$G = F compose F$ satisfies $norm(G(X) - G(Y)) <= q^2 norm(X - Y)$ on $C$. The sequence $Y_j = X_(r - 2 + 2j)$
obeys $Y_(j+1) = G(Y_j)$, and its step from $Y_0$ to $Y_1$ is $X_r - X_(r-2)$, of size $e_r$. Apply Theorem 1,
part 2, to $G$ with $q^2$ in place of $q$. The comparison is the identity
$q^2 (1 - q^(2m))\/(1 - q^2) = q(1 - q^(2m))\/(1-q) dot q\/(1+q)$.
]

Two pure cases show what is gained. Suppose every step is exactly $q$ times the previous one in size.
- _Aligned steps_, $delta_r = q delta_(r-1)$: then $e_r = d_r (1+q)\/q$ and $B_2 = B$. Nothing is lost.
- _Reversing steps_, $delta_r = -q delta_(r-1)$: then $e_r = d_r (1-q)\/q$ and $B_2 = B dot (1-q)\/(1+q)$. At
  $q = 0.83$ this is a factor 10.8. Here $B_2$ equals the true displacement, so it cannot be improved.
In general $e_r <= d_r + d_(r-1)$, so $B_2 <= B dot (1 + d_(r-1)\/d_r) dot q\/(1+q)$: the two-step bound can
exceed the one-step bound only when the last step shrank by more than the factor $q$.

For linear maps the two-step version also repairs the failure of Theorem 2, part 3: no assumption on the signs of
the eigenvalues is needed.

#thm("Corollary (the displacement is sandwiched)")[
Let $F(X) = A X + c$ with $A$ real symmetric and all $abs(lambda_i) <= q < 1$. Let $hat(Q) = e_r \/ e_(r-2)$
(with $e_r, e_(r-2) > 0$). Then for every $m >= 1$
$ e_r (hat(Q) + hat(Q)^2 + dots + hat(Q)^m) <= norm(X_(r+2m) - X_r) <= e_r (q^2 + q^4 + dots + q^(2m)). $
]

#proof[
The two-step differences $eta_j = X_(r-4+2j) - X_(r-6+2j)$ satisfy $eta_(j+1) = A^2 eta_j$, and $A^2$ is symmetric
with eigenvalues $lambda_i^2 >= 0$. The left inequality is Theorem 2, part 3, applied to the recursion
$eta_(j+1) = A^2 eta_j$, whose last observed ratio is $norm(eta_2)\/norm(eta_1) = e_r\/e_(r-2)$. The right
inequality is Theorem 3.
]

#plain[
The truth lies between "extrapolate the shrinkage seen so far" (too small) and "use the worst shrinkage the map
can produce" (safe, but unknown). The gap between the two is set by how spread out the decay rates are. No formula
closes this gap from one run. What can close it is data from other problems of the same kind: a multiplier
calibrated on them (Section 7).
]

= A limit of period two <sec-period>

The two-step view also covers a different behaviour: a state that never settles but ends up alternating between
two states.

#thm("Proposition (two-cycles)")[
Let $G = F compose F$. Suppose $C_a, C_b$ are closed sets with $F(C_a) subset C_b$ and $F(C_b) subset C_a$, and
$norm(G(X) - G(Y)) <= Q norm(X - Y)$ on each of them, $Q < 1$. If $X_r in C_a$ then:
+ $X_(r+2m) -> X_a$ and $X_(r+2m+1) -> X_b$ as $m -> oo$, where $F(X_a) = X_b$ and $F(X_b) = X_a$;
+ $norm(X_(r+2m) - X_r) <= e_r (Q + dots + Q^m)$ whenever $X_(r-2) in C_a$ (Theorem 3 with $Q$ for $q^2$);
+ the scores at even and odd distances from $r$ converge to $s_a = ip(w, (X_a)_L) + b$ and $s_b = ip(w, (X_b)_L) + b$. If
  $tau$ lies strictly between $s_a$ and $s_b$ the flag alternates for ever; if $tau$ lies strictly outside, the
  flag is eventually constant.
If $X_a != X_b$ then $d_t -> norm(X_a - X_b) > 0$ and $d_(t+1)\/d_t -> 1$.
]

#proof[
$G$ maps $C_a$ into itself and contracts, so by Theorem 1 the even iterates converge to the unique fixed point $X_a$
of $G$ in $C_a$. $F(X_a) in C_b$ and $G(F(X_a)) = F(G(X_a)) = F(X_a)$, so $X_b := F(X_a)$ is the fixed point of $G$ in
$C_b$, to which the odd iterates $X_(r+1), X_(r+3), dots in C_b$ converge. $F(X_b) = G(X_a) = X_a$. Part 3 follows
because the score is continuous in the state.
]

So for a two-cycle the one-step rule gives no forecast at all (its ratio tends to 1), while the two-step bound
works unchanged, and a monitor needs either separate thresholds for even and odd budgets or a margin
$abs(s - tau)$ larger than $abs(s_a - s_b)$. This is distinct from the zig-zag of Section 4, where the steps
reverse but shrink ($X_a = X_b$). The two are told apart by $d_t$: it tends to zero for a zig-zag and to a positive
constant for a two-cycle. In both, $e_t$ is small compared with $d_t$ once the shrinkage per step is slow
(for exactly reversing steps $e_t\/d_t = (1-q)\/q$); on the screen, $e_33 \/ d_33 < 0.5$ for 23% of problems of the
32-loop model.

= Token order: every position has its own clock <sec-tri>

So far the state was one big vector. Its internal structure matters. The model is _causal_: the new vector at
position $p$ is computed from the current vectors at positions $1, dots, p$ only. In symbols there are maps $f_p$
with
$ F(X)_p = f_p (x_1, dots, x_p), quad p = 1, dots, L. $ <causal>
Write $X^(<= p) = (x_1, dots, x_p)$ for the first $p$ positions, and let $J$ be the derivative (Jacobian matrix) of
$F$ at some state. Cut $J$ into $d times d$ blocks $J_(p p') = partial f_p \/ partial x_(p')$: block $(p, p')$ says
how position $p$ after the loop responds to a change at position $p'$ before it. The _spectral radius_ $rho(M)$ of
a square matrix is the largest absolute value of its eigenvalues; by Gelfand's formula
$norm(M^t)^(1\/t) -> rho(M)$, so $rho(M)$ is the asymptotic shrinkage factor per step of the recursion
$delta_(t+1) = M delta_t$.

#thm("Theorem 4 (triangular structure)")[
Assume @causal.
+ _Prefixes run on their own._ $X_(t+1)^(<= p)$ is a function of $X_t^(<= p)$ alone. The first $p$ positions follow
  exactly the trajectory they would follow if the prompt ended at token $p$.
+ _Spectrum._ $J_(p p') = 0$ for $p' > p$: $J$ is block lower triangular. Hence
  $det(J - lambda I) = product_p det(J_(p p) - lambda I)$: the eigenvalues of $J$ are those of the diagonal blocks,
  and $rho(J) = max_p rho(J_(p p))$.
+ _Rates._ For the recursion $delta_(t+1) = J delta_t$ put $rho_(<= p) = max_(p' <= p) rho(J_(p' p'))$, which is
  nondecreasing in $p$. For every start, $limsup_t norm(delta_t^(<= p))^(1\/t) <= rho_(<= p)$. For every start
  outside a proper linear subspace (a set of measure zero), $lim_t norm(delta_t^(<= p))^(1\/t) = rho_(<= p)$.
+ _Not normal._ If $J$ is normal then $J_(p p') = 0$ for all $p' != p$. So as soon as one position influences a
  later one, $J$ is not normal.
]

#proof[
_1_ is @causal read for positions $1, dots, p$. _2._ $f_p$ does not depend on $x_(p')$ for $p' > p$, and the
determinant of a block triangular matrix is the product of the determinants of its diagonal blocks.
_3._ Let $M$ be the top-left corner of $J$ made of the blocks with both indices $<= p$. By triangularity the
corresponding corner of $J^t$ is $M^t$, so $delta_(t+1)^(<= p) = M^t delta_1^(<= p)$, and by part 2 applied to $M$,
$rho(M) = rho_(<= p)$. Gelfand's formula gives the upper bound. For the lower bound take an eigenvalue $lambda$ of
$M$ with $abs(lambda) = rho(M)$ and a nonzero (possibly complex) vector $v$ with $v^* M = lambda v^*$. Then
$v^* delta_(t+1)^(<= p) = lambda^t v^* delta_1^(<= p)$, so $norm(delta_(t+1)^(<= p)) >= rho(M)^t abs(v^*
delta_1^(<= p)) \/ norm(v)$; the starts with $v^* delta_1^(<= p) = 0$ form a proper subspace.
_4._ Suppose $J J^top = J^top J$ and compare the traces of the bottom-right $d times d$ block on both sides. On the
left it is $sum_(p' <= L) norm(J_(L p'))_F^2$ (squared entries of the last block row). On the right it is
$sum_(p >= L) norm(J_(p L))_F^2 = norm(J_(L L))_F^2$ (last block column, which has a single nonzero block). So
$J_(L p') = 0$ for $p' < L$. Delete the last position; the remaining matrix is again normal and block triangular;
repeat.
]

#plain[
The theorem is about _prefixes_: the first $p$ positions taken together settle at the rate $rho_(<= p)$, and a
longer prefix cannot settle faster than a shorter one. Since $norm(delta_t^(<= p))^2$ is the sum of the squared
steps at positions $1, dots, p$, at least one of the first $p$ positions decays as slowly as $rho_(<= p)$. The
theorem does _not_ say that this position is $p$ itself. With no coupling at all, $J = mat(0.9, 0; 0, 0.1)$,
position 2 settles at rate 0.1, much faster than position 1. Position $p$ inherits the slow rate of an earlier
position only if two things hold: the slow component is present in the start (excitation), and it reaches position
$p$ through nonzero blocks $J_(p p')$ (coupling). Both fail only on sets of measure zero, so for a trained network
one expects the last position to be as slow as the whole state; this is an expectation to be measured, not a
consequence of the theorem.
]

A two-number example shows both effects of the coupling. Take $x_(t+1) = a x_t$ and $y_(t+1) = b y_t + c x_t$ with
$0 < b <= a < 1$: position 1 is $x$, position 2 is $y$. The steps satisfy the same recursion, and for $a != b$
$ delta y_t = b^(t-1) delta y_1 + c thin delta x_1 (a^(t-1) - b^(t-1))/(a - b). $
(i) If $c != 0$, the step at position 2 decays at the slower rate $a$ of position 1, although its own block has
the faster rate $b$. (ii) If $a = b$ and $delta y_1 = 0$ then $delta y_t = (t-1) a^(t-2) c thin delta x_1$: the step
at position 2 _grows_ for about $1\/log(1\/a)$ loops before it decays, although neither position alone ever
expands. With $a = 0.9$ it peaks at loop 10, at 3.9 times the first step of position 1.

*Consequences for the experiment.*
- _Prediction._ Measured late in the run, the decay rate of the step summed over the first $p$ positions does not
  get faster as $p$ grows (Section 9, P3); this is the theorem, up to finite-time effects. That the rate at position
  $p$ alone behaves the same way is the coupling expectation above, reported separately. The per-position step
  sizes are saved for this purpose. Two cautions: rates measured over loops 16 to 32 (or 64) are finite-time
  slopes, not limits, and a position whose step has reached the rounding floor of the arithmetic has no measurable
  rate; such positions are excluded using the saved per-position state norms.
- _The rising ratios have two possible sources:_ several rates within the last position (Theorem 2), or early
  positions settling first and passing their slower components on (this section). The per-position data separate
  them.
- _Theorem 2 does not apply to the whole state as stated,_ because of part 4. Ratios may fall as well as rise and
  step sizes may grow for a while. This is one more reason not to trust a raw extrapolation.

= Calibration: making any forecast honest <sec-conf>

The previous sections give forecasts whose errors have a known direction but an unknown size. The size can be
learned from other problems. The tool is split-conformal calibration (Papadopoulos et al. 2002; Vovk, Gammerman and
Shafer 2005). It needs one assumption: _exchangeability_, meaning the joint distribution of the problems does not
change when they are reordered. Independent draws from one population are exchangeable.

#thm("Theorem 5 (calibrated multiplier)")[
Let $(R_i, P_i)$, $i = 1, dots, n+1$, be exchangeable pairs of positive numbers: $R_i$ the movement that was
realized on problem $i$ and $P_i$ a forecast for it. Fix $alpha in (0,1)$, let $k = ceil((n+1)(1-alpha))$, and let
$kappa$ be the $k$-th smallest of the ratios $R_1\/P_1, dots, R_n\/P_n$ ($kappa = oo$ if $k > n$). Then
$ PP(R_(n+1) <= kappa P_(n+1)) >= 1 - alpha. $
If ties among the ratios have probability zero, this probability is also at most $1 - alpha + 1\/(n+1)$.
]

#proof[
Put $S_i = R_i\/P_i$; these are exchangeable. If $S_(n+1) > kappa$ then $S_(n+1)$ is strictly larger than at least
$k$ of the other values. Among $n+1$ numbers, at most $n + 1 - k$ of them are strictly larger than at least $k$ of
the others. By exchangeability each index is equally likely to be one of these, so $PP(S_(n+1) > kappa) <= (n+1-k)\/(n+1)
<= alpha$. Without ties exactly $n+1-k$ indices qualify, and $(n+1-k)\/(n+1) > alpha - 1\/(n+1)$.
]

Three remarks on what this does and does not give.

*It is an average statement.* The probability is over the draw of all $n+1$ problems. It does not say that the
bound holds for a particular problem with probability 95%, and it says nothing about problems of a different kind
from those used for calibration.

*Validity is free; tightness has to be earned.* The theorem holds for _any_ forecast, including the constant
$P_i = 1$, which ignores the dynamics. A poor forecast is paid for by a large $kappa$ and loose bounds. So the
question "do the loop dynamics tell us how far the state will still move?" becomes: at equal coverage, are the
calibrated bounds from $B(hat(q))$, from $d_r$, from $e_r$ or from the two-step extrapolation smaller than those
from the constant? The experiment measures this through the fraction of problems whose flag is certified stable,
$abs(s_r - tau) > kappa norm(w) P$.

*Why exchangeability holds in the experiment.* Within one held-out fold, $w$ and $b$ were fitted on the other
folds, $tau$ is a quantile of the fold's own scores (a function that does not depend on the order of the problems),
and the fold is split in two at random: one half calibrates $kappa$ and the other is evaluated. Problems for which a
rule gives no forecast (ratio $>= 1$, or steps at the rounding floor) are left out of both halves; whether a problem
has a forecast depends on that problem alone, so the rest stay exchangeable. They are counted as "not certified".

= From one budget to another <sec-transfer>

Now the question the experiment asks: a monitor is fitted at budget $r$; how much worse can it be at budget
$r'$? For a random problem let $wrong_t in {0,1}$ say whether the answer read at budget $t$ is wrong, and
$flag_t = bold(1){s_t > tau}$. The monitor's error at budget $t$ is $Err(t) = PP(flag_t != wrong_t)$.

#thm("Theorem 6 (transfer inequality)")[
For any two budgets $r, r'$,
$ Err(r') <= Err(r) + PP(flag_(r') != flag_r) + PP(wrong_(r') != wrong_r). $
If $beta >= 0$ is computed at budget $r$ (it may differ from problem to problem) and
$PP(abs(s_(r') - s_r) <= beta) >= 1 - alpha$, then
$ PP(flag_(r') != flag_r) <= PP(abs(s_r - tau) <= beta) + alpha. $
]

#proof[
If $flag_(r') != wrong_(r')$ then at least one of $flag_r != wrong_r$, $flag_(r') != flag_r$, $wrong_(r') != wrong_r$
holds; take probabilities. For the second claim: if $abs(s_(r') - s_r) <= beta < abs(s_r - tau)$ then $s_(r')$ and
$s_r$ are on the same side of $tau$.
]

Together with @cs and Theorem 5 (take $beta = kappa norm(w) P$):
$ Err(r') <= Err(r) + underbrace(PP(abs(s_r - tau) <= kappa norm(w) P), "scores near the threshold")
  + underbrace(alpha, "calibration") + underbrace(PP(wrong_(r') != wrong_r), "the answer itself changes"). $
Each term is measured in the experiment. The last one is outside any monitor's control: on the screen the
correctness of the 32-loop model's answer differed from its value at loop 32 for 7.0% of problems at budget 4, 3.1%
at 8 and 1.2% at 16. The inequality is symmetric in $r$ and $r'$, so $abs(Err(r') - Err(r))$ is at most the sum of
the two change probabilities.

The experiment reports ranking quality (AUROC: the probability that a randomly chosen wrong answer gets a higher
score than a randomly chosen right one, ties counting one half) and not only $Err$. The same idea works there.

#thm("Proposition (ranking version)")[
Fix the labels. Let $s_i$ and $s'_i$ be two scores for each problem $i$ with $abs(s'_i - s_i) <= beta_i$. Let $I$ be
a uniformly chosen wrong problem and $J$ an independent uniformly chosen right one. Then
$ abs(AUC(s') - AUC(s)) <= PP(abs(s_I - s_J) <= beta_I + beta_J). $
]

#proof[
AUROC is the average over pairs $(I, J)$ of a quantity in $[0,1]$ that depends only on the order of the two scores.
If $abs(s_I - s_J) > beta_I + beta_J$ the order of $s'_I, s'_J$ equals that of $s_I, s_J$, and the pair contributes
equally to both averages.
]

= What is tested in the main experiment <sec-pred>

These predictions are written down before the main data exist. The screen numbers come from 96 problems per family
that are not part of the main set.

#table(
  columns: (auto, 1fr, 1fr),
  inset: 6pt, stroke: 0.4pt,
  [], [*Prediction*], [*What would count against it*],
  [P1], [The raw one-step forecast $B(hat(q))$ is exceeded by the path length for most problems at budgets 4 and 8
        (Theorem 2). Screen: 100% at 4; 100% (32-loop model) at 8.],
        [Violation rates near or below 5% at those budgets.],
  [P2], [Where steps reverse (32-loop model, budget 16: median cosine $-0.11$), forecasts built on $e_r$ give
        smaller calibrated bounds on the whole-state displacement than those built on $d_r$; where steps are
        aligned (4-loop model: cosine 0.96 to 1.00) they give no gain (Theorem 3).],
        [No difference in the first case, or a clear gain in the second.],
  [P3], [The late decay rate of the step over the first $p$ positions does not get faster with $p$ (Theorem 4,
        finite-time version: slopes over loops 16 to 32, not limits). Expected in addition, from coupling and not
        from the theorem: the same for position $p$ alone, with positive rank correlation between $p$ and the rate
        for most problems.],
        [Single position rates that systematically speed up towards the end of the prompt refute the coupling
        expectation. The prefix statement cannot fail in the limit; at finite time small dips are possible, and
        their size is reported.],
  [P4], [$Err(r') - Err(r)$ is at most the measured near-threshold mass plus $alpha$ plus the measured answer
        change (Theorem 6). This must hold up to sampling error; the content is how loose it is.],
        [A violation beyond sampling error would mean a mistake in the code, not in the theorem.],
  [P5], [Calibrated bounds from the dynamics certify more flags as stable than the constant forecast at the same
        95% coverage, more so at budget 16 than at budget 4.],
        [Equal or smaller certified fractions: the step sizes would then carry no usable information about the
        monitor's stability beyond what a constant gives.],
)

P5 is the one with a real chance of failing. A first look at the screen data, pooled over families and therefore
only indicative: take the ratio of the realized last-position displacement to the forecast. For the 32-loop model
at budget 16 its 95th percentile is about $10^(0.31) approx 2.0$ times its median for the two-step extrapolation,
against $10^(0.67) approx 4.7$ for the constant (a perfect forecast up to one multiplier would give 1). At budget 4
the figures are 1.7 and 1.9, almost no gain.

= What is borrowed and what is new here

*Borrowed.* Theorem 1 is Banach's fixed-point theorem with its standard error estimate. Part 1 of Theorem 2 is the
log-convexity of moments. Theorem 5 is split-conformal prediction. The facts about block triangular matrices in
Theorem 4 are standard linear algebra. Theorem 6 is a union bound.

*From the closest related work.* Blayney et al. (arXiv 2604.11791) study the same family of looped models. Their
Proposition 4.1 shows that when the whole loop has reached a fixed point, every layer inside the loop repeats its own
state from one loop to the next. Their Proposition 4.2 bounds the change of an attention pattern by the change of
the state. They report that the retrofitted models' steps shrink quickly, and that for another model (Ouro) small
steps at loops 8 to 16 are followed by drift, so small steps alone do not establish convergence. They also detect
rare oscillating tokens. They do not give a forecast from step ratios, a bound on monitor scores, an analysis of
period two, or a monitor comparison (`docs/blayney_notes.md`).

*Specific to this note,* as far as two literature passes found (`docs/monitor_lit_2.md`): the sign of the
extrapolation error and the sandwich (Theorem 2, part 3, and the Corollary); the two-step bound as the remedy for
reversing steps; the use of token order to predict the ordering of rates and to rule out normality; and the
reduction of "the monitor still works at another budget" to three measured quantities. These are small results. The
note claims no more for them than that they are correct, checked numerically, and tell the experiment what to
measure.

#caveat[
All statements about rates and ratios are about the linear recursion $delta_(t+1) = A delta_t$, which describes the
model only close to a fixed point, and Theorem 2 further assumes symmetry, which Theorem 4 says the model does not
have. The guarantees that survive without these assumptions are Theorem 1 and Theorem 3 (given a true contraction
factor, which is unknown), Theorem 5 (given exchangeability) and Theorem 6 (no assumption). The experiment's claims
rest on the last two.
]
