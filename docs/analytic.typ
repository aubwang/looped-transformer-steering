#set document(title: "A first-passage decomposition of the steering lever")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => [#v(0.6em) #it #v(0.2em)]

#let boxed(title, color, body) = block(
  fill: color, inset: 9pt, radius: 3pt, width: 100%, breakable: true,
  [*#title.* #body],
)
#let thm(title, body) = boxed(title, rgb("#eef3fb"), body)
#let plain(body) = boxed("In words", rgb("#fbf3e6"), body)
#let check(body) = boxed("Numerical check", rgb("#eef8ee"), body)
#let caveat(body) = boxed("Careful", rgb("#fdeeee"), body)
#let sgn = math.op("sgn")
#let gh = $hat(g)$

#align(center)[
  #text(16pt, weight: "bold")[A first-passage decomposition of the steering lever]
  #v(0.2em)
  #text(10.5pt)[When a loop's best push points against the readout, and why it turns positive one step later]
  #v(0.2em)
  #text(9.5pt)[Companion to `huginn_addition/WRITEUP.md` §4 and `docs/derivations.pdf` D4–D6. Draft 2 (after
  GPT-6 Astra review), 2026-11-07. Verified by `docs/analytic_check.py`.]
]

#v(0.4em)

#plain[
Split the state space into the readout direction $gh$ and everything orthogonal to it. A push can reach the readout
in two ways: it can start on $gh$ and *return*, or start outside and *arrive* for the first time. The lever (the best
persistent push) is exactly a weighted sum of these two (Theorem 1). The return gain is a sum of three terms
(Theorem 2):
- the self-gain $alpha$,
- a *passage term* $E$, which is positive when $f != 0$ and the outside dynamics has numerical abscissa below 1
  (otherwise it can be negative),
- a *circulation term* $2C$, which is nonzero only if $J$ has a skew part coupling the readout to the outside.

When $sigma > 0$, the lever starts *against* the readout ($c_0 < 0$) exactly when $alpha + E + 2C < 0$. So if the
self-gain does not already overcome the passage term ($alpha + E > 0$), negative circulation is algebraically necessary
(Corollary 2b). Symmetric loops with eigenvalues below 1 and $alpha >= 0$ cannot start negative, though they can be
nearly orthogonal (Corollaries 2a, 2c).

One step later, $c_1 - alpha c_0 = sgn(sigma) E \/ D$ (Theorem 3), a term that contains no separate circulation
contribution. So in the regime $sigma > 0$, $E > 0$, $|alpha| |c_0| < E \/ D$, the lever's response is positive at
$t = 1$ whatever its starting sign.

*Bounds* (@sec-bounds). Let $Gamma = norm(a) \/ norm(g)$ be the loop gain ratio and $theta$ the lever's angle from
the readout. If $1 + Gamma cos theta > 0$ (this is $sigma > 0$), $alpha >= 0$ and $f != 0$, the lever needs
circulation $2 norm(q) >= (1 - omega(K)) Gamma sin theta \/ (1 + Gamma cos theta) - cot theta$. The underlying
Theorem 4 holds with equality in explicit cases. So strong, nearly orthogonal levers force circulation unless
$omega(K)$ is near or above 1. In symmetric loops whose top eigenvalue is bounded away from 1, nearly orthogonal
levers must be weak.

This is a *sufficient mechanism* for the measured first-step flips in Huginn. Existing per-item arrays place all 59
negative-start digit readouts in this regime, with circulation reconstructed through the identity rather than
measured (@sec-huginn). The companion note `freeloop.pdf` shows that this circulation is not specific to readouts:
random directions in the same loops have $E + 2C approx 0$ too, and for random directions $E + 2C$ is, up to
fluctuations of order $d^(-1\/2)$, a function of the loop's eigenvalues. The start sign usually follows the self-gain
$alpha$ (89.5% of digit readouts), and readouts have a self-gain below the probe estimate of the loop's mean
eigenvalue.
]

= Setting and standing assumptions

Vectors are real. $norm(dot)$ is the Euclidean norm, and for matrices the induced 2-norm. $J in RR^(d times d)$ is the
loop Jacobian at the fixed point and $g != 0$ the readout gradient, $gh = g \/ norm(g)$. Let $U in RR^(d times (d-1))$
have orthonormal columns spanning $gh^perp$, so $x = x_0 gh + U x_perp$. In the basis $(gh, U)$,
$ J = mat(alpha, b^top; ell, K), quad
  alpha = gh^top J gh, quad ell = U^top J gh, quad b = U^top J^top gh, quad K = U^top J U. $

#table(columns: (auto, 1fr), stroke: none, inset: 3pt,
  [$alpha$], [*self-gain*: how much of a push along $gh$ is still along $gh$ one step later.],
  [$ell in RR^(d-1)$], [*leak*: where a push along $gh$ lands outside. Note $norm(J gh)^2 = alpha^2 + norm(ell)^2$.],
  [$b in RR^(d-1)$], [*feed*: $b^top x_perp$ is the readout's one-step response to an outside push $U x_perp$.],
  [$K$], [*outside dynamics*: how the outside evolves while it stays outside.],
)

*Lever and response coefficients.* If $rho(J) < 1$,
$ a = sum_(t >= 1) (J^top)^t g = (I - J^top)^(-1) J^top g . $
Whenever $I - J$ is invertible we take the right-hand side as the definition. Then $a = 0$ iff $J^top g = 0$, and we assume
$a != 0$ whenever $hat(a) = a \/ norm(a)$ appears. Response coefficients are in *cosine units*,
$c_t = gh^top J^t hat(a)$, so $c_0 = cos(g, a)$. The write-up's $c_t$ use the unnormalized $g$ and a push of fixed
norm. For the *same* push direction they differ by a per-item factor. The write-up's frozen population pushes differ
from the item lever in direction as well, so their $c_t$ are not rescaled copies of these.

*Derived quantities.* Assume $I - J$ and $I - K$ are invertible. Stability does not give the second: the nilpotent
$J = mat(-1, 1; -1, 1)$ has $K = 1$. Define
$ f = (I - K^top)^(-1) b, quad psi = alpha + ell^top f, quad sigma = 1 - psi, quad D = sqrt(psi^2 + norm(f)^2), $
$ A = (J - J^top) / 2, quad q = U^top A gh, quad E = b^top f, quad C = q^top f, quad
  omega(K) = lambda_max ((K + K^top) / 2) . $
Here $omega(K)$ is the numerical abscissa: $x^top K x <= omega(K) norm(x)^2$ for all $x$. When $rho(K) < 1$,
$f = sum_(n >= 0) (K^top)^n b$ collects the readout gain of paths that start outside, stay outside, and then enter
the readout. Otherwise $f$ is just this resolvent expression. We call $f$ the *first-passage covector*, $psi$ the
*first-return gain*, and $q$ the *circulation vector*.

= The renewal identity

#thm("Lemma 1")[
(a) *Renewal.* Let $r_t = gh^top J^t gh$, $psi_1 = alpha$, and $psi_n = b^top K^(n-2) ell$ for $n >= 2$. Then
$r_0 = 1$, and for all $t >= 1$
$ r_t = sum_(n = 1)^t psi_n r_(t - n) . $
(b) *Green's function.* $gh^top (I - J)^(-1) gh = 1 \/ sigma$ and $det(I - J) = sigma det(I - K)$. In particular
$sigma != 0$. Also $psi = alpha + b^top (I - K)^(-1) ell$.

(c) *Sign of $sigma$.* If every eigenvalue of $J$ has real part $< 1$ (e.g. $rho(J) < 1$) and $omega(K) < 1$, then
$sigma > 0$. Without the second condition $sigma < 0$ is possible even for nilpotent $J$: $J = mat(-2, 1; -4, 2)$ gives
$sigma = -1$.
]

*Proof.* (a) Write $J^t gh = r_t gh + U y_t$. The block form gives $r_(t+1) = alpha r_t + b^top y_t$ and
$y_(t+1) = ell r_t + K y_t$, with $r_0 = 1$, $y_0 = 0$. So $y_t = sum_(m=0)^(t-1) K^(t-1-m) ell r_m$, and
$ r_(t+1) = alpha r_t + sum_(m=0)^(t-1) (b^top K^(t-1-m) ell) r_m = sum_(n=1)^(t+1) psi_n r_(t+1-n) . $
(b) Solve $(I - J) x = gh$ by blocks. The outside rows give $x_perp = (I - K)^(-1) ell x_0$. The readout row becomes
$(1 - alpha - b^top (I - K)^(-1) ell) x_0 = 1$, so $x_0 = 1 \/ sigma$. The determinant identity is the
Schur-complement formula. Finally, $b^top (I - K)^(-1) ell = ell^top f$ by transposing a scalar.
(c) The eigenvalues of $I - J$ have positive real part. Real ones are positive and complex ones come in conjugate pairs
with product $|mu|^2 > 0$, so $det(I - J) > 0$. Every eigenvalue $mu$ of $K$ satisfies
$"Re" mu <= omega(K) < 1$ (take a unit eigenvector $v in CC^(d-1)$: $"Re" mu = "Re" v^* K v = v^* ((K + K^top)\/2) v$).
So $det(I - K) > 0$ by the same argument, and $sigma = det(I - J) \/ det(I - K) > 0$. $square$

*Probability remark.* If $J$ were substochastic and $gh$ a coordinate vector, then:
- $r_t$ would be the probability of being back at that state at time $t$;
- $psi_n$ would be the probability of the *first* return happening at time $n$;
- (a) would be the classical renewal equation.

Here the weights are signed, so returns can cancel. Everything below quantifies that cancellation.

= The lever decomposition

#thm("Theorem 1 (lever = return + first passage)")[
$ a = norm(g) / sigma (psi gh + U f). $
Hence the part of $a$ orthogonal to $g$ is $(norm(g) \/ sigma) U f$, and with $theta = angle(g, a) in [0, pi]$,
$ cos theta = sgn(sigma) psi / D, quad quad sin theta = norm(f) / D, quad quad
  tan theta = sgn(sigma) norm(f) / psi quad (psi != 0) . $
]

*Proof.* $a = (I - J^top)^(-1) J^top g = (I - J^top)^(-1) g - g$. Let $h = (I - J^top)^(-1) gh$. In the basis
$(gh, U)$, $J^top = mat(alpha, ell^top; b, K^top)$, so $(I - J^top) h = gh$ reads
$(1 - alpha) h_0 - ell^top h_perp = 1$ and $- b h_0 + (I - K^top) h_perp = 0$. The second gives $h_perp = h_0 f$, then
the first gives $h_0 = 1 \/ sigma$. So $a = norm(g)(h - gh) = (norm(g) \/ sigma)(psi gh + U f)$. Normalizing gives the
angle formulas. $square$

The lever is nearly orthogonal to the readout exactly when the first-passage gain dominates the first-return gain:
$norm(f) >> |psi|$.

= Return gain: self-gain, passage, circulation

#thm("Theorem 2")[
$ ell = b + 2 q, quad quad psi = alpha + E + 2 C, quad quad E = f^top (I - K) f >= (1 - omega(K)) norm(f)^2 . $
]

*Proof.* $ell - b = U^top (J - J^top) gh = 2 q$, so $psi = alpha + (b + 2 q)^top f = alpha + E + 2 C$. Since
$b = (I - K^top) f$, $E = b^top f = f^top (I - K) f$, and $f^top K f <= omega(K) norm(f)^2$. $square$

#caveat[This is an *algebraic attribution*, not a split into independent causes. For fixed $f$, $E$ sees only the
symmetric part of $K$. But $f$ itself depends on $b$ and on all of $K$, skew part included. So $E$ is not "the return
gain of the symmetrized loop", and $|C|$ also depends on the rest of the dynamics through $f$. What *is* clean is that
$q = 0$, hence $C = 0$, whenever the readout's coupling to the outside is symmetric ($ell = b$).]

#thm("Corollary 2a (symmetric loops)")[
Let $J = J^top$ with eigenvalues in $[lambda_-, lambda_+]$, $lambda_+ < 1$, and $a != 0$. Then $q = 0$, $sigma > 0$,
$psi >= alpha + (1 - lambda_+) norm(f)^2$, and $norm(f) >= norm(b) \/ (1 - lambda_-)$. If moreover $alpha >= 0$:
- $c_0 > 0$;
- if $f != 0$, $tan theta <= 1 \/ ((1 - lambda_+) norm(f))$;
- if $f = 0$, $theta = 0$.
]

*Proof.* $A = 0$ gives $q = 0$. $I - J$ is symmetric positive definite, hence so is its inverse, so
$1\/sigma = gh^top (I - J)^(-1) gh > 0$. Also $f^top K f = (U f)^top J (U f) <= lambda_+ norm(f)^2$, which with
Theorem 2 gives the bound on $psi$. $I - K$ is symmetric with eigenvalues at most $1 - lambda_-$, so
$norm(b) = norm((I - K) f) <= (1 - lambda_-) norm(f)$. If $alpha >= 0$ and $f != 0$, then
$psi >= (1 - lambda_+) norm(f)^2 > 0$, and $tan theta = norm(f) \/ psi$ gives the bound. If $f = 0$, then
$a = (norm(g) psi \/ sigma) gh$ with $psi = alpha$, and $a != 0$ forces $alpha > 0$. $square$

#thm("Proposition 2c (symmetric loops: sharp angle bound and positive first step)")[
Let $J = J^top$ with all eigenvalues $< 1$, and $a != 0$.

(i) $c_1 > 0$ always, even when $c_0 < 0$.

(ii) If the eigenvalues lie in $[lambda_-, lambda_+] subset (0, 1)$, set $m = lambda_- \/ (1 - lambda_-)$ and
$M = lambda_+ \/ (1 - lambda_+)$. Then $cos theta >= 2 sqrt(m M) \/ (m + M)$, and this bound is sharp.
]

*Proof.* Write $J = sum_i lambda_i v_i v_i^top$ with orthonormal $v_i$, and $g_i = v_i^top g$. Then
$a = sum_i mu_i g_i v_i$ with $mu_i = lambda_i \/ (1 - lambda_i)$.

(i) $g^top J a = sum_i lambda_i mu_i g_i^2 = sum_i lambda_i^2 \/ (1 - lambda_i) dot g_i^2 > 0$, unless every
term with $lambda_i != 0$ has $g_i = 0$. That would mean $J g = 0$ and $a = 0$.

(ii) $cos theta = sum_i mu_i g_i^2 \/ (norm(g) sqrt(sum_i mu_i^2 g_i^2))$ with all $mu_i in [m, M]$. This is the
Kantorovich inequality, with equality when $g$ is split between the extreme eigenvectors in the ratio
$g_-^2 \/ g_+^2 = M \/ m$. $square$

So symmetry protects the *sign*, not the *angle*. For $J = op("diag")(epsilon, 1\/2)$ and
$g = (sqrt(1 - epsilon), sqrt(epsilon))$, $C = 0$ but $cos theta = 2 sqrt(epsilon(1 - epsilon)) -> 0$. Near-orthogonality
alone does not require circulation. A *negative* start does, when $sigma > 0$ and $alpha + E > 0$, as the next corollary
shows.

#thm("Corollary 2b (what a negative start requires)")[
If $sigma > 0$ and $c_0 < 0$, then $2 C < -(alpha + E)$. So:
- if $alpha + E > 0$, the circulation must be negative and larger than $(alpha + E)\/2$ in size;
- if moreover $alpha >= 0$ and $omega(K) < 1$, then $norm(q) > (1 - omega(K)) norm(f) \/ 2$.

Without circulation, a negative start needs $alpha < -E$. Example: $J = op("diag")(-1\/2, 0)$, $g = e_1$ has
$C = 0$, $c_0 = -1$, $c_1 = 1\/2$.
]

*Proof.* $c_0 < 0$ with $sigma > 0$ means $psi < 0$, i.e. $2C < -alpha - E$. For the last claim: if $f = 0$ then
$psi = alpha >= 0$, which is impossible, so $f != 0$ and $E >= (1 - omega) norm(f)^2 > 0$. Then
$2 norm(q) norm(f) >= -2C > E >= (1 - omega) norm(f)^2$. $square$

*Correction to `derivations.pdf` D5.* D5 said the lever stays aligned with $g$ if $J$ is "symmetric (more generally,
normal)". The normal case is false, even with positive self-gain. For $J = 0.9 R(60 degree)$ and $g = e_1$ we get
$alpha = 0.45 > 0$, yet $c_0 = -0.42$. A rotation is normal, but its skew part makes $q != 0$. The correct statement
is Corollary 2a: symmetry *and* $alpha >= 0$ give $c_0 > 0$.

= One step later

#thm("Theorem 3 (the first-step correction has the sign of σE)")[
With $s = sgn(sigma)$,
$ c_0 = s (alpha + E + 2 C) / D, quad quad c_1 = alpha c_0 + s E / D, quad quad
  "so" quad c_1 = s (alpha (alpha + E + 2C) + E) / D . $
]

*Proof.* $c_1 = (J^top gh)^top hat(a)$ with $J^top gh = alpha gh + U b$ and $hat(a) = s (psi gh + U f) \/ D$
(Theorem 1). So $c_1 = s (alpha psi + b^top f) \/ D$. Then substitute Theorem 2. $square$

*What this does and does not say.* The circulation does *not* disappear from $c_1$. It enters through $alpha C$ and
through $D$. What is true is narrower but useful:
- The correction $c_1 - alpha c_0 = sgn(sigma) E \/ D$ has the sign of $sigma E$. Since $sigma = 1 - alpha - E - 2C$,
  this is not independent of $C$ globally. But *within* a regime where $sgn(sigma E)$ is fixed (e.g. Lemma 1(c) with
  $f != 0$), $C$ can change the size of the correction, not its sign.
- At $t = 0$, $C$ can make the start negative (Corollary 2b).
- At $t = 1$, if $sigma E > 0$ and $|alpha c_0| < |E| \/ D$, the response is positive, however negative $C$ made the
  start.

#thm("Corollary 3a (sufficient condition for the flip)")[
Suppose every eigenvalue of $J$ has real part $< 1$, $omega(K) < 1$, and $f != 0$. Then $sigma > 0$ and
$ c_1 - alpha c_0 = norm(f) (1 - R_K (f)) sin theta >= (1 - omega(K)) norm(f) sin theta > 0, quad
  R_K (f) = (f^top K f) / norm(f)^2 . $
So a lever with $c_0 < 0$ has $c_1 > 0$ whenever $(1 - omega(K)) norm(f) sin theta > |alpha| |c_0|$.
]

*Proof.* $sigma > 0$ is Lemma 1(c). $E \/ D = norm(f)^2 (1 - R_K (f)) \/ D$ and $norm(f) \/ D = sin theta$. Also
$R_K (f) <= omega(K)$. $square$

The condition is relative. $J = mat(0.01, 0.001; -20, 0)$ is stable with $omega(K) = 0$ and $alpha = 0.01$, yet
$c_0 = -0.995$ and $c_1 = -0.0099$. There $f$ is tiny, and the threshold correctly fails.

#thm("Corollary 3b (quality of the one-step lever)")[
$cos(J^top g, a) = c_1 \/ norm(J^top gh)$ always. If $alpha = 0$, $sigma > 0$ and $f != 0$, then
$cos(J^top g, a) = sin theta dot cos(b, f)$. If in addition $omega(K) < 1$,
$ cos(b, f) >= (1 - omega(K)) / norm(I - K) . $
]

*Proof.* The first identity is the definition of $c_1$. If $alpha = 0$, then $norm(J^top gh) = norm(b)$ and
$c_1 = E \/ D = (b^top f \/ norm(f)) sin theta$. For the bound, $b^top f >= (1 - omega) norm(f)^2 > 0$ and
$norm(b) = norm((I - K^top) f) <= norm(I - K) norm(f)$. $square$

This is the field-of-values quantity in Elman's GMRES bound, which also requires the symmetric part of $I - K$ to be
positive definite. The condition is *sufficient, not necessary*:
- $J = mat(0, 1, -2; 0, 0, 4; 0, 0, 0)$, $g = e_1$ has $omega(K) = 2$ and $cos(b, f) = -0.6$, below the formal bound,
  so the hypothesis cannot be dropped.
- $J = op("diag")(mat(0, 1; 0, 0), mat(0, 4; 0, 0))$, $g = e_1$: the unobserved block makes $omega(K) = 2$, while the
  one-step lever is exact ($a = J^top g$).
- $omega(K) < 1$ alone does not make the one-step lever good: $J = mat(0, 0.001; -500, 0)$ gives
  $cos(J^top g, a) = 0.002$.

= Bounds: how much circulation a lever needs <sec-bounds>

Theorems 1–3 are identities. This section turns them into inequalities whose inputs are either one-step
quantities ($alpha$, $q$, $omega(K)$) or properties of the lever alone. Write
$ Gamma = norm(a) \/ norm(g) $
for the *loop gain ratio*: the best persistent readout gain per unit push, divided by the best immediate gain
$norm(g)$, which a one-shot push along $gh$ achieves.

#thm("Theorem 4 (circulation bound)")[
If $sigma > 0$, $f != 0$ and $0 < theta < pi$, then
$ 2 norm(q) >= alpha / norm(f) + (1 - omega(K)) norm(f) - cot theta . $
Equality holds iff $f^top K f = omega(K) norm(f)^2$ and $q$ is a non-positive multiple of $f$. In particular, every
$2 times 2$ $J$ with $sigma > 0$ and $q f <= 0$ attains it, e.g. the stable $mat(0.05, 0.3; -0.4, 0.5)$, so the bound
cannot be improved in general. It is not attained for every input: the right side can be negative (for
$mat(0, 0.1; 0.2, 0)$ it is $-0.1$, while $2 norm(q) = 0.1$).
]

*Proof.* By Theorem 1 with $sigma > 0$, $cot theta = psi \/ norm(f)$. By Theorem 2, $psi = alpha + E + 2C$ with
$E = norm(f)^2 - f^top K f >= (1 - omega(K)) norm(f)^2$ and $C = q^top f >= -norm(q) norm(f)$. Divide by $norm(f)$ and
rearrange. Equality needs equality in both steps: the first is the Rayleigh condition, the second is Cauchy–Schwarz
with $q$ anti-parallel to $f$ or $q = 0$. In dimension 2, $K$ is a scalar, so the first condition always holds. $square$

#thm("Corollary 4a (in terms of the lever alone)")[
$sigma = 1 \/ (1 + Gamma cos theta)$ and $norm(f) = Gamma sin theta \/ abs(1 + Gamma cos theta)$. If $sigma > 0$
(equivalently $1 + Gamma cos theta > 0$), $alpha >= 0$ and $f != 0$, then
$ 2 norm(q) >= (1 - omega(K)) (Gamma sin theta) / (1 + Gamma cos theta) - cot theta . $
At $theta = 90 degree$ this reads $2 norm(q) >= (1 - omega(K)) Gamma$. Equality needs $alpha = 0$ as well as the
equality conditions of Theorem 4.
]

*Proof.* From Theorem 1, $gh^top a \/ norm(g) = psi \/ sigma$ and $norm(a_perp) \/ norm(g) = norm(f) \/ |sigma|$. With
$x = psi \/ sigma = Gamma cos theta$ we get $sigma = 1 \/ (1 + x)$ and $norm(f) = abs(sigma) Gamma sin theta$.
Substitute into Theorem 4 (which needs $sigma > 0$) and drop $alpha \/ norm(f) >= 0$. $square$

The hypothesis $sigma > 0$ cannot be dropped: $J = mat(0.1, 1; -0.5, 1.2)$, $g = e_1$ is stable with $sigma = -1.6$,
and there $2 norm(q) = 1.5$ while the right side is $1.52$.

In words: *a lever that is both strong ($Gamma$ large) and nearly orthogonal to the readout forces a large
circulation*, unless the outside dynamics is nearly non-dissipative ($omega(K)$ near or above 1).

#thm("Corollary 4b (a uniform threshold)")[
If $sigma > 0$, $alpha >= 0$, $omega(K) < 1$ and $a != 0$, then
$ cot theta >= 2 (sqrt(alpha (1 - omega(K))) - norm(q)) . $
So if $norm(q) < sqrt(alpha (1 - omega(K)))$, then $c_0 > 0$: the lever cannot start against the readout. This
bound does not depend on $f$.
]

*Proof.* If $f = 0$, then $a prop psi gh = alpha gh$, so $a != 0$ forces $alpha > 0$, and then $theta = 0$ and
both claims hold trivially ($cot theta = +infinity$). Otherwise apply
$alpha \/ x + (1 - omega) x >= 2 sqrt(alpha (1 - omega))$ (AM–GM, $x = norm(f) > 0$) to Theorem 4. $square$

#thm("Corollary 4c (symmetric loops: nearly orthogonal levers are weak)")[
Let $J = J^top$ with eigenvalues $< 1$, largest $lambda_+$, $alpha >= 0$, $a != 0$, and $0 < theta <= 90 degree$. If
$(1 - lambda_+) sin^2 theta > cos^2 theta$, then
$ Gamma <= (cos theta) / ((1 - lambda_+) sin^2 theta - cos^2 theta) . $
Writing $theta = 90 degree - phi$, this is $Gamma <= sin phi \/ ((1 - lambda_+) cos^2 phi - sin^2 phi) approx
phi \/ (1 - lambda_+)$ when $phi^2 << 1 - lambda_+$ ($phi$ in radians).
]

*Proof.* Here $q = 0$, $sigma > 0$ (Corollary 2a) and $omega(K) <= lambda_+$. If $f = 0$, then $theta = 0$, which is
excluded. Corollary 4a with $norm(q) = 0$ gives
$(1 - lambda_+) Gamma sin theta <= cot theta (1 + Gamma cos theta)$. Solve for $Gamma$. $square$

Compare Proposition 2c: symmetric loops *can* produce nearly orthogonal levers. When $alpha >= 0$ and $lambda_+$ is
bounded away from 1, Corollary 4c says they are weak. Without the gap the cap fails: for $J = op("diag")(epsilon,
1 - epsilon)$, $g = (sqrt(1 - epsilon), sqrt(epsilon))$, $epsilon = 10^(-4)$, the lever has $Gamma approx 100$ at
$theta approx 89.4 degree$.

The write-up's cosine $0.10$ is between two frozen *population* directions: the lever $d^*$ and the population
readout-gradient direction. It is not the item-level $theta$ of this section. For an item's own $g$ and $a$, both
$theta$ and $Gamma$ remain to be measured (@sec-huginn).

#thm("Proposition 5 (angle of the one-step lever)")[
If $sigma > 0$, $a != 0$, $b != 0$ and $f != 0$, then
$ angle(J^top g, a) <= arctan(|alpha| \/ norm(b)) + angle(b, f) + |90 degree - theta| . $
Moreover $f - b = K^top f$, so $sin angle(b, f) <= norm(K^top f) \/ norm(f)$ whenever this ratio is below 1 (and then
the angle is acute). If $omega(K) < 1$, also $cos angle(b, f) >= (1 - omega(K)) \/ norm(I - K)$.
]

*Proof.* Angles between nonzero vectors satisfy the triangle inequality (they are geodesic distances on the
sphere). Chain $J^top gh = alpha gh + U b$ to $U b$ (angle $arctan(|alpha| \/ norm(b))$), then $U b$ to $U f$
(angle $angle(b, f)$, since $U$ is an isometry), then $U f$ to $a prop psi gh + U f$ (angle
$arctan(|psi| \/ norm(f)) = |90 degree - theta|$, using $sigma > 0$). For the sine bound, the distance from $f$ to the
line through $b$ is at most $norm(f - b) = norm(K^top f)$. If $norm(f - b) < norm(f)$, then
$b^top f > norm(b)^2 \/ 2 > 0$. The cosine bound is Corollary 3b. $square$

When $abs(alpha) << norm(b)$ and $theta approx 90 degree$, the first and third terms are small and the bound is
dominated by $angle(b, f)$: how far one more application of the outside dynamics turns the passage direction. If
$r = norm(K^top hat(f)) < 1$ (one extra VJP), then $angle(b, f) <= arcsin r$. Small $abs(alpha)$ alone is not
enough: $J = mat(0.001, 10^(-6); -1000, 0)$, $g = e_1$ has $sigma = 1$, $theta = 90 degree$, $K = 0$ and $b$
parallel to $f$, yet $angle(J^top g, a) approx 89.9 degree$, because $abs(alpha) \/ norm(b) = 1000$.

#check[Table row 6 (@sec-examples) has $Gamma = 2.05$, $theta = 96.5 degree$, $omega(K) = 0.68$. Corollary 4a requires
$norm(q) >= 0.48$; the actual value is $1.26$. The bound is a worst case: it assumes the circulation is perfectly
anti-aligned with $f$, and that $f$ sits on the top eigenvector of the symmetric part of $K$.]

= Further structure

#thm("Proposition 4 (damping leaves the first-passage covector unchanged)")[
Let $J_beta = (1 - beta) I + beta J$ with $beta > 0$. This is the linearization of $s |-> (1 - beta) s + beta F(s)$,
with the push added to the state before the *whole* relaxed map. Then
$ f_beta = f, quad sigma_beta = beta sigma, quad psi_beta = 1 - beta sigma, quad E_beta = beta E, quad
  C_beta = beta C, quad a_beta = (a + g) / beta - g . $
]

*Proof.* $alpha_beta = 1 - beta + beta alpha$, $ell_beta = beta ell$, $b_beta = beta b$, $q_beta = beta q$, and
$K_beta = (1 - beta) I + beta K$. So $I - K_beta^top = beta (I - K^top)$ and $f_beta = f$. The rest follows from the
definitions, and $a_beta = (I - J_beta^top)^(-1) g - g = beta^(-1) (I - J^top)^(-1) g - g$. $square$

The identities are algebraic. Reading $a_beta$ as a persistent-push response also needs the relaxed loop to converge.
If $sigma > 0$, then $psi_beta = 1 - beta sigma > 0$ for $beta < 1 \/ sigma$. So under strong enough damping the
relaxed loop's lever never starts negative. Pushing inside $F$ only is a different intervention, and this
proposition does not apply to it.

*Remark (several readouts).* Let $Q$ be an orthonormal basis of a $k$-dimensional readout span, and write
$J = mat(A, B^top; L, K)$ in the basis $(Q, U)$. Assume $I - J$ and this new $(d - k)$-dimensional $I - K$ are invertible; the scalar-readout assumption does not imply
the latter. Set $F = (I - K^top)^(-1) B$ and
$Sigma = I - A - B^top (I - K)^(-1) L$. A readout gradient with coordinates $(v, 0)$ then has lever coordinates
$((Sigma^(-top) - I) v, quad F Sigma^(-top) v)$. The scalar sign arguments do not carry over automatically.

*Remark (other frequencies).* As power series near $z = 0$,
$gh^top (I - z J)^(-1) gh = 1 \/ (1 - Psi(z))$ with $Psi(z) = sum_n psi_n z^n = z alpha + z^2 b^top (I - z K)^(-1) ell$.
Evaluating the closed form needs $I - z K$ invertible, and the left side needs $I - z J$ invertible. For $J = mat(1, 1; -1, -1)$, $I + J$ is invertible but $I + K$ is
not, so evaluation at $z = -1$ (the period-2 response, D10) can require analytic continuation.

= Worked examples ($g = e_1$) <sec-examples>

Computed by `docs/analytic_check.py`. It asserts the following on about 3,900 random stable matrices (generic,
strongly non-normal, symmetric, normal), 200 multi-readout cases, and each counterexample quoted in this note:
- the identities and inequalities of Lemma 1, Theorems 1–3, Corollaries 2a–3b, Propositions 2c and 4;
- the bounds of @sec-bounds: Theorem 4 (with equality and strict cases in dimensions 2 and 4), Corollaries 4a–4c
  (the 4a identities for both signs of $sigma$; 4c on about 230 symmetric cases) and Proposition 5;
- the @sec-huginn reconstruction and finite-$T$ identities. The residual identity is checked against a worst-case
  roundoff scale, and only where that scale is informative (about 3,200 cases);
- every table entry, eigenvalues included, to its displayed rounding.

The random draws have $rho(J) < 1$ and damping $beta in (0.1, 1)$. The Kantorovich branch of Proposition 2c is
exercised by few random cases and one sharpness example.

#align(center, table(columns: 9, inset: 4pt, align: center,
  table.header([$J$], [eigenvalues], [$alpha$], [$norm(J gh)$], [$E$], [$C$], [$psi$], [$c_0$], [$c_1$]),
  [$mat(0.1, 2; 0, 0.5)$], [$0.1, 0.5$], [$0.1$], [$0.10$], [$8$], [$-4$], [$0.1$], [$0.025$], [$2.00$],
  [$mat(0.1, 2; -0.2, 0.5)$], [$0.3 plus.minus 0.6 i$], [$0.1$], [$0.22$], [$8$], [$-4.4$], [$-0.7$], [$-0.17$], [$1.95$],
  [$0.9 R(60 degree)$], [$0.45 plus.minus 0.78 i$], [$0.45$], [$0.90$], [$1.10$], [$-1.10$], [$-0.65$], [$-0.42$], [$0.52$],
  [$0.6 mat(0.1, 1; 1, 0.5)$], [$-0.43, 0.79$], [$0.06$], [$0.60$], [$0.51$], [$0$], [$0.57$], [$0.56$], [$0.53$],
  [$mat(-0.5, 0; 0, 0)$], [$-0.5, 0$], [$-0.5$], [$0.50$], [$0$], [$0$], [$-0.5$], [$-1$], [$0.5$],
  [$mat(0, 1.5, 1.5; -0.9, 0.6, 0; 0.74, -0.4, 0.2)$], [$0.6, 0.1 plus.minus 0.48 i$], [$0$], [$1.17$], [$5.63$],
    [$-2.96$], [$-0.30$], [$-0.11$], [$2.11$],
))

- *Row 1.* One-way coupling ($ell = 0$) means $2C = -E$ exactly, so $psi = alpha$. The lever is nearly orthogonal.
- *Row 2.* A small backward leak tips the cancellation negative. The lever reverses, then turns strongly positive.
- *Row 3.* Normal, positive self-gain, and still a reversal: this is the D5 counterexample.
- *Row 4.* Symmetric: $C = 0$, and the lever's cosine stays positive ($0.56$).
- *Row 5.* Symmetric reversal through negative self-gain ($alpha < -E$). This is the alternative route to $c_0 < 0$.
- *Row 6, a Huginn-like toy.* It shows the following qualitative features of the measurements at once:
  - zero self-gain;
  - a readout push that is suppressed but not contracted ($norm(J gh) = 1.17 > rho(J) = 0.6$, so $J$ is not normal);
  - a slightly negative start and a strong positive first step;
  - a one-step lever with cosine $0.994$ to the full lever ($omega(K) = 0.68$).

  The negative start comes from circulation: $2C = -5.93$ slightly over-cancels $E = 5.63$.

Rows 1–2 contract the readout push, so they reproduce the reorientation but not the "suppression without
contraction" observation. Row 6 reproduces both.

= What this says about Huginn, and how to test it <sec-huginn>

*Partial measurement from existing data (added; details in `freeloop.pdf`).* Existing runs saved, per item, the
readout gradient $g$, the 48-step lever $a_48$ and one pullback $J^top g$ (truth: 120 statements; digits: 20 prompts ×
10 centered readouts). These give $alpha$, $x_T$, $sigma_T$, $psi_T$, $E_T$, $Gamma_T$, $c_0$ and $c_1$ per item, and
$C_T$ through the identity $psi_T = alpha + E_T + 2 C_T$. That identity folds the truncation residual into $C_T$, so
$C_T$ is reconstructed, not measured; the direct estimate $tilde(q)^top tilde(f)_T$ needs the JVP below.
- All 120 truth levers start positive. 59 of the 200 digit levers start negative, and *all 59* satisfy the conditions
  of the proposed regime: $sigma_T > 0$, $E_T > 0$, $alpha + E_T > 0$, $c_1 > 0$, and the Corollary 3a condition.
- But $E_T + 2 C_T$ is near zero relative to $E_T$ whether or not the lever starts negative (median $-3.8%$ for truth,
  $+0.9%$ for digits, with 80% of digit readouts in $[-5.5%, 6.8%]$). So the circulation needed for a negative start is present everywhere. What separates negative starts is the
  self-gain: $alpha < 0$ for 53 of the 59, against 15 of the 141 positive starts.
- `freeloop.pdf` measures random directions in the same loops (a pre-registered run). They show the same
  cancellation ($E + 2C$ about $-9%$ of $E$), which for random directions is, up to fluctuations of order
  $d^(-1\/2)$, a function of the spectrum (its Theorems 1–2). So the circulation reading is correct but generic. A
  random-rotation null model that would make it hold for every readout fails the same run's pre-registered criteria.
  What separates readouts is a self-gain below the probe estimate of the loop's mean eigenvalue (all 240 readouts
  tested).
These numbers are asserted by `freeloop_check.py` (Part C).

*What existing data cannot say.* The theorems concern the exact per-item lever and the item's own gradient. The
existing summaries (`rtransfer`, `revmech`) use frozen *population* pushes of norm 3 and medians of medians. So they
cannot identify $alpha$, $E$ or $C$, and an earlier draft's numerical "consistency" estimate is withdrawn. The
qualitative pattern is *compatible* with the regime of Row 6:
- population levers start slightly negative and respond strongly positive at $t = 1$;
- readout-gradient pushes have near-zero one-step readout response.

It does not establish that regime. Centered logits are not an obstacle: $R = z_k - (1\/10) sum_j z_j$ is a valid scalar
readout, provided $g = nabla R$ is used consistently.

*Per-item measurement.* At the settled state of each held-out item:
+ One VJP gives $J^top gh$, hence $alpha = gh^top J^top gh$. This VJP is also the first step of the lever chain.
+ One JVP gives $J gh$. Set $tilde(q) = (1\/2)(I - gh gh^top)(J gh - J^top gh)$, the ambient lift of $q$.
+ The raw item lever $a_T = sum_(t=1)^T (J^top)^t g$ (unnormalized, not a population average) gives *estimates*
  $x_T = gh^top a_T \/ norm(g)$, $sigma_T = 1 \/ (1 + x_T)$, $psi_T = x_T \/ (1 + x_T)$, and
  $tilde(f)_T = sigma_T (I - gh gh^top) a_T \/ norm(g)$. These require $1 + x_T != 0$. For the exact lever
  ($T -> infinity$) they equal $sigma$, $psi$ and the ambient lift of $U f$.
+ Report $E_T = (J^top gh)^top tilde(f)_T$, $C_T = tilde(q)^top tilde(f)_T$, and $1 - E_T \/ norm(tilde(f)_T)^2$ (an
  estimate of $R_K (f)$) when $tilde(f)_T != 0$. Also report $Gamma_T = norm(a_T) \/ norm(g)$ and $theta_T$, the
  inputs of Corollary 4a.
+ *Optional, for the bounds:* $omega(K)$ is the top eigenvalue of $P S P$ restricted to $gh^perp$, where
  $S = (J + J^top)\/2$ and $P = I - gh gh^top$. Lanczos needs one JVP and one VJP per iteration. Comparing the
  measured $norm(tilde(q))$ with the Corollary 4a requirement (which also needs $sigma_T > 0$) shows how much slack
  the worst case leaves. A Lanczos Ritz value is at most $omega(K)$, so substituting it overstates the requirement and
  understates the slack. The comparison is then an estimate; an apparent violation signals an unconverged
  $omega(K)$ or finite-$T$ error, not a failure of the theorem. A certified requirement needs an upper bound on
  $omega(K)$. $norm(K^top hat(f))$ (one more VJP) bounds the one-step lever's angle via Proposition 5.
+ *Error control.* Do not assume $T = 48$ is enough. For strongly non-normal $J$, $rho = 0.78$ does not imply a
  small tail: a 50-dimensional Jordan-type example has over 90% relative error at $T = 48$.
  - The residual is $r_T = (I - J^top) a_T - J^top g = -(J^top)^(T+1) g$, and the lever error is
    $a_T - a = (I - J^top)^(-1) r_T$. A small residual therefore bounds the error only together with a bound on
    $norm((I - J^top)^(-1))$.
  - Agreement between $T = 48$ and $T = 96$ is a diagnostic, not a certificate.
  - The scalar check $psi_T - alpha - E_T - 2 C_T = (sigma_T \/ norm(g)) gh^top r_T$ tests one component of the residual.
  - Treat sign conclusions as established only when their margins clearly exceed the estimated error.
  - Cost: about 50 Jacobian products per item at $T = 48$, about 100 at $T = 96$.

*What the measurement would decide.* The identities hold automatically, so they are not being tested. The
questions are about which regime each item is in.
- *Inside the proposed regime:* on items whose lever starts negative, $sigma > 0$, $E > 0$, $|alpha| |c_0| < E \/ D$ and
  $alpha + E > 0$. Corollary 3a then explains the flip. Negative circulation is algebraically necessary for the
  negative start, because self-gain alone does not overcome $E$. Self-gain may still contribute.
- *Outside it:* if $sigma < 0$ or $E <= 0$ on flipping items, those items are outside the proposed sufficient regime.
  This mechanism then does not account for them, but that alone does not identify another one. For example, the
  nilpotent $J$ of Lemma 1(c) has $sigma = -1$, $E = -1$, and still a positive correction.
- *If $alpha < -E$:* circulation is not algebraically necessary for the negative start, though it may still contribute
  most of it. Example: $J = mat(-0.02, 0.1; -5, 0)$ has $alpha = -0.02$, $E = 0.01$, $2C = -0.51$.
- *Negative passage term:* a negative start can also come from $E < 0$ with $alpha = C = 0$. Example:
  $J = mat(0, 0.1, -0.2; 0.1, 0, 4; -0.2, 0, 0)$ (stable, symmetric readout coupling) has $E = -0.03$ and
  $c_0 = -0.13$. Measuring $E$ separates this case.

The write-up's $|c_1 \/ c_0| = 0.05$ for readout-gradient pushes suggests $|alpha|$ is small, but it must be
re-measured with item gradients. The same applies to $norm(J gh) approx 1.17$.

= Status and limits

- *Classical ingredients.* Lemma 1 is a Schur complement and, in probability language, a renewal equation. The
  symmetric/skew split, the Kantorovich inequality and the field-of-values bound are standard. No novelty search
  has been done.
- *What the note adds.*
  - An exact decomposition of the steering lever into return and first-passage parts.
  - Exact formulas for $c_0$ and $c_1$.
  - Sign statements, each under the stated hypotheses:
    - symmetric $J$ with $lambda_max (J) < 1$ and $a != 0$ gives $c_1 > 0$, and adding $alpha >= 0$ gives $c_0 > 0$;
    - with $sigma > 0$ and $alpha + E > 0$, a negative start requires negative circulation.
  - Sufficient conditions for the first-step flip (Corollary 3a).
  - When $sigma > 0$: a lower bound on the circulation a lever of given angle and loop gain requires (Theorem 4,
    attained in explicit cases; Corollary 4a, attained only if also $alpha = 0$), a uniform no-reversal threshold
    (4b), a cap on nearly orthogonal levers in symmetric loops with a spectral gap (4c), and an angle bound for the
    one-step lever (Proposition 5).
  - A corrected D5.
- *What it does not do.*
  - It does not fully show that Huginn's items are in the regime where these conditions hold. Existing data place
    every negative-start digit readout in it, with circulation reconstructed through the identity and $T = 48$
    (@sec-huginn); a direct circulation measurement (one JVP per item) has not been made. The cancellation
    $E + 2C approx 0$ behind it is shared by random directions and, for them, is a function of the spectrum
    (`freeloop.pdf`).
  - It does not establish circulation as necessary for near-orthogonality; Proposition 2c shows it is not.
  - If the regime is confirmed, it would still not explain how training produced the measured circulation.
- *Scope.* Linear response at a fixed point, real arithmetic, a scalar readout (see the remark for several readouts),
  and the resolvent lever. Finite-$T$ quantities are estimates (@sec-huginn).
