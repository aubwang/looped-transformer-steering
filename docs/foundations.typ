#set document(title: "Foundations: the math behind the Huginn study, from definitions")
#set page(paper: "us-letter", margin: (x: 1in, y: 0.9in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => [#v(0.7em) #it #v(0.2em)]

#let boxed(title, color, body) = block(
  fill: color, inset: 9pt, radius: 3pt, width: 100%, breakable: true,
  [*#title.* #body],
)
#let defn(title, body) = boxed("Definition: " + title, rgb("#eef3fb"), body)
#let thm(title, body) = boxed(title, rgb("#f4eefb"), body)
#let ex(body) = boxed("Example", rgb("#eef8ee"), body)
#let why(body) = boxed("Why we need this", rgb("#fbf3e6"), body)
#let q(body) = boxed("Check yourself", rgb("#f3f3f3"), body)

#align(center)[
  #text(17pt, weight: "bold")[Foundations]
  #v(0.1em)
  #text(12pt)[The math behind the Huginn study, built from definitions]
  #v(0.2em)
  #text(10pt)[Read first. Then `primer.pdf`, then `derivations.pdf`. Draft, 2026-11-07.]
]

#v(0.8em)

*What this assumes.* Sets, functions, real numbers, limits, and the derivative of a function of one variable. Nothing
else. Every other object is defined here, and every fact is proved or checked on a small example.

*How it's organized.* Chapters 1–5 build linear algebra. Chapter 6 is geometric series. Chapter 7 is derivatives in
several variables. Chapter 8 is iterated maps: fixed points, stability, and period-2 cycles. Chapter 9 puts everything
together on a two-dimensional toy loop that reproduces the study's main finding, with every number computable by hand.

*How to read it.* Do every "Check yourself" box before moving on (answers at the end). Each chapter opens with a box
saying why the project needs it.

*One running example.* Most chapters use the same matrix:
$ J = mat(0.1, 2; 0, 0.5). $
By Chapter 9 you will know everything about it, and it will turn out to be a miniature of Huginn's loop.

= Vectors

#why[The model's internal state at one position is a list of 5280 numbers. A "push" adds another such list to it.]

#defn("vector")[A vector in $bb(R)^d$ is an ordered list of $d$ real numbers, written as a column:
$ x = vec(x_1, x_2, dots.v, x_d). $
Add vectors entry by entry, and multiply by a number $a$ entry by entry:
$ x + y = vec(x_1 + y_1, dots.v, x_d + y_d), quad quad a x = vec(a x_1, dots.v, a x_d). $]

#defn("standard basis")[$e_i$ is the vector with 1 in entry $i$ and 0 elsewhere. In $bb(R)^2$:
$e_1 = vec(1, 0)$, $e_2 = vec(0, 1)$. Every vector is a combination of them: $x = x_1 e_1 + dots.c + x_d e_d$.]

#defn("linear combination, span")[A linear combination of $v_1, dots, v_k$ is $a_1 v_1 + dots.c + a_k v_k$. The span
of a set of vectors is the set of all their linear combinations. A "direction" means a nonzero vector, up to scaling.]

#q[Write $vec(3, -2)$ as a combination of $e_1, e_2$. Is $vec(2, 4)$ in the span of $vec(1, 2)$?]

= Linear maps and matrices

#why[Near a fixed point, one loop step acts on small pushes as a linear map. Matrices are how we write linear maps
down.]

#defn("linear map")[A function $A : bb(R)^d -> bb(R)^m$ is linear if for all vectors $x, y$ and numbers $a$:
$ A(x + y) = A x + A y, quad quad A(a x) = a A x. $]

#thm("Fact: a linear map is determined by what it does to the basis")[Since $x = sum_j x_j e_j$, linearity gives
$ A x = sum_j x_j A e_j. $
So $A$ is completely described by the $d$ vectors $A e_1, dots, A e_d$. Writing them as columns gives the *matrix* of
$A$:
$ A = mat(|, , |; A e_1, dots.c, A e_d; |, , |). $
The entry in row $i$, column $j$ is $A_(i j) = (A e_j)_i$.]

#thm("Fact: matrix–vector product")[From $A x = sum_j x_j A e_j$, entry $i$ of $A x$ is
$ (A x)_i = sum_j A_(i j) x_j. $]

#ex[For the running matrix, $J e_1 = vec(0.1, 0)$ and $J e_2 = vec(2, 0.5)$, which are its columns. For
$x = vec(1, 1)$: $J x = 1 dot vec(0.1, 0) + 1 dot vec(2, 0.5) = vec(2.1, 0.5)$.]

#defn("composition and matrix product")[If $B$ then $A$ are applied, the composite $x |-> A(B x)$ is linear. Its
matrix is the product $A B$. Its column $j$ is $A(B e_j)$, which gives
$ (A B)_(i k) = sum_j A_(i j) B_(j k). $
Order matters: in general $A B != B A$. Powers mean repeated application: $A^2 = A A$, $A^t = A dots.c A$
($t$ times), $A^0 = I$, where $I$ is the identity ($I x = x$; ones on the diagonal).]

#defn("inverse")[$A$ (square) is invertible if some $A^(-1)$ has $A^(-1) A = A A^(-1) = I$. For a $2 times 2$ matrix,
$ mat(a, b; c, d)^(-1) = 1 / (a d - b c) mat(d, -b; -c, a), quad "which exists exactly when" a d - b c != 0. $
The number $det mat(a, b; c, d) = a d - b c$ is the determinant. In practice we rarely invert: to find $x$ with
$A x = b$, we solve the equations directly.]

#q[Compute $J^2$. Then compute $J (J e_2)$ and check it equals the second column of $J^2$.]

= Length, angle, and the Cauchy–Schwarz inequality

#why[The study compares directions by their cosine ("the lever is nearly orthogonal to the probe"), and the "best
push" result is Cauchy–Schwarz.]

#defn("inner product, norm")[$x^top y = sum_i x_i y_i$ (also written $x dot y$). The norm (length) is
$norm(x) = sqrt(x^top x)$. Two vectors are orthogonal if $x^top y = 0$.]

#thm("Theorem (Cauchy–Schwarz)")[$|x^top y| <= norm(x) norm(y)$, with equality exactly when one is a multiple of the
other.

*Proof.* If $y = 0$ both sides are 0. Otherwise, for every number $t$,
$ 0 <= norm(x - t y)^2 = norm(x)^2 - 2 t x^top y + t^2 norm(y)^2. $
Take $t = x^top y \/ norm(y)^2$: this gives $0 <= norm(x)^2 - (x^top y)^2 \/ norm(y)^2$, which rearranges to the
claim. Equality forces $norm(x - t y) = 0$, i.e. $x = t y$. #h(1fr) $square$]

#defn("cosine")[For nonzero $x, y$: $cos(x, y) = x^top y \/ (norm(x) norm(y))$. By Cauchy–Schwarz it lies in
$[-1, 1]$. It is 1 for the same direction, $-1$ for opposite, 0 for orthogonal.]

#thm("Corollary: the best direction")[Fix a vector $a$. Among all $u$ with $norm(u) = 1$, the largest value of
$a^top u$ is $norm(a)$, reached at $u = a \/ norm(a)$. And for any unit $u$, $a^top u = norm(a) cos(a, u)$: a
direction gets the fraction $cos(a, u)$ of the best possible value.

*Proof.* Cauchy–Schwarz gives $a^top u <= norm(a)$, with equality when $u$ is a positive multiple of $a$. The second
statement is the definition of cosine with $norm(u) = 1$. #h(1fr) $square$]

#q[Find the cosine between $vec(0.1, 2)$ and $e_1$. Which unit vector $u$ maximizes $3 u_1 + 4 u_2$, and what is the
maximum?]

= The transpose, and why covectors are different from vectors

#why[This is the heart of the study. "Where states differ" (a probe) and "what the readout responds to" (a lever) are
different kinds of object, and the loop moves them differently.]

#defn("transpose")[$A^top$ is $A$ with rows and columns swapped: $(A^top)_(i j) = A_(j i)$. For the running matrix,
$J^top = mat(0.1, 0; 2, 0.5)$.]

#thm("Fact: moving a matrix across an inner product")[For all $x, y$: $(A x)^top y = x^top (A^top y)$.

*Proof.* Both sides equal the double sum $sum_i sum_j A_(i j) x_j y_i$: the left groups it by $i$ first, the right by
$j$ first. #h(1fr) $square$

Also $(A B)^top = B^top A^top$, which you can check the same way.]

#defn("linear functional")[A linear map $ell : bb(R)^d -> bb(R)$ (a map to numbers). Example: $ell(x) = x_1$, "read
the first coordinate".]

#thm("Fact: every linear functional is an inner product")[If $ell$ is linear, then $ell(x) = g^top x$ for exactly
one vector $g$, namely $g_i = ell(e_i)$.

*Proof.* $ell(x) = ell(sum_i x_i e_i) = sum_i x_i ell(e_i) = g^top x$. #h(1fr) $square$

So a functional is represented by a vector $g$, but it is used differently: you *measure* a state with it rather than
*add* it to the state. We call such a $g$ a *covector*.]

#thm("Fact: pulling a functional back through a map")[Measure with $g$ *after* applying $A$:
$ ell(A x) = g^top (A x) = (A^top g)^top x. $
So "apply $A$, then measure with $g$" is the same as "measure with $A^top g$". Vectors are pushed forward by $A$;
covectors are pulled back by $A^top$.]

#ex[Readout $g = e_1$ (read coordinate 1) and the running matrix $J$. After one step, the readout of a push $x$ is
$g^top J x = (J^top g)^top x$, with $J^top g = vec(0.1, 2)$. So one step later, the readout cares mostly about
coordinate *2*, not coordinate 1. Pushing $e_1$ gives readout $0.1$ after one step; pushing $e_2$ gives $2$.]

#q[Using $J^top$, find the vector $h$ with $g^top J^2 x = h^top x$ for $g = e_1$. Check by computing $J^2 e_2$
directly.]

= Eigenvalues and eigenvectors

#why[Eigenvalues say whether repeated application of a matrix shrinks, grows, or flips things. "Spectral radius 0.78"
and "eigenvalue past $-1$" are the two key numbers in the study.]

#defn("eigenvector, eigenvalue")[A nonzero $v$ with $A v = lambda v$. Then $A^t v = lambda^t v$: along $v$, the
matrix is just multiplication by the number $lambda$.]

*Finding them ($2 times 2$).* $A v = lambda v$ with $v != 0$ means $(A - lambda I) v = 0$ has a nonzero solution,
which happens exactly when $det(A - lambda I) = 0$:
$ det mat(a - lambda, b; c, d - lambda) = lambda^2 - (a + d) lambda + (a d - b c) = 0. $
It is a quadratic, so there are two roots (possibly equal, possibly complex).

#ex[For the running matrix, $det(J - lambda I) = (0.1 - lambda)(0.5 - lambda)$, so $lambda = 0.1$ and $0.5$. For a
triangular matrix the eigenvalues are always the diagonal entries.
- $lambda = 0.1$: $(J - 0.1 I) v = mat(0, 2; 0, 0.4) v = 0$ gives $v = e_1$.
- $lambda = 0.5$: $mat(-0.4, 2; 0, 0) v = 0$ gives $v = vec(5, 1)$.
The two eigenvectors have cosine $5 \/ sqrt(26) approx 0.98$: they are *almost parallel*. Remember this.]

#defn("spectral radius")[$rho(A) = max |lambda|$ over all eigenvalues. For $J$: $rho = 0.5$.]

#thm("Fact: powers through eigenvectors")[If $A$ has $d$ independent eigenvectors, collect them as columns of $V$ and
the eigenvalues in a diagonal matrix $Lambda$. Then $A V = V Lambda$, so $A = V Lambda V^(-1)$, and
$ A^t = (V Lambda V^(-1))(V Lambda V^(-1)) dots.c = V Lambda^t V^(-1), $
because the inner $V^(-1) V$ pairs cancel. $Lambda^t$ is diagonal with entries $lambda_i^t$.]

#defn("operator norm")[$norm(A) = max_(norm(x) = 1) norm(A x)$, the most $A$ can stretch a vector. Then
$norm(A x) <= norm(A) norm(x)$ and $norm(A B) <= norm(A) norm(B)$.]

#thm("Theorem: when do powers go to zero?")[If $rho(A) < 1$ then $A^t -> 0$. With $V$ as above,
$ norm(A^t) <= norm(V) norm(V^(-1)) rho(A)^t = kappa(V) rho(A)^t. $
*Proof* (diagonalizable case). $norm(A^t) = norm(V Lambda^t V^(-1)) <= norm(V) norm(Lambda^t) norm(V^(-1))$, and
$norm(Lambda^t) = max |lambda_i|^t = rho^t$. #h(1fr) $square$

(The theorem holds without diagonalizability too, by Gelfand's formula $rho(A) = lim norm(A^t)^(1\/t)$.)]

*Transient growth.* The factor $kappa(V) = norm(V) norm(V^(-1))$ is large when eigenvectors are nearly parallel, and
then $A^t x$ can get *bigger* before it gets smaller, although every $|lambda| < 1$.

#ex[Apply $J$ repeatedly to $e_2$ (length 1):
#align(center, table(columns: 6, stroke: 0.4pt, inset: 5pt,
  [$t$], [1], [2], [3], [4], [5],
  [$J^t e_2$], [$(2, 0.5)$], [$(1.2, 0.25)$], [$(0.62, 0.125)$], [$(0.312, 0.0625)$], [$(0.156, 0.031)$],
  [length], [2.06], [1.23], [0.63], [0.32], [0.16],
))
It doubles first, then halves each step (rate $rho = 0.5$). Compare $e_1$: $0.1, 0.01, 0.001$, never growing.]

*Sign of an eigenvalue.* Along an eigenvector, $lambda^t$ behaves as follows:
- $0 < lambda < 1$: shrinks steadily.
- $-1 < lambda < 0$: shrinks while flipping sign every step.
- $lambda < -1$: grows while flipping sign. This is where period-2 behavior comes from (Chapter 8).
- Complex $lambda = r e^(i theta)$: rotates by $theta$ and scales by $r$ each step, e.g.
  $mat(0.6, -0.6; 0.6, 0.6)$ has $lambda = 0.6 plus.minus 0.6 i$, $|lambda| approx 0.85$: a spiral inward.

#defn("symmetric, normal")[$A$ is symmetric if $A^top = A$. Then eigenvectors for different eigenvalues are
orthogonal: $lambda_1 v_1^top v_2 = (A v_1)^top v_2 = v_1^top (A v_2) = lambda_2 v_1^top v_2$, so
$(lambda_1 - lambda_2) v_1^top v_2 = 0$. So $kappa(V) = 1$ and there is no transient growth. More generally, $A$ is
*normal* if $A A^top = A^top A$; normal matrices also have orthogonal eigenvectors. "Non-normal" means the
eigenvectors are tilted toward each other, like $J$'s.]

#q[Find the eigenvalues of $mat(-1.2, 0; 0, 0.5)$. Starting from $vec(0.01, 1)$, what happens after many steps?]

= Geometric series

#why[A push applied every loop accumulates as $J P + J^2 P + J^3 P + dots.c$. Summing that series is the main formula
of the study.]

#thm("Scalar")[For $lambda != 1$: $1 + lambda + dots.c + lambda^(T-1) = (1 - lambda^T) \/ (1 - lambda)$.
*Proof.* Multiply the sum $S$ by $(1 - lambda)$: every term cancels except $1 - lambda^T$ (telescoping).
#h(1fr) $square$
If $|lambda| < 1$, letting $T -> infinity$ gives $1 \/ (1 - lambda)$.]

#thm("Matrix")[$(I - A)(I + A + dots.c + A^(T-1)) = I - A^T$, by the same telescoping. If $rho(A) < 1$, then
$A^T -> 0$, $I - A$ is invertible, and
$ sum_(t = 0)^infinity A^t = (I - A)^(-1), quad quad sum_(t = 1)^infinity A^t = A (I - A)^(-1). $
The second form just drops the $t = 0$ term ($A^0 = I$). Both $A$ and $(I - A)^(-1)$ are built from powers of $A$, so
they commute.]

#ex[For $J$: $I - J = mat(0.9, -2; 0, 0.5)$. To compute $(sum_(t >= 1) J^t) e_2 = (I - J)^(-1) J e_2$, solve
$(I - J) y = J e_2 = vec(2, 0.5)$: the second row gives $0.5 y_2 = 0.5$, so $y_2 = 1$; the first gives
$0.9 y_1 - 2 = 2$, so $y_1 approx 4.44$. Check: adding the table's columns ($2 + 1.2 + 0.62 + 0.31 + dots.c$) does
approach $4.44$.]

#q[With $lambda = 0.78$, compute $sum_(t >= 1) lambda^t$. Then compute $0.78^(16)$.]

= Derivatives in several variables

#why[The real loop is nonlinear. Near its fixed point it acts like a matrix, the Jacobian $J$. The readout's
sensitivity is a gradient $g$, and computing "the readout's sensitivity $t$ loops back" is $J^top g$, which is what
automatic differentiation does.]

*One variable, restated.* $f'(x)$ is the number for which $f(x + h) = f(x) + f'(x) h + "error"$, with error/$h -> 0$
as $h -> 0$. In words: the best linear approximation to $f$ near $x$.

#defn("Jacobian")[For $F : bb(R)^d -> bb(R)^m$, the Jacobian at $x$ is the matrix $J$ with
$ F(x + h) = F(x) + J h + "error", quad "error" \/ norm(h) -> 0. $
Its entries are partial derivatives: $J_(i j) = partial F_i \/ partial x_j$ (change only $x_j$, see how $F_i$
moves).]

#ex[$F(x_1, x_2) = (x_1 x_2, thin x_1 + x_2^2)$ has $J = mat(x_2, x_1; 1, 2 x_2)$. A linear map plus a constant,
$F(x) = M x + c$, has $J = M$ everywhere.]

#thm("Chain rule")[If $G$ has Jacobian $J_G$ at $x$ and $F$ has Jacobian $J_F$ at $G(x)$, then $F compose G$ has
Jacobian $J_F J_G$ at $x$.
*Why.* $F(G(x + h)) approx F(G(x) + J_G h) approx F(G(x)) + J_F J_G h$. Linear approximations compose by matrix
product.]

#defn("gradient")[For a number-valued $R : bb(R)^d -> bb(R)$, the derivative is a linear functional, represented
(Chapter 4) by the gradient $g = nabla R(x)$, with $g_i = partial R \/ partial x_i$:
$ R(x + h) approx R(x) + g^top h. $
By the best-direction corollary (Chapter 3), the unit $h$ that increases $R$ fastest is $g \/ norm(g)$.]

#thm("Fact: gradients pull back through maps")[The gradient of $x |-> R(F(x))$ is $J^top g$, where $g = nabla R$ at
$F(x)$.
*Proof.* $R(F(x + h)) approx R(F(x) + J h) approx R(F(x)) + g^top J h = R(F(x)) + (J^top g)^top h$. #h(1fr)
$square$

Applying this repeatedly is "backpropagation". Each application is one *vector–Jacobian product* (VJP). We never
build the $5280 times 5280$ matrix $J$; the software computes $J^top w$ for any $w$ directly.]

#q[For $F(x) = (x_1 x_2, thin x_1 + x_2^2)$ at $x = (1, 2)$ and $R(y) = y_1$, compute the gradient of $R compose F$
both directly and as $J^top g$.]

= Iterated maps: fixed points, stability, period-2

#why[The loop is $s_(t+1) = F(s_t)$, applied over and over. Its long-run behavior is the "dynamics" half of the
study.]

#defn("orbit, fixed point, cycle")[Given $f$ and a start $x_0$, the orbit is $x_0, f(x_0), f(f(x_0)), dots$. A fixed
point has $f(x^*) = x^*$. A period-2 cycle is a pair $a != b$ with $f(a) = b$ and $f(b) = a$.]

#thm("Theorem (stability in one variable)")[If $f$ is continuously differentiable, $f(x^*) = x^*$, and
$|f'(x^*)| < 1$, then orbits starting close enough to $x^*$ converge to it.

*Proof.* Pick $k$ with $|f'(x^*)| < k < 1$. By continuity, $|f'| <= k$ on some interval
$[x^* - r, x^* + r]$. For $x$ in it, the mean value theorem gives
$|f(x) - x^*| = |f(x) - f(x^*)| = |f'(xi)| |x - x^*| <= k |x - x^*|$. So the distance to $x^*$ shrinks by at least the
factor $k$ every step and the orbit stays in the interval: distance $<= k^t r -> 0$. #h(1fr) $square$

If $|f'(x^*)| > 1$, nearby orbits move away: $x^*$ is unstable.]

*Linearization.* Write $x_t = x^* + e_t$. Then $e_(t+1) = f(x^* + e_t) - x^* approx f'(x^*) e_t$. Near the fixed point
the error just gets multiplied by $f'(x^*)$ each step. The sign matters as in Chapter 5: positive means a steady
approach, negative means approaching while overshooting back and forth.

*Several variables.* The same holds with $f'(x^*)$ replaced by the Jacobian $J$: $e_(t+1) approx J e_t$, so
$e_t approx J^t e_0$, and $x^*$ is stable if $rho(J) < 1$. That's "the late states are locally attracting, with
$rho approx 0.78$".

*Worked example: the logistic map* $f(x) = r x (1 - x)$. This is the standard textbook example of the period-2
behavior we found.
- Fixed point: $x = r x (1 - x)$ gives $x^* = 1 - 1 \/ r$.
- Slope there: $f'(x) = r (1 - 2 x)$, so $f'(x^*) = r(1 - 2 + 2 \/ r) = 2 - r$.
- $r = 2.8$: slope $-0.8$, so the fixed point is stable and is approached with overshoot. Starting from 0.3, the orbit
  goes $0.588, 0.678, 0.611, 0.666, 0.623, 0.657, dots$ toward $0.643$.
- $r = 3$: slope exactly $-1$. Past this point the fixed point is unstable.
- $r = 3.2$: slope $-1.2$. The orbit settles into a *period-2 cycle* $0.5130 <-> 0.7995$, and the unstable fixed
  point $0.6875$ sits between the two cycle points, exactly as at Huginn's source tokens.

#thm("The period-2 cycle, exactly")[Points of period 2 solve $f(f(x)) = x$. Dividing out the fixed points leaves
$r^2 x^2 - r(r + 1) x + (r + 1) = 0$. Write $r = 3 + mu$ (so the slope at $x^*$ is $-(1 + mu)$). Then:
- the cycle exists for $mu > 0$, and the squared distance between its points is $(a - b)^2 = (r + 1)(r - 3) \/ r^2$,
  which is *proportional to $mu$* for small $mu$;
- its two-step multiplier is $(f compose f)'(a) = f'(a) f'(b) = -r^2 + 2 r + 4 = 1 - 4 mu - mu^2$.
At $r = 3.2$: $(a - b)^2 = 0.082$ and multiplier $0.16$.

This is the *flip* (period-doubling) bifurcation. D7 in `derivations.pdf` derives "squared amplitude $prop mu$" and
"multiplier $approx 1 - 4 mu$" for *any* smooth map at such a crossing. The logistic map shows those are the leading
terms of exact formulas. These are the two signatures you measured in Huginn.]

#thm("Damping (under-relaxation)")[Replace $f$ by $g(x) = (1 - beta) x + beta f(x)$ with $0 < beta <= 1$.
- Same fixed points: $g(x) = x$ exactly when $beta (f(x) - x) = 0$, i.e. $f(x) = x$.
- New slope: $g'(x^*) = 1 - beta + beta f'(x^*)$.
At $r = 3.2$ (slope $-1.2$), choosing $beta = 1.85 \/ 2.2 approx 0.84$ makes the new slope $-0.85$, so the fixed point
$0.6875$ becomes stable and the cycle disappears. Any $beta < 2 \/ 2.2 approx 0.91$ works. This is how the study
stopped blinking without changing answers.]

#q[For the logistic map with $r = 3.4$: what is the slope at $x^*$? Is $x^*$ stable? What $beta$ makes the damped
slope $-0.6$?]

= Putting it together: a toy loop

#why[This chapter is the whole study in two dimensions. Everything is exact and computable by hand.]

*The toy.* State $s in bb(R)^2$, loop $s_(t+1) = F(s_t) = J s_t + c$ with the running matrix $J = mat(0.1, 2; 0, 0.5)$
and $c = vec(1, 1)$. Readout $R(s) = s_1$, so $g = e_1$. (Huginn's $F$ is nonlinear; $J$ plays the role of its
Jacobian at the fixed point. Since this toy is already linear, the formulas below are exact rather than approximate.)

*Fixed point.* $x^* = J x^* + c$ means $(I - J) x^* = c$. Solve: $0.5 x_2 = 1$ gives $x_2 = 2$; $0.9 x_1 - 4 = 1$
gives $x_1 approx 5.56$. Since $rho(J) = 0.5 < 1$, every orbit converges to it.

*One-shot push.* Add $P$ once. The deviation evolves as $J^t P$ (Chapter 5) and dies away. The readout sees
$g^top J^t P$, the first entries of the table in Chapter 5.

*Persistent push.* Add $P$ before every step: $s_(t+1) = F(s_t + P) = J s_t + c + J P$. That is the same loop with $c$
replaced by $c + J P$, so the fixed point moves by
$ delta = (I - J)^(-1) J P quad "(the geometric series, Chapter 6)". $
#align(center, table(columns: 3, stroke: 0.4pt, inset: 6pt,
  [push $P$ (length 1)], [fixed point moves by $delta$], [readout change $g^top delta = delta_1$],
  [$e_1$ (along the readout)], [$(0.111, 0)$], [$0.111$],
  [$e_2$ (orthogonal to the readout)], [$(4.44, 1)$], [$4.44$],
))
Pushing the coordinate you read barely changes the reading; pushing the *other* coordinate changes it 40 times more.

*The lever.* The readout change is a linear functional of the push:
$ Delta R(P) = g^top (I - J)^(-1) J P = (a)^top P, quad a = ((I - J)^(-1) J)^top g. $
By Chapter 4, $a_i = Delta R(e_i)$: the entries of $a$ are exactly the right-hand column of the table, so
$a = (0.111, 4.44)$. By Chapter 3, the best unit push is $a \/ norm(a)$. It gains $norm(a) approx 4.45$, and any
other direction $u$ gets the fraction $cos(u, a)$.

*The probe.* Suppose two classes of inputs differ in the first entry of the context: $c_"True" - c_"False" = vec(1, 0)$.
Their fixed points differ by $(I - J)^(-1) vec(1, 0) = vec(1.11, 0)$. So the states of the two classes differ
*exactly along $e_1$*, and the readout separates them perfectly. A probe finds the direction $e_1$.

*The finding, in miniature.*
- The probe direction is $e_1$. As a push it achieves $0.111$, a fraction $cos(e_1, a) approx 0.025$ of the best.
- The lever points almost along $e_2$, nearly orthogonal to both the probe and the readout gradient.
- Reason: $J$ is non-normal. It carries coordinate 2 into coordinate 1 (the entry $2$), but not the other way.
  *Vectors* (states, probes) are pushed forward by $J$; *covectors* (readouts) are pulled back by $J^top$
  (Chapter 4). For a symmetric $J$ with positive eigenvalues the lever could never point against $g$ (`analytic.pdf`, Corollary 2a).
- The cheap one-step lever is $J^top g = vec(0.1, 2)$ (Chapter 7): already pointing mostly along $e_2$.

In Huginn: the probe-direction push moved the True/False log-odds by 0.25, the loop-aware lever by 8.1 at the same
length, and the two were nearly orthogonal. The toy shows the mechanism in its simplest form. What Huginn adds:
dimension 5280 instead of 2, a nonlinear $F$ (so everything holds near the fixed point), $J$ available only through
products $J v$ and $J^top w$, and many token positions coupled through attention.

#q[In the toy, compute the effect of a persistent push along $u = vec(1, 1) \/ sqrt(2)$ two ways: directly, and as
$norm(a) cos(u, a)$.]

= Answers

*Ch. 1.* $vec(3, -2) = 3 e_1 - 2 e_2$. Yes: $vec(2, 4) = 2 vec(1, 2)$.

*Ch. 2.* $J^2 = mat(0.01, 1.2; 0, 0.25)$. $J (J e_2) = J vec(2, 0.5) = vec(0.2 + 1, 0.25) = vec(1.2, 0.25)$, the
second column. ✓

*Ch. 3.* $0.1 \/ sqrt(0.01 + 4) approx 0.05$. Maximizer $u = (3, 4) \/ 5$, maximum $5$.

*Ch. 4.* $h = (J^2)^top e_1 = (0.01, 1.2)$. Directly, $J^2 e_2 = (1.2, 0.25)$, whose first entry is $1.2 = h_2$. ✓

*Ch. 5.* Eigenvalues $-1.2$ and $0.5$. The second coordinate shrinks as $0.5^t$; the first grows as $(-1.2)^t dot 0.01$,
flipping sign each step. After many steps the state is dominated by an exploding oscillation along $e_1$.

*Ch. 6.* $0.78 \/ 0.22 approx 3.55$. $0.78^(16) approx 0.019$.

*Ch. 7.* $R(F(x)) = x_1 x_2$, gradient $(x_2, x_1) = (2, 1)$. Via $J^top g$: $J = mat(2, 1; 1, 4)$, $g = e_1$,
$J^top e_1 = $ first row of $J$ $= (2, 1)$. ✓

*Ch. 8.* Slope $2 - 3.4 = -1.4$, unstable. $beta = (1 - (-0.6)) \/ (1 - (-1.4)) = 1.6 \/ 2.4 approx 0.667$.

*Ch. 9.* Directly: $Delta R(u) = (0.111 + 4.44) \/ sqrt(2) approx 3.22$. Via cosine: $cos(u, a) =
(0.111 + 4.44) \/ (sqrt(2) dot 4.45) approx 0.723$, and $4.45 times 0.723 approx 3.22$. ✓
