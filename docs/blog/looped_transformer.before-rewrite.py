# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="A looped transformer on eleven words",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np

    np.set_printoptions(precision=2, suppress=True)
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A looped transformer on eleven words

    A language model turns the text so far into a probability for every token that could come next. This notebook
    builds that computation in numpy, one concept at a time, assuming that eleven words are every token there is.
    With eleven tokens and two coordinates per vector, every matrix fits on the screen and every vector can be drawn.

    Parts 1 to 7 build an ordinary transformer: tokens, embeddings, positions, a probability measure over the
    tokens, attention, the MLP, and the whole forward pass. Parts 8 and 9 loop that model and push on it, which is
    what the experiments on Huginn and Recurrent-Llama in this repository measure.

    Everything computed here is a **toy**: the weights are set by hand, not trained. Paragraphs marked
    **Measured** report numbers from the experiments on trained models.

    The text below feeds every part. Change it (using only the eleven words *the, a, on, cat, dog, bird, mat, roof,
    sat, chased, saw*) and the cells that depend on it recompute.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sentence_box = mo.ui.text(value="the cat sat on the", label="Text so far:", full_width=True)
    sentence_box
    return (sentence_box,)


@app.cell(hide_code=True)
def _(ID, mo, sentence_box):
    _typed = sentence_box.value.lower().split()
    _unknown = [word for word in _typed if word not in ID]
    _usable = 2 <= len(_typed) <= 10 and not _unknown
    text = " ".join(_typed) if _usable else "the cat sat on the"
    (
        None
        if _usable
        else mo.callout(
            mo.md(
                (f"Not in the vocabulary: **{', '.join(_unknown)}**. " if _unknown else "Use two to ten words. ")
                + f"Using *{text}* instead."
            ),
            kind="warn",
        )
    )
    return (text,)


@app.cell(hide_code=True)
def _(E, VOCAB, np):
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.figure import Figure

    INK, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#dddcd6", "#fcfcfb"
    BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
    WORD_TINT, POS_TINT, PLAIN_TINT = "#dbe9fb", "#fbe2d6", "#eeede8"
    SEQ = LinearSegmentedColormap.from_list("seq", [SURFACE, "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    DIV = LinearSegmentedColormap.from_list("div", ["#c23b3a", "#f0efec", "#256abf"])

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 120,
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": RULE,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
        }
    )

    LABEL_SHIFT = {
        "the": (-0.08, 0.05, "right"),
        "a": (-0.08, 0.05, "right"),
        "on": (-0.08, 0.0, "right"),
        "saw": (-0.08, 0.0, "right"),
        "sat": (0.0, 0.13, "center"),
        "bird": (-0.08, 0.05, "right"),
        "chased": (0.08, -0.04, "left"),
    }

    def new_fig(ncols=1, width=9.0, height=3.6, **kw):
        """A figure that is not registered with pyplot, so redrawing never piles figures up."""
        fig = Figure(figsize=(width, height), layout="constrained")
        return fig, fig.subplots(1, ncols, **kw)

    def plane(ax, lim=1.6, labels=True, alpha=1.0, skip=()):
        """The two embedding coordinates as a plane, with the eleven words marked."""
        ax.axhline(0, color=RULE, lw=0.8, zorder=0)
        ax.axvline(0, color=RULE, lw=0.8, zorder=0)
        ax.scatter(E[:, 0], E[:, 1], s=22, color=MUTED, alpha=alpha, linewidths=0, zorder=2)
        if labels:
            for word, (x, y) in zip(VOCAB, E):
                if word in skip:
                    continue
                dx, dy, ha = LABEL_SHIFT.get(word, (0.08, 0.05, "left"))
                ax.text(x + dx, y + dy, word, fontsize=8.5, color=MUTED, ha=ha, va="center", alpha=alpha)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ticks = [-1, 0, 1] if lim < 2.2 else [-2, 0, 2]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(length=0, labelsize=8)
        for side in ax.spines.values():
            side.set_visible(False)

    def arrow(ax, start, end, color=INK, lw=1.8, head=11, **kw):
        ax.annotate(
            "",
            xy=tuple(end),
            xytext=tuple(start),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=head),
            **kw,
        )

    def matrix(ax, M, rows=None, cols=None, title=None, tint=PLAIN_TINT, cmap=None, vmin=None, vmax=None,
               fmt="{:.2f}", blank=None, fontsize=9):
        """A matrix drawn as a grid of numbers: one flat tint, or coloured by value when cmap is given."""
        M = np.atleast_2d(np.asarray(M, dtype=float))
        nr, nc = M.shape
        if cmap is None:
            rgb = np.tile(to_rgb(tint), (nr, nc, 1))
        else:
            lo = np.nanmin(M) if vmin is None else vmin
            hi = np.nanmax(M) if vmax is None else vmax
            rgb = cmap((M - lo) / max(hi - lo, 1e-12))[..., :3]
        if blank is not None:
            rgb = np.where(blank[..., None], np.array(to_rgb(SURFACE)), rgb)
        ax.imshow(rgb, aspect="equal")
        for r in range(nr):
            for c in range(nc):
                if blank is not None and blank[r, c]:
                    continue
                dark = rgb[r, c] @ np.array([0.299, 0.587, 0.114]) < 0.55
                ax.text(c, r, fmt.format(M[r, c]), ha="center", va="center", fontsize=fontsize,
                        color="white" if dark else INK)
        ax.set_xticks(np.arange(nc), cols if cols is not None else [""] * nc, fontsize=8.5)
        ax.set_yticks(np.arange(nr), rows if rows is not None else [""] * nr, fontsize=8.5)
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(nc + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(nr + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="both", length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        if title:
            ax.set_title(title, pad=8)

    def hbars(ax, values, title=None, color=BLUE, bad=None, fmt="{:.0%}", ticklabels=None):
        """One horizontal bar per vocabulary token, in vocabulary order."""
        values = np.asarray(values, dtype=float)
        shown = np.nan_to_num(values)
        y = np.arange(len(VOCAB))[::-1]
        colors = [RED if (bad is not None and bad[k]) else color for k in range(len(VOCAB))]
        ax.barh(y, shown, height=0.66, color=colors)
        span = max(np.abs(shown).max(), 1e-9)
        for yy, v, raw in zip(y, shown, values):
            label = fmt.format(raw) if np.isfinite(raw) else "undefined"
            ax.text(v + 0.02 * span * (1 if v >= 0 else -1), yy, label, va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=MUTED)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, [""] * len(VOCAB) if ticklabels is False else (ticklabels or VOCAB))
        ax.set_xticks([])
        ax.tick_params(length=0)
        ax.margins(x=0.34 if shown.min() < 0 else 0.2)
        for side in ax.spines.values():
            side.set_visible(False)
        if title:
            ax.set_title(title)

    def end_label(ax, x, y, label, color=INK, dx=6, dy=0):
        ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=9,
                    color=color)

    return (
        AQUA,
        BLUE,
        DIV,
        INK,
        LABEL_SHIFT,
        MUTED,
        ORANGE,
        POS_TINT,
        RULE,
        SEQ,
        SURFACE,
        WORD_TINT,
        arrow,
        end_label,
        hbars,
        matrix,
        mpl,
        new_fig,
        plane,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Tokens

    Text is split into **tokens**, items from a fixed list called the vocabulary. Real vocabularies have tens of
    thousands of entries: words, pieces of words, digits, punctuation. Ours has eleven whole words, and a text
    becomes a list of indices into it.
    """)
    return


@app.cell
def _(np):
    VOCAB = ["the", "a", "on", "cat", "dog", "bird", "mat", "roof", "sat", "chased", "saw"]
    ID = {word: i for i, word in enumerate(VOCAB)}

    def encode(text):
        """Text to token ids: one integer per word."""
        return np.array([ID[word] for word in text.split()])

    return ID, VOCAB, encode


@app.cell
def _(encode, text):
    words = text.split()
    ids = encode(text)
    n = len(ids)  # number of positions
    ids
    return ids, n, words


@app.cell(hide_code=True)
def _(ids, mo, n, words):
    row_labels = [f"{_i + 1} {_w}" for _i, _w in enumerate(words)]
    mo.md(
        "| position | " + " | ".join(str(_i + 1) for _i in range(n)) + " |\n"
        + "|---|" + "---|" * n + "\n"
        + "| token | " + " | ".join(words) + " |\n"
        + "| id | " + " | ".join(str(_k) for _k in ids) + " |"
    )
    return (row_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Embeddings: a vector for every token

    A model computes with numbers, so every token gets a vector, its **embedding**. The model stores them as the
    rows of a table `E`, one row per vocabulary token, and looking a token up returns its row. Every row has the
    same length $d$: 2 here, 5,280 in Huginn.

    Training learns the numbers in the table. It tends to put tokens that are used alike close together, because
    that helps predict what comes next; the coordinates themselves have no meaning fixed in advance. Here the
    numbers are placed by hand to imitate that.
    """)
    return


@app.cell
def _(ID, np):
    E = np.array([
        [-1.00,  0.40],  # the
        [-0.70,  0.75],  # a
        [-0.85, -0.05],  # on
        [ 0.75,  0.75],  # cat
        [ 1.05,  0.50],  # dog
        [ 0.50,  1.05],  # bird
        [ 1.00, -0.45],  # mat
        [ 0.85, -0.80],  # roof
        [-0.20, -1.10],  # sat
        [ 0.05, -1.20],  # chased
        [-0.60, -0.95],  # saw
    ])
    E[ID["cat"]]  # looking a token up returns its row
    return (E,)


@app.cell(hide_code=True)
def _(BLUE, E, MUTED, SURFACE, VOCAB, WORD_TINT, ids, matrix, new_fig, plane):
    _fig, (_ax, _tab) = new_fig(2, width=9, height=4.4, width_ratios=[1.7, 1])
    plane(_ax)
    _in_text = sorted(set(ids))
    _ax.scatter(E[_in_text, 0], E[_in_text, 1], s=70, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
    for _name, _xy in {"function words": (-1.5, 1.05), "animals": (0.95, 1.3), "things": (1.15, -0.12),
                       "verbs": (-0.75, -1.4)}.items():
        _ax.text(*_xy, _name, fontsize=8.5, style="italic", color=MUTED, alpha=0.75)
    _ax.set_title("Each dot is one row of E (blue: tokens in the text)")
    matrix(_tab, E, rows=VOCAB, cols=["1", "2"], tint=WORD_TINT, title="The table E")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The group names in the figure are ours; the model has no such labels. What it has is distance, and distance can
    be computed:
    """)
    return


@app.cell
def _(E, ID, VOCAB, np):
    def nearest(word, k=2):
        """The k tokens whose embeddings are closest to this word's."""
        distance = np.linalg.norm(E - E[ID[word]], axis=1)
        return [VOCAB[j] for j in np.argsort(distance)[1 : k + 1]]

    {word: nearest(word) for word in ["cat", "mat", "sat", "the"]}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · From a sentence to a matrix

    Look up each token of the text and stack the rows in order:
    """)
    return


@app.cell
def _(E, ids):
    E[ids]  # n rows, one per position
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two things are missing. A token that occurs twice gives two identical rows. And nothing in the numbers records
    the order: shuffling the words only shuffles the rows, and the layers that come next treat every row alike. As
    far as the numbers go, a text is a bag of tokens.
    """)
    return


@app.cell
def _(E, encode):
    def same_rows(a, b):
        return sorted(map(tuple, a)) == sorted(map(tuple, b))

    same_rows(E[encode("the cat sat on the mat")], E[encode("the mat sat on the cat")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Position offsets

    So each row has to carry its position in its own numbers. A counter (1, 2, 3, …) in an extra column fails in
    one of two ways. Left alone, it reads 1,000 at position 1,000 beside word coordinates of about ±1, and the word
    becomes a rounding error. Divided by a maximum length, it turns positions 500 and 501 into 0.500 and 0.501,
    which can hardly be told apart.

    What works is an arrow of fixed length that turns by a fixed angle at every position:

    $$p_i = r\,(\cos i\omega,\ \sin i\omega)$$

    Every arrow has the same length, so it never swamps the word, and neighbours always differ by the same angle
    wherever they are in the text. The toy uses $\omega = 45°$ and $r = 0.15$. Models usually *add* the offset to
    the embedding instead of appending columns, so every row keeps its length $d$. Row $i$ of the matrix is

    $$x_i = e_{w_i} + p_i$$

    where $w_i$ is the token at position $i$.
    """)
    return


@app.cell
def _(E, encode, np):
    OMEGA = np.pi / 4  # 45° per position

    def offsets(n):
        """One arrow per position: all the same length, each turned 45° from the one before."""
        i = np.arange(1, n + 1)
        return 0.15 * np.column_stack([np.cos(i * OMEGA), np.sin(i * OMEGA)])

    def embed(text):
        """Text to matrix: one row per position, embedding plus offset."""
        ids = encode(text)
        return E[ids] + offsets(len(ids))

    return OMEGA, embed, offsets


@app.cell
def _(E, ids, n, offsets):
    P = offsets(n)
    X = E[ids] + P  # the same as embed(text)
    X
    return P, X


@app.cell(hide_code=True)
def _(
    BLUE,
    E,
    INK,
    ORANGE,
    P,
    POS_TINT,
    SURFACE,
    WORD_TINT,
    X,
    arrow,
    ids,
    matrix,
    n,
    new_fig,
    plane,
    row_labels,
    words,
):
    _fig, (_ax, _m1, _m2, _m3) = new_fig(4, width=9.6, height=3.6, width_ratios=[2.5, 1, 1, 1])
    plane(_ax, alpha=0.45, skip=words)
    for _i in range(n):
        _ax.plot(*E[ids[_i]], "o", ms=5, mfc=SURFACE, mec=BLUE, mew=1.2, zorder=3)
        arrow(_ax, E[ids[_i]], X[_i], color=ORANGE, lw=1.6, head=8, zorder=4)
        _out = P[_i] / 0.15
        _ax.text(*(X[_i] + 0.09 * _out), row_labels[_i], fontsize=8, color=INK, zorder=5,
                 ha="left" if _out[0] > 0.3 else "right" if _out[0] < -0.3 else "center",
                 va="bottom" if _out[1] > 0.3 else "top" if _out[1] < -0.3 else "center")
    _ax.set_title("Each word (circle) is shifted by its position's arrow")
    matrix(_m1, E[ids], rows=row_labels, cols=["1", "2"], tint=WORD_TINT, title="E[ids]")
    matrix(_m2, P, rows=[""] * n, cols=["1", "2"], tint=POS_TINT, title="+ P")
    matrix(_m3, X, rows=[""] * n, cols=["1", "2"], title="= X")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `X` has one row per position and one column per coordinate. A longer text adds rows, never columns. The blue
    block depends only on which tokens occur; the orange block would be the same for any text of this length; their
    sum tells two copies of the same word apart.

    ### Why arrows on a circle

    Turning an arrow by $\omega$ is multiplication by one fixed matrix $R$, and it moves every position's arrow onto
    the next one: $R\,p_j = p_{j+1}$. So "the position just before me" is the same computation everywhere in the
    text: turn every arrow one step and see which one lands on mine. For arrows of length 1 the match is

    $$\hat p_i^{\top} R\,\hat p_j = \cos\big((i - j - 1)\,\omega\big)$$

    which is largest, 1, exactly when $j = i - 1$. Attention (part 5) compares rows with this kind of product.
    """)
    return


@app.cell
def _(OMEGA, P, np):
    def rot(angle):
        """The matrix that turns a vector anticlockwise by this angle."""
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    R = rot(OMEGA)
    unit_arrows = P / 0.15
    match = unit_arrows @ R @ unit_arrows.T  # match[i, j]: arrow i against arrow j turned one step

    np.allclose(P[:-1] @ R.T, P[1:])  # one fixed matrix steps every position's arrow to the next
    return match, rot, unit_arrows


@app.cell(hide_code=True)
def _(DIV, ORANGE, RULE, arrow, match, matrix, n, new_fig, np, unit_arrows):
    _fig, (_ax, _m) = new_fig(2, width=8.4, height=3.6, width_ratios=[1, 1.15])
    _circle = np.linspace(0, 2 * np.pi, 200)
    _ax.plot(np.cos(_circle), np.sin(_circle), color=RULE, lw=1)
    for _i in range(n):
        arrow(_ax, (0, 0), unit_arrows[_i], color=ORANGE, lw=1.6, head=10)
        _ax.text(*(1.17 * unit_arrows[_i]), str(_i + 1), ha="center", va="center", fontsize=9)
    _ax.set_xlim(-1.4, 1.4)
    _ax.set_ylim(-1.4, 1.4)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title("The position arrows, numbered")
    _nums = [str(_i + 1) for _i in range(n)]
    matrix(_m, match, rows=_nums, cols=_nums, cmap=DIV, vmin=-1, vmax=1, fontsize=8,
           title="Arrow i (down) against\narrow j turned one step (across)")
    _fig
    return


@app.cell(hide_code=True)
def _(BLUE, INK, RULE, mo, new_fig, np):
    _i = 100
    _j = np.arange(1, 201)
    _periods = [8, 27, 200]

    def _match(periods):
        return np.mean([np.cos((_i - _j) * 2 * np.pi / period) for period in periods], axis=0)

    _fig, (_one, _three) = new_fig(2, width=9, height=2.6, sharey=True)
    for _ax, _vals, _title in [(_one, _match(_periods[:1]), "One circle, 45° per step"),
                               (_three, _match(_periods), "Three circles: periods 8, 27 and 200")]:
        _ax.plot(_j, _vals, color=BLUE, lw=1.3)
        _ax.axhline(0, color=RULE, lw=0.8)
        _ax.plot([_i], [1], "o", ms=6, color=INK)
        _ax.set_title(_title)
        _ax.set_xlabel("position j")
        _ax.set_ylim(-1.1, 1.15)
        _ax.tick_params(length=0)
    _one.set_ylabel("match with position 100")
    mo.accordion(
        {
            "Aside: longer texts need many circles": mo.vstack(
                [
                    mo.md(
                        r"""
                        One circle is not enough for a long text. With 45° steps the arrow comes back around after
                        eight positions, so position 9 lands exactly on position 1. A smaller step avoids that but
                        makes neighbours nearly identical.

                        Real encodings use many circles at once, each in its own pair of coordinates and each
                        turning at its own speed, like the hands of a clock: a fast hand separates neighbours, a
                        slow one keeps distant positions apart. The charts show how well position 100 matches every
                        other position, averaged over the circles in use. With one circle, every eighth position is
                        a perfect look-alike; with three, only position 100 itself scores 1.

                        The original transformer uses $d/2$ circles with speeds $10000^{-2k/d}$ radians per step.
                        Huginn and Recurrent-Llama apply the same rotations inside attention instead of adding
                        offsets (part 5).
                        """
                    ),
                    _fig,
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · A probability measure over the tokens

    The model does not name one next token. Given the text so far, the **history** $w_1, \dots, w_n$, it gives
    every token $v$ a probability,

    $$p(v \mid w_1, \dots, w_n)$$

    read "the probability that the next token is $v$, given that the history is $w_1 \dots w_n$". For our text that
    is eleven numbers, one of them $p(\text{mat} \mid \text{the cat sat on the})$.

    In the language of probability, the next token is the outcome of a random experiment, and the set of possible
    outcomes is its **sample space**. Name each token by its id from part 1, so the sample space is

    $$\mathcal V = \{0, 1, \dots, 10\}$$

    (Python counts from 0.) A probability measure on a finite set is fixed by one number per element, each at least
    0, adding up to 1. The probability of an **event**, any subset $A \subseteq \mathcal V$ such as "the next token
    is an animal" $= \{3, 4, 5\}$, is the sum over its members:

    $$P(A) = \sum_{v \in A} p(v)$$

    So checking a proposed measure means checking two things: nothing negative, total 1.
    """)
    return


@app.cell
def _(ID, np):
    def is_measure(p):
        return bool(np.all(p >= 0) and np.isclose(p.sum(), 1.0))

    def prob(event, p):
        """P(A) for an event A, a set of tokens: add up its members."""
        return float(p[[ID[word] for word in event]].sum())

    return is_measure, prob


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The shape of the model

    A language model of this kind is two maps, one after the other:

    $$
    \underbrace{\mathcal V^*}_{\text{histories}}
    \xrightarrow{\;f_\theta\;}
    \underbrace{\mathbb R^{|\mathcal V|}}_{\text{logits}}
    \xrightarrow{\;\operatorname{softmax}\;}
    \underbrace{\Delta^{|\mathcal V|-1}}_{\text{probability distributions}}
    $$

    - $\mathcal V^*$ is the set of all finite sequences of tokens: every possible history.
    - $f_\theta$ is the learned part. It turns a history into a **logit vector**: $|\mathcal V| = 11$ real numbers,
      one for each token, with no restriction on their sign or size. $\theta$ stands for every adjustable number
      inside it. Parts 5 to 7 all happen inside $f_\theta$.
    - $\operatorname{softmax}$ is fixed; it has no adjustable numbers. It turns any logit vector into a point of
      $\Delta^{|\mathcal V|-1}$, the set of all probability distributions on $\mathcal V$: eleven numbers, each at
      least 0, adding up to 1. (The exponent is $|\mathcal V| - 1$ because the last number is fixed by the other
      ten.)

    The next token is then drawn from that distribution:

    $$W_{n+1} \sim \operatorname{Categorical}\big(\operatorname{softmax}(z)\big), \qquad z = f_\theta(w_1, \dots, w_n)$$

    So the logit vector $z$ determines a probability measure on $\mathcal V$, but $z$ is not what gets sampled. A
    token is.

    ### The learned function, so far

    There is not much inside $f_\theta$ yet. The history becomes the matrix of part 3. The next token follows the
    last position, so the last row of the matrix is read. Each token is then scored by the inner product of its
    embedding with that row:

    $$z_v = e_v^{\top} x_n \qquad \text{for each } v \in \mathcal V$$

    A token scores high when the last row points the same way as its embedding. (Scoring against the embedding table
    itself is what GPT-2 does; many models learn a separate table for this.) For now $\theta$ is just the table `E`.
    """)
    return


@app.cell
def _(E, np, offsets):
    def f_theta(history):
        """The learned part of the model: a history (token ids) to one logit per token."""
        ids = np.array(history)
        X = E[ids] + offsets(len(ids))  # the history as a matrix (part 3)
        h = X[-1]  # its last row
        return E @ h  # z[v] = E[v] · h, for every token id v at once

    return (f_theta,)


@app.cell
def _(X, f_theta, ids):
    h = X[-1]  # the last row for our text, kept for the figures below
    z = f_theta(ids)
    z
    return h, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output has exactly one entry for each element of the sample space. As a function it is
    $z : \mathcal V \to \mathbb{R}$, $v \mapsto z_v$; in code, an array indexed by token id.

    ### Softmax: from logits to a measure

    The logits are indexed by the right set, but they are not probabilities: some are negative, and they add up to
    nothing in particular. **Softmax** repairs both in two moves. Exponentiate every logit, which makes it positive.
    Then divide by the total, which makes the results add up to 1:

    $$p(v \mid w_1, \dots, w_n) = \frac{e^{z_v}}{\sum_{u \in \mathcal V} e^{z_u}}$$
    """)
    return


@app.cell
def _(is_measure, np, z):
    positive = np.exp(z)  # move 1: every entry is now above 0
    total = positive.sum()  # one number
    p = positive / total  # move 2: the entries now add up to 1

    is_measure(p)
    return p, positive, total


@app.cell(hide_code=True)
def _(VOCAB, mo, p, positive, text, total, z):
    _rows = "\n".join(
        f"| {_v} | {VOCAB[_v]} | {z[_v]:+.2f} | {positive[_v]:.2f} | {p[_v]:.1%} |" for _v in range(len(VOCAB))
    )
    mo.md(
        r"| id $v$ | token | logit $z_v$ | $e^{z_v}$ | $p(v \mid \text{history})$ |" + "\n|---|---|---|---|---|\n"
        + _rows
        + f"\n| **total** | | {z.sum():+.2f} | {total:.2f} | {p.sum():.1%} |"
        + f"\n\nThe history is *{text}*. Each probability is its row's $e^{{z_v}}$ divided by {total:.2f}."
    )
    return


@app.cell(hide_code=True)
def _(
    INK,
    MUTED,
    VOCAB,
    arrow,
    h,
    hbars,
    new_fig,
    np,
    p,
    plane,
    positive,
    total,
    z,
):
    _labels = [f"{_k}  {_w}" for _k, _w in enumerate(VOCAB)]
    _lim = max(1.6, float(np.abs(h).max()) + 0.3)
    _fig, (_ax, _bz, _be, _bp) = new_fig(4, width=9.8, height=4.1, width_ratios=[1.55, 1, 1, 1])
    plane(_ax, lim=_lim)
    if np.linalg.norm(h) > 0:
        _perp = np.array([-h[1], h[0]]) / np.linalg.norm(h) * 3 * _lim
        _ax.plot([-_perp[0], _perp[0]], [-_perp[1], _perp[1]], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        arrow(_ax, (0, 0), h, color=INK, lw=2)
    _ax.set_title("Last row (arrow), and where the logit is 0")
    hbars(_bz, z, ticklabels=_labels, color=MUTED, fmt="{:+.2f}", title="logits z")
    _bz.set_xlabel(f"total {z.sum():+.2f}")
    hbars(_be, positive, ticklabels=False, color="#86b6ef", fmt="{:.2f}", title="exponentiate")
    _be.set_xlabel(f"total {total:.2f}")
    hbars(_bp, p, ticklabels=False, title="divide by the total")
    _bp.set_xlabel("total 100%")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Softmax takes a function $\mathcal V \to \mathbb{R}$ and returns a function $\mathcal V \to (0, 1)$ whose values
    add up to 1. That is a probability measure on $\mathcal V$ for any logits whatsoever, trained or not. Only gaps
    between logits matter, $p(a)/p(b) = e^{z_a - z_b}$: each extra unit of gap multiplies the odds by
    $e \approx 2.7$.

    As a function, for the rest of the notebook. Subtracting the largest logit first changes nothing after the
    division; it only avoids overflow.
    """)
    return


@app.cell
def _(f_theta, ids, np, p):
    def softmax(z):
        positive = np.exp(z - z.max(axis=-1, keepdims=True))
        return positive / positive.sum(axis=-1, keepdims=True)

    np.allclose(softmax(f_theta(ids)), p)
    return (softmax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sampling the next token

    For one history the model's output is the whole measure. Writing text means drawing a token from it. Drawing
    many times for the same history shows the measure at work: the share of draws that land in an event approaches
    the event's probability.
    """)
    return


@app.cell
def _(ID, VOCAB, np, p, prob):
    rng = np.random.default_rng(0)
    draws = rng.choice(len(VOCAB), size=10_000, p=p)  # the next token, drawn 10,000 times for the same history
    animals = [ID[word] for word in ("cat", "dog", "bird")]

    {
        "first draw": VOCAB[draws[0]],
        "P(next token is an animal), summed from p": round(prob({"cat", "dog", "bird"}, p), 3),
        "share of animals among the draws": float(np.isin(draws, animals).mean()),
    }
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why exponentiate?

    Two simpler ways of turning the same logits into numbers that add up to 1:
    """)
    return


@app.cell
def _(np):
    def divide_by_total(z):
        return z / z.sum()

    def clip_then_divide(z):
        z = np.maximum(z, 0)
        return z / z.sum()

    return clip_then_divide, divide_by_total


@app.cell(hide_code=True)
def _(
    VOCAB,
    clip_then_divide,
    divide_by_total,
    hbars,
    new_fig,
    np,
    softmax,
    z,
):
    _labels = [f"{_k}  {_w}" for _k, _w in enumerate(VOCAB)]
    _fig, _axes = new_fig(3, width=9.6, height=3.9, width_ratios=[1.25, 1, 1])
    for _ax, (_name, _recipe) in zip(
        _axes, [("divide by the total", divide_by_total), ("clip, then divide", clip_then_divide), ("softmax", softmax)]
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            _values = _recipe(z)
        _bad = ~np.isfinite(_values) | (_values < 0)
        _zeros = int(np.sum(_values == 0))
        if _bad.any():
            _verdict = "not a measure"
        elif _zeros:
            _verdict = f"a measure, but {_zeros} tokens get exactly 0"
        else:
            _verdict = "a measure, every token above 0"
        hbars(_ax, _values, bad=_bad, title=_name, ticklabels=_labels if _ax is _axes[0] else False)
        _ax.set_xlabel(_verdict)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **Divide by the total.** The results add up to 1, but a negative logit becomes a negative "probability" (or,
      if the total is negative, a positive one becomes negative), and a total near 0 blows everything up. Red bars
      are not valid probabilities.
    - **Clip negatives to 0, then divide.** Valid whenever some logit is positive. But every token with a negative
      logit gets exactly 0, and a token given probability 0 that then occurs costs an infinite loss (below).
      Doubling every logit also changes nothing, so the model has no way to say how sure it is.
    - **Softmax.** $e^{z}$ is positive for every $z$ and never exactly 0, so no token is ruled out, and doubling the
      logits squares the odds.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Which measure should it be?

    Softmax guarantees *a* measure. Which one is right is a question about text. Find every place in a large body
    of text where *the cat sat on the* occurs. Say there are 1,000, of which 700 are followed by "mat" and 300 by
    "roof". Those frequencies, $f$, are the measure to aim for.

    Training charges the model a **loss** of $-\log p(w^*)$ at each place, where $w^*$ is the token that actually
    came next: 0 if the model gave it probability 1, growing without limit as that probability falls to 0. Averaged
    over the places, the loss is

    $$L(p) = -\sum_{v \in \mathcal V} f_v \log p(v)$$

    and it is smallest exactly when $p = f$ (Gibbs' inequality). Training adjusts $\theta$ to lower the loss, which
    pushes the model's measure toward the counted one.

    Here is training in miniature. To keep it to two numbers, let the descent move the last row $h$ directly, as
    if it were the only adjustable part. The slope of $L$ with respect to the logits is $p - f$, so with respect to
    $h$ it is $E^{\top}(p - f)$.
    """)
    return


@app.cell
def _(E, ID, VOCAB, h, np, softmax):
    f = np.zeros(len(VOCAB))
    f[ID["mat"]], f[ID["roof"]] = 0.7, 0.3  # counted frequencies after "the cat sat on the"

    def loss(p):
        return float(-np.sum(f[f > 0] * np.log(p[f > 0])))

    h_fit = h.copy()
    history = []
    for _ in range(3000):
        p_fit = softmax(E @ h_fit)
        history.append(loss(p_fit))
        h_fit = h_fit - 0.5 * E.T @ (p_fit - f)  # one step downhill

    h_fit, round(history[0], 3), round(history[-1], 3)
    return h_fit, history, p_fit


@app.cell(hide_code=True)
def _(BLUE, ID, INK, MUTED, history, new_fig, np, p, p_fit):
    _fig, (_curve, _steps, _cmp) = new_fig(3, width=9.6, height=3.1)
    _grid = np.linspace(0.02, 0.98, 200)
    _curve.plot(_grid, -0.7 * np.log(_grid) - 0.3 * np.log(1 - _grid), color=BLUE, lw=2)
    _floor = -0.7 * np.log(0.7) - 0.3 * np.log(0.3)
    _curve.plot([0.7], [_floor], "o", ms=7, color=INK)
    _curve.annotate("lowest at p = 0.7", (0.7, _floor), xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=9)
    _curve.set_ylim(0.5, 2.2)
    _curve.set_xlabel('probability the model gives "mat"')
    _curve.set_title('Average loss against p("mat")')

    _steps.plot(np.arange(1, len(history) + 1), history, color=BLUE, lw=2)
    _steps.axhline(_floor, color=MUTED, lw=1, ls=(0, (4, 3)))
    _steps.text(len(history), _floor - 0.05, "loss of the counted measure", ha="right", va="top", fontsize=8.5,
                color=MUTED)
    _steps.set_ylim(0.35, None)
    _steps.set_xscale("log")
    _steps.set_xlabel("step")
    _steps.set_title("Loss while h moves downhill")

    _groups = ["mat", "roof", "the other nine"]
    _series = [
        ("counted", MUTED, [0.7, 0.3, 0.0]),
        ("model before", "#86b6ef", [p[ID["mat"]], p[ID["roof"]], 1 - p[ID["mat"]] - p[ID["roof"]]]),
        ("model after", BLUE, [p_fit[ID["mat"]], p_fit[ID["roof"]], 1 - p_fit[ID["mat"]] - p_fit[ID["roof"]]]),
    ]
    _pos = np.arange(3)
    for _s, (_name, _color, _vals) in enumerate(_series):
        _cmp.bar(_pos + 0.27 * (_s - 1), _vals, width=0.25, color=_color, label=_name)
        for _k in range(3):
            _cmp.text(_pos[_k] + 0.27 * (_s - 1), _vals[_k] + 0.015, f"{_vals[_k]:.0%}", ha="center", fontsize=7.5,
                      color=MUTED)
    _cmp.set_xticks(_pos, _groups)
    _cmp.set_yticks([])
    _cmp.set_ylim(0, 1.0)
    _cmp.spines["left"].set_visible(False)
    _cmp.legend(frameon=False, fontsize=8.5, loc="upper center", handlelength=1, borderaxespad=0.2)
    _cmp.set_title("Probability of the next token")
    for _ax in (_curve, _steps, _cmp):
        _ax.tick_params(length=0)
    _fig
    return


@app.cell(hide_code=True)
def _(ID, h, h_fit, mo, p_fit):
    mo.md(f"""
    The descent moved $h$ from ({h[0]:.2f}, {h[1]:.2f}) to ({h_fit[0]:.1f}, {h_fit[1]:.1f}), and the model now
    gives "mat" {p_fit[ID["mat"]]:.0%} and "roof" {p_fit[ID["roof"]]:.0%}. Two numbers cannot reach 70 and 30
    exactly, but they get close.

    Real training does the same descent on all of $\\theta$ at once (the table `E` and every weight in the layers
    below), over all the text, so that the right logits come out for each history. That is the part this notebook
    skips: from here on the weights are set by hand.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two different distributions

    There are two probability distributions in this picture, and they live on different spaces.

    1. **Over tokens.** For one fixed history, $\operatorname{softmax}(z)$ is a distribution on $\mathcal V$: which
       token comes next. That is everything above.
    2. **Over logit vectors.** If the history is itself random, for example because the model wrote it one token at
       a time, then the logit vector $Z = f_\theta(W_1, \dots, W_n)$ is random too. Its possible values are the image
       of $f_\theta$,

       $$\mathcal Z_\theta = \{\, f_\theta(w) : w \in \mathcal V^* \,\} \subseteq \mathbb R^{|\mathcal V|}$$

       and its distribution is the **pushforward** of the distribution over histories,
       $\mu_Z = (f_\theta)_{\#}\,\mu_W$. In words: the probability of a logit vector is the total probability of
       the histories that produce it.

    Softmax is applied to one logit vector at a time, never across $\mathcal Z_\theta$.

    With eleven tokens, both can be computed exactly. Let the model write three tokens, the first uniformly at
    random and each later one drawn from its own softmax, and collect the logit vector it would use for the fourth:
    """)
    return


@app.cell
def _(VOCAB, f_theta, np, softmax):
    def history_measure(f, length):
        """The probability of every history of this length when the model writes it itself (first token uniform)."""
        mu = {(v,): 1 / len(VOCAB) for v in range(len(VOCAB))}
        for _ in range(length - 1):
            mu = {w + (v,): m * q for w, m in mu.items() for v, q in enumerate(softmax(f(w)))}
        return mu

    def pushforward(f, mu):
        """The measure on logit vectors induced by a measure on histories."""
        mu_Z = {}
        for w, m in mu.items():
            z = tuple(np.round(f(w), 9))
            mu_Z[z] = mu_Z.get(z, 0.0) + m  # histories that give the same logits pool their probability
        return mu_Z

    mu = history_measure(f_theta, 3)
    mu_Z = pushforward(f_theta, mu)

    {"histories": len(mu), "distinct logit vectors": len(mu_Z), "total probability": round(float(sum(mu_Z.values())), 6)}
    return history_measure, mu, mu_Z, pushforward


@app.cell(hide_code=True)
def _(
    BLUE,
    E,
    LABEL_SHIFT,
    MUTED,
    SURFACE,
    VOCAB,
    f_theta,
    mo,
    mu,
    mu_Z,
    new_fig,
    np,
    plane,
):
    def atoms(ax, measure, scale=4500, names=None):
        """Draw a measure on logit vectors: one dot per vector, at the last row it comes from, area = probability."""
        logits = np.array(list(measure))
        mass = np.array(list(measure.values()))
        rows = logits @ np.linalg.pinv(E).T  # z = E h has exactly one solution h, because E has rank 2
        plane(ax, lim=max(1.6, float(np.abs(rows).max()) + 0.25), labels=False, alpha=0.25)
        ax.scatter(rows[:, 0], rows[:, 1], s=scale * mass, color=BLUE, alpha=0.6, edgecolors=SURFACE,
                   linewidths=0.4, zorder=3)
        if names:
            for row, key, m in zip(rows, measure, mass):
                dx, dy, ha = LABEL_SHIFT.get(names[key], (0.08, 0.05, "left"))
                ax.text(row[0] + 2.2 * dx, row[1] + 2.2 * dy, f"{names[key]} {m:.0%}", ha=ha, va="center",
                        fontsize=8.5, color=MUTED)

    _names = {tuple(np.round(f_theta(_w), 9)): VOCAB[_w[-1]] for _w in mu}
    _fig, _ax = new_fig(1, width=5.6, height=4.6)
    atoms(_ax, mu_Z, names=_names)
    _ax.set_title("The distribution of Z after three tokens")
    mo.vstack(
        [
            _fig,
            mo.md(
                "Each dot is one possible logit vector, drawn at the last row it comes from (a logit vector is "
                "`E @ h`, and `E` has rank 2, so the row can be recovered from it). The dot's area is its "
                "probability, and the label is the last token of the histories that land there. The small grey dots "
                "are the eleven embeddings, for reference."
            ),
        ]
    )
    return (atoms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are 1,331 histories of three tokens, but only 11 possible logit vectors: one for each last token. That
    is the limit of $f_\theta$ as it stands. It reads only the last row of the matrix, and that row was built from
    one token and one position. The measure is written $p(v \mid w_1, \dots, w_n)$, but so far it depends on much
    less than that: any history of the same length that ends in the same token gets the same logits, and so the
    same measure.
    """)
    return


@app.cell
def _(encode, f_theta, ids, n, np, words):
    other = " ".join(["a"] * (n - 1) + words[-1:])  # same length and same last token, different beginning

    other, np.allclose(f_theta(ids), f_theta(encode(other)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Attention: rows gather from earlier rows

    What comes next depends on the whole text, so before the last row is read, information has to move between rows.
    That is the job of **attention**, the only step in which positions talk to each other. A layer never overwrites
    the matrix; it adds a correction, $X \mapsto X + \Delta X$, so whatever was written earlier stays available.

    One attention **head** gives every row three vectors, each a learned linear map of the row:

    - a **query**: what this row is looking for;
    - a **key**: the label this row offers to queries;
    - a **value**: the message this row sends if it is read.

    Row $i$'s weight on row $j$ comes from the inner product of $i$'s query with $j$'s key. Softmax along each row
    turns those scores into weights that are positive and add up to 1. (This is softmax's second job. Here it
    decides where a row reads; in part 4 it decided what the model might write.) A language model predicts left to
    right, so row $i$ may only read rows $j \le i$: the **causal mask**. Row $i$ then receives the weighted average
    of the values, mapped back to $d$ coordinates by a fourth matrix.
    """)
    return


@app.cell
def _(np, softmax):
    def attention(X, WQ, WK, WV, WO, causal=True):
        Q, K, V = X @ WQ, X @ WK, X @ WV  # a query, a key and a value for every row
        S = Q @ K.T / np.sqrt(K.shape[1])  # S[i, j]: how well row i's query matches row j's key
        if causal:
            S = np.where(np.tri(len(X)) > 0, S, -np.inf)  # row i may only read rows j ≤ i
        A = softmax(S)  # each row: weights that are ≥ 0 and add up to 1
        return A, A @ V @ WO  # who reads whom, and the update each row receives

    return (attention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So a head is two things. The matrix $A(X)$ says who reads whom: it is $n \times n$, lower-triangular, each row
    adds up to 1, and it depends on $X$, so the reading pattern changes with the state. A fixed linear map says what
    is done with what was read: $\Delta X = A(X)\,X\,W_V W_O$.

    In the toy head, a row's key and value are the row itself, and its query is the row turned by an angle $\theta$
    and scaled by $\beta$. The angle decides who counts as a match; $\beta$ decides how sharply the best match wins.
    """)
    return


@app.cell
def _(np, rot):
    def toy_head(beta=2.5, theta=np.pi / 4):
        WQ = beta * rot(theta).T  # query: the row, turned by theta and scaled by beta
        WK = np.eye(2)  # key: the row itself
        WV = np.eye(2)  # value: the row itself
        WO = 0.3 * np.eye(2)  # write back 0.3 of what was gathered
        return WQ, WK, WV, WO

    return (toy_head,)


@app.cell(hide_code=True)
def _(mo, n):
    beta = mo.ui.slider(0, 8, step=0.1, value=2.5, label="Sharpness β", show_value=True)
    theta = mo.ui.slider(-180, 180, step=5, value=45, label="Query rotation θ (degrees)", show_value=True)
    looker = mo.ui.slider(1, n, step=1, value=n, label="Row doing the looking", show_value=True)
    causal = mo.ui.checkbox(value=True, label="Only look backwards")
    mo.hstack([beta, theta, looker, causal], justify="start", gap=1.5, wrap=True)
    return beta, causal, looker, theta


@app.cell
def _(X, attention, beta, causal, np, theta, toy_head):
    A, dX = attention(X, *toy_head(beta.value, np.deg2rad(theta.value)), causal=causal.value)
    A.sum(axis=1)  # every row of weights adds up to 1
    return A, dX


@app.cell(hide_code=True)
def _(
    A,
    AQUA,
    BLUE,
    INK,
    MUTED,
    ORANGE,
    SEQ,
    SURFACE,
    X,
    arrow,
    causal,
    dX,
    looker,
    matrix,
    mpl,
    n,
    new_fig,
    np,
    plane,
    rot,
    row_labels,
    theta,
):
    _i = looker.value - 1
    _fig, (_m, _ax) = new_fig(2, width=9.4, height=4.3, width_ratios=[1, 1.1])
    matrix(_m, A, rows=row_labels, cols=[str(_k + 1) for _k in range(n)], cmap=SEQ, vmin=0, vmax=1,
           blank=(np.tri(n) == 0) if causal.value else None, fontsize=8,
           title="A: the row that reads (down), the row that is read (across)")
    _m.add_patch(mpl.patches.Rectangle((-0.5, _i - 0.5), n, 1, fill=False, ec=INK, lw=1.6, zorder=5))

    plane(_ax, labels=False, alpha=0.25)
    _query = rot(np.deg2rad(theta.value)) @ X[_i]
    arrow(_ax, (0, 0), 1.3 * _query / np.linalg.norm(_query), color=ORANGE, lw=1.6)
    _ax.scatter(X[:, 0], X[:, 1], s=30 + 420 * A[_i], color=BLUE, alpha=0.85, edgecolors=SURFACE, linewidths=1.5,
                zorder=3)
    for _k in range(n):
        _ax.text(X[_k, 0], X[_k, 1] + 0.16, row_labels[_k], ha="center", fontsize=8,
                 color=INK if _k == _i else MUTED, fontweight="bold" if _k == _i else "normal")
    _avg = A[_i] @ X
    _ax.plot(*_avg, "o", ms=11, mfc="none", mec=AQUA, mew=2.2, zorder=4)
    arrow(_ax, X[_i], X[_i] + dX[_i], color=INK, lw=1.8, head=9, zorder=5)
    _ax.set_title(f"Row {_i + 1} reads: dot size = weight, ring = what it gathers")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the figure.** The orange arrow is the selected row's query direction. Rows that lie far along it get
    large weights (large dots). The green ring is the weighted average of what the row reads, and the black arrow is
    the update it receives: 0.3 of the ring. With $\beta = 0$ every visible row gets the same weight; a large
    $\beta$ picks out the single best match. Turning $\theta$ changes who is read without moving any row.

    **In real models.** Queries, keys and values are much shorter than the rows (96 against 5,280 in Huginn), which
    is how a layer affords many heads: 55 in Huginn, each gathering something different. Huginn and Recurrent-Llama
    also put position in here instead of adding offsets: before the inner product, row $i$'s query and row $j$'s
    key are each rotated by an angle proportional to their position, so the score depends on the two vectors and on
    the distance $j - i$. Those are the circles of part 3, applied inside attention (rotary position embeddings).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · The MLP: each row computes on its own

    After attention, a row holds its own state plus what it gathered. The **MLP** computes with that, the same way
    for every row and with no traffic between rows:

    $$m(x) = \sum_{r} \sigma(w_r^{\top} x + b_r)\; c_r$$

    Hidden unit $r$ measures a learned combination of the row's coordinates, passes it through a switch $\sigma$,
    and uses the result as the coefficient of a learned write vector $c_r$. The switch is what matters. Without it,
    reading and writing collapse into one linear map. With it, a unit can respond to a combination of features. One
    unit with the switch $\max(0, u)$ computes "both":
    """)
    return


@app.cell
def _():
    def both(a, b):
        return max(0, a + b - 1)  # one hidden unit: weights (1, 1), bias −1

    {(a, b): both(a, b) for a in (0, 1) for b in (0, 1)}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The toy MLP has two units (real ones are wider than $d$: 17,920 units in Huginn) and uses $\tanh$ as its switch:
    steep near 0, level beyond about ±0.5.

    - Unit 1 reads the second coordinate and writes *against* it, with strength $g$.
    - Unit 2 reads the first coordinate, reinforces it a little, and also writes against the second.

    These weights are chosen by hand so that parts 8 and 9 have something to show. The strength $g$ comes back in
    part 8.
    """)
    return


@app.cell
def _(np):
    KAPPA = 4.0  # how sharply the units switch

    W_IN = KAPPA * np.array([[0.0, 1.0],
                             [1.0, 0.0]])  # column r: the direction unit r reads

    def mlp(X, g=2.5):
        W_out = np.array([[0.0, -g],  # row r: what unit r writes
                          [0.5, -0.6]]) / KAPPA
        return np.tanh(X @ W_IN) @ W_out

    return KAPPA, mlp


@app.cell(hide_code=True)
def _(BLUE, KAPPA, MUTED, RULE, mlp, new_fig, np, plane):
    _fig, (_curve, _ax) = new_fig(2, width=9, height=3.9, width_ratios=[1, 1.05])
    _u = np.linspace(-1.5, 1.5, 300)
    _curve.axhline(0, color=RULE, lw=0.8)
    _curve.axvline(0, color=RULE, lw=0.8)
    _curve.plot(_u, -2.5 / KAPPA * np.tanh(KAPPA * _u), color=BLUE, lw=2)
    _curve.annotate("steep: slope −g", (0.03, -0.08), xytext=(0.35, 0.42), fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    _curve.text(1.45, -0.56, "level", ha="right", va="bottom", fontsize=9)
    _curve.set_xlabel("second coordinate of the row")
    _curve.set_ylabel("what unit 1 adds to it")
    _curve.set_title("Unit 1 opposes what it reads (g = 2.5)")
    _curve.tick_params(length=0)

    plane(_ax, labels=False, alpha=0.5)
    _g = np.linspace(-1.4, 1.4, 11)
    _gx, _gy = np.meshgrid(_g, _g)
    _field = mlp(np.column_stack([_gx.ravel(), _gy.ravel()]))
    _ax.quiver(_gx.ravel(), _gy.ravel(), _field[:, 0], _field[:, 1], color=BLUE, alpha=0.75, angles="xy",
               scale_units="xy", scale=2.8, width=0.006, zorder=1)
    _ax.set_title("The update m(x) at every point of the plane")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · The whole model

    A **block** (one layer) is attention and then the MLP, each adding to the matrix. It takes $n \times d$ and
    returns $n \times d$, so blocks can be stacked. After the last one, the last row has taken in the whole text.
    Nothing outside $f_\theta$ changes. Inside it, one line is new: the matrix goes through the block before its
    last row is read. Softmax and sampling stay as they were in part 4.
    """)
    return


@app.cell
def _(
    E,
    attention,
    encode,
    is_measure,
    mlp,
    np,
    offsets,
    softmax,
    text,
    toy_head,
):
    HEAD = toy_head()

    def block(X, g=2.5):
        _, dX = attention(X, *HEAD)
        X = X + dX  # attention: rows gather from earlier rows
        return X + mlp(X, g)  # MLP: each row computes on its own

    def f_block(history):
        """f_theta with one block inside: a history (token ids) to one logit per token."""
        ids = np.array(history)
        X = E[ids] + offsets(len(ids))  # the history as a matrix
        X = block(X)  # new: every row is updated with its context
        return E @ X[-1]  # the last row to logits, as before

    def next_token_probs(text):
        return softmax(f_block(encode(text)))

    is_measure(next_token_probs(text))
    return block, f_block, next_token_probs


@app.cell(hide_code=True)
def _(
    INK,
    X,
    block,
    hbars,
    matrix,
    mpl,
    n,
    new_fig,
    next_token_probs,
    row_labels,
    text,
):
    _H = block(X)
    _probs = next_token_probs(text)
    _fig, (_m1, _m2, _bars) = new_fig(3, width=9.6, height=3.9, width_ratios=[1, 1, 1.9])
    matrix(_m1, X, rows=row_labels, cols=["1", "2"], title="1 · embed")
    matrix(_m2, _H, rows=[""] * n, cols=["1", "2"], title="2 · block")
    _m2.add_patch(mpl.patches.Rectangle((-0.5, n - 1.5), 2, 1, fill=False, ec=INK, lw=1.6, zorder=5))
    hbars(_bars, _probs, title="3 · read out the last row")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To write text, pick a token from the distribution (here, the most probable one), append it, and run the whole
    thing again:
    """)
    return


@app.cell
def _(VOCAB, next_token_probs, np, text):
    def generate(text, n_tokens=4):
        for _ in range(n_tokens):
            text += " " + VOCAB[np.argmax(next_token_probs(text))]
        return text

    generate(text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The mechanics are complete and the training is not, so what it writes is not English. Training would tune `E`,
    the head and the MLP by the descent of part 4.

    Left out: real models normalise the matrix before each sub-step to control its scale, give every layer its own
    weights, and run many heads per layer. An ordinary transformer applies $L$ different blocks, once each.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the block bought

    Part 4 counted the logit vectors the model can produce after a history of three tokens: 11, one for each last
    token. The same count with the block inside $f_\theta$:
    """)
    return


@app.cell
def _(f_block, history_measure, pushforward):
    mu_block = history_measure(f_block, 3)
    mu_Z_block = pushforward(f_block, mu_block)

    {"histories": len(mu_block), "distinct logit vectors": len(mu_Z_block)}
    return mu_Z_block, mu_block


@app.cell(hide_code=True)
def _(atoms, mo, mu_Z, mu_Z_block, mu_block, new_fig):
    _fig, (_before, _after) = new_fig(2, width=9.4, height=4.5)
    atoms(_before, mu_Z)
    _before.set_title(f"No layers: {len(mu_Z):,} possible logit vectors")
    atoms(_after, mu_Z_block)
    _after.set_title(f"One block: {len(mu_Z_block):,} possible logit vectors")
    _all = len(mu_Z_block) == len(mu_block)
    mo.vstack(
        [
            _fig,
            mo.md(
                (
                    f"Every one of the {len(mu_block):,} histories now has its own logit vector. "
                    if _all
                    else f"The {len(mu_block):,} histories now give {len(mu_Z_block):,} different logit vectors. "
                )
                + "The last row has read the rows before it, so histories that end in the same token no longer "
                "collapse, and the measure depends on the whole history, as $p(v \\mid w_1, \\dots, w_n)$ "
                "promised. In both panels a dot's area is its probability when the model writes the three tokens "
                "itself."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Looping the block

    A **looped** transformer keeps one block and applies it again and again, feeding the embedded prompt $e$ back in
    every time:

    $$x_{t+1} = F(x_t, e)$$

    Here $x_t$ is the whole matrix after $t$ loops. More loops mean more computation with no more weights, and the
    number of loops is a setting. Huginn computes $e$ with two layers, loops a core of four layers starting from a
    random state, and reads out through two final layers. The toy averages the state with the prompt and applies the
    block of part 7. The step size $a$ is for later; $a = 1$ is the plain loop.
    """)
    return


@app.cell
def _(block, embed, np):
    def loop_step(S, e, g, a=1.0):
        new = block((S + e) / 2, g)  # feed the prompt back in, then apply the same block
        return (1 - a) * S + a * new  # a = 1: take the whole step; a < 1: damping

    def run_loop(text, loops, g, a=1.0, seed=0):
        e = embed(text)
        S = np.random.default_rng(seed).normal(size=e.shape)  # a random starting state
        states = [S]
        for _ in range(loops):
            S = loop_step(S, e, g, a)
            states.append(S)
        return np.array(states)  # loops + 1 states, each n × d

    return loop_step, run_loop


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Does the state settle as loops are added? That depends on the block. Lower the MLP's feedback strength $g$ and
    every row settles; raise it and more rows keep moving.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    gain = mo.ui.slider(2.0, 4.5, step=0.1, value=3.3, label="Feedback strength g", show_value=True)
    step = mo.ui.slider(0.5, 1.0, step=0.05, value=1.0, label="Step size a (1 = no damping)", show_value=True)
    damp_where = mo.ui.radio(["every position", "only the source position"], value="every position", inline=True,
                             label="Damp:")
    mo.hstack([gain, step, damp_where], justify="start", gap=1.5, wrap=True)
    return damp_where, gain, step


@app.cell
def _(E, damp_where, gain, n, np, run_loop, softmax, step, text):
    LOOPS = 200

    free = run_loop(text, LOOPS, gain.value)  # the plain loop, no damping
    still_moving = np.linalg.norm(free[-1] - free[-2], axis=1) > 1e-6
    source = int(np.argmax(still_moving)) if still_moving.any() else None  # the first row that has not settled

    step_sizes = np.ones((n, 1))
    if damp_where.value == "every position":
        step_sizes[:] = step.value
    elif source is not None:
        step_sizes[source] = step.value

    states = run_loop(text, LOOPS, gain.value, step_sizes)
    movement = np.linalg.norm(states[1:] - states[:-1], axis=2)  # how far each row moved on each loop
    probs_by_loop = softmax(states[:, -1] @ E.T)  # the next-token measure after every loop
    return LOOPS, movement, probs_by_loop, source, states


@app.cell(hide_code=True)
def _(
    LOOPS,
    SEQ,
    VOCAB,
    mo,
    movement,
    n,
    new_fig,
    np,
    probs_by_loop,
    row_labels,
    states,
    words,
):
    _shown = 48
    _fig, _ax = new_fig(1, width=9.4, height=0.5 + 0.34 * n + 0.6)
    _im = _ax.imshow(movement[:_shown].T, aspect="auto", cmap=SEQ, vmin=0, vmax=max(movement[4:].max(), 0.2),
                     extent=(0.5, _shown + 0.5, n - 0.5, -0.5))
    _ax.set_yticks(np.arange(n), row_labels)
    _ax.set_xlabel("loop")
    _ax.tick_params(length=0)
    _ax.set_title("How far each row moves on each loop")
    _fig.colorbar(_im, ax=_ax, pad=0.01, aspect=12).ax.tick_params(length=0, labelsize=8)

    _end = np.linalg.norm(states[-1] - states[-2], axis=1)
    _moving = np.where(_end > 1e-6)[0]
    _period_two = bool(np.abs(states[-1] - states[-3]).max() < 1e-6)
    _pa, _pb = probs_by_loop[-2], probs_by_loop[-1]
    if len(_moving) == 0:
        _text = (f"**Settled.** After {LOOPS} loops no row moves any more, so from here on the next-token measure is "
                 f"the same at every loop count: “{VOCAB[_pb.argmax()]}” leads with {_pb.max():.0%}.")
        _kind = "success"
    elif _period_two:
        _rows = ", ".join(f"{_k + 1} (“{words[_k]}”)" for _k in _moving)
        _text = (f"**A period-two orbit.** Rows {_rows} keep moving and return to the same state every second loop. "
                 f"Every row before row {_moving[0] + 1} has settled, and a row only reads earlier rows, so row "
                 f"{_moving[0] + 1} is not inheriting its motion: it is a source. ")
        if _pa.argmax() != _pb.argmax():
            _text += (f"The top next token **changes with the loop count**: “{VOCAB[_pa.argmax()]}” "
                      f"({_pa.max():.0%}) after {LOOPS - 1} loops, “{VOCAB[_pb.argmax()]}” ({_pb.max():.0%}) "
                      f"after {LOOPS}.")
        else:
            _k = _pb.argmax()
            _text += (f"The top next token is “{VOCAB[_k]}” on both halves of the orbit, but its probability "
                      f"alternates between {_pa[_k]:.1%} and {_pb[_k]:.1%}.")
        _kind = "warn"
    else:
        _text = (f"**Still moving.** After {LOOPS} loops some rows have neither settled nor fallen into a period-two "
                 "orbit.")
        _kind = "neutral"
    mo.vstack([_fig, mo.callout(mo.md(_text), kind=_kind)])
    return


@app.cell(hide_code=True)
def _(
    BLUE,
    LOOPS,
    ORANGE,
    RULE,
    SURFACE,
    VOCAB,
    end_label,
    mpl,
    n,
    new_fig,
    np,
    probs_by_loop,
    row_labels,
    source,
    states,
):
    _fig, (_ax, _pr) = new_fig(2, width=9.4, height=3.6)
    _last = np.arange(LOOPS - 7, LOOPS + 1)
    _rows = [n - 1] if source in (None, n - 1) else [source, n - 1]
    for _k, _color in zip(_rows, (BLUE, ORANGE)):
        _ax.plot(_last, states[_last, _k, 1], "-o", color=_color, lw=2, ms=6, mec=SURFACE, mew=1.5)
        end_label(_ax, _last[-1], states[-1, _k, 1], row_labels[_k], dx=9)
    _ax.set_xticks(_last)
    _ax.set_xlim(_last[0] - 0.4, _last[-1] + 1.6)
    _ax.margins(y=0.35)
    _ax.grid(axis="y", color=RULE, lw=0.6)
    _ax.tick_params(length=0)
    _ax.set_xlabel("loop")
    _ax.set_title("Second coordinate: " + ("source row and last row" if len(_rows) == 2 else "last row"))

    _top = np.argsort(-probs_by_loop[-2:].mean(axis=0))[:2]
    for _token, _color in zip(_top, (BLUE, ORANGE)):
        _pr.plot(_last, probs_by_loop[_last, _token], "-o", color=_color, lw=2, ms=6, mec=SURFACE, mew=1.5)
        end_label(_pr, _last[-1], probs_by_loop[-1, _token], VOCAB[_token], dx=9)
    _pr.set_xticks(_last)
    _pr.set_xlim(_last[0] - 0.4, _last[-1] + 1.2)
    _pr.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    _pr.set_ylim(0, max(0.5, probs_by_loop[_last][:, _top].max() + 0.08))
    _pr.grid(axis="y", color=RULE, lw=0.6)
    _pr.tick_params(length=0)
    _pr.set_xlabel("loops before reading out")
    _pr.set_title("Probability of the two leading next tokens")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now lower the step size $a$, first for every position and then for the source position alone.

    ### Why a row flips

    Suppose the loop has a fixed point $x^* = F(x^*, e)$. A small displacement $\delta$ from it becomes $J\delta$
    after one loop, where $J$ is the derivative of $F$ at $x^*$. Because attention only looks backwards, row $i$'s
    next state depends on rows 1 to $i$ and on nothing later. That makes $J$ block lower-triangular, with one
    $d \times d$ block $J_i$ per position on its diagonal, and the eigenvalues of the whole system are the
    eigenvalues of those blocks taken together.

    - If every eigenvalue lies inside the unit circle, displacements shrink and the state settles.
    - If some row has a real eigenvalue below −1, a displacement there reverses direction and grows on every loop.
      That row ends up alternating between two states, and attention carries the alternation to the rows that read
      it.

    **Damping** replaces the loop by $x \mapsto (1 - a)\,x + a\,F(x, e)$. It keeps every fixed point and moves each
    eigenvalue $\lambda$ to $1 - a + a\lambda$, so a small enough $a$ pulls an eigenvalue back above −1. It also
    gives a way to find the fixed point of a loop that will not settle by itself:
    """)
    return


@app.cell
def _(embed, gain, loop_step, n, np, run_loop, text):
    def jacobian(func, x, eps=1e-5):
        """The derivative of func at x by central differences: one column per coordinate of x."""
        return np.column_stack([(func(x + eps * v) - func(x - eps * v)) / (2 * eps) for v in np.eye(len(x))])

    def own_block(S, e, i, g):
        """J_i: how row i's next state responds to row i's current state."""
        def row_i(x):
            S2 = S.copy()
            S2[i] = x
            return loop_step(S2, e, g)[i]

        return jacobian(row_i, S[i])

    fixed = run_loop(text, 2000, gain.value, a=0.3)[-1]  # a heavily damped run ends on the fixed point
    eigenvalues = [np.linalg.eigvals(own_block(fixed, embed(text), i, gain.value)) for i in range(n)]
    return eigenvalues, own_block


@app.cell(hide_code=True)
def _(eigenvalues, gain, mo, n, np, words):
    def _show(value):
        return f"{value.real:+.2f}" if abs(value.imag) < 1e-9 else f"{value.real:+.2f}{value.imag:+.2f}i"

    def _verdict(values):
        if np.abs(values).max() < 1:
            return "settles on its own"
        if np.any((np.abs(values.imag) < 1e-9) & (values.real < -1)):
            return "**flips**: a real eigenvalue below −1"
        return "does not settle"

    _lines = ["| row | token | eigenvalues of its block | on its own |", "|---|---|---|---|"]
    for _k in range(n):
        _vals = sorted(eigenvalues[_k], key=lambda value: value.real)
        _lines.append(f"| {_k + 1} | {words[_k]} | {', '.join(_show(v) for v in _vals)} | {_verdict(eigenvalues[_k])} |")
    mo.md(
        "\n".join(_lines)
        + f"\n\nAt g = {gain.value:.1f}. A row that moves in the figure above but settles on its own in this table "
        "is being driven by an earlier row."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the toy, the eigenvalue below −1 comes from unit 1 of the MLP. It opposes the second coordinate with slope
    $-g$ where that coordinate is near 0 and hardly at all elsewhere, so the rows that flip are the ones whose state
    sits in the steep part of the switch. Which rows those are depends on the text.

    **Measured (Huginn).** On 200 fresh addition problems, 26 (13%) did not settle but ended on a period-two orbit.
    In each of the 26 the oscillation began at one earlier position and spread to later positions through
    attention, and the derivative at the source position had a real eigenvalue between −1.01 and −1.23. Damping the
    source position stopped its oscillation in 24 of the 26 and stilled the whole text in 15.

    Does the oscillation change what the model writes? Usually not. The two states of the orbit are close, so the
    probabilities differ by a few points, and the top choice changes only where two candidates were nearly tied. In
    the first 44 oscillating prompts examined, no answer digit changed. A targeted search (400 new sums screened for
    near-ties, the 85 closest run for 512 loops) found four answer digits that alternate between the correct digit
    and a wrong one, out of 358 answer positions on an orbit. One of them: after three worked sums, Huginn continues
    `816 + 254 =` with 1070 at 511 loops per token and with 1060 at 512. The chart replays its saved probabilities
    for the third digit. From another random starting state the same prompt still oscillates, but 7 stays ahead.
    These are examples a search was built to find; they do not say how often this happens.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, MUTED, ORANGE, RULE, SURFACE, end_label, mpl, new_fig, np):
    _replay = {
        "Start A (the start used in the search)": ([0.5809, 0.4861] * 4, [0.4065, 0.5015] * 4),
        "Start B (another random start)": ([0.5657, 0.5058] * 4, [0.4218, 0.4818] * 4),
    }
    _loops = np.arange(505, 513)
    _fig, _axes = new_fig(2, width=9.4, height=3.0, sharey=True)
    for _ax, (_title, (_p7, _p6)) in zip(_axes, _replay.items()):
        _ax.plot(_loops, _p7, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1.5)
        _ax.plot(_loops, _p6, "-o", color=ORANGE, lw=2, ms=6, mec=SURFACE, mew=1.5)
        end_label(_ax, _loops[-1], _p7[-1], "7 (correct)", dx=9, dy=-5 if _p7[-1] < _p6[-1] else 5)
        end_label(_ax, _loops[-1], _p6[-1], "6", dx=9, dy=5 if _p7[-1] < _p6[-1] else -5)
        _ax.set_xticks(_loops)
        _ax.set_xlim(504.6, 514.2)
        _ax.set_ylim(0.36, 0.62)
        _ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
        _ax.grid(axis="y", color=RULE, lw=0.6)
        _ax.tick_params(length=0)
        _ax.set_xlabel("loops before reading out")
        _ax.set_title(_title)
    _fig.suptitle("Measured · Huginn · probability of the digit after “816 + 254 = 10”", x=0.01, ha="left",
                  fontsize=10, color=MUTED)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · Pushing on a looped model

    Suppose the state has settled at $x^*$ and you want to raise the score of one token. The readout scores token
    $w$ by $z_w = e_w^{\top} x_{\text{last}}$, so the gradient of that score with respect to the last row is $e_w$:
    the direction in which the score rises fastest right now. The obvious move is to add a small vector $P$ along
    it.

    But the loop keeps transforming the state after the push. A push added once does not last: at an attracting
    fixed point it dies away as $J^t P \to 0$. To move where the state settles, add $P$ before *every* loop,
    $x_{t+1} = F(x_t + P, e)$. Pushing the last row changes no earlier row, so the last row's own block $J$ governs
    the response. To first order the score changes by

    $$\text{at once: } g^{\top} P \qquad\qquad \text{after } T \text{ loops: } g^{\top}\big(J + J^2 + \dots + J^T\big)\,P$$

    with $g = e_w$. The unit direction that makes the second quantity largest is not $g$. It is proportional to
    $(J + \dots + J^T)^{\top} g$, called the **lever** here. The two point the same way only if the loop treats
    every direction alike, and nothing forces it to.

    The toy below uses $g = 2.5$ in the MLP, where the state settles, and pushes the last row for 24 loops.
    """)
    return


@app.cell(hide_code=True)
def _(VOCAB, mo):
    target = mo.ui.dropdown(VOCAB, value="sat", label="Token whose score to raise:")
    target
    return (target,)


@app.cell
def _(E, ID, embed, loop_step, n, np, own_block, run_loop, target, text):
    G_SETTLED, T_PUSH, PUSH_SIZE = 2.5, 24, 0.03

    e_now = embed(text)
    x_star = run_loop(text, 2000, G_SETTLED, a=0.3)[-1]  # the fixed point
    J = own_block(x_star, e_now, n - 1, G_SETTLED)  # the last row's own derivative
    powers = np.array([np.linalg.matrix_power(J, t) for t in range(1, T_PUSH + 1)])  # J, J², …, J^T

    def unit(v):
        return v / np.linalg.norm(v)

    readout = E[ID[target.value]]  # g: the gradient of the target's score
    gradient_push = PUSH_SIZE * unit(readout)
    lever_push = PUSH_SIZE * unit(powers.sum(axis=0).T @ readout)

    def predicted(P):
        """First-order change of the target's score after 1, 2, …, T loops."""
        return np.cumsum(powers @ P, axis=0) @ readout

    def measured(P):
        """Run the loop, adding P to the last row before every loop."""
        S, changes = x_star.copy(), []
        for _ in range(T_PUSH):
            pushed = S.copy()
            pushed[-1] += P
            S = loop_step(pushed, e_now, G_SETTLED)
            changes.append(readout @ (S[-1] - x_star[-1]))
        return np.array(changes)

    J
    return (
        J,
        PUSH_SIZE,
        T_PUSH,
        gradient_push,
        lever_push,
        measured,
        predicted,
        readout,
        unit,
    )


@app.cell(hide_code=True)
def _(
    BLUE,
    E,
    ID,
    INK,
    J,
    MUTED,
    ORANGE,
    PUSH_SIZE,
    RULE,
    SURFACE,
    T_PUSH,
    arrow,
    end_label,
    gradient_push,
    lever_push,
    measured,
    mo,
    new_fig,
    np,
    plane,
    predicted,
    readout,
    target,
    unit,
):
    _settled = bool(np.abs(np.linalg.eigvals(J)).max() < 1)
    _fig, (_ax, _cv) = new_fig(2, width=9.4, height=3.9, width_ratios=[1, 1.35])
    plane(_ax, alpha=0.35)
    _ax.plot(*E[ID[target.value]], "o", ms=9, mfc="none", mec=INK, mew=1.6, zorder=4)
    for _push, _color, _name in [(gradient_push, ORANGE, "gradient"), (lever_push, BLUE, "lever")]:
        _dir = unit(_push)
        arrow(_ax, (0, 0), 1.25 * _dir, color=_color, lw=2.2)
        _ax.text(*(1.42 * _dir), _name, color=INK, fontsize=9, ha="center", va="center",
                 bbox=dict(fc=SURFACE, ec="none", pad=1.5, alpha=0.85))
    _ax.set_title(f"Two unit directions for raising “{target.value}” (circled)")

    _t = np.arange(0, T_PUSH + 1)
    for _push, _color, _name in [(gradient_push, ORANGE, "gradient push"), (lever_push, BLUE, "lever push")]:
        _pred = np.concatenate([[readout @ _push], predicted(_push)])
        _meas = np.concatenate([[readout @ _push], measured(_push)])
        _cv.plot(_t, _pred, color=_color, lw=2)
        _cv.plot(_t, _meas, "o", ms=4.5, color=_color, mec=SURFACE, mew=1)
        end_label(_cv, _t[-1], _pred[-1], _name, dx=8)
    _cv.axhline(0, color=MUTED, lw=0.8)
    _cv.set_xlim(-0.6, T_PUSH + 7.5)
    _cv.set_xticks([0, 4, 8, 12, 16, 20, 24])
    _cv.grid(axis="y", color=RULE, lw=0.6)
    _cv.tick_params(length=0)
    _cv.set_xlabel("loops with the push applied (0 = at once)")
    _cv.set_title(f"Change in the score of “{target.value}”: formula (line), loop (dots)")

    _cos = float(unit(gradient_push) @ unit(lever_push))
    _g_now, _l_now = readout @ gradient_push, readout @ lever_push
    _g_end, _l_end = measured(gradient_push)[-1], measured(lever_push)[-1]
    _summary = mo.md(
        f"""
        | push of length {PUSH_SIZE} along | change at once | change after {T_PUSH} loops |
        |---|---|---|
        | the gradient | {_g_now:+.4f} | {_g_end:+.4f} |
        | the lever | {_l_now:+.4f} | {_l_end:+.4f} |

        Cosine between the two directions: {_cos:+.2f}.
        """
    )
    _warning = mo.callout(
        mo.md("With this text the last row does not settle at g = 2.5, so the first-order formula does not apply "
              "and the dots leave the lines."),
        kind="warn",
    )
    mo.vstack([_fig, _summary] if _settled else [_warning, _fig, _summary])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The gradient wins at once, by construction. After looping, the lever wins, also by construction; the question
    is by how much. With the default text, the last row's $J$ reinforces the first coordinate and opposes the second,
    so a push along the second coordinate is partly undone on the next loop and a push along the first builds up.
    For a token whose embedding points mostly along the second coordinate, such as "sat", the gradient push ends
    below zero and the lever starts below zero before turning positive. A longer push departs from the lines,
    because the formula is first-order.

    **Measured.** Unit directions were averaged over training sums and tested on held-out sums; each push was added
    before every loop. The numbers are the change in the pushed answer's score.

    | direction | Huginn (20 held-out sums, 48 loops, push length 3) | Recurrent-Llama-3.2 (40 held-out sums, push 3% of the state's length) |
    |---|---|---|
    | lever | +1.78 | +0.165 |
    | readout gradient | −0.05 | −0.001 |
    | centroid of the target's states | −0.05 | 0.000 |

    In Huginn the lever was positive on 189 of 200 problem–digit pairs, started slightly negative at once (median
    −0.13) and turned positive after a single loop (+0.94), while the gradient and the centroid sat at the level of
    random directions (±0.05). In Recurrent-Llama the cosine between lever and gradient is −0.005. The chart shows
    its three push lengths.

    The first-order formula predicts the measurements. Ten lever pushes times ten candidate answers give 100 score
    changes per problem; using each problem's own $J$, the formula predicts them with median correlation 0.9997 at
    the 1% push, 0.997 at 3% and 0.976 at 9%, against 0.11 with the loop left out (in Huginn 0.978, against −0.59).

    **The catch.** On Recurrent-Llama the correct answer leads the runner-up by a median of 4.84, about ten times
    the largest change in the chart. No push made its target the model's answer: 0 of 360 tries at every length.
    """)
    return


@app.cell(hide_code=True)
def _(AQUA, BLUE, MUTED, ORANGE, RULE, SURFACE, end_label, new_fig, np):
    _doses = np.array([1, 3, 9])
    _series = [
        ("lever", BLUE, [0.055, 0.165, 0.480], [0.051, 0.154, 0.443], [0.059, 0.177, 0.519]),
        ("gradient", ORANGE, [0.000, -0.001, -0.003], [-0.001, -0.002, -0.006], [0.000, 0.000, 0.001]),
        ("centroid", AQUA, [0.000, 0.000, -0.001], [-0.001, -0.002, -0.005], [0.000, 0.001, 0.004]),
    ]
    _fig, _ax = new_fig(1, width=6.4, height=3.3)
    _ax.axhline(0, color=MUTED, lw=0.8)
    for _name, _color, _v, _lo, _hi in _series:
        _ax.plot(_doses, _v, "-o", color=_color, lw=2, ms=7, mec=SURFACE, mew=1.5, zorder=3)
        _ax.vlines(_doses, _lo, _hi, color=_color, lw=2, zorder=2)
    end_label(_ax, 9, 0.480, "lever", dx=10)
    end_label(_ax, 9, 0.0, "gradient and centroid", dx=10, dy=9)
    _ax.set_xticks(_doses, ["1%", "3%", "9%"])
    _ax.set_xlim(0.4, 12.5)
    _ax.grid(axis="y", color=RULE, lw=0.6)
    _ax.tick_params(length=0)
    _ax.set_xlabel("length of the push, as a share of the length of the state")
    _ax.set_title("Measured · Recurrent-Llama-3.2 · change of the pushed answer's score")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · What this shows, and what it does not

    The toy shows that both mechanisms fit in a few lines of numpy: a row whose own block has an eigenvalue below
    −1 drives a period-two orbit that later rows inherit, and the direction that moves a score most after looping
    is $(J + \dots + J^T)^{\top} g$, not $g$. Its weights were chosen by hand to show exactly that, so it says
    nothing about how often either happens in a trained model.

    The experiments connect a source position's local instability to a change in the model's next-token
    probabilities, and show that damping that position can sometimes stop the oscillation. They do not estimate how
    often answers flip in general use: the answer-changing examples came from searches for near-ties, and the
    outcome also depends on the starting state.

    The pushing experiments show why the response of the loop matters when choosing a direction. They measure
    changes in target scores on the tested tasks and models. In Recurrent-Llama, none of the tested pushes made the
    target the answer, even though its score rose.

    Sources in this repository: `huginn_addition/WRITEUP.md`, `docs/flicker2_design.md`,
    `docs/steering_results.pdf`, `docs/followup_results.md`. Models: Huginn (Geiping and others, 2025),
    Recurrent-Llama-3.2 (McLeish and others, 2025). The left-and-right way of writing an attention head is from
    Elhage and others, "A Mathematical Framework for Transformer Circuits" (2021); the damping rule is in Movahedi
    and others (2026); the lever is the calculus used to train fixed-point models since Bai, Kolter and Koltun
    (2019).
    """)
    return


if __name__ == "__main__":
    app.run()
