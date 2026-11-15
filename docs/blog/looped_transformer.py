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

    np.set_printoptions(precision=3, suppress=True)
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A looped transformer on eleven words

    I've looked around the internet and haven't really found a tutorial and guide on what is a transformer that felt like it was satisfying enough from a motivational and mathematical perspective, so I set out to make my own. My goal was to create something where each next step and additional part of the transformer arrives naturally as a solution to a obstacle that we're trying to overcome, and also to express what a transformer is in a clean, mathematical way.
    """)
    return


@app.cell(hide_code=True)
def _(E, VOCAB, mo, np):
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    INK, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#dddcd6", "#fcfcfb"
    BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
    SEQ = LinearSegmentedColormap.from_list("seq", [SURFACE, "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])

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
        "hat": (-0.08, -0.03, "right"),
        "sat": (0.0, 0.13, "center"),
        "bird": (-0.08, 0.05, "right"),
        "chased": (0.08, -0.04, "left"),
    }

    # ---- figures ----------------------------------------------------------------------------------------------

    def new_fig(ncols=1, width=9.0, height=3.6, **kw):
        """A figure that is not registered with pyplot, so redrawing never piles figures up."""
        fig = Figure(figsize=(width, height), layout="constrained")
        return fig, fig.subplots(1, ncols, **kw)

    def plane(ax, table=None, lim=1.7, labels=True, alpha=1.0, skip=()):
        """The two embedding coordinates as a plane, with the eleven words marked."""
        table = E if table is None else table
        ax.axhline(0, color=RULE, lw=0.8, zorder=0)
        ax.axvline(0, color=RULE, lw=0.8, zorder=0)
        ax.scatter(table[:, 0], table[:, 1], s=22, color=MUTED, alpha=alpha, linewidths=0, zorder=2)
        if labels:
            for word, (x, y) in zip(VOCAB, table):
                if word in skip:
                    continue
                dx, dy, ha = LABEL_SHIFT.get(word, (0.08, 0.05, "left"))
                scale = lim / 1.7
                ax.text(x + dx * scale, y + dy * scale, word, fontsize=8.5, color=MUTED, ha=ha, va="center",
                        alpha=alpha)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ticks = [-1, 0, 1] if lim < 2.2 else [-int(lim * 0.75), 0, int(lim * 0.75)]
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

    def head_picture(ax, X, K, Q, A, update, labels, row=-1, title=None):
        """One head, seen from one row: its query direction, the weight it puts on each key, the update it gets."""
        lim = max(1.7, float(np.abs(np.vstack([X, K, X + update])).max()) + 0.3)
        plane(ax, lim=lim, labels=False, alpha=0.22)
        q = Q[row]
        if np.linalg.norm(q) > 1e-9:
            arrow(ax, (0, 0), 0.8 * lim * q / np.linalg.norm(q), color=ORANGE, lw=1.6)
        ax.scatter(K[:, 0], K[:, 1], s=30 + 420 * A[row], color=BLUE, alpha=0.85, edgecolors=SURFACE,
                   linewidths=1.5, zorder=3)
        me = row % len(X)
        for k in range(len(X)):
            ax.text(K[k, 0], K[k, 1] + 0.1 * lim, labels[k], ha="center", fontsize=8,
                    color=INK if k == me else MUTED, fontweight="bold" if k == me else "normal")
        arrow(ax, X[row], X[row] + update[row], color=INK, lw=1.8, head=9, zorder=5)
        if title:
            ax.set_title(title)

    # ---- numbers on the page ----------------------------------------------------------------------------------

    _CSS = (
        "<style>"
        ".lt-eq{display:flex;flex-wrap:wrap;align-items:center;gap:10px 12px;margin:.55rem 0;max-width:100%;overflow-x:auto;"
        "font-variant-numeric:tabular-nums}"
        ".lt-m{display:inline-flex;flex-direction:column;align-items:center;gap:4px;max-width:100%;overflow-x:auto;flex-shrink:0}"
        ".lt-cap{font-size:.8rem;opacity:.85;white-space:nowrap}"
        ".lt-cap i{font-family:'KaTeX_Math','Times New Roman',serif;font-size:.95rem}"
        ".lt-cap small{opacity:.7;margin-left:4px}"
        ".lt-row{display:inline-flex;align-items:flex-start;align-self:flex-start}"
        "table.lt-t{border-collapse:collapse!important;font-size:.84rem;margin:0!important;width:auto!important;"
        "border:none!important}"
        "table.lt-t td,table.lt-t th{height:22px;line-height:22px;padding:0 7px!important;border:none!important;"
        "text-align:right;white-space:nowrap;background:none}"
        "table.lt-t th{font-weight:400;font-size:.74rem;opacity:.7}"
        "table.lt-t tr{border:none!important;background:none!important}"
        "table.lt-t tr.lt-rule th{border-bottom:1px solid currentColor!important}"
        "table.lt-t tr.lt-foot td{border-top:1px solid currentColor!important;opacity:.85}"
        "table.lt-t td.lt-l,table.lt-t th.lt-l{text-align:left}"
        "table.lt-lab td{font-size:.74rem;opacity:.7;padding:0 6px 0 0!important}"
        ".lt-b{position:relative;padding:0 5px}"
        ".lt-b:before,.lt-b:after{content:'';position:absolute;top:var(--top,0px);bottom:0;width:5px;"
        "border:1.5px solid currentColor;opacity:.65}"
        ".lt-b:before{left:0;border-right:none}.lt-b:after{right:0;border-left:none}"
        "table.lt-t td.lt-hi{background:rgba(42,120,214,.22)}"
        "table.lt-t td.lt-hi2{background:rgba(235,104,52,.26)}"
        "table.lt-t td.lt-dim{opacity:.3}"
        ".lt-op{font-size:1.05rem;padding:0 2px}"
        ".lt-pc{display:flex;align-items:center;gap:6px;min-width:118px}"
        ".lt-pc span:first-child{width:3.3em;text-align:right}"
        ".lt-bar{display:inline-block;height:9px;background:currentColor;opacity:.4;border-radius:1px}"
        "</style>"
    )

    def _txt(v, fmt="{:.2f}"):
        if isinstance(v, str):
            return v
        if v is None or not np.isfinite(v):
            return "·"
        s = fmt.format(v)
        if s.strip("-+0.%") == "":
            s = s.replace("-", "").replace("+", "")
        return s.replace("-", "−")

    def num(v, fmt="{:.2f}"):
        """A number as text, with a real minus sign."""
        return _txt(float(v), fmt)

    def vec(v, fmt="{:.2f}"):
        """A short vector as text: (a, b)."""
        return "(" + ", ".join(_txt(float(t), fmt) for t in v) + ")"

    def pct_cell(p):
        """A probability as a percentage with a small bar."""
        return (f'<div class="lt-pc"><span>{_txt(float(p), "{:.1%}")}</span>'
                f'<span class="lt-bar" style="width:{max(0.0, 70 * float(p)):.1f}px"></span></div>')

    def pct_plain(text):
        """Text aligned like the number of a pct_cell, for a totals row."""
        return f'<div class="lt-pc"><span>{text}</span></div>'

    def grid(M, rows=None, cols=None, fmt="{:.2f}", mark=None, mark2=None, dim=None, name=None, bracket=True,
             shape=True, left=(), foot=None):
        """A matrix as an HTML table: optional row and column labels, optional highlighted cells."""
        M = [list(r) for r in (np.atleast_2d(M) if isinstance(M, np.ndarray) else M)]
        nr, nc = len(M), len(M[0])
        fmts = fmt if isinstance(fmt, (list, tuple)) else [fmt] * nc

        def has(mask, r, c):
            return mask is not None and bool(np.asarray(mask)[r, c])

        body = ""
        if cols is not None:
            body += ('<tr class="' + ("" if bracket else "lt-rule") + '">'
                     + "".join(f'<th class="{"lt-l" if c in left else ""}">{lab}</th>' for c, lab in enumerate(cols))
                     + "</tr>")
        for r in range(nr):
            body += "<tr>"
            for c in range(nc):
                cls = "lt-hi" if has(mark, r, c) else "lt-hi2" if has(mark2, r, c) else ""
                if has(dim, r, c):
                    cls += " lt-dim"
                if c in left:
                    cls += " lt-l"
                body += f'<td class="{cls.strip()}">{_txt(M[r][c], fmts[c])}</td>'
            body += "</tr>"
        if foot is not None:
            body += ('<tr class="lt-foot">'
                     + "".join(f'<td class="{"lt-l" if c in left else ""}">{t}</td>' for c, t in enumerate(foot))
                     + "</tr>")
        table = f'<table class="lt-t">{body}</table>'
        if bracket:
            table = f'<div class="lt-b" style="--top:{22 if cols is not None else 0}px">{table}</div>'
        if rows is not None:
            lab = ("<tr><td>&nbsp;</td></tr>" if cols is not None else "") + "".join(
                f"<tr><td>{r}</td></tr>" for r in rows
            )
            table = f'<div class="lt-row"><table class="lt-t lt-lab">{lab}</table>{table}</div>'
        cap = ""
        if name:
            cap = f'<div class="lt-cap">{name}' + (f"<small>{nr}×{nc}</small>" if shape and bracket else "") + "</div>"
        return f'<div class="lt-m">{cap}{table}</div>'

    def show(*parts):
        """Lay matrices and symbols out in one row, like an equation."""
        html = "".join(p if p.lstrip().startswith("<") else f'<span class="lt-op">{p}</span>' for p in parts)
        return mo.Html(_CSS + f'<div class="lt-eq">{html}</div>')

    def vocab_table(columns, name=None, mark=None, foot=None):
        """One row per vocabulary word. columns: (header, values, kind); kind is 'pct', 'vec', 'text' or a format."""
        header = ["word"] + [c[0] for c in columns]
        body = []
        for v, word in enumerate(VOCAB):
            row = [word]
            for _, values, kind in columns:
                x = values[v]
                if kind == "pct":
                    row.append(pct_cell(x))
                elif kind == "vec":
                    row.append(vec(x))
                elif kind == "text":
                    row.append(str(x))
                else:
                    row.append(_txt(float(x), kind))
            body.append(row)
        return grid(body, cols=header, bracket=False, name=name, mark=mark, left=(0,), foot=foot)

    def top_words(p, k=3):
        """The k most probable next words, as text."""
        return ", ".join(f"{VOCAB[j]} {p[j]:.0%}" for j in np.argsort(-np.asarray(p))[:k])

    def last_row_mask(shape):
        """A highlight mask that marks the last row of a matrix."""
        mask = np.zeros(shape, dtype=bool)
        mask[-1] = True
        return mask

    return (
        AQUA,
        BLUE,
        INK,
        LABEL_SHIFT,
        MUTED,
        ORANGE,
        RULE,
        SEQ,
        SURFACE,
        arrow,
        end_label,
        grid,
        hbars,
        head_picture,
        last_row_mask,
        mpl,
        new_fig,
        num,
        pct_cell,
        pct_plain,
        plane,
        show,
        top_words,
        vec,
        vocab_table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part I · Building a transformer

    ## 1 · Eleven words and five sentences

    Let's suppose that we are living in a world where the English language has only $11$ words:
    """)
    return


@app.cell
def _(np):
    VOCAB = ["the", "a", "on", "cat", "dog", "bird", "mat", "hat", "grass", "sat", "chased"]
    ID = {word: i for i, word in enumerate(VOCAB)}

    CORPUS = [
        "the cat sat on the mat",
        "the cat sat on the mat",
        "the cat sat on the mat",
        "the cat sat on the hat",
        "the cat sat on the grass",
    ]


    def encode(text):
        """Text to token ids: one integer per word."""
        return np.array([ID[word] for word in text.split()])

    return CORPUS, ID, VOCAB, encode


@app.cell(hide_code=True)
def _(CORPUS, VOCAB, mo):
    _groups = [VOCAB[_i:_i + 4] for _i in range(0, len(VOCAB), 4)]
    _set_rows = [r",\ ".join(r"\text{" + _word + "}" for _word in _group) for _group in _groups]
    _set = r"\begin{aligned}\mathcal V=\{&" + r",\\ &".join(_set_rows) + r"\}.\end{aligned}"
    mo.vstack([
        mo.md(rf"""
        Let's call this our **vocabulary**, written $\mathcal V$:

        $${_set}$$

        Now, suppose you go out into the world and collect the following five sentences from various books.
        You decide to store them as your **training data**:
        """),
        mo.Html(
            '<ol style="font-size:1.2rem;line-height:1.7;padding-left:1.5rem;list-style:decimal">'
            + "".join(f"<li>{_s}</li>" for _s in CORPUS) + '</ol>'
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now you wonder: could we use these examples to predict what comes next? Suppose someone gives us
    the beginning of a sentence:

    **the cat sat on the ____**

    The words before the blank are the **context**. We are working with our vocabulary $\mathcal V$,
    so any word $v\in\mathcal V$ is a candidate to fill the blank. But which candidates are more likely
    after this particular context?

    Let's look at the training data to make a more informed guess. All five sentences begin with
    *the cat sat on the*. Count the words that come next:
    """)
    return


@app.cell
def _(CORPUS, ID, VOCAB, np):
    def continuations(context):
        """Count which word follows this context in the five sentences. None if the context never occurs."""
        words = context.split()
        counts = np.zeros(len(VOCAB))
        for sentence in CORPUS:
            s = sentence.split()
            if s[: len(words)] == words and len(s) > len(words):
                counts[ID[s[len(words)]]] += 1
        return counts if counts.sum() > 0 else None

    QUESTION = "the cat sat on the"
    counts = continuations(QUESTION)
    target = counts / counts.sum()  # the observed distribution for this context
    return QUESTION, continuations, counts, target


@app.cell(hide_code=True)
def _(QUESTION, VOCAB, counts, grid, mo, pct_cell, pct_plain, show, target):
    _total = int(counts.sum())
    mo.vstack([
        show(grid(
            [[f"<b>{_word}</b>", str(int(_count)), f"{int(_count)}/{_total}"]
             for _word, _count in zip(VOCAB, counts) if _count > 0],
            cols=["word in the blank", "count", "share of observations"],
            bracket=False,
            left=(0,),
        )),
        mo.md(r"""
        *Mat* follows this context in three of our five observations; *hat* and *grass* each follow it once.
        That gives us a reason to guess *mat* more often than either of the others.

        What about the other words in $\mathcal V$? They are still candidates, but none of them filled
        this blank in our five examples. Let's extend the table to include the whole vocabulary,
        giving those words a count of zero. A zero here means **not observed in this data**.
        """),
        mo.md(f"**{QUESTION} ____**"),
        show(grid(
            [[f"<b>{_word}</b>", f"{int(_count)}", f"{int(_count)}/{_total}", pct_cell(_frequency)]
             for _word, _count, _frequency in zip(VOCAB, counts, target)],
            cols=["word in the blank", "count", "share", "frequency"],
            foot=["total", str(_total), f"{_total}/{_total}", pct_plain("100.0%")],
            bracket=False,
            left=(0,),
        )),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This table gives the **observed frequency** of each word after *the cat sat on the*.
    We can write it as a function $f$, which takes a word from our vocabulary and returns its share
    of the observations:

    $$f:\mathcal V\longrightarrow[0,1],\qquad f(v)=\frac{\operatorname{count}(v)}{5}.$$

    For example,

    $$\begin{aligned}
    f(\text{mat})&=\frac35, & f(\text{hat})&=\frac15,\\
    f(\text{grass})&=\frac15, & f(\text{bird})&=0.
    \end{aligned}$$

    The other seven words also have frequency zero in these observations.

    Summing over every candidate accounts for all five observations:

    $$\begin{aligned}
    \sum_{v\in\mathcal V}f(v)
    &=f(\text{mat})+f(\text{hat})\\
    &\quad+f(\text{grass})+f(\text{bird})\\
    &\quad+f(\text{the})+f(\text{a})+f(\text{on})\\
    &\quad+f(\text{cat})+f(\text{dog})\\
    &\quad+f(\text{sat})+f(\text{chased})\\[4pt]
    &=\frac{3+1+1+0+\cdots+0}{5}\\
    &=1.
    \end{aligned}$$

    We can also ask about a group of words. Such a group is an **event**, a subset $A\subseteq\mathcal V$.
    We assign it a probability by adding the frequencies of its members:

    $$P(A)=\sum_{v\in A}f(v).$$

    For *mat or hat*, we combine the two one-word sets using a union:

    $$\{\text{mat}\}\cup\{\text{hat}\}=\{\text{mat},\text{hat}\}.$$

    These sets have no word in common, so their probabilities add:

    $$\begin{aligned}
    P(\{\text{mat}\}\cup\{\text{hat}\})
    &=P(\{\text{mat}\})+P(\{\text{hat}\})\\
    &=f(\text{mat})+f(\text{hat})\\
    &=\frac35+\frac15=\frac45.
    \end{aligned}$$

    This rule defines a **probability measure** on our finite vocabulary.
    """)
    return


@app.cell
def _(np):
    def is_measure(p):
        """Check nonnegative entries and total mass one on the vocabulary."""
        return bool(np.all(p >= 0) and np.isclose(p.sum(), 1.0))
    return (is_measure,)


@app.cell
def _(QUESTION):
    text = QUESTION
    return (text,)


@app.cell
def _(continuations, text):
    _counts = continuations(text)
    f_obs = None if _counts is None else _counts / _counts.sum()
    return (f_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Turning words into numbers

    We want to predict after new contexts too, such as ***a*** *cat sat on the*. Similar contexts should help
    us make similar predictions.

    We'd like our model to use what words have in common. For example, *cat* and *dog* can both fill
    the same place in a sentence. If we represent words with numbers, we can compare them through
    calculations and use those comparisons to make predictions. The question is: what numbers would
    capture useful similarities between words?

    Let's try a simple idea: assign a different number to each word. Here is one possible assignment,
    starting at zero. We'll call these numbers **word IDs**.
    """)
    return


@app.cell(hide_code=True)
def _(ID, VOCAB, grid, mo, show):
    mo.vstack([
        show(grid([[_word, str(ID[_word])] for _word in VOCAB],
                  cols=["word", "ID"], bracket=False, left=(0,))),
        mo.md("Use this dictionary to replace each word in the current context with its ID:"),
    ])
    return


@app.cell
def _(encode, text):
    words = text.split()
    ids = encode(text)
    n = len(ids)  # number of positions
    return ids, n, words


@app.cell(hide_code=True)
def _(grid, ids, n, show, words):
    row_labels = [f"{_i + 1} {_w}" for _i, _w in enumerate(words)]
    show(
        grid(
            [words, [str(_k) for _k in ids]],
            rows=["word", "id"],
            cols=[f"position {_i + 1}" for _i in range(n)],
            bracket=False,
        )
    )
    return (row_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we have numbers. Do their differences tell us anything useful about the words? *Cat* has ID 3,
    *dog* has ID 4, and *on* has ID 2. The equal numerical gaps do not mean that *cat* is equally similar
    to *dog* and *on*. Rearranging the dictionary would change those gaps without changing the language.
    IDs are useful for looking words up, but their numerical distances do not capture the similarities
    we want our model to use.

    What numbers would be more useful? For prediction, we'd like words that play similar roles to have
    something in common in their numerical representations. *Cat* and *dog*, for example, can both go in
    *the ____ sat on the mat*. *Mat* plays a different role in that sentence. A useful representation could
    let a model share what it learns about *cat* with *dog*, while still telling the two apart.

    Instead of one ID, give each word a vector: a short list of adjustable numbers. That vector is its
    **embedding**. We store one vector per word as a row of an **embedding matrix**, $E$. The ID now tells
    us which row to retrieve. With eleven words and $d$ numbers per word, $E$ has shape $11\times d$.

    A model usually learns these numbers by adjusting them as it practices predicting words.
    We'll see how that works in section 4.

    For now, **we'll make the table by hand**, placing the animals near one another and the places near
    one another. We'll use just two coordinates per word ($d=2$), so we can inspect every number and draw
    each embedding as a point. These positions are our starting choices.
    """)
    return


@app.cell
def _(np):
    E = np.array([
        [-1.00,  0.40],  # the
        [-0.70,  0.75],  # a
        [-0.85, -0.05],  # on
        [ 0.75,  0.75],  # cat
        [ 1.05,  0.50],  # dog
        [ 0.50,  1.05],  # bird
        [ 1.00, -0.45],  # mat
        [ 0.85, -0.80],  # hat
        [ 1.25, -0.75],  # grass
        [-0.20, -1.10],  # sat
        [ 0.05, -1.20],  # chased
    ])
    return (E,)


@app.cell(hide_code=True)
def _(
    E,
    ID,
    INK,
    MUTED,
    ORANGE,
    VOCAB,
    arrow,
    grid,
    mo,
    new_fig,
    np,
    plane,
    show,
):
    _fig, _ax = new_fig(1, width=5.0, height=4.6)
    plane(_ax)
    for _name, _xy in {"function words": (-1.6, 1.1), "animals": (0.95, 1.3), "places": (1.0, -1.15),
                       "verbs": (-0.85, -1.45)}.items():
        _ax.text(*_xy, _name, fontsize=8.5, style="italic", color=MUTED, alpha=0.75)
    for _word, _color in (("cat", INK), ("dog", ORANGE), ("mat", MUTED), ("on", MUTED)):
        arrow(_ax, (0, 0), E[ID[_word]], color=_color, lw=1.5, head=10)
    _cat_angle = np.arctan2(*E[ID["cat"]][::-1])
    _dog_angle = np.arctan2(*E[ID["dog"]][::-1])
    _arc = np.linspace(_dog_angle, _cat_angle, 60)
    _ax.plot(0.45 * np.cos(_arc), 0.45 * np.sin(_arc), color=ORANGE, lw=1.6)
    _ax.text(0.46, 0.29, f"{np.degrees(_cat_angle - _dog_angle):.1f}°", color=ORANGE, fontsize=9)
    _ax.set_title("Compare directions from the origin")
    mo.hstack(
        [_fig, show(grid(E, rows=VOCAB, cols=["coordinate 1", "coordinate 2"], name="<i>E</i>"))],
        justify="start",
        align="center",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each point is the tip of a word vector starting at the origin. To compare **directions**, look
    at the angle between the arrows: the *cat* and *dog* arrows make a small angle, while *cat* and
    *mat* point farther apart. This is the similarity we'll focus on here.

    We chose these coordinates by hand to reflect similar uses of words. The group names help us
    read the picture; the model receives only the coordinates.
    """)
    return


@app.cell(hide_code=True)
def _(E, ID, mo, np):
    _cat = E[ID["cat"]]
    _dog = E[ID["dog"]]
    _worked_pairs = []
    _angles = {}
    for _word in ("dog", "mat", "on"):
        _other = E[ID[_word]]
        _cosine = (_cat @ _other) / (np.linalg.norm(_cat) * np.linalg.norm(_other))
        _angles[_word] = np.degrees(np.arccos(np.clip(_cosine, -1, 1)))
        _lead = {"dog": "Start with **cat and dog**.",
                 "mat": "Keep the cat vector fixed. Now compare **cat and mat**.",
                 "on": "Finally, compare **cat and on**."}[_word]
        _worked_pairs.append(mo.md(rf"""
        {_lead} Multiply matching coordinates and add:

        $$\begin{{aligned}}
        e_{{\mathrm{{cat}}}}^\top e_{{\mathrm{{{_word}}}}}
        &= ({_cat[0]:.2f})({_other[0]:.2f})+({_cat[1]:.2f})({_other[1]:.2f})\\
        &= {_cat @ _other:.4f}.
        \end{{aligned}}$$
        """))
    mo.vstack([
        mo.md(r"""
        **How do we compare the vectors numerically?** For two word vectors $u=(u_1,u_2)$ and
        $v=(v_1,v_2)$, their **inner product**, also called the **dot product**, is

        $$u^\top v = u\cdot v = u_1v_1+u_2v_2.$$

        Let's apply it to the arrows in the picture.
        """),
        *_worked_pairs,
        mo.md(rf"""
        Of these three comparisons, *dog* has the largest inner product with *cat*: **{_cat @ _dog:.4f}**,
        followed by *mat* at **{_cat @ E[ID['mat']]:.4f}**, then *on* at **{_cat @ E[ID['on']]:.4f}**. It also makes the smallest angle
        with the cat arrow: **{_angles['dog']:.1f}°**, versus **{_angles['mat']:.1f}°** for mat and
        **{_angles['on']:.1f}°** for on. This matches the similarity we intended when placing the words.
        The geometry expresses that choice; it is not independent evidence about their meanings.
        """),
        mo.md(r"""
        **Alignment means how nearly two vectors point in the same direction.** Draw both arrows
        starting at the origin, and call the angle between them $\theta$. For nonzero vectors,

        $$u^\top v = \lVert u\rVert\,\lVert v\rVert\cos\theta.$$

        Here $\lVert u\rVert$ and $\lVert v\rVert$ are the arrow lengths. The directional part is
        $\cos\theta$, also called **cosine similarity**: it is 1 for the same direction, 0 for
        perpendicular arrows, and −1 for opposite directions. A smaller angle means stronger alignment.

        The inner product multiplies this alignment by both lengths. With lengths held fixed,
        a larger inner product means a smaller angle. If we want to compare direction alone,
        we divide out the lengths and use cosine similarity.
        """),
        mo.accordion({"Does distance between the points mean anything?": mo.md(r"""
            Yes. Distance measures how far apart the endpoints are, while cosine similarity measures
            direction. Raw distance depends on both direction and length. Which is useful depends on
            how the representation is trained and used; neither automatically measures meaning.

            For unit-length vectors, they give exactly the same ranking because

            $$\lVert u-v\rVert^2=2-2\cos\theta.$$

            We focus on angles here to separate directional similarity from the effect of vector length.
        """)}),
        mo.md(r"""
        **What does alignment mean for the words?** In our toy, we placed *cat* and *dog* in similar
        directions because they can play similar roles: either can fill *the ____ sat on the mat*.
        We want the model to reuse something it learns about one animal when it encounters another.
        Their alignment is our numerical way of expressing that intended similarity.

        The model gets no label saying “animal.” It only gets the coordinates. Similar directions
        make the vectors respond similarly to dot-product comparisons once their lengths are taken
        into account. In a trained model, those comparisons become useful by practicing prediction.
        Alignment can reflect similar usage; it does not necessarily mean the words are synonyms.

        **And what would a longer vector mean?** Keep its direction fixed and stretch it. Its
        contribution to every raw dot product grows in magnitude: positive scores become more
        positive, and negative scores become more negative. Direction determines which comparisons
        it favors; length scales how strongly it contributes to those comparisons.
        """),
        mo.md(rf"""
        For example, stretch our cat vector from $({_cat[0]:.2f},\ {_cat[1]:.2f})$ to
        $({2 * _cat[0]:.2f},\ {2 * _cat[1]:.2f})$. Its direction has not changed, but its dot product
        with *dog* doubles from **{_cat @ _dog:.4f}** to **{(2 * _cat) @ _dog:.4f}**.
        Its dot product with *on* doubles from **{_cat @ E[ID['on']]:.4f}** to
        **{(2 * _cat) @ E[ID['on']]:.4f}**.
        """),
        mo.md(r"""
        That does **not** make it “twice as cat-like.” Length alone does not tell us a word's
        importance, frequency, or the model's confidence. In this hand-made table, the lengths are
        choices we made. In a trained model, their role depends on what training and the later
        calculations do with them. The concrete meaning here is their effect on the comparison scores.

        We'll use those scores to build our predictor.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where is each word in the sentence?

    Take **the cat sat on the ____**. Looking up each word's embedding gives this matrix,
    with one row for each word before the blank:
    """)
    return


@app.cell(hide_code=True)
def _(E, QUESTION, encode, grid, np, show):
    _example_ids = encode(QUESTION)
    _example_words = QUESTION.split()
    _repeated = np.array([list(_example_ids).count(_id) > 1 for _id in _example_ids])
    show(grid(
        E[_example_ids],
        rows=[f"{_i + 1} {_word}" for _i, _word in enumerate(_example_words)],
        cols=["coordinate 1", "coordinate 2"],
        name="Embeddings before position offsets",
        mark=np.repeat(_repeated[:, None], 2, axis=1),
    ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first and fifth rows are identical: both copies of *the* become $(-1.00,\ 0.40)$.
    We can tell them apart by reading the row labels, but those labels aren't part of the vectors.
    The matrix still stores the words in order; **an individual word vector does not say where it came
    from**. A calculation using just that vector cannot distinguish the first *the* from the fifth.

    We can put position information into each vector by adding a **position offset** $p_i$:

    $$x_i = e_{w_i} + p_i.$$

    Here $e_{w_i}$ represents the word at position $i$, and $p_i$ represents its position.

    **Why not just use the index numbers?** A word ID tells us *which word*: both copies of *the*
    have ID 0, so adding that ID would leave them identical. A position number tells us *where*:
    these copies are at positions 1 and 5.

    Let's try it: add the position number to both coordinates.
    """)
    return


@app.cell(hide_code=True)
def _(E, ID, grid, mo, show):
    _the = E[ID["the"]]
    _cat, _dog, _on = (E[ID[_word]] for _word in ("cat", "dog", "on"))
    _pair_rows = [["No offsets", f"{_cat @ _dog:,.2f}", f"{_cat @ _on:,.2f}"]]
    for _last in (5, 5000):
        _pair_rows.append([
            f"{_last:,} and {_last - 1:,}",
            f"{(_cat + _last) @ (_dog + _last - 1):,.2f}",
            f"{(_cat + _last) @ (_on + _last - 1):,.2f}",
        ])
    mo.vstack([
        mo.md(r"""
        **First, a five-word context.** Both copies of *the* start with the embedding
        $(-1.00,\ 0.40)$. Add 1 at the first position and 5 at the fifth:
        """),
        show(grid(
            [[str(_i), f"{(_the + _i)[0]:,.2f}", f"{(_the + _i)[1]:,.2f}"] for _i in (1, 5)],
            cols=["position", "coordinate 1", "coordinate 2"], bracket=False,
        )),
        mo.md(r"""
        The two copies are now different. So far, the rule does distinguish positions.

        **Now imagine a 5,000-word sequence.** This is a thought experiment with the same word
        embeddings, beyond our notebook's eight-word input limit. At position 5,000, *the* would become

        $$(-1.00 + 5000,\ 0.40 + 5000) = (4999.00,\ 5000.40).$$

        The word contributed numbers around 1; the position contributed 5,000 to each coordinate.

        **What happens when we compare vectors?** Use the inner product we just introduced.
        Without offsets, *cat* has a positive dot product with *dog* and a
        negative one with *on*. Keep the words fixed and move the pair later in the sequence:
        """),
        show(grid(_pair_rows, cols=["positions", "cat · dog", "cat · on"], bracket=False, left=(0,))),
        mo.md(rf"""
        In each numbered row, *cat* is at the first position listed and the other word is at the
        second. At positions 5,000 and 4,999, both scores are around 50 million. The position offsets
        alone contribute $2\times5000\times4999 = 49{{,}}990{{,}}000$ to each score.

        Position can even reverse the comparison. For *cat* at position 5,000, *dog* at position 1
        scores **{(_cat + 5000) @ (_dog + 1):,.2f}**, while *on* at position 4,999 scores
        **{(_cat + 5000) @ (_on + 4999):,.2f}**. The less similar word wins because it appears later.

        The scores are not identical, and this does not imply equal prediction probabilities.
        It shows why directly adding large indices is a poor fit for the comparisons we want:
        position size can dominate the contribution from the words.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Multiplying the index by a small number delays that problem, but the offsets still grow without
    bound. An index can work as a position encoding; it just doesn't give us all the properties
    we'd like here:

    - **Different positions get different offsets** within the context we support.
    - **Offsets stay small**, so adding later positions doesn't swamp the word coordinates.
    - **Moving one position has a consistent rule**, so the relation between positions can be reused
      throughout the sentence.

    A small arrow rotating around a circle gives us these properties. Its length stays fixed, and
    each position turns it by the same angle. We use turns of 45°, giving eight distinct offsets
    before the arrow repeats. That's enough for our two-to-eight-word contexts, though longer
    contexts need more than this one circle. The table uses coordinates rounded to two decimals.
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, INK, OMEGA, ORANGE, RULE, arrow, mo, n, new_fig, np):
    _unit = np.column_stack([np.cos(np.arange(1, n + 1) * OMEGA), np.sin(np.arange(1, n + 1) * OMEGA)])
    _R = np.array([[np.cos(OMEGA), -np.sin(OMEGA)], [np.sin(OMEGA), np.cos(OMEGA)]])
    _steps_on = bool(np.allclose(_unit[:-1] @ _R.T, _unit[1:]))

    _circle_fig, _circle_ax = new_fig(width=4.8, height=4.5)
    _circle = np.linspace(0, 2 * np.pi, 240)
    _radius = 0.15
    _circle_ax.plot(_radius * np.cos(_circle), _radius * np.sin(_circle), color=RULE, lw=1.5)
    _circle_ax.axhline(0, color=RULE, lw=0.8)
    _circle_ax.axvline(0, color=RULE, lw=0.8)
    for _i in range(1, 9):
        _point = _radius * np.array([np.cos(_i * OMEGA), np.sin(_i * OMEGA)])
        _circle_ax.plot(*_point, "o", color=INK, ms=3)
        _circle_ax.text(*(1.17 * _point), str(_i), ha="center", va="center", fontsize=10)
    _p1 = _radius * np.array([np.cos(OMEGA), np.sin(OMEGA)])
    _p2 = _radius * np.array([np.cos(2 * OMEGA), np.sin(2 * OMEGA)])
    arrow(_circle_ax, (0, 0), _p1, color=BLUE, lw=2, head=12)
    arrow(_circle_ax, (0, 0), _p2, color=ORANGE, lw=2, head=12)
    _circle_ax.plot([_p1[0], _p1[0]], [0, _p1[1]], "--", color=BLUE, lw=1)
    _circle_ax.plot([0, _p1[0]], [_p1[1], _p1[1]], "--", color=BLUE, lw=1)
    _arc = np.linspace(OMEGA, 2 * OMEGA, 40)
    _circle_ax.plot(0.065 * np.cos(_arc), 0.065 * np.sin(_arc), color=ORANGE, lw=1.5)
    arrow(_circle_ax, (0.065 * np.cos(_arc[-3]), 0.065 * np.sin(_arc[-3])),
          (0.065 * np.cos(_arc[-1]), 0.065 * np.sin(_arc[-1])), color=ORANGE, lw=1.5, head=8)
    _circle_ax.text(0.022, 0.078, "45°", color=ORANGE, fontsize=10)
    _circle_ax.text(0.066, 0.025, "$p_1$", color=BLUE, fontsize=11)
    _circle_ax.text(-0.024, 0.10, "$p_2$", color=ORANGE, fontsize=11)
    _circle_ax.set(xlim=(-0.195, 0.195), ylim=(-0.195, 0.195),
                   xlabel="horizontal coordinate", ylabel="vertical coordinate", aspect="equal")
    _circle_ax.set_xticks([-0.15, 0, 0.15])
    _circle_ax.set_yticks([-0.15, 0, 0.15])
    _circle_ax.set_title("Eight positions on a circle of radius 0.15")

    _fig, (_one, _three) = new_fig(2, width=7.2, height=2.9)
    _j = np.arange(1, 201)

    def _match(periods):
        return np.mean([np.cos((100 - _j) * 2 * np.pi / _period) for _period in periods], axis=0)

    for _a, _vals, _title in [(_one, _match([8]), "One circle, 45° per step"),
                              (_three, _match([8, 27, 200]), "Three circles: periods 8, 27, 200")]:
        _a.plot(_j, _vals, color=BLUE, lw=1.2)
        _a.axhline(0, color=RULE, lw=0.8)
        _a.plot([100], [1], "o", ms=5, color=INK)
        _a.set_title(_title)
        _a.set_xlabel("position j")
        _a.set_ylim(-1.1, 1.15)
        _a.tick_params(length=0)
    _one.set_ylabel("match with position 100")

    mo.accordion(
        {
            "How the circle produces position offsets": mo.vstack([
                mo.md(r"""
                Start with an arrow of length 0.15 pointing right. Turn it 45° for position 1,
                another 45° for position 2, and so on. The numbered dots below are the eight
                positions; the colored arrows show the first two.
                """),
                _circle_fig,
                mo.md(r"""
                The dashed lines read off the two coordinates of $p_1$: about 0.11 across and
                0.11 up. At $p_2$, the arrow points straight up: 0 across and 0.15 up.
                Sine and cosine give these coordinates at any angle:

                $$p_i = r\begin{bmatrix}\cos(i\omega)\\\sin(i\omega)\end{bmatrix},
                \qquad r=0.15,\quad\omega=45^\circ.$$

                Every unrounded arrow has length 0.15, so later positions don't make it grow.

                **The matrix that turns the arrow.** A counterclockwise turn by $\omega$ is

                $$R=\begin{bmatrix}
                \cos\omega & -\sin\omega\\
                \sin\omega & \cos\omega
                \end{bmatrix}.$$

                At 45°, sine and cosine both equal $1/\sqrt2$, giving

                $$R=\frac{1}{\sqrt2}\begin{bmatrix}1&-1\\1&1\end{bmatrix}
                \approx\begin{bmatrix}0.7071&-0.7071\\0.7071&0.7071\end{bmatrix}.$$

                Apply it to the first arrow:

                $$R\,p_1=
                \frac{1}{\sqrt2}\begin{bmatrix}1&-1\\1&1\end{bmatrix}
                \frac{0.15}{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}
                =\begin{bmatrix}0\\0.15\end{bmatrix}=p_2.$$

                The first coordinate cancels; the second adds. The result is the upward arrow
                in the diagram. The same matrix takes each unrounded offset to the next:
                $R\,p_j=p_{j+1}$. Rounding the coordinates makes this relation approximate in our table.
                """),
                mo.md(f"The rotation relation holds for the unrounded offsets in this context: **{_steps_on}**."),
                mo.accordion({"Why longer contexts need more than one circle": mo.vstack([
                    mo.md(r"""
                    After eight turns, the arrow comes back around: position 9 coincides with position 1.
                    Our toy stops at eight words. To distinguish more positions, we can use several circles,
                    each turning at a different speed in its own pair of coordinates.

                    These charts compare each position with position 100, averaging the cosine of the
                    angle difference over the circles. With one circle, every eighth position matches
                    perfectly. With these three circles, only position 100 matches perfectly in the range shown.

                    The original transformer uses $d/2$ circles. Huginn and Recurrent-Llama apply rotations
                    inside the attention step of section 6 instead of adding offsets.
                    """),
                    _fig,
                ])}),
            ]),
        }
    )
    return


@app.cell
def _(E, encode, np):
    OMEGA = np.pi / 4  # 45° per position

    def offsets(n):
        """One short arrow per position: all the same length, each turned 45° from the one before."""
        i = np.arange(1, n + 1)
        return np.round(0.15 * np.column_stack([np.cos(i * OMEGA), np.sin(i * OMEGA)]), 2)

    def embed(text):
        """Text to matrix: one row per position, embedding plus position offset."""
        ids = encode(text)
        return E[ids] + offsets(len(ids))

    return OMEGA, embed, offsets


@app.cell
def _(E, ids, n, offsets):
    X_tok = E[ids]  # one embedding per position
    P = offsets(n)  # one offset per position
    X = X_tok + P  # the same as embed(text)
    return P, X, X_tok


@app.cell(hide_code=True)
def _(P, X, X_tok, grid, ids, mo, n, np, row_labels, show, text):
    _repeat = np.array([list(ids).count(_k) > 1 for _k in ids])
    _mark = np.repeat(_repeat[:, None], 2, axis=1)
    _note = (
        "The highlighted rows hold the same word. They are identical on the left and different on the right."
        if _repeat.any()
        else "No word occurs twice in this context. Try one where a word repeats."
    )
    mo.vstack(
        [
            mo.md(f"Let's represent this sentence beginning: **{text} ____**. "
                  "First look up its word embeddings, then add the position offsets. "
                  "The resulting matrix **X** is our representation of this sequence of words:"),
            show(
                grid(X_tok, rows=row_labels, cols=["1", "2"], name="Word embeddings<br><i>E</i>[ids]", mark=_mark),
                "+",
                grid(P, rows=[f"<i>p</i><sub>{_i + 1}</sub>" for _i in range(n)], cols=["1", "2"], name="Position offsets<br><i>P</i>"),
                "=",
                grid(X, rows=row_labels, cols=["1", "2"], name="Sequence representation<br><i>X</i>", mark=_mark),
            ),
            mo.md(_note),
        ]
    )
    return


@app.cell(hide_code=True)
def _(X, mo, n, words):
    _numeric_rows = r" \\ ".join(f"{_row[0]:.2f} & {_row[1]:.2f}" for _row in X)
    _vector_rows = r" \\ ".join(rf"x_{{{_i + 1}}}^\top" for _i in range(n))
    mo.md(rf"""
    We can write the same matrix as a stack of word-position vectors:

    $$X = \begin{{bmatrix}}{_numeric_rows}\end{{bmatrix}}
        = \begin{{bmatrix}}{_vector_rows}\end{{bmatrix}}.$$

    For the first word, **{words[0]}**, the vector and its row in $X$ are

    $$x_1 = \begin{{bmatrix}}{X[0, 0]:.2f} \\ {X[0, 1]:.2f}\end{{bmatrix}},
    \qquad x_1^\top = \begin{{bmatrix}}{X[0, 0]:.2f} & {X[0, 1]:.2f}\end{{bmatrix}}.$$

    The transpose symbol $\top$ turns the column into a row. Stacking one row per position gives
    $X\in\mathbb R^{{n\times d}}$: here, {n} words with two coordinates each, so ${n}\times2$.
    A longer context adds rows, never columns.

    This matrix $X$ is our **representation of this particular sequence of words**. Each $x_i$
    contains the word at position $i$ and its position offset; it hasn't yet gathered information
    from the other words.

    We have turned a sequence of words into numbers. Now we want a function that takes those
    numbers and predicts the next word.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · From a sequence to a prediction

    Our input is $X$, the representation of the words so far. What should a function of $X$ return
    if its job is to “guess the next word”?

    A single answer would hide the alternatives. In our five training examples, *the cat sat on the*
    is followed by *mat* three times, *hat* once, and *grass* once. We'd like the function to return
    a **distribution over our vocabulary**: one probability for every possible next word.
    """)
    return


@app.cell(hide_code=True)
def _(VOCAB, mo):
    _outputs = "".join(
        f'<div style="display:grid;grid-template-columns:1fr 2rem 2rem;gap:.5rem;line-height:1.8">'
        f'<span>{_word}</span><span aria-hidden="true">→</span><strong>?</strong></div>'
        for _word in VOCAB
    )
    mo.vstack([
        mo.Html(
            '<div style="display:flex;align-items:center;gap:1rem;max-width:350px;margin:1rem auto">'
            '<div style="text-align:center"><i style="font-family:serif;font-size:2rem">X</i>'
            '<div style="font-size:.8rem">sequence</div></div>'
            '<span style="font-size:1.6rem" aria-label="maps to">→</span>'
            '<div style="flex:1;min-width:0;border-left:2px solid currentColor;padding-left:1rem">'
            '<div style="font-size:.8rem;margin-bottom:.6rem">next word → probability</div>'
            + _outputs + '</div></div>'
        ),
        mo.md(r"""
        Each **?** is a probability our predictor still has to compute. There is one for every
        word in the vocabulary, and all eleven must be nonnegative and sum to 1.
        Our task is to fill in these eleven numbers.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For that observed context, we'd like its probabilities to come close to the frequencies in
    the data: 0.6 for *mat*, 0.2 for *hat*, 0.2 for *grass*, and little mass on the other words.
    For a new context, it still needs to produce a prediction, even when we have no counts to look up.
    We can then choose the highest-probability word or sample from the distribution.

    There are two things to solve:

    1. **How should the function use $X$ to score each candidate word?** The scores should help it
       assign more probability to words supported by the training data.
    2. **How do we turn those scores into valid probabilities?** Every value must be nonnegative,
       and the values across the vocabulary must add up to 1.

    Let's work backward and solve the second problem first.

    ### Suppose we already have the scores

    Assume a function $s$ takes a candidate word $v$ and the sequence representation $X$, and
    returns a finite real number. For a context of $n$ words, its input and output types are

    $$s:\mathcal V\times\mathbb R^{n\times d}\longrightarrow\mathbb R.$$

    That is, a candidate word and a matrix go in; one real-valued score comes out:

    $$(v,X)\longmapsto s(v,X)\in\mathbb R.$$

    This description leaves an enormous range of possible functions. One could ignore the
    context and store a fixed score for each word; another could calculate its score from every
    row of $X$. Even the constant function

    $$s(v,X)=1\quad\text{for every }v\text{ and }X$$

    qualifies. It always returns a real number, although it makes no distinction between words
    or contexts. Specifying the inputs and output does not yet tell us which function will
    make useful predictions.

    Hold $X$ fixed and run this function for each of our eleven candidate words. We'll name the
    score for candidate $v$ with the shorthand

    $$z_v:=s(v,X).$$

    Call these eleven scores **logits**. Higher means the scoring function favors that word more,
    but the numbers can be negative and need not sum to anything in particular. We haven't yet
    chosen how $s$ works.

    ### A trick for turning any eleven scores into probabilities

    Set aside where the logits came from. Take any eleven finite real numbers, one per word.
    They may be positive, negative, zero, or all the same. We want eleven nonnegative numbers
    that add up to 1.

    First exponentiate each score. Every $e^{z_v}$ is positive. Then add them to get a total:

    $$S=\sum_{u\in\mathcal V}e^{z_u}>0.$$

    Finally, divide each positive number by that total. Call the result $p_v$, the probability
    assigned to word $v$:

    $$p_v=\frac{e^{z_v}}{S}.$$

    These two operations together are called **softmax**. Every $p_v$ is positive, and summing
    all eleven of them gives

    $$\sum_{v\in\mathcal V}p_v
      =\frac{\sum_{v\in\mathcal V}e^{z_v}}{S}
      =\frac{S}{S}=1.$$

    This works for any finite real scores. We never needed to know how the function $s$ computed them.

    **Try the constant function.** If $s(v,X)=1$, our eleven logits are

    $$z=(1,1,1,1,1,1,1,1,1,1,1).$$

    Exponentiating gives eleven copies of $e$, so $S=11e$. Every word therefore receives

    $$p_v=\frac{e}{11e}=\frac{1}{11}\approx0.0909.$$

    And the total probability is $11\times\frac{1}{11}=1$. This is the uniform distribution:
    *the*, *a*, *on*, and every other word each get about 9.09%. These are probabilities over
    the **eleven candidate words**, with one output per logit.

    As in section 1, we can assign a probability to any set of words $A$ by adding their entries:

    $$P(A)=\sum_{v\in A}p_v,\qquad A\subseteq\mathcal V.$$

    Nonnegative entries and total mass 1 give us a **probability measure on the vocabulary**.
    For the uniform example, $P(\{\text{mat},\text{hat}\})=2/11$.

    So softmax solves the normalization problem even for our constant scoring function. But
    giving every word the same probability doesn't match the training data. Softmax preserves
    the order of the scores; to favor *mat* over *hat*, the scoring function has to give *mat*
    a higher score in the first place.
    """)
    return


@app.cell
def _(encode, np, offsets):
    def softmax(z):
        positive = np.exp(z - z.max(axis=-1, keepdims=True))  # subtracting the largest logit only avoids overflow
        return positive / positive.sum(axis=-1, keepdims=True)

    def first_predictor(table, context):
        """Read the last row, score every word against it, normalize."""
        ids = encode(context)
        x_n = table[ids[-1]] + offsets(len(ids))[-1]  # the last row of X
        z = table @ x_n  # one logit per word
        return x_n, z, softmax(z)

    return first_predictor, softmax


@app.cell
def _(E, X, first_predictor, is_measure, np, text):
    x_last, z0, p0 = first_predictor(E, text)

    exp_z0 = np.exp(z0)  # repair 1: every entry is now above 0
    denominator0 = exp_z0.sum()  # one number
    # repair 2: exp_z0 / denominator0 adds up to 1, and equals p0

    {
        "x_n is the last row of X": bool(np.array_equal(x_last, X[-1])),
        "exp(z) / total equals softmax(z)": bool(np.allclose(exp_z0 / denominator0, p0)),
        "the output is a probability measure": is_measure(p0),
    }
    return denominator0, exp_z0, p0, x_last, z0


@app.cell(hide_code=True)
def _(
    E,
    ID,
    VOCAB,
    denominator0,
    exp_z0,
    grid,
    mo,
    np,
    num,
    p0,
    pct_plain,
    show,
    text,
    vocab_table,
    x_last,
    z0,
):
    _examples = []
    for _candidate in ("a", "cat", "mat"):
        _i = ID[_candidate]
        _examples.append(mo.md(rf"""
        For **{_candidate}**, take its embedding and multiply matching coordinates with $x_n$:

        $$\begin{{aligned}}
        z_{{\text{{{_candidate}}}}}
        &=e_{{\text{{{_candidate}}}}}^\top x_n\\
        &=({E[_i, 0]:.2f})({x_last[0]:.2f})+({E[_i, 1]:.2f})({x_last[1]:.2f})\\
        &={z0[_i]:.4f}.
        \end{{aligned}}$$
        """))
    _last_word = text.split()[-1]
    _occurrence = sum(_word == _last_word for _word in text.split())
    _occurrence_note = ('In this sentence, that is the **second “the”**.'
                        if _last_word == "the" and _occurrence == 2 else '')
    _w = "mat"
    _k = ID[_w]
    _winner = int(np.argmax(z0))
    _matrix_mark = np.zeros_like(E, dtype=bool)
    _matrix_mark[_k] = True
    _score_mark = np.zeros((len(VOCAB), 1), dtype=bool)
    _score_mark[_k] = True
    _mark = np.zeros((len(VOCAB), 5), dtype=bool)
    _mark[_k, 1:3] = True
    mo.vstack([
        mo.md(rf"""
        ### What should produce the scores?

        Let's try a simple guess: use the word immediately before the blank as a clue to what comes next.
        Our context is **{text} ____**. The last word is **{text.split()[-1]}**, at position {len(text.split())}.
        {_occurrence_note}

        Its vector is the last row of $X$, written as a column:

        $$x_n=x_{{{len(text.split())}}}=e_{{\text{{{_last_word}}}}}+p_{{{len(text.split())}}}
        =\begin{{bmatrix}}{x_last[0]:.2f}\\{x_last[1]:.2f}\end{{bmatrix}}.$$

        How could we compare that clue with a candidate next word? We already have one comparison:
        the **dot product**. Let's try giving each candidate $v$ the score

        $$s(v,X)=e_v^\top x_n=z_v\in\mathbb R.$$

        A higher score means a stronger match under this comparison: direction and vector length both
        contribute. A lower score means a weaker match; vectors pointing in opposite directions have
        a negative dot product. Our guess is that a stronger match deserves a higher next-word probability.
        We'll test whether that guess works.

        **One candidate at a time.** Keep $x_n$ fixed and change only the candidate's embedding:
        """),
        *_examples,
        mo.md(r"""
        Each multiplication gives **one real number for one word**. Repeating it for all eleven words
        gives eleven scores. We can write those calculations together by stacking the candidate embeddings:

        $$\underbrace{\begin{bmatrix}
        e_{\text{the}}^\top\\e_{\text{a}}^\top\\e_{\text{on}}^\top\\
        e_{\text{cat}}^\top\\\vdots\\e_{\text{chased}}^\top
        \end{bmatrix}}_{E}\,x_n
        =\underbrace{\begin{bmatrix}
        e_{\text{the}}^\top x_n\\e_{\text{a}}^\top x_n\\e_{\text{on}}^\top x_n\\
        e_{\text{cat}}^\top x_n\\\vdots\\e_{\text{chased}}^\top x_n
        \end{bmatrix}}_{z}.$$

        Uppercase $E$ is our whole embedding table. Lowercase $e_v$ is one word's vector.
        So $z=E x_n$ is an $11\times2$ matrix times a $2\times1$ column, giving an
        $11\times1$ column of real scores. Here is the numerical calculation; the *mat* row is highlighted:
        """),
        show(
            grid(E, rows=VOCAB, name="<i>E</i>", mark=_matrix_mark),
            "×",
            grid(x_last[:, None], name="<i>x<sub>n</sub></i>"),
            "=",
            grid(z0[:, None], rows=VOCAB, name="<i>z</i>", fmt="{:.4f}", mark=_score_mark),
        ),
        mo.md(rf"""
        All eleven scores are finite real numbers, so we can use the softmax transformation we just built:

        $$p=\operatorname{{softmax}}(z)=\operatorname{{softmax}}(E x_n).$$

        It turns these scores into positive probabilities that sum to one. Higher scores get higher
        probabilities. The highest score belongs to **{VOCAB[_winner]}** ({z0[_winner]:.4f}), which gets
        **{p0[_winner]:.1%}**. The guess is **{text} {VOCAB[_winner]}**.

        In this first attempt, $x_n$ contains only the last word and its position. The earlier words
        do not affect the scores. Also, matching the last word is not the same as being a good continuation:
        the repeated **the** above shows why we need to test this rule against the data.
        """),
        mo.accordion({"Turn all eleven scores into probabilities": mo.vstack([
            show(vocab_table(
                [
                    ("embedding e<sub>v</sub>", E, "vec"),
                    ("logit z<sub>v</sub>", z0, "{:.4f}"),
                    ("exp(z<sub>v</sub>)", exp_z0, "{:.4f}"),
                    ("probability p<sub>v</sub>", p0, "pct"),
                ],
                mark=_mark,
                foot=["total", "", num(z0.sum(), "{:.4f}"), num(denominator0, "{:.4f}"), pct_plain("100.0%")],
            )),
            mo.md(f"For **{_w}**, exp({z0[_k]:.4f}) = {exp_z0[_k]:.4f}. "
                  f"Divide by the total {denominator0:.4f} to get {p0[_k]:.1%}."),
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have a complete predictor: $X\mapsto\operatorname{softmax}(E\,x_n)$.
    Its probabilities are valid by construction. Next, we need to measure how well they fit the data.

    Two softmax properties will matter: only score differences affect the probabilities,
    since $p_a/p_b=e^{z_a-z_b}$, and finite scores never give probability exactly zero.

    Reading only the last row and reusing $E$ to score candidates are choices we can test and change.
    """)
    return


@app.cell(hide_code=True)
def _(VOCAB, hbars, mo, new_fig, np, softmax, z0):
    def _divide(z):
        return z / z.sum()

    def _clip(z):
        z = np.maximum(z, 0)
        return z / z.sum()

    _fig, _axes = new_fig(3, width=9.6, height=3.7, width_ratios=[1.25, 1, 1])
    for _ax, (_name, _recipe) in zip(
        _axes, [("divide by the total", _divide), ("clip, then divide", _clip), ("softmax", softmax)]
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            _values = _recipe(z0)
        _bad = ~np.isfinite(_values) | (_values < 0)
        _zeros = int(np.sum(_values == 0))
        if _bad.any():
            _verdict = "not a measure"
        elif _zeros:
            _verdict = f"a measure, but {_zeros} words get exactly 0"
        else:
            _verdict = "a measure, every word above 0"
        hbars(_ax, _values, bad=_bad, title=_name, ticklabels=VOCAB if _ax is _axes[0] else False)
        _ax.set_xlabel(_verdict)

    mo.accordion(
        {
            "Aside: why exponentiate? Two simpler recipes and how they fail": mo.vstack(
                [
                    mo.md(
                        r"""
                        The same logits, turned into numbers that add up to 1 in three ways.

                        - **Divide by the total.** The results add up to 1, but a negative logit becomes a negative
                          "probability" (or, if the total is negative, a positive one does), and a total near 0
                          blows everything up. Red bars are not valid probabilities.
                        - **Clip negatives to 0, then divide.** Valid whenever some logit is positive. But every
                          word with a negative logit gets exactly 0, and the loss of the next section is infinite
                          when a word given 0 then occurs. Doubling every logit also changes nothing, so the model
                          cannot adjust confidence by that global rescaling alone.
                        - **Softmax.** Positive everywhere, never exactly 0, and doubling the logits squares the
                          odds.
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
    ## 4 · How wrong is it, and what can be changed?

    The predictor produces a measure. The data produced a measure too, for the one context it knows about. This
    section uses that context, *the cat sat on the*, because it is the only one for which the data has
    more than one thing to say.
    """)
    return


@app.cell
def _(E, QUESTION, first_predictor, np, target):
    def nll(p, f):
        """Average of -log p(word that came next) over data in which word v came next a share f[v] of the time."""
        return float(-np.sum(f[f > 0] * np.log(p[f > 0])))

    _, _, p_question = first_predictor(E, QUESTION)

    loss_start = nll(p_question, target)
    loss_floor = nll(target, target)  # the loss of a model that reproduced the observed distribution exactly
    {"loss of the section 3 predictor": round(loss_start, 4), "loss of a perfect model (the floor)": round(loss_floor, 4)}
    return loss_floor, loss_start, nll, p_question


@app.cell(hide_code=True)
def _(ID, loss_floor, loss_start, mo, num, p_question):
    _m, _h, _g = (p_question[ID[_w]] for _w in ("mat", "hat", "grass"))
    mo.md(
        rf"""
    ### A number for "how wrong"

    We want to choose a model that gives high probability to the data we observed. To make that precise,
    suppose we have $n$ observed endings $y_1,\ldots,y_n$, all after the same context. Our model gives
    each possible ending $v$ a probability $p(v)$.

    **Define the likelihood.** Treat the observations as independent draws from the same distribution $p$.
    Under that assumption, their joint probability is a product:

    $$\boxed{{\mathcal L(p)=\prod_{{i=1}}^{{n}}p(y_i)}}.$$

    Hold the observations fixed and vary $p$. This function of $p$ is the **likelihood**: it tells us how
    much probability the model assigns to the particular data we collected. We want to **maximize** it.

    Here $n=5$, and our observed endings are

    $$\text{{mat}},\quad\text{{mat}},\quad\text{{mat}},\quad\text{{hat}},\quad\text{{grass}}.$$

    So the likelihood is

    $$\mathcal L(p)=p(\text{{mat}})^3p(\text{{hat}})p(\text{{grass}}).$$

    **Group repeated words.** Recall that $f(v)$ is the observed frequency from section 1:

    $$f(v)=\frac{{\operatorname{{count}}(v)}}{{n}},\qquad nf(v)=\operatorname{{count}}(v).$$

    Each occurrence contributes one factor. Since *mat* occurs three times,

    $$\underbrace{{p(\text{{mat}})\,p(\text{{mat}})\,p(\text{{mat}})}}_{{\text{{three occurrences}}}}
    =p(\text{{mat}})^3.$$

    The exponent is its count, which we can also write using its frequency:

    $$nf(\text{{mat}})=5\cdot\frac35=3=\operatorname{{count}}(\text{{mat}}).$$

    The same rewriting works for every word: replace its repeated factors by $p(v)^{{nf(v)}}$.
    Multiplying over the words we observed gives the same likelihood:

    $$\mathcal L(p)=\prod_{{\substack{{v\in\mathcal V\\f(v)>0}}}}p(v)^{{nf(v)}}.$$

    **Define a loss to minimize.** We want a number that gets smaller when the likelihood gets larger.
    We choose

    $$\boxed{{L(p)=-\frac1n\ln\mathcal L(p)}}.$$

    Each part has a purpose:

    - **The logarithm** turns products into sums. It is strictly increasing (monotonic): if
      $\mathcal L(p_1)>\mathcal L(p_2)>0$, then $\ln\mathcal L(p_1)>\ln\mathcal L(p_2)$.
      It preserves which model has the larger likelihood.
    - **The minus sign** reverses that ordering: higher likelihood gives lower loss.
    - **Dividing by $n$** gives an average per observation. Since $n$ is fixed and positive,
      this scaling does not change which model minimizes the loss.

    Now substitute the likelihood and simplify:

    $$\begin{{aligned}}
    L(p)&=-\frac1n\ln\mathcal L(p)\\
    &=-\frac1n\ln\prod_{{v:f(v)>0}}p(v)^{{nf(v)}}\\
    &=-\frac1n\sum_{{v:f(v)>0}}nf(v)\ln p(v)\\
    &=-\sum_{{v\in\mathcal V}}f(v)\ln p(v).
    \end{{aligned}}$$

    This is the **average negative log-likelihood**, also called the **cross-entropy** from $f$ to $p$.
    Minimizing it is equivalent to maximizing the original likelihood.

    With our five observations, the same calculation is

    $$\begin{{aligned}}
    L(p)&=-\frac15\ln\big[p(\text{{mat}})^3p(\text{{hat}})p(\text{{grass}})\big]\\
    &=-\big[\tfrac35\ln p(\text{{mat}})\\
    &\qquad+\tfrac15\ln p(\text{{hat}})+\tfrac15\ln p(\text{{grass}})\big].
    \end{{aligned}}$$

    Each observed word contributes $-\ln p(v)$. Assigning an observed word probability $0.9$ costs
    about $0.105$; assigning it $0.01$ costs about $4.605$. The loss penalizes being confident that an
    outcome will not happen when it does. Probability assigned to unobserved words also leaves less
    for the observed ones, because the probabilities must sum to one.

    **Keep the two distributions separate.** We are comparing them for the same context,
    *the cat sat on the*:

    - $f(v)$ is **what we observed**: the fraction of our five examples ending in $v$. It is fixed by the data.
    - $p(v)$ is **what the model predicts**: its softmax probability for $v$. It changes when we change the model's weights.

    | Next word | Observed $f(v)$ | Model's current $p(v)$ |
    |:--|--:|--:|
    | mat | 0.60 | {_m:.4f} |
    | hat | 0.20 | {_h:.4f} |
    | grass | 0.20 | {_g:.4f} |
    | All other words combined | 0.00 | {1-_m-_h-_g:.4f} |

    The $f(v)$ values weight the loss, while the $p(v)$ values go inside the logarithms:

    $$\begin{{aligned}}
    L(p)&=-\big[0.6\ln {_m:.4f}\\
    &\qquad+0.2\ln {_h:.4f}+0.2\ln {_g:.4f}\big]\\
    &\approx {loss_start:.3f}.
    \end{{aligned}}$$

    **What would an ideal fit look like?** The model's probabilities $p$ would match the observed
    frequencies $f$: $p=f$. Does that give zero loss? Substituting $f$ into the loss gives

    $$\begin{{aligned}}
    L(f)=H(f)&=-\sum_{{v\in\mathcal V}}f(v)\ln f(v)\\
    &=-\big[0.6\ln0.6+0.2\ln0.2+0.2\ln0.2\big]\\
    &\approx {loss_floor:.3f}.
    \end{{aligned}}$$

    This is the **entropy of the observed distribution**: the baseline loss that remains even with
    an exact fit, because these observations contain three different endings.

    **How far above that baseline is a model $p$?** Define its excess loss as

    $$D_{{\mathrm{{KL}}}}(f\|p):=L(p)-H(f).$$

    Expanding the difference gives the **Kullback–Leibler divergence**, or **KL divergence**:

    $$\begin{{aligned}}
    D_{{\mathrm{{KL}}}}(f\|p)
    &=-\sum_{{v:f(v)>0}}f(v)\ln p(v)\\
    &\quad+\sum_{{v:f(v)>0}}f(v)\ln f(v)\\
    &=\sum_{{v:f(v)>0}}f(v)\ln\frac{{f(v)}}{{p(v)}}.
    \end{{aligned}}$$

    It is nonnegative and equals zero exactly when $p=f$. Thus $H(f)$ is the minimum loss over all
    distributions $p$, and minimizing our loss means minimizing the excess above that baseline:

    $$\boxed{{L(p)=H(f)+D_{{\mathrm{{KL}}}}(f\|p)}}.$$

    For our current model, the excess is ${loss_start:.3f}-{loss_floor:.3f}={loss_start-loss_floor:.3f}$.
    Here “ideal” means matching the observed data; it does not assert that $f$ is the true distribution.
    We use $0\ln0=0$; assigning $p(v)=0$ to an observed word gives infinite loss.

    ### What can be changed

    Let's pause and take inventory of what we have built:

    | Component | Its role |
    |:--|:--|
    | Training data and frequencies $f$ | Tell us what was observed |
    | Embedding table $E$ | Gives each word its two numbers |
    | Position offsets | Add position information to the embeddings |
    | Scoring rule $z=E x_n$ | Compares candidates with the last word-position vector |
    | Softmax $p=\operatorname{{softmax}}(z)$ | Turns scores into probabilities |
    | Loss $L(p)$ | Measures how well those probabilities fit the data |

    Our goal is to **make the loss smaller**. We could try changing the scoring rule. But first,
    let's ask a narrower question: if we keep that rule, the position offsets, softmax, and the loss
    formula fixed, do we still have room to improve? The training data stays fixed too.

    Yes: we can change the **22 entries of $E$**, which we chose by hand. The formulas stay the same,
    but different embeddings give different $x_n$ and candidate vectors, hence different scores,
    probabilities, and loss.

    These adjustable numbers are the model's **parameters**, also called its **weights**. We use
    $\theta$ as a collective name for them. **Training** means finding values of these parameters
    that lower the loss. For this first model, that means adjusting $E$.

    ### Which way to change them

    Changing $E$ changes the logits, which changes the loss. First, how does the loss change when
    we nudge just one logit $z_v$, keeping the others fixed? Throughout this calculation, the data $f$ stays fixed.

    **Write the loss in terms of logits.** Call the softmax denominator $Z$:

    $$Z=\sum_{{u\in\mathcal V}}e^{{z_u}},\qquad p(u)=\frac{{e^{{z_u}}}}{{Z}}.$$

    Taking the logarithm gives $\ln p(u)=z_u-\ln Z$. Substitute that into the loss:

    $$\begin{{aligned}}
    L&=-\sum_u f(u)\big(z_u-\ln Z\big)\\
    &=-\sum_u f(u)z_u+\Big(\sum_u f(u)\Big)\ln Z\\
    &=-\sum_u f(u)z_u+\ln Z,
    \end{{aligned}}$$

    because the observed frequencies sum to one. Here $u$ runs over all vocabulary words;
    $v$ names the particular word whose logit we are changing.

    **Differentiate the two terms.** Start with the first sum. The frequencies $f(u)$ come from the
    data, so they stay constant while we change a logit. For example, its nonzero terms are

    $$-\sum_u f(u)z_u
    =-0.6z_{{\text{{mat}}}}-0.2z_{{\text{{hat}}}}-0.2z_{{\text{{grass}}}}.$$

    If we vary only $z_{{\text{{mat}}}}$, the last two terms are constant. The first is a straight line
    with slope $-0.6$: increasing $z_{{\text{{mat}}}}$ by $\Delta$ changes this sum by $-0.6\Delta$.
    Thus its derivative is $-0.6=-f(\text{{mat}})$.

    In general, split out the term for the word $v$:

    $$-\sum_u f(u)z_u=-f(v)z_v-\sum_{{u\ne v}}f(u)z_u.$$

    With respect to $z_v$, we have $\partial z_v/\partial z_v=1$, while every other $z_u$ is held fixed.
    Using the constant-multiple rule and the fact that a constant has derivative zero,

    $$\begin{{aligned}}
    \frac{{\partial}}{{\partial z_v}}\Big[-\sum_u f(u)z_u\Big]
    &=-f(v)\cdot1-0\\
    &=-f(v).
    \end{{aligned}}$$

    We are differentiating the logit, not the frequency; $f(v)$ remains as its fixed coefficient.

    In the second term, use the chain rule. Only $e^{{z_v}}$ in $Z$ changes with $z_v$:

    $$\begin{{aligned}}
    \frac{{\partial\ln Z}}{{\partial z_v}}
    &=\frac1Z\frac{{\partial Z}}{{\partial z_v}}\\
    &=\frac{{e^{{z_v}}}}Z=p(v).
    \end{{aligned}}$$

    Adding the derivatives gives

    $$\boxed{{\frac{{\partial L}}{{\partial z_v}}=p(v)-f(v)}}.$$

    **Read the sign.** If $p(v)>f(v)$, the slope is positive: a small decrease in $z_v$ lowers the loss.
    If $p(v)<f(v)$, the slope is negative: a small increase lowers the loss. For *mat* right now,

    $$\frac{{\partial L}}{{\partial z_{{\text{{mat}}}}}}
    ={_m:.4f}-0.6\approx{_m-0.6:.4f}.$$

    So, holding the other logits fixed, increasing the *mat* logit would lower the loss.

    We adjust $E$, rather than setting the logits independently. The chain rule carries these slopes
    back to $E$, which enters twice: each row scores a candidate, and the row for *the* also supplies
    the last-word vector $x_n$. One step of **gradient descent** moves each entry of $E$ a small distance
    opposite its derivative. Then we recompute and repeat.
    """
    )
    return


@app.cell
def _(E, QUESTION, encode, np, offsets, softmax, target):
    def slope_of_loss(table, context, f):
        """dL/dE for the first predictor on one context, by the chain rule."""
        ids = encode(context)
        last = ids[-1]
        x_n = table[last] + offsets(len(ids))[-1]
        slope_z = softmax(table @ x_n) - f  # dL/dz
        slope = np.outer(slope_z, x_n)  # E as the scorer: z = E x_n
        slope[last] += table.T @ slope_z  # E as the source of x_n
        return slope

    def train_table(table, context, f, steps, rate=0.5):
        """Gradient descent on the 22 numbers of E. Returns the table after 0, 1, ..., steps steps."""
        tables = [table.copy()]
        for _ in range(steps):
            table = table - rate * slope_of_loss(table, context, f)
            tables.append(table.copy())
        return tables

    tables = train_table(E, QUESTION, target, steps=2000)
    E_trained = tables[-1]
    return E_trained, slope_of_loss, tables


@app.cell
def _(E, QUESTION, first_predictor, nll, np, slope_of_loss, target):
    def slope_by_nudging(table, context, f, eps=1e-6):
        """The same slope without calculus: nudge each of the 22 numbers and watch the loss."""
        slope = np.zeros_like(table)
        for r in range(table.shape[0]):
            for c in range(table.shape[1]):
                up, down = table.copy(), table.copy()
                up[r, c] += eps
                down[r, c] -= eps
                slope[r, c] = (nll(first_predictor(up, context)[2], f) - nll(first_predictor(down, context)[2], f)) / (2 * eps)
        return slope

    {
        "the chain-rule slope equals the slope found by nudging": bool(
            np.allclose(slope_of_loss(E, QUESTION, target), slope_by_nudging(E, QUESTION, target), atol=1e-6)
        )
    }
    return


@app.cell(hide_code=True)
def _(mo):
    train_step = mo.ui.slider(
        steps=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
        value=0,
        label="Training steps taken",
        show_value=True,
        full_width=True,
    )
    mo.vstack(
        [
            mo.md(
                "The check above confirms the chain-rule slope against plain nudging. Now run the descent. Drag "
                "the slider to move through it; every number below is the state of the predictor after that many "
                "steps."
            ),
            train_step,
        ]
    )
    return (train_step,)


@app.cell(hide_code=True)
def _(
    BLUE,
    E,
    ID,
    INK,
    MUTED,
    QUESTION,
    RULE,
    SURFACE,
    VOCAB,
    first_predictor,
    grid,
    loss_floor,
    mo,
    new_fig,
    nll,
    np,
    plane,
    show,
    tables,
    target,
    train_step,
    vec,
    vocab_table,
):
    _t = train_step.value
    _E = tables[_t]
    _x, _z, _p = first_predictor(_E, QUESTION)
    _loss = nll(_p, target)
    _losses = [nll(first_predictor(_tab, QUESTION)[2], target) for _tab in tables]
    _seen = [ID[_w] for _w in ("mat", "hat", "grass")]
    _rest = 1 - _p[_seen].sum()

    _fig, (_ax, _curve) = new_fig(2, width=9.4, height=3.9, width_ratios=[1, 1.25])
    _lim = max(1.7, float(np.abs(_E).max()) + 0.6)
    plane(_ax, table=_E, lim=_lim)
    _ax.scatter(E[:, 0], E[:, 1], s=18, facecolors="none", edgecolors=MUTED, linewidths=0.8, zorder=1)
    for _v in range(len(VOCAB)):
        _ax.plot([E[_v, 0], _E[_v, 0]], [E[_v, 1], _E[_v, 1]], color=RULE, lw=1, zorder=0)
    _ax.plot(*_x, "x", ms=7, mew=2, color=INK, zorder=4)
    _ax.set_title(f"E after {_t} steps (rings: start, ×: the last row)")

    _steps = np.arange(len(_losses))
    _curve.plot(_steps + 1, _losses, color=BLUE, lw=2)
    _curve.axhline(loss_floor, color=MUTED, lw=1, ls=(0, (4, 3)))
    _curve.text(len(_losses), loss_floor - 0.06, f"floor {loss_floor:.3f}", ha="right", va="top", fontsize=8.5,
                color=MUTED)
    _curve.plot([_t + 1], [_loss], "o", ms=8, color=INK, mec=SURFACE, mew=1.5, zorder=4)
    _curve.set_xscale("log")
    _curve.set_ylim(0.6, None)
    _curve.set_xlabel("steps taken + 1")
    _curve.set_title("Loss during the descent")
    _curve.tick_params(length=0)

    mo.vstack(
        [
            mo.hstack(
                [
                    show(
                        vocab_table(
                            [
                                ("row of E", _E, "vec"),
                                ("logit", _z, "{:.3f}"),
                                ("model p(v)", _p, "pct"),
                                ("observed f(v)", target, "pct"),
                            ],
                            name=f"After {_t} steps",
                        )
                    ),
                    show(
                        grid(
                            [
                                ["steps taken", str(_t)],
                                ["last row x<sub>n</sub>", vec(_x)],
                                ["loss L", f"<b>{_loss:.4f}</b>"],
                                ["the floor", f"{loss_floor:.4f}"],
                                ["gap to the floor", f"{_loss - loss_floor:.4f}"],
                                ["probability left on the eight words that never came next", f"{_rest:.2%}"],
                            ],
                            bracket=False,
                            left=(0,),
                        )
                    ),
                ],
                justify="start",
                align="start",
                gap=2,
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(E_trained, ID, QUESTION, first_predictor, loss_floor, mo, nll, target):
    _p = first_predictor(E_trained, QUESTION)[2]
    _rest = 1 - sum(_p[ID[_w]] for _w in ("mat", "hat", "grass"))
    mo.md(
        rf"""
    After 2,000 steps the model gives mat {_p[ID["mat"]]:.1%}, hat {_p[ID["hat"]]:.1%} and grass
    {_p[ID["grass"]]:.1%}, with a loss of {nll(_p, target):.4f} against a floor of {loss_floor:.4f}. Those numbers
    are **trained**: they came out of the loop above, starting from the hand-set table.

    **The floor is never reached.** The observed distribution gives eight words probability exactly 0. Softmax
    cannot: $e^{{z}}$ is positive for every finite $z$. So $p = f$ is unattainable with finite logits, and the
    descent can only keep widening the gaps between the three words that occurred and the eight that did not. After
    2,000 steps the eight still share {_rest:.2%}. More steps shrink that number and never make it 0. This is a
    property of softmax, and a reasonable one: a model that says "never" pays an infinite loss the first time it is
    wrong.

    **What this run did not do.** It fitted one context. The five sentences contain four shorter contexts as well
    (*the* is followed by *cat*, and so on), and a full training run would descend on all of them at once.

    From here on the notebook goes back to the hand-set table of section 2. The defect found next does not depend
    on which numbers are in $E$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · What the last row cannot know

    We will now compare predictions for different contexts. Write $p(v\mid X)$ for the probability
    of word $v$ given sequence representation $X$; the bar means “given.” Writing a dot in place
    of $v$, as in $p(\cdot\mid X)$, refers to the whole list of word probabilities. We can also
    write the words of the context after the bar instead of their matrix representation.

    Look again at what the predictor reads. It reads $x_n = e_{w_n} + p_n$: the embedding of the last word plus the
    offset of the last position. So

    $$p(\,\cdot \mid w_1, \dots, w_n) \text{ depends only on } w_n \text{ and } n.$$

    Take any two contexts of the same length that end in the same word. Their matrices differ, but their last rows
    are the same row, so their logits are equal and so are their measures. This holds for every possible table $E$,
    trained or not.

    Here is a second context to compare with the first. It must have the same length and the same last word.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    contrast_box = mo.ui.text(value="on the mat sat the", label="Second context:", full_width=True)
    contrast_box
    return (contrast_box,)


@app.cell(hide_code=True)
def _(ID, contrast_box, mo, words):
    def _fallback(ws):
        if ws == "the cat sat on the".split():
            return "on the mat sat the".split()
        alt = ws[:-1][::-1] + ws[-1:]
        if alt == ws:
            alt = [("a" if _w != "a" else "the") for _w in ws[:-1]] + ws[-1:]
        return alt

    _typed = contrast_box.value.lower().split()
    _ok = (
        all(_w in ID for _w in _typed)
        and len(_typed) == len(words)
        and _typed[-1] == words[-1]
        and _typed != words
    )
    text_b = " ".join(_typed if _ok else _fallback(words))
    (
        None
        if _ok
        else mo.callout(
            mo.md(
                f"The second context has to be a different context with {len(words)} words that ends in "
                f"*{words[-1]}*. Using *{text_b}* instead."
            ),
            kind="warn",
        )
    )
    return (text_b,)


@app.cell
def _(E, E_trained, embed, first_predictor, np, p0, text, text_b):
    X_b = embed(text_b)
    _, z0_b, p0_b = first_predictor(E, text_b)

    {
        "the two matrices are equal": bool(np.array_equal(embed(text), X_b)),
        "their last rows are equal": bool(np.array_equal(embed(text)[-1], X_b[-1])),
        "the two predictions are equal": bool(np.array_equal(p0, p0_b)),
        "and equal with the trained table too": bool(
            np.array_equal(first_predictor(E_trained, text)[2], first_predictor(E_trained, text_b)[2])
        ),
    }
    return X_b, p0_b


@app.cell(hide_code=True)
def _(
    E_trained,
    ID,
    X,
    X_b,
    first_predictor,
    grid,
    last_row_mask,
    mo,
    p0,
    p0_b,
    row_labels,
    show,
    text,
    text_b,
    vocab_table,
):
    _labels_b = [f"{_i + 1} {_w}" for _i, _w in enumerate(text_b.split())]
    _p_tr = first_predictor(E_trained, text)[2]
    mo.vstack(
        [
            show(
                grid(X, rows=row_labels, cols=["1", "2"], name=f"<i>X</i> for “{text}”", mark=last_row_mask(X.shape)),
                "&nbsp;",
                grid(X_b, rows=_labels_b, cols=["1", "2"], name=f"<i>X</i> for “{text_b}”",
                     mark=last_row_mask(X_b.shape)),
                "&nbsp;",
                vocab_table([(f"p(v | {text})", p0, "pct"), (f"p(v | {text_b})", p0_b, "pct")]),
            ),
            mo.md(
                f"The matrices differ in the rows that are not highlighted. The predictor never looks at those. "
                f"Training does not help: the table trained in section 4 gives *mat* {_p_tr[ID['mat']]:.1%} after "
                f"*{text}*, and the same {_p_tr[ID['mat']]:.1%} after *{text_b}*."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the default pair the failure is easy to feel. *The cat sat on the …* should continue with a place. *On the
    mat sat the …* should continue with an animal. One measure cannot be right for both.

    **What this does and does not show.** The five sentences alone do not prove the predictor inadequate. A
    function of the last word and the position can tell apart every context that occurs in the data: *the* stands
    at position 1 (followed by *cat*) and at position 5 (followed by *mat*, *hat* or *grass*), and those two rows
    differ by their position offsets. Section 4 showed that the predictor can fit 60/20/20. The second context
    above is not in the data, and we are not adding it. It is a counterexample to the *form* of the predictor: it
    shows a distinction that matters for language and that no choice of $E$ can express.

    So the next requirement is clear, and it is forced. **Before the last row is read, it has to come to depend on
    the other rows.** Row $n$ of $X$ is an *initial* description of one word at one position. What we need in its
    place is a description of that position *in its context*. Building that is the rest of Part I.
    """)
    return


@app.cell(hide_code=True)
def _(
    BLUE,
    E,
    LABEL_SHIFT,
    MUTED,
    SURFACE,
    VOCAB,
    first_predictor,
    mo,
    new_fig,
    np,
    plane,
    softmax,
):
    def history_measure(logits_of, length):
        """The probability of every context of this length when the model writes it itself (first word uniform)."""
        mu = {(_v,): 1 / len(VOCAB) for _v in range(len(VOCAB))}
        for _ in range(length - 1):
            mu = {_w + (_v,): _m * _q for _w, _m in mu.items() for _v, _q in enumerate(softmax(logits_of(_w)))}
        return mu

    def pushforward(logits_of, mu):
        """The measure on logit vectors induced by a measure on contexts."""
        mu_Z = {}
        for _w, _m in mu.items():
            _z = tuple(np.round(logits_of(_w), 9))
            mu_Z[_z] = mu_Z.get(_z, 0.0) + _m  # contexts that give the same logits pool their probability
        return mu_Z

    def atoms(ax, measure, scale=4500, names=None):
        """Draw a measure on logit vectors: one dot per vector, at the last row it comes from, area = probability."""
        logits = np.array(list(measure))
        mass = np.array(list(measure.values()))
        rows = logits @ np.linalg.pinv(E).T  # z = E h has exactly one solution h, because E has rank 2
        plane(ax, lim=max(1.7, float(np.abs(rows).max()) + 0.25), labels=False, alpha=0.25)
        ax.scatter(rows[:, 0], rows[:, 1], s=scale * mass, color=BLUE, alpha=0.6, edgecolors=SURFACE,
                   linewidths=0.4, zorder=3)
        if names:
            for row, key, m in zip(rows, measure, mass):
                dx, dy, ha = LABEL_SHIFT.get(names[key], (0.08, 0.05, "left"))
                ax.text(row[0] + 2.2 * dx, row[1] + 2.2 * dy, f"{names[key]} {m:.0%}", ha=ha, va="center",
                        fontsize=8.5, color=MUTED)

    def _first(context_ids):
        return first_predictor(E, " ".join(VOCAB[_k] for _k in context_ids))[1]

    mu_first = history_measure(_first, 3)
    mu_Z_first = pushforward(_first, mu_first)
    _names = {tuple(np.round(_first(_w), 9)): VOCAB[_w[-1]] for _w in mu_first}
    _fig, _ax = new_fig(1, width=5.2, height=4.4)
    atoms(_ax, mu_Z_first, names=_names)
    _ax.set_title("The distribution of Z after three words")

    mo.accordion(
        {
            "Aside: counting what the first predictor can say (pushforward measures)": mo.vstack(
                [
                    mo.md(
                        rf"""
                        There are two probability distributions in play, on different spaces. For one fixed
                        context, softmax of the logits is a distribution on $\mathcal V$: which word comes next.
                        But if the context is itself random, for example because the model wrote it one word at a
                        time, then the logit vector $Z$ is random too. Its distribution is the **pushforward** of
                        the distribution over contexts: the probability of a logit vector is the total probability
                        of the contexts that produce it.

                        With eleven words this can be computed exactly. Let the first predictor write three words,
                        the first uniformly at random and each later one from its own softmax. There are
                        {len(mu_first):,} contexts of three words, and the predictor can produce only
                        **{len(mu_Z_first)}** different logit vectors from them: one per last word. That is the
                        limitation of this section, counted. Each dot below is one possible logit vector, drawn at
                        the last row it comes from, with area equal to its probability. Section 12 repeats the
                        count for the finished layer.
                        """
                    ),
                    _fig,
                ]
            )
        }
    )
    return atoms, history_measure, mu_Z_first, pushforward


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Letting rows read earlier rows

    Our first scoring rule was $s(v,X)=e_v^\top x_n$. It compared each candidate with only the last
    word-position vector. In *the cat sat on the*, that vector represents the final **the**; it contains
    no information about **cat**, **sat**, or **on**. Changing earlier words cannot change the prediction
    if the last word and its position stay the same.

    Instead of using only $x_n$, let's combine the word vectors with a **weighted sum**:

    $$h_n=\alpha_{n1}x_1+\alpha_{n2}x_2+\cdots+\alpha_{nn}x_n
    =\sum_{j=1}^{n}\alpha_{nj}x_j.$$

    Each weight says how much of one word's vector to include. The result is one vector $h_n$
    containing contributions from the context. We then use it in place of $x_n$:

    $$s(v,X)=e_v^\top h_n.$$

    Now an earlier word can affect the score through its contribution to the sum.

    We can apply the same rule at every position, using only the words available up to that point.
    For example, at position 3:

    $$h_3=\alpha_{31}x_1+\alpha_{32}x_2+\alpha_{33}x_3.$$

    This combines **the**, **cat**, and **sat** to help predict what comes next.
    The first subscript in $\alpha_{32}$ says which row receives the information (row 3);
    the second says which row it comes from (row 2, *cat*). Rows 4 and 5 contribute nothing:
    row 3 must predict what follows *the cat sat* without seeing *on the*.
    """)
    return


@app.cell(hide_code=True)
def _(X, mo, words):
    import html as _html

    _row_count = len(words)
    _height = 100 + 52 * _row_count
    _rows_svg = []
    for _i, (_word, _row) in enumerate(zip(words, X), start=1):
        _y = 94 + 52 * (_i - 1)
        _color = "#4d9de0" if _i <= 3 else "#888888"
        _rows_svg.append(
            f'<text x="12" y="{_y}" fill="{_color}" font-size="17">{_html.escape(_word)}</text>'
            f'<path d="M 76 {_y-5} H 111" stroke="{_color}" marker-end="url(#row-arrow)"/>'
            f'<rect x="123" y="{_y-23}" width="143" height="35" rx="4" fill="{_color}" fill-opacity=".12"/>'
            f'<text x="132" y="{_y}" fill="currentColor" font-size="15" font-family="monospace">{_row[0]:.2f}  {_row[1]:.2f}</text>'
            f'<text x="278" y="{_y}" fill="{_color}" font-size="13">x<tspan baseline-shift="sub" font-size="10">{_i}</tspan>ᵀ</text>'
        )
        if _i <= 3:
            _target_y = 143 + 16 * (_i - 1)
            _rows_svg.append(
                f'<path d="M 314 {_y-5} C 342 {_y-5}, 350 {_target_y}, 375 {_target_y}" '
                'fill="none" stroke="#4d9de0" stroke-width="1.6" marker-end="url(#row-arrow)"/>'
            )
    _svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 {_height}" '
        'role="img" aria-label="Words map to rows of X. The first three rows combine into h3; later rows are excluded." '
        'style="width:100%;max-width:650px;display:block;margin:1rem auto">'
        '<defs><marker id="row-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6" fill="#4d9de0"/></marker></defs>'
        '<text x="12" y="28" fill="currentColor" font-size="16">Sentence</text>'
        '<text x="134" y="28" fill="currentColor" font-size="16">Matrix X</text>'
        '<text x="119" y="51" fill="currentColor" font-size="11">embedding + position</text>'
        f'<path d="M 122 66 H 116 V {_height-32} H 122 M 269 66 H 275 V {_height-32} H 269" '
        'fill="none" stroke="currentColor" stroke-width="1.5"/>'
        + ''.join(_rows_svg)
        + '<rect x="382" y="132" width="78" height="56" rx="8" fill="#4d9de0" fill-opacity=".16" '
        'stroke="#4d9de0"/>'
        '<text x="407" y="168" fill="currentColor" font-size="23">h₃</text>'
        '<text x="337" y="218" fill="currentColor" font-size="12">weighted sum of</text>'
        '<text x="337" y="235" fill="currentColor" font-size="12">rows 1, 2, and 3</text>'
        '</svg>'
    )
    mo.vstack([
        mo.Html(_svg),
        mo.md("Each word maps to one row of $X$. To build $h_3$, follow only the arrows from the first three "
              "rows, multiplying them by $\\alpha_{31}$, $\\alpha_{32}$, and $\\alpha_{33}$ before adding. "
              "The later rows have no arrows into $h_3$."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The result depends on the weights $\alpha_{ij}$. We try three ways of choosing them, each addressing a
    limitation of the previous choice.

    ### Start with equal weights

    Give every permitted row the same weight, $\alpha_{ij} = 1/i$. Row $i$ becomes the average of rows 1 to $i$.
    """)
    return


@app.cell
def _(X, X_b, n, np):
    U = np.tril(np.ones((n, n))) / np.arange(1, n + 1)[:, None]  # row i: weight 1/i on rows 1..i, 0 on later rows
    H_avg = U @ X
    H_avg_b = U @ X_b
    return H_avg, H_avg_b, U


@app.cell(hide_code=True)
def _(H_avg, X, mo):
    mo.stop(len(X) < 3)
    mo.md(rf"""
    **At row 3, equal weights mean one third each:**

    $$\begin{{aligned}}
    h_3&=\tfrac13x_1+\tfrac13x_2+\tfrac13x_3\\
    &=\tfrac13\left(
    \begin{{bmatrix}}{X[0,0]:.2f}\\{X[0,1]:.2f}\end{{bmatrix}}
    +\begin{{bmatrix}}{X[1,0]:.2f}\\{X[1,1]:.2f}\end{{bmatrix}}
    +\begin{{bmatrix}}{X[2,0]:.2f}\\{X[2,1]:.2f}\end{{bmatrix}}
    \right)\\
    &=\begin{{bmatrix}}{H_avg[2,0]:.3f}\\{H_avg[2,1]:.3f}\end{{bmatrix}}.
    \end{{aligned}}$$

    We average each coordinate separately. This is the third row of $H$ in the matrix calculation below.
    """)
    return

@app.cell(hide_code=True)
def _(
    E,
    H_avg,
    U,
    VOCAB,
    X,
    embed,
    grid,
    last_row_mask,
    mo,
    n,
    np,
    row_labels,
    show,
    softmax,
    text,
    vec,
    words,
):
    _h_last = H_avg[-1]
    _new_scores = E @ _h_last
    _new_probs = softmax(_new_scores)
    _winner = int(np.argmax(_new_probs))
    _nums = [str(_i + 1) for _i in range(n)]
    _nice = {"the cat sat on the": "the sat cat on the"}
    _alt = _nice.get(text, " ".join(words[:-1][::-1] + words[-1:]))
    if _alt == text:
        _alt = " ".join(words[1:-1] + words[:1] + words[-1:])
    _same = bool(np.allclose((U @ embed(_alt))[-1], H_avg[-1]))
    _order = (
        f"For *{text}*, the final average is {vec(H_avg[-1], '{:.3f}')}. "
        f"Reordering those words into *{_alt}* gives exactly the same average. "
        "So multiplying by $E$ gives the same scores, and softmax gives the same probabilities."
        if _alt != text
        else "This context has only one distinct ordering of its words, so there is nothing to compare."
    )
    mo.vstack(
        [
            show(
                grid(U, rows=row_labels, cols=_nums, name="weights <i>α</i>", mark=last_row_mask(U.shape),
                     dim=np.triu(np.ones((n, n)), 1)),
                "×",
                grid(X, cols=["1", "2"], name="<i>X</i>", mark=np.ones(X.shape, dtype=bool)),
                "=",
                grid(H_avg, rows=row_labels, cols=["1", "2"], name="<i>H</i>", mark=last_row_mask(X.shape)),
            ),
            mo.md(rf"""
            Blue marks the weights and rows used to produce the last row of $H$.

            **How do we turn $H$ into a prediction?** Take its last row, now a combination of all
            the words in the context, and write it as a column:

            $$h_n=\begin{{bmatrix}}{_h_last[0]:.3f}\\{_h_last[1]:.3f}\end{{bmatrix}}.$$

            Use the same embedding table $E$ to compute a score for every possible next word:

            $$z=E h_n.$$

            Each row below corresponds to one word in our vocabulary:
            """),
            show(
                grid(E, rows=VOCAB, name="<i>E</i>"), "×",
                grid(_h_last[:,None], name="<i>h<sub>n</sub></i>", fmt="{:.3f}"), "=",
                grid(_new_scores[:,None], rows=VOCAB, name="new scores <i>z</i>", fmt="{:.4f}"),
            ),
            mo.md(rf"""
            Finally, apply softmax to those scores:

            $$p(v)=\frac{{e^{{z_v}}}}{{\sum_{{u\in\mathcal V}}e^{{z_u}}}}.$$

            **The model's next-word prediction for “{text} ____”**

            The table gives the calculated probability of each possible next word for this context.
            """),
            show(grid(
                [[_word, f"{_score:.4f}", f"{_prob:.1%}"]
                 for _word, _score, _prob in zip(VOCAB, _new_scores, _new_probs)],
                cols=["possible next word", "score", "probability"], bracket=False, left=(0,),
            )),
            mo.md(f"The highest probability is **{VOCAB[_winner]}** ({_new_probs[_winner]:.1%}). "
                  f"Choosing it gives **{text} {VOCAB[_winner]}**. "
                  "We changed how the context vector is built; scoring and softmax work as before."),
            mo.md(
                f"""
    **What fails: the final average cannot distinguish word order.** Each row of $X$ combines a word
    with its position. But averaging all the rows separates those two contributions:

    $$h_n=\\frac{{1}}{{n}}\\sum_{{j=1}}^{{n}}(e_{{w_j}}+p_j)
    =\\frac{{1}}{{n}}\\left(\\sum_{{j=1}}^{{n}}e_{{w_j}}+\\sum_{{j=1}}^{{n}}p_j\\right).$$

    The first sum keeps which words appear and how many times each appears. The second keeps the
    position offsets. Neither sum keeps **which word was at which position**.

    For example, swap *cat* and *sat* at positions 2 and 3. Before the swap, their contribution is

    $$(e_{{\\text{{cat}}}}+p_2)+(e_{{\\text{{sat}}}}+p_3).$$

    After the swap, it is

    $$(e_{{\\text{{sat}}}}+p_2)+(e_{{\\text{{cat}}}}+p_3).$$

    Both add up to $e_{{\\text{{cat}}}}+e_{{\\text{{sat}}}}+p_2+p_3$.
    The individual rows change, but their sum does not. Adding position offsets helped us distinguish
    the rows; averaging them equally loses the association between each word and its position.

    {_order}

    There is a second limitation: every word contributes the same fraction, $1/n$. The model cannot
    give one word more influence because it is more useful for predicting what comes next.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From fixed mixing weights to context-dependent attention

    Equal averaging gives every word the same influence. What if we let training choose a different
    coefficient for each pair of positions, then kept those coefficients fixed for every sentence?
    For example, suppose it chose

    $$\alpha_{52}=0.3.$$

    That says: whenever we build row 5's output, take $0.3$ times the vector at position 2. The vector
    changes with the word, so the output can change. But the **amount of influence** given to that
    position stays the same, whatever word is there.

    **The useful relationship can move.** Step outside our eleven-word vocabulary for a short grammar
    example. Both sentence beginnings have five words and end with *cabinet*:

    | Position | 1 | 2 | 3 | 4 | 5 | Next word |
    |---|---|---|---|---|---|---|
    | A | the | **keys** | beside | the | cabinet | **are** |
    | B | beside | the | keys | the | **cabinet** | **is** |

    Read them as “The keys beside the cabinet **are**…” and “Beside the keys, the cabinet **is**…”.
    To choose *are* or *is*, the relevant relationship is to the grammatical subject. In A, that is
    *keys* at position 2. In B, it is *cabinet* at position 5; position 2 contains *the*.
    A fixed $\alpha_{52}$ cannot give position 2 more influence in A and less in B.

    This does not prescribe one correct set of attention weights. It illustrates the flexibility we
    want: **choose contributions using the words and their context, not only their position numbers**.
    Early layers may need to build up contextual information before a later layer can recognize a
    relationship such as “grammatical subject.”

    **What if $\alpha_{ij}$ varied with the words being compared?** Instead of assigning a constant
    to the position pair $(i,j)$, calculate its weight from the input. For example, the contribution
    from position 2 to position 5 can change when those positions contain different words.

    We can make that change explicit in the notation:

    $$\underbrace{\alpha_{ij}}_{\text{fixed mixing weight}}
    \quad\longrightarrow\quad
    \underbrace{\alpha_{ij}(X)}_{\text{weight computed for this input}}.$$

    We already represent each word and its position by a vector $x_i=e_{w_i}+p_i$. So compare
    $x_i$ with $x_j$ to get a score, then normalize the scores over all permitted contributors.
    This lets the weight depend on the words being compared, their positions, and the alternatives
    available in that context. The rule stays the same; its result changes with the input.

    We'll call this **context-dependent attention** to emphasize the change. The analogy is deciding
    how much attention to give each word while reading. Mathematically, $\alpha_{ij}$ says how much
    row $j$ contributes to the weighted sum for row $i$; it does not measure mental effort or establish
    the word's overall importance. “Context-dependent” is an explanatory label here: this dependence
    is already part of the standard attention mechanism.

    The dependence can include word identity and position, and in later layers the surrounding
    information already incorporated into each row. We first need a rule for computing these weights;
    then we can give that rule parameters to learn.

    We already know a way to compare two vectors: their inner product. Start with

    $$s_{ij}=x_i^\top x_j.$$

    A higher score gives row $j$ more influence on row $i$. Turn the scores into weights using softmax
    over the permitted positions:

    $$\alpha_{ij}(X)=\frac{e^{s_{ij}}}{\sum_{k\le i}e^{s_{ik}}},\qquad
    h_i=\sum_{j\le i}\alpha_{ij}(X)x_j.$$

    These weights are positive and sum to one. Changing the words can now change how much each
    row contributes. From here on we write $\alpha_{ij}$ for brevity, with the dependence on $X$
    understood. Next we will give the comparison rule its own adjustable parameters, $W_Q$ and $W_K$.
    """)
    return


@app.cell(hide_code=True)
def _(VOCAB, embed, grid, mo, np, show, softmax):
    _weights = np.zeros((len(VOCAB), len(VOCAB)))
    for _r, _word5 in enumerate(VOCAB):
        for _c, _word2 in enumerate(VOCAB):
            _input = embed(f"the {_word2} sat on {_word5}")
            _weights[_r, _c] = softmax(_input @ _input[4])[1]
    _row_words = ["the", "cat", "on"]
    _col_words = ["cat", "dog", "bird"]
    _rids = [VOCAB.index(_w) for _w in _row_words]
    _cids = [VOCAB.index(_w) for _w in _col_words]
    _mark = np.zeros((3, 3), dtype=bool)
    _mark[0, 0] = True
    _full_mark = np.zeros_like(_weights, dtype=bool)
    _full_mark[VOCAB.index("the"), VOCAB.index("cat")] = True
    _example = embed("the cat sat on the")
    _scores = _example @ _example[4]
    _weight = softmax(_scores)[1]
    mo.vstack([
        mo.md(r"""
        **Picture the dependence as a word-by-word lookup table.** Fix positions 1, 3, and 4,
        and vary only the words at positions 2 and 5:

        $$\text{the}\quad\boxed{\text{word at 2}}\quad\text{sat}\quad\text{on}
        \quad\boxed{\text{word at 5}}.$$

        Read **down the rows for the word at position 5** (receiving information), then
        **across the columns for the word at position 2** (contributing information).
        The cell contains the computed attention weight $\alpha_{52}$ for that input.
        Here is a small part of the table:
        """),
        show(grid(_weights[np.ix_(_rids, _cids)],
                  rows=[f"5: {_w}" for _w in _row_words],
                  cols=[f"2: {_w}" for _w in _col_words],
                  fmt="{:.1%}", mark=_mark, name="Computed weight α₅₂", shape=False)),
        mo.md(rf"""
        **For “the cat sat on the”:** choose the **the** row and the **cat** column.
        The highlighted cell is **{_weight:.1%}**. Thus position 5 takes {_weight:.1%} of
        position 2's vector in its weighted sum. It is one of five contributions to $h_5$,
        not a probability of predicting *cat* next.

        This entry was computed by comparing row 5 with all five rows, then taking the weight
        assigned to row 2:

        $$\alpha_{{52}}=\frac{{e^{{x_5^\top x_2}}}}{{\sum_{{j=1}}^5 e^{{x_5^\top x_j}}}}
        =\frac{{e^{{{_scores[1]:.3f}}}}}{{{np.exp(_scores).sum():.3f}}}
        \approx {_weight:.4f}.$$
        """),
        mo.accordion({"Show every vocabulary-word pair": show(grid(
            _weights, rows=[f"5: {_w}" for _w in VOCAB], cols=[f"2: {_w}" for _w in VOCAB],
            fmt="{:.1%}", mark=_full_mark, name="Computed weight α₅₂", shape=False,
        ))}),
        mo.md(r"""
        A fixed coefficient would put the same number, such as $0.3$, in every cell.
        This table changes with the words. Some substitutions make odd sentences; we are probing
        the calculation, not claiming that every combination is grammatical.

        **This is a table of outputs, not a table of learned parameters.** We generated each cell
        using the same dot-product-and-softmax rule. It applies to the template above: changing
        the other words can change the denominator and hence these weights too. Each cell comes
        from a different input, so the rows of this table do not need to sum to one. For any one
        input, it is the five weights $\alpha_{51},\ldots,\alpha_{55}$ that sum to one.
        """),
    ])
    return


@app.cell
def _(X, np, softmax):
    def causal_softmax(S):
        """Scores to weights: row i keeps only rows j ≤ i, then softmax along the row."""
        allowed = np.tri(len(S)) > 0
        return softmax(np.where(allowed, S, -np.inf))  # exp(-inf) = 0: a forbidden row gets weight exactly 0

    S_raw = X @ X.T  # S_raw[i, j] = x_i · x_j
    A_raw = causal_softmax(S_raw)
    return A_raw, S_raw, causal_softmax


@app.cell(hide_code=True)
def _(A_raw, S_raw, grid, last_row_mask, mo, n, np, row_labels, show, words):
    _nums = [f"{_i + 1} {_w}" for _i, _w in enumerate(words)]
    _upper = np.triu(np.ones((n, n)), 1)
    _best = int(np.argmax(A_raw[-1]))
    _selfish = int(np.sum(np.argmax(A_raw, axis=1) == np.arange(n)))
    mo.vstack(
        [
            show(
                grid(S_raw, rows=row_labels, cols=_nums, name="scores <i>S</i> = <i>X X</i><sup>⊤</sup>",
                     mark=last_row_mask(S_raw.shape), dim=_upper),
                "→",
                grid(A_raw, rows=row_labels, cols=_nums, name="weights <i>A</i>: softmax along each row",
                     mark=last_row_mask(A_raw.shape), dim=_upper),
            ),
            mo.md(
                f"""
    Row $i$ of $S$ holds row $i$'s scores for every row, and row $i$ of $A$ the weights they turn into. Faded
    entries are the forbidden ones, $j > i$; they get weight 0. Each row of $A$ adds up to 1.

    The last row, *{words[-1]}*, puts its largest weight ({A_raw[-1, _best]:.0%}) on
    row {_best + 1}, *{words[_best]}*.

    Raw dot products compare the original word-and-position vectors directly. But resembling the current
    word is not the same as being useful for predicting what follows it. We want training to adjust
    **what makes two rows a useful match**.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Q and K: learn how to compute the attention weights

    We already have context-dependent attention: even the direct dot product changes with its inputs.
    Queries and keys give us a trainable rule for computing those weights.

    So far, training could change the embeddings, but the comparison itself was fixed: multiply matching
    coordinates and add. Give the comparison its own adjustable numbers. A matrix can mix and rescale
    coordinates before we take their dot product.

    **Which row are we predicting from?** Suppose we have read *the cat sat* and want to predict
    the next word:

    $$\underbrace{\text{the}\quad\text{cat}\quad\text{sat}}_{\text{known input: positions 1, 2, 3}}
    \quad\underbrace{\text{\_\_\_}}_{\text{predict position 4}}.$$

    Here we build the output for **row $i=3$**, the row containing *sat*. We already know that word.
    The output $h_3$ will help us predict the word **after sat**, which is *on* in our training sentences.
    We are not trying to reconstruct *sat* or guess its embedding.

    **Where can that row get information?** The index $j$ runs over the known input rows:
    $j=1$ is *the*, $j=2$ is *cat*, and $j=3$ is *sat* itself. Each is a possible contributor to $h_3$.
    Even if a longer sentence is present during training, positions 4 and beyond are masked out when
    computing this output: position 4 would reveal the answer.

    So $i$ names the **output position receiving information**, and $j$ names an **input position
    supplying information**. Both refer to words already in the input. The candidate next word is
    a separate vocabulary choice, scored later from $h_i$.

    Give the two roles separate transformations:

    $$q_i=W_Q^\top x_i,\qquad k_j=W_K^\top x_j,$$
    $$s_{ij}=q_i^\top k_j.$$

    These matrices do not belong to one particular word: **the same $W_Q$ and $W_K$ act on every
    row**. In our example, each takes two coordinates and produces two new coordinates. Think of this
    as learning a space for comparisons, rather than adding a new dimension to the original row.

    We call $q_i$ a **query** and $k_j$ a **key**. Keep the query from *sat*, $q_3$, fixed and
    compare it with the three permitted keys:

    $$\begin{aligned}
    s_{31}&=q_3^\top k_1 &&\text{(sat compares with the)},\\
    s_{32}&=q_3^\top k_2 &&\text{(sat compares with cat)},\\
    s_{33}&=q_3^\top k_3 &&\text{(sat compares with itself)}.
    \end{aligned}$$

    For example, $s_{32}$ helps decide **how much the cat row contributes to the sat row's output**.
    It is not a score for predicting *cat* next. Softmax turns these three scores into weights:

    $$(\alpha_{31},\alpha_{32},\alpha_{33})
    =\operatorname{softmax}(s_{31},s_{32},s_{33}),$$
    $$h_3=\alpha_{31}x_1+\alpha_{32}x_2+\alpha_{33}x_3.$$

    Then $E h_3$ gives scores for the possible **next words**, and another softmax turns those
    scores into a prediction for position 4. The first softmax chooses how to combine input positions;
    the second produces probabilities over our vocabulary. Later we will let each input row send
    a transformed value instead of its original $x_j$.

    A key is therefore **a row's vector for being compared with a query**. The query $q_3$ from *sat*
    compares with the key $k_2$ from *cat* to determine one contribution to $h_3$. When we instead
    build $h_5$ to predict after *the cat sat on the*, the query $q_5$ compares with that same key
    $k_2$. Different queries can give the same key different scores.

    **Why a separate key transformation?** We could compare $q_3$ with the original $x_2$; that is
    the special case $W_K=I$. Learning $W_K$ lets training change which combinations of *cat*'s
    coordinates participate in this comparison. The query $q_2$ from *cat* has a different job:
    it compares with keys at positions 1 and 2 to build $h_2$, used to predict position 3.
    The matrices let the same input row have different representations for these two roles.

    **What new freedom does this give us?** Substituting the definitions shows the scoring rule:

    $$s_{ij}=x_i^\top\underbrace{W_QW_K^\top}_{\text{learned comparison}}x_j.$$

    With both matrices equal to the identity, we recover $x_i^\top x_j$. With other entries, a coordinate
    of one row can be compared with a different coordinate of the other, and some combinations can count
    more than others. The score need not be symmetric: how useful row $j$ is to row $i$ can differ from
    how useful row $i$ is to row $j$.

    We apply the same two matrices to every row. Stacking the queries and keys gives

    $$Q=XW_Q,\qquad K=XW_K,\qquad S=QK^\top.$$

    Entry $(i,j)$ of $S$ is exactly the single score $q_i^\top k_j$. Mask future positions and apply
    softmax along each row to get $A$, just as before. Thus $Q$ and $K$ are intermediate results;
    the adjustable parameters are the entries of $W_Q$ and $W_K$ (eight numbers in our two-dimensional toy).

    **Why should learning these matrices help?** Their entries change the scores, which change the weights,
    which change the combined vector and its next-word prediction. The loss therefore has a path back
    to these entries through the same chain rule we used for $E$. Training can favor comparisons that
    improve prediction. This gives the model a useful choice of scoring rules; it does not guarantee
    that every choice, or every training run, will work well.

    Below, two sliders let us inspect that choice directly. They set $W_Q$ to rotate by $\theta$ and
    stretch by $\beta$, while $W_K$ stays the identity. At the default, the query rotates a quarter turn
    and doubles in length. These are hand-set examples, not learned optimal weights. Training would
    adjust individual matrix entries, without restricting them to these two slider settings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    beta1 = mo.ui.slider(0, 6, step=0.5, value=2, label="Stretch β", show_value=True, debounce=True)
    theta1 = mo.ui.slider(-180, 180, step=15, value=90, label="Turn θ (degrees)", show_value=True, debounce=True)
    mo.hstack([beta1, theta1], justify="start", gap=2)
    return beta1, theta1


@app.cell
def _(X, beta1, causal_softmax, np, theta1):
    def turn(degrees, stretch=1.0):
        """The 2×2 matrix that turns a row vector anticlockwise by this angle and stretches it."""
        a = np.deg2rad(degrees)
        return np.round(stretch * np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]), 10)

    WQ1 = turn(theta1.value, beta1.value)
    WK1 = np.eye(2)

    Q1 = X @ WQ1  # one query per row
    K1 = X @ WK1  # one key per row
    S1 = Q1 @ K1.T  # S1[i, j] = q_i · k_j
    A1 = causal_softmax(S1)  # each row: weights over rows j ≤ i
    return A1, K1, Q1, S1, WK1, WQ1, turn


@app.cell(hide_code=True)
def _(WK1, WQ1, causal_softmax, embed, mo):
    _sentences = ["the cat sat on the", "the bird sat on the"]
    _rows = []
    for _sentence in _sentences:
        _x = embed(_sentence)
        _scores = (_x @ WQ1) @ (_x @ WK1).T
        _attention = causal_softmax(_scores)
        _rows.append(f"| {_sentence} | 0.300 | {_scores[4,1]:.3f} | {_attention[4,1]:.3f} |")
    mo.vstack([
        mo.md(r"""
        **Same learned rule, different computed weights.** Return to our eleven-word vocabulary.
        Here we change only the word at position 2. Both sentences use exactly the same $E$, position
        offsets, $W_Q$, and $W_K$. With the current slider settings:
        """),
        mo.md("| Five-word context | Fixed coefficient α₅₂ | Score q₅ · k₂ | Computed α₅₂ |\n"
              "|---|---:|---:|---:|\n" + "\n".join(_rows)),
        mo.md(r"""
        The fixed coefficient is an illustrative choice: it stays at $0.3$ in both sentences.
        For attention, the query $q_5$ stays the same here because the final word and its position
        stay the same. Changing *cat* to *bird* changes $x_2$ and its key $k_2$. This can change the
        score and hence the weight:

        $$\alpha_{52}=\frac{\exp(q_5^\top k_2)}{\sum_{j=1}^{5}\exp(q_5^\top k_j)}.$$

        The denominator compares this match with all the permitted alternatives. Changing another
        key can therefore change $\alpha_{52}$ too. Some parameter settings give equal weights;
        for example, setting the query stretch to zero makes every score zero.

        **What is fixed, and what is recomputed?** Training adjusts the entries of $W_Q$ and $W_K$.
        During prediction those matrices are fixed, but queries, keys, scores, and attention weights
        are calculated from each new input. No new sentence information is created: the matrices
        control how the existing information is compared.

        The computed weight is still **one number**, just like the fixed coefficient. Content
        dependence comes from making it a function of the input, not from adding an “importance
        dimension.” These hand-set numbers demonstrate that dependence; they do not show that
        *cat* or *bird* deserves more attention in a trained model.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo, n, words):
    score_query = mo.ui.dropdown({f"{i+1}: {w}": i for i, w in enumerate(words)},
                                value=f"{min(3,n)}: {words[min(3,n)-1]}", label="Query row")
    score_key = mo.ui.dropdown({f"{i+1}: {w}": i for i, w in enumerate(words)},
                              value=f"1: {words[0]}", label="Key row")
    mo.hstack([score_query, score_key], justify="start", wrap=True)
    return score_key, score_query


@app.cell(hide_code=True)
def _(A1, BLUE, Figure, K1, ORANGE, Q1, S1, mo, np, score_query, words):
    _i = score_query.value
    _q = Q1[_i]
    _keys = K1[:_i+1]
    _fig = Figure(figsize=(6, 5), layout="constrained")
    _ax = _fig.subplots()
    _lim = max(1.5, float(np.abs(np.vstack([_keys, _q])).max()) * 1.3)
    _ax.axhline(0, color="#bbb", lw=0.7)
    _ax.axvline(0, color="#bbb", lw=0.7)
    for _j, _k in enumerate(_keys):
        _ax.annotate("", xy=_k, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=BLUE, alpha=0.6))
        _ax.annotate(f"key {_j+1}: {words[_j]}", xy=_k, xytext=(5, 7),
                     textcoords="offset points", fontsize=9, color=BLUE)
    _ax.annotate("", xy=_q, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
    _ax.annotate(f"query {_i+1}: {words[_i]}", xy=_q, xytext=(5, -14),
                 textcoords="offset points", fontsize=9, color=ORANGE)
    _ax.set(xlim=(-_lim, _lim), ylim=(-_lim, _lim), aspect="equal",
            xlabel="comparison coordinate 1", ylabel="comparison coordinate 2",
            title="One query compared with every permitted key")
    _rows = "\n".join(
        f"| {_j+1}: {words[_j]} | {S1[_i,_j]:.3f} | {A1[_i,_j]:.1%} |"
        for _j in range(_i+1)
    )
    mo.vstack([
        mo.md("**Plot the comparison.** The orange arrow is the selected row's query. "
              "The blue arrows are the keys it may compare with. All arrows start at the origin "
              "and are drawn at their actual lengths. Change the query row or the sliders above."),
        _fig,
        mo.md("| Key being compared | Query · key | Attention weight |\n"
              "|---|---:|---:|\n" + _rows),
        mo.md("A larger dot product gets a larger weight. Both direction and length matter; "
              "distance between arrow tips is not the score. These weights tell this row how much "
              "to take from each value, which we introduce next. The sliders are hand-set, "
              "so a high weight here need not mean the row is useful for prediction."),
    ])
    return


@app.cell(hide_code=True)
def _(
    A1,
    K1,
    Q1,
    S1,
    WK1,
    WQ1,
    X,
    grid,
    mo,
    n,
    np,
    num,
    row_labels,
    score_key,
    score_query,
    show,
):
    _i, _j = score_query.value, score_key.value
    _allowed = _j <= _i
    _mark = np.zeros_like(S1, dtype=bool)
    _mark[_i, _j] = True
    mo.vstack([
        mo.md(r"First inspect one score: $q_i=W_Q^\top x_i$, $k_j=W_K^\top x_j$, $s_{ij}=q_i^\top k_j$."),
        show(grid([X[_i]], name=f"x_{_i+1}"), "×", grid(WQ1, name="W_Q"), "=",
             grid([Q1[_i]], name=f"q_{_i+1}", mark=np.ones((1, 2), dtype=bool))),
        show(grid([X[_j]], name=f"x_{_j+1}"), "×", grid(WK1, name="W_K"), "=",
             grid([K1[_j]], name=f"k_{_j+1}", mark=np.ones((1, 2), dtype=bool))),
        mo.md(f"**Multiply matching coordinates and add:** "
              f"({num(Q1[_i,0], '{:.5f}')})({num(K1[_j,0], '{:.5f}')}) + "
              f"({num(Q1[_i,1], '{:.5f}')})({num(K1[_j,1], '{:.5f}')}) = **{S1[_i,_j]:.5f}**. "
              + (f"After normalizing the permitted scores, its weight is {A1[_i,_j]:.5f}." if _allowed else
                 "This key is in the future: masking replaces the score by −∞ before softmax, so its weight is 0.")),
        mo.accordion({
            "Normalize this query's permitted scores": mo.vstack([
                mo.md(r"$\alpha_{ij}=e^{s_{ij}-m}/\sum_{k\le i}e^{s_{ik}-m}$, where $m=\max_{k\le i}s_{ik}$. "
                      "Subtracting the same maximum changes neither the weights nor their sum."),
                show(grid([S1[_i, :_i+1], np.exp(S1[_i, :_i+1]-S1[_i, :_i+1].max()), A1[_i, :_i+1]],
                          rows=["score", "exp(score − max)", "weight"], cols=row_labels[:_i+1], fmt="{:.5f}")),
            ]),
            "Expand full Q, K, score and attention matrices": mo.vstack([
                show(grid(Q1, rows=row_labels, name="Q"), grid(K1, rows=row_labels, name="K")),
                show(grid(S1, rows=row_labels, cols=list(range(1,n+1)), name="Q Kᵀ", mark=_mark,
                          dim=np.triu(np.ones((n,n)),1)), "→",
                     grid(A1, rows=row_labels, cols=list(range(1,n+1)), name="A", mark=_mark)),
            ]),
        }),
        mo.md("The score determines whom to read. The normalized weight determines how much of that row's "
              "message to receive. Values, next, determine the message itself. Displayed decimals are rounded."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Two softmaxes, two different sample spaces.**

    - The softmax of this section runs along one row of scores. Its output, a row of $A$, is a distribution over
      the **positions** $1, \dots, i$ of the text. Nothing is ever drawn from it. It is used as mixing weights.
    - The softmax of section 3 runs over the eleven logits. Its output is a distribution over the **vocabulary**,
      and the next word is drawn from it.

    Same formula, different jobs. The first decides where a row reads. The second decides what the model writes.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The computed coefficients $\alpha_{ij}$ are the **attention weights**. The learned matrices
    $W_Q$ and $W_K$ are **parameters of the rule that produces them**. Attention is the full operation:
    compute scores, normalize them into weights, and combine the vectors using those weights.
    The Q/K comparison is one part of that operation. The restriction $j\le i$ is the **causal mask**.

    ### What happens to position information?

    Each query and key still starts from a word **at a particular position**:

    $$q_i=W_Q^\top(e_{w_i}+p_i),\qquad
    k_j=W_K^\top(e_{w_j}+p_j).$$

    Swapping two words changes which embedding is paired with each position offset. That can change
    their queries and keys, and therefore their attention weights. The causal mask also keeps track of
    which positions each row may use.

    Recall why equal averaging failed: swapping *cat* and *sat* left the sum unchanged. With unequal
    weights $a$ and $b$ on their two positions, compare the contributions before and after a swap:

    $$\begin{aligned}
    \text{before: }&a(e_{\text{cat}}+p_2)+b(e_{\text{sat}}+p_3),\\
    \text{after: }&a(e_{\text{sat}}+p_2)+b(e_{\text{cat}}+p_3).
    \end{aligned}$$

    Subtracting gives

    $$(a-b)(e_{\text{cat}}-e_{\text{sat}}).$$

    Equal averaging had $a=b$, so this was always zero. With different weights and different embeddings,
    it is nonzero. This calculation holds the weights fixed to isolate the effect; in attention,
    swapping words can change the weights too.

    We can still split a weighted sum into word and position contributions. The difference is that
    the word contributions now have weights tied to their rows, so a swap no longer has to cancel.
    Attention **can distinguish word orders**; it does not preserve every detail automatically.
    Equal weights or transformations that discard useful coordinates can still lose information.
    Position offsets make order available, and training must learn to use it.

    ## 7 · What should a row send?

    The weights are settled. Now look at what is being averaged. So far each row contributes itself, $x_j$. Here is
    the last row after averaging raw rows with the weights of section 6, read out with $E$ as before.
    """)
    return


@app.cell
def _(A1, E, X, softmax):
    H_copy = A1 @ X  # every row replaced by a weighted average of raw rows
    p_copy = softmax(E @ H_copy[-1])
    return H_copy, p_copy


@app.cell(hide_code=True)
def _(A1, H_copy, mo, np, p_copy, top_words, vec):
    _best = int(np.argmax(A1[-1]))
    mo.md(
        rf"""
    The last row becomes {vec(H_copy[-1], "{:.3f}")}, and the three most probable next words are
    {top_words(p_copy)}.

    **Choosing a row and choosing what to take from it are different decisions.** Queries and keys
    control which rows receive weight. So far, a selected row sends its entire original vector $x_j$.
    Give training a separate way to change that contribution:

    $$v_j=W_V^\top x_j,\qquad h_i=\sum_{{j\le i}}\alpha_{{ij}}v_j.$$

    We call $v_j$ the row's **value**. If we change $W_V$ while holding $X$, $W_Q$, and $W_K$ fixed,
    the attention weights stay exactly the same, but the vector sent by each row changes. For example,
    a row containing *sat* could supply a contribution that helps score possible places next. Copying
    its original embedding does not give training this separate control.

    Stack the values to perform every weighted sum at once:

    $$V=XW_V,\qquad H=AV.$$

    Here $V$ is the matrix of value vectors; $\mathcal V$ is still our vocabulary. Each output row
    is a weighted sum of value rows. We then read the last row with $E$ and softmax as before.

    Setting $W_V=I$ recovers the raw-row average. Other choices can transform the information being
    combined. As with queries and keys, the value map acts on the word-plus-position vector; it can
    retain useful position distinctions, but it can also discard them. Its four entries are trained
    through their effect on the prediction loss.

    The three maps therefore control **what to look for** ($W_Q$), **what to match against** ($W_K$),
    and **what to send** ($W_V$). Below, two sliders illustrate the value map with a rotation
    $\varphi$ and stretch $\gamma$; the default rotates each value a quarter turn.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    gamma1 = mo.ui.slider(0, 2, step=0.25, value=1, label="Value stretch γ", show_value=True, debounce=True)
    phi1 = mo.ui.slider(-180, 180, step=15, value=90, label="Value turn φ (degrees)", show_value=True, debounce=True)
    mo.hstack([gamma1, phi1], justify="start", gap=2)
    return gamma1, phi1


@app.cell
def _(A1, X, gamma1, phi1, turn):
    WV1 = turn(phi1.value, gamma1.value)

    V1 = X @ WV1  # one value per row
    AV1 = A1 @ V1  # row i: the weighted average of the values of rows j ≤ i
    return AV1, V1, WV1


@app.cell(hide_code=True)
def _(mo):
    value_coordinate = mo.ui.radio({"first coordinate": 0, "second coordinate": 1},
                                   value="first coordinate", label="Inspect weighted sum", inline=True)
    value_coordinate
    return (value_coordinate,)


@app.cell(hide_code=True)
def _(
    A1,
    AV1,
    V1,
    WV1,
    X,
    grid,
    mo,
    np,
    row_labels,
    score_query,
    show,
    value_coordinate,
):
    _i, _c = score_query.value, value_coordinate.value
    _terms = A1[_i, :_i+1] * V1[:_i+1, _c]
    mo.vstack([
        mo.md(r"$v_j=W_V^\top x_j$, then $(AV)_{ic}=\sum_{j\le i}\alpha_{ij}v_{jc}$. "
              f"Inspect row {_i+1}, coordinate {_c+1}, using the query selected above."),
        show(grid([X[_i]], name=f"x_{_i+1}"), "×", grid(WV1, name="W_V"), "=",
             grid([V1[_i]], name=f"v_{_i+1}")),
        show(grid(np.column_stack([A1[_i,:_i+1], V1[:_i+1,_c], _terms]), rows=row_labels[:_i+1],
                  cols=["weight α", f"value coordinate {_c+1}", "product αv"], fmt="{:.5f}",
                  mark=np.ones((_i+1,3),dtype=bool))),
        mo.md("Add the products: " + " + ".join(f"({v:.5f})" for v in _terms)
              + f" = **{AV1[_i,_c]:.5f}** (rounded display). This is one coordinate of the message received."),
        mo.accordion({"Expand the complete value and weighted-sum calculation": mo.vstack([
            show(grid(X, rows=row_labels, name="X"), "×", grid(WV1, name="W_V"), "=",
                 grid(V1, rows=row_labels, name="V")),
            show(grid(A1, name="A"), "×", grid(V1, name="V"), "=", grid(AV1, rows=row_labels, name="AV")),
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Do not throw away what we had

    Should $AV$ *replace* $X$? If it does, a row's own content survives only through the weight it happens to put
    on itself. There is also a cleaner objection. Compare the family of predictors we can express now with the
    family of section 3. With replacement, setting $W_V=0$ does not switch attention off: it gives
    $h_n=0$, all logits 0, and the uniform distribution. Recovering the input would require special routing or
    parameter choices; it is not built in. We want a direct way to preserve the original predictor.

    Adding instead of replacing fixes both:

    $$h_i = x_i + \sum_{j \le i} \alpha_{ij}\, v_j, \qquad H = X + A V$$

    Now $W_V = 0$ gives exactly the section 3 predictor, and anything else is a correction to it. The logits split
    the same way:

    $$E\,h_n = \underbrace{E\,x_n}_{\text{section 3}} + \underbrace{E\,(AV)_n}_{\text{what attention adds}}$$

    Keeping the input and adding to it is called a **residual connection**.
    """)
    return


@app.cell
def _(AV1, E, X, softmax):
    H1 = X + AV1  # every row: what it was, plus what it gathered

    z1 = E @ H1[-1]
    p1 = softmax(z1)
    return H1, p1, z1


@app.cell(hide_code=True)
def _(
    AV1,
    E,
    H1,
    X,
    f_obs,
    grid,
    last_row_mask,
    mo,
    nll,
    p0,
    p1,
    row_labels,
    show,
    text,
    top_words,
    vocab_table,
    z0,
    z1,
):
    _columns = [
        ("section 3 logit", z0, "{:.3f}"),
        ("+ e<sub>v</sub>·(AV)<sub>n</sub>", E @ AV1[-1], "{:+.3f}"),
        ("= new logit", z1, "{:.3f}"),
        ("section 3 p(v)", p0, "pct"),
        ("with attention p(v)", p1, "pct"),
    ]
    if f_obs is not None:
        _columns.append(("observed f(v)", f_obs, "pct"))
        _verdict = (
            f"On *{text}*, the loss against the observed distribution moves from **{nll(p0, f_obs):.3f}** to "
            f"**{nll(p1, f_obs):.3f}** ({'lower' if nll(p1, f_obs) < nll(p0, f_obs) else 'higher'})."
        )
    else:
        _verdict = f"The data has no observed distribution for *{text}*, so there is no loss to compare."
    mo.vstack(
        [
            show(
                grid(X, rows=row_labels, name="<i>X</i>", mark=last_row_mask(X.shape)),
                "+",
                grid(AV1, name="<i>A V</i>", mark=last_row_mask(X.shape)),
                "=",
                grid(H1, rows=row_labels, name="<i>H</i>", mark=last_row_mask(X.shape)),
            ),
            show(vocab_table(_columns)),
            mo.md(
                f"""
    The most probable next words were {top_words(p0)}. They are now {top_words(p1)}. {_verdict}

    **Nothing guarantees that this is an improvement.** $W_Q$ and $W_V$ were set by hand, and the sliders make the
    loss worse as easily as better. Set the value stretch γ to 0 and the last two probability columns coincide: the
    section 3 predictor is still in there. Finding weights that lower the loss is what training is for, and it
    would proceed exactly as in section 4, with the twelve new numbers added to $\\theta$.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Second experiment · Does exchanging information actually help?

    The original five sentences share one prefix, so they cannot test this question. Keep them intact.
    For this **separate synthetic task**, use six contexts: *the/a cat sat on a* → *mat*,
    *the/a dog sat on a* → *hat*, and *the/a bird sat on a* → *grass*. This arbitrary animal-to-place rule
    is a controlled memory task, not a claim about English. All contexts end in *a* at position 5.

    Train two models from identical initial parameters. Both use two-dimensional embeddings, the same position
    offsets, one residual head, and a separate linear readout with bias over all eleven words:

    $$q=W_Q^\top x_n,\quad K=XW_K,\quad V=XW_V,\quad
    h_n=x_n+\sum_j\alpha_jv_j,\quad p=\operatorname{softmax}(U^\top h_n+b).$$

    **A (last-row only)** masks out every key except the last: its weight there is exactly 1.
    **B (attention)** permits all five keys and learns their softmax weights. This mask is the only difference.
    Each has 67 parameter entries; A's query and key parameters have zero gradient because there is only one
    permitted key. Both train embeddings, values, and readout. An MLP on A's last row could not resolve the
    collision either: its input would still be identical for all six contexts.

    We minimize mean next-token cross-entropy on exactly these six examples, using full-batch Adam,
    learning rate 0.03, 1,200 updates, and seeds 7, 17, and 29. Both receive the same optimization budget.
    This tests ability to fit the task, **not generalization**. No weights from this experiment replace the
    hand-set weights in the continuing worked example.
    """)
    return


@app.cell
def _(E, ID, encode, np, offsets, softmax):
    SECOND_CONTEXTS = [f"{article} {animal} sat on a" for article in ("the", "a")
                       for animal in ("cat", "dog", "bird")]
    SECOND_TARGETS = np.array([ID[word] for word in ("mat", "hat", "grass") * 2])
    second_ids = np.array([encode(c) for c in SECOND_CONTEXTS])

    def comparison_init(seed):
        rng = np.random.default_rng(seed)
        return {"E": E.copy(), "Q": rng.normal(0, 0.25, (2, 2)),
                "K": rng.normal(0, 0.25, (2, 2)), "V": rng.normal(0, 0.25, (2, 2)),
                "U": rng.normal(0, 0.25, (2, len(E))), "b": np.zeros(len(E))}

    def comparison_loss_grad(params, allow_earlier, token_ids=second_ids, labels=SECOND_TARGETS):
        """Full-batch cross-entropy and analytic derivatives of the actual masked-attention model."""
        x = params["E"][token_ids] + offsets(token_ids.shape[1])
        q, k, v = x[:, -1] @ params["Q"], x @ params["K"], x @ params["V"]
        scores = np.einsum("bd,bnd->bn", q, k)
        if not allow_earlier:
            scores[:, :-1] = -np.inf
        a = softmax(scores)
        h = x[:, -1] + np.einsum("bn,bnd->bd", a, v)
        z = h @ params["U"] + params["b"]
        logp = z - np.max(z, axis=1, keepdims=True)
        logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
        p = np.exp(logp)
        loss = -float(logp[np.arange(len(labels)), labels].mean())
        dz = p.copy()
        dz[np.arange(len(labels)), labels] -= 1
        dz /= len(labels)
        grad = {"U": h.T @ dz, "b": dz.sum(axis=0)}
        dh = dz @ params["U"].T
        dv = a[:, :, None] * dh[:, None, :]
        da = np.einsum("bd,bnd->bn", dh, v)
        ds = a * (da - (da * a).sum(axis=1, keepdims=True))
        dq = np.einsum("bn,bnd->bd", ds, k)
        dk = ds[:, :, None] * q[:, None, :]
        grad["Q"] = x[:, -1].T @ dq
        grad["K"] = np.einsum("bni,bnj->ij", x, dk)
        grad["V"] = np.einsum("bni,bnj->ij", x, dv)
        dx = dk @ params["K"].T + dv @ params["V"].T
        dx[:, -1] += dh + dq @ params["Q"].T
        grad["E"] = np.zeros_like(params["E"])
        np.add.at(grad["E"], token_ids, dx)
        return loss, grad, p, a

    def comparison_train(seed, allow_earlier, steps=1200):
        params = comparison_init(seed)
        first = {k: np.zeros_like(v) for k, v in params.items()}
        second = {k: np.zeros_like(v) for k, v in params.items()}
        losses = []
        for step in range(steps + 1):
            loss, grad, probabilities, weights = comparison_loss_grad(params, allow_earlier)
            losses.append(loss)
            if step == steps:
                break
            for key in params:
                first[key] = 0.9 * first[key] + 0.1 * grad[key]
                second[key] = 0.999 * second[key] + 0.001 * grad[key] ** 2
                m = first[key] / (1 - 0.9 ** (step + 1))
                v = second[key] / (1 - 0.999 ** (step + 1))
                params[key] -= 0.03 * m / (np.sqrt(v) + 1e-8)
        return {"params": params, "losses": np.array(losses), "p": probabilities, "attention": weights}

    comparison_runs = {(seed, name): comparison_train(seed, allow)
                       for seed in (7, 17, 29) for name, allow in (("A", False), ("B", True))}
    return SECOND_CONTEXTS, SECOND_TARGETS, comparison_runs


@app.cell(hide_code=True)
def _(mo):
    comparison_seed = mo.ui.dropdown({str(seed): seed for seed in (7, 17, 29)}, value="7", label="Inspect training seed")
    return (comparison_seed,)


@app.cell(hide_code=True)
def _(
    BLUE,
    ORANGE,
    SECOND_CONTEXTS,
    SECOND_TARGETS,
    VOCAB,
    comparison_runs,
    comparison_seed,
    grid,
    mo,
    new_fig,
    np,
    show,
):
    _fig, _ax = new_fig(1, width=6.5, height=3.0)
    _summary = []
    _selected_seed = comparison_seed.value
    for (_seed, _name), _run in comparison_runs.items():
        _accuracy = float(np.mean(_run["p"].argmax(axis=1) == SECOND_TARGETS))
        _summary.append([str(_seed), _name, f'{_run["losses"][0]:.4f}', f'{_run["losses"][-1]:.6f}',
                         f'{_accuracy:.0%}'])
        _ax.plot(_run["losses"], color=ORANGE if _name == "A" else BLUE,
                 alpha=1 if _seed == _selected_seed else 0.25,
                 label=f"Model {_name}, seed {_seed}" if _seed == _selected_seed else None)
    _ax.axhline(np.log(3), color=ORANGE, linestyle=":", label="A's lower bound: log 3")
    _ax.set_xlabel("optimizer updates")
    _ax.set_ylabel("mean cross-entropy (nats)")
    _ax.legend(frameon=False)
    _a, _b = comparison_runs[_selected_seed, "A"], comparison_runs[_selected_seed, "B"]
    _rows = [[c, VOCAB[y], VOCAB[pa.argmax()], f"{pa[y]:.2%}", VOCAB[pb.argmax()], f"{pb[y]:.2%}"]
             for c, y, pa, pb in zip(SECOND_CONTEXTS, SECOND_TARGETS, _a["p"], _b["p"])]
    _successes = [seed for seed in (7, 17, 29)
                  if np.array_equal(comparison_runs[seed, "B"]["p"].argmax(axis=1), SECOND_TARGETS)
                  and comparison_runs[seed, "B"]["losses"][-1] < 0.05]
    _failures = [seed for seed in (7,17,29) if seed not in _successes]
    _failure_note = " ".join(
        f"Seed {seed}: loss fell by only {comparison_runs[seed,'B']['losses'][-201] - comparison_runs[seed,'B']['losses'][-1]:.3g} "
        f"in the final 200 updates; mean attention to the animal position is {comparison_runs[seed,'B']['attention'][:,1].mean():.2%}."
        for seed in _failures)
    _passed = len(_successes) == 3
    mo.vstack([
        comparison_seed,
        _fig,
        show(grid(_summary, cols=["seed", "model", "initial loss", "final loss", "training accuracy"],
                  bracket=False, left=(1,))),
        show(grid(_rows, cols=[f"context (seed {_selected_seed})", "target", "A predicts", "A p(target)", "B predicts", "B p(target)"],
                  bracket=False, left=(0, 1, 2, 4))),
        mo.md("**Result:** " + (
            "B fits all six examples for all three seeds with loss below 0.05. The architectural change solves "
            "this constructed training task." if _passed else
            f"B fits the task in {len(_successes)} of 3 seeds (successful seeds: {_successes}). Its successful runs demonstrate "
            "the capability, but optimization does not reliably find it. " + _failure_note
            + " Inspect a failed seed below: weak routing to the distinguishing token can leave nearly identical predictions. "
            "This is an observed optimization failure, not a proof of its precise cause."
        ) + r""" A must return one distribution for all six inputs, regardless of training. With three equally
        frequent targets its best cross-entropy has infimum $\log 3$ and its deterministic accuracy cannot
        exceed $1/3$. The remaining eight words keep a positive softmax probability at finite logits, so the
        infimum is approached rather than attained. B can read the animal token. This result does not show
        attention is the only solution: even a suitable pooled-context model could solve this task."""),
        mo.accordion({
            f"Inspect learned attention and every trained parameter (seed {_selected_seed})": mo.vstack([
                show(grid(_b["attention"], rows=SECOND_CONTEXTS,
                          cols=[f"position {i + 1}" for i in range(5)], name="B's trained attention", fmt="{:.4f}")),
                *[show(grid(_run["params"][_key] if _run["params"][_key].ndim == 2 else [_run["params"][_key]],
                            name=f"Model {_name}: {_key}", fmt="{:.4f}"))
                  for _name, _run in (("A", _a), ("B", _b)) for _key in _run["params"]],
            ]),
            f"All eleven next-token probabilities (seed {_selected_seed})": mo.vstack([
                show(grid(_run["p"], rows=SECOND_CONTEXTS, cols=VOCAB, name=f"Model {_name}", fmt="{:.4f}"))
                for _name, _run in (("A", _a), ("B", _b))]),
        }),
        mo.md("Now return to the **hand-set** worked example. The comparison above tests whether trained information "
              "exchange helps; its displayed losses and predictions determine the answer."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hidden states

    The rows of $H$ are a new kind of object. $h_i$ is not an input: no lookup in a table produces it. It is not an
    output: it is not a probability. It is a vector the model computes on the way from one to the other. Such
    vectors are called **hidden states**, or **representations**: $h_i$ is the model's representation of position
    $i$ *in its context*.

    Keep the two apart. $x_i$ is the initial representation, a function of one word and one position. $h_i$ is a
    function of every row up to $i$. Everything the model will ever say about the next word has to pass through
    $h_n$, which makes $h_n$ the natural place to look when we ask, in Part II, what the model "knows".
    """)
    return


@app.cell(hide_code=True)
def _(A1, AV1, K1, Q1, X, head_picture, mo, new_fig, row_labels):
    _fig, _ax = new_fig(1, width=5.4, height=4.8)
    head_picture(_ax, X, K1, Q1, A1, AV1, row_labels, title="The last row's view of the head")
    mo.hstack(
        [
            _fig,
            mo.md(
                """
                **The same step as a picture.** Each dot is the key of one row; with $W_K = I$ a key is the row
                itself. The orange arrow is the direction of the last row's query. A key that lies far along that
                direction gets a high score, and the dot's size is the attention weight it ends up with. The black
                arrow is the update the last row receives: from $x_n$ to $h_n = x_n + (AV)_n$.

                Turn θ and the orange arrow swings round to pick out other rows. Raise β and the largest dot takes
                everything. Turn φ and the black arrow rotates without any dot changing size: the weights say where
                to read, the values say what is brought back.
                """
            ),
        ],
        justify="start",
        align="center",
        gap=2,
        widths=[1.1, 1],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 · The whole computation so far, live

    Sections 2 to 8 in one place. Change the context or a slider, and every number below recomputes from the same
    arrays the sections above used. Blue marks the cells that produce the prediction: the last row, the weights it
    reads with, and what it reads.
    """)
    return


@app.cell(hide_code=True)
def _(beta1, gamma1, mo, phi1, theta1):
    mo.hstack([beta1, theta1, gamma1, phi1], justify="start", gap=1.5, wrap=True)
    return


@app.cell(hide_code=True)
def _(
    A1,
    AV1,
    E,
    H1,
    K1,
    P,
    Q1,
    S1,
    V1,
    WK1,
    WQ1,
    WV1,
    X,
    X_tok,
    f_obs,
    grid,
    ids,
    last_row_mask,
    mo,
    n,
    nll,
    np,
    p0,
    p1,
    row_labels,
    show,
    text,
    vocab_table,
    words,
    z0,
    z1,
):
    _nums = [f"{_i + 1} {_w}" for _i, _w in enumerate(words)]
    _upper = np.triu(np.ones((n, n)), 1)
    _last = last_row_mask(X.shape)
    _all = np.ones(X.shape, dtype=bool)
    _columns = [
        ("baseline p(v)", p0, "pct"),
        ("new logit", z1, "{:.3f}"),
        ("new p(v)", p1, "pct"),
    ]
    if f_obs is not None:
        _columns.append(("observed f(v)", f_obs, "pct"))
        _loss = (
            f"**8 · Against the data.** Loss of the baseline: {nll(p0, f_obs):.3f}. Loss with attention: "
            f"{nll(p1, f_obs):.3f}. The floor for this context is {nll(f_obs, f_obs):.3f}."
        )
    else:
        _loss = (
            f"**8 · Against the data.** *{text}* does not occur in the five sentences, so there is no observed "
            "distribution to compare with."
        )
    mo.accordion(
        {
            "1 · Words and ids": mo.vstack([
            show(grid([words, [str(_k) for _k in ids]], rows=["word", "id"],
                      cols=[f"position {_i + 1}" for _i in range(n)], bracket=False)),
            ]),
            "2–3 · Embeddings, offsets and X": mo.vstack([
            show(
                grid(X_tok, rows=row_labels, name="<i>E</i>[ids]"), "+", grid(P, name="<i>P</i>"), "=",
                grid(X, rows=row_labels, name="<i>X</i>", mark=_last),
            ),
            ]),
            "4 · Baseline logits and probabilities": mo.vstack([
            show(
                grid([X[-1]], name="<i>x<sub>n</sub></i>", mark=np.ones((1, 2), dtype=bool)),
                "→",
                vocab_table([("embedding e<sub>v</sub>", E, "vec"), ("baseline logit e<sub>v</sub>·x<sub>n</sub>", z0, "{:.3f}"),
                             ("baseline p(v)", p0, "pct")]),
            ),
            ]),
            "5 · Attention scores and weights": mo.vstack([
            show(
                grid(WQ1, name="<i>W<sub>Q</sub></i>"), grid(WK1, name="<i>W<sub>K</sub></i>"),
                grid(Q1, rows=row_labels, name="<i>Q</i> = <i>X W<sub>Q</sub></i>", mark=_last),
                grid(K1, name="<i>K</i> = <i>X W<sub>K</sub></i>", mark=_all),
            ),
            show(
                grid(S1, rows=row_labels, cols=_nums, name="<i>Q K</i><sup>⊤</sup>", mark=last_row_mask(S1.shape),
                     dim=_upper),
                "→",
                grid(A1, rows=row_labels, cols=_nums, name="<i>A</i>", mark=last_row_mask(A1.shape), dim=_upper),
            ),
            ]),
            "6 · Values and updated representations": mo.vstack([
            show(
                grid(WV1, name="<i>W<sub>V</sub></i>"),
                grid(V1, rows=row_labels, name="<i>V</i> = <i>X W<sub>V</sub></i>", mark=_all),
                grid(AV1, name="<i>A V</i>", mark=_last),
                "→",
                grid(X, rows=row_labels, name="<i>X</i>", mark=_last), "+", grid(AV1, name="<i>A V</i>", mark=_last),
                "=",
                grid(H1, rows=row_labels, name="<i>H</i>", mark=_last),
            ),
            ]),
            "7 · New logits and probabilities": mo.vstack([
            show(
                grid([H1[-1]], name="<i>h<sub>n</sub></i>", mark=np.ones((1, 2), dtype=bool)),
                "→",
                vocab_table(_columns),
            ),
            mo.md(_loss),
            ]),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Asking more than one question per row

    The construction of sections 6 to 8, with its three matrices, is called an attention **head**. Look at what one
    head gives the last row: one row of weights, and one matrix $W_V$ applied to whatever those weights select.
    That is one question with one way of using the answer.

    Predicting the word after *the cat sat on the* plausibly needs two things from the context: what is being done
    (*sat on*) and who is doing it (*cat*). Multiple heads provide separate routes for these signals:

    - The score is linear in the key, $s_{nj}=q_n^{\top}k_j$. For exactly opposite keys $k$ and $-k$,
      their scores have opposite signs. A negative cosine alone does **not** imply this restriction: a query
      can score both non-antipodal keys positively. Separate queries allow separate scoring preferences.
    - Even with the weight split between the two rows, both values pass through the same $W_V$. The message that
      *sat* should send and the message that *cat* should send are produced by one matrix.

    This does not prove that one head fails on this corpus. It motivates another design choice: run several heads side by side, each with its own $W_Q$, $W_K$, $W_V$, and add
    all their updates to the same row:

    $$H = X + A^{(1)} V^{(1)} + A^{(2)} V^{(2)}$$

    Here is a second head with its own four sliders. At the default it is the first head with both quarter turns
    reversed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    beta2 = mo.ui.slider(0, 6, step=0.5, value=2, label="Head 2: stretch β", show_value=True, debounce=True)
    theta2 = mo.ui.slider(-180, 180, step=15, value=-90, label="turn θ", show_value=True, debounce=True)
    gamma2 = mo.ui.slider(0, 2, step=0.25, value=1, label="value stretch γ", show_value=True, debounce=True)
    phi2 = mo.ui.slider(-180, 180, step=15, value=-90, label="value turn φ", show_value=True, debounce=True)
    mo.hstack([beta2, theta2, gamma2, phi2], justify="start", gap=1.5, wrap=True)
    return beta2, gamma2, phi2, theta2


@app.cell
def _(
    AV1,
    E,
    WK1,
    WQ1,
    WV1,
    X,
    beta2,
    causal_softmax,
    gamma2,
    np,
    phi2,
    softmax,
    theta2,
    turn,
):
    def head(X, WQ, WK, WV):
        """Sections 6 and 7 as one function: queries, keys, values, scores, weights, and the update."""
        Q, K, V = X @ WQ, X @ WK, X @ WV
        S = Q @ K.T
        A = causal_softmax(S)
        return Q, K, V, S, A, A @ V

    WQ2, WK2, WV2 = turn(theta2.value, beta2.value), np.eye(2), turn(phi2.value, gamma2.value)
    heads = [(WQ1, WK1, WV1), (WQ2, WK2, WV2)]  # every attention weight of the model, in one list

    Q2, K2, V2, S2, A2, AV2 = head(X, *heads[1])
    H2 = X + AV1 + AV2  # both heads add to the same rows

    z2 = E @ H2[-1]
    p2 = softmax(z2)
    return A2, AV2, H2, K2, Q2, V2, WQ2, WV2, head, heads, p2


@app.cell(hide_code=True)
def _(
    A1,
    A2,
    AV1,
    AV2,
    H2,
    K1,
    K2,
    Q2,
    V2,
    WQ2,
    WV2,
    X,
    f_obs,
    grid,
    head_picture,
    last_row_mask,
    mo,
    n,
    new_fig,
    nll,
    np,
    num,
    p0,
    p1,
    p2,
    row_labels,
    show,
    text,
    top_words,
    words,
):
    _j1, _j2 = int(np.argmax(A1[-1])), int(np.argmax(A2[-1]))
    _cos = float(K1[_j1] @ K1[_j2] / (np.linalg.norm(K1[_j1]) * np.linalg.norm(K1[_j2])))
    _last = last_row_mask(X.shape)
    _both = grid(
        np.column_stack([A1[-1], A2[-1]]),
        rows=[f"{_i + 1} {_w}" for _i, _w in enumerate(words)],
        cols=["head 1", "head 2"],
        name=f"The last row's weights",
        bracket=False,
        mark=np.column_stack([np.arange(n) == _j1, np.arange(n) == _j2]),
    )
    _fig, _ax = new_fig(1, width=4.6, height=4.2)
    head_picture(_ax, X, K2, Q2, A2, AV2, row_labels, title="The last row's view of head 2")
    _opposed = (
        f"Head 1's favourite is row {_j1 + 1}, *{words[_j1]}*; head 2's is row {_j2 + 1}, *{words[_j2]}*. The "
        f"cosine between those two keys is {num(_cos, '{:+.2f}')}"
        + ". This cosine alone does not determine what scores one query could assign."
        if _j1 != _j2
        else f"With these settings both heads favour row {_j1 + 1}, *{words[_j1]}*."
    )
    _cancel = (
        " Row 1 is unchanged: it has only itself to read, and with these two value matrices its two messages "
        "cancel."
        if np.allclose(AV1[0] + AV2[0], 0)
        else ""
    )
    _loss = (
        f" Loss on *{text}*: {nll(p0, f_obs):.3f} with no attention, {nll(p1, f_obs):.3f} with head 1, "
        f"**{nll(p2, f_obs):.3f}** with both."
        if f_obs is not None
        else ""
    )
    mo.vstack(
        [
            show(
                grid(WQ2, name="<i>W<sub>Q</sub></i><sup>(2)</sup>"),
                grid(WV2, name="<i>W<sub>V</sub></i><sup>(2)</sup>"),
                grid(V2, rows=row_labels, name="<i>V</i><sup>(2)</sup>", mark=np.ones(X.shape, dtype=bool)),
                "&nbsp;",
                _both,
            ),
            show(
                grid(X, rows=row_labels, name="<i>X</i>", mark=_last),
                "+",
                grid(AV1, name="<i>A</i><sup>(1)</sup><i>V</i><sup>(1)</sup>", mark=_last),
                "+",
                grid(AV2, name="<i>A</i><sup>(2)</sup><i>V</i><sup>(2)</sup>", mark=_last),
                "=",
                grid(H2, rows=row_labels, name="<i>H</i>", mark=_last),
            ),
            mo.hstack(
                [
                    _fig,
                    mo.md(
                        f"""
    {_opposed}{_cancel}

    The most probable next words are now {top_words(p2)}.{_loss}

    As before, these are hand-set weights and the numbers are whatever they are. What changed for certain is what
    the model *can* express: two questions per row, each with its own way of using the answer. Real models run many
    heads per layer (55 in Huginn), each with queries and keys much shorter than the rows.
    """
                    ),
                ],
                justify="start",
                align="center",
                gap=2,
                widths=[1, 1.2],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11 · Votes that only add

    Hold the attention weights fixed for a moment and look at how the last row depends on what it read:

    $$E\,h_n = E\,x_n + \sum_{\text{heads}} \sum_{j \le n} \alpha_{nj}\; E\,W_V^{\top} x_j$$

    Each row that is read casts a vote on every word's logit, the vote depends on that row alone, and the votes
    add. That has a consequence. Take four contexts that differ in two places, such as *cat* or *dog* as the
    animal and *sat* or *chased* as the verb, and suppose the weights are the same in all four. Then for every word
    $v$,

    $$z_v(\text{cat}, \text{sat}) + z_v(\text{dog}, \text{chased}) = z_v(\text{cat}, \text{chased}) + z_v(\text{dog}, \text{sat})$$

    because each side adds up the same four votes. This is an additive identity for **logits with fixed weights**,
    not for probabilities after softmax. It rules out an exact AND truth table at that logit, but an additive
    score followed by a threshold can classify AND. Content-dependent attention need not satisfy the identity:
    its weights change with the input, so attention alone is already nonlinear and can model interactions.

    An MLP adds a different capability: a nonlinear transformation of the mixed information at each position,
    after the choice of where to read.

    Adding one more linear matrix cannot break the fixed-weight additive identity. If we want that extra
    transformation to be nonlinear, one simple choice is to bend a single number:

    $$a = w^{\top} h + b, \qquad \text{output } \sigma(a)\; c$$

    This unit measures one combination $a$ of the row's coordinates, bends it through a fixed curve $\sigma$, and
    uses the result as the coefficient of a vector $c$ that it writes back. The unit is called a **neuron**. $a$ is
    its **pre-activation** and $\sigma(a)$ its **activation**; $w$, $b$ and $c$ are its weights. With
    $\sigma(u) = \max(0, u)$, one neuron already computes "both":
    """)
    return


@app.cell
def _():
    def both(a, b):
        return max(0, a + b - 1)  # one neuron: w = (1, 1), bias −1

    {(a, b): both(a, b) for a in (0, 1) for b in (0, 1)}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The output is exactly 1 when both inputs are 1 and exactly 0 otherwise, which no additive scalar function
    can reproduce on all four inputs. This illustrates a useful primitive, not a necessity proof for MLPs. A sum of $k$ such neurons,
    applied to each row separately and added to it, is called an **MLP**:

    $$m(h) = \sum_{r=1}^{k} \sigma\big(w_r^{\top} h + b_r\big)\, c_r = W_{\text{out}}^{\top}\sigma\big(W_{\text{in}}^{\top}h + b\big), \qquad y = h + m(h)$$

    Ours has $k = 2$ neurons (Huginn has 17,920) and uses $\sigma = \tanh$: steep near 0 and approaching ±1 at large pre-activations.
    The weights are **hand-set**, chosen so that Part II has something to investigate:

    - Neuron 1 reads the second coordinate, shifted by a bias, and writes *against* it with strength $g$.
    - Neuron 2 reads the first coordinate and writes along it, reinforcing it a little.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    gain = mo.ui.slider(0, 4.5, step=0.1, value=2.5, label="Neuron 1's write strength g", show_value=True,
                        debounce=True)
    gain
    return (gain,)


@app.cell
def _(H2, gain, np):
    W_IN = np.array([[0.0, 4.0],
                     [4.0, 0.0]])  # column r: the direction neuron r reads
    B_IN = np.array([0.8, 0.0])  # one bias per neuron

    def w_out(g):
        return np.array([[0.00, -g / 4],  # row r: the vector neuron r writes
                         [0.25, 0.00]])

    def mlp(H, g):
        """Every row separately: two pre-activations, two activations, one written vector."""
        return np.tanh(H @ W_IN + B_IN) @ w_out(g)

    pre2 = H2 @ W_IN + B_IN  # pre-activations: one row per position, one column per neuron
    act2 = np.tanh(pre2)  # activations
    Y1 = H2 + act2 @ w_out(gain.value)  # the same as H2 + mlp(H2, gain.value)
    return B_IN, W_IN, Y1, act2, mlp, pre2, w_out


@app.cell(hide_code=True)
def _(
    B_IN,
    E,
    H2,
    W_IN,
    Y1,
    act2,
    f_obs,
    gain,
    grid,
    last_row_mask,
    mo,
    nll,
    num,
    p2,
    pre2,
    row_labels,
    show,
    softmax,
    text,
    top_words,
    vec,
    w_out,
):
    p3 = softmax(E @ Y1[-1])
    _Wo = w_out(gain.value)
    _h = H2[-1]
    _rows = [
        [
            f"neuron {_r + 1}",
            vec(W_IN[:, _r], "{:.0f}"),
            num(B_IN[_r], "{:.1f}"),
            f"({num(W_IN[0, _r], '{:.0f}')})({num(_h[0], '{:.3f}')}) + ({num(W_IN[1, _r], '{:.0f}')})"
            f"({num(_h[1], '{:.3f}')}) + {num(B_IN[_r], '{:.1f}')} = <b>{num(pre2[-1, _r], '{:.3f}')}</b>",
            f"tanh = <b>{num(act2[-1, _r], '{:.3f}')}</b>",
            vec(_Wo[_r], "{:.3f}"),
            vec(act2[-1, _r] * _Wo[_r], "{:.3f}"),
        ]
        for _r in range(2)
    ]
    _last = last_row_mask(H2.shape)
    _loss = (
        f" Loss on *{text}*: {nll(p2, f_obs):.3f} before the MLP, **{nll(p3, f_obs):.3f}** after."
        if f_obs is not None
        else ""
    )
    mo.vstack(
        [
            show(
                grid(W_IN, name="<i>W</i><sub>in</sub>"),
                grid([B_IN], name="<i>b</i>", fmt="{:.1f}"),
                grid(_Wo, name="<i>W</i><sub>out</sub>", fmt="{:.3f}"),
            ),
            show(
                grid(
                    [[label, _rows[0][i], _rows[1][i]] for i, label in enumerate(
                        ["neuron", "reads w", "bias b", "pre-activation a = w·h + b", "activation σ(a)",
                         "writes c", "contribution σ(a)·c"])][1:],
                    cols=["calculation", "neuron 1", "neuron 2"],
                    bracket=False,
                    left=(0,),
                    name=f"The two neurons on the last row, h<sub>n</sub> = {vec(_h, '{:.3f}')}",
                )
            ),
            mo.accordion({"Expand every row through the MLP": show(
                grid(H2, rows=row_labels, name="<i>H</i>, after attention", mark=_last),
                "→",
                grid(pre2, name="pre-activations", cols=["neuron 1", "neuron 2"], mark=_last),
                "→",
                grid(act2, name="activations", cols=["neuron 1", "neuron 2"], mark=_last),
                "→",
                grid(Y1 - H2, name="<i>m</i>(<i>H</i>)", mark=_last),
                "→",
                grid(Y1, rows=row_labels, name="<i>Y</i> = <i>H</i> + <i>m</i>(<i>H</i>)", mark=_last),
            )}),
            mo.md(
                f"""
    Every row goes through the same two neurons, separately: there is no traffic between rows in this step. The
    most probable next words after the MLP are {top_words(p3)}.{_loss} Once more the weights are hand-set, and
    nothing says the MLP should help here.

    **A neuron is not a coordinate of the row.** The row $h$ lives in $\\mathbb R^d$, here $d = 2$. The activations
    live in a different space, $\\mathbb R^k$, with one coordinate per neuron; that $k$ is also 2 in this toy is a
    coincidence of its size. In the activation space the axes are special, because $\\sigma$ acts on each
    coordinate separately. In the row space nothing acts coordinate by coordinate: every matrix that touches a row
    could be re-expressed in a rotated basis without changing what the model computes. That difference matters in
    Part II.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12 · Reading rows that have already read

    Attention followed by the MLP, each adding to the rows, is one **layer**. Written as a single map, with
    $\theta$ standing for all its weights:

    $$T_\theta(X) = H + m(H), \qquad H = X + \sum_{\text{heads}} A^{(h)} V^{(h)}$$
    """)
    return


@app.cell
def _(Y1, embed, gain, head, heads, mlp, np, text):
    def attend(X, heads):
        """The rows after attention: X plus the update of every head."""
        return X + sum(head(X, *weights)[-1] for weights in heads)

    def layer(X, heads, g):
        """One layer, T_θ: attention, then the MLP, each adding to the rows."""
        H = attend(X, heads)
        return H + mlp(H, g)

    {"layer() reproduces sections 6 to 11": bool(np.allclose(layer(embed(text), heads, gain.value), Y1))}
    return attend, layer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One layer still has a blind spot, and it is the limitation of section 5 one level up. In a single layer, the
    weight $\alpha_{nj}$ compares row $n$ with the *initial* row $j$, and the value $v_j = W_V^\top x_j$ is computed
    from the initial row $j$. So what *sat* sends is the same whoever sat. A row can be read, but only as the bare
    word it started as.

    The repair is to apply a layer to the output of a layer. That is possible because $T$ returns a matrix of the
    same shape it was given, which is the second reason for insisting on that in section 6. In the second pass,
    every key and value is computed from a row that has already gathered from its own context.

    Here is that claim as an experiment. Change only the first word of the context and compare the values that head
    1 sends, in a first pass and in a second.
    """)
    return


@app.cell
def _(X, Y1, embed, gain, heads, layer, words):
    text_var = " ".join([("a" if words[0] != "a" else "the")] + words[1:])  # the context with its first word changed
    X_var = embed(text_var)

    Y2 = layer(Y1, heads, gain.value)  # two passes through the same layer

    WV = heads[0][2]
    V_pass1, V_pass1_var = X @ WV, X_var @ WV  # what each row sends in the first pass
    V_pass2, V_pass2_var = Y1 @ WV, layer(X_var, heads, gain.value) @ WV  # and in the second
    return V_pass1, V_pass1_var, V_pass2, V_pass2_var, Y2, text_var


@app.cell(hide_code=True)
def _(
    V_pass1,
    V_pass1_var,
    V_pass2,
    V_pass2_var,
    grid,
    mo,
    np,
    row_labels,
    show,
    text,
    text_var,
):
    _d1 = ~np.isclose(V_pass1, V_pass1_var)
    _d2 = ~np.isclose(V_pass2, V_pass2_var)
    _labels_var = [f"{_i + 1} {_w}" for _i, _w in enumerate(text_var.split())]
    mo.vstack(
        [
            show(
                grid(V_pass1, rows=row_labels, name=f"first pass, “{text}”", mark2=_d1),
                grid(V_pass1_var, rows=_labels_var, name=f"first pass, “{text_var}”", mark2=_d1),
                "&nbsp;&nbsp;",
                grid(V_pass2, rows=row_labels, name=f"second pass, “{text}”", mark2=_d2),
                grid(V_pass2_var, rows=_labels_var, name=f"second pass, “{text_var}”", mark2=_d2),
            ),
            mo.md(
                f"Orange marks every entry that differs between the two contexts. In the first pass the only row "
                f"that can differ is the one whose word changed ({int(_d1.any(axis=1).sum())} of {len(_d1)} actually do). In the "
                f"second pass {int(_d2.any(axis=1).sum())} of {len(_d2)} rows differ. The architecture allows messages "
                "to depend on preceding context, but some weights (for example, zero value matrices) remove that effect."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A transformer

    That is the whole architecture. A **transformer** embeds the context, applies $L$ layers in a row, each with
    its own weights, reads the last row and normalizes:

    $$p(\,\cdot \mid w_1 \dots w_n) = \operatorname{softmax}\!\big(E\,[\,T_{\theta_L} \circ \dots \circ T_{\theta_1}(X)\,]_n\big)$$

    To write text, sample a word from that distribution or choose its most probable word (the greedy rule below), append it to the context, and
    run the whole computation again on the longer context. That outer loop is called **autoregressive generation**.
    Each turn of it adds one row.
    """)
    return


@app.cell
def _(E, VOCAB, embed, gain, heads, layer, np, softmax, text):
    def next_word_probs(text):
        return softmax(E @ layer(embed(text), heads, gain.value)[-1])

    def generate(text, n_words=3):
        for _ in range(n_words):
            text += " " + VOCAB[np.argmax(next_word_probs(text))]
        return text

    {"the context, continued by three words": generate(text)}
    return


@app.cell(hide_code=True)
def _(
    E,
    H1,
    H2,
    VOCAB,
    X,
    Y1,
    Y2,
    f_obs,
    grid,
    mo,
    nll,
    np,
    show,
    softmax,
    text,
    vec,
):
    _stages = [
        ("section 3: the last row of X", X[-1]),
        ("section 8: one head", H1[-1]),
        ("section 10: two heads", H2[-1]),
        ("section 11: one full layer", Y1[-1]),
        ("this section: the layer twice", Y2[-1]),
    ]
    _seen = [] if f_obs is None else [_v for _v in range(len(VOCAB)) if f_obs[_v] > 0]
    _rows = []
    for _name, _h in _stages:
        _p = softmax(E @ _h)
        _row = [_name, vec(_h, "{:.3f}"), f"{VOCAB[int(np.argmax(_p))]} {_p.max():.0%}"]
        _row += [f"{_p[_v]:.1%}" for _v in _seen]
        _row += [f"{nll(_p, f_obs):.3f}"] if f_obs is not None else []
        _rows.append(_row)
    _cols = ["what is read out", "the row", "top word"] + [f"p({VOCAB[_v]})" for _v in _seen] + (
        ["loss"] if f_obs is not None else []
    )
    _foot = None
    if f_obs is not None:
        _foot = ["observed", "", ""] + [f"{f_obs[_v]:.1%}" for _v in _seen] + [f"{nll(f_obs, f_obs):.3f}"]
    mo.vstack(
        [
            show(grid(_rows, cols=_cols, bracket=False, left=(0,), foot=_foot,
                      name=f"Every stage of Part I on “{text}”")),
            mo.md(
                """
    The generated text above is not English, because almost none of these weights were trained. Where the loss
    in the table falls from one stage to the next, the credit goes to how the hand-set weights were picked (by
    looking at the picture in section 2 and aiming the values of *sat* and *cat* at the places), not to learning.
    Move a slider and it can as easily rise. The mechanics are complete; the training is not. Training would tune
    every number in $\\theta$ by the descent of section 4, on every position of every sentence.

    We have hand-set weights for only one layer, so the last row of the table applies that same layer twice. Hold
    on to that. Using one set of weights again and again is the idea Part II starts from.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What was forced, what was chosen

    | Section | The problem | The repair | Forced or chosen |
    |---|---|---|---|
    | 2 | Words are names, not quantities | A vector per word, plus a position offset | Numbers are forced; this encoding is chosen |
    | 3 | A matrix is not a probability measure | Score the last row against $E$, then softmax | Non-negative and summing to 1 is forced; the rest is chosen |
    | 4 | The measure is far from the data | Descend the loss over the parameters | The data forces a target; the loss and the descent are chosen |
    | 5–6 | The last row ignores every earlier word | Each row takes a weighted average of earlier rows | Depending on earlier rows is forced; averaging is chosen |
    | 6 | Fixed mixing weights cannot adapt to the words | Context-dependent attention weights computed from the input | Chosen |
    | 6 | Direct dot products fix how rows are compared | Q/K parameters let training adjust the scoring rule | Chosen |
    | 7 | What is useful to send may differ from the embedding | Values | Chosen |
    | 8 | Replacement has no direct off switch | Add the update instead of replacing | Chosen |
    | 10 | One question per row | Several heads | Chosen |
    | 11 | Fixed-weight mixing is additive in the values | A nonlinear position-wise transformation | Chosen; attention is already nonlinear |
    | 12 | Rows are read as bare words | Apply a layer to the output of a layer | Chosen |

    Left out of the toy: real models rescale the rows before each step to keep their size under control, use
    queries and keys much shorter than the rows with a fourth matrix that maps values back to $d$ coordinates,
    divide the scores by $\sqrt{d_k}$, and give every layer its own weights. None of those changes the logic above.

    The model so far has 22 entries in $E$, 12 in each of two heads and 10 in the MLP: 56 numbers.
    The sliders explore only a constrained subset of these entries. Trained language
    models have billions.
    """)
    return


@app.cell(hide_code=True)
def _(
    E,
    VOCAB,
    atoms,
    embed,
    gain,
    heads,
    history_measure,
    layer,
    mo,
    mu_Z_first,
    new_fig,
    pushforward,
):
    def _layered(context_ids):
        return E @ layer(embed(" ".join(VOCAB[_k] for _k in context_ids)), heads, gain.value)[-1]

    _mu = history_measure(_layered, 3)
    _mu_Z = pushforward(_layered, _mu)
    _fig, (_before, _after) = new_fig(2, width=9.2, height=4.3)
    atoms(_before, mu_Z_first)
    _before.set_title(f"Section 3: {len(mu_Z_first):,} possible logit vectors")
    atoms(_after, _mu_Z)
    _after.set_title(f"One layer: {len(_mu_Z):,} possible logit vectors")
    mo.accordion(
        {
            "Aside: the count of section 5, repeated for the layer": mo.vstack(
                [
                    mo.md(
                        f"Of the {len(_mu):,} contexts of three words, the section 3 predictor could produce "
                        f"{len(mu_Z_first)} different logit vectors. With one layer in between, the same contexts "
                        f"produce **{len(_mu_Z):,}**. Contexts that end in the same word no longer collapse. In "
                        "both panels a dot's area is its probability when the model writes the three words itself."
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
    # Part II · Investigating a transformer

    Part I ended with a function we can run. Part II asks what happens inside it.

    That may sound like an odd question about a model we built by hand, with all 56 weights on the screen. But try
    to answer this from the weights alone: *when the last row is read, does it know whether an animal has been
    mentioned?* The weights do not say. The answer depends on what the weights do to particular inputs. For a
    trained model with billions of weights the gap is far wider: every weight is known, and inspection of them
    explains almost nothing. So the questions have to be put to the hidden states, as experiments. The toy is the
    place where every answer can be checked.

    ## 13 · What does a hidden state contain?

    Take the hidden state that Part I worked hardest to build: $h_n$, the last row after attention and before the
    MLP. Section 5 complained that the last row knew nothing about the earlier words, and sections 6 to 10 were the
    repair. Did the repair work? Is there information about the earlier words in $h_n$, in a form that can be read
    out?

    Five sentences cannot answer that. There is too little variety in them. So we set up a diagnostic experiment:
    a set of contexts, chosen by us to vary one property, used only to examine the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Diagnostic experiment. This is not training data.** 24 contexts in two families, all five words long.

    - **Family A**, *the/a* + animal + *sat on* + *the/a*, for example *a dog sat on the*. An animal has been
      named, and a place should come next.
    - **Family B**, *on* + *the/a* + place + *sat* + *the/a*, for example *on the mat sat the*. No animal yet, and
      an animal should come next.

    The property to detect is **which family a context belongs to**. Nothing in the model is trained on these
    contexts. The model stays exactly as Part I left it: hand-set.
    """),
        kind="info",
    )
    return


@app.cell
def _(attend, embed, heads, np):
    FAMILY_A = [f"{d1} {animal} sat on {d2}" for d1 in ("the", "a") for animal in ("cat", "dog", "bird")
                for d2 in ("the", "a")]
    FAMILY_B = [f"on {d1} {place} sat {d2}" for d1 in ("the", "a") for place in ("mat", "hat", "grass")
                for d2 in ("the", "a")]
    DIAG = FAMILY_A + FAMILY_B
    diag_y = np.array([1] * len(FAMILY_A) + [0] * len(FAMILY_B))  # 1 = family A: an animal has appeared

    diag_x = np.array([embed(c)[-1] for c in DIAG])  # the last row before any layer: the initial representation
    diag_h = np.array([attend(embed(c), heads)[-1] for c in DIAG])  # the last row after attention: the hidden state
    return DIAG, diag_h, diag_x, diag_y


@app.cell(hide_code=True)
def _(diag_h, diag_x, diag_y, mo, np):
    _values, _inverse = np.unique(diag_x, axis=0, return_inverse=True)
    _inverse = np.ravel(_inverse)
    _best = sum(
        max(int(np.sum((_inverse == _k) & (diag_y == 1))), int(np.sum((_inverse == _k) & (diag_y == 0))))
        for _k in range(len(_values))
    ) / len(diag_y)
    eye_accuracy = float(np.mean((diag_h[:, 1] < 0) == (diag_y == 1)))
    mo.md(
        rf"""
    **Before attention there is nothing to find.** Across the 24 contexts the initial last row $x_n$ takes only
    {len(_values)} distinct values, one for each possible last word, and each value occurs equally often in both
    families. So no function of $x_n$ whatever, linear or not, can identify the family more than {_best:.0%} of the
    time. That is section 5 again, as a measurement.

    **After attention** the last row depends on the whole context. The left panel below shows the 24 hidden
    states $h_n$. With the default weights, family A sits low and family B sits high, so try the simplest possible
    reading: call a context family A when the second coordinate of $h_n$ is negative. That rule is right for
    {eye_accuracy:.0%} of the 24.

    ### Features and probes

    A property of the input that can be computed from a hidden state is called a **feature** of the
    representation. The simplest features are read along a direction:

    $$s(h) = u^{{\top}} h + b$$

    positive for one class and negative for the other. Geometrically $u$ is a direction in the space the rows live
    in, and $s$ measures how far along it the state lies.

    Instead of choosing $u$ by eye, fit it. A **linear probe** is a classifier of exactly this form, trained on
    pairs (hidden state, label). Its loss is the negative log-likelihood of section 4 with two outcomes instead of
    eleven, taking $P(\text{{family A}} \mid h) = 1 / (1 + e^{{-s(h)}})$, and it is trained by the same descent. To
    test transfer to new nouns, train it on the 16 cat/dog/mat/hat contexts and test it on the 8 bird/grass
    contexts. Both sets are balanced. This is stronger than a random split but still shares the two templates.
    """
    )
    return


@app.cell
def _(DIAG, diag_h, diag_y, np):
    def train_probe(H, y, steps=3000, rate=1.0, shrink=1e-3):
        """Fit s(h) = u·h + b by gradient descent, so that 1 / (1 + exp(-s)) matches the labels y."""
        centre = H.mean(axis=0)
        u, b = np.zeros(H.shape[1]), 0.0
        for _ in range(steps):
            p = 1 / (1 + np.exp(-np.clip((H - centre) @ u + b, -60, 60)))  # the probe's probability of class 1
            u = u - rate * ((H - centre).T @ (p - y) / len(y) + shrink * u)
            b = b - rate * np.mean(p - y)
        return u, b - centre @ u

    def accuracy(H, y, u, b):
        return float(np.mean((H @ u + b > 0) == (y == 1)))

    def unit(v):
        """The same direction at length 1 (or zero, if there is no direction to speak of)."""
        length = np.linalg.norm(v)
        return v / length if length > 1e-12 else np.zeros_like(v)

    # Hold out both bird and grass across all article combinations. Each split is balanced.
    test_idx = np.array([i for i, c in enumerate(DIAG) if "bird" in c.split() or "grass" in c.split()])
    train_idx = np.array([i for i in range(len(DIAG)) if i not in test_idx])

    probe_u, probe_b = train_probe(diag_h[train_idx], diag_y[train_idx])  # the only training in this section
    probe_dir = unit(probe_u)  # the probe's direction, at unit length

    {
        "accuracy on the 16 training contexts": accuracy(diag_h[train_idx], diag_y[train_idx], probe_u, probe_b),
        "accuracy on the 8 held-out contexts": accuracy(diag_h[test_idx], diag_y[test_idx], probe_u, probe_b),
    }
    return (
        accuracy,
        probe_b,
        probe_dir,
        probe_u,
        test_idx,
        train_idx,
        train_probe,
        unit,
    )


@app.cell
def _(E, ID, diag_h, gain, mlp):
    PLACES, ANIMALS = ["mat", "hat", "grass"], ["cat", "dog", "bird"]
    d_place = E[[ID[w] for w in PLACES]].mean(axis=0) - E[[ID[w] for w in ANIMALS]].mean(axis=0)

    def place_preference(row):
        """Mean logit of the three places minus mean logit of the three animals, for a row about to be read out."""
        return row @ d_place

    diag_out = diag_h + mlp(diag_h, gain.value)  # the same 24 last rows after the MLP: what is actually read out
    diag_r = place_preference(diag_out)
    return d_place, diag_r


@app.cell(hide_code=True)
def _(
    BLUE,
    B_IN,
    DIAG,
    INK,
    MUTED,
    ORANGE,
    RULE,
    SURFACE,
    W_IN,
    arrow,
    diag_h,
    diag_r,
    diag_y,
    new_fig,
    np,
    probe_b,
    probe_dir,
    probe_u,
    test_idx,
    text,
):
    _fig, (_ax, _beh) = new_fig(2, width=9.6, height=4.5, width_ratios=[1.3, 1])
    _lim = max(1.1, float(np.abs(diag_h).max()) + 0.25)
    _ax.axhline(0, color=RULE, lw=0.8, zorder=0)
    _ax.axvline(0, color=RULE, lw=0.8, zorder=0)
    _held = np.isin(np.arange(len(DIAG)), test_idx)
    for _cls, _color, _marker, _name in [(1, BLUE, "o", "family A"), (0, ORANGE, "s", "family B")]:
        for _is_test in (False, True):
            _sel = (diag_y == _cls) & (_held == _is_test)
            _ax.scatter(diag_h[_sel, 0], diag_h[_sel, 1], s=46, marker=_marker,
                        facecolors=SURFACE if _is_test else _color, edgecolors=_color, linewidths=1.6, zorder=3,
                        label=_name + (" (held out)" if _is_test else ""))
    if np.linalg.norm(probe_u) > 1e-12:
        _foot = -probe_b * probe_u / (probe_u @ probe_u)
        _along = np.array([-probe_dir[1], probe_dir[0]])
        _ax.plot(*np.column_stack([_foot - 3 * _lim * _along, _foot + 3 * _lim * _along]), color=INK, lw=1.4, zorder=2)
        arrow(_ax, _foot, _foot + 0.45 * _lim * probe_dir, color=INK, lw=1.8, head=10, zorder=4)
        _ax.text(*(_foot + 0.55 * _lim * probe_dir), "u", fontsize=10, style="italic", ha="center", va="center")
    else:
        _ax.text(0.5, 0.95, "Zero probe direction: no separating line", transform=_ax.transAxes, ha="center")
    for _r, _style in [(0, (0, (4, 3))), (1, (0, (1, 3)))]:
        _w = W_IN[:, _r]
        _p = -B_IN[_r] * _w / (_w @ _w)
        _t = np.array([-_w[1], _w[0]]) / np.linalg.norm(_w)
        _ax.plot(*np.column_stack([_p - 3 * _lim * _t, _p + 3 * _lim * _t]), color=MUTED, lw=1, ls=_style, zorder=1)
        _ax.text(*(_p + 0.72 * _lim * _t * (-1 if _r == 0 else 1)), f"neuron {_r + 1}\nswitches here", fontsize=7.5,
                 color=MUTED, ha="center", va="center", bbox=dict(fc=SURFACE, ec="none", pad=1))
    if text in DIAG:
        _k = DIAG.index(text)
        _ax.plot(*diag_h[_k], "o", ms=13, mfc="none", mec=INK, mew=1.3, zorder=5)
    _ax.set_xlim(-_lim, _lim)
    _ax.set_ylim(-_lim, _lim)
    _ax.set_aspect("equal")
    _ax.tick_params(length=0, labelsize=8)
    _ax.legend(frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9, fontsize=7.5, loc="lower left",
               handletextpad=0.2, borderaxespad=0.3)
    _ax.set_title("The hidden state h_n of the 24 contexts, and the probe")

    _rng = np.random.default_rng(1)
    for _cls, _color, _marker, _x in [(1, BLUE, "o", 0), (0, ORANGE, "s", 1)]:
        _vals = diag_r[diag_y == _cls]
        _beh.scatter(_x + _rng.uniform(-0.13, 0.13, len(_vals)), _vals, s=40, marker=_marker, color=_color,
                     alpha=0.85, edgecolors=SURFACE, linewidths=1, zorder=3)
        _beh.plot([_x - 0.25, _x + 0.25], [_vals.mean()] * 2, color=INK, lw=1.6, zorder=4)
    _beh.axhline(0, color=RULE, lw=0.8)
    _beh.set_xticks([0, 1], ["family A\n(place should follow)", "family B\n(animal should follow)"])
    _beh.set_xlim(-0.6, 1.6)
    _beh.tick_params(length=0)
    _beh.set_ylabel("place preference r of the output")
    _beh.set_title("What the model does with it (bars: means)")
    _fig
    return


@app.cell(hide_code=True)
def _(
    B_IN,
    DIAG,
    W_IN,
    accuracy,
    diag_h,
    diag_r,
    diag_y,
    grid,
    mo,
    np,
    num,
    probe_b,
    probe_dir,
    probe_u,
    show,
    test_idx,
    train_idx,
    vec,
):
    _acc_train = accuracy(diag_h[train_idx], diag_y[train_idx], probe_u, probe_b)
    _acc_test = accuracy(diag_h[test_idx], diag_y[test_idx], probe_u, probe_b)
    _w1 = W_IN[:, 0] / np.linalg.norm(W_IN[:, 0])
    _angle = float(np.degrees(np.arccos(min(1.0, abs(float(probe_dir @ _w1))))))
    _pre1 = diag_h @ W_IN[:, 0] + B_IN[0]
    _acc_n1 = float(np.mean((_pre1 > 0) == (diag_y == 1)))
    _acc_n1 = max(_acc_n1, 1 - _acc_n1)
    _a, _b = diag_r[diag_y == 1], diag_r[diag_y == 0]
    _pairs = float(np.mean((_a[:, None] > _b[None, :]) + 0.5 * (_a[:, None] == _b[None, :])))
    _scores = diag_h @ probe_u + probe_b
    _rows = [
        [
            DIAG[_k],
            "A" if diag_y[_k] else "B",
            "held out" if _k in test_idx else "train",
            vec(diag_h[_k], "{:.3f}"),
            num(_scores[_k], "{:+.2f}"),
            "A" if _scores[_k] > 0 else "B",
            num(diag_r[_k], "{:+.3f}"),
        ]
        for _k in range(len(DIAG))
    ]
    _near = (
        f"The probe's direction lies {_angle:.0f}° from the direction neuron 1 reads. In this toy the feature sits "
        f"close to a neuron: the sign of neuron 1's pre-activation alone separates the families in {_acc_n1:.0%} "
        "of the 24 contexts. That closeness is an accident of size, with two neurons in a two-dimensional space."
        if _angle < 30
        else f"The probe's direction lies {_angle:.0f}° from the direction neuron 1 reads, and the sign of neuron "
        f"1's pre-activation alone separates the families in {_acc_n1:.0%} of the 24 contexts. This compares "
        "two candidate readings; it does not prove that no individual neuron carries the information."
    )
    _has_direction = bool(np.linalg.norm(probe_u) > 1e-12)
    _status = ("The held-out score exceeds the balanced 50% baseline; this is evidence of decoding on this split."
               if _acc_test > 0.5 else
               "The held-out score does not exceed the balanced 50% baseline; this fit does not demonstrate useful decoding.")
    if not _has_direction:
        _status += " The direction has zero norm: no unit direction or boundary exists, and a probe-direction push does nothing."
        _near = "There is no direction to compare with a neuron."
    mo.vstack(
        [
            mo.md(
                f"""
    **The trained probe** is three numbers: $u$ = {vec(probe_u)} and $b$ = {num(probe_b)}. The normalized direction (stored as zero when undefined) is {vec(probe_dir)}. It classifies {_acc_train:.0%} of its 16 training contexts and
    **{_acc_test:.0%} of the 8 held-out contexts** correctly. In the left panel, the black line is where
    $s(h) = 0$ and the arrow follows $u$, when its norm is nonzero. Filled markers were used for training; hollow ones were held out.

    Be exact about what was trained. Sections 4 and 8 trained predictors. Section 4 trained a
    *model*, the table $E$, on the five sentences. Here we trained a *probe*, on pairs of a hidden state and a
    label that we assigned. The descent adjusted $u$ and $b$, and nothing else. The model's weights were not
    touched, and the model never sees $u$. A probe is a measuring instrument that we build outside the model and
    hold against one of its hidden states. What it finds is a fact about this hand-set model on these 24
    contexts; nothing here was learned by the model. (The small `shrink` term in the code keeps $u$ finite. Without regularization, the unregularized logistic loss on perfectly separable data has no finite minimizer, for the same reason the logits of section 4 kept
    growing.)

    **A feature is not a neuron.** A neuron is a part of the model: it has weights we can look up, and its
    activation is a coordinate of the MLP's hidden vector. A feature is a pattern we go looking for, and the
    direction that reads it is whatever the search finds. {_near} With thousands of dimensions there are vastly
    more directions than neurons, and a feature generally lines up with none of them.

    ### What the probe has shown, and what it has not

    {_status}

    Family A/B is confounded with noun category, word order, and the positions of *on* and *sat*. A classifier
    could distinguish the templates without representing “a place should follow.” Holding out nouns prevents
    memorizing those exact noun identities, but does not remove the template shortcut. These are **hand-set**
    representations, not a semantic feature discovered by model training.

    It has not shown that the model *does* anything with that information. For that we have to look at behaviour.
    Define the model's **place preference** as the mean logit of the three places minus the mean logit of the three
    animals. It is a linear function of the row that is read out:

    $$r = d^{{\\top}} y_n, \\qquad d = \\tfrac13(e_\\text{{mat}} + e_\\text{{hat}} + e_\\text{{grass}}) - \\tfrac13(e_\\text{{cat}} + e_\\text{{dog}} + e_\\text{{bird}})$$

    If the model used the family the way English does, $r$ would be clearly higher in family A than in family B.
    The right panel shows $r$ for the same 24 contexts, computed from the output of the layer. The means are
    {num(_a.mean(), "{:+.3f}")} for family A and {num(_b.mean(), "{:+.3f}")} for family B, and an A context has the
    higher $r$ in {_pairs:.0%} of the 144 A–B pairs (50% means chance pairwise ranking, not statistical independence; ties count half). Compare that with the
    probe, which names the family correctly for {accuracy(diag_h, diag_y, probe_u, probe_b):.0%} of the 24 at
    $h_n$. What can be read from a hidden state and what the model's output reflects are two separate
    measurements. **Decodable is not the same as used.**
    """
            ),
            mo.accordion(
                {
                    "All 24 contexts: hidden state, probe reading, place preference": show(
                        grid(
                            _rows,
                            cols=["context", "family", "probe saw it in", "h_n", "s(h) = u·h + b", "probe says",
                                  "place preference r"],
                            bracket=False,
                            left=(0,),
                        )
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(attend, embed, heads, np, probe_b, probe_u):
    # Exchange noun categories while keeping each template and both article slots unchanged.
    CONTROL_CONTEXTS = ([f"{a} {noun} sat on {b}" for a in ("the", "a")
                         for noun in ("mat", "hat", "grass") for b in ("the", "a")]
                        + [f"on {a} {noun} sat {b}" for a in ("the", "a")
                           for noun in ("cat", "dog", "bird") for b in ("the", "a")])
    control_template = np.array([1] * 12 + [0] * 12)
    control_noun = 1 - control_template  # animal present; opposed to the template label in these controls
    control_h = np.array([attend(embed(c), heads)[-1] for c in CONTROL_CONTEXTS])
    control_pred = (control_h @ probe_u + probe_b > 0).astype(int)
    return CONTROL_CONTEXTS, control_noun, control_pred, control_template


@app.cell(hide_code=True)
def _(
    CONTROL_CONTEXTS,
    control_noun,
    control_pred,
    control_template,
    grid,
    mo,
    np,
    show,
):
    _template_acc = float(np.mean(control_pred == control_template))
    _noun_acc = float(np.mean(control_pred == control_noun))
    mo.accordion({"Control experiment: noun category versus word-order template": mo.vstack([
        mo.md(f"""
        In the original diagnostic set, “animal present” and “template A” always agree. Now exchange noun
        categories: *the mat sat on the* has template A with a place noun; *on the cat sat the* has template B
        with an animal noun. Articles, length and last word are balanced exactly as before. These are deliberately
        unusual strings, **not new language-model training data or English next-word labels**.

        Keep the fitted probe frozen. On these 24 controls its predictions agree with **template** in
        {_template_acc:.0%} of cases and with **noun category** in {_noun_acc:.0%}. Those labels oppose each
        other here. This tests which cue the reading follows under a controlled distribution shift; it does
        not establish a semantic concept or prove the absence of other shortcuts. Inspect the rows rather
        than inferring “the model understands what should come next” from the original accuracy.
        """),
        show(grid([[c, "A" if t else "B", "animal" if y else "place", "A" if p else "B"]
                   for c,t,y,p in zip(CONTROL_CONTEXTS, control_template, control_noun, control_pred)],
                  cols=["control context", "template", "noun category", "frozen probe says"],
                  bracket=False, left=(0,1,2,3))),
    ])})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14 · Does the model use it? Interventions

    Observation alone cannot settle whether a direction matters to the model. Two things can vary together without
    one causing the other. The remedy is the one every experimental science uses: change one thing ourselves and
    see what follows. Take the hidden state, move it along the probe's direction, and rerun everything downstream:

    $$h' = h + \alpha\,\hat u, \qquad y' = h' + m(h'), \qquad r' = d^{\top} y'$$

    where $\hat u = u / \lVert u \rVert$ when the norm is nonzero (otherwise we use a zero, no-op edit).
    This is a **causal intervention**. Nothing else in the forward pass is
    touched, so whatever changes in the output was caused by the change we made.

    There is a naive prediction to test it against. If $\hat u$ were simply "the family, as the model stores it",
    and nothing downstream reacted to the move, then $r$ would change in a straight line, by $\alpha\,d^{\top}\hat u$.
    Pick a context and a push.
    """)
    return


@app.cell(hide_code=True)
def _(DIAG, mo, text):
    push_context = mo.ui.dropdown(
        ([text] if text not in DIAG else []) + DIAG,
        value=text,
        label="Context to intervene on:",
    )
    push_alpha = mo.ui.slider(-1.5, 1.5, step=0.05, value=0.0, label="Push α along the probe direction",
                              show_value=True, full_width=True)
    mo.vstack([push_context, push_alpha])
    return push_alpha, push_context


@app.cell
def _(
    attend,
    d_place,
    diag_h,
    embed,
    gain,
    heads,
    mlp,
    np,
    probe_b,
    probe_dir,
    probe_u,
    push_context,
    unit,
):
    def after_mlp(h, g):
        """Everything downstream of the hidden state h: the MLP adds to it, and the result is what gets read out."""
        return h + mlp(h[None, :], g)[0]

    def jacobian(func, x, eps=1e-5):
        """The derivative of func at x by nudging: column j is the response to a nudge of coordinate j."""
        return np.column_stack([(func(x + eps * v) - func(x - eps * v)) / (2 * eps) for v in np.eye(len(x))])

    h_site = attend(embed(push_context.value), heads)[-1]  # the hidden state we intervene on

    alphas = np.linspace(-1.5, 1.5, 121)
    pushed = h_site + alphas[:, None] * probe_dir  # h' for every push size
    probe_line = pushed @ probe_u + probe_b  # what the probe reads at each h'
    r_actual = np.array([d_place @ after_mlp(h, gain.value) for h in pushed])  # recomputed downstream
    r_naive = d_place @ after_mlp(h_site, gain.value) + alphas * (d_place @ probe_dir)  # if nothing reacted

    def best_direction(h, g):
        """The direction at h that raises r fastest: the readout direction d, carried back through the MLP."""
        return unit(jacobian(lambda v: after_mlp(v, g), h).T @ d_place)

    cos_all = np.array([best_direction(h, gain.value) @ probe_dir
                        if np.linalg.norm(probe_dir) > 0 and np.linalg.norm(best_direction(h, gain.value)) > 0
                        else np.nan for h in diag_h])
    return (
        after_mlp,
        alphas,
        best_direction,
        cos_all,
        h_site,
        jacobian,
        probe_line,
        r_actual,
        r_naive,
    )


@app.cell(hide_code=True)
def _(
    BLUE,
    B_IN,
    E,
    ID,
    INK,
    MUTED,
    ORANGE,
    RULE,
    SURFACE,
    VOCAB,
    W_IN,
    after_mlp,
    alphas,
    best_direction,
    cos_all,
    d_place,
    diag_r,
    diag_y,
    gain,
    grid,
    h_site,
    mo,
    new_fig,
    np,
    num,
    probe_b,
    probe_dir,
    probe_line,
    probe_u,
    push_alpha,
    push_context,
    r_actual,
    r_naive,
    show,
    softmax,
    vec,
):
    _a = push_alpha.value
    _g = gain.value
    _h0, _h1 = h_site, h_site + _a * probe_dir
    _s0, _s1 = _h0 @ probe_u + probe_b, _h1 @ probe_u + probe_b
    _y0, _y1 = after_mlp(_h0, _g), after_mlp(_h1, _g)
    _r0, _r1 = d_place @ _y0, d_place @ _y1
    _p_before, _p_after = softmax(E @ _y0), softmax(E @ _y1)
    _norm = float(np.linalg.norm(probe_u))
    _flip = -_s0 / _norm if _norm > 1e-12 else None
    _flip_note = "The probe direction is zero: its score cannot change and no flip is defined."
    if _flip is not None:
        _r_flip = d_place @ after_mlp(_h0 + _flip * probe_dir, _g)
        _flip_note = (f"The score crosses zero at α = {_flip:+.2f} "
                      + ("(inside the displayed range). " if abs(_flip) <= 1.5 else "(outside the displayed range). ")
                      + f"At that push the recomputed place preference changes by {_r_flip - _r0:+.3f}. "
                      "The input context and its label have not changed; only the probe verdict has.")
    _pre = _h0 @ W_IN[:, 0] + B_IN[0]
    _slope = 1 - _g * (1 - np.tanh(_pre) ** 2)
    _best = best_direction(_h0, _g)

    _fig, (_ps, _pr, _pc) = new_fig(3, width=9.8, height=3.5, width_ratios=[1, 1.25, 0.8])
    _ps.axhline(0, color=RULE, lw=0.8)
    _ps.plot(alphas, probe_line, color=INK, lw=2)
    _ps.plot([_a], [_s1], "o", ms=8, color=INK, mec=SURFACE, mew=1.5, zorder=4)
    if _flip is not None and abs(_flip) < 1.5:
        _ps.axvline(_flip, color=MUTED, lw=1, ls=(0, (4, 3)))
    _ps.set_xlabel("push α")
    _ps.set_title("What the probe reads")
    _ps.text(0.03, 0.95, "above 0: family A", transform=_ps.transAxes, fontsize=8, color=MUTED, va="top")
    _ps.text(0.03, 0.05, "below 0: family B", transform=_ps.transAxes, fontsize=8, color=MUTED, va="bottom")

    _pr.axhline(_r0, color=RULE, lw=0.8)
    _pr.plot(alphas, r_naive, color=MUTED, lw=1.4, ls=(0, (4, 3)), label="if nothing downstream reacted")
    _pr.plot(alphas, r_actual, color=BLUE, lw=2.2, label="recomputed through the MLP")
    _pr.plot([_a], [_r1], "o", ms=8, color=INK, mec=SURFACE, mew=1.5, zorder=4)
    if _flip is not None and abs(_flip) < 1.5:
        _pr.axvline(_flip, color=MUTED, lw=1, ls=(0, (4, 3)))
    _pr.set_xlabel("push α")
    _pr.set_title("What the model does: place preference r")
    _pr.legend(frameon=False, fontsize=8, loc="upper left")

    _rng = np.random.default_rng(2)
    for _cls, _color, _marker in [(1, BLUE, "o"), (0, ORANGE, "s")]:
        _vals = cos_all[diag_y == _cls]
        _pc.scatter(_vals, _cls + _rng.uniform(-0.15, 0.15, len(_vals)), s=34, marker=_marker, color=_color,
                    alpha=0.85, edgecolors=SURFACE, linewidths=1)
    _pc.axvline(0, color=RULE, lw=0.8)
    _pc.set_xlim(-1.08, 1.08)
    _pc.set_ylim(-0.6, 1.6)
    _pc.set_yticks([1, 0], ["A", "B"])
    _pc.set_xlabel("cosine")
    _pc.set_title("Probe direction against\nthe locally best direction")
    for _axis in (_ps, _pr, _pc):
        _axis.tick_params(length=0)

    _other = 1 - int(_s0 > 0)
    _gap = diag_r[diag_y == _other].mean() - diag_r[diag_y == int(_s0 > 0)].mean()
    _rows = [
        ["hidden state h", vec(_h0, "{:.3f}"), vec(_h1, "{:.3f}")],
        ["probe reading s(h)", num(_s0, "{:+.2f}") + (" (A)" if _s0 > 0 else " (B)"),
         num(_s1, "{:+.2f}") + (" (A)" if _s1 > 0 else " (B)")],
        ["neuron activations", vec(np.tanh(_h0 @ W_IN + B_IN), "{:.3f}"), vec(np.tanh(_h1 @ W_IN + B_IN), "{:.3f}")],
        ["row read out, y", vec(_y0, "{:.3f}"), vec(_y1, "{:.3f}")],
        ["place preference r", num(_r0, "{:+.3f}"), num(_r1, "{:+.3f}")],
        ["P(mat or hat or grass)", f"{sum(_p_before[ID[w]] for w in ('mat','hat','grass')):.2%}",
         f"{sum(_p_after[ID[w]] for w in ('mat','hat','grass')):.2%}"],
        ["most probable next token", VOCAB[_p_before.argmax()], VOCAB[_p_after.argmax()]],
    ]
    mo.vstack(
        [
            _fig,
            show(grid(_rows, cols=["", "untouched", f"pushed by α = {num(_a, '{:+.2f}')}"], bracket=False, left=(0,),
                      name=f"“{push_context.value}”")),
            mo.md(
                f"""
    **Reading the figure.** The probe score is affine in the push. {_flip_note}
    Middle: the dashed line assumes the added vector passes through unchanged; the blue curve recomputes the
    actual MLP and readout. At the selected α = {_a:+.2f}, the measured change in $r$ is {_r1 - _r0:+.3f}.
    This is an output-logit contrast, not accuracy or the total probability of a place token. The table also
    recomputes the total place probability and the most probable next token; those need not change together.

    **Why the curve bends.** The MLP sits between the hidden state and the output. With the default weights,
    neuron 1 reads a direction close to the one we are pushing along (section 13) and writes against it. For a small push the second coordinate of the
    output moves by a factor $1 - g\\,(1 - \\tanh^2 a_1)$ times the push, where $a_1$ is neuron 1's pre-activation.
    For this context $a_1$ = {num(_pre, "{:.3f}")}, so the factor is 1 − {num(_g, "{:.1f}")}·(1 −
    {num(np.tanh(_pre) ** 2, "{:.3f}")}) = **{num(_slope, "{:+.3f}")}**. Where the neuron is saturated the factor
    is near 1 and the push passes through. When $g(1-\\tanh^2 a_1)>1$, the factor
    is negative, and the push comes out *reversed*. The same intervention has different effects in different
    states.

    **Which direction would work.** At a given state, the direction that raises $r$ fastest is the gradient of
    $r$ with respect to $h$. By the chain rule that is the readout direction $d$ carried back through the
    derivative of everything downstream, $(\\partial y / \\partial h)^{{\\top}} d$. For this context it is
    {vec(_best)}, against the probe's {vec(probe_dir)}: cosine {num(float(_best @ probe_dir), "{:+.2f}") if np.linalg.norm(_best) > 0 and np.linalg.norm(probe_dir) > 0 else "undefined"}. The
    right panel shows that cosine for all 24 contexts. It ranges from {num(cos_all.min(), "{:+.2f}")} to
    {num(cos_all.max(), "{:+.2f}")}, and in {int(np.sum(cos_all < 0))} of the 24 it is negative: there, pushing
    along the probe direction locally lowers what the best direction would raise. If either direction is zero,
    the cosine is undefined (shown as a dot), and no directional comparison is claimed.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Three conclusions about this model with its default weights, each supported by the numbers above.

    1. **A direction can be highly predictive.** The held-out accuracy above tests whether this fit reads the family; success is not guaranteed
       at every slider setting and does not identify a semantic mechanism.
    2. **Moving along it need not produce the predicted change in behaviour.** The output's response is bent,
       cancelled or reversed by what lies downstream, depending on the state.
    3. **The direction that best changes behaviour is a different object.** It is the readout direction carried
       back through the downstream computation. It depends on the state and on the weights after the site, and it
       coincides with the probe direction only where those happen to agree.

    A probe answers "what can be read here?". An intervention answers "what happens if this changes?". The second
    question is the one about mechanism, and it can only be answered by recomputing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15 · Why stop after one pass?

    The model of Part I transforms the rows a fixed number of times, once per layer, and the number of layers is
    settled before any input is seen. An easy context and a hard one get the same amount of computation. Why should
    that be? What if the model could keep refining its hidden state for as long as that helped?

    The simplest way to try uses what we already have. Take the one layer and apply it again and again:

    $$H^{(0)} = X, \qquad H^{(k+1)} = T_\theta\big(H^{(k)}\big)$$

    The weights stay the same from pass to pass. Only the state changes. Section 12 took the first two steps of
    this already.
    """)
    return


@app.cell
def _(X, gain, heads, layer, np):
    plain = [X]
    for _ in range(12):
        plain.append(layer(plain[-1], heads, gain.value))  # the same layer, applied to its own output

    {"largest entry of the state after 0, 1, 2, … passes": [round(float(np.abs(H).max()), 2) for H in plain]}
    return (plain,)


@app.cell(hide_code=True)
def _(grid, mo, np, plain, row_labels, show):
    _size = [float(np.abs(_H).max()) for _H in plain]
    _grew = _size[-1] > 10 * _size[0]
    mo.vstack([
        mo.accordion({f"Inspect H^({k})": show(grid(plain[k], rows=row_labels, fmt="{:.4f}", name=f"H^({k})"))
                      for k in range(4)}),
        mo.md(
        (
            f"**It grows rapidly in this run.** The largest entry of the state goes from {_size[0]:.2f} to {_size[4]:.2f} after four "
            f"passes and {_size[-1]:.1f} after twelve. "
            if _grew
            else f"With these weights the largest entry goes from {_size[0]:.2f} to {_size[-1]:.2f} in twelve "
            "passes. This is a finite-run observation. "
        )
        + r"""
    Residual updates can reinforce or cancel the current state. With these hand-set weights, repeated additions
    can amplify it; residual connections do not inherently force growth. A finite run does not prove divergence.

    **The repair** is to hand the prompt back in on every pass. Before applying the layer, average the current
    state with $X$:

    $$H^{(k+1)} = F\big(H^{(k)}\big), \qquad F(H) = T_\theta\big(\tfrac12 (H + X)\big)$$

    The average reinjects the prompt and attenuates the old state before the layer. It does not by itself
    guarantee boundedness or convergence for arbitrary layer weights. It is a design choice, the simplest of its kind;
    Huginn learns its mixing as a linear map of the state and the embedded prompt. With $H^{(0)} = X$, the first
    step is $F(X) = T_\theta(X)$, so iteration 1 is exactly the one-layer model of Part I. A model that reuses one
    layer this way is called a **looped** or **recurrent-depth** transformer.

    For a fixed prompt, $F$ is one fixed map from $\mathbb R^{n \times d}$ to itself, applied over and over. That
    makes the sequence $H^{(0)}, H^{(1)}, H^{(2)}, \dots$ a **discrete dynamical system**: a state, and a rule
    that produces the next state from the current one. The questions to put to it are the classical ones. Where
    does the state go? Does it settle? If it is disturbed, does the disturbance die out or grow?
    """
    ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Two loops. Keep them apart.**

    - **Generation** (section 12) is a loop over *words*. Predict, draw a word, append it, and start again on a
      context that is one row longer. Each turn commits to a word that cannot be taken back.
    - **Recurrence** (this section) is a loop over *iterations* $k$ on one fixed context. The same $n$ rows are
      transformed again and again, and no word is produced. Everything happens in the hidden state.

    A looped model that writes text runs the second loop inside the first: some number of iterations for every
    word it writes.
    """),
        kind="info",
    )
    return


@app.cell
def _(attend, mlp, mo, np):
    def loop_step(S, X, heads, g, a=1.0):
        """One iteration: feed the prompt back in, apply the layer. a < 1 takes only part of the step (damping)."""
        H = attend((S + X) / 2, heads)
        return (1 - a) * S + a * (H + mlp(H, g))

    def run_loop(X, heads, g, loops, a=1.0, seed=None):
        """Iterate from the prompt (or from a random state). Returns every state and every post-attention state."""
        S = X.copy() if seed is None else np.random.default_rng(seed).normal(size=X.shape)
        states, mids = [S], []
        for _ in range(loops):
            H = attend((S + X) / 2, heads)  # the hidden state after attention, inside this iteration
            S = (1 - a) * S + a * (H + mlp(H, g))
            mo.stop(not np.all(np.isfinite(S)) or np.max(np.abs(S)) > 1e12,
                    mo.callout(mo.md("This recurrent run exceeded the numerical safety limit (an entry larger "
                                     "than 1e12 or nonfinite). The affected calculation is stopped; no fixed point "
                                     "or stability conclusion is reported. Reduce the attention value strengths "
                                     "or change their rotations to inspect a bounded trajectory."), kind="warn"))
            states.append(S)
            mids.append(H)
        return np.array(states), np.array(mids)

    def keeps_moving(states):
        """Finite-run diagnostic: did any of the last 20 steps exceed 1e-6 in this row?"""
        tail_steps = np.linalg.norm(np.diff(states[-21:], axis=0), axis=-1)
        return np.max(tail_steps, axis=0) > 1e-6

    return keeps_moving, loop_step, run_loop


@app.cell(hide_code=True)
def _(gain, mo, n):
    loop_k = mo.ui.slider(0, 40, step=1, value=1, label="Iteration k to inspect", show_value=True, full_width=True)
    start_box = mo.ui.dropdown(
        {"the prompt X": None, "random state 1": 1, "random state 2": 2, "random state 3": 3},
        value="the prompt X",
        label="Start from",
    )
    inspect_row = mo.ui.slider(1, n, step=1, value=n, label="Row whose neurons to show", show_value=True)
    mo.vstack([loop_k, mo.hstack([gain, start_box, inspect_row], justify="start", gap=1.5, wrap=True)])
    return inspect_row, loop_k, start_box


@app.cell
def _(
    B_IN,
    E,
    W_IN,
    X,
    gain,
    heads,
    np,
    probe_b,
    probe_u,
    run_loop,
    softmax,
    start_box,
):
    K_MAX = 400

    loop_states, loop_mids = run_loop(X, heads, gain.value, K_MAX, seed=start_box.value)

    loop_moves = np.linalg.norm(loop_states[1:] - loop_states[:-1], axis=2)  # how far each row moved, per iteration
    loop_probs = softmax(loop_states[:, -1] @ E.T)  # the next-word distribution if we stopped after k iterations
    loop_acts = np.tanh(loop_mids @ W_IN + B_IN)  # every neuron activation, at every iteration, for every row
    loop_probe = loop_mids[:, -1] @ probe_u + probe_b  # the section 13 probe, held against the last row each time
    return (
        loop_acts,
        loop_mids,
        loop_moves,
        loop_probe,
        loop_probs,
        loop_states,
    )


@app.cell(hide_code=True)
def _(
    B_IN,
    VOCAB,
    W_IN,
    f_obs,
    grid,
    inspect_row,
    loop_acts,
    loop_k,
    loop_mids,
    loop_moves,
    loop_probe,
    loop_probs,
    loop_states,
    mo,
    nll,
    np,
    num,
    row_labels,
    show,
    start_box,
    top_words,
    vec,
):
    _k = loop_k.value
    _i = inspect_row.value - 1
    _cur = loop_states[_k]
    _mask = np.zeros(_cur.shape, dtype=bool)
    _mask[_i] = True
    if _k == 0:
        _matrices = show(grid(_cur, rows=row_labels, name="<i>H</i><sup>(0)</sup>, the starting state", mark=_mask))
        _neurons = "At k = 0 this is the initial embedding state: no recurrent MLP activation exists yet. A probe could be fitted here, but the displayed probe was trained after attention."
    else:
        _prev = loop_states[_k - 1]
        _matrices = show(
            grid(_prev, rows=row_labels, name=f"<i>H</i><sup>({_k - 1})</sup>", mark=_mask),
            "→",
            grid(_cur, rows=row_labels, name=f"<i>H</i><sup>({_k})</sup>", mark=_mask),
            "&nbsp;",
            grid(_cur - _prev, name=f"<i>H</i><sup>({_k})</sup> − <i>H</i><sup>({_k - 1})</sup>", fmt="{:+.3f}",
                 mark=_mask),
            grid(loop_moves[_k - 1][:, None], name="size of each row's move", fmt="{:.4f}", shape=False),
        )
        _pre = loop_mids[_k - 1, _i] @ W_IN + B_IN
        _neurons = (
            f"**Neurons of row {_i + 1} during iteration {_k}.** Hidden state after attention "
            f"{vec(loop_mids[_k - 1, _i], '{:.3f}')}; pre-activations {vec(_pre, '{:.3f}')}; activations "
            f"**{vec(loop_acts[_k - 1, _i], '{:.3f}')}**"
            + (f" (they were {vec(loop_acts[_k - 2, _i], '{:.3f}')} one iteration earlier)." if _k > 1 else ".")
            + f" **Probe of section 13 on the last row:** s = {num(loop_probe[_k - 1], '{:+.2f}')}, which reads as "
            f"family {'A' if loop_probe[_k - 1] > 0 else 'B'}."
        )

    _shown = list(range(0, 13)) + ([_k] if _k > 12 else [])
    _rows = []
    for _j in _shown:
        _total = float(np.linalg.norm(loop_states[_j] - loop_states[_j - 1])) if _j > 0 else None
        _before = float(np.linalg.norm(loop_states[_j - 1] - loop_states[_j - 2])) if _j > 1 else None
        _rows.append(
            [
                str(_j),
                "·" if _total is None else num(_total, "{:.4f}"),
                "·" if not _before else num(_total / _before, "{:.2f}"),
                vec(loop_states[_j, -1], "{:.3f}"),
                "·" if _j == 0 else vec(loop_acts[_j - 1, _i], "{:.2f}"),
                "·" if _j == 0 else num(loop_probe[_j - 1], "{:+.2f}"),
                f"{VOCAB[int(np.argmax(loop_probs[_j]))]} {loop_probs[_j].max():.0%}",
            ]
            + ([num(nll(loop_probs[_j], f_obs), "{:.3f}")] if f_obs is not None else [])
        )
    _table_mask = np.zeros((len(_rows), len(_rows[0])), dtype=bool)
    _table_mask[_shown.index(_k)] = True
    mo.vstack(
        [
            _matrices,
            mo.md(
                _neurons
                + f"\n\n**Next word if we stopped here:** {top_words(loop_probs[_k])}."
                + (f" Loss against the data: {nll(loop_probs[_k], f_obs):.3f}." if f_obs is not None else "")
            ),
            show(
                grid(
                    _rows,
                    cols=["k", "size of the step ‖H(k) − H(k−1)‖", "ratio to the step before", "last row",
                          f"activations of row {_i + 1}", "probe reading", "top next word"]
                    + (["loss"] if f_obs is not None else []),
                    bracket=False,
                    mark=_table_mask,
                    name=f"Iteration by iteration, starting from {start_box.selected_key}",
                )
            ),
        ]
    )
    return


@app.cell
def _(
    DIAG,
    accuracy,
    diag_y,
    embed,
    gain,
    heads,
    loop_step,
    np,
    probe_b,
    probe_u,
    run_loop,
    test_idx,
    train_idx,
    train_probe,
):
    K_DIAG = 20
    _runs = [run_loop(embed(c), heads, gain.value, 60) for c in DIAG]
    diag_mids = np.array([mids[:K_DIAG, -1] for _, mids in _runs])  # (24 contexts, 20 iterations, 2): h_n each time
    _candidates = [run_loop(embed(c), heads, gain.value, 1500, a=0.3)[0][-1] for c in DIAG]
    diag_residuals = np.array([np.linalg.norm(loop_step(h, embed(c), heads, gain.value) - h)
                              for c, h in zip(DIAG, _candidates)])
    diag_settled = np.array([h[-1] for h in _candidates])  # candidates; see diag_residuals before calling them fixed

    acc_fixed = [accuracy(diag_mids[test_idx, k], diag_y[test_idx], probe_u, probe_b) for k in range(K_DIAG)]  # one probe, reused
    acc_refit = [
        accuracy(diag_mids[test_idx, k], diag_y[test_idx], *train_probe(diag_mids[train_idx, k], diag_y[train_idx], steps=800))
        for k in range(K_DIAG)
    ]  # a new probe fitted at every iteration
    return K_DIAG, acc_fixed, acc_refit, diag_residuals, diag_settled


@app.cell(hide_code=True)
def _(
    AQUA,
    BLUE,
    INK,
    K_DIAG,
    MUTED,
    ORANGE,
    RULE,
    SEQ,
    SURFACE,
    VOCAB,
    acc_fixed,
    acc_refit,
    end_label,
    f_obs,
    inspect_row,
    loop_acts,
    loop_moves,
    loop_probe,
    loop_probs,
    mo,
    mpl,
    n,
    new_fig,
    nll,
    np,
    row_labels,
):
    _shown = 40
    _i = inspect_row.value - 1
    _ks = np.arange(1, _shown + 1)

    _fig1, (_mv, _pb) = new_fig(2, width=9.6, height=3.0, width_ratios=[1.25, 1])
    _im = _mv.imshow(loop_moves[:_shown].T, aspect="auto", cmap=SEQ, vmin=0,
                     vmax=max(float(loop_moves[3:_shown].max()), 0.2), extent=(0.5, _shown + 0.5, n - 0.5, -0.5))
    _mv.set_yticks(np.arange(n), row_labels)
    _mv.set_xlabel("iteration k")
    _mv.tick_params(length=0)
    _mv.set_title("How far each row moves at each iteration")
    _fig1.colorbar(_im, ax=_mv, pad=0.01, aspect=12).ax.tick_params(length=0, labelsize=8)
    _top = np.argsort(-loop_probs[1 : _shown + 1].mean(axis=0))[:3]
    for _token, _color in zip(_top, (BLUE, ORANGE, AQUA)):
        _pb.plot(np.arange(0, _shown + 1), loop_probs[: _shown + 1, _token], color=_color, lw=1.8)
        end_label(_pb, _shown, loop_probs[_shown, _token], VOCAB[_token], dx=6)
    _pb.set_xlim(0, _shown + 7)
    _pb.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    _pb.set_ylim(0, None)
    _pb.grid(axis="y", color=RULE, lw=0.6)
    _pb.tick_params(length=0)
    _pb.set_xlabel("iterations before reading out")
    _pb.set_title("Probability of three next words")

    _fig2, (_na, _pr, _ac) = new_fig(3, width=9.6, height=2.9)
    for _r, _color in [(0, BLUE), (1, ORANGE)]:
        _na.plot(_ks, loop_acts[:_shown, _i, _r], "-o", color=_color, lw=1.4, ms=3, label=f"neuron {_r + 1}")
    _na.axhline(0, color=RULE, lw=0.8)
    _na.legend(frameon=False, fontsize=8, loc="center right")
    _na.set_xlim(0, _shown + 1)
    _na.set_ylim(-1.1, 1.1)
    _na.set_xlabel("iteration k")
    _na.set_title(f"Activations of row {_i + 1}")
    _pr.axhline(0, color=MUTED, lw=0.8)
    _pr.plot(_ks, loop_probe[:_shown], "-o", color=INK, lw=1.4, ms=3)
    _pr.set_xlabel("iteration k")
    _pr.set_title("Probe reading of the last row\n(above 0: family A)")
    _kd = np.arange(1, K_DIAG + 1)
    _ac.plot(_kd, acc_refit, "-o", color=AQUA, lw=1.6, ms=3.5, mec=SURFACE, label="refitted at each k")
    _ac.plot(_kd, acc_fixed, "-o", color=INK, lw=1.6, ms=3.5, mec=SURFACE, label="the probe of section 13")
    _ac.axhline(0.5, color=RULE, lw=0.8)
    _ac.legend(frameon=False, fontsize=8, loc="lower right")
    _ac.set_xlim(0, K_DIAG + 1)
    _ac.set_ylim(-0.04, 1.04)
    _ac.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    _ac.set_xticks([1, 5, 10, 15, 20])
    _ac.set_xlabel("iteration k")
    _ac.set_title("Probe accuracy on the 8\nheld-out noun contexts")
    for _axis in (_na, _pr, _ac):
        _axis.tick_params(length=0)

    _worst = int(np.argmin(acc_fixed))
    _loss = (
        f" On this context the loss against the data is {nll(loop_probs[1], f_obs):.3f} after one iteration and "
        f"{nll(loop_probs[_shown], f_obs):.3f} after {_shown}."
        if f_obs is not None
        else ""
    )
    mo.vstack(
        [
            _fig1,
            _fig2,
            mo.md(
                f"""
    **What to look at.** Drag *k* and watch the matrix: the state changes at every iteration, by less and less or
    not. The table lists the size of each step. The neurons of a row are recomputed at every iteration from that
    row's current hidden state, so an activation is a function of the iteration as well as of the input.

    **More iterations do not mean a better prediction.**{_loss} Nothing trained these weights so that the place the
    loop goes to is a good answer. In a trained looped model that is exactly what training is for; here we are
    studying the dynamics, not the quality.

    **A probe belongs to the place and the moment it was trained on.** The probe of section 13 was fitted to
    $h_n$ during iteration 1. Held against $h_n$ at later iterations (black curve, bottom right) its accuracy is
    {acc_fixed[0]:.0%} at iteration 1, falls to {acc_fixed[_worst]:.0%} at iteration {_worst + 1}, and is
    {acc_fixed[-1]:.0%} at iteration {K_DIAG}. A probe refitted at each iteration (green) reaches
    {min(acc_refit):.0%} or better throughout on the held-out nouns. Where a refitted probe succeeds and the
    fixed probe fails, this supports a change in how the information is decoded. Where both fail, these fits
    cannot distinguish absent information from information they cannot read. Reading with one frozen probe measures two
    things at once, and they have to be separated.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16 · Does the sequence settle?

    Look at the step sizes in the table of section 15. Whether they shrink toward 0 is the first question to ask
    of any iteration.

    **Fixed points.** Suppose the sequence does converge, to some state $H^*$. Because $F$ is continuous, applying
    it to both sides of $H^{(k)} \to H^*$ gives

    $$H^* = F(H^*)$$

    A state that the map leaves unchanged is called a **fixed point**. So "the loop settles" means precisely "the
    state approaches a fixed point of $F$". To find one even when the plain loop does not settle, take only part
    of each step, $H \leftarrow (1 - a) H + a F(H)$ with $a = 0.3$. That changes how the state moves without
    changing where the fixed points are.
    """)
    return


@app.cell
def _(X, gain, heads, jacobian, keeps_moving, loop_step, np, run_loop):
    H_star = run_loop(X, heads, gain.value, 1500, a=0.3)[0][-1]  # a candidate only; certify its residual below
    residual = float(np.linalg.norm(loop_step(H_star, X, heads, gain.value) - H_star))  # Frobenius norm
    fixed_valid = bool(np.isfinite(residual) and residual < 1e-8)

    def F_flat(v):
        """F as a map on one long vector of n·d numbers, so that its derivative is an ordinary matrix."""
        return loop_step(v.reshape(X.shape), X, heads, gain.value).ravel()

    J_full = jacobian(F_flat, H_star.ravel())  # the derivative of F at H*: (n·d) × (n·d)
    eig_full = np.linalg.eigvals(J_full)
    rho = float(np.abs(eig_full).max())  # spectral radius: the largest |eigenvalue|
    stretch = float(np.linalg.norm(J_full, 2))  # the most any small displacement can grow in one step

    _ends = [run_loop(X, heads, gain.value, 400, seed=s)[0] for s in range(1, 21)]
    starts_settled = sum(not keeps_moving(st).any() for st in _ends)
    starts_same = sum(bool(np.abs(st[-1] - H_star).max() < 1e-5) for st in _ends)

    {"Frobenius residual at candidate": residual, "spectral radius": round(rho, 4), "largest stretch": round(stretch, 4)}
    return (
        H_star,
        J_full,
        eig_full,
        fixed_valid,
        residual,
        rho,
        starts_same,
        starts_settled,
        stretch,
    )


@app.cell(hide_code=True)
def _(
    H_star,
    J_full,
    X,
    eig_full,
    fixed_valid,
    gain,
    grid,
    heads,
    jacobian,
    loop_step,
    mo,
    n,
    np,
    residual,
    rho,
    row_labels,
    show,
    starts_same,
    starts_settled,
    stretch,
):
    def own_block(S, X, heads, i, g):
        """Derivative of row i's next state with respect to row i's current state."""
        def _row(x):
            S2 = S.copy()
            S2[i] = x
            return loop_step(S2, X, heads, g)[i]
        return jacobian(_row, S[i])

    row_eigs = [np.linalg.eigvals(own_block(H_star, X, heads, i, gain.value)) for i in range(n)]
    _blocks_match = bool(np.allclose(np.sort_complex(np.concatenate(row_eigs)),
                                    np.sort_complex(eig_full), atol=1e-5))
    _upper_zero = all(np.abs(J_full[2*i:2*i+2, 2*j:2*j+2]).max() < 1e-6
                      for i in range(n) for j in range(i+1, n))
    _status = ("The candidate passes the absolute residual tolerance 1e-8. We treat it as an approximate fixed "
               "point for the local numerical analysis below; this is not a rigorous root certificate."
               if fixed_valid else
               "The solver did not reach the residual tolerance 1e-8. This is not an established fixed point. "
               "Its derivative below is exploratory; no fixed-point stability verdict is inferred from it.")
    _stability = ("Locally asymptotically stable according to this numerical linearization."
                  if rho < 1 - 1e-6 else "Unstable according to this numerical linearization."
                  if rho > 1 + 1e-6 else "Near the unit-circle boundary: this numerical linearization is inconclusive.")
    _local = mo.md(rf"""
    **Why linearize?** Near an exact fixed point of a continuously differentiable map,

    $$F(H^*+\delta)=H^*+J\delta+o(\|\delta\|),\qquad J=DF(H^*).$$

    The Jacobian records how each output coordinate responds to each input coordinate. We estimate it by
    central differences. For small perturbations and a fixed number of steps, the leading response is
    $J^k\delta$; a nonlinear trajectory is not exactly this matrix power.

    If every eigenvalue has magnitude strictly below 1, the fixed point is locally asymptotically stable.
    An eigenvalue strictly outside the unit circle implies instability, not repulsion in every direction.
    Eigenvalues on the unit circle require further analysis. Transient growth and mode-dependent decay can
    occur; the spectral radius is not an exact per-step shrinkage factor.

    Here $\rho(J)={rho:.6f}$. **{_stability}** Small residual and a numerical Jacobian provide evidence,
    especially when the spectrum is well away from 1; they are not a global convergence proof.

    **Different starts.** Of 20 seeded random starts, {starts_settled} have all last 20 step norms below 1e-6;
    {starts_same} end within 1e-5 of this candidate in the maximum-entry norm. This finite sample does not prove
    uniqueness or convergence from all starts.
    """) if fixed_valid else mo.callout(mo.md(_status), kind="warn")
    mo.vstack([
        mo.md(rf"The damped solver's candidate has residual $\|F(H_c)-H_c\|_F={residual:.3e}$. {_status}"),
        show(grid(H_star, rows=row_labels, name="Candidate H_c", fmt="{:.4f}")),
        _local,
        mo.accordion({
            "Inspect the full Jacobian and its causal blocks": mo.vstack([
                mo.md(f"Flattening {n} rows of two coordinates gives a {2*n} × {2*n} derivative. "
                      f"Upper blocks are zero: `{_upper_zero}`. The block spectra match the full spectrum: "
                      f"`{_blocks_match}`. These algebraic checks apply even to an unverified candidate."),
                show(grid(J_full, name="DF(H_c)", fmt="{:+.3f}", dim=np.abs(J_full) < 5e-7)),
                mo.md("Eigenvalues: " + ", ".join(f"{v.real:+.4f}{v.imag:+.4f}i" for v in eig_full)),
            ]),
            "Optional theory: contraction mappings and their assumptions": mo.md(rf"""
            On a nonempty complete metric space (for example, a closed subset of Euclidean space), a map
            **into that same space** satisfying $\|F(a)-F(b)\|\le q\|a-b\|$ for a uniform $q<1$ has one fixed
            point, and iteration converges to it from every start in that space (Banach's theorem).

            The spectral norm $\|DF(H_c)\|_2={stretch:.4f}$ measures maximal infinitesimal stretch at one point.
            It is not a uniform bound over a region. A derivative norm strictly below 1, continuity, and an exact
            fixed point give a sufficiently small invariant contracting neighborhood. A norm above 1 rules out
            Euclidean contraction there, but need not rule out stability: transient amplification is possible.
            If $\rho(J)<1$ at an exact fixed point, a suitable equivalent norm gives a local contraction.
            None of these local statements establishes a global bound. When the candidate fails its residual
            check, we have not established the fixed point required by those local conclusions.
            """),
        }),
        mo.md("**Stability and interventions.** Near a locally asymptotically stable fixed point, sufficiently "
              "small perturbations decay. Section 18 tests how a one-off edit differs from a repeated edit."),
    ])
    return own_block, row_eigs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17 · When it does not settle

    Neuron 1 writes against what it reads, with strength $g$. So far $g$ has been a number we set and left alone.
    Sweep it. For each value of $g$, run the loop 300 times from the prompt and record where the state is on the
    last two iterations.
    """)
    return


@app.cell
def _(X, heads, keeps_moving, np, run_loop):
    G_SWEEP = np.round(np.linspace(1.5, 4.5, 41), 3)
    _tails = [run_loop(X, heads, g, 300)[0][-22:] for g in G_SWEEP]  # the end of each run
    sweep_ends = np.array([tail[-2:] for tail in _tails])  # (41 values of g, 2 states, n, d)
    sweep_moving = np.array([keeps_moving(tail) for tail in _tails])  # (41 values of g, n rows)
    return G_SWEEP, sweep_ends, sweep_moving


@app.cell(hide_code=True)
def _(mo):
    damp_a = mo.ui.slider(0.5, 1.0, step=0.05, value=1.0, label="Step size a (1 = no damping)", show_value=True,
                          debounce=True)
    damp_where = mo.ui.radio(["every row", "only the source row"], value="every row", inline=True, label="Damp:")
    return damp_a, damp_where


@app.cell
def _(
    E,
    X,
    damp_a,
    damp_where,
    gain,
    heads,
    keeps_moving,
    loop_states,
    n,
    np,
    run_loop,
    softmax,
):
    still_moving = keeps_moving(loop_states)
    source = int(np.argmax(still_moving)) if still_moving.any() else None  # the first row that has not settled

    step_sizes = np.ones((n, 1))
    if damp_where.value == "every row":
        step_sizes[:] = damp_a.value
    elif source is not None:
        step_sizes[source] = damp_a.value

    damped_states, _ = run_loop(X, heads, gain.value, 400, a=step_sizes)
    damped_moves = np.linalg.norm(damped_states[1:] - damped_states[:-1], axis=2)
    damped_probs = softmax(damped_states[:, -1] @ E.T)
    return (
        damped_moves,
        damped_probs,
        damped_states,
        source,
        step_sizes,
        still_moving,
    )


@app.cell(hide_code=True)
def _(
    BLUE,
    G_SWEEP,
    INK,
    VOCAB,
    gain,
    inspect_row,
    loop_acts,
    loop_probs,
    loop_states,
    mo,
    n,
    new_fig,
    np,
    num,
    row_labels,
    source,
    still_moving,
    sweep_ends,
    sweep_moving,
    words,
):
    _i = inspect_row.value - 1
    _fig, (_ax, _all) = new_fig(2, width=9.6, height=3.3, width_ratios=[1.25, 1])
    for _phase in (0, 1):
        _ax.plot(G_SWEEP, sweep_ends[:, _phase, _i, 1], "o", ms=3.2, color=BLUE)
    _ax.axvline(gain.value, color=INK, lw=1)
    _ax.text(gain.value, _ax.get_ylim()[1], f" g = {gain.value:.1f}", fontsize=8.5, va="top")
    _ax.set_xlabel("neuron 1's write strength g")
    _ax.set_ylabel("second coordinate")
    _ax.set_title(f"Row {_i + 1} on the last two of 300 iterations")
    _im = _all.imshow(sweep_moving.T, aspect="auto", cmap="Blues", vmin=0, vmax=1.6,
                      extent=(G_SWEEP[0], G_SWEEP[-1], n - 0.5, -0.5))
    _all.axvline(gain.value, color=INK, lw=1)
    _all.set_yticks(np.arange(n), row_labels)
    _all.set_xlabel("g")
    _all.set_title("Rows still moving at the end of the run")
    for _axis in (_ax, _all):
        _axis.tick_params(length=0)

    _moving = np.where(still_moving)[0]
    _period_two = bool(np.max(np.abs(loop_states[-20:] - loop_states[-22:-2])) < 1e-5)
    _pa, _pb = loop_probs[-2], loop_probs[-1]
    if len(_moving) == 0:
        _text = (f"**Settled at g = {gain.value:.1f}.** All last 20 step norms are below 1e-6 after 400 iterations; the next-word "
                 f"distribution at the final iteration is: “{VOCAB[_pb.argmax()]}” leads with "
                 f"{_pb.max():.0%}. Raise g (the slider is in section 15, and again below).")
        _kind = "success"
    elif _period_two:
        _rows = ", ".join(f"{_k + 1} (“{words[_k]}”)" for _k in _moving)
        _text = (f"**Numerical evidence of a period-two orbit at g = {gain.value:.1f}.** Rows {_rows} keep moving and return to the same "
                 f"state to within 1e-5 every second iteration throughout the last 20 steps. Every row before row {source + 1} has settled numerically, and a row reads "
                 f"only earlier rows, so row {source + 1} is not inheriting its motion. It is a source. Neuron 1 "
                 f"of that row alternates between activations {num(loop_acts[-2, source, 0], '{:+.2f}')} and "
                 f"{num(loop_acts[-1, source, 0], '{:+.2f}')}. ")
        if _pa.argmax() != _pb.argmax():
            _text += (f"The most probable next word **depends on when you stop**: “{VOCAB[_pa.argmax()]}” "
                      f"({_pa.max():.0%}) after 399 iterations, “{VOCAB[_pb.argmax()]}” ({_pb.max():.0%}) after "
                      "400.")
        else:
            _top = _pb.argmax()
            _text += (f"The most probable next word is “{VOCAB[_top]}” on both halves of the orbit, but its "
                      f"probability alternates between {_pa[_top]:.1%} and {_pb[_top]:.1%}.")
        _kind = "warn"
    else:
        _text = (f"**Still moving at g = {gain.value:.1f}.** After 400 iterations some rows have neither settled "
                 "nor fallen into a period-two orbit.")
        _kind = "neutral"
    mo.vstack(
        [
            _fig,
            mo.md(
                "Left: one dot per iteration. Where the two dots coincide the row has settled. Where they separate, "
                "this suggests an oscillation; two distinct final states alone do not establish a **period-two orbit**. Right: for every row, the "
                "values of g at which its last 20 steps exceed the movement tolerance. The "
                "vertical line marks the current value of g."
            ),
            mo.callout(mo.md(_text), kind=_kind),
        ]
    )
    return


@app.cell(hide_code=True)
def _(damp_a, damp_where, gain, mo):
    mo.vstack(
        [
            mo.md(
                r"""
    ### Why a row flips

    When section 16 finds a small-residual candidate, we can ask whether it is locally unstable even though
    damping reached it. A failed solve establishes neither existence nor absence of a fixed point. The table
    below labels its eigenvalues as exploratory whenever the residual check fails.

    - If both are inside the unit circle, a displacement of that row shrinks, as long as the earlier rows hold
      still.
    - If one is real and below −1, a displacement along its direction changes sign and grows at every iteration.
      Nonlinear saturation may limit that growth; a period-two orbit is one possible outcome, not a consequence
      of the eigenvalue alone. Rows that
      read it through attention alternate with it, even if their own eigenvalues are harmless.

    In the toy the eigenvalue below −1 comes from neuron 1. Where its pre-activation is near 0 it answers a
    displacement of the second coordinate with $-g$ times that displacement, so the row's block has an eigenvalue
    near $\tfrac12(1 - g)$ times what attention contributes. Which rows sit in that steep part depends on the
    text.

    **Damping** replaces the step by $H \leftarrow (1 - a)\,H + a\,F(H)$. It keeps every fixed point and moves each
    eigenvalue $\lambda$ to $1-a+a\lambda$ for uniform damping. For a real $\lambda<-1$, stability along that
    mode requires $0<a<2/(1-\lambda)$; equality is marginal. Modes with real part at least 1 cannot be made
    strictly stable this way. Row-specific damping transforms each diagonal block separately here because the
    Jacobian is block lower-triangular. Try it on every row, then on the source row alone.
    """
            ),
            mo.hstack([gain, damp_a, damp_where], justify="start", gap=1.5, wrap=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    BLUE,
    ORANGE,
    RULE,
    SEQ,
    SURFACE,
    VOCAB,
    damp_a,
    damped_moves,
    damped_probs,
    damped_states,
    end_label,
    fixed_valid,
    gain,
    grid,
    keeps_moving,
    mo,
    mpl,
    n,
    new_fig,
    np,
    row_eigs,
    row_labels,
    show,
    source,
    step_sizes,
    words,
):
    def _show_eig(value):
        return f"{value.real:+.2f}".replace("-", "−") if abs(value.imag) < 1e-7 else f"{value.real:+.2f}{value.imag:+.2f}i"

    def _verdict(values):
        if not fixed_valid:
            return "exploratory derivative only"
        if np.abs(values).max() < 1:
            return "locally stable with earlier rows fixed"
        if np.any((np.abs(values.imag) < 1e-7) & (values.real < -1)):
            return "unstable sign-alternating linear mode"
        return "unstable or marginal linearization"

    _rows = []
    for _k in range(n):
        _vals = sorted(row_eigs[_k], key=lambda _v: _v.real)
        _flip = [_v.real for _v in _vals if abs(_v.imag) < 1e-7 and _v.real < -1]
        _damped = [1 - step_sizes[_k, 0] + step_sizes[_k, 0] * _v for _v in _vals]
        _rows.append(
            [
                f"{_k + 1} {words[_k]}",
                ", ".join(_show_eig(_v) for _v in _vals),
                _verdict(row_eigs[_k]),
                f"{2 / (1 - _flip[0]):.2f}" if _flip and fixed_valid else "·",
                f"{step_sizes[_k, 0]:.2f}",
                ", ".join(_show_eig(_v) for _v in _damped),
            ]
        )

    _fig, (_mv, _co, _pr) = new_fig(3, width=9.8, height=3.2, width_ratios=[1.3, 1, 1])
    _shown = 48
    _im = _mv.imshow(damped_moves[:_shown].T, aspect="auto", cmap=SEQ, vmin=0,
                     vmax=max(float(damped_moves[4:].max()), 0.2), extent=(0.5, _shown + 0.5, n - 0.5, -0.5))
    _mv.set_yticks(np.arange(n), row_labels)
    _mv.set_xlabel("iteration")
    _mv.set_title("Movement with the damping above")
    _last = np.arange(len(damped_states) - 8, len(damped_states))
    _watch = [n - 1] if source in (None, n - 1) else [source, n - 1]
    for _k, _color in zip(_watch, (BLUE, ORANGE)):
        _co.plot(_last, damped_states[_last, _k, 1], "-o", color=_color, lw=2, ms=5, mec=SURFACE, mew=1.2)
        end_label(_co, _last[-1], damped_states[-1, _k, 1], row_labels[_k], dx=7)
    _co.set_xlim(_last[0] - 0.4, _last[-1] + 2.6)
    _seen = damped_states[_last][:, _watch, 1]
    _mid, _half = (_seen.max() + _seen.min()) / 2, max(0.7 * (_seen.max() - _seen.min()), 0.08)
    _co.set_ylim(_mid - _half, _mid + _half)
    _co.grid(axis="y", color=RULE, lw=0.6)
    _co.set_xticks(_last[::2])
    _co.set_xlabel("iteration")
    _co.set_title("Second coordinate, last 8 iterations")
    _top = np.argsort(-damped_probs[-2:].mean(axis=0))[:2]
    for _token, _color in zip(_top, (BLUE, ORANGE)):
        _pr.plot(_last, damped_probs[_last, _token], "-o", color=_color, lw=2, ms=5, mec=SURFACE, mew=1.2)
        end_label(_pr, _last[-1], damped_probs[-1, _token], VOCAB[_token], dx=7)
    _pr.set_xlim(_last[0] - 0.4, _last[-1] + 2.6)
    _pr.set_xticks(_last[::2])
    _pr.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
    _pr.set_ylim(0, max(0.3, float(damped_probs[_last][:, _top].max()) + 0.06))
    _pr.grid(axis="y", color=RULE, lw=0.6)
    _pr.set_xlabel("iterations before reading out")
    _pr.set_title("Two leading next words")
    for _axis in (_mv, _co, _pr):
        _axis.tick_params(length=0)

    _still = [f"{_k + 1}" for _k in np.where(keeps_moving(damped_states))[0]]
    mo.vstack(
        [
            show(
                grid(
                    _rows,
                    cols=["row", "its two eigenvalues", "on its own", "damping needed a*", "step size a now",
                          "eigenvalues after damping"],
                    bracket=False,
                    left=(0, 2),
                    name=f"Each row's derivative block at the candidate (verified: {fixed_valid}), g = {gain.value:.1f}",
                )
            ),
            _fig,
            mo.md(
                (
                    f"With the damping set above (a = {damp_a.value:.2f}), "
                    + ("no row is moving after 400 iterations." if not _still
                       else f"rows {', '.join(_still)} are still moving after 400 iterations.")
                )
                + " Local eigenvalues describe infinitesimal perturbations at the candidate; they do not by themselves "
                "explain a finite trajectory. Earlier rows can also drive later ones."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Aside: what replaces a fixed point (invariant measures)": mo.md(
                r"""
                When the state settles, "where the loop ends up" is a point, $H^*$. On a period-two orbit there is
                no such point, but there is still a well-defined answer to "where is the state, on average, in the
                long run?": half the time at one state $a$ of the orbit and half the time at the other, $b$. The
                measure $\mu = \tfrac12 \delta_a + \tfrac12 \delta_b$ is **invariant** under $F$: push it forward
                through $F$ and you get it back, because $F$ swaps $a$ and $b$. A fixed point is the special case
                $\mu = \delta_{H^*}$.

                The practical reading is modest. On an orbit, any quantity computed from the state has two values,
                one per phase, and its long-run average is the mean of the two. A reported probability for a
                looped model on an orbit therefore needs either the iteration count it was read at or the average
                over both phases.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## 18 · Pushing on a looped model

        Section 14 edited the last row after attention and reran the MLP and readout. Now edit the last row
        **between recurrent iterations** and run the whole loop downstream. We still measure the actual
        output contrast $r=d^\top h_n$: mean place logit minus mean animal logit.

        First verify a small-residual, locally stable candidate in section 16. Then compare two protocols:

        - **Once:** add $P$ before the first remaining iteration; run all later steps without another edit.
        - **Every iteration:** add the same $P$ before each remaining iteration, so $H_{k+1}=F(H_k+P)$.

        Compare each edited run with an unedited run from the same candidate and read out after 24 iterations.
        The immediate change before any iteration is $d^\top P$; it need not predict the eventual change.
        At a locally asymptotically stable fixed point, sufficiently small one-off edits eventually decay.
        Repeated edits keep perturbing the computation and can have a persistent effect.

        Test four directions with push norm 0.03: the held-out-evaluated **probe**, the **difference of class
        means** (both fitted only on the 16 training noun contexts), the immediate **readout gradient**, and
        the **lever**, which optimizes the linearized effect after 24 iterations. A zero vector is labeled
        as a no-op, not a unit direction. The lever is a local prediction; the actual loop tests it.
        """),
        mo.accordion({"Optional theory: accumulated intervention-response operators": mo.md(r"""
        Let $J$ be the last row's derivative block at an exact fixed point. Causality means changing that row
        cannot change earlier rows. For a one-off small edit, $\delta_T\approx J^T P$, where the superscript
        $T$ here is an integer power, not a transpose. For edits before **every** step,

        $$\delta_{k+1}\approx J\delta_k+JP,\qquad
        \Delta r_T\approx d^\top R_TP,\qquad R_T=\sum_{t=1}^T J^t.$$

        Among unit vectors the linearized effect is maximal along
        $R_T^\top d/\|R_T^\top d\|$, if this vector is nonzero. This is the lever.
        If $\rho(J)<1$, the matrix series converges to $(I-J)^{-1}J$. This is the **first-order** limiting
        response, not the exact finite-edit equilibrium. If edits were added *after* each step, the recurrence
        and the series would instead start with $P$ and $I$; intervention timing matters.

        Away from a fixed point, use the different Jacobian at each trajectory step and ordered products,
        not powers of a single matrix. A failed fixed-point solve therefore disables the numerical comparison
        below. Stability of the unedited fixed point alone does not ensure a finite edit stays in its basin.
        """)}),
    ])
    return


@app.cell
def _(
    H_star,
    X,
    d_place,
    diag_settled,
    diag_y,
    gain,
    heads,
    loop_step,
    n,
    np,
    own_block,
    train_idx,
    train_probe,
    unit,
):
    T_PUSH, PUSH_SIZE = 24, 0.03

    J_last = own_block(H_star, X, heads, n - 1, gain.value)  # the last row's own derivative at the fixed point
    powers = np.array([np.linalg.matrix_power(J_last, t) for t in range(1, T_PUSH + 1)])  # J, J², …, J^T

    loop_probe_u, loop_probe_b = train_probe(diag_settled[train_idx], diag_y[train_idx])  # fit on the training nouns only
    directions = {
        "probe direction": unit(loop_probe_u),
        "difference of class means": unit(diag_settled[train_idx][diag_y[train_idx] == 1].mean(axis=0) - diag_settled[train_idx][diag_y[train_idx] == 0].mean(axis=0)),
        "readout gradient": unit(d_place),
        "lever": unit(powers.sum(axis=0).T @ d_place),
    }

    def predicted(P):
        """First-order change of r after 1, 2, …, T iterations of pushing."""
        return np.cumsum(powers @ P, axis=0) @ d_place

    def measured(P, every=True):
        """Run the loop from the fixed point, adding P to the last row before every iteration (or only the first)."""
        S, baseline, changes = H_star.copy(), H_star.copy(), []
        for t in range(T_PUSH):
            pushed = S.copy()
            if every or t == 0:
                pushed[-1] += P
            S = loop_step(pushed, X, heads, gain.value)
            baseline = loop_step(baseline, X, heads, gain.value)
            changes.append(d_place @ (S[-1] - baseline[-1]))
        return np.array(changes)

    return (
        J_last,
        PUSH_SIZE,
        T_PUSH,
        directions,
        loop_probe_b,
        loop_probe_u,
        measured,
        predicted,
    )


@app.cell(hide_code=True)
def _(
    AQUA,
    BLUE,
    J_last,
    MUTED,
    ORANGE,
    PUSH_SIZE,
    RULE,
    SURFACE,
    T_PUSH,
    accuracy,
    arrow,
    d_place,
    diag_residuals,
    diag_settled,
    diag_y,
    directions,
    fixed_valid,
    gain,
    grid,
    loop_probe_b,
    loop_probe_u,
    measured,
    mo,
    new_fig,
    np,
    num,
    plane,
    predicted,
    residual,
    rho,
    show,
    test_idx,
    text,
    vec,
):
    _settled = fixed_valid and rho < 1 - 1e-6 and bool(np.all(diag_residuals < 1e-8))
    mo.stop(not _settled, mo.callout(mo.md(
        f"Fixed-point intervention comparison omitted: prompt residual {residual:.2e}, spectral radius {rho:.3f}, "
        f"largest diagnostic residual {diag_residuals.max():.2e}. It requires verified small-residual "
        "candidates and local stability at the prompt. Change the weights to meet these conditions."
    ), kind="warn"))
    _colors = {"probe direction": AQUA, "difference of class means": MUTED, "readout gradient": ORANGE, "lever": BLUE}
    _fig, (_ax, _cv) = new_fig(2, width=9.6, height=4.0, width_ratios=[1, 1.4])
    plane(_ax, labels=False, alpha=0.2)
    for _name, _dir in directions.items():
        arrow(_ax, (0, 0), 1.3 * _dir, color=_colors[_name], lw=2.2)
    _ax.set_title("Four unit directions for the last row")

    _t = np.arange(0, T_PUSH + 1)
    _rows = []
    for _name, _dir in directions.items():
        _P = PUSH_SIZE * _dir
        _pred = np.concatenate([[d_place @ _P], predicted(_P)])
        _meas = np.concatenate([[d_place @ _P], measured(_P)])
        _cv.plot(_t, _pred, color=_colors[_name], lw=2, label=_name)
        _cv.plot(_t, _meas, "o", ms=4, color=_colors[_name], mec=SURFACE, mew=0.8)
        _rows.append(
            [_name + (" (no-op)" if np.linalg.norm(_dir) == 0 else ""), vec(_dir),
             num(float(_dir @ directions["lever"]), "{:+.2f}") if np.linalg.norm(_dir) > 0 and np.linalg.norm(directions["lever"]) > 0 else "undefined",
             num(d_place @ _P, "{:+.4f}"),
             num(_meas[1], "{:+.4f}"), num(_meas[-1], "{:+.4f}"), num(_pred[-1], "{:+.4f}")]
        )
    _cv.axhline(0, color=RULE, lw=0.8)
    _cv.legend(frameon=False, fontsize=8, loc="upper right", ncols=2)
    _cv.set_xlim(-0.6, T_PUSH + 0.6)
    _cv.set_xticks([0, 4, 8, 12, 16, 20, 24])
    _cv.grid(axis="y", color=RULE, lw=0.6)
    _cv.tick_params(length=0)
    _cv.set_xlabel("iterations with the push applied (0 = at once)")
    _cv.set_title("Change in place preference r: formula (line), actual loop (dots)")

    _once = measured(0.1 * directions["readout gradient"], every=False)
    _acc = accuracy(diag_settled[test_idx], diag_y[test_idx], loop_probe_u, loop_probe_b)
    _g_row, _l_row, _p_row = _rows[2], _rows[3], _rows[0]
    _body = [
        mo.md(
            f"""
    **A push given once** (length 0.1 along the readout gradient, on *{text}*). At once, $r$ rises by
    {num(0.1 * np.linalg.norm(d_place), "{:+.3f}")}. Over the next six iterations what remains of that is
    {", ".join(num(_v, "{:+.3f}") for _v in _once[:6])}. The table and trajectory quantify the remaining effect; local stability predicts eventual decay for sufficiently small edits.

    **A push given at every iteration.** The last row's own block at the fixed point is $J$ = {vec(J_last[0])} /
    {vec(J_last[1])} (its two rows), with eigenvalues
    {", ".join(num(_v.real, "{:+.2f}") for _v in np.linalg.eigvals(J_last))}.
    """
        ),
        show(
            grid(
                _rows,
                cols=["direction", "unit vector", "cosine with the lever", "change in r at once",
                      "after 1 iteration", f"after {T_PUSH} iterations", "formula predicts"],
                bracket=False,
                left=(0,),
                name=f"Pushes of length {PUSH_SIZE} on the last row, g = {gain.value:.1f}",
            )
        ),
        _fig,
        mo.md(
            f"""
    The probe for this site classifies {_acc:.0%} of the 8 held-out candidate states correctly (baseline 50%).
    This is evidence of useful decoding only when it beats that baseline; a zero direction is a no-op. As a push it changes $r$ by {_p_row[5]} after {T_PUSH} iterations.

    The readout gradient wins at once, by construction: {_g_row[3]}. After {T_PUSH} iterations of the loop it
    stands at **{_g_row[5]}**. The lever starts at {_l_row[3]} and ends at **{_l_row[5]}**. The dots are the actual loop and the lines the first-order approximation. Their largest absolute
    discrepancy over these directions and 24 iterations is
    {max(float(np.max(np.abs(measured(PUSH_SIZE * v) - predicted(PUSH_SIZE * v)))) for v in directions.values()):.3e}.
    Small pushes need not remain in the linear regime near marginal stability; compare rather than assume agreement.

    With the default weights the sign change has a plain cause. $r$ depends mostly on the second coordinate, and
    the loop answers a push on the second coordinate by pushing back harder, through neuron 1. So the direction
    that raises $r$ at once lowers it once the loop has responded, and the direction that works looks wrong on the
    first step.
    """
        ),
    ]
    mo.vstack(_body)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The recurrent experiment tests three related distinctions (subject to its residual and stability checks).

    1. **A direction can be highly predictive.** A probe may read the family from a settled state; the held-out score tests that here.
    2. **Moving along it need not cause the predicted change.** A sufficiently small push given once decays near an asymptotically stable
       fixed point. A push given every iteration is reshaped by the loop: its first-order limiting effect, when $\rho(J)<1$, is
       $d^{\top}(I - J)^{-1} J\,P$, which can be small, or have the opposite sign to its immediate effect.
    3. **The direction that steers is a different object.** It is the readout direction carried back through the
       dynamics. Neither the probe nor the readout itself reveals it. The Jacobian of the loop does.

    Everything in Part II was arranged by hand: the weights were chosen partly so that these effects would show.
    The toy establishes that the effects are possible and how to compute them. Whether trained models behave this
    way is an empirical question, and it is the subject of Part III.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part III · The same questions, asked of trained models

    The experiments in this repository put the questions of Part II to **Huginn-0125** (Geiping and others, 2025),
    a looped transformer trained on text, and repeated the steering experiment on **Recurrent-Llama-3.2** (McLeish
    and others, 2025). Everything in this part is **Measured**: the numbers come from the write-ups listed at the
    end, not from this notebook.

    Huginn's rows have 5,280 coordinates. Two layers embed the prompt. A core of four layers is then applied again
    and again to a state that starts as random noise, with the embedded prompt mixed back in before every
    iteration. Two final layers read out the next token. The correspondence with the toy:

    | In the toy | In Huginn |
    |---|---|
    | $F(H) = T_\theta\big(\tfrac12(H + X)\big)$ | the core block applied to a learned linear map of the state and the embedded prompt |
    | start from $X$, or from a random state | start from a seeded random state |
    | the last row | the last token, where the answer is predicted |
    | place preference $r = d^{\top} h_n$ | a digit's logit minus the average over digits, or $\log p(\text{True}) - \log p(\text{False})$ |
    | the context *the cat sat on the* | few-shot addition problems; true-or-false statements about cities and number comparisons |
    | $g$, set by a slider | nothing: every weight was trained |

    ## 19 · Do the states settle?

    **Mostly, yes.** At the positions where Huginn predicts an answer digit, late states are close to stationary,
    and the largest eigenvalue magnitude found for the local dynamics has a median of 0.78 (the toy's spectral
    radius is shown in section 16). The answer settles long before the state does: among positions that
    converged, the most probable digit stops changing at a median of 7 iterations and the state at a median of 74.

    **Sometimes, no.** On 200 fresh addition problems, 26 (13%) did not settle and ended on a period-two orbit. In
    each of the 26 the oscillation began at one earlier token and spread to later ones through attention, and the
    derivative at that source token had a real eigenvalue between −1.01 and −1.23, the case of section 17. Damping
    the source token alone stopped its oscillation in 24 of the 26. It stilled the whole text in only 15, because
    other rows often keep moving for reasons of their own.

    **Does the oscillation change what the model writes?** Usually not. The two states of an orbit are close, so
    the probabilities differ by a few points, and the top choice changes only where two candidates were nearly
    tied. In the first 44 oscillating prompts examined, no answer digit changed. A targeted search (400 new sums
    screened for near-ties, the 85 closest run for 512 iterations) found four answer digits that alternate between
    the correct digit and a wrong one, out of 358 answer positions on an orbit. One of them: after three worked
    sums, Huginn continues `816 + 254 =` with 1070 at 511 iterations per token and with 1060 at 512. The chart
    replays its saved probabilities for the third digit. From another random starting state the same prompt still
    oscillates, but 7 stays ahead. These examples were found by a search built to find them; they do not say how
    often this happens.
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
        _ax.set_xlabel("iterations before reading out")
        _ax.set_title(_title)
    _fig.suptitle("Measured · Huginn · probability of the digit after “816 + 254 = 10”", x=0.01, ha="left",
                  fontsize=10, color=MUTED)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 20 · Reading is not controlling

    **A property can be read.** Whether a statement is true can be decoded from Huginn's state at the end of the
    statement by a simple linear probe (the difference of class means), with an AUROC of 0.997 on statements about
    cities and 0.914 on number comparisons. That is decoding of a label, as in section 13. It is not evidence that
    the model has a notion of truth.

    **A push given once is erased.** Pushing a digit's state along a fitted direction moved the immediate readout
    toward the intended neighbouring digit by 2.3 in log-odds. Under continued iteration the push decayed, and the
    final answer equalled the unpushed one on 168 of 168 prompts. The same kind of edit applied at every iteration
    settled at a shifted state.

    **The probe direction is a weak lever.** On 60 held-out true-or-false statements, with pushes of equal length
    applied at every iteration, the median shift of the True/False log-odds was 0.25 along the probe direction,
    0.93 along the readout gradient, and 8.13 along the lever. The lever is nearly orthogonal to both of the others
    (cosine 0.10 with the readout gradient, 0.03 with the probe direction). What it moves is the readout: it raises the token "True". The experiments designed to
    separate that from changing the model's judgment were inconclusive.

    **The same on digits.** Unit directions were averaged over training sums and tested on held-out sums, each
    push added before every iteration. The numbers are the change in the pushed answer's score.

    | direction | Huginn (20 held-out sums, 48 iterations, push length 3) | Recurrent-Llama-3.2 (40 held-out sums, push 3% of the state's length) |
    |---|---|---|
    | lever | +1.78 | +0.165 |
    | readout gradient | −0.05 | −0.001 |
    | centroid of the target's states | −0.05 | 0.000 |

    In Huginn the lever was positive on 189 of 200 problem–digit pairs. It started slightly negative at once
    (median −0.13) and turned positive after a single iteration (+0.94), while the gradient and the centroid sat at
    the level of random directions (±0.05). That is the pattern of the table in section 18, in a model nobody
    arranged. In Recurrent-Llama the cosine between lever and gradient is −0.005. The chart shows its three push
    lengths.

    **The first-order formula predicts the measurements.** Ten lever pushes times ten candidate answers give 100
    score changes per problem. Using each problem's own $J$, the formula of section 18 predicts them with median
    correlation 0.9997 at the 1% push, 0.997 at 3% and 0.976 at 9%, against 0.11 with the loop left out (in Huginn
    0.978, against −0.59).

    **The catch.** On Recurrent-Llama the correct answer leads the runner-up by a median of 4.84, about ten times
    the largest change in the chart. No push made its target the model's answer: 0 of 360 tries at every length.
    And in Huginn, in a retrospective test where a probe chose which way to steer, steering added no accuracy
    beyond what the probe's own verdict already gave.
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
    ## 21 · What this shows, and what it does not

    **The toy** shows that each mechanism fits in a few lines of numpy and can be checked by hand. A hidden state
    can carry a diagnostic label perfectly while the output fails to reflect it. In some settings, a locally
    unstable row drives a numerically observed period-two orbit that later rows inherit. The direction maximizing the first-order finite-horizon readout change at a fixed point is
    $(J + \dots + J^T)^{\top} d$, which need not resemble either the probe direction or $d$. Its weights were
    chosen by hand to show exactly that, so the toy says nothing about how often any of this happens in a trained
    model.

    **The experiments on dynamics** connect a source position's local instability to a change in the model's
    next-token probabilities, and show that damping that position can sometimes stop the oscillation. They do not
    estimate how often answers flip in general use: the answer-changing examples came from searches for near-ties,
    and the outcome also depends on the starting state.

    **The experiments on control** show that the response of the loop decides which direction steers a readout,
    and that a local linear model of the loop predicts the size of the effect. They measure changes in target
    scores on the tested tasks and models. In Recurrent-Llama none of the tested pushes made the target the answer,
    even though its score rose. In neither model do they show that a concept, as opposed to a readout, was
    changed.

    **A way of asking.** The thread through all three parts is a habit more than a result. Say what the hidden
    state is and where it sits. Separate what can be read from it from what the model does with it. Test the
    second by changing the state and recomputing. And when the model is a loop, remember that everything downstream
    of an intervention includes the loop.

    Sources in this repository: `huginn_addition/WRITEUP.md`, `docs/flicker2_design.md`,
    `docs/steering_results.pdf`, `docs/followup_results.md`. Models: Huginn (Geiping and others, 2025),
    Recurrent-Llama-3.2 (McLeish and others, 2025). The way of writing an attention head as "where to read" and
    "what to bring back" is from Elhage and others, "A Mathematical Framework for Transformer Circuits" (2021);
    the damping rule is in Movahedi and others (2026); the lever is the calculus used to train fixed-point models
    since Bai, Kolter and Koltun (2019).
    """)
    return


if __name__ == "__main__":
    app.run()
