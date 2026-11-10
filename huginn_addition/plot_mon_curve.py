"""Plot saved learning curves without fitting or recomputing estimates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

COLORS = {'L32': '#0072B2', 'D': '#D55E00'}
FITS = {'probe': ('-', 'o'), 'confidence': (':', 's'),
        'confidence+probe': ('--', '^')}
FAMILIES = ('add3', 'add4', 'add3w', 'arc_easy', 'family_mean')


def make_figures(curve, out):
    data = json.loads(Path(curve).read_text())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    families = [f for f in FAMILIES if f in data['curves']]
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
    points = []
    for added in (False, True):
        name = 'fig_curve_added' if added else 'fig_curve'
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 5.8), sharey=True)
        axes = list(axes.flat)
        for ax in axes:
            ax.set_visible(False)
        for ax, family in zip(axes, families):
            ax.set_visible(True)
            ax.set_title('mean of the four families' if family == 'family_mean' else family)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.2)
            ticks = set()
            if added:
                ax.axhline(0, color='.5', lw=.7)
            for model, color in COLORS.items():
                curves = data['curves'][family][model]
                for fit in (('added',) if added else FITS):
                    rows = data['derived'][family][model]['added'] if added else curves[fit]
                    xs, ys = [], []
                    for fraction in sorted(rows, key=float):
                        row = rows[fraction]
                        x = curves['confidence+probe'][fraction]['training_items'] if added else row['training_items']
                        y = row['auroc']
                        if added:
                            values = row['draw_values']
                            low, high = (min(values), max(values)) if len(values) > 1 else (None, None)
                        else:
                            low, high = row['draw_range'] if row['draws'] > 1 else (None, None)
                        points.append(dict(figure=name, panel=family,
                                           series=model if added else f'{model} {fit}',
                                           x=x, y=y, low=low, high=high))
                        if low is not None:
                            ax.vlines(x, low, high, color=color, linewidth=.9)
                            ax.plot([x, x], [low, high], ls='none', marker='_', color=color, markersize=4)
                        xs.append(x)
                        ys.append(y)
                        ticks.add(x)
                    style, marker = ('-', 'o') if added else FITS[fit]
                    ax.plot(xs, ys, color=color, ls=style, marker=marker, markersize=3, lw=.9)
            ax.set_xscale('log', base=2)
            positions = []
            for x in sorted(ticks):  # the two models' sizes can differ by one problem: one tick for both
                if not positions or x > 1.02 * positions[-1]:
                    positions.append(x)
            ax.set_xticks(positions, [f'{x:.0f}' for x in positions], rotation=45, ha='right')
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel('Training problems')
            ax.set_ylabel('Added AUROC' if added else 'AUROC')
        axes[-1].set_visible(True)
        axes[-1].axis('off')
        handles = [Line2D([], [], color=c, label=m, lw=.9) for m, c in COLORS.items()]
        if not added:
            handles += [Line2D([], [], color='.3', ls=ls, marker=mk, markersize=3,
                               label=fit.replace('+', ' + ')) for fit, (ls, mk) in FITS.items()]
        axes[-1].legend(handles=handles, loc='center', frameon=False, fontsize=8,
                        handlelength=1.5, labelspacing=.9)
        fig.tight_layout(pad=.7, h_pad=1.4, w_pad=.9)
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}', dpi=200)
        plt.close(fig)
    (out / 'curve_figures.json').write_text(json.dumps({'points': points}, indent=2) + '\n')
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curve', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    make_figures(args.curve, args.out)
