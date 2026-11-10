"""Publication figures for summarize_mon.py; no fitting or bootstrap work.

Points use the median across seeds; intervals are the envelope only when every
seed has an interval. family_mean is read from the analysis, never recomputed.
"""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {'L4': '#009E73', 'L8': '#E69F00', 'L16': '#CC79A7',
          'L32': '#0072B2', 'D': '#D55E00'}
MODELS = list(COLORS)
NATIVE = '#444444'      # budget figures: the probe fitted at the target loop count


def load_summary(inputs):
    """Import the existing CLI module without changing it or printing its tables."""
    spec = importlib.util.spec_from_file_location('mon_summary', Path(__file__).with_name('summarize_mon.py'))
    module = importlib.util.module_from_spec(spec)
    old = sys.argv
    try:
        sys.argv = ['summarize_mon.py', '--inputs', *map(str, inputs), '--sections', '__load_only__']
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)
    finally:
        sys.argv = old
    return module


def aggregate(s, section, family, key, field='auroc', interval='interval'):
    """Use summary's accessors and exactly its median/envelope/missing policy."""
    values, bounds = [], []
    for seed in s.SEEDS:
        e = s.get(section, seed, family, key)
        v = s.sub(e, field)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        values.append(v)
        iv = s.sub(e, interval) if interval else None
        if isinstance(iv, list) and len(iv) == 2 and all(isinstance(x, (int, float)) for x in iv):
            bounds.append(iv)
    complete = len(bounds) == len(values)
    return dict(value=s.statistics.median(values),
                lower=min(x[0] for x in bounds) if complete else None,
                upper=max(x[1] for x in bounds) if complete else None,
                n_seeds=len(values))


def make_figures(inputs, out):
    s = load_summary(inputs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    points, omitted = [], []
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'savefig.bbox': 'tight'})

    def keys(section, family):
        return sorted({k for seed in s.SEEDS for k in
                       s.RES[seed].get(section, {}).get(family, {}).get('estimates', {})})

    def canvas():
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 5.8), sharey=True)
        axes = list(axes.flat)
        for ax, fam in zip(axes, s.FAMS):
            ax.set_title('mean of the four families' if fam == 'family_mean' else fam)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.2)
        axes[-1].axis('off')
        return fig, axes

    def point(ax, name, fam, series, x, pos, section, key, color,
              field='auroc', interval='interval', marker='o', size=4):
        a = aggregate(s, section, fam, key, field, interval)
        if a is None:
            omitted.append(dict(figure=name, panel=fam, key=key, reason='missing point in at least one seed'))
            return None
        points.append(dict(figure=name, panel=fam, series=series, x=x,
                           section=section, key=key, field=field, interval=interval, **a))
        lo, hi = a['lower'], a['upper']
        if lo is not None:
            # Draw ends directly: valid intervals need not bracket the median.
            ax.vlines(pos, lo, hi, color=color, linewidth=.9)
            ax.plot([pos, pos], [lo, hi], linestyle='none', marker='_', color=color, markersize=4)
        ax.plot(pos, a['value'], marker=marker, markersize=size, linestyle='none',
                color=color, markerfacecolor=color if lo is not None else 'white')
        return a['value']

    def finish(fig, axes, name, ylabel, handles):
        for ax in axes[:len(s.FAMS)]:
            ax.set_ylabel(ylabel)
        handles = handles + [Line2D([], [], color='black', marker='o', mfc='white', ls='',
                                   label='Open: interval unavailable')]
        axes[-1].legend(handles=handles, loc='center', frameon=False, fontsize=8,
                        handlelength=1.5, labelspacing=.9)
        fig.tight_layout(pad=.7, h_pad=1.4, w_pad=.9)
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}', dpi=200)
        plt.close(fig)

    model_handles = lambda ms: [Line2D([], [], color=COLORS[m], marker='o', ls='', label=m) for m in ms]
    fig, axes = canvas()
    for ax, fam in zip(axes, s.FAMS):
        for i, (r, c) in enumerate((r, c) for r in ('L32', 'D') for c in ('L32', 'D')):
            point(ax, 'fig_table', fam, f'{r} errors / {c} state', f'{r}/{c}', i,
                  'm1a', f'table/{r}/{c}', COLORS[c])
        ax.axhline(.5, color='.5', ls='--', lw=.7)
        ax.set_xticks(range(4), ['L32 / L32', 'L32 / D', 'D / L32', 'D / D'], rotation=45, ha='right')
        ax.set_xlabel('Errors / state')
    finish(fig, axes, 'fig_table', 'Probe AUROC', model_handles(['L32', 'D']))

    fig, axes = canvas()
    types = ['confidence', 'probe', 'confidence+probe', 'output_identity', 'privileged_ground_truth', 'early']
    labels = ['Confidence', 'Probe', 'Both', 'Output identity', 'Ground truth', 'Early state']
    for ax, fam in zip(axes, s.FAMS):
        for m, offset in [('L32', -.12), ('D', .12)]:
            for i, t in enumerate(types):
                control = i >= 3
                key = f'{m}/{t}' if control else f'fixed/{m}/{t}'
                point(ax, 'fig_confidence', fam, m, t, i+offset, 'm1c', key,
                      '.5' if control else COLORS[m], marker='o' if m == 'L32' else 's', size=3 if control else 4)
        ax.set_xticks(range(6), labels, rotation=60, ha='right')
        ax.set_xlabel('Prediction information')
    finish(fig, axes, 'fig_confidence', 'AUROC', [
        Line2D([], [], color=COLORS[m], marker=mk, ls='', label=m) for m, mk in [('L32','o'), ('D','s')]] +
        [Line2D([], [], color='.5', marker='o', ls='', label='Grey: controls')])

    fig, axes = canvas()
    for ax, fam in zip(axes, s.FAMS):
        for i, m in enumerate(MODELS):
            for t, offset, marker in [('probe', -.1, 'o'), ('confidence', .1, 's')]:
                key = f'{m}/{m[1:]}/{t}' if m != 'D' else f'D/16/{t}'
                point(ax, 'fig_loopcounts', fam, f'{m} {t}', m, i+offset, 'm1s', key, COLORS[m], marker=marker)
        ax.set_xticks(range(5), MODELS)
        ax.set_xlabel('Model (trained loop count)')
    finish(fig, axes, 'fig_loopcounts', 'AUROC', model_handles(MODELS) + [
        Line2D([], [], color='.3', marker=mk, ls='', label=t) for t, mk in [('Probe','o'), ('Confidence','s')]])

    for m, src, name in [('L32',32,'fig_budget'), ('L4',4,'fig_budget_L4')]:
        pattern = re.compile(rf'{m}/{src}->(\d+)$')
        targets = sorted({int(match[1]) for fam in s.FAMS for k in keys('m3a', fam)
                          if (match := pattern.fullmatch(k))})
        if not targets:
            omitted.append(dict(figure=name, reason='no transfer targets'))
            continue
        fig, axes = canvas()
        for ax, fam in zip(axes, s.FAMS):
            for native, style in [(False, '-'), (True, '--')]:
                xs, ys = [], []
                for r in targets:
                    key = f'{m}/native{r}' if native else f'{m}/{src}->{r}'
                    y = point(ax, name, fam, 'native' if native else 'transfer', r,
                              r * (1.03 if native else .97), 'm3a', key, NATIVE if native else COLORS[m],
                              field='auroc' if native else 'transfer_auroc',
                              interval='interval' if native else 'transfer_interval',
                              marker='^' if r in (48,63,64) else ('s' if native else 'o'), size=3)
                    xs.append(r); ys.append(float('nan') if y is None else y)
                ax.plot(xs, ys, style, color=NATIVE if native else COLORS[m], lw=.8, alpha=.6)
            ax.set_xscale('log', base=2)
            ax.set_xticks([1,4,16,64], ['1','4','16','64'])
            ax.set_xlabel('Target loop count')
        finish(fig, axes, name, 'Probe AUROC', [
            Line2D([], [], color=COLORS[m], marker='o', label=f'Fitted at {src}, applied at r'),
            Line2D([], [], color=NATIVE, marker='s', ls='--', label='Fitted at r'),
            Line2D([], [], color='.3', marker='^', ls='', label='48, 63, 64: smaller sample')])

    fig, axes = canvas()
    for ax, fam in zip(axes, s.FAMS):
        for m in MODELS:
            counts = sorted({int(r) for seed in s.SEEDS for r in
                             (s.sub(s.get('dyn', seed, fam, m), 'accuracy') or {})})
            xs, ys = [], []
            for r in counts:
                y = point(ax, 'fig_accuracy', fam, m, r, r, 'dyn', m, COLORS[m],
                          field=f'accuracy.{r}.accuracy', interval=None, size=2.5)
                xs.append(r); ys.append(float('nan') if y is None else y)
            if counts:
                ax.plot(xs, ys, color=COLORS[m], lw=.9)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1,4,16,64], ['1','4','16','64'])
        ax.set_xlabel('Readout loop count')
    if not any(p['figure'] == 'fig_accuracy' and p['series'] == 'D' for p in points):
        omitted.append(dict(figure='fig_accuracy', reason='D accuracy absent'))
    finish(fig, axes, 'fig_accuracy', 'Accuracy', model_handles(MODELS[:-1]))
    (out / 'figures.json').write_text(json.dumps(dict(points=points, omitted=omitted,
        aggregation='Median over seeds; interval envelope only if available in every seed; family_mean read directly.'), indent=2) + '\n')
    return points


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    make_figures(args.inputs, args.out)
