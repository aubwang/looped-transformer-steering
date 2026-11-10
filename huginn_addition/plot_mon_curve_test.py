"""Small end-to-end check of saved values and figure outputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = {'curves': {}, 'derived': {}}
        expected = []
        # Deliberately different family means and derived values catch recomputation.
        for family, offset in [('add3', 0), ('family_mean', .07)]:
            data['curves'][family] = {}
            data['derived'][family] = {}
            for model, delta in [('L32', 0), ('D', .03)]:
                data['curves'][family][model] = {}
                added = {}
                for fit, extra in [('probe', 0), ('confidence', .1), ('confidence+probe', .2)]:
                    cells = {}
                    for fraction, x, draws in [('0.125', 13, 2), ('1.0', 101, 1)]:
                        y = .51 + offset + delta + extra
                        low, high = y - .02, y + .04
                        cells[fraction] = dict(training_items=x, auroc=y, draws=draws,
                                               draw_aurocs=[low, high][:draws], draw_range=[low, high])
                        expected.append(dict(figure='fig_curve', panel=family, series=f'{model} {fit}',
                                             x=x, y=y, low=low if draws > 1 else None,
                                             high=high if draws > 1 else None))
                        if fit == 'confidence+probe':
                            value = -.017 + offset + delta
                            values = [-.04, .06] if draws > 1 else [value]
                            added[fraction] = dict(auroc=value, draw_values=values)
                            expected.append(dict(figure='fig_curve_added', panel=family, series=model,
                                                 x=x, y=value, low=-.04 if draws > 1 else None,
                                                 high=.06 if draws > 1 else None))
                    data['curves'][family][model][fit] = cells
                data['derived'][family][model] = {'added': added}
        curve = root / 'curve.json'
        curve.write_text(json.dumps(data))
        subprocess.run([sys.executable, '-I', str(Path(__file__).with_name('plot_mon_curve.py')),
                        '--curve', str(curve), '--out', str(root / 'plots')], check=True)
        for name in ('fig_curve', 'fig_curve_added'):
            for ext in ('pdf', 'png'):
                assert (root / 'plots' / f'{name}.{ext}').stat().st_size > 0
        actual = json.loads((root / 'plots' / 'curve_figures.json').read_text())['points']
        key = lambda p: (p['figure'], p['panel'], p['series'], p['x'])
        assert sorted(actual, key=key) == sorted(expected, key=key)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
