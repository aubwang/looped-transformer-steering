"""Numbers and figures for docs/steering_results.typ. Nothing is fitted here: every value is read from saved run
outputs (and, for Huginn, from docs/steering_gen/huginn_facts.json, whose entries carry their own sources).
Usage (from the repository root):  python huginn_addition/steering_report.py
Writes docs/steering_gen/numbers.json (formatted strings + raw values) and docs/steering_gen/figs/*.pdf|png.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
GEN = ROOT / 'docs' / 'steering_gen'
TRAIN, PRED, PRED2, EVAL = (RES / n for n in ('llever_train-20261110-171953', 'llever_pred-20261110-172208',
                                             'llever_pred-20261110-174655', 'llever_eval-20261110-172520'))
SCREEN, CONFIRM, EXIST = RES / 'lateshift-20261110-171533', RES / 'lateshift-20261110-172550', RES / 'lateshift_existing-20261110'
DOSES = (.01, .03, .09)
# Read-space self-gain (docs/freeloop.typ, docs/lread_design.md): saved summaries of the four runs.
CTRL_T, CTRL_D, OURO = RES / 'ctrlprobe-20261107-144949', RES / 'ctrlprobe_digits-20261107-144951', RES / 'ouro-20261107-150835'
LREAD = RES / 'lread-run-20261110-221504'
LREAD_SMOKES = (RES / 'lread-smoke-20261110-220912', RES / 'lread-smoke-20261110-221413')
BLUE, ORANGE, AQUA, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e'  # validated: dataviz palette slots 1-3


def sgn(x, nd=3):
    """Fixed decimals with a true minus sign and an explicit plus for positive effects."""
    s = f'{abs(x):.{nd}f}'
    return ('−' if x < 0 and float(s) != 0 else '+' if x > 0 and float(s) != 0 else '') + s


def num(x, nd=3):
    s = f'{abs(x):.{nd}f}'
    return ('−' if x < 0 and float(s) != 0 else '') + s


def iv(c, nd=3):
    return f'[{num(c[0], nd)}, {num(c[1], nd)}]'


def pct(x, nd=1):
    return f'{100 * x:.{nd}f}%'


def sci(x):
    m, e = f'{x:.1e}'.split('e')
    return f'{m} × 10' + str(int(e)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))


def lever():
    N = {}
    A = json.loads((EVAL / 'analysis_summary.json').read_text()); E1 = json.loads((PRED2 / 'partial_summary.json').read_text())
    metas = {k: json.loads((p / 'meta.json').read_text()) for k, p in (('train', TRAIN), ('pred', PRED), ('pred2', PRED2), ('eval', EVAL))}
    smoke = json.loads((RES / 'llever_smoke-20261110-171809' / 'meta.json').read_text())
    zp = np.load(PRED / 'llever.npz'); ze = np.load(EVAL / 'llever.npz'); zt = np.load(TRAIN / 'llever.npz'); push = np.load(EVAL / 'pushes.npz')
    m = float(push['m_ref']); ev = np.where(zp['split'] == 'eval')[0]; G = zp['G'][ev]; AP = zp['AP'][ev]; P = push['P']
    rp, rr = ze['row_push'], ze['row_rho']; times = [int(t) for t in ze['times']]
    rows = lambda off, rho: [int(np.flatnonzero((rp == off + k) & np.isclose(rr, rho))[0]) for k in range(10)]
    N['verdict'] = A['verdict']; N['m_ref'] = num(m, 1); N['n_eval'] = str(len(ze['pid']))
    N['g1_max'] = sci(max(max(mm['gates']['G1']) for mm in metas.values()))
    N['adjoint'] = sci(A['adjoint_error']); N['tail_q95'] = sci(max(A['tails'][k]['q95'] for k in A['tails']))
    N['tail_max'] = sci(float(max(zt['tail'].max(), zp['tail'].max())))
    N['sham_correct'] = str(A['sham_correct']); N['sham_drift'] = sci(A['sham_drift']); N['base_drift_median'] = sci(float(np.median(ze['base_drift'])))
    N['base_drift_max'] = sci(float(ze['base_drift'].max()))
    N['correct_32_64'] = f"{int((zt['answer32'] == zt['ans_token']).sum() + (zp['answer32'][ev] == zp['ans_token'][ev]).sum())} of {len(zt['pid']) + len(ev)}"
    N['cos_L_G'] = num(float(np.mean(np.sum(P[:10] * P[10:20], -1))), 3); N['cos_L_C'] = num(float(np.mean(np.sum(P[:10] * P[20:30], -1))), 3)
    raw = dict(times=times, doses=list(DOSES))
    # Answers after the forced run (whole-vocabulary argmax, array `top`): pairs (problem, pushed candidate k != own answer).
    cls = {int(a): int(t) for a, t in zip(ze['ans'], ze['ans_token'])}; tok = np.array([cls[70 + k] for k in range(10)])
    own = ze['ans'] - 70; assert (ze['top'][:, 0] == ze['ans_token']).all() and rp[0] < 0
    for rho in DOSES:
        t = f'{int(round(rho * 100)):02d}'
        for f, off in (('L', 0), ('G', 10), ('C', 20)):
            top = ze['top'][:, rows(off, rho)]; other = own[:, None] != np.arange(10)[None]
            N[f'flip_target_{f}_{t}'] = str(int(((top == tok[None]) & other).sum()))
            N[f'flip_other_{f}_{t}'] = str(int(((top != tok[None]) & (top != ze['ans_token'][:, None]) & other).sum()))
        N['flip_pairs'] = str(int(other.sum()))
    assert all(N[f'flip_target_{f}_{t}'] == '0' for f in 'LGC' for t in ('01', '03', '09')), 'write-up says no target flips'
    assert all(N[f'flip_other_{f}_{t}'] == '0' for f in 'GC' for t in ('01', '03', '09')) and N['flip_other_L_01'] == N['flip_other_L_03'] == '0'
    N['flip_other_09'] = N['flip_other_L_09']
    # ---- Computed after the main results, from saved arrays (suggested by results/steering_results_review4.md).
    z2 = np.load(PRED2 / 'llever.npz'); ev2 = np.where(z2['split'] == 'eval')[0]; assert int(z2['partial_t'][0]) == 1
    assert np.allclose(np.linalg.norm(P[:10], axis=1), 1)
    JL = np.linalg.norm(z2['AP_partial'][ev2, 0], axis=-1)                      # |J L_k|, unit levers, [problem, k]
    N['jl_median'] = num(float(np.median(JL)), 2); N['jl_min'] = num(float(JL.min()), 2); N['jl_max'] = num(float(JL.max()), 2)
    N['jl_q25'], N['jl_q75'] = (num(float(q), 2) for q in np.quantile(JL, [.25, .75]))
    for f, lo, hi in (('L', 0, 10), ('G', 10, 20), ('C', 20, 30), ('R', 30, 32)):  # endpoint displacement at q over |P|, main dose
        rws = [int(np.flatnonzero((rp == k) & np.isclose(rr, .03))[0]) for k in range(lo, hi)]
        N[f'disp_{f}_03'] = num(float(np.median(ze['displacement'][:, rws] * ze['norms'][:, None] / (.03 * m))), 2)
    rws = [int(np.flatnonzero((rp == k) & np.isclose(rr, .03))[0]) for k in (30, 31)]
    N['random_abs_03'] = num(float(np.median(np.abs(ze['Z'][:, rws, -1] - ze['Z'][:, 0:1, -1]))), 4)
    z0 = ze['Z'][:, 0, -1]; zo = z0[np.arange(len(own)), own]; zr = z0.copy(); zr[np.arange(len(own)), own] = -np.inf
    mar = zo - zr.max(1)                                                         # lead of the own answer over the runner-up candidate
    N['margin_median'], N['margin_min'], N['margin_max'] = (num(float(v), 2) for v in (np.median(mar), mar.min(), mar.max()))
    lead = 0                                                                     # pushed candidate leading among the ten at a saved time
    for off in (0, 10, 20):
        for rho in DOSES:
            for k, r in enumerate(rows(off, rho)):
                lead += int(((ze['Z'][:, r].argmax(-1) == k) & (own[:, None] != k)).any(1).sum())
    assert lead == 0, 'write-up says the pushed candidate never leads at a saved time'
    N['lead_any'] = str(lead); N['saved_times'] = ', '.join(str(t) for t in times[:-1])
    for rho in DOSES:
        d = A['doses'][str(rho)]; t = f'{int(round(rho * 100)):02d}'; p1, p3 = d['P1'], d['P3']
        N[f'norm_{t}'] = num(rho * m, 2)
        for f in 'LGC':
            N[f'eff_{f}_{t}'] = sgn(p1['mean_target_effect'][f]); N[f'eff_{f}_ci_{t}'] = iv(p1['target_ci'][f])
        N[f'L_minus_G_{t}'] = iv(p1['L_minus_G_ci'])
        for k, v in p3.items():
            N[f'corr_{k}_{t}'] = num(v['median_corr'], 4 if v['median_corr'] > .999 else 3)
        N[f'slope_FULL_{t}'] = num(p3['FULL']['median_slope'], 2); N[f'err_FULL_{t}'] = pct(p3['FULL']['median_norm_error'])
        for k, v in d['FULL_minus_ci'].items():
            N[f'FULL_minus_{k}_{t}'] = iv(v, 2)
        dzL = ze['Z'][:, rows(0, rho)] - ze['Z'][:, 0:1]; dzG = ze['Z'][:, rows(10, rho)] - ze['Z'][:, 0:1]
        tL = np.einsum('jktk->jkt', dzL); tG = np.einsum('jktk->jkt', dzG)                       # target effects [problem, k, time]
        c0L = rho * m * np.einsum('jkd,kd->jk', G, P[:10]); c0G = rho * m * np.einsum('jkd,kd->jk', G, P[10:20])
        N[f'imm_L_{t}'] = sgn(float(c0L.mean())); N[f'imm_L_neg_{t}'] = pct(float((c0L < 0).mean()), 0); N[f'imm_G_{t}'] = sgn(float(c0G.mean()), 2)
        N[f'ratio_c0_{t}'] = num(float(c0L.mean() / c0G.mean()), 3)                 # Proposition 3: three readings of one number
        N[f'ratio_eff_{t}'] = num(float(tG[:, :, -1].mean() / tL[:, :, -1].mean()), 3)
        for i, tt in enumerate(times):
            N[f'course_L_{t}_t{tt}'] = sgn(float(tL[:, :, i].mean())); N[f'course_G_{t}_t{tt}'] = sgn(float(tG[:, :, i].mean()))
        end = dzL[:, :, -1].reshape(len(dzL), -1)
        disc = [float(np.median(np.linalg.norm(dzL[:, :, i].reshape(len(dzL), -1) - end, axis=1) / np.linalg.norm(end, axis=1))) for i in range(len(times))]
        for i, tt in enumerate(times[:-1]):
            N[f'disc_{t}_t{tt}'] = pct(disc[i], 2 if disc[i] < .001 else 1)
        N[f'corr_t1_end_{t}'] = num(float(np.median([np.corrcoef(dzL[j, :, 0].ravel(), dzL[j, :, -1].ravel())[0, 1] for j in range(len(dzL))])), 2)
        e1 = E1['doses'][str(rho)]
        for tt, v in e1.items():
            N[f'e1_err_{t}_t{tt}'] = pct(v['endpoint']['median_norm_error']); N[f'e1_corr_{t}_t{tt}'] = num(v['endpoint']['median_corr'], 4 if v['endpoint']['median_corr'] > .999 else 3)
            N[f'e1_excess_{t}_t{tt}'] = '[' + ', '.join(f'{100 * x:.1f}' for x in v['endpoint']['error_minus_FULL_ci']) + ']'
            if 'matched_time' in v:
                N[f'e1_matched_{t}_t{tt}'] = pct(v['matched_time']['median_norm_error'])
        raw[t] = dict(course_L=[float(tL[:, :, i].mean()) for i in range(len(times))], course_G=[float(tG[:, :, i].mean()) for i in range(len(times))],
                      imm_L=float(c0L.mean()), imm_G=float(c0G.mean()), disc=disc,
                      e1_t=[int(k) for k in e1], e1_err=[v['endpoint']['median_norm_error'] for v in e1.values()],
                      pred_full=(rho * m * np.einsum('jmd,jkd->jkm', G, AP)).ravel().tolist() if rho == .03 else None,
                      pred_id=(rho * m * np.einsum('jmd,kd->jkm', G, P[:10])).ravel().tolist() if rho == .03 else None,
                      obs=dzL[:, :, -1].ravel().tolist() if rho == .03 else None)
    N['dose_ratio_low'] = num(A['dose_ratios']['L']['0.01']['observed'], 3); N['dose_ratio_high'] = num(A['dose_ratios']['L']['0.09']['observed'], 2)
    N['e1_reproduces'] = sci(max(E1['reproduces_pred_run'].values())) if max(E1['reproduces_pred_run'].values()) > 0 else 'exactly'
    N['per_unit_norm'] = num(A['doses']['0.03']['P1']['mean_target_effect']['L'] / (.03 * m), 2)
    secs = dict(smoke=smoke['runtime_s'], train=metas['train']['runtime_s'], pred=metas['pred']['runtime_s'],
                eval=sum(s['runtime_s'] for s in metas['eval']['shards']), e1=metas['pred2']['runtime_s'])
    N['gpu_lever_main'] = num((secs['smoke'] + secs['train'] + secs['pred'] + secs['eval']) / 60, 1); N['gpu_lever_e1'] = num(secs['e1'] / 60, 1)
    N['peak_gb'] = num(smoke['timings']['benchmark']['peak_gb'], 1)
    return N, raw


def screen():
    N = {}; S = json.loads((SCREEN / 'analysis_summary.json').read_text()); C = json.loads((CONFIRM / 'analysis_summary.json').read_text())
    X = json.loads((EXIST / 'summary.json').read_text())
    rows, tot = [], dict(ch=0, ww=0, wr=0, rw=0)
    for f, v in S['families'].items():
        ch, tr, a = v['changes'], v['transitions'], v['accuracy']
        k = round(ch['8,32']['rate'] * v['n']); tot['ch'] += k; tot['ww'] += tr['wrong_wrong_changed']['k']; tot['wr'] += tr['wrong_right']; tot['rw'] += tr['right_wrong']
        rows.append(dict(fam=f, verdict=v['verdict'].capitalize(), change=f"{100 * ch['8,32']['rate']:.0f}% [{100 * ch['8,32']['ci95'][0]:.0f}, {100 * ch['8,32']['ci95'][1]:.0f}]",
                         c16=f"{100 * ch['16,32']['rate']:.0f}%", c31=f"{100 * ch['31,32']['rate']:.0f}%", acc8=f"{100 * a['8']['rate']:.0f}%", acc32=f"{100 * a['32']['rate']:.0f}%",
                         accD=f"{100 * v['accuracy_D']['rate']:.0f}%", flips=f"{tr['wrong_right']} / {tr['right_wrong']}",
                         x=a['32']['rate'], y=ch['8,32']['rate'], xd=v['accuracy_D']['rate'], chance=v['chance']))
    assert tot['ch'] == tot['ww'] + tot['wr'] + tot['rw']
    N.update(n_fam=str(len(rows)), n_per=str(S['families']['add3']['n']), changes_total=str(tot['ch']), changes_ww=str(tot['ww']), changes_wr=str(tot['wr']), changes_rw=str(tot['rw']))
    N['max_c16'] = pct(max(v['changes']['16,32']['rate'] for v in S['families'].values()), 0); N['max_c31'] = pct(max(v['changes']['31,32']['rate'] for v in S['families'].values()), 0)
    conf = []
    for f, v in C['families'].items():
        ch, tr, a = v['changes'], v['transitions'], v['accuracy']; n = v['n']
        conf.append(dict(fam=f, n=n, change=f"{100 * ch['8,32']['rate']:.1f}% [{100 * ch['8,32']['ci95'][0]:.1f}, {100 * ch['8,32']['ci95'][1]:.1f}]",
                         acc8=pct(a['8']['rate']), acc32=pct(a['32']['rate']), accD=pct(v['accuracy_D']['rate']), verdict=v['verdict'],
                         k32=round(a['32']['rate'] * n), kD=round(v['accuracy_D']['rate'] * n), kch=round(ch['8,32']['rate'] * n), ww=tr['wrong_wrong_changed']['k']))
    sc = {k: json.loads((p / 'containers.json').read_text()) for k, p in (('smoke', RES / 'lateshift_smoke-20261110-171352'), ('screen', SCREEN), ('confirm', CONFIRM))}
    sec = {k: sum(v['timing']['total_seconds'] for v in c.values()) for k, c in sc.items()}
    N['gpu_screen'] = num(sec['screen'] / 60, 1); N['gpu_screen_all'] = num(sum(sec.values()) / 60, 1)
    # baseline from the earlier monitor data: L32, loops (8, 32)
    L = X['models']['L32']; pooled = L['pooled']; c = pooled['changes']['8,32']
    N['base_change'] = pct(c['rate']); N['base_n'] = f"{pooled['n']:,}"
    N['base_acc8'] = pct(pooled['accuracy']['8']['rate'], 2); N['base_acc32'] = pct(pooled['accuracy']['32']['rate'], 2)
    fam = [v['changes']['8,32']['rate'] for k, v in L.items() if k != 'pooled']
    N['base_lo'] = pct(min(fam)); N['base_hi'] = pct(max(fam))
    return N, rows, conf


def figures(raw, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = GEN / 'figs'; out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
                         'savefig.bbox': 'tight', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6, 'grid.color': '#d9d8d2', 'grid.linewidth': .5})

    def save(fig, name):
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'{name}.{ext}', dpi=200)
        plt.close(fig)

    # Fig. 1: predicted against measured response, primary dose (4,000 entries: 40 problems x 10 pushes x 10 classes)
    r = raw['03']; obs = np.array(r['obs'])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, key, title in ((axes[0], 'pred_full', "with the loop:  r · G · (J + … + J⁴⁸) · L"), (axes[1], 'pred_id', 'without the loop:  r · G · L')):
        p = np.array(r[key]); lim = 1.05 * max(abs(obs).max(), abs(np.array(r['pred_full'])).max())
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=.7, zorder=1)
        ax.scatter(p, obs, s=4, color=BLUE, alpha=.35, linewidths=0, zorder=2, rasterized=True)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal'); ax.grid(True); ax.set_axisbelow(True)
        ax.set_title(title); ax.set_xlabel('predicted change of a candidate score')
    axes[0].set_ylabel('measured change after 48 loops'); axes[0].annotate('equal', (lim * .78, lim * .78), xytext=(lim * .42, lim * .9), fontsize=8, color=MUTED)
    save(fig, 'fig_pred')

    # Fig. 2: target effect over forced loops, lever against readout gradient; hollow marker at 0 = derivative prediction
    T = raw['times']; xs = list(range(1, len(T) + 1))
    fig, ax = plt.subplots(figsize=(4.9, 2.9))
    for key, imm, col, name in (('course_L', 'imm_L', BLUE, 'lever L'), ('course_G', 'imm_G', ORANGE, 'gradient push U')):
        y = r[key]
        ax.plot([0, 1], [r[imm], y[0]], color=col, lw=1.2, ls=(0, (3, 2)))
        ax.plot(xs, y, color=col, lw=2, marker='o', ms=5, mec='white', mew=1)
        ax.plot([0], [r[imm]], marker='o', ms=6.5, mfc='white', mec=col, mew=1.6, ls='none')
    ax.annotate('lever L', (xs[-1], r['course_L'][-1]), xytext=(4, 8), textcoords='offset points', ha='right', color=INK, fontsize=8.5)
    ax.annotate('gradient push U', (xs[-1], r['course_G'][-1]), xytext=(4, 8), textcoords='offset points', ha='right', color=INK, fontsize=8.5)
    ax.text(.3, .9, 'immediate effect\n(derivative, not measured)', va='center', fontsize=8, color=MUTED)
    ax.axhline(0, color=MUTED, lw=.6); ax.set_xticks([0] + xs); ax.set_xticklabels(['0'] + [str(t) for t in T]); ax.grid(axis='y'); ax.set_axisbelow(True)
    ax.set_xlabel('forced loops before the readout'); ax.set_ylabel('mean change of the target score')
    save(fig, 'fig_course')

    # Fig. 3: what repetition adds: error of the t-pass prediction against the final response
    fig, ax = plt.subplots(figsize=(4.9, 2.9))
    for t, col, name, off in (('01', BLUE, 'push 1% of state norm', (6, -11)), ('03', ORANGE, 'push 3% of state norm', (6, 6))):
        d = raw[t]; x = list(range(len(d['e1_t']))); y = [100 * v for v in d['e1_err']]
        ax.plot(x, y, color=col, lw=2, marker='o', ms=5, mec='white', mew=1)
        ax.annotate(name, (x[-1], y[-1]), xytext=off, textcoords='offset points', ha='right', fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in raw['01']['e1_t']]); ax.set_ylim(0, 30); ax.grid(axis='y'); ax.set_axisbelow(True)
    ax.set_xlabel('passes through the block used in the prediction'); ax.set_ylabel('error against the final response (%)')
    save(fig, 'fig_passes')

    # Fig. 4: the screen: late answer changes against accuracy, one point per task family
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((10, 15), 80, 45, facecolor='#eef3fb', edgecolor=MUTED, lw=.7, ls=(0, (4, 3)), zorder=0))
    xs_ = [100 * q['x'] for q in rows]; ys_ = [100 * q['y'] for q in rows]
    ax.scatter(xs_, ys_, s=34, color=BLUE, edgecolors='white', linewidths=1, zorder=3)
    lab = {'var4': (6, 3), 'var4s': (6, -10), 'mul2add': (7, 3), 'paren': (-5, 7), 'ptr3': (7, -9), 'story2': (7, 3), 'add3': (-5, 7)}
    for q in rows:
        if q['fam'] in lab:
            ax.annotate(q['fam'], (100 * q['x'], 100 * q['y']), xytext=lab[q['fam']], textcoords='offset points', fontsize=7.5 if q['fam'].startswith('var') else 8, color=INK)
    ax.text(50, 57, 'illustration of the first-screen thresholds for L32', ha='center', va='top', fontsize=8, color=MUTED)
    ax.set_xlim(-3, 100); ax.set_ylim(-2, 60); ax.grid(True); ax.set_axisbelow(True)
    ax.set_xlabel('L32 answers correct at 32 loops (%)'); ax.set_ylabel('answer changed between loops 8 and 32 (%)')
    save(fig, 'fig_screen')


def huginn(H):
    """Formatted Huginn numbers; every value comes from docs/steering_gen/huginn_facts.json (value + source per key)."""
    v = lambda k: json.loads(H[k]['value']) if isinstance(H[k]['value'], str) and H[k]['value'].startswith('[') else H[k]['value']
    N = {}
    for f in 'lgc':
        N[f'eff_{f.upper()}'] = sgn(v(f's4_p1_mean_target_effect_{f}'), 2)
    N['L_minus_G'] = iv(v('s4_p1_l_minus_g_target_ci'), 2); N['random_abs'] = num(v('s4_p1_random_median_abs'), 3)
    for k in ('full', 'averaged', 'swapped', 'swapped_same_digit', 'identity'):
        N[f'corr_{k}'] = num(v(f's4_p3_median_corr_{k}'), 3)
    N['slope_full'] = num(v('s4_p3_median_slope_full'), 2); N['err_full'] = pct(v('s4_p3_median_norm_error_full'))
    for k in ('identity', 'averaged', 'swapped', 'swapped_same_digit'):
        N[f'full_minus_{k}'] = iv(v(f's4_p3_full_minus_{k}_ci'), 2)
    N['adjoint'] = sci(v('s4_adjoint_rel_err_median')); N['tail_q95'] = sci(max(v('s4_train_tail_q95'), v('s4_pred_tail_q95')))
    N['displacement'] = num(v('s4_endpoint_displacement_l_median'), 2)
    for f in 'lg':  # linearised immediate (c0) and one-loop (c1) target responses of the population pushes, norm 3
        N[f'c0_{f.upper()}'] = sgn(v(f's6_summary_r2_{f}_c0'), 2); N[f'c1_{f.upper()}'] = sgn(v(f's6_summary_r2_{f}_c1'), 2 if f == 'l' else 3)
        N[f'inspan_{f.upper()}'] = num(v(f's6_summary_r2_{f}_inspan'), 2); N[f'gain_{f.upper()}'] = num(v(f's6_summary_r2_{f}_gain'), 2)
    N['rho_span'] = num(v('s6_summary_r1_median_rho'), 2)
    for f, name in (('a', 'adapter'), ('w', 'Ws'), ('b', 'onestep'), ('l', 'lever')):
        N[f'cheap_{name}'] = sgn(v(f's8_target_effect_mean_{f}_mean'), 2)
        if f != 'l':
            N[f'cheap_{name}_share'] = pct(v(f's8_ratio_of_means_to_l_{f}'), 0)
    N['ovl_a'] = sgn(v('s9_h1_a_excess_mean'), 2) + ' ' + iv(v('s9_h1_a_ci97_5'), 2); N['ovl_b'] = sgn(v('s9_h2_b_excess_mean'), 2) + ' ' + iv(v('s9_h2_b_ci97_5'), 2)
    N['ovl_a_ctrl_share'] = pct(v('s9_h1_a_control_mean') / v('s9_h1_a_observed_mean'), 0); N['ovl_b_ctrl_share'] = pct(v('s9_h2_b_control_mean') / v('s9_h2_b_observed_mean'), 0)
    N['dstar'] = num(v('s3_p1_dstar'), 1); N['probe'] = num(v('s3_p1_mm'), 2)
    N['per_unit_norm'] = num(v('s4_p1_mean_target_effect_l') / 3.0, 2)
    N['_sources'] = sorted({H[k]['source'].split(':')[0] for k in H if k.startswith(('s3_p1', 's4_', 's6_summary_r', 's8_', 's9_'))})
    return N


def readspace(lread=None):
    """One column per (model, contexts): mean self-gain of random directions, of sampled read directions (C3), excess."""
    lread = Path(lread or LREAD)
    L = json.loads((lread / 'lread_summary.json').read_text()); lm = json.loads((lread / 'meta.json').read_text())
    lz = np.load(lread / 'lread.npz')
    cols = []

    def col(name, ev, rot, n, extra=None, nd=3):
        c = dict(name=name, n=str(n), tau=num(ev['tau'][0], nd), tau_ci=iv(ev['tau'][1:], nd),
                 c3_alpha=num(rot['C3']['alpha'], nd), c3_excess=sgn(ev['C3']['Delta'][0], nd), c3_ci=iv(ev['C3']['Delta'][1:], nd),
                 c3_below=pct(ev['C3']['below_all'], 0), c3_Z=sgn(ev['C3']['Z'][0], 1),
                 task_alpha=num(rot['task']['alpha'], nd), task_excess=sgn(ev['task']['Delta'][0], nd), task_ci=iv(ev['task']['Delta'][1:], nd),
                 probe_norm=num(rot['probes']['norm1'], 2), c3_norm=num(rot['C3']['norm1'], 2), task_norm=num(rot['task']['norm1'], 2),
                 probe_cos=num(rot['probes']['cos1'], 2), c3_cos=num(rot['C3']['cos1'], 2), task_cos=num(rot['task']['cos1'], 2))
        c.update(extra or {}); cols.append(c)

    def huginn_rot(run):
        """Medians for the Huginn control runs, from their saved arrays with analyze_ouro.rotation (same code as the others)."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from analyze_ouro import rotation
        z = np.load(run / 'ctrlprobe.npz'); d = z['done']
        return rotation(z['gram3'][d], z['ugram'][d], z['x1'][d], z['fam'], z['G'].shape[-1])

    for name, run in (('Huginn, true/false', CTRL_T), ('Huginn, digits', CTRL_D)):
        ev = json.loads((run / 'ctrl_summary.json').read_text())['main']
        col(name, ev, huginn_rot(run), ev['n'], dict(reading='deficit'))
    for key, name in (('A', 'L32, sums'), ('B', 'L32, true/false')):
        m = L[key]['main']; pr = m['primary']
        col(name, m['evaluate'], m['rotation'], L[key]['gates']['n_done'],
            dict(reading=pr['reading'].lower() + ('; ' + pr['sublabel'].lower() if pr.get('sublabel') else ''), tau_near_zero=pr.get('tau_near_zero')), nd=4)
    o = json.loads((OURO / 'ouro_summary.json').read_text()); e = o['loop_k3']
    col('Ouro-1.4B, true/false', e, e['rotation'], o['gates']['n_done'], dict(reading=o['PRIMARY'].lower() + ('; positive excess' if e['C3']['Delta'][1] > 0 else '')))
    A, B = L['A'], L['B']; gA = A['gates']
    N = dict(run=lread.name, verdict_A=A['reading'], verdict_B=B['reading'], sublabel_A=A['main']['primary'].get('sublabel') or '',
             branch_A=A['main']['primary'].get('branch', ''), tau_near_zero_A=str(A['main']['primary'].get('tau_near_zero')),
             meaning=L['meaning_for_writeup'], coord_A=A['main']['coordinate_control']['reading'],
             c1_A=A['main']['evaluate']['C1']['reading'], c2_A=A['main']['evaluate']['C2']['reading'], c3_read_A=A['main']['evaluate']['C3']['reading'],
             c3flip_A=sgn(A['main']['evaluate']['C3flip']['Delta'][0], 3) + ' ' + iv(A['main']['evaluate']['C3flip']['Delta'][1:], 3),
             c1_excess_A=sgn(A['main']['evaluate']['C1']['Delta'][0], 3), c2_excess_A=sgn(A['main']['evaluate']['C2']['Delta'][0], 3),
             g1_max=sci(max(lm['gates']['G1'])), adjoint_max=sci(max(max(L[k]['gates']['adjoint_by_family'].values()) for k in 'AB')),
             identity=sci(gA['identity_relative_error']), state_identity=sci(gA['state_relative_error']),
             drift_A=sci(gA['median_drift']), drift_A_max=sci(gA['max_drift']), correct_A=f"{gA['n_correct']} of {gA['n_done']}",
             correct_B=f"{int(lz['B_correct'][lz['B_done']].sum())} of {int(lz['B_done'].sum())}",
             drift_B=sci(float(np.median(lz['B_drift'][lz['B_done']]))), drift_B_max=sci(float(lz['B_drift'][lz['B_done']].max())),
             dropped_A=str(A.get('drift_excluded', 0)), dropped_B=str(B.get('drift_excluded', 0)),
             gpu_min=num(lm['runtime_s'] / 60, 1),
             **{f'{f}_{k}': sgn(L[k]['main']['evaluate'][f]['Delta'][0], 4) + ' ' + iv(L[k]['main']['evaluate'][f]['Delta'][1:], 4)
                for k in 'AB' for f in ('C1', 'C2', 'C3flip')},
             **dict(zip(('rms_A', 'rms_A_min', 'rms_A_max'), (num(float(v), 2) for v in (
                 lambda q: (np.median(q), q.min(), q.max()))(np.sqrt(lz['A_gram3'][:, (lz['A_fam'] == 4) | (lz['A_fam'] == 5), 1, 1].mean(1)))))),
             gpu_min_all=num((lm['runtime_s'] + sum(json.loads((q / 'meta.json').read_text())['runtime_s'] for q in LREAD_SMOKES)) / 60, 1),
             n_probe_values=str(A['main']['absolute_alpha']['probes']['n']),
             **{f'{fam}_{st}_{k}': num(L[k]['main']['absolute_alpha'][fam][st], 3) for k in 'AB' for fam in ('probes', 'C3', 'task') for st in ('min', 'max', 'median')},
             tau_A=num(A['main']['evaluate']['tau'][0], 4), tau_A_ci=iv(A['main']['evaluate']['tau'][1:], 4),
             c3_A=sgn(A['main']['evaluate']['C3']['Delta'][0], 4), c3_A_ci=iv(A['main']['evaluate']['C3']['Delta'][1:], 4),
             c3_A_lo=num(abs(A['main']['evaluate']['C3']['Delta'][1]), 4),
             c3_B=sgn(B['main']['evaluate']['C3']['Delta'][0], 3), c3_B_ci=iv(B['main']['evaluate']['C3']['Delta'][1:], 3),
             task_A=sgn(A['main']['evaluate']['task']['Delta'][0], 4), task_A_ci=iv(A['main']['evaluate']['task']['Delta'][1:], 4),
             sens_B=B['sensitivity_drift_le_1e-3']['primary']['reading'].lower(), sens_B_n=str(B['sensitivity_drift_le_1e-3']['n']),
             above5_A=str(A['behaviour']['drift']['above_1e_5']), sd_pred=num(A['main']['evaluate']['probe_sd']['pred_rms'], 4),
             sd_obs=num(A['main']['evaluate']['probe_sd']['obs_rms'], 4))
    return N, cols


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    N, raw = lever(); S, rows, conf = screen(); N.update(S)
    hf = GEN / 'huginn_facts.json'
    out = dict(N=N, screen_rows=[{k: v for k, v in q.items() if k not in ('x', 'y', 'xd', 'chance')} for q in rows], confirm_rows=conf,
               H=huginn(json.loads(hf.read_text())))
    if (LREAD / 'lread_summary.json').exists():
        out['RS'], out['rs_cols'] = readspace()
    (GEN / 'numbers.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    figures(raw, rows)
    print(json.dumps(out['H'], indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
