"""Precision simulation for docs/oversight_monitor.md §7 / docs/mon_spec.md B.

Construction: y1 ~ Bernoulli(p); independently with probability .7 set y2=y1,
otherwise draw y2 ~ Bernoulli(p). Thus P(agreement)=.7+.3*(p²+(1-p)²),
and chance-corrected agreement (Cohen's kappa) is .70 at every error rate.
For model m, X_m[i,j] = delta*y_m[i] + z_m[i] + .2*e_m[i,j], d=2048.
Paired z and paired e are Gaussian with correlation .7 across models, with
independence across items, coordinates and the two noise blocks. Conditional
on each label the features are multivariate Gaussian. The sufficient score is
the coordinate mean; its conditional variance is v=1+.04/d, so its population
AUROC is Phi(delta/sqrt(2*v)). Set delta=sqrt(2*v)*Phi^-1(.8). The library
fits the genuine 2048-dimensional matrices, with no access to this direction.
A large independent sample verifies the analytic optimum numerically without
materializing redundant feature coordinates: coordinate-mean noise is N(0,v).
This low effective rank construction is deliberately learnable at these n;
it does not model a general high-dimensional isotropic representation.

Both models use one shared outer-fold assignment balanced over the four joint
correctness categories, as the 2x2 table of the analysis does. Bootstrap strata
are those folds (one family), with identical item resamples for both models. AUROC is always averaged over folds,
never pooled across independently calibrated fold scores. Intervals condition
on the fitted scores, so omit refitting/selection uncertainty by design.

Default: pilot the most expensive n=1500 cell, project runtime for 40 repeats
per cell, and reduce repeats to fit a conservative 25-minute compute budget
(reserving time for variability). --reps can request an explicit count. Each
replication runs real 5x4 nested CV with all nine lambdas for both models.
--jobs runs replications in parallel processes; seeds, and so results, do not
depend on it.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import concurrent.futures
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
import time
import numpy as np
from scipy import special
from mon_probe import nested_cv, auroc, paired_bootstrap, make_joint_folds

D = 2048
V = 1 + .04 / D
DELTA = np.sqrt(2*V) * special.ndtri(.8)


def generate(n, p, seed):
    rng = np.random.default_rng(seed)
    y1 = (rng.random(n) < p).astype(int)
    y2 = np.where(rng.random(n)<.7, y1, rng.random(n)<p).astype(int)
    z1 = rng.normal(size=n); z2 = .7*z1 + np.sqrt(1-.7**2)*rng.normal(size=n)
    e1 = rng.normal(size=(n,D))
    e2 = .7*e1 + np.sqrt(1-.7**2)*rng.normal(size=(n,D))
    return ((DELTA*y1+z1)[:,None]+.2*e1, y1,
            (DELTA*y2+z2)[:,None]+.2*e2, y2)


def replication(n, p, seed, n_boot):
    start = time.perf_counter()
    x1,y1,x2,y2 = generate(n,p,seed)
    shared = make_joint_folds(np.column_stack((y1,y2)),5,seed+29)
    r1 = nested_cv({'state':x1},y1,seeds=(0,),folds=[shared])[0]
    r2 = nested_cv({'state':x2},y2,seeds=(0,),folds=[shared])[0]
    def stat(idx):
        vals=[]
        for r,y in ((r1,y1),(r2,y2)):
            vals.append(np.mean([auroc(r['scores'][idx[r['folds'][idx]==f]],
                                      y[idx[r['folds'][idx]==f]]) for f in range(5)]))
        return vals[0]-vals[1]
    est,low,high,draws = paired_bootstrap(stat,np.column_stack([r1['folds'],r2['folds']]),
                                         n_boot=n_boot,seed=seed+17)
    # Analysis policy: if any resample loses a class in a fold, the interval is unavailable (never drop draws).
    bad=int((~np.isfinite(draws)).sum())
    return dict(auc1=r1['mean_auroc'],auc2=r2['mean_auroc'],difference=float(est),
                interval=[None,None] if bad else [float(low),float(high)],
                width=None if bad else float(high-low),undefined_bootstrap_draws=bad,
                seconds=time.perf_counter()-start,
                nonconverged=sum(not m.converged for r in (r1,r2) for m in r['models']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps',type=int,default=None)
    parser.add_argument('--budget-seconds',type=float,default=1500)
    parser.add_argument('--n-boot',type=int,default=2000)
    parser.add_argument('--jobs',type=int,default=1)
    args=parser.parse_args()
    if args.reps is not None and args.reps < 2:
        parser.error('--reps must be at least 2 for a sample standard deviation')
    start=time.perf_counter()
    rng=np.random.default_rng(92471)
    verification={}
    for p in (.1,.3,.5):
        y=(rng.random(400000)<p).astype(int)
        value=auroc(DELTA*y+np.sqrt(V)*rng.normal(size=len(y)),y)
        assert abs(value-.8)<.006
        verification[str(p)]=value
    print('Population AUROC: analytic=0.800000; numerical='+str(verification),flush=True)
    # A pilot is retained as the first replicate in its cell.
    pilot=replication(1500,.3,30001500,args.n_boot)
    projected=pilot['seconds']*6*40
    reps=args.reps if args.reps is not None else min(40,max(2,int(args.budget_seconds/(6*pilot['seconds']*1.25))))
    print(f'Pilot pair: {pilot["seconds"]:.2f}s; projected 40/cell: {projected/60:.1f} min; using {reps}/cell',flush=True)
    cells=[]
    todo=[(p,n,rep) for p in (.1,.3,.5) for n in (800,1500) for rep in range(reps) if (p,n,rep)!=(.3,1500,0)]
    done={(.3,1500,0):pilot}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(replication,n,p,1000000+int(p*100)*100000+n*50+rep,args.n_boot):(p,n,rep) for p,n,rep in todo}
        for future in concurrent.futures.as_completed(futures):
            p,n,rep=futures[future]; row=done[p,n,rep]=future.result()
            print(f'p={p:.2f} n={n} rep={rep+1}/{reps}: auc={row["auc1"]:.4f}/{row["auc2"]:.4f}, {row["seconds"]:.1f}s',flush=True)
    for p in (.1,.3,.5):
        for n in (800,1500):
            runs=[done[p,n,rep] for rep in range(reps)]
            cells.append(dict(error_rate=p,n=n,replications=reps,
                              sd_auroc=float(np.std([r['auc1'] for r in runs],ddof=1)),
                              sd_paired_difference=float(np.std([r['difference'] for r in runs],ddof=1)),
                              intervals_unavailable=sum(r['width'] is None for r in runs),
                              mean_paired_interval_width=float(np.mean([r['width'] for r in runs if r['width'] is not None])),
                              interval_covers_zero=float(np.mean([r['interval'][0]<=0<=r['interval'][1] for r in runs if r['width'] is not None])),
                              nonconverged_outer_fits=int(sum(r['nonconverged'] for r in runs)),runs=runs))
    output=dict(d=D,analytic_auroc=.8,numerical_auroc=verification,
                requested_replications=40,actual_replications=reps,
                projected_40_seconds=projected,seconds=time.perf_counter()-start,
                n_boot=args.n_boot,cells=cells)
    destination=Path(__file__).resolve().parents[1]/'results'/'mon_power_sim.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(output,indent=2)+'\n')
    print('\nerror_rate    n  reps  SD(AUROC)  SD(paired diff)  mean CI width',flush=True)
    for c in cells:
        print(f'{c["error_rate"]:10.2f} {c["n"]:4d} {reps:5d} {c["sd_auroc"]:10.4f} {c["sd_paired_difference"]:16.4f} {c["mean_paired_interval_width"]:14.4f}',flush=True)
    print(f'Wrote {destination}; elapsed {output["seconds"]:.1f}s',flush=True)
    if reps<40:
        print('Reduced replications for runtime; SD estimates are imprecise at this replication count.',flush=True)


if __name__=='__main__':
    main()
