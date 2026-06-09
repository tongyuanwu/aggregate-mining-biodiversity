"""
Generate the cumulative-measure (BAU -> CE) scenario ladder for Supplementary
Figure 8-10 at n_sim=10,000, then plot the per-year MEDIAN (the figure draws a
smooth central trajectory; n_sim=1 gives jagged single-draw lines).

The 7-rung ladder adds CE levers one at a time, in the SI-figure wedge order:
    BAU -> +more intensive use -> +lifetime extension -> +component reuse ->
    +end-of-life recycling -> +aggregate substitution -> +clinker substitution(=CE)

Endpoints reuse the production 10,000-run scenarios already on disk:
    BAU  == production 'BAU'  (levers = {} )            -> not recomputed
    CE   == production 'CE'   (levers = ALL)            -> not recomputed
(verified bit-identical). Only the 5 INTERMEDIATE rungs are computed here, on the
production 10,000-run 'historical' basis, and each scenario folder is PRUNED to the
three <material>_extraction.npz the figure needs (a full 10k folder is ~180 MB;
pruned it is a few MB), since these cumulative scenarios are figure-only.

Run (use the wty env so numpy's LAPACK loads correctly):
    conda run -n wty python 1-IMAGINE-Concrete/run-wedge-scenarios.py --smoke
    conda run -n wty python 1-IMAGINE-Concrete/run-wedge-scenarios.py
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from IMAGINE_Concrete import run_projection, load_shared_data, RESULTS, DATA

N_SIM = 10000
SEED = 42
HIST_SUBDIR = 'historical'   # production 10,000-run historical basis
KEEP = {'gravel_extraction.npz', 'sand_extraction.npz', 'limestone_extraction.npz'}

LEVER_COL = {
    'more_intensive_use': 'intensive_use',
    'lifetime_extension': 'lifetime_ext',
    'reuse_rate':         'reuse',
    'recycle_rate':       'recycling',
    'sand_sub':           'sand_substitution',
    'clinker_target':     'clinker_ratio_target',
}
ALL_LEVERS = set(LEVER_COL)

# only the 5 INTERMEDIATE rungs (BAU == production BAU, CE == production CE)
INTERMEDIATE = [
    ('wedge_MIU', {'more_intensive_use'}),
    ('wedge_LE',  {'more_intensive_use', 'lifetime_extension'}),
    ('wedge_RE',  {'more_intensive_use', 'lifetime_extension', 'reuse_rate'}),
    ('wedge_RC',  {'more_intensive_use', 'lifetime_extension', 'reuse_rate', 'recycle_rate'}),
    ('wedge_SS',  {'more_intensive_use', 'lifetime_extension', 'reuse_rate', 'recycle_rate', 'sand_sub'}),
]


def _load_levers():
    ce = pd.read_csv(DATA / 'CE-parameters-142.csv').set_index('economy')
    oss = pd.read_csv(DATA / 'oresand-supply-cap-142.csv').set_index('economy')
    return ce, oss


def ce_kwargs(economy, ce, oss, levers):
    r = ce.loc[economy]
    kw = {}
    for arg, col in LEVER_COL.items():
        on = arg in levers
        if arg == 'clinker_target':
            kw[arg] = float(r[col]) if on else None
        else:
            kw[arg] = float(r[col]) if on else 0.0
    kw['oresand_supply'] = float(oss.loc[economy, 'oresand_supply_kt_yr']) if 'sand_sub' in levers else np.inf
    return kw


def run_and_prune(scen, economy, levers, ce, oss):
    d = RESULTS / 'material-flows' / economy / scen
    # idempotent: skip economies already finished & pruned (resume-friendly)
    if d.exists() and {p.name for p in d.glob('*.npz')} == KEEP:
        return
    run_projection(scen, economy, n_sim=N_SIM, seed=SEED, hist_subdir=HIST_SUBDIR,
                   **ce_kwargs(economy, ce, oss, levers))
    for f in d.glob('*.npz'):
        if f.name not in KEEP:
            f.unlink()


def run_all(economies, ce, oss, n_jobs=12, scenarios=INTERMEDIATE):
    for scen, levers in scenarios:
        print(f'>>> {scen}  levers={sorted(levers)}', flush=True)
        Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
            delayed(run_and_prune)(scen, e, levers, ce, oss) for e in economies)


def smoke(economy='CHN'):
    ce, oss = _load_levers()
    scen, levers = INTERMEDIATE[0]
    run_and_prune(scen, economy, levers, ce, oss)
    d = RESULTS / 'material-flows' / economy / scen
    files = sorted(p.name for p in d.glob('*.npz'))
    for k in KEEP:
        a = np.load(d / k)['data']
        print(f'  {k:22s} shape={a.shape}  median[:3]={np.median(a, axis=0)[:3]}')
    print(f'  pruned to {len(files)} files: {files}')
    print('SMOKE OK')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--n-jobs', type=int, default=12)
    p.add_argument('--only', default='', help='comma-sep short names, e.g. RC,SS (default: all 5)')
    args = p.parse_args()

    if args.smoke:
        shared = load_shared_data()
        smoke('CHN')
    else:
        shared = load_shared_data()
        economies_142 = shared['df_cement_production'].columns.tolist()
        ce, oss = _load_levers()
        scenarios = INTERMEDIATE
        if args.only:
            want = {f'wedge_{s.strip()}' for s in args.only.split(',')}
            scenarios = [(scen, lev) for scen, lev in INTERMEDIATE if scen in want]
        print('scenarios:', [s for s, _ in scenarios], flush=True)
        run_all(economies_142, ce, oss, n_jobs=args.n_jobs, scenarios=scenarios)
        print('DONE: intermediate rungs (n_sim=10,000, pruned) for', len(economies_142), 'economies')
