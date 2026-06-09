"""Regenerate the β0.6 27-combo (D,R,A) factorial for CON (BAU·β0.6) and IAP (CE·β0.6),
reusing the EXACT pipeline functions by exec-ing only the setup cells of:
  3-mining-area-simulation/5-2-map-extraction.ipynb        (cells <=7: allocate_demand_to_extraction)
  3-mining-area-simulation/6-estimate-mining-area.ipynb    (cells <=7: estimate_mining_area)
  4-biodiversity-risk-assessment/6-biodiversity-risk-assessment.ipynb (cells <=3: assess_biodiversity_risk)
The loop/usage cells (which run the full sweep) are NOT exec-ed.

Phases (only the MISSING combos are generated; existing central+anti-diagonal reused):
  A  12 allocations  : BAU,CE × β0.6 × 6 missing (D,R)
  B  48 footprints   : BAU,CE × β0.6 × 24 missing (D,R,A)   [joblib]
  C  54 overlaps     : BAU(→CON),CE(→IAP) × β0.6 × all 27 (D,R,A) → global sums + factorial CSV

Usage:  python _regen_b06_factorial.py test   # 1 combo per phase, timed
        python _regen_b06_factorial.py full   # the whole thing
"""
import os, sys, json, time
from pathlib import Path

REPO = Path(r'D:/wty/Documents/GitHub/aggregate-mining-biodiversity')
os.chdir(REPO / '3-mining-area-simulation')          # so each notebook's Path.cwd().parent == REPO
MODE = sys.argv[1] if len(sys.argv) > 1 else 'full'

def exec_setup(nb_rel, upto):
    """exec code cells with index <= upto from a notebook into THIS module's globals()."""
    nb = json.load(open(REPO / nb_rel, encoding='utf-8'))
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code' or i > upto:
            continue
        src = ''.join(c['source'])
        if src.strip():
            exec(compile(src, f'{nb_rel}#cell{i}', 'exec'), globals())

t0 = time.time()
print('[setup] exec-ing pipeline setup cells ...', flush=True)
exec_setup('3-mining-area-simulation/5-2-map-extraction.ipynb', 7)              # allocate_demand_to_extraction
exec_setup('3-mining-area-simulation/6-estimate-mining-area.ipynb', 7)          # estimate_mining_area, fit, predict_interval
exec_setup('4-biodiversity-risk-assessment/6-biodiversity-risk-assessment.ipynb', 3)  # assess_biodiversity_risk
from joblib import Parallel, delayed
import numpy as np, pandas as pd
assert all(k in globals() for k in
           ('allocate_demand_to_extraction', 'estimate_mining_area', 'assess_biodiversity_risk',
            'trade_codes', 'a_1', 'mining_suitability_1km', 'RESULTS')), 'setup incomplete'
print(f'[setup] done in {time.time()-t0:.1f}s | RESULTS={RESULTS}', flush=True)

LEVELS = ['L', 'M', 'H']
EXISTING_DR  = {('M', 'M'), ('L', 'H'), ('H', 'L')}                  # central + anti-diagonal corners
EXISTING_DRA = {('M', 'M', 'M'), ('L', 'H', 'L'), ('H', 'L', 'H')}
ALLOC_MISSING_DR  = [(d, r) for d in LEVELS for r in LEVELS if (d, r) not in EXISTING_DR]            # 6
FOOT_MISSING_DRA  = [(d, r, a) for d in LEVELS for r in LEVELS for a in LEVELS
                     if (d, r, a) not in EXISTING_DRA]                                               # 24
ALL_DRA           = [(d, r, a) for d in LEVELS for r in LEVELS for a in LEVELS]                      # 27
SCENS = ['BAU', 'CE']
BETA  = 0.6
N_JOBS = 8

# preloaded shared inputs for the footprint step (mirrors nb6 cell 13)
suit_1km       = mining_suitability_1km
fit            = (a_1, b_1, popt_1, pcov_1, dof_1)
cap_by_reserve = {lvl: rio.open(RESULTS / 'mining-area' / f'aggregate-reserve-1km-{lvl}.tif').read(1)
                  for lvl in LEVELS}

if MODE == 'test':
    ALLOC_MISSING_DR = [('M', 'H')]
    FOOT_MISSING_DRA = [('M', 'H', 'M')]
    ALL_DRA          = [('M', 'H', 'M')]
    SCENS_C          = [('BAU', 'CON')]
    N_JOBS           = 2
else:
    SCENS_C = [('BAU', 'CON'), ('CE', 'IAP')]

# ---------------- Phase A: allocations ----------------
tA = time.time()
print(f'\n[A] allocations: {len(SCENS)*len(ALLOC_MISSING_DR)} combos', flush=True)
for scen in SCENS:
    for d, r in ALLOC_MISSING_DR:
        ts = time.time()
        allocate_demand_to_extraction(scen, d, r, beta=BETA, max_r=3, max_r_surplus=10)
        print(f'[A] {scen} B{BETA} D{d}-R{r}  ({time.time()-ts:.1f}s)', flush=True)
print(f'[A] done in {time.time()-tA:.1f}s', flush=True)

# ---------------- Phase B: footprints (joblib) ----------------
tB = time.time()
foot_combos = [(scen, d, r, a) for scen in SCENS for (d, r, a) in FOOT_MISSING_DRA]
print(f'\n[B] footprints: {len(foot_combos)} combos, n_jobs={N_JOBS}', flush=True)
msgs = Parallel(n_jobs=N_JOBS, backend='loky', verbose=5)(
    delayed(estimate_mining_area)(scen, BETA, demand_level=d, reserve_level=r, area_level=a,
                                  suit_1km=suit_1km, cap_1km=cap_by_reserve[r], fit=fit)
    for scen, d, r, a in foot_combos)
for m in msgs:
    print('  ', m, flush=True)
print(f'[B] done in {time.time()-tB:.1f}s', flush=True)

# ---------------- Phase C: overlap → global sums + factorial CSV ----------------
tC = time.time()
recs = []
print(f'\n[C] overlaps: {len(SCENS_C)*len(ALL_DRA)} combos', flush=True)
for scen, label in SCENS_C:
    for d in (LEVELS if MODE != 'test' else ['M']):
        for r in (LEVELS if MODE != 'test' else ['H']):
            for a in (LEVELS if MODE != 'test' else ['M']):
                overlap, _ = assess_biodiversity_risk(scen, BETA, demand_level=d,
                                                      reserve_level=r, area_level=a)
                recs.append({'scenario': label, 'beta': BETA, 'demand': d, 'reserve': r,
                             'area': a, 'overlap': float(np.sum(overlap))})
fac = pd.DataFrame(recs)
out_csv = RESULTS / 'biodiversity-risk' / (
    'biodiversity-risk-summary-B0.6-signposts-DRA-factorial.csv' if MODE != 'test'
    else '_TEST-b06-factorial.csv')
fac.to_csv(out_csv, index=False)
print(f'[C] done in {time.time()-tC:.1f}s | wrote {out_csv} ({len(fac)} rows)', flush=True)
print(fac.round(3).to_string(index=False), flush=True)
print(f'\n[ALL] total {time.time()-t0:.1f}s', flush=True)
