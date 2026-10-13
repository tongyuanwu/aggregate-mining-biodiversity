"""Resume the spatial pipeline after the SSP2 MFA groups pass validation.

Usage: python run-mfa-downstream.py prepare|preflight|run <MFA run directory>
Other SSP paths may continue concurrently. Model equations come from frozen copies of the existing notebooks. Outputs are
computed in isolation, checked, compared with saved originals, then installed.
"""
import contextlib
import functools
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

# Reuse the Python 3.12 raster runtime used by the prior spatial refresh.
# The MFA conda environment has a broken MINPACK DLL, so it is not used here.
if sys.version_info[:2] == (3, 12):
    _repo = next(p for p in Path(__file__).resolve().parents if (p / '1-IMAGINE-Concrete').is_dir())
    sys.path.insert(0, str(_repo / 'outputs/mining-area-nonnegative-20261010/runtime'))

import numpy as np
import pandas as pd
import rasterio as rio
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from scipy.stats import t
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm as _tqdm

NOTEBOOKS = {
    'demand': ('3-mining-area-simulation/1-downscale-extraction-demand.ipynb', [2, 3, 4]),
    'allocation': ('3-mining-area-simulation/5-2-map-extraction.ipynb', [3, 4, 5, 6, 7]),
    'area': ('3-mining-area-simulation/6-estimate-mining-area.ipynb', [4, 5, 6, 7]),
    'risk': ('4-biodiversity-risk-assessment/6-biodiversity-risk-assessment.ipynb', [3]),
    'cost': ('3-mining-area-simulation/5-3-cost-analysis.ipynb', [3, 4, 5, 6, 8]),
}
BASE_SCENARIOS = ['BAU', 'MIU', 'LE', 'RE', 'RC', 'SS', 'CCR', 'CE']
MATERIALS = ['sand', 'gravel', 'limestone']
FUTURE_RESULTS = Path(r'F:\aggregates-mining-biodiversity\results')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tqdm(iterable, **kwargs):
    return _tqdm(iterable, disable=True, **kwargs)


def progress(base, phase, **kwargs):
    item = dict(status='running', phase=phase, updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kwargs)
    dump(Path(base) / 'downstream/progress.json', item)
    print(json.dumps(item, ensure_ascii=False), flush=True)


def paths(base):
    base = Path(base).resolve()
    return base, base / 'downstream', base.parents[1]


def ssp2_readiness(base):
    """Require accepted outputs for history and every SSP2 group, including wedges.

    A marker is written only after calculation, validation, comparison and output
    installation succeed. Unrelated SSP failures do not invalidate these results.
    """
    base = Path(base)
    plan = read(base / 'run-plan.json')
    scenarios = [s for s in plan['scenarios'] if not s.startswith('BAU_SSP')]
    if 'historical' not in scenarios or len(scenarios) != 23:
        raise ValueError('Expected history plus 22 SSP2 scenario groups')
    completed = 0
    missing = []
    for scenario in scenarios:
        for economy in plan['economies']:
            marker = base / 'state' / scenario / f'{economy}.json'
            comparison = base / 'comparisons' / scenario / f'{economy}.json'
            if marker.exists() and read(marker).get('status') == 'complete' and comparison.exists():
                completed += 1
            else:
                missing.append(f'{economy}/{scenario}')
    return dict(ready=not missing, completed=completed, total=len(scenarios)*len(plan['economies']),
                scenarios=scenarios, missing=missing)


def copy_once(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    return dict(original=str(source), snapshot=str(target), sha256=sha(target))


def prepare(base):
    base, out, repo = paths(base)
    if (out / 'manifest.json').exists():
        return read(out / 'manifest.json')
    if shutil.disk_usage(base).free < 35 * 2**30:
        raise RuntimeError('At least 35 GiB free space is required for downstream staging')
    originals = repo / 'results'
    records, previous = [], {}
    for relative, _ in NOTEBOOKS.values():
        records.append(copy_once(repo / relative, out / 'code' / Path(relative).name))
    records.append(copy_once(repo / '3-mining-area-simulation/area_capacity.py', out / 'code/area_capacity.py'))
    copy_once(Path(__file__).resolve(), out / 'code/downstream_worker.py')
    copy_once(repo / '3-mining-area-simulation/mfa_downstream_report.py', out / 'code/downstream_report.py')
    records.append(copy_once(repo / 'data/country-pseudo-values.csv', out / 'inputs/data/country-pseudo-values.csv'))
    static = [originals / n for n in ['pseudo-codes-economies-142.csv', 'historical-trade-matrix.csv',
        'bilateral-distance-matrix.csv', 'sweep-config.json', 'extraction-area-relationship.csv',
        'global-mining-suitability-1km.tif']]
    for folder, pattern in [('economy-rasters', 'economy-raster-eck4-*.tif'),
                            ('country-rasters', 'country-raster-eck4-*.tif'),
                            ('mining-area', 'aggregate-reserve-*.tif'),
                            ('mining-area', 'effective-extraction-capacity-*.tif'),
                            ('area-of-habitat-50km', '*.tif')]:
        static += [p for p in (originals / folder).glob(pattern) if 'supplement' not in p.name]
    for p in static:
        records.append(copy_once(p, out / 'inputs/results' / p.relative_to(originals)))
    for year in [2020, 2025, 2030, 2035, 2040, 2045, 2050]:
        p = FUTURE_RESULTS / 'building-mineral-rasters' / f'global-building-minerals-{year}-SSP2-50km.tif'
        records.append(copy_once(p, out / 'inputs/future-results' / p.relative_to(FUTURE_RESULTS)))
    for folder in ['extraction-rasters', 'mining-area', 'biodiversity-risk', 'economy-rasters']:
        for p in (originals / folder).iterdir():
            if not p.is_file() or (folder == 'mining-area' and not p.name.startswith('mining-area-expansion-')):
                continue
            if folder == 'economy-rasters' and 'supplement' not in p.name:
                continue
            rel = p.relative_to(originals)
            row = copy_once(p, out / 'before/results' / rel)
            if sha(p) != row['sha256']:
                raise RuntimeError(f'Baseline changed during snapshot: {p}')
            previous[rel.as_posix()] = row['sha256']
    cfg = read(out / 'inputs/results/sweep-config.json')
    scenarios = list(dict.fromkeys(BASE_SCENARIOS + cfg['SCENARIOS']))
    cases = list(dict.fromkeys(
        [(s, 0., d, r, a) for s in BASE_SCENARIOS for d, r, a in itertools.product('LMH', repeat=3)] +
        [(s, b, d, r, a) for s in ['BAU', 'CE'] for b in [.6, .7] for d, r, a in itertools.product('LMH', repeat=3)] +
        [(s, b, 'M', 'M', 'M') for s in cfg['SCENARIOS'] for b in cfg['BETAS']]))
    assert len(cases) == 461 and len(scenarios) == 17
    manifest = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), inputs=records,
                    before=previous, cases=cases, scenarios=scenarios, sweep=cfg,
                    source_results=str(originals), mfa_results=str(FUTURE_RESULTS / 'material-flows'),
                    model_changes='None; existing notebook functions, with input/output paths redirected',
                    scope='Existing 17 spatial-demand scenarios and 461 D/R/A/beta combinations; SSP and wedge MFA scenarios remain MFA-only as in the original pipeline',
                    knee_tolerance=.001, comparison_beta=.7)
    dump(out / 'manifest.json', manifest)
    setup_stage(base, 'staged')
    progress(base, 'waiting_for_ssp2', cases=len(cases), scenarios=len(scenarios))
    return manifest


def setup_stage(base, stage):
    _, out, _ = paths(base)
    results = out / stage / 'results'
    for p in (out / 'inputs/results').rglob('*'):
        if p.is_file():
            target = results / p.relative_to(out / 'inputs/results')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(p, target)
    for folder in ['extraction-rasters', 'mining-area', 'biodiversity-risk']:
        (results / folder).mkdir(parents=True, exist_ok=True)
    return results


@functools.lru_cache(None)
def namespace(base, stage='staged'):
    _, out, _ = paths(base)
    sys.path.insert(0, str(out / 'code'))
    ns = dict(np=np, pd=pd, rio=rio, Path=Path, tqdm=tqdm, json=json,
              distance_transform_edt=distance_transform_edt, curve_fit=curve_fit, t=t,
              ROOT=out / stage, RESULTS=out / stage / 'results', DATA=out / 'inputs/data',
              MFA_RESULTS=out / 'inputs/material-flows', BUILDING_RESULTS=out / 'inputs/future-results')
    for name in ['demand', 'allocation', 'area', 'risk']:
        load_cells(out, name, ns)
    original_supplement = ns['economy_supplement']

    @functools.lru_cache(None)
    def cached_supplement(res):
        saved = ns['RESULTS'] / 'economy-rasters' / f'economy-raster-eck4-{res}-supplement.tif'
        if saved.exists():
            with rio.open(saved) as src:
                return np.array([int(c) for c in src.descriptions]), src.read().astype(bool)
        return original_supplement(res)

    # Generate masks once in the coordinator; parallel workers read that copy.
    ns['economy_supplement'] = cached_supplement
    return ns


def load_cells(out, name, ns):
    relative, indices = NOTEBOOKS[name]
    nb = read(out / 'code' / Path(relative).name)
    for i in indices:
        source = ''.join(nb['cells'][i]['source'])
        # Path adaptation only; keep the scientific functions byte-for-byte otherwise.
        source = source.replace("Path(r'F:\\aggregates-mining-biodiversity\\results\\material-flows')", 'MFA_RESULTS')
        source = source.replace("Path(r'F:\\aggregates-mining-biodiversity\\results')", 'BUILDING_RESULTS')
        exec(compile(source, f'{relative}#{i}', 'exec'), ns)


def tag(case):
    s, b, d, r, a = case
    return f'{s}-B{b:.2f}-50km-D{d}-R{r}-A{a}'


def raster(path):
    with rio.open(path) as src:
        value = src.read(1)
    if not np.isfinite(value).all() or value.min() < -1e-8:
        raise ValueError(f'Invalid raster values: {path}')
    return value


def npz_signature(path):
    with zipfile.ZipFile(path) as z:
        return [[v.filename, v.CRC, v.file_size] for v in z.infolist()]


def freeze_mfa(base, scenarios):
    base, out, _ = paths(base)
    economies = read(base / 'run-plan.json')['economies']
    for i, s in enumerate(scenarios, 1):
        mark = out / 'state/mfa-inputs' / f'{s}.json'
        if mark.exists():
            continue
        copied = []
        for economy in economies:
            state = read(base / 'state' / s / f'{economy}.json')
            if state['status'] != 'complete':
                raise ValueError(f'MFA unfinished: {economy}/{s}')
            signatures = {x['file']: x['npz_signature'] for x in state['file_manifest']}
            for name in [f'{m}_extraction.npz' for m in MATERIALS] + ['split_ratio_fine.npz', 'split_ratio_coarse.npz', 'split_ratio_RawMat_Ca.npz']:
                rel = Path(economy) / s / name
                source = FUTURE_RESULTS / 'material-flows' / rel
                if npz_signature(source) != signatures[name]:
                    raise ValueError(f'Installed MFA result differs from accepted output: {rel}')
                dst = out / 'inputs/material-flows' / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dst)
                if npz_signature(dst) != signatures[name]:
                    raise ValueError(f'MFA input copy failed: {rel}')
                copied.append(dict(file=rel.as_posix(), npz_signature=signatures[name]))
        dump(mark, dict(status='complete', files=copied))
        progress(base, 'freezing_mfa_inputs', completed=i, total=len(scenarios))


def demand_task(base, scenario):
    _, out, _ = paths(base)
    mark = out / 'state/demand' / f'{scenario}.json'
    if mark.exists():
        return read(mark)
    ns = namespace(base)
    results = ns['RESULTS']
    checks = []
    with (out / 'logs' / f'demand-{scenario}.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        for mat in MATERIALS:
            for lo, hi in zip([2023, 2026, 2031, 2036, 2041, 2046], [2025, 2030, 2035, 2040, 2045, 2050]):
                ns['distribute_national_extraction_to_pixel'](mat, lo, hi, scenario)
            ns['add_extraction_demand_together'](mat, scenario)
            expected = np.zeros(3)
            for economy in ns['pseudo_codes_economies_142']['economies_142'].unique():
                turnover, growth = ns['spatial_demand_percentiles'](economy, scenario, mat)
                expected += (turnover + growth).sum(axis=1) / 1000  # kt -> Mt
            for k, level in enumerate('LMH'):
                a = raster(results / 'extraction-rasters' / f'{mat}-extraction-demand-2023-2050-{scenario}-50km-{level}.tif')
                observed = float(a.sum(dtype=float)) / 1e6
                if not np.isclose(observed, expected[k], rtol=2e-6, atol=1e-5):
                    raise ValueError(f'Spatial demand not conserved: {scenario}/{mat}/{level}: {observed} vs {expected[k]}')
                checks.append(dict(material=mat, level=level, raster_Mt=observed, expected_component_quantile_sum_Mt=float(expected[k])))
        ns['spatial_demand_percentiles'].cache_clear()
    row = dict(status='complete', scenario=scenario, checks=checks)
    dump(mark, row)
    return row


def allocation_task(base, case, stage='staged'):
    _, out, _ = paths(base)
    marker = out / ('state/allocation' if stage == 'staged' else 'preflight/state') / f'{tag(case)}.json'
    if marker.exists():
        return read(marker)
    ns = namespace(base, stage)
    s, b, d, r, a = case
    logfile = out / 'logs' / f'alloc-{stage}-{tag(case)}.log'
    with logfile.open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        values, summary, matrix, flows = ns['allocate_demand_to_extraction'](s, d, r, beta=b, area_level=a)
    cap = raster(ns['RESULTS'] / 'mining-area' / f'effective-extraction-capacity-50km-R{r}-A{a}.tif')
    if not np.isfinite(values).all() or values.min() < -1e-8 or (values-cap).max() > 1e-6:
        raise ValueError(f'Invalid capacity-constrained allocation: {tag(case)}')
    for arr in [matrix.to_numpy(), summary[['deficit', 'surplus', 'import', 'export']].to_numpy(), flows[['alloc_Mt', 'dist_km']].to_numpy()]:
        if not np.isfinite(arr).all() or arr.min() < -1e-8:
            raise ValueError(f'Invalid trade/transport values: {tag(case)}')
    if not np.isclose(flows.alloc_Mt.sum(), values.sum(), rtol=1e-10, atol=1e-6):
        raise ValueError(f'Transport flow does not match extraction: {tag(case)}')
    demand = sum(raster(ns['RESULTS'] / 'extraction-rasters' / f'{m}-extraction-demand-2023-2050-{s}-50km-{d}.tif') for m in MATERIALS)
    demanded = float(demand.sum(dtype=float)) / 1e6
    extracted = float(values.sum(dtype=float))
    unmet = demanded-extracted
    # Float32 demand assembly permits small rounding differences. Real unmet demand
    # remains visible and stops publication if coverage falls below 99.999%.
    if abs(unmet) > max(.001, demanded * 1e-5):
        raise ValueError(f'Demand not covered by allocation: {tag(case)}: {unmet} Mt')
    row = dict(status='complete', case=list(case), extraction_Mt=extracted, demand_Mt=demanded,
               unmet_Mt=unmet, max_capacity_excess_Mt=float((values-cap).max()))
    dump(marker, row)
    return row


def area_task(base, case, suit, cap, fit, stage='staged'):
    ns = namespace(base, stage)
    s, b, d, r, a = case
    target = ns['RESULTS'] / 'mining-area' / f'mining-area-expansion-2050-{tag(case)}.tif'
    marker = Path(base) / 'downstream' / ('state/area' if stage == 'staged' else 'preflight/area-state') / f'{tag(case)}.json'
    if marker.exists():
        return read(marker)
    ns['estimate_mining_area'](s, b, d, r, a, suit_1km=suit, cap_1km=cap, fit=fit)
    values = raster(target)
    with rio.open(target) as src:
        tags = src.tags()
    maximum = float(tags['max_subcell_area_km2'])
    error = float(tags['max_mass_error_Mt'])
    if maximum > 1 + 1e-9 or error > 1e-5:
        raise ValueError(f'Area/mass check failed: {tag(case)}')
    row = dict(status='complete', case=list(case), area_km2=float(values.sum(dtype=float)), max_subcell_area_km2=maximum, max_mass_error_Mt=error)
    dump(marker, row)
    return row


def risk_task(base, case, stage='staged'):
    ns = namespace(base, stage)
    s, b, d, r, a = case
    marker = Path(base) / 'downstream' / ('state/risk' if stage == 'staged' else 'preflight/risk-state') / f'{tag(case)}.json'
    if marker.exists():
        return read(marker)
    with contextlib.redirect_stdout(sys.stderr):
        overlap, taxa = ns['assess_biodiversity_risk'](s, b, d, r, a)
    area = raster(ns['RESULTS'] / 'mining-area' / f'mining-area-expansion-2050-{tag(case)}.tif')
    rwr = raster(ns['RESULTS'] / 'area-of-habitat-50km/rarity-weighted-richness-vertebrates-50km.tif')
    np.testing.assert_allclose(overlap, area / 2500 * rwr, rtol=1e-12, atol=1e-15)
    if not np.isfinite(taxa.to_numpy()).all() or taxa.to_numpy().min() < -1e-10 or overlap.min() < 0:
        raise ValueError(f'Invalid biodiversity results: {tag(case)}')
    row = dict(status='complete', case=list(case), overlap=float(overlap.sum(dtype=float)))
    dump(marker, row)
    return row


def summarize(base, cases):
    _, out, _ = paths(base)
    ns = namespace(base)
    results = ns['RESULTS']
    cfg = read(out / 'manifest.json')['sweep']
    sums = {tuple(c): read(out / 'state/risk' / f'{tag(c)}.json')['overlap'] for c in cases}
    riskdir = results / 'biodiversity-risk'
    pd.DataFrame([dict(scenario=s, beta=b, overlap=sums[(s,b,'M','M','M')]) for s in cfg['SCENARIOS'] for b in cfg['BETAS']]).to_csv(riskdir / 'biodiversity-risk-summary-DM-RM.csv', index=False)
    for b, labels, name in [(0., [(s,s) for s in BASE_SCENARIOS], 'B0'), (.6, [('BAU','CON'),('CE','IAP')], 'B0.6-signposts'), (.7, [('BAU','CON'),('CE','IAP')], 'B0.7-signposts')]:
        pd.DataFrame([dict(scenario=label,beta=b,demand=d,reserve=r,area=a,overlap=sums[(s,b,d,r,a)]) for s,label in labels for d,r,a in itertools.product('LMH',repeat=3)]).to_csv(riskdir / f'biodiversity-risk-summary-{name}-DRA-factorial.csv', index=False)
    bounds=[]
    for s,b in [('BAU',0.),('BAU',.7),('CE',0.),('CE',.7)]:
        rows=[dict(scenario=s,beta=b,demand=d,reserve=r,area=a,overlap=sums[(s,b,d,r,a)]) for d,r,a in itertools.product('LMH',repeat=3)]
        bounds += [dict(min(rows,key=lambda x:x['overlap']),bound='lower'),dict(max(rows,key=lambda x:x['overlap']),bound='upper')]
    pd.DataFrame(bounds).to_csv(riskdir / 'biodiversity-risk-bounds-signposts.csv', index=False)
    ns.update(FLOWDIR=results/'extraction-rasters', SCENARIOS=cfg['SCENARIOS'], BETAS=cfg['BETAS'],
              DEMAND='M',RESERVE='M',RES='50km',FREIGHT_RATES={'low':.08,'central':.12,'high':.15}, YEARS=28)
    load_cells(out, 'cost', ns)
    cost=ns['cost_df']
    sweep=cost[['scenario','beta','work_Gt_km','mean_dist_km','rwr_exposure','alloc_Gt']].rename(columns={'work_Gt_km':'domestic_work_Gt_km','alloc_Gt':'total_alloc_Gt'})
    sweep.to_csv(results / 'extraction-rasters/tradeoff-sweep-summary-DM-RM.csv', index=False)
    return knee(results)


def knee(results):
    cost=pd.read_csv(results/'extraction-rasters/cost-analysis-DM-RM.csv')
    risk=pd.read_csv(results/'biodiversity-risk/biodiversity-risk-summary-DM-RM.csv')
    frame=cost.merge(risk,on=['scenario','beta'])
    fr=frame[frame.scenario=='BAU'].sort_values('beta')
    red=1-fr.overlap.to_numpy()/fr.overlap.iloc[0]
    cst=fr.cost_Tusd_central.to_numpy()
    if cst[-1]==cst[0] or red[-1]==red[0]:
        raise ValueError('Cannot normalize a constant cost/risk frontier')
    net=(red-red[0])/(red[-1]-red[0])-(cst-cst[0])/(cst[-1]-cst[0])
    beta=fr.beta.to_numpy()
    plateau=beta[net>=net.max()-.001]
    return dict(beta=float(plateau.min()),tolerance=.001,candidates=plateau.tolist(),net_benefit=net.tolist(),betas=beta.tolist())


def batch(base, cases, phase, jobs=8):
    _, out, _ = paths(base)
    sys.path.insert(0,str(out/'code'))
    worker=importlib.import_module('downstream_worker')
    args = {}
    ns=namespace(base)
    if phase=='area':
        suit=rio.open(ns['RESULTS']/'global-mining-suitability-1km.tif').read(1)
        caps={r:rio.open(ns['RESULTS']/'mining-area'/f'aggregate-reserve-1km-{r}.tif').read(1) for r in 'LMH'}
        fit=tuple(ns[k] for k in ['a_1','b_1','popt_1','pcov_1','dof_1'])
    function=getattr(worker,phase+'_task')
    work=(delayed(function)(str(base),c,**(dict(suit=suit,cap=caps[c[3]],fit=fit) if phase=='area' else args)) for c in cases)
    # Joblib adds long memmap filenames; keep the prefix short on Windows.
    temp = Path('D:/caches') / (Path(base).name + '-spatial-jobs')
    temp.mkdir(parents=True, exist_ok=True)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=jobs,temp_folder=str(temp),return_as='generator_unordered')(work)
        for i,result in enumerate(results,1):
            progress(base,phase,completed=i,total=len(cases),last=result.get('case',result.get('scenario')))


def preflight(base):
    base,out,_=paths(base)
    manifest=read(out/'manifest.json')
    results=setup_stage(base,'preflight')
    (out/'logs').mkdir(exist_ok=True)
    for p in (out/'before/results/extraction-rasters').glob('*extraction-demand-2023-2050-BAU-50km-*.tif'):
        shutil.copy2(p,results/'extraction-rasters'/p.name)
    case=('BAU',0.,'M','M','M')
    allocation_task(str(base),case,'preflight')
    ns=namespace(str(base),'preflight')
    suit=rio.open(results/'global-mining-suitability-1km.tif').read(1)
    cap=rio.open(results/'mining-area/aggregate-reserve-1km-M.tif').read(1)
    fit=tuple(ns[k] for k in ['a_1','b_1','popt_1','pcov_1','dof_1'])
    area_task(str(base),case,suit,cap,fit,'preflight')
    risk_task(str(base),case,'preflight')
    checked=[]
    for folder,prefix in [('extraction-rasters','allocated-extraction-2023-2050'),('mining-area','mining-area-expansion-2050'),('biodiversity-risk','mining-biodiversity-overlap-2050')]:
        rel=Path(folder)/f'{prefix}-{tag(case)}.tif'
        actual=raster(results/rel);expected=raster(out/'before/results'/rel)
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-8)
        checked.append(dict(file=str(rel),maximum_absolute_difference=float(np.max(abs(actual-expected)))))
    dump(out/'preflight-validation.json',dict(status='passed',checks=checked))
    print('Preflight reproduces the saved global BAU allocation, area and biodiversity results.',flush=True)


def publish(base):
    _,out,repo=paths(base)
    manifest=read(out/'manifest.json')
    items=[]
    for folder in ['extraction-rasters','mining-area','biodiversity-risk','economy-rasters']:
        for source in (out/'staged/results'/folder).iterdir():
            if not source.is_file():continue
            if folder=='mining-area' and not source.name.startswith('mining-area-expansion-'):continue
            if folder=='economy-rasters' and 'supplement' not in source.name:continue
            rel=source.relative_to(out/'staged/results');target=repo/'results'/rel
            new=sha(source);old=sha(target) if target.exists() else None
            expected=manifest['before'].get(rel.as_posix())
            if folder=='economy-rasters' and old==new:continue
            if old not in (expected,new):
                raise RuntimeError(f'Official result changed concurrently: {target}')
            items.append((source,target,rel,new,expected))
    # Validate every destination before replacing any. Each replacement is atomic;
    # the manifest retains the complete baseline and supports safe resume.
    for i,(source,target,rel,new,old) in enumerate(items,1):
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or sha(target)!=new:
            temp=target.with_name(target.name+'.mfa-downstream.tmp')
            shutil.copy2(source,temp)
            if sha(temp)!=new:raise RuntimeError(f'Copy validation failed: {rel}')
            os.replace(temp,target)
        if i%500==0:progress(base,'installing',completed=i,total=len(items))
    dump(out/'published-manifest.json',[dict(file=rel.as_posix(),before_sha256=old,after_sha256=new) for _,_,rel,new,old in items])


def run(base):
    base,out,_=paths(base)
    manifest=read(out/'manifest.json')
    if (out/'completion.json').exists():return
    (out/'logs').mkdir(exist_ok=True)
    import msvcrt
    lock=(out/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    while True:
        readiness=ssp2_readiness(base)
        if readiness['ready']:
            dump(out/'ssp2-release.json',dict(released=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**readiness))
            break
        upstream=read(base/'progress.json') if (base/'progress.json').exists() else {}
        if upstream.get('status')=='failed':
            progress(base,'waiting_for_ssp2_recovery',completed=readiness['completed'],total=readiness['total'],upstream_error=upstream.get('error'))
        else:
            progress(base,'waiting_for_ssp2',completed=readiness['completed'],total=readiness['total'],upstream_scenario=upstream.get('scenario'))
        time.sleep(30)
    for row in manifest['inputs']:
        if sha(row['snapshot'])!=row['sha256']:raise ValueError(f"Frozen input changed: {row['snapshot']}")
    if read(out/'preflight-validation.json')['status']!='passed':raise ValueError('Preflight not passed')
    freeze_mfa(base,manifest['scenarios'])
    # Missing-economy masks are generated once before parallel demand workers.
    namespace(str(base))['economy_supplement']('50km')
    batch(base,manifest['scenarios'],'demand',jobs=4)
    cases=manifest['cases']
    for phase in ['allocation','area','risk']:batch(base,cases,phase)
    selected=summarize(base,cases)
    old_selected=knee(out/'before/results')
    dump(out/'knee-comparison.json',dict(before=old_selected,after=selected,fixed_comparison_beta=.7))
    # If the same selection rule moves the signpost, compute its full uncertainty
    # ensemble too, while retaining beta=.7 for direct before/after comparisons.
    more=[(s,selected['beta'],d,r,a) for s in ['BAU','CE'] for d,r,a in itertools.product('LMH',repeat=3)
          if (s,selected['beta'],d,r,a) not in {tuple(c) for c in cases}]
    if more:
        for phase in ['allocation','area','risk']:batch(base,more,phase)
        cases=cases+more
    dump(out/'computed-cases.json',cases)
    progress(base,'comparing',cases=len(cases))
    # Reporting uses the working XML/matplotlib environment; spatial calculation
    # uses the prior raster runtime, whose base conda has a broken expat DLL.
    subprocess.run([r'D:\caches\conda-envs\wty\python.exe', str(out/'code/downstream_report.py'), str(base)], check=True)
    progress(base,'installing',cases=len(cases))
    publish(base)
    completed=dict(status='complete',finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                   scenarios=len(manifest['scenarios']),cases=len(cases),selected_beta=selected['beta'],
                   report=str(out/'下游前后结果对比.html'),old_results=str(out/'before/results'))
    dump(out/'completion.json',completed)
    dump(out/'progress.json',completed)
    # The remaining SSP runs and MFA workbook may still be in progress. The
    # combined monitor records full completion once both coordinators finish.
    dump(base/'ssp2-chain-completion.json',dict(status='complete',ssp2=read(out/'ssp2-release.json'),downstream=completed))
    print(json.dumps(completed,ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    command,run_base=sys.argv[1:3]
    try:
        globals()[command](str(Path(run_base).resolve()))
    except Exception:
        if command=='run':
            dump(Path(run_base)/'downstream/progress.json',dict(status='failed',traceback=traceback.format_exc(),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        raise
