"""Physical 1 km² footprint constraint used by allocation and area estimation.

Geological reserves remain unchanged. Effective extraction capacity is the
minimum of those reserves and the inverse of the selected area relationship.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.optimize import brentq, curve_fit
from scipy.stats import t


@lru_cache(None)
def critical_value(dof):
    return t.ppf(0.975, dof)


def area_km2(extraction, popt, pcov, dof, level='M'):
    if level not in ('L', 'M', 'H'):
        raise ValueError('Area level must be L, M or H')
    q = np.asarray(extraction, dtype=np.float64)
    if np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError('Extraction must be finite and non-negative')
    a, b = popt
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        raise ValueError('Expected finite, non-negative area relationship')
    safe = np.where(q > 0, q, 1e-12)
    best = a * safe ** b
    if level != 'M':
        g1 = safe ** b
        g2 = best * np.log(safe)
        variance = pcov[0, 0]*g1*g1 + 2*pcov[0, 1]*g1*g2 + pcov[1, 1]*g2*g2
        best = best + (-1 if level == 'L' else 1)*critical_value(dof)*np.sqrt(np.maximum(variance, 0))
    return np.where(q > 0, np.maximum(best, 0)/100, 0.)


def extraction_limit(popt, pcov, dof, level='M', max_area=1.):
    """First positive crossing of the area limit, validated below the crossing."""
    if max_area <= 0 or not np.isfinite(max_area):
        raise ValueError('Invalid cell area')
    edge = lambda q: float(area_km2(q, popt, pcov, dof, level)) - max_area
    high = 1.
    for _ in range(80):
        if edge(high) >= 0:
            break
        high *= 2
    else:
        raise ValueError('No finite extraction limit for area relationship')
    limit = brentq(edge, 0., high, xtol=1e-12)
    if np.max(area_km2(np.geomspace(1e-12, limit, 10000), popt, pcov, dof, level)) > max_area + 1e-9:
        raise ValueError('Area relationship exceeds cap before inverse threshold')
    return limit


def assign_extraction(total, capacities):
    """Fill already-ranked sub-cells without losing volume or exceeding capacity."""
    capacities = np.asarray(capacities, dtype=np.float64)
    if not np.isfinite(total) or total < 0 or np.any(~np.isfinite(capacities)) or np.any(capacities < 0):
        raise ValueError('Invalid extraction or capacities')
    assigned = np.zeros(capacities.size, dtype=np.float64)
    remaining = float(total)
    for k, cap in enumerate(capacities):
        take = min(remaining, float(cap))
        assigned[k] = take
        remaining -= take
        if remaining <= 0:
            break
    tolerance = max(1e-8, total*1e-10)
    if remaining > tolerance or abs(assigned.sum()-total) > tolerance:
        raise ValueError(f'Insufficient effective capacity: {remaining} Mt unallocated')
    return assigned


def prepare_effective_capacity(results, reserve_level, area_level, res='50km'):
    """Cache area-limited coarse capacity, preserving geological-reserve files."""
    import pandas as pd
    import rasterio as rio
    results = Path(results)
    target = results/'mining-area'/f'effective-extraction-capacity-{res}-R{reserve_level}-A{area_level}.tif'
    if target.exists():
        return target
    rel = pd.read_csv(results/'extraction-area-relationship.csv')
    popt, pcov = curve_fit(lambda x, a, b: a*x**b, rel['E (Mt)'].values, rel['A (ha)'].values)
    limit = extraction_limit(popt, pcov, len(rel)-2, area_level)
    with rio.open(results/'mining-area'/f'aggregate-reserve-1km-{reserve_level}.tif') as fine:
        reserve = fine.read(1)
        fine_transform = fine.transform
    with rio.open(results/'mining-area'/f'aggregate-reserve-{res}-{reserve_level}.tif') as coarse:
        original = coarse.read(1)
        profile = coarse.profile
        transform = coarse.transform
    scale = int(round(abs(transform.a)/1000))
    off_i = int(round((transform.f-fine_transform.f)/1000))
    off_j = int(round((fine_transform.c-transform.c)/1000))
    if off_i < 0 or off_j < 0 or off_j + reserve.shape[1] > original.shape[1]*scale:
        raise ValueError('Unexpected fine/coarse grid alignment')
    capacity = np.zeros(original.shape, dtype=np.float64)
    for i in range(original.shape[0]):
        lo, hi = max(0, i*scale-off_i), min(reserve.shape[0], (i+1)*scale-off_i)
        if hi <= lo:
            continue
        strip = np.zeros((hi-lo, original.shape[1]*scale), dtype=np.float64)
        strip[:, off_j:off_j+reserve.shape[1]] = np.minimum(reserve[lo:hi].astype(np.float64), limit)
        capacity[i] = strip.reshape(hi-lo, original.shape[1], scale).sum(axis=(0, 2))
    capacity = np.minimum(capacity, original)
    if np.any(~np.isfinite(capacity)) or np.any(capacity < 0):
        raise ValueError('Invalid effective capacity')
    profile.update(dtype='float64', count=1, compress='lzw')
    with rio.open(target, 'w', **profile) as dst:
        dst.write(capacity, 1)
        dst.update_tags(area_cap_km2='1', extraction_limit_Mt=str(limit), reserve_level=reserve_level, area_level=area_level)
    return target
