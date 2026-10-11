import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from scipy.special import ndtr
from pathlib import Path


# ============ Configuration ============
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# RESULTS = ROOT / "results"
RESULTS = Path(r"F:\aggregates-mining-biodiversity\results")

# ============ Tool functions ============
def make_constant_values(df, n_sim):
    values = np.array(df).astype(np.float32)  # shape (years,)
    return np.tile(values, (n_sim, 1))

def make_uncertain_values(df, std, span, run, rng):
    values = np.asarray(df, dtype=np.float32).ravel()
    return rng.normal(values, values * std, size=(run, span)).astype(np.float32)

def make_stock_array(array_2D):
    n, s = array_2D.shape
    out = np.empty((n, s, s), dtype=np.float32)
    out[:] = array_2D.astype(np.float32, copy=False)[:, :, None]
    j = np.arange(s)
    out[:, j[:, None] > j[None, :]] = np.nan
    return out

def cal_dem_rate(lft_2D):
    # former "cal_dem_rate_matrix2"
    run, span = lft_2D.shape
    j = np.arange(span, dtype=np.float32)
    invalid = j[:, None] > j[None, :]

    lft_3D = np.empty((run, span, span), dtype=np.float32)
    lft_3D[:] = lft_2D.astype(np.float32, copy=False)[:, :, None]
    lft_3D[:, invalid] = np.nan

    age_2d = (j[None, :] - j[:, None]).astype(np.float32)
    age_3D = np.broadcast_to(age_2d, (run, span, span)).copy()
    age_3D[:, invalid] = np.nan

    # ndtr((age - lft) / (lft * 0.2)) = ndtr(5 * (age/lft - 1))
    z = 5.0 * (age_3D / lft_3D - 1.0)
    ccc = ndtr(z).astype(np.float32)
    sur_rate_3D = (1.0 - ccc).astype(np.float32)

    dem_rate_3D = np.full_like(ccc, np.nan)
    dem_rate_3D[:, :, 1:] = ccc[:, :, 1:] - ccc[:, :, :-1]
    d = np.arange(span)
    dem_rate_3D[:, d, d] = 0.0
    return sur_rate_3D, dem_rate_3D

def cal_dem_rate_conditional(lft_mat, start=1931, mid=2022, end=2050):
    # former "cal_dem_rate_matrix"
    """
    优化点：
    - 不再 np.repeat 物化 lft_3D（节省 ~100MB 分配 + 写回）
    - norm.cdf → ndtr：z = 5*(age/lft - 1)，避开 scipy 包装层
    - 去掉 gg 的 28 次 tile，直接广播 (n_sim, n_hist, 1) 的 denom
    """
    n_sim, n_hist = lft_mat.shape
    span = end - mid + 1

    age_2D = np.array([[j - i for j in range(mid, end + 1)] for i in range(start, mid + 1)],
                      dtype=np.float32)                              # (92, 29)

    # z = (age - lft)/(lft*0.2) = 5*(age/lft - 1)，全程广播
    lft_3D_view = lft_mat[:, :, None]                                # (n_sim, 92, 1)
    z = 5.0 * (age_2D[None, :, :] / lft_3D_view - 1.0)               # (n_sim, 92, 29)
    ddd = ndtr(z).astype(np.float32)

    denom = 1.0 - ddd[:, :, [0]]                                     # (n_sim, 92, 1)，沿年份恒定
    mask = denom == 0
    safe_denom = np.where(mask, 1.0, denom)

    sur_rate = np.where(mask, 0.0, (1.0 - ddd[:, :, 1:]) / safe_denom)
    dem_rate = np.where(mask, 0.0, np.diff(ddd, axis=2) / safe_denom)

    return sur_rate, dem_rate


def fit_logistic_with_constraints(
        hist_values, future_growth,
        start=1931, mid=2022, end=2050,
        r_grid=np.linspace(0.05, 0.8, 100),
        t0_grid=np.linspace(1950, 2040, 120),
        recent_weight=3.0, recent_from=1990):

    def logistic_4p(t, A, K, r, t0):
        return A + (K - A) / (1.0 + np.exp(-r * (t - t0)))

    hist_years = np.arange(start, mid + 1)
    future_years = np.arange(mid + 1, end + 1)
    y_mid = hist_values[-1]
    y_end = y_mid * future_growth

    # --- Non-growth case: keep history as-is, linear bridge to y_end ---
    if future_growth <= 1.0 + 1e-6:
        future_fit = np.linspace(y_mid, y_end, len(future_years) + 1)[1:]
        values = np.concatenate([hist_values, future_fit])
        return pd.Series(values, index=np.arange(start, end + 1))

    # --- Growth case: original logistic fit ---
    def solve_AK_for_endpoints(r, t0):
        L1 = 1.0 / (1.0 + np.exp(-r * (mid - t0)))
        L2 = 1.0 / (1.0 + np.exp(-r * (end - t0)))
        # Closed-form 2x2 solve of M @ [A, K] = [y_mid, y_end] with
        #   M = [[1-L1, L1], [1-L2, L2]]  ->  det = L2 - L1 (caller guarantees |L2-L1|>1e-6).
        # Identical to np.linalg.solve but avoids a LAPACK call that crashes natively
        # (Windows 0xc06d007f) in this env's numpy when run in the main process / kernel.
        det = L2 - L1
        A = (L2 * y_mid - L1 * y_end) / det
        K = ((1.0 - L1) * y_end - (1.0 - L2) * y_mid) / det
        return np.array([A, K], dtype=float)

    w = np.ones_like(hist_years, dtype=float)
    w[hist_years >= recent_from] = recent_weight

    best = {"err": np.inf, "r": None, "t0": None, "A": None, "K": None}
    for r in r_grid:
        for t0 in t0_grid:
            L1 = 1.0 / (1.0 + np.exp(-r * (mid - t0)))
            L2 = 1.0 / (1.0 + np.exp(-r * (end - t0)))
            if abs(L2 - L1) < 1e-6:
                continue
            A, K = solve_AK_for_endpoints(r, t0)
            if not (K > A):
                continue
            y_hat = logistic_4p(hist_years, A, K, r, t0)
            err = np.average((y_hat - hist_values) ** 2, weights=w)
            if err < best["err"]:
                best = {"err": err, "r": r, "t0": t0, "A": A, "K": K}

    years = np.arange(start, end + 1, dtype=float)
    values = logistic_4p(years, best["A"], best["K"], best["r"], best["t0"])
    return pd.Series(values, years)


# ============ load input data（for 142 economies） ============
def load_shared_data():
    region_country = pd.read_excel(DATA / "contents.xlsx", sheet_name="region_division")
    region_country.loc[28, "region"] = "NA"
    region_country.loc[171, "region"] = "NA"

    df_cement_production = pd.read_csv(DATA / "production-142.csv", index_col=0)
    economies_142 = df_cement_production.columns.tolist()

    return {
        "region_country": region_country,
        "df_cement_production": df_cement_production,
        "df_cement_export":    pd.read_csv(RESULTS / "trade-data" / "cement-export-142.csv", index_col=0),
        "df_cement_import": pd.read_csv(RESULTS / 'trade-data' / 'cement-import-142.csv', index_col=0),
        "df_clinker_export": pd.read_csv(RESULTS / 'trade-data' / 'clinker-export-142.csv', index_col=0),
        "df_clinker_import": pd.read_csv(RESULTS / 'trade-data' / 'clinker-import-142.csv', index_col=0),
        "df_cementPro_export": pd.read_csv(RESULTS / 'trade-data' / 'cement-product-export-142.csv', index_col=0),
        "df_cementPro_import": pd.read_csv(RESULTS / 'trade-data' / 'cement-product-import-142.csv', index_col=0),
        "df_limestone_export": pd.read_csv(RESULTS / 'trade-data' / 'limestone-export-142.csv', index_col=0),
        "df_limestone_import": pd.read_csv(RESULTS / 'trade-data' / 'limestone-import-142.csv', index_col=0),
        "df_gravel_export": pd.read_csv(RESULTS / 'trade-data' / 'gravel-crushed-stone-export-142.csv', index_col=0),
        "df_gravel_import": pd.read_csv(RESULTS / 'trade-data' / 'gravel-crushed-stone-import-142.csv', index_col=0),
        "df_sand_export": pd.read_csv(RESULTS / 'trade-data' / 'other-sand-export-142.csv', index_col=0),
        "df_sand_import": pd.read_csv(RESULTS / 'trade-data' / 'other-sand-import-142.csv', index_col=0),

        "df_concrete_ratio": pd.read_excel(DATA / 'params-1.xlsx', sheet_name='concrete_ratio', index_col=0),
        "df_C15_ratio": pd.read_excel(DATA / 'params-1.xlsx', sheet_name='C15_ratio', index_col=0),
        "df_C16_23_ratio": pd.read_excel(DATA / 'params-1.xlsx', sheet_name='C16_23_ratio', index_col=0),
        "df_C24_35_ratio": pd.read_excel(DATA / 'params-1.xlsx', sheet_name='C24_35_ratio', index_col=0),
        "df_C35_ratio": pd.read_excel(DATA / 'params-1.xlsx', sheet_name='C35_ratio', index_col=0),
        "df_cement_content_Pro": pd.read_excel(DATA / 'params-1.xlsx', sheet_name='cement_in_Pro', index_col=0),

        # load GNR params 
        "GNR_CCR": pd.read_excel(DATA / 'GNR-params.xlsx', sheet_name='CCR', index_col=0),
        "GNR_ratio_gypsum": pd.read_excel(DATA / 'GNR-params.xlsx', sheet_name='ratio_gypsum', index_col=0),
        "GNR_ratio_limestone": pd.read_excel(DATA / 'GNR-params.xlsx', sheet_name='ratio_limestone', index_col=0),
        "GNR_ratio_pozzolana": pd.read_excel(DATA / 'GNR-params.xlsx', sheet_name='ratio_pozzolana', index_col=0),
        "GNR_ratio_slag": pd.read_excel(DATA / 'GNR-params.xlsx', sheet_name='ratio_slag', index_col=0),
        "GNR_ratio_flyash": pd.read_excel(DATA / 'GNR-params.xlsx', sheet_name='ratio_flyash', index_col=0),
        "GNR_ratio_others": pd.read_excel(DATA / 'GNR-params.xlsx', sheet_name='ratio_others', index_col=0),
        "GNR_binder_slag": pd.read_excel(DATA / 'GNR-params.xlsx', sheet_name='binder_slag', index_col=0),
        "GNR_binder_FAP": pd.read_excel(DATA / 'GNR-params.xlsx', sheet_name='binder_FAP', index_col=0),

        # sectoral params
        "economies_142": economies_142,
        "df_ratio_RES": pd.read_excel(DATA / 'sector-params.xlsx', sheet_name='ratio_RES', index_col=0)[economies_142],
        "df_ratio_NRES": pd.read_excel(DATA / 'sector-params.xlsx', sheet_name='ratio_NRES', index_col=0)[economies_142],
        "df_ratio_CIV": pd.read_excel(DATA / 'sector-params.xlsx', sheet_name='ratio_CIV', index_col=0)[economies_142],

        "df_lft_RES": pd.read_excel(DATA / 'sector-params.xlsx', sheet_name='lft_RES', index_col=0)[economies_142],
        "df_lft_NRES": pd.read_excel(DATA / 'sector-params.xlsx', sheet_name='lft_NRES', index_col=0)[economies_142],
        "df_lft_CIV": pd.read_excel(DATA / 'sector-params.xlsx', sheet_name='lft_CIV', index_col=0)[economies_142],

        "df_other_params": pd.read_excel(DATA / 'other-params.xlsx', index_col=0)

    }



# ============ main function ============
def run_historical(
    economy: str,
    shared: dict,
    start: int = 1931,
    end: int = 2022,
    n_sim: int = 10_000,
    seed: int = 42,
    std: float = 0.05,
    hist_subdir: str = "historical",
):

    rng = np.random.default_rng(seed)
    span = end - start + 1
    out_dir = RESULTS / "material-flows" / economy / hist_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_data(var, generated_data, out_dir=out_dir):
        np.savez_compressed(out_dir / f'{var}.npz', data=generated_data.astype(np.float32))

    """Run historical MFA simulation for a single economy."""

    for var in [
        'cement_production',
        'cement_export',
        'cement_import',
        'clinker_export',
        'clinker_import',
        'cementPro_export',
        'cementPro_import',
        'limestone_export',
        'limestone_import',
        'gravel_export',
        'gravel_import',
        'sand_export',
        'sand_import',
    ]:
        generated_data = make_constant_values(shared[f"df_{var}"].loc[start:end, economy].values, n_sim)
        save_data(var, generated_data)

    
    CCR_mode = np.where(shared["GNR_CCR"].loc[start:end, economy].values > 0.95, 0.95, shared["GNR_CCR"].loc[start:end, economy]).astype(np.float32)
    CCR = rng.triangular(np.ones(span)*0.5, CCR_mode, np.ones(span)*0.95, size=(n_sim, span)).astype(np.float32)
    save_data('CCR', CCR)

    for var in [
        'ratio_gypsum',
        'ratio_limestone',
        'ratio_pozzolana',
        'ratio_slag',
        'ratio_flyash',
        'ratio_others',
        'binder_slag',
        'binder_FAP'
    ]:
        generated_data = make_uncertain_values(shared[f'GNR_{var}'].loc[start:end, economy], std, span, n_sim, rng)
        save_data(var, generated_data)

    concrete_ratio = make_uncertain_values(shared['df_concrete_ratio'].loc[start:end, economy].values, std, span, n_sim, rng)
    mortar_ratio = 100 - concrete_ratio
    save_data('concrete_ratio', concrete_ratio)
    save_data('mortar_ratio', mortar_ratio)

    r_C15_ = make_uncertain_values(shared['df_C15_ratio'].loc[start:end, economy].values, std, span, n_sim, rng)
    r_C16_23_ = make_uncertain_values(shared['df_C16_23_ratio'].loc[start:end, economy].values, std, span, n_sim, rng)
    r_C24_35_ = make_uncertain_values(shared['df_C24_35_ratio'].loc[start:end, economy].values, std, span, n_sim, rng)
    r_C35_ = make_uncertain_values(shared['df_C35_ratio'].loc[start:end, economy].values, std, span, n_sim, rng)

    r_C15 = r_C15_ / (r_C15_ + r_C16_23_ + r_C24_35_ + r_C35_)
    r_C16_23 = r_C16_23_ / (r_C15_ + r_C16_23_ + r_C24_35_ + r_C35_)
    r_C24_35 = r_C24_35_ / (r_C15_ + r_C16_23_ + r_C24_35_ + r_C35_)
    r_C35 = r_C35_ / (r_C15_ + r_C16_23_ + r_C24_35_ + r_C35_)

    save_data('r_C15', r_C15)
    save_data('r_C16_23', r_C16_23)
    save_data('r_C24_35', r_C24_35)
    save_data('r_C35', r_C35)

    del r_C15_, r_C16_23_, r_C24_35_, r_C35_

    fine_to_binder_C15 = make_uncertain_values(2.542840011, std, span, n_sim, rng)
    fine_to_binder_C16_23 = make_uncertain_values(2.855202053, std, span, n_sim, rng)
    fine_to_binder_C24_35 = make_uncertain_values(2.083908811, std, span, n_sim, rng)
    fine_to_binder_C35 = make_uncertain_values(1.728617340, std, span, n_sim, rng)
    water_to_binder_C15 = make_uncertain_values(0.659834197, std, span, n_sim, rng)
    water_to_binder_C16_23 = make_uncertain_values(0.619416533, std, span, n_sim, rng)
    water_to_binder_C24_35 = make_uncertain_values(0.496178881, std, span, n_sim, rng)
    water_to_binder_C35 = make_uncertain_values(0.395957047, std, span, n_sim, rng)
    coarse_to_binder_C15 = make_uncertain_values(4.182638806, std, span, n_sim, rng)
    coarse_to_binder_C16_23 = make_uncertain_values(3.271740668, std, span, n_sim, rng)
    coarse_to_binder_C24_35 = make_uncertain_values(2.584728036, std, span, n_sim, rng)
    coarse_to_binder_C35 = make_uncertain_values(2.471397967, std, span, n_sim, rng)
    fine_to_binder_mortar = make_uncertain_values(2.720389651, std, span, n_sim, rng)
    water_to_binder_mortar = make_uncertain_values(0.506558494, std, span, n_sim, rng)

    save_data('fine_to_binder_C15', fine_to_binder_C15)
    save_data('fine_to_binder_C16_23', fine_to_binder_C16_23)
    save_data('fine_to_binder_C24_35', fine_to_binder_C24_35)
    save_data('fine_to_binder_C35', fine_to_binder_C35)
    save_data('water_to_binder_C15', water_to_binder_C15)
    save_data('water_to_binder_C16_23', water_to_binder_C16_23)
    save_data('water_to_binder_C24_35', water_to_binder_C24_35)
    save_data('water_to_binder_C35', water_to_binder_C35)
    save_data('coarse_to_binder_C15', coarse_to_binder_C15)
    save_data('coarse_to_binder_C16_23', coarse_to_binder_C16_23)
    save_data('coarse_to_binder_C24_35', coarse_to_binder_C24_35)
    save_data('coarse_to_binder_C35', coarse_to_binder_C35)
    save_data('fine_to_binder_mortar', fine_to_binder_mortar)
    save_data('water_to_binder_mortar', water_to_binder_mortar)

    region_country = shared['region_country']
    aa = region_country[region_country['ISO-3'] == economy]['cement_content_in_products'].values.astype(np.float32)
    cement_content = make_uncertain_values(aa, std, span, n_sim, rng)
    save_data('cement_content', cement_content)

    r_RES_ = make_uncertain_values(shared['df_ratio_RES'].loc[start:end, economy].values, std, span, n_sim, rng)
    r_NRES_ = make_uncertain_values(shared['df_ratio_NRES'].loc[start:end, economy].values, std, span, n_sim, rng)
    r_CIV_ = make_uncertain_values(shared['df_ratio_CIV'].loc[start:end, economy].values, std, span, n_sim, rng)

    r_RES = r_RES_ / (r_RES_ + r_NRES_ + r_CIV_)
    r_NRES = r_NRES_ / (r_RES_ + r_NRES_ + r_CIV_)
    r_CIV = r_CIV_ / (r_RES_ + r_NRES_ + r_CIV_)

    save_data('r_RES', r_RES)
    save_data('r_NRES', r_NRES)
    save_data('r_CIV', r_CIV)

    del r_RES_, r_NRES_, r_CIV_
    del r_RES, r_NRES, r_CIV

    lft_RES = make_uncertain_values(shared['df_lft_RES'].loc[start:end, economy].values, std, span, n_sim, rng)
    lft_NRES = make_uncertain_values(shared['df_lft_NRES'].loc[start:end, economy].values, std, span, n_sim, rng)
    lft_CIV = make_uncertain_values(shared['df_lft_CIV'].loc[start:end, economy].values, std, span, n_sim, rng)

    save_data('lft_RES', lft_RES)
    save_data('lft_NRES', lft_NRES)
    save_data('lft_CIV', lft_CIV)

    del lft_RES, lft_NRES, lft_CIV

    r_waste = rng.triangular(0.01, 0.015, 0.03, size=(n_sim, span)).astype(np.float32)
    save_data('r_waste', r_waste)
    # del r_waste

    r_Ca = make_uncertain_values(1.45402, std, span, n_sim, rng)   # calcareous ratio 
    r_Al = make_uncertain_values(0.09921, std, span, n_sim, rng)    # aluminous ratio 
    r_Fe = make_uncertain_values(0.01798, std, span, n_sim, rng)   # ferrous ratio 
    r_Si = make_uncertain_values(0.09350, std, span, n_sim, rng) # siliceous ratio
    r_CKD = make_uncertain_values(0.06, std, span, n_sim, rng)

    save_data('r_Ca', r_Ca)
    save_data('r_Al', r_Al)
    save_data('r_Fe', r_Fe)
    save_data('r_Si', r_Si)
    save_data('r_CKD', r_CKD)

    del r_Ca, r_Al, r_Fe, r_Si, r_CKD

    cement_production = np.load(out_dir / 'cement_production.npz')['data']
    cement_export = np.load(out_dir / 'cement_export.npz')['data']
    cement_import = np.load(out_dir / 'cement_import.npz')['data']
    binder_slag = np.load(out_dir / 'binder_slag.npz')['data']
    binder_FAP = np.load(out_dir / 'binder_FAP.npz')['data']
    cementPro_export = np.load(out_dir / 'cementPro_export.npz')['data']
    cementPro_import = np.load(out_dir / 'cementPro_import.npz')['data']
    cement_content = np.load(out_dir / 'cement_content.npz')['data']

    #********************    CALCULATION     *************************
    cement_consumption_ = cement_production - cement_export + cement_import
    cement_consumption = np.where(cement_consumption_ >= 0, cement_consumption_, 0)
    cement_sub_slag = cement_consumption * binder_slag
    cement_sub_FAP = cement_consumption * binder_FAP
    all_binder = cement_consumption + cement_sub_slag + cement_sub_FAP  # cement_consumption * (1 + binder_slag + binder_FAP)

    binder_in_exportPro = cementPro_export * cement_content
    binder_in_importPro = cementPro_import * cement_content

    gggg = binder_in_importPro - binder_in_exportPro
    binder_for_concrete = (all_binder * concrete_ratio / 100) + gggg
    binder_for_mortar = all_binder * mortar_ratio / 100

    concrete_waste = (binder_for_concrete * r_waste).astype(np.float32)
    mortar_waste = (binder_for_mortar * r_waste).astype(np.float32)

    save_data('cement_consumption',  cement_consumption)
    save_data('cement_sub_slag',     cement_sub_slag)
    save_data('cement_sub_FAP',      cement_sub_FAP)
    save_data('all_binder',          all_binder)
    save_data('binder_in_exportPro', binder_in_exportPro)
    save_data('binder_in_importPro', binder_in_importPro)
    save_data('binder_for_concrete', binder_for_concrete)
    save_data('binder_for_mortar',   binder_for_mortar)
    save_data('concrete_waste',      concrete_waste)
    save_data('mortar_waste',        mortar_waste)


    strengths = ['C15', 'C16_23', 'C24_35', 'C35']
    apps = ['RES', 'NRES', 'CIV']

    for app in apps:  
        # 1. create create value array for survival and demolition rates 
        lft_app = np.load(out_dir / f'lft_{app}.npz')['data']
        r_app = np.load(out_dir / f'r_{app}.npz')['data']

        sur_rate_app, dem_rate_app = cal_dem_rate(lft_app)

        # 2. allocate inflow for different application 
        binder_for_concrete_app = binder_for_concrete * (1 - r_waste) * r_app
        binder_for_mortar_app = binder_for_mortar * (1 - r_waste) * r_app

        save_data(f'binder_for_concrete_{app}', binder_for_concrete_app)
        save_data(f'binder_for_mortar_{app}',   binder_for_mortar_app)

        # 3. calculate outflow and stock 
        initial_array = make_stock_array(binder_for_concrete_app)
        S_binder_for_concrete_app = initial_array * sur_rate_app
        O_binder_for_concrete_app = initial_array * dem_rate_app
        
        a_shape = S_binder_for_concrete_app.shape
        S_fine_for_concrete_app   = np.zeros(shape=a_shape)
        S_coarse_for_concrete_app = np.zeros(shape=a_shape)
        S_water_for_concrete_app  = np.zeros(shape=a_shape)
        O_fine_for_concrete_app   = np.zeros(shape=a_shape)
        O_coarse_for_concrete_app = np.zeros(shape=a_shape)
        O_water_for_concrete_app  = np.zeros(shape=a_shape)
        S_concrete_app            = np.zeros(shape=a_shape)
        O_concrete_app            = np.zeros(shape=a_shape)

        for strength in tqdm(strengths):
            r_s   = np.load(out_dir / f'r_{strength}.npz')['data'][:, :, None]
            f2b_s = np.load(out_dir / f'fine_to_binder_{strength}.npz')['data'][:, :, None]
            c2b_s = np.load(out_dir / f'coarse_to_binder_{strength}.npz')['data'][:, :, None]
            w2b_s = np.load(out_dir / f'water_to_binder_{strength}.npz')['data'][:, :, None]

            S_binder_for_concrete_app_strength = S_binder_for_concrete_app          * r_s
            S_fine_for_concrete_app_strength   = S_binder_for_concrete_app_strength * f2b_s
            S_coarse_for_concrete_app_strength = S_binder_for_concrete_app_strength * c2b_s
            S_water_for_concrete_app_strength  = S_binder_for_concrete_app_strength * w2b_s
            S_concrete_app_strength = np.sum([S_binder_for_concrete_app_strength, S_fine_for_concrete_app_strength, S_coarse_for_concrete_app_strength, S_water_for_concrete_app_strength], axis=0)

            O_binder_for_concrete_app_strength = O_binder_for_concrete_app          * r_s
            O_fine_for_concrete_app_strength   = O_binder_for_concrete_app_strength * f2b_s
            O_coarse_for_concrete_app_strength = O_binder_for_concrete_app_strength * c2b_s
            O_water_for_concrete_app_strength  = O_binder_for_concrete_app_strength * w2b_s
            O_concrete_app_strength = np.sum([O_binder_for_concrete_app_strength, O_fine_for_concrete_app_strength, O_coarse_for_concrete_app_strength, O_water_for_concrete_app_strength], axis=0)

            save_data(f'fine_for_concrete_{app}_{strength}',   np.diagonal(S_fine_for_concrete_app_strength.T))
            save_data(f'coarse_for_concrete_{app}_{strength}',   np.diagonal(S_coarse_for_concrete_app_strength.T))

            save_data(f'S_binder_for_concrete_{app}_{strength}_annual',  np.nansum(S_binder_for_concrete_app_strength, axis=1))
            save_data(f'S_fine_for_concrete_{app}_{strength}_annual',    np.nansum(S_fine_for_concrete_app_strength,   axis=1))
            save_data(f'S_coarse_for_concrete_{app}_{strength}_annual',  np.nansum(S_coarse_for_concrete_app_strength, axis=1))
            save_data(f'S_water_for_concrete_{app}_{strength}_annual',   np.nansum(S_water_for_concrete_app_strength,  axis=1))
            save_data(f'S_concrete_{app}_{strength}_annual',             np.nansum(S_concrete_app_strength,            axis=1))

            save_data(f'O_binder_for_concrete_{app}_{strength}_annual',  np.nansum(O_binder_for_concrete_app_strength, axis=1))
            save_data(f'O_fine_for_concrete_{app}_{strength}_annual',    np.nansum(O_fine_for_concrete_app_strength,   axis=1))
            save_data(f'O_coarse_for_concrete_{app}_{strength}_annual',  np.nansum(O_coarse_for_concrete_app_strength, axis=1))
            save_data(f'O_water_for_concrete_{app}_{strength}_annual',   np.nansum(O_water_for_concrete_app_strength,  axis=1))
            save_data(f'O_concrete_{app}_{strength}_annual',             np.nansum(O_concrete_app_strength,            axis=1))
            
            S_fine_for_concrete_app   += S_fine_for_concrete_app_strength
            S_coarse_for_concrete_app += S_coarse_for_concrete_app_strength
            S_water_for_concrete_app  += S_water_for_concrete_app_strength
            
            O_binder_for_concrete_app += O_binder_for_concrete_app_strength
            O_fine_for_concrete_app   += O_fine_for_concrete_app_strength
            O_coarse_for_concrete_app += O_coarse_for_concrete_app_strength
            O_water_for_concrete_app  += O_water_for_concrete_app_strength
            
            S_concrete_app            += S_concrete_app_strength
            O_concrete_app            += O_concrete_app_strength

            S_binder_for_concrete_app_strength_cohort = S_binder_for_concrete_app_strength[:, :, -1]
            save_data(f'S_binder_for_concrete_{app}_{strength}_cohort', S_binder_for_concrete_app_strength_cohort)


        save_data(f'concrete_{app}',           np.diagonal(S_concrete_app.T))
        save_data(f'fine_for_concrete_{app}',  np.diagonal(S_fine_for_concrete_app.T))
        save_data(f'coarse_{app}',             np.diagonal(S_coarse_for_concrete_app.T))
        save_data(f'water_for_concrete_{app}', np.diagonal(S_water_for_concrete_app.T))

        save_data(f'S_binder_for_concrete_{app}_annual', np.nansum(S_binder_for_concrete_app, axis=1))
        save_data(f'S_fine_for_concrete_{app}_annual',   np.nansum(S_fine_for_concrete_app,   axis=1))
        save_data(f'S_coarse_for_concrete_{app}_annual', np.nansum(S_coarse_for_concrete_app, axis=1))
        save_data(f'S_water_for_concrete_{app}_annual',  np.nansum(S_water_for_concrete_app,  axis=1))

        save_data(f'O_binder_for_concrete_{app}_annual', np.nansum(O_binder_for_concrete_app, axis=1))
        save_data(f'O_fine_for_concrete_{app}_annual',   np.nansum(O_fine_for_concrete_app,   axis=1))
        save_data(f'O_coarse_for_concrete_{app}_annual', np.nansum(O_coarse_for_concrete_app, axis=1))
        save_data(f'O_water_for_concrete_{app}_annual',  np.nansum(O_water_for_concrete_app,  axis=1))

        save_data(f'S_concrete_{app}_annual', np.nansum(S_concrete_app, axis=1))
        save_data(f'O_concrete_{app}_annual', np.nansum(O_concrete_app, axis=1))

        S_binder_for_concrete_app_cohort = S_binder_for_concrete_app[:, :, -1]
        save_data(f'S_binder_for_concrete_{app}_cohort', S_binder_for_concrete_app_cohort)


        initial_array2 = make_stock_array(binder_for_mortar_app)
        S_binder_for_mortar_app = initial_array2 * sur_rate_app
        O_binder_for_mortar_app = initial_array2 * dem_rate_app

        S_fine_for_mortar_app  = S_binder_for_mortar_app * fine_to_binder_mortar[:, :, None]
        S_water_for_mortar_app = S_binder_for_mortar_app * water_to_binder_mortar[:, :, None]

        O_fine_for_mortar_app  = O_binder_for_mortar_app * fine_to_binder_mortar[:, :, None]
        O_water_for_mortar_app = O_binder_for_mortar_app * water_to_binder_mortar[:, :, None]

        S_mortar_app = np.sum([S_binder_for_mortar_app, S_fine_for_mortar_app, S_water_for_mortar_app], axis=0)
        O_mortar_app = np.sum([O_binder_for_mortar_app, O_fine_for_mortar_app, O_water_for_mortar_app], axis=0)

        save_data(f'mortar_{app}',           np.diagonal(S_mortar_app.T))
        save_data(f'fine_for_mortar_{app}',  np.diagonal(S_fine_for_mortar_app.T))
        save_data(f'water_for_mortar_{app}', np.diagonal(S_water_for_mortar_app.T))

        save_data(f'S_binder_for_mortar_{app}_annual', np.nansum(S_binder_for_mortar_app, axis=1))
        save_data(f'S_fine_for_mortar_{app}_annual',   np.nansum(S_fine_for_mortar_app,   axis=1))
        save_data(f'S_water_for_mortar_{app}_annual',  np.nansum(S_water_for_mortar_app,  axis=1))

        save_data(f'O_binder_for_mortar_{app}_annual', np.nansum(O_binder_for_mortar_app, axis=1))
        save_data(f'O_fine_for_mortar_{app}_annual',   np.nansum(O_fine_for_mortar_app,   axis=1))
        save_data(f'O_water_for_mortar_{app}_annual',  np.nansum(O_water_for_mortar_app,  axis=1))

        save_data(f'S_mortar_{app}_annual', np.nansum(S_mortar_app, axis=1))
        save_data(f'O_mortar_{app}_annual', np.nansum(O_mortar_app, axis=1))

        S_binder_for_mortar_app_cohort = S_binder_for_mortar_app[:, :, -1]
        save_data(f'S_binder_for_mortar_{app}_cohort', S_binder_for_mortar_app_cohort)

    
    for strength in strengths:
        fine_for_concrete_RES_strength  = np.load(out_dir / f'fine_for_concrete_RES_{strength}.npz')['data']
        fine_for_concrete_NRES_strength = np.load(out_dir / f'fine_for_concrete_NRES_{strength}.npz')['data']
        fine_for_concrete_CIV_strength  = np.load(out_dir / f'fine_for_concrete_CIV_{strength}.npz')['data']
        fine_for_concrete_strength      = fine_for_concrete_RES_strength + fine_for_concrete_NRES_strength + fine_for_concrete_CIV_strength
        save_data(f'fine_for_concrete_{strength}', fine_for_concrete_strength)

    S_binder_for_concrete_annual = np.zeros(shape=(n_sim, span))
    S_fine_for_concrete_annual   = np.zeros(shape=(n_sim, span))
    S_coarse_for_concrete_annual = np.zeros(shape=(n_sim, span))
    S_water_for_concrete_annual  = np.zeros(shape=(n_sim, span))
    S_concrete_annual            = np.zeros(shape=(n_sim, span))

    S_binder_for_mortar_annual   = np.zeros(shape=(n_sim, span))
    S_fine_for_mortar_annual     = np.zeros(shape=(n_sim, span))
    S_water_for_mortar_annual    = np.zeros(shape=(n_sim, span))
    S_mortar_annual              = np.zeros(shape=(n_sim, span))

    O_binder_for_concrete_annual = np.zeros(shape=(n_sim, span))
    O_fine_for_concrete_annual   = np.zeros(shape=(n_sim, span))
    O_coarse_for_concrete_annual = np.zeros(shape=(n_sim, span))
    O_water_for_concrete_annual  = np.zeros(shape=(n_sim, span))
    O_concrete_annual            = np.zeros(shape=(n_sim, span))

    O_binder_for_mortar_annual   = np.zeros(shape=(n_sim, span))
    O_fine_for_mortar_annual     = np.zeros(shape=(n_sim, span))
    O_water_for_mortar_annual    = np.zeros(shape=(n_sim, span))
    O_mortar_annual              = np.zeros(shape=(n_sim, span))

    for app in apps:
        S_binder_for_concrete_app_annual = np.load(out_dir / f'S_binder_for_concrete_{app}_annual.npz', )['data']
        S_fine_for_concrete_app_annual   = np.load(out_dir / f'S_fine_for_concrete_{app}_annual.npz', )['data']
        S_coarse_for_concrete_app_annual = np.load(out_dir / f'S_coarse_for_concrete_{app}_annual.npz', )['data']
        S_water_for_concrete_app_annual  = np.load(out_dir / f'S_water_for_concrete_{app}_annual.npz', )['data']
        S_concrete_app_annual            = np.load(out_dir / f'S_concrete_{app}_annual.npz', )['data']

        S_binder_for_concrete_annual += S_binder_for_concrete_app_annual
        S_fine_for_concrete_annual   += S_fine_for_concrete_app_annual
        S_coarse_for_concrete_annual += S_coarse_for_concrete_app_annual
        S_water_for_concrete_annual  += S_water_for_concrete_app_annual
        S_concrete_annual            += S_concrete_app_annual

        S_binder_for_mortar_app_annual = np.load(out_dir / f'S_binder_for_mortar_{app}_annual.npz', )['data']
        S_fine_for_mortar_app_annual   = np.load(out_dir / f'S_fine_for_mortar_{app}_annual.npz', )['data']
        S_water_for_mortar_app_annual  = np.load(out_dir / f'S_water_for_mortar_{app}_annual.npz', )['data']
        S_mortar_app_annual            = np.load(out_dir / f'S_mortar_{app}_annual.npz', )['data']

        S_binder_for_mortar_annual += S_binder_for_mortar_app_annual
        S_fine_for_mortar_annual   += S_fine_for_mortar_app_annual
        S_water_for_mortar_annual  += S_water_for_mortar_app_annual
        S_mortar_annual            += S_mortar_app_annual

        O_binder_for_concrete_app_annual = np.load(out_dir / f'O_binder_for_concrete_{app}_annual.npz', )['data']
        O_fine_for_concrete_app_annual   = np.load(out_dir / f'O_fine_for_concrete_{app}_annual.npz', )['data']
        O_coarse_for_concrete_app_annual = np.load(out_dir / f'O_coarse_for_concrete_{app}_annual.npz', )['data']
        O_water_for_concrete_app_annual  = np.load(out_dir / f'O_water_for_concrete_{app}_annual.npz', )['data']
        O_concrete_app_annual            = np.load(out_dir / f'O_concrete_{app}_annual.npz', )['data']

        O_binder_for_concrete_annual += O_binder_for_concrete_app_annual
        O_fine_for_concrete_annual   += O_fine_for_concrete_app_annual
        O_coarse_for_concrete_annual += O_coarse_for_concrete_app_annual
        O_water_for_concrete_annual  += O_water_for_concrete_app_annual
        O_concrete_annual            += O_concrete_app_annual

        O_binder_for_mortar_app_annual = np.load(out_dir / f'O_binder_for_mortar_{app}_annual.npz', )['data']
        O_fine_for_mortar_app_annual   = np.load(out_dir / f'O_fine_for_mortar_{app}_annual.npz', )['data']
        O_water_for_mortar_app_annual  = np.load(out_dir / f'O_water_for_mortar_{app}_annual.npz', )['data']
        O_mortar_app_annual            = np.load(out_dir / f'O_mortar_{app}_annual.npz', )['data']

        O_binder_for_mortar_annual += O_binder_for_mortar_app_annual
        O_fine_for_mortar_annual   += O_fine_for_mortar_app_annual
        O_water_for_mortar_annual  += O_water_for_mortar_app_annual
        O_mortar_annual            += O_mortar_app_annual
    
    save_data('S_binder_for_concrete_annual', S_binder_for_concrete_annual)
    save_data('S_fine_for_concrete_annual',   S_fine_for_concrete_annual)
    save_data('S_coarse_for_concrete_annual', S_coarse_for_concrete_annual)
    save_data('S_water_for_concrete_annual',  S_water_for_concrete_annual)
    save_data('S_concrete_annual',            S_concrete_annual)

    save_data('S_binder_for_mortar_annual', S_binder_for_mortar_annual)
    save_data('S_fine_for_mortar_annual',   S_fine_for_mortar_annual)
    save_data('S_water_for_mortar_annual',  S_water_for_mortar_annual)
    save_data('S_mortar_annual',            S_mortar_annual)

    save_data('O_binder_for_concrete_annual', O_binder_for_concrete_annual)
    save_data('O_fine_for_concrete_annual',   O_fine_for_concrete_annual)
    save_data('O_coarse_for_concrete_annual', O_coarse_for_concrete_annual)
    save_data('O_water_for_concrete_annual',  O_water_for_concrete_annual)
    save_data('O_concrete_annual',            O_concrete_annual)

    save_data('O_binder_for_mortar_annual', O_binder_for_mortar_annual)
    save_data('O_fine_for_mortar_annual',   O_fine_for_mortar_annual)
    save_data('O_water_for_mortar_annual',  O_water_for_mortar_annual)
    save_data('O_mortar_annual',            O_mortar_annual)

    S_binder_RES_annual = np.zeros(shape=(n_sim, span))
    S_binder_NRES_annual = np.zeros(shape=(n_sim, span))
    S_binder_CIV_annual = np.zeros(shape=(n_sim, span))
    S_fine_RES_annual = np.zeros(shape=(n_sim, span))
    S_fine_NRES_annual = np.zeros(shape=(n_sim, span))
    S_fine_CIV_annual = np.zeros(shape=(n_sim, span))

    S_binder_annual = np.zeros(shape=(n_sim, span))
    S_fine_annual = np.zeros(shape=(n_sim, span))
    S_water_annual = np.zeros(shape=(n_sim, span))

    S_RES_annual = np.zeros(shape=(n_sim, span))
    S_NRES_annual = np.zeros(shape=(n_sim, span))
    S_CIV_annual = np.zeros(shape=(n_sim, span))
    S_Total_annual = np.zeros(shape=(n_sim, span))

    O_binder_annual = np.zeros(shape=(n_sim, span))
    O_fine_annual = np.zeros(shape=(n_sim, span))
    O_water_annual = np.zeros(shape=(n_sim, span))

    O_RES_annual = np.zeros(shape=(n_sim, span))
    O_NRES_annual = np.zeros(shape=(n_sim, span))
    O_CIV_annual = np.zeros(shape=(n_sim, span))
    O_Total_annual = np.zeros(shape=(n_sim, span))

    for material in ['concrete', 'mortar']:
        S_binder_for_material_RES_annual = np.load(out_dir / f'S_binder_for_{material}_RES_annual.npz')['data']
        S_binder_for_material_NRES_annual = np.load(out_dir / f'S_binder_for_{material}_NRES_annual.npz')['data']
        S_binder_for_material_CIV_annual = np.load(out_dir / f'S_binder_for_{material}_CIV_annual.npz')['data']

        S_fine_for_material_RES_annual = np.load(out_dir / f'S_fine_for_{material}_RES_annual.npz')['data']
        S_fine_for_material_NRES_annual = np.load(out_dir / f'S_fine_for_{material}_NRES_annual.npz')['data']
        S_fine_for_material_CIV_annual = np.load(out_dir / f'S_fine_for_{material}_CIV_annual.npz')['data']

        S_binder_for_material_annual = np.load(out_dir / f'S_binder_for_{material}_annual.npz')['data']
        S_fine_for_material_annual = np.load(out_dir / f'S_fine_for_{material}_annual.npz')['data']
        S_water_for_material_annual = np.load(out_dir / f'S_water_for_{material}_annual.npz')['data']

        S_material_RES_annual = np.load(out_dir / f'S_{material}_RES_annual.npz')['data']
        S_material_NRES_annual = np.load(out_dir / f'S_{material}_NRES_annual.npz')['data']
        S_material_CIV_annual = np.load(out_dir / f'S_{material}_CIV_annual.npz')['data']
        S_material_annual = np.load(out_dir / f'S_{material}_annual.npz')['data']

        O_binder_for_material_annual = np.load(out_dir / f'O_binder_for_{material}_annual.npz')['data']
        O_fine_for_material_annual = np.load(out_dir / f'O_fine_for_{material}_annual.npz')['data']
        O_water_for_material_annual = np.load(out_dir / f'O_water_for_{material}_annual.npz')['data']

        O_material_RES_annual = np.load(out_dir / f'O_{material}_RES_annual.npz')['data']
        O_material_NRES_annual = np.load(out_dir / f'O_{material}_NRES_annual.npz')['data']
        O_material_CIV_annual = np.load(out_dir / f'O_{material}_CIV_annual.npz')['data']
        O_material_annual = np.load(out_dir / f'O_{material}_annual.npz')['data']

        S_binder_RES_annual += S_binder_for_material_RES_annual
        S_binder_NRES_annual += S_binder_for_material_NRES_annual
        S_binder_CIV_annual += S_binder_for_material_CIV_annual

        S_fine_RES_annual += S_fine_for_material_RES_annual
        S_fine_NRES_annual += S_fine_for_material_NRES_annual
        S_fine_CIV_annual += S_fine_for_material_CIV_annual

        S_binder_annual += S_binder_for_material_annual
        S_fine_annual += S_fine_for_material_annual
        S_water_annual += S_water_for_material_annual

        S_RES_annual += S_material_RES_annual
        S_NRES_annual += S_material_NRES_annual
        S_CIV_annual += S_material_CIV_annual
        
        S_Total_annual += S_material_annual

        O_binder_annual += O_binder_for_material_annual
        O_fine_annual += O_fine_for_material_annual
        O_water_annual += O_water_for_material_annual

        O_RES_annual += O_material_RES_annual
        O_NRES_annual += O_material_NRES_annual
        O_CIV_annual += O_material_CIV_annual

        O_Total_annual += O_material_annual


    S_coarse_RES_annual = np.load(out_dir / 'S_coarse_for_concrete_RES_annual.npz')['data']
    S_coarse_NRES_annual = np.load(out_dir / 'S_coarse_for_concrete_NRES_annual.npz')['data']
    S_coarse_CIV_annual = np.load(out_dir / 'S_coarse_for_concrete_CIV_annual.npz')['data']

    S_coarse_annual = np.load(out_dir / 'S_coarse_for_concrete_annual.npz')['data']
    O_coarse_annual = np.load(out_dir / 'O_coarse_for_concrete_annual.npz')['data']

    save_data('S_binder_RES_annual', S_binder_RES_annual)
    save_data('S_binder_NRES_annual', S_binder_NRES_annual)
    save_data('S_binder_CIV_annual', S_binder_CIV_annual)

    save_data('S_fine_RES_annual', S_fine_RES_annual)
    save_data('S_fine_NRES_annual', S_fine_NRES_annual)
    save_data('S_fine_CIV_annual', S_fine_CIV_annual)

    save_data('S_coarse_RES_annual', S_coarse_RES_annual)
    save_data('S_coarse_NRES_annual', S_coarse_NRES_annual)
    save_data('S_coarse_CIV_annual', S_coarse_CIV_annual)

    save_data('S_binder_annual', S_binder_annual)
    save_data('S_fine_annual', S_fine_annual)
    save_data('S_coarse_annual', S_coarse_annual)
    save_data('S_water_annual', S_water_annual)
    save_data('S_RES_annual', S_RES_annual)
    save_data('S_NRES_annual', S_NRES_annual)
    save_data('S_CIV_annual', S_CIV_annual)
    save_data('S_Total_annual', S_Total_annual)

    save_data('O_binder_annual', O_binder_annual)
    save_data('O_fine_annual', O_fine_annual)
    save_data('O_coarse_annual', O_coarse_annual)
    save_data('O_water_annual', O_water_annual)
    save_data('O_RES_annual', O_RES_annual)
    save_data('O_NRES_annual', O_NRES_annual)
    save_data('O_CIV_annual', O_CIV_annual)
    save_data('O_Total_annual', O_Total_annual)

    fine_total = np.zeros(shape=(n_sim, span))
    coarse_total = np.zeros(shape=(n_sim, span))
    water_total = np.zeros(shape=(n_sim, span))
    concrete = np.zeros(shape=(n_sim, span))
    mortar = np.zeros(shape=(n_sim, span))

    for app in apps:
        fine_for_concrete_app = np.load(out_dir / f'fine_for_concrete_{app}.npz')['data']
        fine_for_mortar_app = np.load(out_dir / f'fine_for_mortar_{app}.npz')['data']
        water_for_concrete_app = np.load(out_dir / f'water_for_concrete_{app}.npz')['data']
        water_for_mortar_app = np.load(out_dir / f'water_for_mortar_{app}.npz')['data']

        fine_app = fine_for_concrete_app + fine_for_mortar_app
        water_app = water_for_concrete_app + water_for_mortar_app
        coarse_app = np.load(out_dir / f'coarse_{app}.npz')['data']

        fine_total += fine_app
        coarse_total += coarse_app
        water_total += water_app

        concrete += np.load(out_dir / f'concrete_{app}.npz')['data']
        mortar += np.load(out_dir / f'mortar_{app}.npz')['data']

        save_data(f'fine_{app}', fine_app)
    

    save_data('fine_total',   fine_total)
    save_data('coarse_total', coarse_total)
    save_data('water_total',  water_total)
    save_data('concrete',     concrete)
    save_data('mortar',       mortar)

    sand_export = np.load(out_dir / 'sand_export.npz')['data']
    sand_import = np.load(out_dir / 'sand_import.npz')['data']
    gravel_export = np.load(out_dir / 'gravel_export.npz')['data']
    gravel_import = np.load(out_dir / 'gravel_import.npz')['data']

    sand_production_ = fine_total + sand_export - sand_import
    gravel_production_ = coarse_total + gravel_export - gravel_import
    excess_sand = np.where(sand_production_ < 0, -sand_production_, 0)
    excess_gravel = np.where(gravel_production_ < 0, -gravel_production_, 0)
    sand_stock = excess_sand.cumsum(axis=1)
    gravel_stock = excess_gravel.cumsum(axis=1)

    consumed_sand_stock = np.where((sand_stock > 0) & (sand_production_ > 0), np.where(sand_production_ <= sand_stock, sand_production_, sand_stock), 0)
    consumed_gravel_stock = np.where((gravel_stock > 0) & (gravel_production_ > 0), np.where(gravel_production_ <= gravel_stock, gravel_production_, gravel_stock), 0)

    sand_stock = sand_stock - consumed_sand_stock
    gravel_stock = gravel_stock - consumed_gravel_stock

    sand_production = np.where(sand_production_ >= consumed_sand_stock, sand_production_ - consumed_sand_stock, 0)
    gravel_production = np.where(gravel_production_ >= consumed_gravel_stock, gravel_production_ - consumed_gravel_stock, 0)

    EoLWaste = O_Total_annual

    region = region_country[region_country['ISO-3'] == economy]['region'].values[0]
    aa = shared['df_other_params'].loc['rec_rate', region]
    bb = shared['df_other_params'].loc['down_rate', region]
    rec_rate = make_uncertain_values([aa], std, span, n_sim, rng)
    down_rate = make_uncertain_values([bb], std, span, n_sim, rng)

    EoLWaste_recycled = EoLWaste * rec_rate
    EoLWaste_downcycled = EoLWaste * down_rate 
    EoLWaste_disposed = EoLWaste * (1 - rec_rate - down_rate)

    aggregate_recycled_ = EoLWaste_recycled 
    # aggregate_virgin = fine_total + (coarse_total - aggregate_recycled)

    sand_substitution_rate = 0
    sand_substitution = sand_production * sand_substitution_rate

    aggregate_recycled = np.where(aggregate_recycled_ > gravel_production, gravel_production, aggregate_recycled_)
    gravel_extraction = gravel_production - aggregate_recycled 

    sand_extraction = sand_production - sand_substitution
    aggregate_virgin = gravel_extraction + sand_extraction 

    ratio_gypsum = np.load(out_dir / 'ratio_gypsum.npz')['data']
    ratio_limestone = np.load(out_dir / 'ratio_limestone.npz')['data']
    ratio_pozzolana = np.load(out_dir / 'ratio_pozzolana.npz')['data']
    ratio_slag = np.load(out_dir / 'ratio_slag.npz')['data']
    ratio_flyash = np.load(out_dir / 'ratio_flyash.npz')['data']
    ratio_others = np.load(out_dir / 'ratio_others.npz')['data']

    SCM_gypsum = cement_production * ratio_gypsum
    SCM_limestone = cement_production * ratio_limestone
    SCM_pozzolana = cement_production * ratio_pozzolana
    SCM_slag = cement_production * ratio_slag 
    SCM_flyash = cement_production * ratio_flyash
    SCM_others = cement_production * ratio_others 

    clinker_consumption = cement_production * CCR

    clinker_export = np.load(out_dir / 'clinker_export.npz')['data']
    clinker_import = np.load(out_dir / 'clinker_import.npz')['data']
    limestone_export = np.load(out_dir / 'limestone_export.npz')['data']
    limestone_import = np.load(out_dir / 'limestone_import.npz')['data']

    clinker_production_ = clinker_consumption + clinker_export - clinker_import
    excess_clinker = np.where(clinker_production_ < 0, -clinker_production_, 0)
    gg = excess_clinker.cumsum(axis=1)
    clinker_stock = gg.copy()
    clinker_stock[:, 3:] = gg[:, 3:] - gg[:, :-3]
    clinker_production = np.where(clinker_production_ >= clinker_stock, clinker_production_ - clinker_stock, 0)

    r_CKD = np.load(out_dir / 'r_CKD.npz')['data']
    r_Ca = np.load(out_dir / 'r_Ca.npz')['data']
    r_Al = np.load(out_dir / 'r_Al.npz')['data']
    r_Fe = np.load(out_dir / 'r_Fe.npz')['data']
    r_Si = np.load(out_dir / 'r_Si.npz')['data']

    CKD = clinker_production * r_CKD  

    RawMat_Ca = clinker_production * r_Ca 
    RawMat_Al = clinker_production * r_Al 
    RawMat_Fe = clinker_production * r_Fe 
    RawMat_Si = clinker_production * r_Si 

    limestone_ = RawMat_Ca + limestone_export - limestone_import
    excess_limestone = np.where(limestone_ < 0, -limestone_, 0)

    limestone_stock = excess_limestone.cumsum(axis=1)
    consumed_limestone_stock = np.where((limestone_stock > 0) & (limestone_ > 0), np.where(limestone_ <= limestone_stock, limestone_, limestone_stock), 0)
    limestone_stock = limestone_stock - consumed_limestone_stock
    limestone_extraction = np.where(limestone_ >= consumed_limestone_stock, limestone_ - consumed_limestone_stock, 0)

    save_data('aggregate_virgin',   aggregate_virgin)
    save_data('aggregate_recycled', aggregate_recycled)

    save_data('excess_sand',   excess_sand)
    save_data('excess_gravel', excess_gravel)

    save_data('sand_stock',   sand_stock)
    save_data('gravel_stock', gravel_stock)

    save_data('sand_production',   sand_production)
    save_data('gravel_production', gravel_production)

    save_data('sand_extraction',   sand_extraction)
    save_data('gravel_extraction', gravel_extraction)

    save_data('rec_rate',  rec_rate)
    save_data('down_rate', down_rate)

    save_data('EoLWaste',            EoLWaste)
    save_data('EoLWaste_recycled',   EoLWaste_recycled)
    save_data('EoLWaste_downcycled', EoLWaste_downcycled)
    save_data('EoLWaste_disposed',   EoLWaste_disposed)

    save_data('SCM_gypsum',    SCM_gypsum)
    save_data('SCM_limestone', SCM_limestone)
    save_data('SCM_pozzolana', SCM_pozzolana)
    save_data('SCM_slag',      SCM_slag)
    save_data('SCM_flyash',    SCM_flyash)
    save_data('SCM_others',    SCM_others)

    save_data('clinker_consumption', clinker_consumption)
    save_data('clinker_production',  clinker_production)
    save_data('excess_clinker',      excess_clinker)
    save_data('clinker_stock',       clinker_stock)

    save_data('CKD', CKD)

    save_data('RawMat_Ca', RawMat_Ca)
    save_data('RawMat_Al', RawMat_Al)
    save_data('RawMat_Fe', RawMat_Fe)
    save_data('RawMat_Si', RawMat_Si)

    save_data('excess_limestone',     excess_limestone)
    save_data('limestone_stock',      limestone_stock)
    save_data('limestone_extraction', limestone_extraction)
    
    return None


#-------------- future --------------------------
def process_a_flow(
        economy,
        material, 
        app, 
        strength, 
        sur_rate, 
        dem_rate, 
        fut_sur_rate, 
        fut_dem_rate,
        future_stock_trajectory,
        hist_dir,
        output_dir, 
        n_sim=10000,
        seed=42
        ):
    
    def save_data(var, generated_data, out_dir=output_dir):
        np.savez_compressed(out_dir / f'{var}.npz', data=generated_data.astype(np.float32))

    mid, end = 2022, 2050
    span = end - mid
    std = 0.05
    rng = np.random.default_rng(seed)

    coarse_to_binder_params = {'C15': 4.182638806, 'C16_23': 3.271740668, 'C24_35': 2.584728036, 'C35': 2.471397967}
    fine_to_binder_params   = {'C15': 2.542840011, 'C16_23': 2.855202053, 'C24_35': 2.083908811, 'C35': 1.728617340}

    if material == 'concrete':
        stock_cohort     = np.load(hist_dir / f'S_binder_for_concrete_{app}_{strength}_cohort.npz')['data']
        his_stock_annual = np.load(hist_dir / f'S_binder_for_concrete_{app}_{strength}_annual.npz')['data']

        alpha = coarse_to_binder_params[strength]
        beta  = fine_to_binder_params[strength]

        coarse_to_binder = make_uncertain_values(alpha, std, span, n_sim, rng)
        fine_to_binder   = make_uncertain_values(beta,  std, span, n_sim, rng)

    else:
        stock_cohort     = np.load(hist_dir / f'S_binder_for_mortar_{app}_cohort.npz')['data']
        his_stock_annual = np.load(hist_dir / f'S_binder_for_mortar_{app}_annual.npz')['data']
        fine_to_binder   = make_uncertain_values(2.720389651, std, span, n_sim, rng)

    fut_his_stock_array_ = np.array([stock_cohort for i in range(span)]).transpose(1, 2, 0)
    fut_his_stock_array = fut_his_stock_array_ * sur_rate
    fut_his_outflow_array = fut_his_stock_array_ * dem_rate
    fut_his_stock_annual = np.nansum(fut_his_stock_array, axis=1)
    fut_his_outflow_annual = np.nansum(fut_his_outflow_array, axis=1)

    current_stock = his_stock_annual[:, -1]
    future_stock_annual = current_stock[:, None] * future_stock_trajectory[None, :]

    the_gap = future_stock_annual[:, 1:] - fut_his_stock_annual

    fut_fut_stock_array = np.full(shape=(n_sim, span, span), fill_value=np.nan).astype(np.float32)
    fut_fut_outflow_array = np.full(shape=(n_sim, span, span), fill_value=np.nan).astype(np.float32)
    future_inflows = np.empty((span, n_sim), dtype=np.float32)
    col_sums = np.zeros((n_sim, span), dtype=np.float32)

    for t in range(span):
        a_inflow = the_gap[:, t] - col_sums[:, t]
        row_stock   = a_inflow[:, None] * fut_sur_rate[:, t, :]
        row_outflow = a_inflow[:, None] * fut_dem_rate[:, t, :]
        fut_fut_stock_array[:, t, :]   = row_stock
        fut_fut_outflow_array[:, t, :] = row_outflow
        future_inflows[t] = a_inflow
        col_sums += np.nan_to_num(row_stock, nan=0.0)
    
    fut_fut_stock_annual   = np.nansum(fut_fut_stock_array, axis=1)
    fut_fut_outflow_annual = np.nansum(fut_fut_outflow_array, axis=1)

    future_inflows_annual  = future_inflows.T
    future_outflows_annual = fut_his_outflow_annual + fut_fut_outflow_annual

    future_fine_inflow  = future_inflows_annual * fine_to_binder
    future_fine_outflow = future_outflows_annual * fine_to_binder

    prefix = f'{material}_{app}_{strength}' if material == 'concrete' else f'{material}_{app}'

    save_data(f'S_binder_for_{prefix}_his_annual', fut_his_stock_annual)
    save_data(f'O_binder_for_{prefix}_his_annual', fut_his_outflow_annual)
    save_data(f'S_binder_for_{prefix}_annual', future_stock_annual)
    save_data(f'I_binder_for_{prefix}_annual', future_inflows_annual)
    save_data(f'O_binder_for_{prefix}_annual', future_outflows_annual)
    save_data(f'I_fine_for_{prefix}_annual', future_fine_inflow)
    save_data(f'O_fine_for_{prefix}_annual', future_fine_outflow)

    if material == 'concrete':
        futuer_coarse_inflow  = future_inflows_annual  * coarse_to_binder
        future_coarse_outflow = future_outflows_annual * coarse_to_binder
        save_data(f'I_coarse_for_{prefix}_annual', futuer_coarse_inflow)
        save_data(f'O_coarse_for_{prefix}_annual', future_coarse_outflow)
    

variable_list = [
    'I_binder_annual',
    'I_fine_annual',
    'I_coarse_annual',
    'O_binder_annual',
    'O_fine_annual',
    'O_coarse_annual',
    'reuse_factors',
    'recycle_factors',
    'sand_sub_factors',
    'fut_CCR',
    'I_binder_for_concrete_annual_ru',
    'I_binder_for_mortar_annual_ru',
    'I_binder_for_concrete_annual_v',
    'I_binder_for_mortar_annual_v',
    'I_binder_annual_ru',
    'I_binder_annual_v',
    'p_binder_slag',
    'p_binder_FAP',
    'p_binder_in_exportPro',
    'p_binder_in_importPro',
    'p_cement_export',
    'p_cement_import',
    'p_clinker_export',
    'p_clinker_import',
    # 'p_sand_export',
    # 'p_sand_import',
    # 'p_gravel_export',
    # 'p_gravel_import',
    # 'p_limestone_export',
    # 'p_limestone_import',
    # 'gravel_export',
    # 'gravel_import',
    # 'sand_export',
    # 'sand_import',
    # 'excess_gravel',
    # 'excess_sand',
    # 'gravel_stock',
    # 'sand_stock',
    # 'consumed_gravel_stock',
    # 'consumed_sand_stock',
    'gravel_production',
    'sand_production',
    'reused_and_recycled_gravel',
    'p_reused_gravel',
    'p_recycled_gravel',
    'reused_gravel',
    'recycled_gravel',
    'gravel_extraction',
    'reused_and_recycled_sand',
    'p_reused_sand',
    'p_recycled_sand',
    'reused_sand',
    'recycled_sand',
    'substituted_sand',
    'sand_extraction',
    'all_binder',
    'cement_consumption',
    'cement_production',
    'clinker_consumption',
    'excess_clinker',
    'clinker_stock',
    'clinker_production',
    'r_Ca',
    'RawMat_Ca',
    # 'limestone_export',
    # 'limestone_import',
    # 'excess_limestone',
    # 'limestone_stock',
    # 'consumed_limestone_stock',
    # 'limestone_stock',
    'limestone_extraction',
    'S_coarse_annual',
    'S_fine_annual',
    'S_binder_annual',
    'split_ratio_coarse',
    'split_ratio_fine',
    'split_ratio_RawMat_Ca',
    'gravel_extraction_for_stock_growth',
    'gravel_extraction_for_stock_turnover',
    'sand_extraction_for_stock_growth',
    'sand_extraction_for_stock_turnover',
    'limestone_extraction_for_stock_growth',
    'limestone_extraction_for_stock_turnover'
]


def run_projection(
    scenario,
    economy,
    more_intensive_use=0,
    lifetime_extension=0,
    reuse_rate=0,
    recycle_rate=0,
    sand_sub=0,
    oresand_supply=np.inf,
    clinker_target=None,
    end=2050,
    mid=2022,
    ssp='SSP2',
    n_sim=10000,
    seed=42,
    ambition=1.0,    # global CE-ambition multiplier phi in [0,1]: 0 -> BAU, 1 -> full CE target
    hist_subdir='historical',
):

    hist_dir   = RESULTS / 'material-flows' / economy / hist_subdir
    output_dir = RESULTS / 'material-flows' / economy / scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def save_data(var, generated_data, out_dir=output_dir):
        np.savez_compressed(out_dir / f'{var}.npz', data=generated_data.astype(np.float32))

    stock_growth_of_economies_142 = pd.read_excel(RESULTS / 'future-stock-growth.xlsx', sheet_name=ssp, index_col=0)
    # if_developed = region_country[region_country['ISO-3'] == economy]['developed'].values[0]
    # std = 0.05 if if_developed == 1 else 0.1
    std = 0.05
    
    # --- global CE-ambition multiplier (phi in [0,1]): scale every lever's 2050 target ---
    # 0 -> BAU (no measures), 1 -> full CE target. Backward-compatible (default ambition=1.0).
    # clinker is an absolute k-factor, so it is scaled inside its own ramp (fut_CCR) below.
    more_intensive_use = ambition * more_intensive_use
    lifetime_extension = ambition * lifetime_extension
    reuse_rate         = ambition * reuse_rate
    recycle_rate       = ambition * recycle_rate
    sand_sub           = ambition * sand_sub

    lft_extension = 1 + lifetime_extension
    span = end - mid

    S_binder_annual = np.load(hist_dir / 'S_binder_annual.npz')['data']

    historical_stock_trajectory = np.median(S_binder_annual, axis=0) / np.median(S_binder_annual, axis=0)[-1]
    K = stock_growth_of_economies_142.loc[end, economy]
    fitted_stock_trajectory = fit_logistic_with_constraints(historical_stock_trajectory, K)
    future_stock_trajectory_ = fitted_stock_trajectory.loc[mid:end].values
    fut_stock_factor = np.linspace(1, 1 - more_intensive_use, span+1)
    future_stock_trajectory = future_stock_trajectory_ * fut_stock_factor

    for app in ['RES', 'NRES', 'CIV']:
        lft = np.load(hist_dir / f'lft_{app}.npz')['data']
        fut_lft = np.array([lft[:, -1] * lft_extension for i in range(span)]).T
        sur_rate, dem_rate = cal_dem_rate_conditional(lft)
        fut_sur_rate, fut_dem_rate = cal_dem_rate(fut_lft)

        for strength in ['C15', 'C16_23', 'C24_35', 'C35']:
            process_a_flow(
                economy,
                'concrete', 
                app, 
                strength,
                sur_rate,
                dem_rate,
                fut_sur_rate,
                fut_dem_rate,
                future_stock_trajectory,
                hist_dir,
                output_dir,
                n_sim=n_sim,
                seed=seed)

        binder_stock = np.nansum([np.load(output_dir / f'S_binder_for_concrete_{app}_{strength}_annual.npz')['data'] for strength in ['C15', 'C16_23', 'C24_35', 'C35']], axis=0)
        binder_inflow = np.nansum([np.load(output_dir / f'I_binder_for_concrete_{app}_{strength}_annual.npz')['data'] for strength in ['C15', 'C16_23', 'C24_35', 'C35']], axis=0)
        binder_outflow = np.nansum([np.load(output_dir / f'O_binder_for_concrete_{app}_{strength}_annual.npz')['data'] for strength in ['C15', 'C16_23', 'C24_35', 'C35']], axis=0)
        coarse_inflow = np.nansum([np.load(output_dir / f'I_coarse_for_concrete_{app}_{strength}_annual.npz')['data'] for strength in ['C15', 'C16_23', 'C24_35', 'C35']], axis=0)
        coarse_outflow = np.nansum([np.load(output_dir / f'O_coarse_for_concrete_{app}_{strength}_annual.npz')['data'] for strength in ['C15', 'C16_23', 'C24_35', 'C35']], axis=0)
        fine_inflow = np.nansum([np.load(output_dir / f'I_fine_for_concrete_{app}_{strength}_annual.npz')['data'] for strength in ['C15', 'C16_23', 'C24_35', 'C35']], axis=0)
        fine_outflow = np.nansum([np.load(output_dir / f'O_fine_for_concrete_{app}_{strength}_annual.npz')['data'] for strength in ['C15', 'C16_23', 'C24_35', 'C35']], axis=0)

        save_data(f'S_binder_for_concrete_{app}_annual', binder_stock)
        save_data(f'I_binder_for_concrete_{app}_annual', binder_inflow)
        save_data(f'O_binder_for_concrete_{app}_annual', binder_outflow)
        save_data(f'I_coarse_for_concrete_{app}_annual', coarse_inflow)
        save_data(f'O_coarse_for_concrete_{app}_annual', coarse_outflow)
        save_data(f'I_fine_for_concrete_{app}_annual',   fine_inflow)
        save_data(f'O_fine_for_concrete_{app}_annual',   fine_outflow)

        process_a_flow(
            economy,
            'mortar', 
            app,
            '',
            sur_rate,
            dem_rate,
            fut_sur_rate,
            fut_dem_rate,
            future_stock_trajectory,
            hist_dir,
            output_dir,
            n_sim=n_sim,
            seed=seed
            )

    for material in ['concrete', 'mortar']:
        binder_stock = np.nansum([np.load(output_dir / f'S_binder_for_{material}_{app}_annual.npz')['data'] for app in ['RES', 'NRES', 'CIV']], axis=0).astype(np.float32)
        binder_inflow = np.nansum([np.load(output_dir / f'I_binder_for_{material}_{app}_annual.npz')['data'] for app in ['RES', 'NRES', 'CIV']], axis=0).astype(np.float32)
        binder_outflow = np.nansum([np.load(output_dir / f'O_binder_for_{material}_{app}_annual.npz')['data'] for app in ['RES', 'NRES', 'CIV']], axis=0).astype(np.float32)
        fine_inflow = np.nansum([np.load(output_dir / f'I_fine_for_{material}_{app}_annual.npz')['data'] for app in ['RES', 'NRES', 'CIV']], axis=0).astype(np.float32)
        fine_outflow = np.nansum([np.load(output_dir / f'O_fine_for_{material}_{app}_annual.npz')['data'] for app in ['RES', 'NRES', 'CIV']], axis=0).astype(np.float32)

        save_data(f'S_binder_for_{material}_annual', binder_stock)
        save_data(f'I_binder_for_{material}_annual', binder_inflow)
        save_data(f'O_binder_for_{material}_annual', binder_outflow)
        save_data(f'I_fine_for_{material}_annual',   fine_inflow)
        save_data(f'O_fine_for_{material}_annual',   fine_outflow)

        if material == 'concrete':
            coarse_inflow  = np.nansum([np.load(output_dir / f'I_coarse_for_{material}_{app}_annual.npz')['data'] for app in ['RES', 'NRES', 'CIV']], axis=0).astype(np.float32)
            coarse_outflow = np.nansum([np.load(output_dir / f'O_coarse_for_{material}_{app}_annual.npz')['data'] for app in ['RES', 'NRES', 'CIV']], axis=0).astype(np.float32)

            save_data(f'I_coarse_for_{material}_annual', coarse_inflow)
            save_data(f'O_coarse_for_{material}_annual', coarse_outflow)
    
    I_binder_for_concrete_annual = np.load(output_dir / 'I_binder_for_concrete_annual.npz')['data']
    I_binder_for_mortar_annual =   np.load(output_dir / 'I_binder_for_mortar_annual.npz')['data']
    I_binder_annual = I_binder_for_concrete_annual + I_binder_for_mortar_annual
    I_fine_annual = np.load(output_dir / 'I_fine_for_concrete_annual.npz')['data'] + np.load(output_dir / 'I_fine_for_mortar_annual.npz')['data']
    I_coarse_annual = np.load(output_dir / 'I_coarse_for_concrete_annual.npz')['data']
    O_binder_annual = np.load(output_dir / 'O_binder_for_concrete_annual.npz')['data'] + np.load(output_dir / 'O_binder_for_mortar_annual.npz')['data']
    O_fine_annual = np.load(output_dir / 'O_fine_for_concrete_annual.npz')['data'] + np.load(output_dir / 'O_fine_for_mortar_annual.npz')['data']
    O_coarse_annual = np.load(output_dir / 'O_coarse_for_concrete_annual.npz')['data']

    reuse_factors = np.tile(np.linspace(0, reuse_rate, span+1)[1:], (n_sim, 1))
    recycle_factors = np.tile(np.linspace(0, recycle_rate, span+1)[1:], (n_sim, 1))
    sand_sub_factors = np.tile(np.linspace(0, sand_sub, span+1)[1:], (n_sim, 1))

    his_CCR = np.load(hist_dir / 'CCR.npz')['data']
    base_CCR = his_CCR[:, -1][:, None]                                   # 2022 clinker-to-cement ratio (per sim)
    if clinker_target is None:
        fut_CCR = np.repeat(base_CCR, span, axis=1)                      # no clinker change (BAU): flat at base
    else:
        ccr_ramp = np.linspace(0, 1, span+1)[1:][None, :]                # 0 -> 1 over 2023..2050
        # ramp base toward the absolute target, reduction only (never increase a base already below target)
        fut_CCR = base_CCR - ambition * np.maximum(base_CCR - clinker_target, 0.0) * ccr_ramp

    I_binder_for_concrete_annual_ru = np.load(output_dir / 'O_binder_for_concrete_annual.npz')['data'] * reuse_factors 
    I_binder_for_mortar_annual_ru = np.load(output_dir / 'O_binder_for_mortar_annual.npz')['data'] * reuse_factors 
    I_binder_for_concrete_annual_v = I_binder_for_concrete_annual - I_binder_for_concrete_annual_ru 
    I_binder_for_mortar_annual_v = I_binder_for_mortar_annual - I_binder_for_mortar_annual_ru
    I_binder_annual_ru = I_binder_for_concrete_annual_ru + I_binder_for_mortar_annual_ru
    I_binder_annual_v =  I_binder_for_concrete_annual_v + I_binder_for_mortar_annual_v

    p_binder_slag = np.load(hist_dir / 'binder_slag.npz')['data'][:, -1]
    p_binder_FAP  = np.load(hist_dir / 'binder_FAP.npz')['data'][:, -1]

    p_binder_in_exportPro   = np.load(hist_dir / 'binder_in_exportPro.npz')['data'][:, -1] / np.load(hist_dir / 'binder_for_concrete.npz')['data'][:, -1]
    p_binder_in_importPro   = np.load(hist_dir / 'binder_in_importPro.npz')['data'][:, -1] / np.load(hist_dir / 'binder_for_concrete.npz')['data'][:, -1]

    his_cement_consumption  = np.load(hist_dir / 'cement_consumption.npz')['data']
    his_cement_export       = np.load(hist_dir / 'cement_export.npz')['data']
    his_cement_import       = np.load(hist_dir / 'cement_import.npz')['data']

    his_clinker_consumption = np.load(hist_dir / 'clinker_consumption.npz')['data']
    his_clinker_export      = np.load(hist_dir / 'clinker_export.npz')['data']
    his_clinker_import      = np.load(hist_dir / 'clinker_import.npz')['data']

    # his_fine_total  = np.load(hist_dir / 'fine_total.npz')['data']
    # his_sand_export = np.load(hist_dir / 'sand_export.npz')['data']
    # his_sand_import = np.load(hist_dir / 'sand_import.npz')['data']

    # his_coarse_total  = np.load(hist_dir / 'coarse_total.npz')['data']
    # his_gravel_export = np.load(hist_dir / 'gravel_export.npz')['data']
    # his_gravel_import = np.load(hist_dir / 'gravel_import.npz')['data']

    # his_RawMat_Ca        = np.load(hist_dir / 'RawMat_Ca.npz')['data']
    # his_limestone_export = np.load(hist_dir / 'limestone_export.npz')['data']
    # his_limestone_import = np.load(hist_dir / 'limestone_import.npz')['data']

    p_cement_export = np.divide(his_cement_export[:, -1], his_cement_consumption[:, -1], out=np.zeros_like(his_cement_export[:, -1], dtype=np.float32), where=his_cement_consumption[:, -1] != 0)
    p_cement_import = np.divide(his_cement_import[:, -1], his_cement_consumption[:, -1], out=np.zeros_like(his_cement_import[:, -1], dtype=np.float32), where=his_cement_consumption[:, -1] != 0)

    p_clinker_export = np.divide(his_clinker_export[:, -1], his_clinker_consumption[:, -1], out=np.zeros_like(his_clinker_export[:, -1], dtype=np.float32), where=his_clinker_consumption[:, -1] != 0)
    p_clinker_import = np.divide(his_clinker_import[:, -1], his_clinker_consumption[:, -1], out=np.zeros_like(his_clinker_import[:, -1], dtype=np.float32), where=his_clinker_consumption[:, -1] != 0)

    # p_sand_export = np.divide(his_sand_export[:, -1], his_fine_total[:, -1], out=np.zeros_like(his_sand_export[:, -1], dtype=np.float32), where=his_fine_total[:, -1] != 0)
    # p_sand_import = np.divide(his_sand_import[:, -1], his_fine_total[:, -1], out=np.zeros_like(his_sand_import[:, -1], dtype=np.float32), where=his_fine_total[:, -1] != 0)

    # p_gravel_export = np.divide(his_gravel_export[:, -1], his_coarse_total[:, -1], out=np.zeros_like(his_gravel_export[:, -1], dtype=np.float32), where=his_coarse_total[:, -1] != 0)
    # p_gravel_import = np.divide(his_gravel_import[:, -1], his_coarse_total[:, -1], out=np.zeros_like(his_gravel_import[:, -1], dtype=np.float32), where=his_coarse_total[:, -1] != 0)

    # p_limestone_export = np.divide(his_limestone_export[:, -1], his_RawMat_Ca[:, -1], out=np.zeros_like(his_limestone_export[:, -1], dtype=np.float32), where=his_RawMat_Ca[:, -1] != 0)
    # p_limestone_import = np.divide(his_limestone_import[:, -1], his_RawMat_Ca[:, -1], out=np.zeros_like(his_limestone_import[:, -1], dtype=np.float32), where=his_RawMat_Ca[:, -1] != 0)

    # gravel_production_ = I_coarse_annual / (1 + p_gravel_export - p_gravel_import)[:, None]
    # sand_production_ = I_fine_annual / (1 + p_sand_export - p_sand_import)[:, None]

    # gravel_export = I_coarse_annual * p_gravel_export[:, None]
    # gravel_import = I_coarse_annual * p_gravel_import[:, None]

    # sand_export = I_fine_annual * p_sand_export[:, None]
    # sand_import = I_fine_annual * p_sand_import[:, None]

    # excess_gravel = np.where(gravel_production_ < 0 , -gravel_production_, 0)
    # excess_sand = np.where(sand_production_ < 0 , -sand_production_, 0)

    # his_gravel_stock = np.load(hist_dir / 'gravel_stock.npz')['data']
    # his_sand_stock = np.load(hist_dir / 'sand_stock.npz')['data']
    # his_limestone_stock = np.load(hist_dir / 'limestone_stock.npz')['data']

    # gravel_stock = excess_gravel.cumsum(axis=1) + his_gravel_stock[:, -1][:, None]
    # sand_stock = excess_sand.cumsum(axis=1) + his_sand_stock[:, -1][:, None]

    # consumed_gravel_stock = np.where((gravel_stock > 0) & (gravel_production_ > 0), np.where(gravel_production_ <= gravel_stock, gravel_production_, gravel_stock), 0)
    # consumed_sand_stock = np.where((sand_stock > 0) & (sand_production_ > 0), np.where(sand_production_ <= sand_stock, sand_production_, sand_stock), 0)

    # gravel_production = np.where(gravel_production_ >= consumed_gravel_stock, gravel_production_ - consumed_gravel_stock, 0)
    # sand_production = np.where(sand_production_ >= consumed_sand_stock, sand_production_ - consumed_sand_stock, 0)

    gravel_production = I_coarse_annual
    sand_production   = I_fine_annual

    reused_gravel_ = O_coarse_annual * reuse_factors
    recycled_gravel_ = O_coarse_annual * recycle_factors
    reused_and_recycled_gravel_ = reused_gravel_ + recycled_gravel_

    reused_and_recycled_gravel = np.where(gravel_production >= reused_and_recycled_gravel_, reused_and_recycled_gravel_, gravel_production)

    p_reused_gravel = np.divide(reused_gravel_, reused_and_recycled_gravel_, out=np.zeros_like(reused_gravel_, dtype=np.float32), where=reused_and_recycled_gravel_ != 0)
    p_recycled_gravel = np.divide(recycled_gravel_, reused_and_recycled_gravel_, out=np.zeros_like(recycled_gravel_, dtype=np.float32), where=reused_and_recycled_gravel_ != 0)
    reused_gravel = reused_and_recycled_gravel * p_reused_gravel
    recycled_gravel = reused_and_recycled_gravel * p_recycled_gravel
    gravel_extraction = gravel_production - reused_and_recycled_gravel

    reused_sand_ = O_fine_annual * reuse_factors
    recycled_sand_ = O_fine_annual * recycle_factors
    reused_and_recycled_sand_ = reused_sand_ + recycled_sand_ 
    reused_and_recycled_sand = np.where(sand_production >= reused_and_recycled_sand_, reused_and_recycled_sand_, sand_production)

    p_reused_sand = np.divide(reused_sand_, reused_and_recycled_sand_, out=np.zeros_like(reused_sand_, dtype=np.float32), where=reused_and_recycled_sand_ != 0)
    p_recycled_sand = np.divide(recycled_sand_, reused_and_recycled_sand_, out=np.zeros_like(recycled_sand_, dtype=np.float32), where=reused_and_recycled_sand_ != 0)
    reused_sand = reused_and_recycled_sand * p_reused_sand
    recycled_sand = reused_and_recycled_sand * p_recycled_sand
    substituted_sand = np.minimum(
        (sand_production - reused_and_recycled_sand) * sand_sub_factors,
        oresand_supply)                                                  # constant annual ore-sand supply ceiling (kt/yr)
    sand_extraction = sand_production - (reused_and_recycled_sand + substituted_sand)

    all_binder = I_binder_for_concrete_annual_v * (1 - p_binder_in_importPro + p_binder_in_exportPro)[:, None] + I_binder_for_mortar_annual_v
    cement_consumption = all_binder / (1 + p_binder_slag + p_binder_FAP)[:, None]
    cement_production = cement_consumption / (1 - p_cement_export + p_cement_import)[:, None]
    clinker_consumption = cement_production * fut_CCR
    clinker_production_ = clinker_consumption / (1 - p_clinker_export + p_clinker_import)[:, None]
    excess_clinker = np.where(clinker_production_ < 0, -clinker_production_, 0)
    hh = excess_clinker.cumsum(axis=0)
    clinker_stock = hh.copy()
    clinker_stock[3:] = hh[3:] - hh[:-3]
    clinker_production = np.where(clinker_production_ >= clinker_stock, clinker_production_ - clinker_stock, 0)

    # r_Ca = np.vstack([data1['r_Ca'], data2['r_Ca']])[:, -1]
    r_Ca = make_uncertain_values(1.45402, std, span, n_sim, rng)
    RawMat_Ca = clinker_production * r_Ca

    # limestone_ = RawMat_Ca / (1 - p_limestone_export + p_limestone_import)[:, None]
    # limestone_export = RawMat_Ca * p_limestone_export[:, None]
    # limestone_import = RawMat_Ca * p_limestone_import[:, None]
    # excess_limestone = np.where(limestone_ < 0, -limestone_, 0)
    # limestone_stock = excess_limestone.cumsum(axis=1) + his_limestone_stock[:, -1][:, None]
    # consumed_limestone_stock = np.where((limestone_stock > 0) & (limestone_ > 0), np.where(limestone_ <= limestone_stock, limestone_, limestone_stock), 0)
    # limestone_stock = limestone_stock - consumed_limestone_stock
    # limestone_extraction = np.where(limestone_ >= consumed_limestone_stock, limestone_ - consumed_limestone_stock, 0)

    limestone_extraction = RawMat_Ca

    S_coarse_annual = (np.load(hist_dir / 'S_coarse_annual.npz')['data'][:, -1] * future_stock_trajectory[:, None]).T
    S_fine_annual   = (np.load(hist_dir / 'S_fine_annual.npz')['data'][:, -1] * future_stock_trajectory[:, None]).T
    S_binder_annual = (np.load(hist_dir / 'S_binder_annual.npz')['data'][:, -1] * future_stock_trajectory[:, None]).T

    S_coarse_growth = np.diff(S_coarse_annual, axis=1)
    S_fine_growth = np.diff(S_fine_annual, axis=1)
    S_binder_growth = np.diff(S_binder_annual, axis=1)

    split_ratio_coarse = O_coarse_annual / I_coarse_annual
    split_ratio_fine = O_fine_annual / I_fine_annual
    split_ratio_RawMat_Ca = O_binder_annual / I_binder_annual

    gravel_extraction_for_stock_growth = (1 - split_ratio_coarse) * gravel_extraction
    sand_extraction_for_stock_growth = (1 - split_ratio_fine) * sand_extraction
    limestone_extraction_for_stock_growth = (1 - split_ratio_RawMat_Ca) * limestone_extraction

    gravel_extraction_for_stock_turnover = split_ratio_coarse * gravel_extraction
    sand_extraction_for_stock_turnover = split_ratio_fine * sand_extraction
    limestone_extraction_for_stock_turnover = split_ratio_RawMat_Ca * limestone_extraction
    
    local_vars = locals()
    for var in variable_list:
        an_array = local_vars[var].astype(np.float32)
        save_data(var, an_array)