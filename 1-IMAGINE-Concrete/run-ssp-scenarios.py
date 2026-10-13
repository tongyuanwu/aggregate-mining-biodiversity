"""Run BAU extraction projections for the non-SSP2 stock pathways.

The production BAU results were generated with ``run_projection``'s default
SSP2 pathway.  This runner generates the otherwise identical BAU case for
SSP1, SSP3, SSP4, and SSP5.  Each economy is written to its own scenario
folder (for example ``BAU_SSP1``), then pruned immediately to the three
extraction arrays required for plotting and uncertainty aggregation.

Run with the ``wty`` environment so NumPy/SciPy use the working project stack::

    conda run -n wty python 1-IMAGINE-Concrete/run-ssp-scenarios.py --smoke --only SSP1
    conda run -n wty python 1-IMAGINE-Concrete/run-ssp-scenarios.py

The full command is resumable: an economy is skipped only when all three
retained arrays exist, have the expected shape and dtype, and no extra NPZ
files remain in that scenario folder.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


sys.path.insert(0, str(Path(__file__).resolve().parent))

from IMAGINE_Concrete import DATA, RESULTS, load_shared_data, run_projection


N_SIM = 10_000
SEED = 42
HIST_SUBDIR = "historical"
EXPECTED_SHAPE = (N_SIM, 28)  # 2023-2050
DEFAULT_SSPS = ("SSP1", "SSP3", "SSP4", "SSP5")
KEEP = {
    "gravel_extraction.npz",
    "sand_extraction.npz",
    "limestone_extraction.npz",
}


def scenario_name(ssp):
    return f"BAU_{ssp}"


def output_dir(economy, ssp):
    return RESULTS / "material-flows" / economy / scenario_name(ssp)


def required_outputs_are_valid(directory):
    """Check the three retained arrays independently of intermediate files."""
    if not directory.is_dir():
        return False
    try:
        for filename in KEEP:
            with np.load(directory / filename) as archive:
                values = archive["data"]
                if values.shape != EXPECTED_SHAPE or values.dtype != np.float32:
                    return False
                if not np.isfinite(values).all() or np.any(values < 0):
                    return False
    except (OSError, KeyError, ValueError):
        return False

    return True


def retained_outputs_are_complete(directory):
    """Return True only for a valid, fully pruned extraction result folder."""
    if {path.name for path in directory.glob("*.npz")} != KEEP:
        return False
    return required_outputs_are_valid(directory)


def run_and_prune(ssp, economy):
    """Run one economy's BAU projection and retain only extraction arrays."""
    directory = output_dir(economy, ssp)
    if retained_outputs_are_complete(directory):
        return f"SKIP {ssp} {economy}"

    resumed_after_interruption = required_outputs_are_valid(directory)
    if not resumed_after_interruption:
        run_projection(
            scenario_name(ssp),
            economy,
            more_intensive_use=0.0,
            lifetime_extension=0.0,
            reuse_rate=0.0,
            recycle_rate=0.0,
            sand_sub=0.0,
            oresand_supply=np.inf,
            clinker_target=None,
            ssp=ssp,
            n_sim=N_SIM,
            seed=SEED,
            hist_subdir=HIST_SUBDIR,
        )

        missing = KEEP.difference(path.name for path in directory.glob("*.npz"))
        if missing:
            raise RuntimeError(
                f"{ssp} {economy} finished without required outputs: {sorted(missing)}"
            )

    # run_projection needs its intermediate NPZ files while it is running.
    # Prune only after it returns and all three required outputs are present.
    for path in directory.glob("*.npz"):
        if path.name not in KEEP:
            path.unlink()

    if not retained_outputs_are_complete(directory):
        raise RuntimeError(f"Post-prune validation failed for {ssp} {economy}")

    if resumed_after_interruption:
        return f"DONE {ssp} {economy} (pruned existing outputs)"
    return f"DONE {ssp} {economy}"


def parse_ssps(only):
    if not only:
        return list(DEFAULT_SSPS)

    requested = [item.strip().upper() for item in only.split(",") if item.strip()]
    invalid = sorted(set(requested).difference(DEFAULT_SSPS))
    if invalid:
        allowed = ", ".join(DEFAULT_SSPS)
        raise ValueError(f"Unsupported SSP(s): {invalid}. Allowed values: {allowed}")

    # Preserve the user's order while avoiding accidental duplicate work.
    return list(dict.fromkeys(requested))


def validate_inputs(ssps, economies):
    workbook = RESULTS / "future-stock-growth.xlsx"
    sheets = set(pd.ExcelFile(workbook).sheet_names)
    missing_sheets = sorted(set(ssps).difference(sheets))
    if missing_sheets:
        raise FileNotFoundError(
            f"Missing SSP sheet(s) in {workbook}: {missing_sheets}"
        )

    missing_history = [
        economy
        for economy in economies
        if not (
            RESULTS
            / "material-flows"
            / economy
            / HIST_SUBDIR
            / "S_binder_annual.npz"
        ).exists()
    ]
    if missing_history:
        raise FileNotFoundError(
            f"Missing historical inputs for {len(missing_history)} economies: "
            f"{missing_history[:10]}"
        )


def run_ssps(ssps, economies, n_jobs):
    for ssp in ssps:
        print(
            f">>> {scenario_name(ssp)}: {len(economies)} economies, "
            f"n_jobs={n_jobs}, n_sim={N_SIM:,}",
            flush=True,
        )
        messages = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
            delayed(run_and_prune)(ssp, economy) for economy in economies
        )
        completed = sum(message.startswith("DONE") for message in messages)
        skipped = sum(message.startswith("SKIP") for message in messages)
        print(
            f"<<< {scenario_name(ssp)} complete: "
            f"computed={completed}, resumed/skipped={skipped}",
            flush=True,
        )


def smoke(ssp, economy):
    print(f">>> smoke test: {ssp} {economy}", flush=True)
    print(run_and_prune(ssp, economy), flush=True)

    directory = output_dir(economy, ssp)
    for filename in sorted(KEEP):
        with np.load(directory / filename) as archive:
            values = archive["data"]
            medians = np.median(values, axis=0)
            print(
                f"  {filename:26s} shape={values.shape} dtype={values.dtype} "
                f"median[:3]={medians[:3]}",
                flush=True,
            )
    print(f"  retained files: {sorted(path.name for path in directory.glob('*.npz'))}")
    print("SMOKE OK", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run non-SSP2 BAU projections and retain extraction arrays only."
    )
    parser.add_argument("--smoke", action="store_true", help="run one economy only")
    parser.add_argument("--economy", default="CHN", help="economy used by --smoke")
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated SSP names, e.g. SSP1,SSP3 (default: SSP1,3,4,5)",
    )
    args = parser.parse_args()

    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least 1")

    try:
        ssps = parse_ssps(args.only)
    except ValueError as error:
        parser.error(str(error))

    shared = load_shared_data()
    economies = shared["df_cement_production"].columns.tolist()
    if args.economy not in economies:
        parser.error(f"unknown economy for --smoke: {args.economy}")

    selected_economies = [args.economy] if args.smoke else economies
    validate_inputs(ssps, selected_economies)

    free_gib = shutil.disk_usage(RESULTS).free / 2**30
    print(f"results root: {RESULTS}", flush=True)
    print(f"free disk space: {free_gib:.1f} GiB", flush=True)
    print(f"SSPs: {ssps}", flush=True)

    if args.smoke:
        smoke(ssps[0], args.economy)
    else:
        run_ssps(ssps, economies, args.n_jobs)
        print(
            f"DONE: {len(ssps)} BAU SSP scenarios, {len(economies)} economies, "
            "pruned to extraction only",
            flush=True,
        )


if __name__ == "__main__":
    main()
