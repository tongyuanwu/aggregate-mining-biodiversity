"""Resume a complete MFA rerun using the project's economy-parallel approach.

The run directory contains an immutable model/input snapshot. Each economy is
computed on the SSD, checked, compared, and only then installed in RESULTS.
Changed original files are preserved on the same volume before atomic replacement.
No downstream area/biodiversity calculations are invoked by this runner.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from zipfile import ZipFile

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed, parallel_config

OFFICIAL = Path(r"F:\aggregates-mining-biodiversity\results")
MATERIALS = ("sand", "gravel", "limestone")
LEVER_COL = {
    "more_intensive_use": "intensive_use", "lifetime_extension": "lifetime_ext",
    "reuse_rate": "reuse", "recycle_rate": "recycling",
    "sand_sub": "sand_substitution", "clinker_target": "clinker_ratio_target",
}
ALL_LEVERS = set(LEVER_COL)
ISO = dict(zip(("MIU", "LE", "RE", "RC", "SS", "CCR"), LEVER_COL))
WEDGE = {
    "wedge_MIU": {"more_intensive_use"},
    "wedge_LE": {"more_intensive_use", "lifetime_extension"},
    "wedge_RE": {"more_intensive_use", "lifetime_extension", "reuse_rate"},
    "wedge_RC": {"more_intensive_use", "lifetime_extension", "reuse_rate", "recycle_rate"},
    "wedge_SS": ALL_LEVERS - {"clinker_target"},
}
SCENARIOS = ["BAU", "CE", *ISO, *[f"CE{x}" for x in range(10, 100, 10)],
             *WEDGE, *[f"BAU_SSP{x}" for x in (1, 3, 4, 5)]]
_CACHE = {}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_model(base):
    key = str(base)
    if key not in _CACHE:
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        path = base / "code" / manifest.get("model_file", "model.py")
        if sha256(path) != manifest["model_sha256"]:
            raise RuntimeError("Frozen model changed; use a separate run directory.")
        spec = importlib.util.spec_from_file_location("frozen_corrected_mfa", path)
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        model.DATA = base / "inputs" / "data"
        model.RESULTS = base / "stage"
        model.tqdm = lambda iterable, *a, **kw: iterable
        shared = model.load_shared_data()
        adjustments_path = base / "input-adjustments.json"
        if adjustments_path.exists():
            # Explicit, recorded modelling assumptions; raw workbooks stay intact.
            for adjustment in json.loads(adjustments_path.read_text(encoding="utf-8")):
                table = shared[adjustment["parameter"]]
                year, economy = adjustment["year"], adjustment["economy"]
                if not pd.isna(table.loc[year, economy]):
                    raise ValueError(f"Input adjustment would replace an observed value: {economy}/{year}")
                table.loc[year, economy] = adjustment["value"]
        _CACHE[key] = (model, shared,
                       pd.read_csv(model.DATA / "CE-parameters-142.csv").set_index("economy"),
                       pd.read_csv(model.DATA / "oresand-supply-cap-142.csv").set_index("economy"))
    return _CACHE[key]


def scenario_kwargs(economy, scenario, ce, oss):
    ambition, ssp = 1.0, "SSP2"
    if scenario == "BAU" or scenario.startswith("BAU_SSP"):
        levers = set()
        if scenario.startswith("BAU_"):
            ssp = scenario.removeprefix("BAU_")
    elif scenario in ISO:
        levers = {ISO[scenario]}
    elif scenario in WEDGE:
        levers = WEDGE[scenario]
    else:
        levers = ALL_LEVERS
        if scenario != "CE":
            ambition = int(scenario[2:]) / 100
    row = ce.loc[economy]
    kw = {arg: float(row[col]) if arg in levers else (None if arg == "clinker_target" else 0.0)
          for arg, col in LEVER_COL.items()}
    kw.update(oresand_supply=float(oss.loc[economy, "oresand_supply_kt_yr"]) if "sand_sub" in levers else np.inf,
              ambition=ambition, ssp=ssp, n_sim=10000, seed=42, hist_subdir="historical")
    return kw


def load_array(path):
    with np.load(path) as z:
        return z["data"]


def zip_signature(path):
    with ZipFile(path) as z:
        return [(i.filename, i.CRC, i.file_size) for i in z.infolist()]


def statistics(a):
    annual = np.percentile(a, [2.5, 50, 97.5], axis=0)
    cumulative = a.sum(axis=1, dtype=np.float64)
    return {
        "annual_mean_kt": a.mean(axis=0, dtype=np.float64).tolist(),
        "annual_p025_kt": annual[0].tolist(), "annual_median_kt": annual[1].tolist(),
        "annual_p975_kt": annual[2].tolist(),
        "cumulative_mean_kt": float(cumulative.mean()),
        "cumulative_quantiles_kt": np.percentile(cumulative, [2.5, 50, 97.5]).tolist(),
        "negative_values": int(np.count_nonzero(a < 0)),
    }


def compare_and_validate(base, economy, scenario, stage, official, backup):
    n_years = 92 if scenario == "historical" else 28
    result = {"economy": economy, "scenario": scenario, "unit": "kt", "n_sim": 10000,
              "years": list(range(1931, 2023) if scenario == "historical" else range(2023, 2051)),
              "materials": {}, "checked_at": now()}
    old_total = np.zeros((10000, n_years), dtype=np.float64)
    new_total = np.zeros_like(old_total)
    for material in MATERIALS:
        name = f"{material}_extraction.npz"
        old_path = backup / name if (backup / name).exists() else official / name
        old, new = load_array(old_path), load_array(stage / name)
        if old.shape != (10000, n_years) or new.shape != old.shape:
            raise ValueError(f"Unexpected extraction shape: {economy}/{scenario}/{name}: {old.shape}, {new.shape}")
        if not np.isfinite(old).all() or not np.isfinite(new).all() or np.any(new < 0):
            raise ValueError(f"Invalid extraction values: {economy}/{scenario}/{name}")
        result["materials"][material] = {"before": statistics(old), "after": statistics(new),
                                         "max_absolute_draw_change_kt": float(np.max(np.abs(new.astype(np.float64) - old)))}
        old_total += old
        new_total += new
    result["materials"]["total"] = {"before": statistics(old_total), "after": statistics(new_total)}
    # Saved production/inflows/stocks should remain physical. Split ratios and
    # growth/turnover components are allowed to reflect contracting stocks.
    checked = 0
    for p in stage.glob("*.npz"):
        name = p.stem
        if name.startswith(("I_", "S_", "O_")) or name in ("clinker_production", "cement_production", "sand_stock", "gravel_stock", "limestone_stock", "clinker_stock"):
            a = load_array(p)
            if not np.isfinite(a).all():
                raise ValueError(f"Nonfinite saved physical flow: {economy}/{scenario}/{p.name}")
            if scenario != "historical" and name.startswith(("I_", "S_", "O_")) and np.min(a) < -1e-5:
                raise ValueError(f"Negative future physical flow: {economy}/{scenario}/{p.name}: {np.min(a)}")
            checked += 1
    result["physical_arrays_checked"] = checked
    write_json(base / "comparisons" / scenario / f"{economy}.json", result)
    return result


def install_outputs(base, stage, official, backup, keep):
    """All old bytes remain recoverable: changed files in backup, others in place."""
    official.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)
    records = []
    for name in sorted(keep):
        src, dst, old = stage / name, official / name, backup / name
        new_sig = zip_signature(src)
        if dst.exists() and zip_signature(dst) == new_sig:
            records.append({"file": name, "status": "unchanged", "npz_signature": new_sig})
            continue
        if shutil.disk_usage(OFFICIAL).free < 15 * 2**30:
            raise RuntimeError("Less than 15 GiB free on result drive; originals retained, rerun can resume.")
        record = {"file": name, "status": "updated" if dst.exists() else "created",
                  "new_sha256": sha256(src), "npz_signature": new_sig}
        if dst.exists() and not old.exists():
            # Both paths are on F:. No file in this backup is ever opened for writing.
            try:
                os.link(dst, old)
            except OSError:
                shutil.copy2(dst, old)
            if sha256(old) != sha256(dst):
                raise RuntimeError(f"Backup verification failed: {dst}")
        if old.exists():
            record["old_sha256"] = sha256(old)
        temp = dst.with_name(dst.name + ".mfa-rerun-tmp")
        shutil.copy2(src, temp)
        if sha256(temp) != record["new_sha256"]:
            raise RuntimeError(f"Copy verification failed: {dst}")
        os.replace(temp, dst)
        records.append(record)
    return records


def clear_stage_npz(base, directory):
    directory = directory.resolve()
    if not directory.is_relative_to((base / "stage" / "material-flows").resolve()):
        raise ValueError(f"Cleanup outside rerun staging directory: {directory}")
    for path in directory.glob("*.npz"):
        path.unlink()


def run_one(base_string, backup_string, economy, scenario):
    base, backup_root = Path(base_string), Path(backup_string)
    marker = base / "state" / scenario / f"{economy}.json"
    if marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            return {"economy": economy, "scenario": scenario, "status": "resumed"}
    stage = base / "stage" / "material-flows" / economy / scenario
    official = OFFICIAL / "material-flows" / economy / scenario
    backup = backup_root / "material-flows" / economy / scenario
    start = time.monotonic()
    log_path = base / "logs" / f"{scenario}-{economy}.log"
    write_json(marker, {"status": "running", "started": now(), "pid": os.getpid()})
    try:
        with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print(now(), "START", economy, scenario, flush=True)
            model, shared, ce, oss = load_model(base)
            pilot = base / "state" / f"pilot-{economy}-{scenario}.json"
            if not pilot.exists():
                clear_stage_npz(base, stage)
                if scenario == "historical":
                    model.run_historical(economy, shared, n_sim=10000, seed=42)
                else:
                    model.run_projection(scenario, economy, **scenario_kwargs(economy, scenario, ce, oss))
            # Figure-only runs retain exactly the original three extraction arrays.
            keep = {f"{x}_extraction.npz" for x in MATERIALS} if scenario in WEDGE or scenario.startswith("BAU_SSP") else {p.name for p in stage.glob("*.npz")}
            if not keep:
                raise RuntimeError(f"No staged output: {economy}/{scenario}")
            comparison = compare_and_validate(base, economy, scenario, stage, official, backup)
            records = install_outputs(base, stage, official, backup, keep)
            state = {"status": "complete", "economy": economy, "scenario": scenario,
                     "finished": now(), "seconds": time.monotonic() - start,
                     "files": len(records), "changed_files": sum(r["status"] != "unchanged" for r in records),
                     "physical_arrays_checked": comparison["physical_arrays_checked"],
                     "peak_memory_gib": psutil.Process().memory_info().peak_wset / 2**30,
                     "file_manifest": records}
            write_json(marker, state)
            if scenario != "historical":
                clear_stage_npz(base, stage)
            print(now(), "COMPLETE", economy, scenario, state["seconds"], flush=True)
        gc.collect()
        return {k: v for k, v in state.items() if k != "file_manifest"}
    except Exception:
        error = traceback.format_exc()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(error)
        write_json(marker, {"status": "failed", "economy": economy, "scenario": scenario,
                            "time": now(), "error": error})
        raise


def refresh_summary(base, economies, backup):
    """A readable interim/final comparison, updated after every scenario."""
    summaries, countries = [], []
    for scenario in ["historical", *SCENARIOS]:
        paths = [base / "comparisons" / scenario / f"{e}.json" for e in economies]
        markers = [base / "state" / scenario / f"{e}.json" for e in economies]
        if not all(p.exists() for p in paths + markers):
            continue
        if not all(json.loads(p.read_text(encoding="utf-8")).get("status") == "complete" for p in markers):
            continue
        jobs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
        for material in (*MATERIALS, "total"):
            before = np.sum([j["materials"][material]["before"]["annual_mean_kt"] for j in jobs], axis=0)
            after = np.sum([j["materials"][material]["after"]["annual_mean_kt"] for j in jobs], axis=0)
            bm = np.sum([j["materials"][material]["before"]["annual_median_kt"] for j in jobs], axis=0)
            am = np.sum([j["materials"][material]["after"]["annual_median_kt"] for j in jobs], axis=0)
            summaries.append({"scenario": scenario, "material": material, "years": jobs[0]["years"],
                              "before_annual_mean_Gt": (before / 1e6).tolist(), "after_annual_mean_Gt": (after / 1e6).tolist(),
                              "before_sum_country_medians_Gt": (bm / 1e6).tolist(), "after_sum_country_medians_Gt": (am / 1e6).tolist(),
                              "before_cumulative_mean_Gt": float(before.sum() / 1e6),
                              "after_cumulative_mean_Gt": float(after.sum() / 1e6),
                              "change_percent": float((after.sum() / before.sum() - 1) * 100) if before.sum() else None})
        for j in jobs:
            for material in (*MATERIALS, "total"):
                stats = j["materials"][material]
                b, a = stats["before"]["cumulative_mean_kt"] / 1000, stats["after"]["cumulative_mean_kt"] / 1000
                countries.append({"economy": j["economy"], "scenario": scenario, "material": material,
                                  "before_cumulative_mean_Mt": b, "after_cumulative_mean_Mt": a,
                                  "change_Mt": a - b, "change_percent": (a / b - 1) * 100 if b else None})
    write_json(base / "global-comparison.json", summaries)
    write_json(base / "economy-comparison.json", countries)
    completed = [s for s in ["historical", *SCENARIOS] if any(r["scenario"] == s for r in summaries)]
    complete = len(completed) == 27
    names = {"sand": "砂", "gravel": "砾石", "limestone": "石灰石", "total": "三类材料合计"}
    lines = ["MFA 修正后全量重算与前后对比", f"更新时间：{now()}",
             f"状态：{'全部完成' if complete else '计算进行中'}；已完整完成 {len(completed)}/27 个历史期或未来情景。",
             "范围：142 个经济体；每个经济体 10,000 次模拟；随机种子 42。",
             "历史期 1931—2022 年；未来期 2023—2050 年。",
             "对照是本次重算前磁盘上的正式结果；使用冻结的修正后代码及当前输入。",
             "包含此前六项修正，以及试跑发现的补充修正：实际胶凝材料再利用量不超过当年需求，避免负的新料需求和虚假熟料入库。",
             "下表采用模拟均值：先对每个经济体的 10,000 次模拟取均值，再跨经济体和年份求和。",
             "global-comparison.json 另保留逐年结果及各经济体中位数之和，便于与原汇总口径对照。",
             "本次仅更新 MFA；下游开采面积、生物多样性、图件和论文数值尚需另行刷新。",
             f"差异备份目录：{backup}",
             "原始文件恢复规则：备份目录中存在的文件使用备份；其余文件原值未变，仍在正式目录。",
             "", "全球累计开采量（Gt；此前 → 修正后；相对变化）："]
    for row in summaries:
        pct = row["change_percent"]
        lines.append(f"{row['scenario']:12s} {names[row['material']]:8s}  {row['before_cumulative_mean_Gt']:.6f} → {row['after_cumulative_mean_Gt']:.6f}  {pct:+.3f}%" if pct is not None else f"{row['scenario']} {names[row['material']]} 原值为零")
    for scenario in completed:
        if scenario not in ("historical", "BAU", "CE"):
            continue
        lines.extend(["", f"{scenario}：累计总开采量绝对变化最大的 10 个经济体（Mt）"])
        subset = sorted((r for r in countries if r["scenario"] == scenario and r["material"] == "total"), key=lambda r: abs(r["change_Mt"]), reverse=True)[:10]
        for r in subset:
            pct = f"{r['change_percent']:+.3f}%" if r["change_percent"] is not None else "原值为零"
            lines.append(f"{r['economy']}  {r['before_cumulative_mean_Mt']:.3f} → {r['after_cumulative_mean_Mt']:.3f}  {pct}")
    (base / "重算进度与结果对比.txt").write_text("\n".join(lines), encoding="utf-8-sig")
    return complete


def execute_batch(base, backup, economies, scenario, n_jobs):
    # Importable frozen worker functions keep one input cache per process and
    # avoid serializing the parent's loaded scientific module with every task.
    sys.path.insert(0, str(base / "code"))
    worker_module = __import__("mfa_rerun_worker")
    print(now(), "START BATCH", scenario, "economies", len(economies), "workers", n_jobs, flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        stream = Parallel(n_jobs=n_jobs, return_as="generator_unordered", batch_size=1, pre_dispatch=n_jobs)(
            delayed(worker_module.run_one)(str(base), str(backup), e, scenario) for e in economies)
        for index, result in enumerate(stream, 1):
            print(now(), scenario, f"{index}/{len(economies)}", json.dumps(result), flush=True)
            write_json(base / "progress.json", {"status": "running", "scenario": scenario,
                       "completed_in_batch": index, "economies": len(economies), "updated": now(),
                       "last_result": result, "backup": str(backup)})
    refresh_summary(base, economies, backup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--historical-jobs", type=int, default=2)
    parser.add_argument("--future-jobs", type=int, default=10)
    parser.add_argument("--only", default="", help="Optional comma-separated scenario names for a checked resume.")
    args = parser.parse_args()
    base = Path(args.run_root).resolve()
    import msvcrt
    run_lock = (base / "run.lock").open("a+b")
    if run_lock.seek(0, 2) == 0:
        run_lock.write(b"0")
        run_lock.flush()
    run_lock.seek(0)
    try:
        msvcrt.locking(run_lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        raise RuntimeError("This run directory already has an active coordinator.") from error
    backup = OFFICIAL.parent / "rerun-backups" / base.name
    backup.mkdir(parents=True, exist_ok=True)
    model, shared, ce, oss = load_model(base)
    economies = list(shared["economies_142"])
    # Prioritize representative economies without changing any sample or input.
    first = ["CHN", "HKG", "USA", "CAN", "FIN", "SGP", "GRC", "MTQ", "IND", "DEU"]
    economies = [e for e in first if e in economies] + [e for e in economies if e not in first]
    for entry in json.loads((base / "manifest.json").read_text(encoding="utf-8"))["inputs"]:
        if sha256(Path(entry["snapshot"])) != entry["sha256"]:
            raise RuntimeError(f"Frozen input changed: {entry['snapshot']}")
    plan = {"started": now(), "economies": economies, "scenarios": ["historical", *SCENARIOS],
            "jobs": 27 * len(economies), "n_sim": 10000, "seed": 42,
            "historical_workers": args.historical_jobs, "future_workers": args.future_jobs,
            "backup": str(backup), "official": str(OFFICIAL), "python": sys.executable,
            "numpy": np.__version__, "pandas": pd.__version__}
    write_json(base / "run-plan.json", plan)
    shutil.copy2(base / "manifest.json", backup / "manifest.json")
    print(json.dumps(plan), flush=True)
    todo = args.only.split(",") if args.only else ["historical", *SCENARIOS]
    if any(s not in ["historical", *SCENARIOS] for s in todo):
        raise ValueError("Unknown scenario in --only")
    if "historical" not in todo:
        if not all((base / "state" / "historical" / f"{e}.json").exists() for e in economies):
            raise RuntimeError("Complete this run's historical batch before future-only resume.")
    for scenario in todo:
        execute_batch(base, backup, economies, scenario,
                      args.historical_jobs if scenario == "historical" else args.future_jobs)
    complete = refresh_summary(base, economies, backup)
    if complete:
        report_script = base / "code" / "finish_report.py"
        if report_script.exists():
            import subprocess
            subprocess.run([sys.executable, str(report_script), str(base)], check=True)
        write_json(base / "progress.json", {"status": "complete", "updated": now(), "jobs": 27 * len(economies), "backup": str(backup)})
        print(now(), "ALL MFA RERUNS AND COMPARISONS COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if "--run-root" in sys.argv:
            failure_base = Path(sys.argv[sys.argv.index("--run-root") + 1]).resolve()
            write_json(failure_base / "progress.json", {"status": "failed", "updated": now(),
                       "error": traceback.format_exc()})
        traceback.print_exc()
        raise
