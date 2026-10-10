# Future biodiversity risks of construction aggregate mining

*Code and analysis pipeline for the manuscript by Wu, Cao, Torres et al. (2026)*

This repository contains the full computational workflow for projecting global construction aggregate (sand, gravel, and limestone-for-cement) mining expansion through 2050 and assessing its overlap with terrestrial vertebrate biodiversity. The pipeline integrates material-flow analysis, machine-learning suitability modelling, trade-aware spatial allocation, and rarity-weighted biodiversity mapping into a single end-to-end framework, and evaluates how demand-side circular-economy measures and supply-side conservation-aware siting can jointly reduce mining's biodiversity footprint.

**Headline finding.** Under business-as-usual, projected construction aggregate extraction between 2023 and 2050 is approximately **447 Gt**, with a mining footprint of **~10,143 km²** and a total **mining-biodiversity overlap of ~2.6 species-range equivalents**. Combining an ambitious circular economy with conservation-aware siting reduces that overlap by **about 58%**. Numerical summaries and their output sources are given below.

---

## Table of contents

1. [Repository structure](#1-repository-structure)
2. [Workflow overview](#2-workflow-overview)
3. [Scenario and uncertainty framework](#3-scenario-and-uncertainty-framework)
4. [Software environment](#4-software-environment)
5. [Quarry data extraction from OpenStreetMap](#5-quarry-data-extraction-from-openstreetmap)
6. [Data acquisition](#6-data-acquisition)
7. [Section 0 — Data preparation](#7-section-0--data-preparation)
8. [Section 1 — IMAGINE-Concrete: material flow analysis](#8-section-1--imagine-concrete-material-flow-analysis)
9. [Section 2 — Mining suitability model](#9-section-2--mining-suitability-model)
10. [Section 3 — Mining area simulation](#10-section-3--mining-area-simulation)
11. [Section 4 — Biodiversity risk assessment](#11-section-4--biodiversity-risk-assessment)
12. [Section 9 — Figures](#12-section-9--figures)
13. [Key results summary](#13-key-results-summary)
14. [Citation](#14-citation)

---

## 1. Repository structure

The repository primarily distributes source code and this README. The `data/`, `results/`, `figures/`, and `text/` folders are excluded by `.gitignore` and are **not** distributed with the repository. Reproduction requires the external datasets and model-specific inputs listed in [Section 6](#6-data-acquisition), together with configuration of the paths used by the notebooks; cloning this repository alone does not supply those inputs.

```
aggregate-mining-biodiversity/
│
├── 0-data-preparation/             # Rasterize the common Eckert IV analysis grid
│   ├── 1-make-country-raster.ipynb
│   ├── 2-make-economy-raster.ipynb
│   ├── 3-make-protected-area-raster.ipynb
│   ├── 4-make-biome-raster.ipynb            ← WWF TEOW biomes
│   ├── 4-make-biome-raster-2017.ipynb       ← RESOLVE Ecoregions 2017 (primary)
│   ├── 5-make-income-group-raster.ipynb
│   ├── 8-make-region-mask.ipynb
│   ├── 9-reproject-GDP-data.ipynb
│   ├── 9-reproject-population-grids.ipynb
│   └── 10-reproject-building-mineral-grid.ipynb
│
├── 1-IMAGINE-Concrete/             # Material flow analysis & extraction demand
│   ├── IMAGINE_Concrete.py             ← core probabilistic MFA module
│   ├── run-wedge-scenarios.py          ← CLI for circular-economy wedge ladder
│   ├── 1-get-trade-data.ipynb
│   ├── 2-compile-trade-data.ipynb
│   ├── 3-determine-future-stock.ipynb
│   ├── 4-make-CE-parameters.ipynb
│   ├── 5-estimate-oresand-supply-cap.ipynb
│   ├── 6-run-simulations.ipynb
│   ├── 7-check-results.ipynb
│   └── 8-compile-results.ipynb
│
├── 2-mining-suitability-model/     # Global Random Forest suitability classifier
│   ├── 1-filter-aggregate-quarries.ipynb
│   ├── 2-rasterize-quarry-area.ipynb
│   ├── 3-1-forest.ipynb  …  3-9-water-mask.ipynb   ← feature engineering (9 notebooks)
│   ├── 4-train-models.ipynb
│   ├── 5-predict-mining-suitability.ipynb
│   └── 6-sample-effect-test.ipynb
│
├── 3-mining-area-simulation/       # Spatial demand allocation & footprint projection
│   ├── 1-downscale-extraction-demand.ipynb
│   ├── 2-1-process-desert-map.ipynb
│   ├── 2-2-estimate-mineable-land.ipynb
│   ├── 3-fit-extraction-area-relationship.ipynb
│   ├── 4-estimate-aggregate-reserve.ipynb
│   ├── 5-1-bilateral-trade-seed.ipynb
│   ├── 5-1-trade-gravity-matrix.ipynb
│   ├── 5-2-map-extraction.ipynb
│   ├── 5-3-cost-analysis.ipynb
│   ├── 6-estimate-mining-area.ipynb
│   └── _regen_b06_factorial.py         ← regenerate β=0.6 (CON/IAP) uncertainty factorial
│
├── 4-biodiversity-risk-assessment/ # Species ranges, Area of Habitat, overlap
│   ├── 1-make-range-raster-for-amphibians.ipynb
│   ├── 1-make-range-raster-for-birds.ipynb
│   ├── 1-make-range-raster-for-mammals.ipynb
│   ├── 1-make-range-raster-for-reptiles.ipynb
│   ├── 2-collect-and-compile-IUCN-assessments.ipynb
│   ├── 3-resample-habitat-class-maps.ipynb
│   ├── 4-make-AOH.ipynb
│   ├── 5-make-rarity-richness-maps.ipynb
│   └── 6-biodiversity-risk-assessment.ipynb
│
├── 9-draw-figures/                 # Publication figures (main, extended-data, supplementary)
│   ├── Fig-1.ipynb … Fig-4.ipynb
│   ├── Extended-Data-Figure-2.ipynb, -3, -5
│   ├── Supplementary-Figure-1.ipynb … Supplementary-Figure-23.ipynb   (SF-15 not used)
│   └── _verify.py                      ← legacy figure helper (contains an obsolete notebook path)
│
├── data/                           # Input data (download separately — see Section 6; git-ignored)
├── results/                        # Intermediate & final outputs (git-ignored)
├── figures/                        # Publication-ready figures (git-ignored)
├── text/                           # Manuscript, SI, response (git-ignored)
└── README.md
```

---

## 2. Workflow overview

The analytical sections are largely sequential. The biodiversity layers (Section 4, steps 1–5) can be prepared alongside demand and suitability modelling. Rarity-weighted richness is needed for conservation-aware allocation in Section 3; the final overlap assessment (Section 4, step 6) then combines those richness layers with the projected mining footprints.

```
Section 0 → Section 1 ─────────────────┐
Section 0 → Section 2 ─────────────────┼→ Section 3 → Section 4 (step 6) → Section 9
Section 4 (steps 1–5; AOH and RWR) ────┘                  ↑
              └─────────────────────────────────────────┘
```

| Section | Role | Key output |
|---------|------|------------|
| 0 — Data preparation | Build the common equal-area analysis grid and rasterize administrative, conservation, and socioeconomic layers | Base-layer GeoTIFFs at 1–100 km in Eckert IV |
| 1 — IMAGINE-Concrete | Project sand, gravel, and limestone extraction demand to 2050 for 142 economies | Probabilistic (10,000-run) annual demand by scenario |
| 2 — Mining suitability | Train and apply a global Random Forest classifier on 136,666 quarries | 1 km global suitability raster (probability 0–1) |
| 3 — Mining area simulation | Allocate demand spatially under a φ×β scenario sweep and convert to footprint area | Mining-area rasters (km²) across scenarios, β, and uncertainty levels |
| 4 — Biodiversity assessment | Map vertebrate Area of Habitat, rarity-weighted richness, and overlay mining | Mining-biodiversity overlap (species-range equivalents) |
| 9 — Figures | Generate all main, extended-data, and supplementary figures | Multi-panel maps and charts |

**Spatial reference system used throughout:** Eckert IV equal-area projection (ESRI:54012). Primary analysis is at 1 km; outputs are aggregated to 2, 5, 10, 30, 50, and 100 km as needed, with **50 km** used for the mining-area and biodiversity-overlap layers.

---

## 3. Scenario and uncertainty framework

The study compares four **signpost scenarios** that sit on a continuous **two-lever plane**. Understanding this plane is essential to reading Sections 3, 4, and the figures.

- **φ (phi) — circular-economy ambition** (demand side, 0 → 1). Scales the bundle of demand-side measures from business-as-usual (φ=0) to a full circular economy (φ=1). In code the φ axis is realized as 11 demand scenarios: `BAU, CE10, CE20, …, CE90, CE`.
- **β (beta) — conservation priority** (supply side, 0 → 1). Reweights how each cell's demand is allocated among candidate supply cells, trading haul distance against avoiding high-biodiversity cells. In code the β axis is 13 values: `0.0, 0.1, …, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0`.

The full sweep is 11 φ × 13 β = **143 combinations**. The spatial allocator writes `results/sweep-config.json` for downstream analysis and figure notebooks.

| Signpost | (φ, β) | Meaning |
|----------|--------|---------|
| **BAU** | (0, 0) | Business-as-usual demand, least-cost siting |
| **CE** | (1, 0) | Full circular economy, least-cost siting |
| **CON** | (0, 0.6) | BAU demand, conservation-aware siting at the cost-benefit "knee" |
| **IAP** | (1, 0.6) | Integrated Action Portfolio — full CE **and** conservation siting |

> **β_CON = 0.6** is the selected "knee" of the normalized cost-benefit frontier. It is derived in `Supplementary-Figure-22.ipynb` (the cheaper end of the tied β=0.6–0.7 knee) and set as `BETA_CON` in `Fig-4.ipynb`.

> **Important.** CON is a **siting-weight lever, not an extra land-exclusion mask.** The base of mineable land (which excludes urban areas, water bodies, deserts, and all IUCN Ia–VI protected areas) is **identical across all four scenarios**. The difference between BAU and CON lies entirely in the β allocation score, not in which land is available.

**Three-dimensional uncertainty.** The signpost scenarios are evaluated across three uncertainty dimensions, each at low/central/high (L/M/H) levels:

1. **D — extraction demand** (from the 10,000-run Monte Carlo material-flow model),
2. **R — aggregate reserve** (from lithology thickness/density/recovery parameters),
3. **A — extraction-to-area relationship** (from the fitted power law).

The **central case is D-M / R-M / A-M (often tagged `DM-RM`)**. The figure bounds use the **anti-diagonal** corners — lower bound `DL-RH-AL`, upper bound `DH-RL-AH` — reflecting the opposite role of reserve availability in the allocation. The full 3³ = 27-combination factorial is evaluated for sensitivity analysis at the signposts and for the individual demand-side scenarios at β=0. Spatial uncertainty ranges are scenario envelopes, not Monte Carlo confidence intervals.

---

## 4. Software environment

The analysis uses Python notebooks and helper scripts, with external geospatial tools for preprocessing. Key dependencies:

- **Geospatial:** `rasterio`, `geopandas`, `shapely`, `fiona`, `pyproj`, `cartopy`
- **Scientific computing:** `numpy`, `scipy`, `pandas`, `scikit-learn`, `joblib`
- **I/O & plotting:** `openpyxl`, `matplotlib`, `requests`, `tqdm`
- **Notebook execution:** `jupyter`, `nbformat`
- **System tools (for OSM preprocessing):** `osmium-tool` and `GDAL`/`ogr2ogr`

A conda environment is recommended because several steps depend on native libraries (LAPACK, GDAL, PROJ). Heavy steps (`6-run-simulations`, `6-estimate-mining-area`, the AOH builders) use `joblib` parallelism and benefit from a multi-core machine with ample RAM and disk.

Before running, review each notebook's input/output paths and working-directory assumptions. Some scripts reference local drive locations outside the repository. Set these paths to the corresponding data and results directories on your system, and configure any data-provider API credentials. A version-pinned environment specification is not currently included.

---

## 5. Quarry data extraction from OpenStreetMap

The quarry training dataset is built from an OpenStreetMap snapshot dated **2021-01-01**, chosen to align temporally with the other input layers (land cover, population, protected areas, all approximately year 2020). Using a fixed snapshot ensures full reproducibility.

### 5.1 Install osmium-tool

```bash
sudo apt update
sudo apt install osmium-tool
```

### 5.2 Download OSM regional extracts

Obtain the regional `.osm.pbf` files for the **2021-01-01 snapshot used in this study** and place them in your working directory. Check the [Geofabrik raw directory index](https://download.geofabrik.de/) for available archived extracts; availability of the 2021 files is not guaranteed. If unavailable there, the original archived inputs are needed to reproduce the same quarry sample. The expected filenames are:

- africa-210101.osm.pbf
- asia-210101.osm.pbf
- australia-oceania-210101.osm.pbf
- central-america-210101.osm.pbf
- europe-210101.osm.pbf
- north-america-210101.osm.pbf
- south-america-210101.osm.pbf

### 5.3 Filter quarry features

Extract features tagged `landuse=quarry` from each regional file:

```bash
for f in *-210101.osm.pbf; do
  echo "Processing $f..."
  osmium tags-filter "$f" wr/landuse=quarry -o "${f%.osm.pbf}-quarries.osm.pbf"
done
```

### 5.4 Convert to GeoPackage

Use [GDAL](https://gdal.org/)'s `ogr2ogr` to convert each file into a [GeoPackage](https://www.geopackage.org/), extracting only the `multipolygons` layer:

```bash
sudo apt install gdal-bin

for f in *-quarries.osm.pbf; do
  ogr2ogr -f "GPKG" "${f%.osm.pbf}.gpkg" "$f" multipolygons -where "landuse='quarry'"
done
```

### 5.5 Merge into a single global file

```bash
ogrmerge.py -f GPKG -o global-quarries-210101.gpkg -single -nln quarries *-quarries.gpkg
```

The result, `data/global-quarries-210101.gpkg` (148,214 features), is the raw input to [Section 2](#9-section-2--mining-suitability-model). Notebook `2-mining-suitability-model/1-filter-aggregate-quarries.ipynb` then removes non-aggregate mines (intersecting the Maus et al. `global_mining_polygons_v2.gpkg`) to produce `data/global-aggregate-quarries-210101.gpkg` (136,666 aggregate quarries, 19,142 km²).

---

## 6. Data acquisition

The repository does not include the research inputs. Obtain the external datasets and the model-specific auxiliary/parameter files below, and place them at the paths expected by the relevant notebook. Third-party data may require registration or an access request. Provider links identify the sources; they do not supply the author-compiled parameter tables or replace the exact versions used in the analysis.

### 6.1 External datasets

| Dataset | Source | Used in |
|---------|--------|---------|
| Global cement production | [Andrew (2025)](https://zenodo.org/records/14931651) | IMAGINE-Concrete |
| International trade data | [UN Comtrade Database](https://comtradeplus.un.org/) | IMAGINE-Concrete |
| Global population grid (current) | [WorldPop](https://hub.worldpop.org/doi/10.5258/SOTON/WP00647) | Data prep, stock projection |
| Future population grids (SSP1–5) | [Wang et al. (2022)](https://doi.org/10.6084/m9.figshare.19608594.v3) | Data prep, stock projection |
| Future GDP grids (SSP1–5) | SSP gridded GDP projections | Data prep, stock projection |
| Building material stocks | [WSF3D Material Stock](https://download.geoservice.dlr.de/WSF3D_MATSTOCK/files/MaterialStockGlobal/) | Data prep, stock projection |
| OpenStreetMap regional extracts | [Geofabrik](https://download.geofabrik.de/) | Mining suitability |
| Maus et al. global mining polygons (v2) | [Maus et al.](https://doi.pangaea.de/10.1594/PANGAEA.942325) | Quarry filtering, ore-sand cap |
| Travel time to cities | [Oxford MAP (2015)](https://developers.google.com/earth-engine/datasets/catalog/Oxford_MAP_accessibility_to_cities_2015_v1_0) | Mining suitability |
| Distance to freshwater | [Kummu et al.](https://zenodo.org/records/4956396) | Mining suitability |
| GMTED2010 elevation | [USGS](https://developers.google.com/earth-engine/datasets/catalog/USGS_GMTED2010_FULL) | Suitability, AOH |
| Global Lithological Map (GLiM) | [Hartmann & Moosdorf](https://doi.pangaea.de/10.1594/PANGAEA.788537) | Suitability, reserve |
| World Database on Protected Areas | [Protected Planet](https://www.protectedplanet.net/en/thematic-areas/wdpa) | Data prep, mineable land |
| Global Lakes and Wetlands Database | [GLWD v2](https://figshare.com/articles/dataset/Global_Lakes_and_Wetlands_Database_GLWD_version_2_0/28519994) | Mining suitability |
| Global forest cover | [Hansen et al.](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2024_v1_12) | Mining suitability |
| Global cropland extent | [Potapov et al.](https://doi.org/10.5281/zenodo.12527546) | Mining suitability |
| ESA CCI Land Cover (urban / bare land) | [ESA CCI](http://maps.elie.ucl.ac.be/CCI/viewer/download.php) | Mining suitability |
| IUCN species range polygons | [IUCN Red List](https://www.iucnredlist.org/resources/spatial-data-download) | Biodiversity assessment |
| BirdLife range maps (BOTW) | [BirdLife International](http://datazone.birdlife.org/species/requestdis) | Biodiversity assessment |
| Global terrestrial habitat map | [Jung et al.](https://uploads.users.earthengine.app/view/habitat-types-map) | Biodiversity assessment |
| IUCN Red List API (assessments) | [IUCN Red List API v4](https://api.iucnredlist.org/) | Biodiversity assessment |
| Country administrative boundaries | [GADM](https://gadm.org/) | All sections |
| RESOLVE Ecoregions 2017 / WWF TEOW | [RESOLVE](https://ecoregions.appspot.com/) | Data prep |
| Desert / sand classification (GSDS30) | [Zenodo](https://zenodo.org/records/10441041) | Mining area simulation |
| Trade gravity data | [CEPII Gravity (v202211)](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8) | Mining area simulation |
| Natural Earth populated places | [Natural Earth](https://www.naturalearthdata.com/) | Figures (city markers) |

### 6.2 Auxiliary / parameter files (small, model-specific)

These are read directly from `data/`:

| File | Contents |
|------|----------|
| `production-142.csv` | Cement production 1931–2022 for the 142 economies |
| `reporter-with-economy-142.csv` | UN Comtrade reporter → 142-economy mapping |
| `country-pseudo-values.csv` | Country ISO3 ↔ raster pseudo-code lookup |
| `contents.xlsx` | Economy / merged-country (parent–child) definitions |
| `params-1.xlsx`, `sector-params.xlsx`, `other-params.xlsx`, `GNR-params.xlsx` | Concrete/mortar ratios, building lifetimes, recycling rates, supplementary-cementitious-material ratios |
| `CE-parameters-142.csv` | Income-group circular-economy 2050 targets (output of notebook 1-4) |
| `oresand-supply-cap-142.csv` | Per-economy ore-sand supply ceiling (output of notebook 1-5) |
| `global-quarries-210101.gpkg`, `global-aggregate-quarries-210101.gpkg` | OSM quarry polygons (raw and aggregate-filtered) |
| `global_mining_polygons_v2.gpkg` | Maus et al. mining polygons (non-aggregate filtering) |
| `GLiM.tif` | Global Lithological Map raster |
| `Gravity_csv_V202211/`, `GSDS30/`, `China-map/`, `simplified-world-map/`, `ne_10m_populated_places/` | CEPII gravity, desert/sand classes, China provincial map, world map, city points |

---

## 7. Section 0 — Data preparation

**Folder:** `0-data-preparation/`

These notebooks establish the common geospatial framework: a global Eckert IV equal-area grid at multiple resolutions, with administrative, conservation, and socioeconomic layers rasterized onto it. All outputs are compressed GeoTIFFs under `results/`.

| Notebook | Purpose | Key output |
|----------|---------|------------|
| `1-make-country-raster` | Rasterize GADM country boundaries (253 countries + Hong Kong + Macau = 255 units) | Country-ID rasters at 1, 2, 5, 10, 30, 50, 100 km |
| `2-make-economy-raster` | Rasterize the 142 cement-reporting economies (parent–child country merges) | Economy-ID rasters at 1, 10, 50 km + `pseudo-codes-economies-142.csv` |
| `3-make-protected-area-raster` | Rasterize 283,894 WDPA polygons (Jul 2024) by IUCN category | Merged + per-category masks (1 km) and 50 km aggregates |
| `4-make-biome-raster` | Rasterize WWF TEOW 14-biome classification | Biome rasters at 1, 10, 50 km |
| `4-make-biome-raster-2017` | Rasterize RESOLVE Ecoregions 2017 (847 ecoregions → 14 biomes) — the **primary** biome layer | `resolve-biome-raster-eck4-*` at 1, 2, 5, 10, 50 km |
| `5-make-income-group-raster` | Rasterize World Bank income groups (L / LM / UM / H) | Income-group raster at 1 km |
| `8-make-region-mask` | Define 15 biogeographic regions + USA/Canada/China state-province masks | `region-mask-*` (1/10/50 km) + `state-raster-eck4-1km` |
| `9-reproject-GDP-data` | Reproject GDP-per-capita grids (2020 + 2025–2050 under SSP1–5) to Eckert IV | GDP rasters at 1 km |
| `9-reproject-population-grids` | Reproject WorldPop 2020 + SSP1–5 future population to Eckert IV | Population rasters at 1, 10, 50 km |
| `10-reproject-building-mineral-grid` | Reproject WSF3D building-material stock density (≈401 Gt global) | `features/building-mineral-stock-1km.tif` |

**Pseudo-code conventions:**
- WDPA IUCN categories: Ia = 8, Ib = 7, II = 6, III = 5, IV = 4, V = 3, VI = 2; not-reported/applicable = 1; unprotected = 0.
- Income groups: Low = 1, Lower-Middle = 2, Upper-Middle = 3, High = 4.
- Biogeographic regions (1–15): Middle East, LAC, North Africa, West Africa, East Africa, Central Africa, Southern Africa, Europe, Southeast Asia, Oceania, Southern Asia, Eastern Asia, Central & Northern Asia, Northern America, China.

---

## 8. Section 1 — IMAGINE-Concrete: material flow analysis

**Folder:** `1-IMAGINE-Concrete/`

IMAGINE-Concrete is a global material-flow-analysis (MFA) model that projects extraction demand for construction aggregates (sand, gravel, limestone-for-cement) through 2050, driven by building-stock dynamics across **142 economies**. Uncertainty is propagated with **10,000 Monte Carlo simulations** over a historical period of **1931–2022** and a projection horizon of **2023–2050**.

### Core modules

| File | Description |
|------|-------------|
| `IMAGINE_Concrete.py` | Core MFA engine. Key functions: `run_historical()` (1931–2022 baseline), `run_projection()` (2023–2050 per scenario, with CE-lever and φ-ambition parameterization), `fit_logistic_with_constraints()` (4-parameter logistic stock projection), `cal_dem_rate_conditional()` (age-cohort demolition rates). Tracks 3 building sectors (residential, non-residential, civil) and 4 concrete strength classes (C15, C16–23, C24–35, C35). |
| `run-wedge-scenarios.py` | Standalone, resumable CLI that generates the cumulative circular-economy "wedge" ladder (`wedge_MIU → wedge_LE → wedge_RE → wedge_RC → wedge_SS`) at n_sim = 10,000, pruning each output folder to the three extraction arrays for Supplementary Figures 8–10. Supports `--smoke`, `--n-jobs`, `--only`. |

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `1-get-trade-data` | Download bilateral trade flows (cement, clinker, limestone, sand, gravel) from the UN Comtrade API |
| `2-compile-trade-data` | Aggregate raw bilateral matrices into 142-economy import/export time series (1931–2022) |
| `3-determine-future-stock` | Project per-economy building-material stock to 2050 under SSP1–5 from GDP × population-density bins → `future-stock-growth.xlsx` |
| `4-make-CE-parameters` | Define income-group-differentiated 2050 targets for the 6 circular-economy levers → `data/CE-parameters-142.csv` |
| `5-estimate-oresand-supply-cap` | Estimate per-economy ore-sand (mine-tailings) supply ceilings → `data/oresand-supply-cap-142.csv` |
| `6-run-simulations` | Run all 142 economies across the scenario set (joblib parallel); produce per-economy `.npz` flow archives |
| `7-check-results` | Validation/QA — compare the 10,000-run results against an earlier 2,000-run baseline |
| `8-compile-results` | Aggregate to global totals and export `results-from-IMAGINE-Concrete.xlsx` (extraction, in-use stock, outflows) |

### Demand-side circular-economy levers

The full circular economy (φ=1) bundles six levers, each phased in by World Bank income group:

| Lever | Strategy |
|-------|----------|
| MIU | More intensive use |
| LE | Building lifetime extension |
| RE | Component re-use |
| RC | End-of-life recycling of aggregates |
| SS | Sand substitution with recovered ore-sand |
| CCR | Clinker-to-cement ratio reduction / clinker substitution |

Figure 2 compares **individual-lever scenarios** (`MIU`, `LE`, `RE`, `RC`, `SS`, `CCR`) and the combined `CE` scenario against `BAU`. Supplementary Figures 8–10 use a separate **cumulative wedge ladder** (`BAU → wedge_MIU → wedge_LE → wedge_RE → wedge_RC → wedge_SS → CE`). The **φ-graded sweep** (`CE10 … CE90`) used in the spatial analysis scales the levers together (see [Section 3](#3-scenario-and-uncertainty-framework)).

### Key demand estimates (2023–2050 cumulative, BAU vs CE)

| Material | BAU | CE | Reduction |
|----------|-----|-----|-----------|
| Sand | 207.3 Gt (203.0–212.0) | 160.6 Gt (157.5–163.9) | −22% |
| Gravel | 145.6 Gt (141.2–150.3) | 118.1 Gt (114.7–121.6) | −19% |
| Limestone (for cement) | 94.0 Gt (70.5–112.8) | 74.0 Gt (62.4–81.7) | −21% |
| **Total** | **446.7 Gt (421.5–469.0)** | **352.7 Gt (339.3–363.7)** | **−21%** |

Values are the medians and 2.5th–97.5th percentile intervals in `results/Data-for-Fig-2a.csv`, as checked on 10 October 2026. Figure 2 aggregates the Monte Carlo draws over economies and years (and materials for the total) **before** taking percentiles. Summing separately calculated medians or percentile endpoints does not give the same global statistics.

---

## 9. Section 2 — Mining suitability model

**Folder:** `2-mining-suitability-model/`

A Random Forest classifier is trained on **136,666** globally distributed aggregate quarries to predict where future mining is likely to occur.

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `1-filter-aggregate-quarries` | Remove 11,548 non-aggregate mines (intersecting Maus et al. polygons) from 148,214 OSM features → 136,666 aggregate quarries (19,142 km²) |
| `2-rasterize-quarry-area` | Rasterize aggregate quarries (and Maus mines) to per-cell area at 1 km and 50 km |
| `3-1` … `3-9` | Feature engineering — prepare the predictor rasters at 1 km (forest, cropland, urban, bare land, accessibility, distance-to-water, elevation/slope, GLiM lithology, water mask) |
| `4-train-models` | Compile training data with two-tier pseudo-absence sampling; train and cross-validate the Random Forest |
| `5-predict-mining-suitability` | Apply the model globally → `results/global-mining-suitability-1km.tif` (probability ×10,000, uint16) |
| `6-sample-effect-test` | Sensitivity test — downsample EU+NA presences (5–100%) to check robustness to data-rich regions |

### Training data & features

| Item | Value |
|------|-------|
| Total samples | 601,390 (195,765 quarry presences, 405,625 pseudo-absences) |
| Pseudo-absence sampling | Two-tier: 30–300 km buffer ring + country background (≈3:1 target per country; ≈2:1 realized globally) |
| Biogeographic regions | 15 |
| Country-region groups | 312 |
| Predictors | 34 environmental features + 15 region one-hot indicators = 49 model inputs |

**Feature categories:** topography (elevation, slope); accessibility (population density, travel time to cities); land use (forest, cropland, urban, bare land); water (distance to freshwater, water-body mask); geology (**16 GLiM class masks**); seven protected-area category masks (Ia, Ib, II, III, IV, V, VI); and building-mineral stock density. The 34 environmental predictors are supplemented by 15 region indicators.

### Model configuration & performance

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | Random Forest |
| Trees | 400 |
| Max depth | 20 |
| Min samples per leaf | 5 |
| Class weight | balanced |
| Validation | Grouped cross-validation using region/country or subnational groups; regional evaluation uses up to 5 folds |

| Region | AUC | κ |
|--------|-----|---|
| Mean ± SD across 73 region/fold evaluations | 0.832 ± 0.085 | — |
| North America (best) | 0.895 | 0.602 |
| East Africa (weakest) | 0.748 | 0.347 |

**Outputs:** `results/trained-models/rf-global.pkl` (model) and `rf-global-features.pkl` (feature list); `results/global-mining-suitability-1km.tif` (global suitability raster); per-region CV metrics in `results/cv_results_per_region_global_model.csv` and `results/figure-data/`.

The AUC summary above is the unweighted mean and sample standard deviation of the 73 rows in `cv_results_per_region_global_model.csv`, as checked on 10 October 2026; it is not a pooled global AUC. The training notebook also contains other validation configurations.

---

## 10. Section 3 — Mining area simulation

**Folder:** `3-mining-area-simulation/`

This section converts the extraction demands from Section 1, the suitability map from Section 2, and the reserve/lithology layers into spatially explicit mining footprints (km²) under the [φ×β scenario sweep](#3-scenario-and-uncertainty-framework).

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `1-downscale-extraction-demand` | Downscale national demand to 50 km cells using building-material stock density as proxy; produce L/M/H (2.5 / 50 / 97.5 percentile) rasters per material, scenario, and 5-year period |
| `2-1-process-desert-map` | Reproject the GSDS30 sand/desert classification to 1 km; output fractional aeolian-sand coverage for exclusion |
| `2-2-estimate-mineable-land` | Build the mineable-land base by categorical exclusion (urban, water, existing quarries, deserts, IUCN Ia–VI) and a single suitability threshold τ = empirical P50 of quarry suitability |
| `3-fit-extraction-area-relationship` | Fit the power law A = 9.421·E^0.548 (n = 113 quarries; E in Mt, A in ha). The 95% confidence band of the fitted mean relationship is propagated into the footprint calculation; an individual-quarry prediction band is also examined as a diagnostic |
| `4-estimate-aggregate-reserve` | Estimate per-cell reserve (Mt) = mineable area × lithology factor (thickness × density × recovery), at L/M/H levels |
| `5-1-bilateral-trade-seed` | Build the observed bilateral trade seed matrix (2018–2022 average) to preserve real trade corridors |
| `5-1-trade-gravity-matrix` | Calibrate country export/import propensities and the CEPII distance matrix used in allocation |
| `5-2-map-extraction` | Unified spatial allocator — distribute global demand across cells using score = (1−β)·distance + β·rarity-weighted-richness, with two-stage trade; **writes `results/sweep-config.json`** |
| `5-3-cost-analysis` | Convert domestic transport work (Mt·km) to monetary cost; quantify the conservation premium and CE demand-reduction benefit across all scenario×β combinations |
| `6-estimate-mining-area` | Convert allocated extraction (Mt) into footprint area (km²) via the power law, distributing within 1 km sub-cells ranked by suitability×reserve; propagate D×R×A uncertainty |
| `_regen_b06_factorial.py` | Helper that regenerates the β=0.6 (CON/IAP) cells of the 27-combination D×R×A factorial by re-executing the relevant pipeline functions |

### Mineable land (shared base)

The mineable-land base is the **same for all scenarios** (see [Section 3 framework](#3-scenario-and-uncertainty-framework)). Excluded areas are urban land, water bodies, existing quarries, aeolian-sand deserts, and all IUCN Ia–VI protected areas. Of ~109 M km² of land, ~3.8 M km² (3.5%) clears both the exclusions and the empirical suitability threshold. The BAU↔CON contrast is produced by the **β allocation weight**, not by a different land mask.

**Outputs:** allocated-extraction and trade rasters/CSVs in `results/extraction-rasters/`; reserve and footprint rasters in `results/mining-area/`; the cost-analysis table `cost-analysis-DM-RM.csv`; and the canonical `results/sweep-config.json`.

---

## 11. Section 4 — Biodiversity risk assessment

**Folder:** `4-biodiversity-risk-assessment/`

Vertebrate species ranges and habitat data are combined into rarity-weighted richness layers, which are then overlaid with the projected mining footprints to compute the headline **mining-biodiversity overlap** metric.

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `1-make-range-raster-for-amphibians` / `-birds` / `-mammals` / `-reptiles` | Rasterize IUCN/BirdLife range polygons to 1 km, with bounding-box indices for memory efficiency |
| `2-collect-and-compile-IUCN-assessments` | Query the IUCN Red List API for habitat, elevation, and Red-List category data, then parse into structured CSVs |
| `3-resample-habitat-class-maps` | Resample the Jung et al. terrestrial habitat classification (levels 1 and 2) to 1 km Eckert IV |
| `4-make-AOH` | Compute Area of Habitat (AOH) = range ∩ suitable habitat classes ∩ elevation band at 1 km; upscale to 10 km (presence) and 50 km (area) |
| `5-make-rarity-richness-maps` | Build rarity-weighted richness (RWR) at 50 km, per taxon and combined, overall and by threat category (CR/EN/VU/NT) |
| `6-biodiversity-risk-assessment` | Overlay mining footprints on RWR to compute mining-biodiversity overlap across the scenario×β sweep, the 27-combination D×R×A factorial, and the signpost anti-diagonal bounds |

### Metrics

| Metric | Definition |
|--------|------------|
| **AOH** (Area of Habitat) | Intersection of a species' range with its suitable habitat classes and elevation band |
| **RWR** (rarity-weighted richness) | Σ over species of (cell AOH ÷ species' total AOH), so range-restricted species are not masked by widespread ones |
| **Mining-biodiversity overlap** | For each 50 km cell, (mining footprint in km² ÷ full cell area of 2,500 km²) × RWR, summed globally. Expressed in **species-range equivalents**. This is a forward-looking exposure/risk metric, not an observed impact. |

> **Terminology note.** This metric was formerly called "affected species range" (ASR). The current term is **"mining-biodiversity overlap"**, in **species-range equivalents**; some legacy comments still use ASR.

### Coverage

The manuscript reports **29,327 terrestrial vertebrates** with a valid Area of Habitat — 5,237 mammals, 9,927 birds, 6,215 amphibians, and 7,948 reptiles. The range-rasterization step processes the larger full IUCN/BirdLife polygon sets; AOH/elevation/habitat filtering then defines the analysed subset. RWR layers are produced for each taxon and for all vertebrates combined, overall and stratified by threat category.

---

## 12. Section 9 — Figures

**Folder:** `9-draw-figures/`

The publication figure notebooks in this directory use the outputs of Sections 1–4. `_verify.py` is a legacy helper that still references the absent `Fig-3-new.ipynb`; it is not a current validation of all figure notebooks. Run the relevant notebooks with their required inputs to regenerate the figures.

### Main figures

| Notebook | Content |
|----------|---------|
| `Fig-1` | BAU global overview — RWR and mining-footprint maps, with mining-biodiversity overlap by taxon and threat status |
| `Fig-2` | Individual circular-economy measures and combined CE — reductions in extraction, mining area, and mining-biodiversity overlap relative to BAU |
| `Fig-3` | Conservation siting (CON, β=0.6) vs BAU — overlap-change field, regional insets, and a waterfall decomposition of avoided vs transferred overlap |
| `Fig-4` | BAU / CE / CON / IAP comparison — overlap with 27-combination uncertainty, the ΔOverlap field over the (φ, β) plane, and global sub-additivity across the same 27 uncertainty combinations |

### Extended-Data figures

| Notebook | Content |
|----------|---------|
| `Extended-Data-Figure-2` | Building-material stock distribution and its stratification by population density and GDP |
| `Extended-Data-Figure-3` | Global mining-suitability model output at 1 km |
| `Extended-Data-Figure-5` | Four focal-region zoom-ins (RWR, BAU mining, CON mining, overlap-change) |

### Supplementary figures

| Notebook(s) | Content |
|-------------|---------|
| `Supplementary-Figure-1` | Map of the 142 reporting economies |
| `Supplementary-Figure-2` | Building-stock age structure by economy (2022) |
| `Supplementary-Figure-3` | Global historical aggregate extraction and in-use material stock (1931–2022) |
| `Supplementary-Figure-4` / `-5` / `-6` | Per-economy historical sand / gravel / limestone flows (5 pages each, 30 economies per page) |
| `Supplementary-Figure-7` | Per-economy building-stock projections under SSP1–5 |
| `Supplementary-Figure-8` / `-9` / `-10` | Per-economy CE wedge decomposition of sand / gravel / limestone savings (5 pages each) |
| `Supplementary-Figure-11` | Global quarry distribution over the 15 biogeographic regions (suitability-model training data) |
| `Supplementary-Figure-12` | Suitability-model per-region performance (AUC, F1, κ) |
| `Supplementary-Figure-13` | Suitability-model robustness to EU+NA subsampling |
| `Supplementary-Figure-14` | Extraction-to-area power-law fit |
| `Supplementary-Figure-16` … `-20` | RWR maps for all vertebrates, mammals, birds, amphibians, reptiles (each: total + CR/EN/VU/NT) |
| `Supplementary-Figure-21` | International aggregate trade 2023–2050 (BAU) — balances, top traders, bilateral matrix |
| `Supplementary-Figure-22` | Derivation of the conservation-priority knee β_CON = 0.6 |
| `Supplementary-Figure-23` | Per-country mining-biodiversity-overlap waterfalls (BAU → CON) |

---

## 13. Key results summary

The headline metric is **mining-biodiversity overlap** in species-range equivalents. Extraction intervals are Monte Carlo percentiles (see Section 8); footprint and overlap ranges use the anti-diagonal spatial uncertainty combinations (see Section 3). Values are rounded from the current Figure 2 tables and the central/bounds signpost tables in `results/biodiversity-risk/`, checked on 10 October 2026.

| Indicator | Value |
|-----------|-------|
| Aggregate quarries identified (global, 2021) | 136,666 features, 19,142 km² |
| Quarry distribution | Europe 65,097, N. America 23,992, Asia 22,146, Africa 13,998, S. America 8,778, Oceania 1,809, C. America 846 |
| Cumulative aggregate extraction 2023–2050 (BAU) | 446.7 Gt (421.5–469.0) |
| — sand / gravel / limestone | 207.3 / 145.6 / 94.0 Gt |
| Cumulative extraction (CE) | 352.7 Gt (339.3–363.7), −21% vs BAU |
| Mining footprint disturbed by 2050 (BAU) | ~10,143 km² (7,077–16,306) |
| Mining footprint (CE) | 8,441 km² (5,973–13,288), −17% vs BAU |
| **Total mining-biodiversity overlap (BAU)** | **2.6 species-range equivalents (1.9–4.5)** |
| Overlap (CE) | 2.3 (1.7–3.7), −14% vs BAU |
| Overlap (CON) | 1.3 (0.9–2.3), ~−50% vs BAU |
| **Overlap (IAP)** | **1.1 (0.8–1.9), −58% vs BAU** |
| Suitability-model AUC (mean ± SD over 73 region/fold rows) | 0.832 ± 0.085 |
| Vertebrate species assessed (with AOH; manuscript count) | 29,327 |

---

## 14. Citation

If you use this code or data, please cite:

> Wu, T., Cao, Z., Torres, A., et al. (2026). *Future biodiversity risks of construction aggregate mining.*

Journal and DOI information will be added once confirmed.

For questions or issues, please open a GitHub issue in this repository.
