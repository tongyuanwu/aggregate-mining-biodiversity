# Future biodiversity risks of land-sourced aggregate mining

*Code and analysis pipeline for the manuscript by Wu, Cao, Torres et al. (2026)*

This repository contains analysis code for projecting global construction aggregate (sand, gravel, and limestone-for-cement) mining expansion through 2050 and assessing its overlap with terrestrial vertebrate biodiversity. The pipeline integrates material-flow analysis, machine-learning suitability modelling, trade-aware spatial allocation, and rarity-weighted biodiversity mapping, and evaluates how demand-side circular-economy measures and supply-side conservation-aware siting can jointly reduce mining's biodiversity footprint.

**Study scope.** Extraction estimates cover aggregates embodied in **concrete and mortar**, including limestone used for cement. They do not represent all construction-mineral extraction (for example, unbound road aggregates and non-cement limestone uses are outside the model boundary). The main scenario analysis uses **SSP2**; SSP1–5 stock projections provide the broader socioeconomic context. Mining footprints represent **additional disturbance during 2023–2050**, and mining-biodiversity overlap measures potential terrestrial exposure, not observed species losses.

**Headline finding.** Under business-as-usual, projected construction aggregate extraction between 2023 and 2050 is approximately **446 Gt**, with a mining footprint of **~10,160 km²** and a total **mining-biodiversity overlap of ~2.6 species-range equivalents**. Combining an ambitious circular economy with conservation-aware siting reduces that overlap by **about 58%**. Numerical summaries and their output sources are given below.

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

The repository primarily distributes source code and this README, together with the small source PDFs used to export Extended Data Figures 1 and 4. The `data/`, `results/`, `figures/`, and `text/` folders are excluded by `.gitignore` and are **not** distributed with the repository. Selected study outputs are available in the [Zenodo data record](https://doi.org/10.5281/zenodo.23290764), with its contents described in [Section 6](#6-data-acquisition). Reproduction requires the external datasets and model-specific inputs listed there, together with configuration of notebook paths; neither cloning this repository nor downloading the output archive supplies all upstream inputs.

```
aggregate-mining-biodiversity/
│
├── 0-data-preparation/             # Rasterize the common Eckert IV analysis grid
│   ├── 1-make-country-raster.ipynb
│   ├── 2-make-economy-raster.ipynb
│   ├── 3-make-protected-area-raster.ipynb
│   ├── 4-make-biome-raster.ipynb            ← WWF TEOW biomes (used by Fig-1)
│   ├── 4-make-biome-raster-2017.ipynb       ← alternative RESOLVE Ecoregions 2017 layers
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
│   ├── Supplementary-Figure-1.ipynb … Supplementary-Figure-23.ipynb   (SF-15 is a separate schematic)
│   ├── final_figure_export.py           ← final-size PDF/PNG/JPG export with editable PDF text
│   ├── export_extended_data_schematics.py  ← export ED1 and ED4 from source PDF artwork
│   └── _verify.py                      ← legacy figure helper (contains an obsolete notebook path)
│
├── source data/figure artwork/     # Source PDFs for Extended Data Figures 1 and 4
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
| 2 — Mining suitability | Train and apply a global Random Forest classifier using rasterized presences from 136,666 quarry polygons and sampled pseudo-absences | 1 km global suitability raster (divide stored uint16 values by 10,000 for scores on 0–1) |
| 3 — Mining area simulation | Allocate demand spatially under a φ×β scenario sweep and convert to footprint area | Mining-area rasters (km²) across scenarios, β, and uncertainty levels |
| 4 — Biodiversity assessment | Map vertebrate Area of Habitat, rarity-weighted richness, and overlay mining | Mining-biodiversity overlap (species-range equivalents) |
| 9 — Figures | Generate the analytical main, extended-data, and supplementary figures; schematic exceptions are listed in Section 12 | Multi-panel maps and charts |

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

> **Important.** CON changes the **siting weight**. The mineable-land base is **identical across all four scenarios**: exclusions cover urban land, water, existing mining footprints, aeolian sand, IUCN Ia–VI protected areas, Antarctica, and cells outside the country mask. The same suitability threshold and reserve assumptions apply at a given uncertainty setting.

**Three-dimensional uncertainty.** The signpost scenarios are evaluated across three uncertainty dimensions, each at low/central/high (L/M/H) levels:

1. **D — extraction demand** (from the 10,000-run Monte Carlo material-flow model),
2. **R — aggregate reserve** (from lithology thickness/density/recovery parameters),
3. **A — extraction-to-area relationship** (from the fitted power law).

The **central case is `DM-RM-AM`**. Allocated-extraction filenames use `DM-RM`, because area uncertainty enters only when extraction is converted to footprint. The full 3³ = **27-combination factorial** is evaluated at the four signposts and for individual demand-side scenarios at β=0; the 143-point (φ, β) surface uses the central settings.

Figure 1c,d uses the **minimum and maximum across all 27 combinations**, calculated separately for each biome or taxonomic group after summing its grid cells or Red List categories. Bars show the central `DM-RM-AM` outcome. Figure 1's global summary bounds are calculated from the 27 global totals, not by summing biome or taxon extrema. For example, the mangrove lower endpoint is **46.46700 km²** (`DL-RM-AL`); the fixed `DL-RH-AL` reference gives 47.89601 km² and is not that biome's minimum.

Figure 2c,e also calculates ranges from the **minimum and maximum across all 27 combinations**. For reductions, BAU and the intervention are paired under the same D/R/A setting before finding the range. Some figure routines also read the designated **anti-diagonal** rasters (`DL-RH-AL` and `DH-RL-AH`); these agree with the global signpost overlap extrema in the checked results, but are not a general substitute for checking all combinations for regional totals or scenario differences. Spatial ranges are sensitivity envelopes, not Monte Carlo confidence intervals. The annual Figure 1a trajectories and cumulative Figure 2a statistics sum economy/material/year percentile summaries, as detailed in Section 8.

---

## 4. Software environment

The analysis uses Python notebooks and helper scripts, with external geospatial tools for preprocessing. Key dependencies:

- **Geospatial:** `rasterio`, `geopandas`, `shapely`, `fiona`, `pyproj`, `cartopy`, `affine`, and the GDAL Python bindings (`osgeo`)
- **Scientific computing:** `numpy`, `scipy`, `pandas`, `scikit-learn`, `joblib`
- **I/O & plotting:** `openpyxl`, `netCDF4`, `matplotlib`, `seaborn`, `cmocean`, `cmcrameri`, `adjustText`, `generativepy`, `Pillow`, `requests`, `tqdm`
- **Final figure export:** `pypdf`; `PyMuPDF` is also required by `export_extended_data_schematics.py`
- **Additional notebook imports:** `comtradeapicall` for trade acquisition; `xgboost` is imported by model-development notebooks although the final suitability model is a Random Forest
- **Notebook execution:** `jupyter`, `nbformat`
- **System tools (for OSM preprocessing):** `osmium-tool` and `GDAL`/`ogr2ogr`

A conda environment is recommended because several steps depend on native libraries (LAPACK, GDAL, PROJ). Heavy steps (`6-run-simulations`, `6-estimate-mining-area`, the AOH builders) use `joblib` parallelism and benefit from a multi-core machine with ample RAM and disk.

Before running, review each notebook's input/output paths and working-directory assumptions. Some scripts reference local drive locations outside the repository. Set these paths to the corresponding data and results directories on your system, and configure any data-provider API credentials. A version-pinned environment specification is not currently included.

Most notebooks assume their **working directory is their numbered section folder**, with `Path.cwd().parent` as the repository root. In particular, the MFA engine and several compilation/plotting cells use a separate material-flow results directory, while spatial notebooks use the repository's `results/`. Configure both consistently, including `IMAGINE_Concrete.py`, `8-compile-results.ipynb`, `Fig-2.ipynb`, and `_regen_b06_factorial.py`; ensure downstream inputs such as `results/material-flows/results-from-IMAGINE-Concrete.xlsx` are available at the paths actually read. The OSM shell examples below use Bash on Linux/WSL, not PowerShell syntax.

---

## 5. Quarry data extraction from OpenStreetMap

The quarry training dataset is built from an OpenStreetMap snapshot dated **2021-01-01**. Reproducing the quarry sample requires those archived extracts and the same filtering inputs. Input dates vary: population is based on 2020, while the protected-area notebook reads the **July 2024 WDPA** release.

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
mkdir -p data
ogrmerge.py -f GPKG -o data/global-quarries-210101.gpkg -single -nln quarries *-quarries.gpkg
```

Run this merge from the repository root, with the regional GeoPackages in that directory, or adjust the paths. The result, `data/global-quarries-210101.gpkg` (148,214 features in the study inputs), is the raw input to [Section 2](#9-section-2--mining-suitability-model). Notebook `2-mining-suitability-model/1-filter-aggregate-quarries.ipynb` excludes OSM polygons intersecting the Maus et al. `global_mining_polygons_v2.gpkg` inventory to produce `data/global-aggregate-quarries-210101.gpkg` (136,666 retained quarries, 19,142 km²).

---

## 6. Data acquisition

The repository does not include the research inputs. Obtain the external datasets and the model-specific auxiliary/parameter files below, and place them at the paths expected by the relevant notebook. Third-party data may require registration or an access request. Provider links identify the sources; they do not supply the author-compiled parameter tables or replace the exact versions used in the analysis.

**Study output archive.** The [Zenodo record](https://doi.org/10.5281/zenodo.23290764), checked on 11 October 2026, contains:

- `global extraction.zip`: annual BAU and CE extraction for 142 economies, three materials, 2023–2050, retaining 10,000 draws (kt).
- `mining area.zip` and `mining-biodiversity overlap.zip`: 50 km rasters for BAU, CE, CON and IAP, with 27 D/R/A combinations per scenario.
- `rarity-weighted richness.zip`: 25 taxon/combined and threat-category RWR rasters at 50 km.
- `figure data.zip`: eight figure-source GeoTIFFs, including the 1 km suitability raster and the building-stock map.

These are selected outputs, not a complete input/environment bundle. In particular, the extraction archive contains BAU and CE only, so the individual-lever and intermediate-φ outputs must be generated separately. The numerical Source Data workbooks supplied with the manuscript are separate from this archive's listed files.

### 6.1 External datasets

| Dataset | Source | Used in |
|---------|--------|---------|
| Global cement production | [Andrew (2025)](https://zenodo.org/records/14931651) | IMAGINE-Concrete |
| International trade data | [UN Comtrade Database](https://comtradeplus.un.org/) | IMAGINE-Concrete |
| Global population grid (current) | [WorldPop](https://hub.worldpop.org/doi/10.5258/SOTON/WP00647) | Data prep, stock projection |
| Future population grids (SSP1–5) | [Wang et al. (2022)](https://doi.org/10.6084/m9.figshare.19608594.v3) | Data prep, stock projection |
| Historical and future GDP grids (SSP1–5; v7) | [Wang & Sun, global gridded GDP](https://doi.org/10.5281/zenodo.7898409) | Data prep, stock projection |
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

### 6.2 Model inputs and supporting spatial files

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

The footprint calibration also requires **`results/extraction-area-relationship.csv`**, with 113 observations and columns `E (Mt)` and `A (ha)`. It is an input to the fitting notebook despite its location under `results/`; the notebook reads the table and does not reconstruct the original pit measurements. Generated prerequisites include `results/future-stock-growth.xlsx`, the compiled trade inputs, and the AOH catalogues used to build RWR. These paths should be checked before running dependent stages.

---

## 7. Section 0 — Data preparation

**Folder:** `0-data-preparation/`

These notebooks establish the common geospatial framework: a global Eckert IV equal-area grid at multiple resolutions, with administrative, conservation, and socioeconomic layers rasterized onto it. All outputs are compressed GeoTIFFs under `results/`.

| Notebook | Purpose | Key output |
|----------|---------|------------|
| `1-make-country-raster` | Rasterize GADM country boundaries (253 countries + Hong Kong + Macau = 255 units) | Country-ID rasters at 1, 2, 5, 10, 30, 50, 100 km |
| `2-make-economy-raster` | Rasterize the 142 cement-reporting economies (parent–child country merges) | Economy-ID rasters at 1, 10, 50 km + `pseudo-codes-economies-142.csv` |
| `3-make-protected-area-raster` | Rasterize 283,894 WDPA polygons (Jul 2024) by IUCN category | Merged + per-category masks (1 km) and 50 km aggregates |
| `4-make-biome-raster` | Rasterize WWF TEOW 14-biome classification; **`Fig-1.ipynb` reads these layers** | `biome-raster-eck4-*` at 1, 10, 50 km |
| `4-make-biome-raster-2017` | Rasterize alternative RESOLVE Ecoregions 2017 layers (847 ecoregions → 14 biomes) | `resolve-biome-raster-eck4-*` at 1, 2, 5, 10, 50 km |
| `5-make-income-group-raster` | Rasterize World Bank income groups (L / LM / UM / H) | Income-group raster at 1 km |
| `8-make-region-mask` | Define 15 biogeographic regions + USA/Canada/China state-province masks | `region-mask-*` (1/10/50 km) + `state-raster-eck4-1km` |
| `9-reproject-GDP-data` | Reproject total GDP per grid cell (2020 + 2025–2050 under SSP1–5), preserving sums; per-capita GDP is derived with population in the stock analysis | GDP rasters at 1 km |
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

The production BAU/CE and lever scenarios use **SSP2**, the default `ssp` argument of `run_projection()`. The SSP1–5 stock trajectories in Supplementary Figure 7 are a separate comparison, not five socioeconomic pathways crossed with every φ/β combination.

### Core modules

| File | Description |
|------|-------------|
| `IMAGINE_Concrete.py` | Core MFA engine. Key functions: `run_historical()` (1931–2022 baseline), `run_projection()` (2023–2050 per scenario, with CE-lever and φ-ambition parameterization), `fit_logistic_with_constraints()` (4-parameter logistic stock projection), `cal_dem_rate_conditional()` (age-cohort demolition rates). Tracks 3 building sectors (residential, non-residential, civil) and 4 concrete strength classes (C15, C16–23, C24–35, C35+; the last uses `C35` in variable names). |
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
| `8-compile-results` | Export economy/year/material percentile summaries to `material-flows/results-from-IMAGINE-Concrete.xlsx` (extraction, in-use stock, outflows); summing these summaries does not produce global Monte Carlo percentiles |

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
| Sand | 207.3 Gt (190.5–224.7) | 160.6 Gt (147.5–174.2) | −22% |
| Gravel | 145.6 Gt (130.2–161.7) | 118.1 Gt (105.6–131.0) | −19% |
| Limestone (for cement) | 93.6 Gt (69.0–116.3) | 73.7 Gt (60.2–86.0) | −21% |
| **Total** | **446.5 Gt (389.7–502.7)** | **352.4 Gt (313.3–391.2)** | −21% |

Figure 1a and Figure 2a use the same economy-first reporting convention, restored on **11 October 2026**. For each economy, material and year, calculate the median and 2.5th/97.5th percentiles of 10,000 Monte Carlo outcomes, then sum matching percentile estimates across economies and materials. Figure 2a additionally sums across 2023–2050. For strategy reductions, calculate local percentiles of paired BAU-minus-strategy outcomes before summing. These aggregate ranges sum local percentile bounds; they are not 95% probability intervals of global totals.

The cumulative BAU estimate is **446.4723 Gt (389.6975–502.7322)**, reported as **446 Gt (390–503)**. CE is **352.3941 Gt (313.3472–391.2463)**, reported as **352 Gt (313–391)**. Component reuse reduces extraction by **16.7320 Gt (14.5316–18.9248)**, reported as **17 Gt (15–19)**. `results/Data-for-Fig-2a.csv` contains the plotted summaries; `results/Data-for-Fig-2a-economy-summaries.csv` and the Figure 2a Source Data detail rows provide the economy/material components. The panel overlays gray points for the 10,000 draw-wise global Monte Carlo outcomes, saved in `results/Data-for-Fig-2a-draws.csv`. These points show the global draw distribution; the bars and error bars use the economy-first summaries described above, so the error bars are not calculated as percentiles of the gray points.

---

## 9. Section 2 — Mining suitability model

**Folder:** `2-mining-suitability-model/`

A Random Forest classifier is trained on **136,666** globally distributed aggregate quarries to predict where future mining is likely to occur.

The quarry count refers to **polygons**. Training uses their rasterized 1 km presence cells plus sampled pseudo-absences; polygon counts and training-sample counts are different quantities.

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
| North America (highest regional mean AUC) | 0.895 | 0.602 |
| East Africa (lowest regional mean AUC) | 0.748 | 0.347 |

**Outputs:** `results/trained-models/rf-global.pkl` (model) and `rf-global-features.pkl` (feature list); `results/global-mining-suitability-1km.tif` (global suitability raster); per-region CV metrics in `results/cv_results_per_region_global_model.csv` and `results/figure-data/`.

The AUC summary above is the unweighted mean and sample standard deviation of the 73 rows in `cv_results_per_region_global_model.csv`, checked on 11 October 2026; it is not a pooled global AUC. Eastern Asia has three valid folds and each of the other 14 regions has five. The regional rows show fold means. The configuration table describes the final saved Random Forest; the training notebook also contains earlier validation configurations.

---

## 10. Section 3 — Mining area simulation

**Folder:** `3-mining-area-simulation/`

This section converts the extraction demands from Section 1, the suitability map from Section 2, and the reserve/lithology layers into spatially explicit mining footprints (km²) under the [φ×β scenario sweep](#3-scenario-and-uncertainty-framework).

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `1-downscale-extraction-demand` | Split extraction into stock turnover and stock growth: distribute turnover by existing building stock and growth by positive SSP2 stock increments; write 50 km L/M/H demand rasters for 2023–2025 and then 5-year periods through 2050 |
| `2-1-process-desert-map` | Reproject the GSDS30 sand/desert classification to 1 km; output fractional aeolian-sand coverage for exclusion |
| `2-2-estimate-mineable-land` | Apply fractional/categorical land exclusions and a single threshold τ = the **quarry-area-weighted P50** of observed-quarry suitability (0.8240 in the saved diagnostics) |
| `3-fit-extraction-area-relationship` | Fit the power law A = 9.421·E^0.548 (n = 113 quarries; E in Mt, A in ha). The 95% confidence band of the fitted mean relationship is propagated into the footprint calculation; an individual-quarry prediction band is also examined as a diagnostic |
| `4-estimate-aggregate-reserve` | Estimate per-cell reserve (Mt) = mineable area × lithology factor (thickness × density × recovery), at L/M/H levels |
| `5-1-bilateral-trade-seed` | Build the observed bilateral trade seed matrix (2018–2022 average) to preserve real trade corridors |
| `5-1-trade-gravity-matrix` | Calibrate country export/import propensities and the CEPII distance matrix used in allocation |
| `5-2-map-extraction` | Allocate to the lowest score = (1−β)·**normalized distance** + β·**normalized RWR**, with min–max normalization within candidate sets and two-stage trade; **writes `results/sweep-config.json`** |
| `5-3-cost-analysis` | Convert domestic transport work (Mt·km) to monetary cost; quantify the conservation premium and CE demand-reduction benefit across all scenario×β combinations |
| `6-estimate-mining-area` | Convert allocated extraction (Mt) into footprint area (km²) via the power law, distributing within 1 km sub-cells ranked by suitability×reserve; propagate D×R×A uncertainty |
| `_regen_b06_factorial.py` | Helper that regenerates the β=0.6 (CON/IAP) cells of the 27-combination D×R×A factorial by re-executing the relevant pipeline functions |

### Mineable land (shared base)

The mineable-land base is the **same for all scenarios** (see [Section 3 framework](#3-scenario-and-uncertainty-framework)). The exclusions leave approximately **109.46 million km² of available land**; applying the single quarry-area-weighted suitability threshold then retains approximately **3.82 million km² of mineable land** (3.5% of that available base). These are successive stages, not the total global land area. L/M/H reserve cases vary extractable thickness and recovery, not the land mask or threshold.

Spatial demand L/M/H rasters sum economy/year percentile trajectories for stock turnover and growth. They are deterministic demand settings for the spatial sensitivity analysis, not the combined-extraction summaries in Figure 2a. The domestic allocator searches a square window extending three 50 km cells in each direction (ten for Indonesia and the Philippines); it ranks candidates by Euclidean distance/RWR within that window. Reachable-surplus screening separately uses a ten-cell distance limit before trade allocation. The footprint calculation applies the fitted relationship to 1 km sub-cell contributions before summing to 50 km; negative lower-band areas are clipped to zero at the sub-cell stage.

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

The publication figure notebooks in this directory use the outputs of Sections 1–4. `_verify.py` is a legacy helper that still references the absent `Fig-3-new.ipynb`; it is not a current validation of all figure notebooks. Run the relevant notebooks with their required inputs to regenerate the analytical figures. **Extended Data Figures 1 and 4 and Supplementary Figure 15 are manuscript schematics without corresponding drawing notebooks in this repository.**

The main-figure and analytical Extended Data notebooks use `final_figure_export.py` to export PDF, PNG and JPG files at a maximum width of 180 mm, with editable TrueType text in the PDF. For Extended Data Figures 1 and 4, run `python 9-draw-figures/export_extended_data_schematics.py` from the repository root; it reads the included PDFs in `source data/figure artwork/` and writes the final exports to `figures/` (or the directory specified by `--output-dir`).

### Main figures

| Notebook | Content |
|----------|---------|
| `Fig-1` | BAU overview: (a) annual extraction, (b) mining footprint over RWR, (c) mining area by biome, (d) overlap by taxon/threat category, and (e) overlap map; panels c,d include 27 sensitivity outcomes |
| `Fig-2` | Individual circular-economy measures and combined CE — reductions in extraction, mining area, and mining-biodiversity overlap relative to BAU |
| `Fig-3` | Conservation siting (CON, β=0.6) vs BAU — conservation-priority/cost trade-offs, avoided-versus-transferred overlap, overlap-change map, and regional insets |
| `Fig-4` | BAU / CE / CON / IAP comparison — overlap with 27-combination uncertainty, the ΔOverlap field over the (φ, β) plane, and global sub-additivity across the same 27 uncertainty combinations |

### Extended-Data figures

| Notebook | Content |
|----------|---------|
| No notebook — Extended Data Figure 1 | Overall analytical framework schematic, including IMAGINE-Concrete |
| `Extended-Data-Figure-2` | Building-material stock distribution and its stratification by population density and GDP |
| `Extended-Data-Figure-3` | Global mining-suitability model output at 1 km |
| No notebook — Extended Data Figure 4 | Demand allocation and mining-area estimation schematic |
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
| No notebook — Supplementary Figure 15 | Species Area of Habitat workflow schematic, included in the final SI |
| `Supplementary-Figure-16` … `-20` | RWR maps for all vertebrates, mammals, birds, amphibians, reptiles (each: total + CR/EN/VU/NT) |
| `Supplementary-Figure-21` | International aggregate trade 2023–2050 (BAU) — balances, top traders, bilateral matrix |
| `Supplementary-Figure-22` | Derivation of the conservation-priority knee β_CON = 0.6 |
| `Supplementary-Figure-23` | Per-country mining-biodiversity-overlap waterfalls (BAU → CON) |

---

## 13. Key results summary

The headline metric is **mining-biodiversity overlap** in species-range equivalents. Extraction ranges sum economy/material/year percentile estimates (see Section 8); footprint and overlap ranges follow the spatial sensitivity calculations described in Section 3. Values are rounded from the Figure 2 tables and signpost tables in `results/biodiversity-risk/`, rechecked on **11 October 2026**. Percentage reductions use unrounded central values, so they need not equal percentages calculated from the displayed rounded overlap values.

| Indicator | Value |
|-----------|-------|
| Aggregate quarries identified (global, 2021) | 136,666 features, 19,142 km² |
| Quarry distribution | Europe 65,097, N. America 23,992, Asia 22,146, Africa 13,998, S. America 8,778, Oceania 1,809, C. America 846 |
| Cumulative aggregate extraction 2023–2050 (BAU) | 446.5 Gt (389.7–502.7) |
| — sand / gravel / limestone | 207.3 / 145.6 / 93.6 Gt |
| Cumulative extraction (CE) | 352.4 Gt (313.3–391.2), −21% vs BAU |
| Additional mining footprint, 2023–2050 (BAU) | ~10,160 km² (7,089–16,336) |
| Mining footprint (CE) | 8,455 km² (5,984–13,307), −17% vs BAU |
| **Total mining-biodiversity overlap (BAU)** | **2.6 species-range equivalents (1.9–4.6)** |
| Overlap (CE) | 2.3 (1.7–3.7), −14% vs BAU |
| Overlap (CON) | 1.3 (0.9–2.3), ~−50% vs BAU |
| **Overlap (IAP)** | **1.1 (0.8–1.9), −58% vs BAU** |
| Suitability-model AUC (mean ± SD over 73 region/fold rows) | 0.832 ± 0.085 |
| Vertebrate species assessed (with AOH; manuscript count) | 29,327 |

---

## 14. Citation

If you use this code or data, please cite:

> Wu, T., Cao, Z., Torres, A., et al. (2026). *Future biodiversity risks of land-sourced aggregate mining.*

Journal and DOI information will be added once confirmed.

For questions or issues, please open a GitHub issue in this repository.
