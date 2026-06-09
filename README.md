# Mapping future biodiversity risks of global construction aggregate mining

*Code and analysis pipeline for the manuscript by Wu, Cao, Torres et al. (2026)*

This repository contains the full computational workflow for projecting global construction aggregate (sand, gravel, and limestone) mining expansion through 2050 and assessing its impacts on vertebrate biodiversity. The pipeline integrates material flow analysis, machine-learning-based suitability modeling, spatial planning, and species richness mapping into a single end-to-end framework.

---

## Table of contents

1. [Repository structure](#1-repository-structure)
2. [Workflow overview](#2-workflow-overview)
3. [Quarry data extraction from OpenStreetMap](#3-quarry-data-extraction-from-openstreetmap)
4. [Data acquisition](#4-data-acquisition)
5. [Section 0 — Data preparation](#5-section-0--data-preparation)
6. [Section 1 — IMAGINE-Concrete: material flow analysis](#6-section-1--imagine-concrete-material-flow-analysis)
7. [Section 2 — Mining suitability model](#7-section-2--mining-suitability-model)
8. [Section 3 — Mining area simulation](#8-section-3--mining-area-simulation)
9. [Section 4 — Biodiversity risk assessment](#9-section-4--biodiversity-risk-assessment)
10. [Section 9 — Figures](#10-section-9--figures)
11. [Key results summary](#11-key-results-summary)
12. [Citation](#12-citation)

---

## 1. Repository structure

```
aggregate-mining-biodiversity/
│
├── 0-data-preparation/           # Rasterize base layers (countries, PAs, biomes, …)
│   ├── 1-make-country-raster.ipynb
│   ├── 2-make-economy-raster.ipynb
│   ├── 3-make-protected-area-raster.ipynb
│   ├── 4-make-biome-raster.ipynb
│   ├── 5-make-income-group-raster.ipynb
│   ├── 8-make-region-mask.ipynb
│   ├── 9-reproject-GDP-data.ipynb
│   ├── 9-reproject-population-grids.ipynb
│   └── 10-reproject-building-mineral-grid.ipynb
│
├── 1-IMAGINE-Concrete/           # Material flow analysis & extraction demand
│   ├── IMAGINE_Concrete.py           ← core MFA module
│   ├── 1-get-trade-data.ipynb
│   ├── 2-compile-trade-data.ipynb
│   ├── 3-determine-future-stock.ipynb
│   ├── 4-run-simulations.ipynb
│   └── 5-check-results.ipynb
│
├── 2-mining-suitability-model/   # Random Forest suitability classifier
│   ├── 1-filter-aggregate-quarries.ipynb
│   ├── 2-rasterize-quarry-area.ipynb
│   ├── 3-1-forest.ipynb  …  3-9-water-mask.ipynb   ← feature engineering (9 notebooks)
│   ├── 4-train-models.ipynb
│   ├── 5-predict-mining-suitability.ipynb
│   ├── 6-sample-effect-test.ipynb
│   ├── Extended-Data-Fig-3.ipynb
│   └── Supplementary-Figure-10.ipynb
│
├── 3-mining-area-simulation/     # Spatial demand allocation & area projection
│   ├── 0-1-trade-gravity-matrix.ipynb
│   ├── 0-2-desert-map.ipynb
│   ├── 1-downscaling-extraction-demand.ipynb
│   ├── 2-estimate-available-land-for-mining.ipynb
│   ├── 3-fit-extraction-area-relationship.ipynb
│   ├── 4-estimating-extraction-capacity.ipynb
│   ├── 5-1-demand-capacity-allocation.ipynb
│   ├── 5-2-strategic-planning.ipynb
│   └── 6-mining-area-estimation.ipynb
│
├── 4-biodiversity-risk-assessment/   # Species range mapping & richness
│   ├── 1-make-range-raster-for-amphibians.ipynb
│   ├── 1-make-range-raster-for-birds.ipynb
│   ├── 1-make-range-raster-for-mammals.ipynb
│   ├── 1-make-range-raster-for-reptiles.ipynb
│   ├── 2-collect-IUCN-assessments.ipynb
│   ├── 3-compile-assessment-reports.ipynb
│   ├── 4-resample-habitat-class-map-1.ipynb
│   ├── 4-resample-habitat-class-map-2.ipynb
│   ├── 5-make-AOH-10km.ipynb
│   ├── 5-make-AOH-50km.ipynb
│   ├── 5-make-AOH-standard.ipynb
│   ├── 6-make-species-richness-maps.ipynb
│   ├── 7-make-rarity-richness-maps.ipynb
│   ├── 8-display-species-richness-maps.ipynb
│   ├── 9-species-richness-comparison.ipynb
│   ├── biodiversity-impact-assessment.ipynb
│   └── make-RWR-maps.ipynb
│
├── 9-draw-figures/               # Publication figures
│   ├── Extended-Data-Figure-2.ipynb
│   ├── Supplementary-Figure-1.ipynb
│   ├── Supplementary-Figure-3.ipynb
│   ├── Supplementary-Figure-7.ipynb
│   └── Supplementary-Figures-4-6.ipynb
│
├── data/                         # Input data (download separately — see Section 4)
├── results/                      # All intermediate and final outputs (GeoTIFFs, CSVs, pickles)
├── figures/                      # Publication-ready figures
└── README.md
```

---

## 2. Workflow overview

The five analytical sections are largely sequential, though the biodiversity assessment (Section 4) can be run independently. The recommended execution order is:

```
Section 0  →  Section 1  ─┐
                            ├→  Section 3  →  Section 4  →  Section 9
Section 2  ────────────────┘
```

| Section | Role | Key output |
|---------|------|------------|
| 0 — Data preparation | Rasterize administrative, conservation, and socioeconomic boundaries | Base-layer GeoTIFFs at 1 km, 10 km, 50 km resolutions in Eckert IV |
| 1 — IMAGINE-Concrete | Project extraction demand for sand, gravel, and limestone to 2050 | Scenario-specific annual demand estimates (Gt) for 142 economies |
| 2 — Mining suitability | Train and apply a global Random Forest classifier | 1 km global suitability raster (probability 0–1) |
| 3 — Mining area simulation | Allocate demand spatially, estimate mining footprint | Mining area rasters (BAU and CON scenarios, 2023–2050) |
| 4 — Biodiversity assessment | Overlay mining projections on vertebrate species ranges | Species richness and rarity-weighted richness impact maps |
| 9 — Figures | Generate all publication figures | Multi-panel maps and charts |

**Spatial reference system used throughout:** Eckert IV equal-area projection (ESRI:54012), with primary analysis at 1 km resolution and aggregated outputs at 10 km, 30 km, 50 km, and 100 km.

---

## 3. Quarry data extraction from OpenStreetMap

The quarry training dataset is built from an OpenStreetMap snapshot dated **2021-01-01**, chosen to align temporally with other input layers (land cover, population, protected areas, all approximately year 2020). Using a fixed snapshot ensures full reproducibility.

### 3.1 Install osmium-tool

```bash
sudo apt update
sudo apt install osmium-tool
```

### 3.2 Download OSM regional extracts

Download the regional `.osm.pbf` files from the [Geofabrik raw directory index](https://download.geofabrik.de/) and place them in your working directory:

- https://download.geofabrik.de/africa-210101.osm.pbf
- https://download.geofabrik.de/asia-210101.osm.pbf
- https://download.geofabrik.de/australia-oceania-210101.osm.pbf
- https://download.geofabrik.de/central-america-210101.osm.pbf
- https://download.geofabrik.de/europe-210101.osm.pbf
- https://download.geofabrik.de/north-america-210101.osm.pbf
- https://download.geofabrik.de/south-america-210101.osm.pbf

> Older Geofabrik snapshots are accessible via the [raw directory index](https://download.geofabrik.de/) (click "raw directory index" on each region's page).

### 3.3 Filter quarry features

Run the following loop to extract features tagged as `landuse=quarry` from each regional file. The output will be saved as `<region>-quarries.osm.pbf`.

```bash
for f in *-210101.osm.pbf; do
  echo "Processing $f..."
  osmium tags-filter "$f" wr/landuse=quarry -o "${f%.osm.pbf}-quarries.osm.pbf"
done
```

### 3.4 Convert to GeoPackage format

We use [GDAL](https://gdal.org/)'s `ogr2ogr` utility to convert each file into a [GeoPackage](https://www.geopackage.org/) (`.gpkg`), extracting only the `multipolygons` layer.

```bash
sudo apt install gdal-bin

for f in *-quarries.osm.pbf; do
  ogr2ogr -f "GPKG" "${f%.osm.pbf}.gpkg" "$f" multipolygons -where "landuse='quarry'"
done
```

### 3.5 Merge into a single global file

```bash
ogrmerge.py -f GPKG -o global-quarries-210101.gpkg -single -nln quarries *-quarries.gpkg
```

**Flags:** `-single` merges all inputs into one layer; `-nln quarries` names the merged layer `quarries`.

The result, `global-quarries-210101.gpkg`, serves as the primary input for the mining suitability model in [Section 7](#7-section-2--mining-suitability-model).

---

## 4. Data acquisition

All input datasets are openly available. Because of GitHub file-size limits, the `data/` folder is intentionally empty — download each dataset and place it in `data/` following the structure shown in Section 4.2.

### 4.1 Dataset list

| Dataset | Source | Used in |
|---------|--------|---------|
| Global cement production | [Andrew (2025)](https://zenodo.org/records/14931651) | IMAGINE-Concrete |
| International trade data | [UN Comtrade Database](https://comtradeplus.un.org/) | IMAGINE-Concrete |
| Global population grid (current) | [WorldPop](https://hub.worldpop.org/doi/10.5258/SOTON/WP00647) | Mining area simulation |
| Future population grid (SSP2) | [Wang et al. (2022)](https://doi.org/10.6084/m9.figshare.19608594.v3) | Mining area simulation |
| Building material stocks | [WSF3D Material Stock](https://download.geoservice.dlr.de/WSF3D_MATSTOCK/files/MaterialStockGlobal/) | Mining area simulation |
| OpenStreetMap regional extracts | [Geofabrik](https://download.geofabrik.de/) | Mining suitability |
| Travel time to cities | [Oxford MAP (2015)](https://developers.google.com/earth-engine/datasets/catalog/Oxford_MAP_accessibility_to_cities_2015_v1_0) | Mining suitability |
| Distance to freshwater | [Kummu et al.](https://zenodo.org/records/4956396) | Mining suitability |
| GMTED2010 elevation | [USGS](https://developers.google.com/earth-engine/datasets/catalog/USGS_GMTED2010_FULL) | Mining suitability |
| Global Lithological Map (GLiM) | [Hartmann & Moosdorf](https://doi.pangaea.de/10.1594/PANGAEA.788537) | Mining suitability |
| World Database on Protected Areas | [Protected Planet](https://www.protectedplanet.net/en/thematic-areas/wdpa) | Spatial planning |
| Global Lakes and Wetlands Database | [GLWD v2](https://figshare.com/articles/dataset/Global_Lakes_and_Wetlands_Database_GLWD_version_2_0/28519994) | Mining suitability |
| Global forest cover | [Hansen et al.](https://developers.google.com/earth-engine/datasets/catalog/UMD_hansen_global_forest_change_2024_v1_12) | Spatial planning |
| Global cropland extent | [Potapov et al.](https://doi.org/10.5281/zenodo.12527546) | Spatial planning |
| ESA CCI Land Cover (urban / bare land) | [ESA CCI](http://maps.elie.ucl.ac.be/CCI/viewer/download.php) | Mining suitability |
| IUCN species range polygons | [IUCN Red List](https://www.iucnredlist.org/resources/spatial-data-download) | Biodiversity assessment |
| Global terrestrial habitat map | [Jung et al.](https://uploads.users.earthengine.app/view/habitat-types-map) | Biodiversity assessment |
| Country administrative boundaries | [GADM](https://gadm.org/) | All sections |
| Desert classification | [Zenodo](https://zenodo.org/records/10441041) | Mining area simulation |
| Trade gravity data | [CEPII Gravity (v202211)](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8) | Mining area simulation |

### 4.2 Expected folder structure

```
data/
├── Gravity_csv_V202211/          # CEPII gravity trade data (142 economies)
├── GSDS30/                       # Geological suitability dataset
├── China-map/                    # China province boundaries
├── simplified-world-map/         # World administrative boundaries
├── production-142.csv            # Cement production by economy
├── reporter-with-economy-142.csv # Economy metadata for 142 reporters
├── country-pseudo-values.csv     # Country ISO to raster pseudo-code mapping
└── [IUCN species ranges, habitat map, WDPA, etc.]
```

---

## 5. Section 0 — Data preparation

**Folder:** `0-data-preparation/`

These notebooks establish the common geospatial framework used throughout the study: a global Eckert IV equal-area grid at multiple resolutions. All rasters are stored as compressed GeoTIFFs in `results/`.

| Notebook | Purpose | Key output |
|----------|---------|------------|
| `1-make-country-raster` | Rasterize GADM country boundaries (253 countries + HK + Macao) | Country-ID rasters at 1, 2, 5, 10, 30, 50, 100 km |
| `2-make-economy-raster` | Rasterize the 142 cement-reporting economies | Economy-ID rasters at 1, 10, 50 km |
| `3-make-protected-area-raster` | Rasterize WDPA polygons by IUCN category (Ia–VI) | PA category rasters + individual category binary masks |
| `4-make-biome-raster` | Rasterize WWF TEOW 14-biome classification | Biome rasters at 1, 10, 50 km |
| `5-make-income-group-raster` | Rasterize World Bank income group classifications | Income group rasters |
| `8-make-region-mask` | Define 15 biogeographic regions for stratified modeling | Region-mask rasters |
| `9-reproject-GDP-data` | Reproject GDP per capita grids into Eckert IV | GDP rasters at 50 km |
| `9-reproject-population-grids` | Reproject WorldPop current and SSP2 future population grids | Population rasters (current + 2025–2050) |
| `10-reproject-building-mineral-grid` | Reproject WSF3D building material stock density | Building mineral rasters at 50 km |

**Coordinate system details:**
- Projection: Eckert IV (ESRI:54012)
- Primary resolution: 1 km (1,000 m × 1,000 m pixels)
- Aggregated resolutions: 10 km, 30 km, 50 km, 100 km
- WDPA pseudo-codes: Ia = 8, Ib = 7, II = 6, III = 5, IV = 4, V = 3, VI = 2

---

## 6. Section 1 — IMAGINE-Concrete: material flow analysis

**Folder:** `1-IMAGINE-Concrete/`

IMAGINE-Concrete is a global material flow analysis (MFA) model that projects extraction demand for construction aggregates (sand, gravel, limestone) through 2050, driven by building stock dynamics across 142 economies.

### Core module: `IMAGINE_Concrete.py`

| Function | Description |
|----------|-------------|
| `run_historical()` | Simulate material flows 1931–2022 with probabilistic uncertainty |
| `fit_logistic_with_constraints()` | Fit a 4-parameter logistic curve to project building stocks to 2050 |
| `cal_dem_rate_conditional()` | Compute age-cohort demolition rates for stock turnover |

**Uncertainty:** 10,000 Monte Carlo simulations propagate parameter uncertainty through the entire model chain.

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `1-get-trade-data` | Download bilateral trade flows from UN Comtrade API for cement, clinker, limestone, sand, and gravel |
| `2-compile-trade-data` | Organize trade flows into economy-level import/export matrices (10 CSV files in `results/trade-data/`) |
| `3-determine-future-stock` | Fit logistic growth curves to historical building-stock data; project stocks to 2050 per economy |
| `4-run-simulations` | Execute MFA for all 142 economies under 8 scenarios; decompose demand into stock-growth and stock-turnover components |
| `5-check-results` | Validate and summarize results; produce scenario comparison tables |

### Demand scenarios

| Scenario | Description |
|----------|------------|
| BAU | Business-as-Usual — current trends continue |
| MIU | Material intensity reduction |
| LE | Lifetime extension of buildings |
| RE | Material re-use |
| RC | Recycled aggregates partially substitute virgin materials |
| SS | Sufficiency-focused building stock |
| CCR | Combined circular economy measures |
| CE | Full circular economy transition |

### Key demand estimates (2023–2050 cumulative)

| Material | BAU | CE (low-demand) |
|----------|-----|-----------------|
| Sand | ~381 Gt | ~287 Gt |
| Gravel | ~247 Gt | ~205 Gt |
| Limestone | ~161 Gt | ~125 Gt |

---

## 7. Section 2 — Mining suitability model

**Folder:** `2-mining-suitability-model/`

A Random Forest classifier is trained on 136,666 globally distributed construction aggregate quarries to predict where future mining is likely to occur, using 34 environmental, accessibility, geological, and land-use features.

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `1-filter-aggregate-quarries` | Remove 11,548 non-aggregate quarries (from Global Mining Atlas) from the 148,214 OSM features; retain 136,666 aggregate quarries (19,142 km²) |
| `2-rasterize-quarry-area` | Convert quarry polygons to a 1 km binary presence raster |
| `3-1` through `3-9` | Feature engineering: prepare 34 raster layers at 1 km resolution |
| `4-train-models` | Train the Random Forest model with stratified cross-validation |
| `5-predict-mining-suitability` | Apply the trained model globally; output 1 km suitability raster |
| `6-sample-effect-test` | Sensitivity analysis on pseudo-absence sampling strategy |
| `Extended-Data-Fig-3` | Regional performance visualization |
| `Supplementary-Figure-10` | Additional model diagnostics |

### Training data

| Item | Value |
|------|-------|
| Total samples | 601,390 |
| Presence (quarries) | 195,765 |
| Pseudo-absences | 405,625 |
| Absence sampling | Two-tier: buffer ring 30–300 km + country background |
| Biogeographic regions | 15 |
| Country-level groups | 312 |

### Features (34 total)

| Category | Features |
|----------|----------|
| Topography | Elevation (GMTED2010), slope |
| Accessibility | Population density, travel time to nearest city |
| Land use | Forest cover, cropland extent, urban area, bare land |
| Water | Distance to freshwater (Kummu et al.), water body mask (GLWD) |
| Geology | 8 GLiM rock-type classes (su, vb, ss, pb, sm, sc, va, mt) |
| Protected areas | 7 IUCN category masks (Ia–VI + unprotected) |
| Built environment | Building mineral stock density (WSF3D) |

### Model configuration & performance

| Hyperparameter | Value |
|---------------|-------|
| Algorithm | Random Forest |
| Trees | 400 |
| Max depth | 20 |
| Min samples per leaf | 5 |
| Validation | 5-fold GroupKFold (grouped by country) |

| Region | AUC | F1 | κ |
|--------|-----|----|---|
| Global (mean ± SD) | 0.833 ± 0.085 | 0.599 ± 0.136 | 0.442 |
| North America (best) | 0.895 | — | 0.602 |
| East Africa (weakest) | 0.748 | — | 0.347 |

**Output:** `results/trained-models/rf-global.pkl` — serialized Random Forest model; `results/mining-suitability/` — global 1 km suitability raster.

---

## 8. Section 3 — Mining area simulation

**Folder:** `3-mining-area-simulation/`

This section combines extraction demands from Section 1 with the suitability map from Section 2 and conservation constraints to project the spatial footprint of mining expansion through 2050.

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `0-1-trade-gravity-matrix` | Build bilateral trade flow matrices using the CEPII gravity model; used to distribute demand between importing and exporting economies |
| `0-2-desert-map` | Create a binary desert mask (from Zenodo) to exclude hyper-arid areas from available land |
| `1-downscaling-extraction-demand` | Downscale national extraction demands to 50 km pixel level using building material stock density as a spatial proxy; produces rasters for all materials and scenarios with uncertainty bounds (2.5th, 50th, 97.5th percentile) |
| `2-estimate-available-land-for-mining` | Compute available land under BAU and CON planning scenarios (see below) |
| `3-fit-extraction-area-relationship` | Fit empirical relationships between extraction volume and mining area using observed quarry data |
| `4-estimating-extraction-capacity` | Estimate regional extraction capacity from suitability scores and available land |
| `5-1-demand-capacity-allocation` | Allocate global demand across pixels weighted by suitability, within regional capacity constraints |
| `5-2-strategic-planning` | Apply strategic constraints to minimize impacts on protected areas and biodiversity hotspots |
| `6-mining-area-estimation` | Compute final mining area estimates (km²) per scenario and year, with uncertainty ranges |

### Land availability scenarios

| Scenario | Excluded from mining |
|----------|---------------------|
| BAU | Urban areas, water bodies, IUCN categories Ia–IV protected areas |
| CON (Conservation) | Urban, water, forest, cropland, all IUCN categories (Ia–VI) protected areas, deserts |

---

## 9. Section 4 — Biodiversity risk assessment

**Folder:** `4-biodiversity-risk-assessment/`

Species range maps and habitat suitability data are combined to quantify how projected mining expansion affects global vertebrate biodiversity, focusing on species richness and the concentration of geographically restricted (rare) species.

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `1-make-range-raster-for-[taxon]` (×4) | Rasterize IUCN range polygons to 1 km for amphibians (7,640 spp.), birds (~10,364 spp.), mammals (~5,237 spp.), reptiles (~7,948 spp.); store with bounding-box indices for memory efficiency |
| `2-collect-IUCN-assessments` | Query the IUCN Red List API for habitat and threat assessment data for each species |
| `3-compile-assessment-reports` | Parse assessment JSON files into structured CSV threat databases |
| `4-resample-habitat-class-map-1/2` | Reproject and resample the Jung et al. global habitat classification to match the species range rasters |
| `5-make-AOH-[10km|50km|standard]` | Compute Area of Habitat (AOH) = species range ∩ suitable habitat classes; output bounding-box-optimized rasters at three resolutions |
| `6-make-species-richness-maps` | Sum species presence layers to create richness maps by taxon and IUCN threat category (CR, EN, VU, NT, LC) |
| `7-make-rarity-richness-maps` | Weight species by 1/AOH to emphasize range-restricted species; produce rarity-weighted richness maps |
| `8-display-species-richness-maps` | Visualize global biodiversity patterns |
| `9-species-richness-comparison` | Validate richness maps against published datasets |
| `biodiversity-impact-assessment` | Overlay mining projections on richness and rarity layers; quantify affected species and area |
| `make-RWR-maps` | Generate rarity-weighted richness maps for all taxa combined |

### Biodiversity metrics

| Metric | Definition |
|--------|------------|
| Species richness | Number of species present per grid cell |
| Rarity-weighted richness | Σ (1 / AOH_species) per cell — emphasizes geographically restricted species |
| AOH | Area of Habitat: intersection of species range and suitable habitat classes |

### Key biodiversity statistics (at 50 km resolution)

| Metric | Value |
|--------|-------|
| Total vertebrate species covered | ~31,000 |
| Max total vertebrate richness per cell | 1,230 species |
| Max critically endangered richness per cell | 217 species |
| Max endangered richness per cell | 768 species |

---

## 10. Section 9 — Figures

**Folder:** `9-draw-figures/`

Publication-quality figures are generated from the outputs of Sections 1–4.

| Notebook | Content |
|----------|---------|
| `Extended-Data-Figure-2` | Global mining suitability map with regional insets |
| `Supplementary-Figure-1` | Quarry distribution and training data coverage |
| `Supplementary-Figure-3` | Scenario comparison of extraction demands |
| `Supplementary-Figure-7` | Biodiversity richness hotspot maps |
| `Supplementary-Figures-4-6` | Mining area projections under BAU and CON scenarios |

---

## 11. Key results summary

| Indicator | Value |
|-----------|-------|
| Aggregate quarries identified (global, 2021) | 136,666 features, 19,142 km² |
| Quarry distribution | Europe 65k, N. America 24k, Asia 22k, Africa 14k |
| Sand demand 2023–2050 (BAU) | ~381 Gt |
| Sand demand 2023–2050 (CE scenario) | ~287 Gt |
| Gravel demand 2023–2050 (BAU) | ~247 Gt |
| Limestone demand 2023–2050 (BAU) | ~161 Gt |
| Suitability model AUC (global) | 0.833 ± 0.085 |
| Vertebrate species assessed | ~31,000 |
| Max vertebrate richness (50 km cell) | 1,230 species |

---

## 12. Citation

If you use this code or data, please cite:

> Wu, T., Cao, J., Torres, A., et al. (2026). *Mapping future biodiversity risks of global construction aggregate mining.* [Journal name]. [DOI].

For questions or issues, please open a GitHub issue in this repository.
