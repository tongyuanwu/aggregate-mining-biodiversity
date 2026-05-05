# Mapping future biodiversity risks of global construction aggregate mining
*Code and analysis pipeline for the manuscript by Wu, Cao, Torres et al. (2026) *  
This is ...

## 1. Quarry data extraction from OpenStreetMap

### 1.1 Install osmium-tool

`osmium-tool` is required to filter quarry features from OpenStreetMap (OSM) data.

```bash
sudo apt update
sudo apt install osmium-tool
```

### 1.2 Download OSM regional extracts

Download the regional `.osm.pbf` files from the [Geofabrik raw directory index](https://download.geofabrik.de/) and place them in your working directory:

- https://download.geofabrik.de/africa-210101.osm.pbf
- https://download.geofabrik.de/asia-210101.osm.pbf
- https://download.geofabrik.de/australia-oceania-210101.osm.pbf
- https://download.geofabrik.de/central-america-210101.osm.pbf
- https://download.geofabrik.de/europe-210101.osm.pbf
- https://download.geofabrik.de/north-america-210101.osm.pbf
- https://download.geofabrik.de/south-america-210101.osm.pbf


> **Note on the `210101` version:** We use the OpenStreetMap snapshot dated **2021-01-01** to ensure temporal alignment with other spatial datasets used in this study (e.g., population, land cover, and protected area layers, which are largely available for the year 2020). Using a fixed snapshot also ensures reproducibility, as OSM is continuously updated and live downloads would yield different results over time. Older Geofabrik snapshots are accessible via the [raw directory index](https://download.geofabrik.de/) (click "raw directory index" on each region's page).

### 1.3 Filter quarry features

Run the following loop to extract features tagged as `landuse=quarry` from each regional file. The output will be saved as `<region>-quarries.osm.pbf`.

```bash
for f in *-210101.osm.pbf; do
  echo "Processing $f..."
  osmium tags-filter "$f" wr/landuse=quarry -o "${f%.osm.pbf}-quarries.osm.pbf"
done
```

### 1.4 Convert to GeoPackage format
The filtered `.osm.pbf` files contain quarry features as raw OSM elements. To make them easier to work with in GIS software (e.g., QGIS) and Python (e.g., `geopandas`), we convert each file into a [GeoPackage](https://www.geopackage.org/) (`.gpkg`) file.

We use [GDAL](https://gdal.org/)'s `ogr2ogr` utility for the conversion, extracting only the `multipolygons` layer (where quarry boundaries are stored) and keeping features tagged with `landuse=quarry`.

#### Install GDAL
```bash
sudo apt install gdal-bin
```

#### Convert each regional file
```bash
for f in *-quarries.osm.pbf; do
  ogr2ogr -f "GPKG" "${f%.osm.pbf}.gpkg" "$f" multipolygons -where "landuse='quarry'"
done
```


### 1.5 Merge into a single global file

To simplify downstream processing, we merge all regional GeoPackages into a single global file containing every quarry polygon worldwide. This uses GDAL's `ogrmerge.py` utility, which combines multiple vector files into one.

```bash
ogrmerge.py -f GPKG -o global-quarries-210101.gpkg -single -nln quarries *-quarries.gpkg
```

**Flags explained:**
- `-f GPKG` — output format is GeoPackage
- `-o global-quarries-210101.gpkg` — output filename
- `-single` — merge all inputs into a single layer (instead of one layer per input file)
- `-nln quarries` — name the merged layer `quarries`

The result, `global-quarries-210101.gpkg`, contains all quarry polygons globally and serves as the input for the mining-suitability model in [Section 2](#2-mining-suitability-modeling).

## 2. Data acquisition

All input datasets used in this study are openly available. Due to file size limits on GitHub, the `data/` folder in this repository is intentionally empty — please download each dataset from the sources listed below and place them in `data/` following the structure shown.

### 2.1 Dataset list

| Dataset | Source | Used in |
|---|---|---|
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
| ESA CCI Land Cover (urban/bare) | [ESA CCI](http://maps.elie.ucl.ac.be/CCI/viewer/download.php) | Mining suitability |
| IUCN species range polygons | [IUCN Red List](https://www.iucnredlist.org/resources/spatial-data-download) | Biodiversity assessment |
| Global terrestrial habitat map | [Jung et al.](https://uploads.users.earthengine.app/view/habitat-types-map) | Biodiversity assessment |
| Country administrative boundaries | [GADM](https://gadm.org/) | All modules |

### 2.2 Expected folder structure

After downloading, organize the files in `data/` as follows: