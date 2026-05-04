# Aggregate mining & biodiversity

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

### 1.4 XXX

```bash
sudo apt install gdal-bin
```


