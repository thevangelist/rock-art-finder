#!/bin/bash
# Rock Art Finder — setup and run

echo "=== Installing dependencies ==="
pip3 install geopandas rasterio shapely requests folium numpy scipy elevation --quiet

echo ""
echo "=== Fetching known rock art sites from Museovirasto ==="
python3 fetch_sites.py

echo ""
echo "=== Scoring candidate locations ==="
python3 score.py

echo ""
echo "=== Opening map ==="
open map.html
