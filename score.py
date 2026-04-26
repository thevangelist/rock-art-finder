"""
Rock art likelihood scorer for Central Finland.

Scoring model based on known Finnish rock art site characteristics:
1. Elevation band — sites cluster in the ancient shoreline elevation range
2. Current proximity to water — sites are on lakeshores/cliffs
3. Slope — moderate to steep rock faces preferred
4. Clustering — new sites tend to be near known ones

Inputs:
  - known_sites.geojson   (from fetch_sites.py or manually placed)
  - Elevation: fetched from open-elevation API or local DEM

Output:
  - candidate_sites.geojson  (scored grid points)
  - map.html                 (interactive Folium map)
"""

import json
import math
import os
import gzip
import numpy as np
import folium
from folium.plugins import HeatMap
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point, shape
import warnings
warnings.filterwarnings("ignore")

# ── Study area (Central Finland / Jyväskylä–Hankasalmi–Konnevesi) ────────────
BBOX = {
    "min_lon": 25.3,
    "max_lon": 26.8,
    "min_lat": 61.55,
    "max_lat": 62.65,
}
GRID_STEP = 0.004          # ~300 m grid spacing

# Elevation band calibration from confirmed sites:
# Sites sit 5–40m above their current lake surface depending on the water body.
# Rather than one global band, we score based on distance above the nearest lake.
# Actual site elevations (asl): Saraakallio 83m, Toussunlinna 79m, Pyhänpää 101m, Uittovuori 116m
# Lake surface elevations in region: Päijänne ~77m, Saraavesi ~82m, Konnevesi ~95m, Keitele ~99m
# → sites sit roughly 2–20m above their lake surface
ELEV_ABOVE_LAKE_MIN = 1    # metres above local lake surface
ELEV_ABOVE_LAKE_MAX = 25   # metres above local lake surface (Pyhänpää ~24m above Päijänne)
ELEV_ABOVE_LAKE_PEAK = 8   # sweet spot — most sites here

# Approximate lake surface elevations (asl) for scoring
LAKE_SURFACES = [
    {"name": "Päijänne",    "elev": 77.5, "lat": 62.05, "lon": 25.55, "radius_km": 40},
    {"name": "Saraavesi",   "elev": 82.0, "lat": 62.48, "lon": 25.90, "radius_km": 10},
    {"name": "Hankavesi",   "elev": 82.0, "lat": 62.43, "lon": 26.24, "radius_km": 12},
    {"name": "Konnevesi",   "elev": 95.5, "lat": 62.57, "lon": 26.55, "radius_km": 15},
    {"name": "Keitele",     "elev": 99.0, "lat": 62.70, "lon": 26.20, "radius_km": 25},
    {"name": "Kynsivesi",   "elev": 82.0, "lat": 62.30, "lon": 26.10, "radius_km": 10},
    {"name": "Leppävesi",   "elev": 82.0, "lat": 62.37, "lon": 25.85, "radius_km": 8},
]

CLUSTER_RADIUS_KM = 20.0   # km — distance within which proximity bonus applies

# Water route corridors: sites tend to be ON travel routes between lake systems
# Scoring bonus for points near lake-to-lake narrows or straits
WATER_ROUTE_BONUS = 0.15

# Known rock art sites with confirmed coordinates
# access: "boat" = typically reachable only by water, critical for route scoring
MANUAL_KNOWN_SITES = [
    {
        "name": "Saraakallio I (Laukaa)",
        "lon": 25.9973, "lat": 62.4176,
        "access": "boat",
        "note": "One of Fennoscandia's largest rock art complexes. Hundreds of figures spanning thousands of years.",
        "elev_above_water": 2,
    },
    {
        "name": "Saraakallio II (Laukaa)",
        "lon": 25.9920, "lat": 62.4195,
        "access": "boat",
        "note": "Secondary site ~300m from Saraakallio I on Saraavesi shore.",
        "elev_above_water": 2,
    },
    {
        "name": "Toussunlinna (Laukaa / Hankavesi)",
        "lon": 26.2413, "lat": 62.4332,
        "access": "boat",
        "note": "Vertical cliff face rising straight from lake. Elk figure. Practically inaccessible except by water.",
        "elev_above_water": 5,
    },
    {
        "name": "Pyhänpää (Päijänne south)",
        "lon": 25.4745, "lat": 61.6418,
        "access": "mixed",
        "note": "Multiple painting groups on southern Päijänne shore. Paintings notably high above current water.",
        "elev_above_water": 15,
    },
    {
        "name": "Uittovuori (Konnevesi)",
        "lon": 26.5492, "lat": 62.5701,
        "access": "boat",
        "note": "On the water travel corridor between Keitele and Konnevesi — route site.",
        "elev_above_water": 3,
    },
    {
        "name": "Paanalansaari (Hankavesi area)",
        "lon": 26.2411, "lat": 62.3786,
        "access": "boat",
        "note": "Rock painting on Paanalansaari island. Approx coords from map search.",
        "elev_above_water": 3,
    },
    {
        "name": "Halsvuori (Laukaa)",
        "lon": 25.8124, "lat": 62.3699,
        "access": "mixed",
        "note": "Tourist attraction, rock painting.",
        "elev_above_water": 5,
    },
    {
        "name": "Kumpusaari (Konnevesi)",
        "lon": 26.6699, "lat": 62.5411,
        "access": "boat",
        "note": "Island site near Uittovuori on Konnevesi water route.",
        "elev_above_water": 3,
    },
    {
        "name": "Hakavuori",
        "lon": 25.7516, "lat": 61.9882,
        "access": "mixed",
        "note": "Rock painting site.",
        "elev_above_water": 5,
    },
]


def load_known_sites():
    try:
        with open("known_sites.geojson") as f:
            fc = json.load(f)
        pts = []
        for feat in fc["features"]:
            coords = feat["geometry"]["coordinates"]
            if feat["geometry"]["type"] == "Point":
                pts.append({"lon": coords[0], "lat": coords[1],
                            "name": feat.get("properties", {}).get("kohdenimi", "?")})
        if pts:
            print(f"Loaded {len(pts)} known sites from known_sites.geojson")
            return pts
    except FileNotFoundError:
        pass
    print("known_sites.geojson not found — using manually placed sites from screenshot")
    return MANUAL_KNOWN_SITES


_TILE_CACHE = {}

def load_tile_array(tile_lat, tile_lon):
    """Load and cache one SRTM1 tile as a float32 array. Decompresses once."""
    key = (tile_lat, tile_lon)
    if key in _TILE_CACHE:
        return _TILE_CACHE[key]
    fname = f"N{tile_lat:02d}E{tile_lon:03d}.hgt.gz"
    if not os.path.exists(fname):
        _TILE_CACHE[key] = None
        return None
    SIZE = 3601
    with gzip.open(fname, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.dtype('>i2')).reshape(SIZE, SIZE).astype(np.float32)
    raw[raw == -32768] = np.nan
    _TILE_CACHE[key] = raw
    return raw


def load_all_elevations(grid):
    """Vectorized elevation lookup: loads each tile once, then indexes all points at once."""
    print("  Decompressing SRTM tiles...")
    SIZE = 3601

    lats = np.array([p["lat"] for p in grid])
    lons = np.array([p["lon"] for p in grid])

    tile_lats = np.floor(lats).astype(int)
    tile_lons = np.floor(lons).astype(int)

    # Pre-load all unique tiles
    for tl, tlon in set(zip(tile_lats.tolist(), tile_lons.tolist())):
        tile = load_tile_array(tl, tlon)
        status = "ok" if tile is not None else "missing"
        print(f"    N{tl:02d}E{tlon:03d}: {status}")

    # Vectorized row/col calculation
    rows = np.clip(((tile_lats + 1 - lats) * (SIZE - 1)).astype(int), 0, SIZE - 1)
    cols = np.clip(((lons - tile_lons) * (SIZE - 1)).astype(int), 0, SIZE - 1)

    elevs = np.full(len(grid), np.nan)
    for tl, tlon in set(zip(tile_lats.tolist(), tile_lons.tolist())):
        tile = _TILE_CACHE.get((tl, tlon))
        if tile is None:
            continue
        mask = (tile_lats == tl) & (tile_lons == tlon)
        elevs[mask] = tile[rows[mask], cols[mask]]

    valid = int((~np.isnan(elevs)).sum())
    print(f"  {valid}/{len(grid)} valid elevation values")
    return elevs.tolist()


def load_slope_and_aspect(grid):
    """
    Compute slope steepness and aspect (facing direction) from SRTM tiles.
    Returns (slope_deg, aspect_deg) arrays — aspect 0=N, 90=E, 180=S, 270=W.
    Finnish rock art is on south/southwest facing steep cliff faces.
    """
    lats = np.array([p["lat"] for p in grid])
    lons = np.array([p["lon"] for p in grid])
    tile_lats = np.floor(lats).astype(int)
    tile_lons = np.floor(lons).astype(int)
    SIZE = 3601

    rows = np.clip(((tile_lats + 1 - lats) * (SIZE - 1)).astype(int), 1, SIZE - 2)
    cols = np.clip(((lons - tile_lons) * (SIZE - 1)).astype(int), 1, SIZE - 2)

    # Approximate cell size in metres at this latitude
    lat_mean = lats.mean()
    cell_m_ns = 111320.0 / (SIZE - 1)           # ~30m N-S
    cell_m_ew = cell_m_ns * math.cos(math.radians(lat_mean))

    slopes = np.full(len(grid), np.nan)
    aspects = np.full(len(grid), np.nan)

    for tl, tlon in set(zip(tile_lats.tolist(), tile_lons.tolist())):
        tile = _TILE_CACHE.get((tl, tlon))
        if tile is None:
            continue
        mask = (tile_lats == tl) & (tile_lons == tlon)
        r = rows[mask]
        c = cols[mask]
        # Central differences for gradient
        dz_ns = (tile[r-1, c].astype(float) - tile[r+1, c].astype(float)) / (2 * cell_m_ns)
        dz_ew = (tile[r, c+1].astype(float) - tile[r, c-1].astype(float)) / (2 * cell_m_ew)
        slope_rad = np.arctan(np.sqrt(dz_ns**2 + dz_ew**2))
        slopes[mask] = np.degrees(slope_rad)
        # Aspect: 0=N, 90=E, 180=S, 270=W
        aspect_rad = np.arctan2(dz_ew, dz_ns)
        aspects[mask] = (np.degrees(aspect_rad) + 360) % 360

    return slopes, aspects


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def nearest_lake_surface(lat, lon):
    """Return the surface elevation of the nearest lake body."""
    best = min(LAKE_SURFACES, key=lambda l: haversine_km(lat, lon, l["lat"], l["lon"]))
    return best["elev"]


def elevation_score(elev, lat=None, lon=None):
    """Score based on height above the nearest lake surface.
    Rock art sits 1–25m above water, peaking around 5–10m."""
    if elev is None or elev != elev:  # None or NaN
        return 0.3
    lake_elev = nearest_lake_surface(lat, lon) if lat else 82.0
    above = elev - lake_elev
    if above < ELEV_ABOVE_LAKE_MIN or above > ELEV_ABOVE_LAKE_MAX + 10:
        return 0.0
    if above < ELEV_ABOVE_LAKE_MIN:
        return 0.0
    # Triangle peak at ELEV_ABOVE_LAKE_PEAK, drops off either side
    if above <= ELEV_ABOVE_LAKE_PEAK:
        return above / ELEV_ABOVE_LAKE_PEAK
    else:
        return max(0.0, 1.0 - (above - ELEV_ABOVE_LAKE_PEAK) / (ELEV_ABOVE_LAKE_MAX - ELEV_ABOVE_LAKE_PEAK))


def proximity_score(lat, lon, known_sites):
    """Score based on distance to nearest known site."""
    if not known_sites:
        return 0.5
    min_dist = min(haversine_km(lat, lon, s["lat"], s["lon"]) for s in known_sites)
    # Peaks at 2–8 km from a known site (not on top of it, not too far)
    if min_dist < 0.5:
        return 0.2   # too close — likely same site
    elif min_dist < CLUSTER_RADIUS_KM:
        return 1.0 - (min_dist / CLUSTER_RADIUS_KM) * 0.4
    else:
        return max(0.0, 1.0 - (min_dist - CLUSTER_RADIUS_KM) / 30.0)


def build_grid():
    lats = np.arange(BBOX["min_lat"], BBOX["max_lat"], GRID_STEP)
    lons = np.arange(BBOX["min_lon"], BBOX["max_lon"], GRID_STEP)
    grid = [{"lat": float(lat), "lon": float(lon)} for lat in lats for lon in lons]
    print(f"Grid: {len(grid)} points ({len(lats)} × {len(lons)})")
    return grid


def cliff_score(slopes, aspects):
    """
    Score for steep, south/southwest facing cliff faces.
    - Steep (>25°) scores high for steepness
    - Facing 135–270° (SE to W, peaking at S=180°) scores high for aspect
    Finnish rock art almost always faces S or SW where sun hits the wall.
    """
    # Steepness: ramp up from 15° to 35°, cap at 1.0
    steep = np.clip((slopes - 15.0) / 20.0, 0.0, 1.0)

    # Aspect: gaussian centred on 202.5° (SSW), std ~60°
    # Wrap-safe angular difference
    diff = np.abs(((aspects - 202.5) + 180) % 360 - 180)
    facing = np.exp(-(diff**2) / (2 * 60.0**2))

    return steep * facing


def score_grid(grid, known_sites, elevations, slopes=None, aspects=None):
    """Fully vectorized scoring — runs in seconds on 57k points."""
    lats = np.array([p["lat"] for p in grid])
    lons = np.array([p["lon"] for p in grid])
    elevs = np.array([e if e is not None else np.nan for e in elevations])

    R = 6371.0
    def vec_haversine(lat1, lon1, lat2_arr, lon2_arr):
        dlat = np.radians(lat2_arr - lat1)
        dlon = np.radians(lon2_arr - lon1)
        a = np.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * np.cos(np.radians(lat2_arr)) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # ── Elevation score (vectorized) ──────────────────────────────────────────
    lake_lats = np.array([l["lat"] for l in LAKE_SURFACES])
    lake_lons = np.array([l["lon"] for l in LAKE_SURFACES])
    lake_elvs = np.array([l["elev"] for l in LAKE_SURFACES])

    # For each grid point find nearest lake
    lake_dists = np.stack([
        np.sqrt((lats - ll)**2 + (lons - lo)**2)   # cheap approx for nearest
        for ll, lo in zip(lake_lats, lake_lons)
    ], axis=1)
    nearest_lake_idx = np.argmin(lake_dists, axis=1)
    lake_surface = lake_elvs[nearest_lake_idx]
    above = elevs - lake_surface

    e_score = np.where(
        np.isnan(above) | (above < ELEV_ABOVE_LAKE_MIN) | (above > ELEV_ABOVE_LAKE_MAX + 10),
        0.0,
        np.where(
            above <= ELEV_ABOVE_LAKE_PEAK,
            above / ELEV_ABOVE_LAKE_PEAK,
            np.clip(1.0 - (above - ELEV_ABOVE_LAKE_PEAK) / (ELEV_ABOVE_LAKE_MAX - ELEV_ABOVE_LAKE_PEAK), 0, 1)
        )
    )

    # ── Proximity score (vectorized) ──────────────────────────────────────────
    site_lats = np.array([s["lat"] for s in known_sites])
    site_lons = np.array([s["lon"] for s in known_sites])
    site_dists = np.stack([
        vec_haversine(site_lats[i], site_lons[i], lats, lons)
        for i in range(len(known_sites))
    ], axis=1)
    min_dist = np.min(site_dists, axis=1)

    p_score = np.where(
        min_dist < 0.5, 0.2,
        np.where(
            min_dist < CLUSTER_RADIUS_KM,
            1.0 - (min_dist / CLUSTER_RADIUS_KM) * 0.4,
            np.clip(1.0 - (min_dist - CLUSTER_RADIUS_KM) / 30.0, 0, 1)
        )
    )

    # ── Approach/terrain score — local elevation std dev ─────────────────────
    # Use a simple gaussian smooth of elevation variance as proxy for roughness
    from scipy.ndimage import uniform_filter, generic_filter
    nlats = len(np.unique(lats))
    nlons = len(np.unique(lons))
    if not np.all(np.isnan(elevs)) and nlats * nlons == len(elevs):
        elev_grid = elevs.reshape(nlats, nlons)
        local_std = generic_filter(np.nan_to_num(elev_grid), np.std, size=3)
        a_score_grid = np.exp(-local_std**2 / 300.0)
        a_score = a_score_grid.ravel()
    else:
        a_score = np.full(len(grid), 0.5)

    # ── Water route score (vectorized) ────────────────────────────────────────
    corridors = [
        (62.48, 25.91), (62.43, 26.24), (62.57, 26.55),
        (62.15, 25.70), (62.40, 26.05), (62.30, 26.10),
    ]
    corr_dists = np.stack([
        vec_haversine(clat, clon, lats, lons)
        for clat, clon in corridors
    ], axis=1)
    min_corr = np.min(corr_dists, axis=1)
    w_score = np.where(
        min_corr < 3.0, WATER_ROUTE_BONUS,
        np.where(min_corr < 8.0, WATER_ROUTE_BONUS * (1 - (min_corr - 3.0) / 5.0), 0.0)
    )

    # ── Cliff aspect + steepness ──────────────────────────────────────────────
    if slopes is not None and aspects is not None:
        slopes_arr = np.array([s if s == s else 0.0 for s in slopes])
        aspects_arr = np.array([a if a == a else 0.0 for a in aspects])
        c_score = cliff_score(slopes_arr, aspects_arr)
    else:
        c_score = np.full(len(grid), 0.5)

    # ── Water mask — zero out points at or below lake surface ─────────────────
    water_mask = above <= 2.0
    e_score = np.where(water_mask, 0.0, e_score)

    # ── Combine ───────────────────────────────────────────────────────────────
    # Cliff score is additive (SRTM 30m resolution too coarse to gate on)
    total = np.clip(
        0.40*e_score + 0.20*c_score + 0.25*p_score + 0.10*a_score + 0.05*(1 + w_score),
        0, 1
    )

    results = []
    for i in range(len(grid)):
        elev_val = float(elevs[i]) if not np.isnan(elevs[i]) else None
        results.append({
            "lat": float(lats[i]),
            "lon": float(lons[i]),
            "elevation": round(elev_val, 1) if elev_val else None,
            "elev_score": round(float(e_score[i]), 3),
            "cliff_score": round(float(c_score[i]), 3),
            "prox_score": round(float(p_score[i]), 3),
            "approach_score": round(float(a_score[i]), 3),
            "route_bonus": round(float(w_score[i]), 3),
            "score": round(float(total[i]), 3),
        })
    return results


def build_map(scored, known_sites):
    center_lat = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
    center_lon = (BBOX["min_lon"] + BBOX["max_lon"]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10,
                   tiles="OpenStreetMap")

    # Add a satellite layer option
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Heatmap of candidate scores
    heat_data = [[r["lat"], r["lon"], r["score"]] for r in scored if r["score"] > 0.05]
    print(f"  Heatmap points: {len(heat_data)}")
    HeatMap(heat_data, radius=15, blur=25, min_opacity=0.05, max_opacity=0.45,
            gradient={0.0: "blue", 0.5: "lime", 0.75: "yellow", 1.0: "red"},
            name="Likelihood heatmap", show=False).add_to(m)

    # Top candidates as markers
    top = sorted(scored, key=lambda x: x["score"], reverse=True)[:30]
    candidate_group = folium.FeatureGroup(name="Top 30 candidates")
    for r in top:
        elev_str = f"{r['elevation']:.0f}m" if r['elevation'] else "?"
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=8,
            color="orange",
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Candidate #{top.index(r)+1}</b><br>"
                f"<b style='font-size:15px'>{round(r['score']*100)}% likelihood</b><br>"
                f"{'🟢 High' if r['score']>=0.7 else '🟡 Medium' if r['score']>=0.45 else '🔴 Low'}<br><br>"
                f"Elevation: {elev_str}<br>"
                f"Elev score: {r['elev_score']:.2f} · Cliff: {r.get('cliff_score',0):.2f}<br>"
                f"Proximity: {r['prox_score']:.2f} · Approach: {r.get('approach_score',0):.2f}<br><br>"
                f"<a href='https://www.google.com/maps?q={r['lat']:.5f},{r['lon']:.5f}' "
                f"target='_blank'>📍 Open in Google Maps</a>",
                max_width=220,
            ),
        ).add_to(candidate_group)
    candidate_group.add_to(m)

    # Known sites
    known_group = folium.FeatureGroup(name="Known rock art sites (red)")
    for s in known_sites:
        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=10,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.9,
            popup=folium.Popup(
                f"<b>{s['name']}</b><br>"
                f"Access: {s.get('access','?')}<br>"
                f"<small>{s.get('note','')}</small>",
                max_width=250,
            ),
        ).add_to(known_group)
    known_group.add_to(m)

    folium.LayerControl().add_to(m)

    map_name = m.get_name()
    candidates_json = json.dumps([
        {"lat": r["lat"], "lon": r["lon"], "score": r["score"],
         "elevation": r.get("elevation")}
        for r in sorted(scored, key=lambda x: x["score"], reverse=True)[:200]
    ])

    gps_js = """
    <style>
    #nearest-modal {
      display:none; position:fixed; top:0; left:0; width:100%; height:100%;
      background:rgba(0,0,0,0.5); z-index:9999; align-items:center; justify-content:center;
    }
    #nearest-modal.open { display:flex; }
    #nearest-box {
      background:white; border-radius:12px; width:90%; max-width:380px;
      max-height:80vh; overflow-y:auto; font-family:sans-serif;
      box-shadow:0 8px 32px rgba(0,0,0,0.4);
    }
    #nearest-box h3 {
      margin:0; padding:16px; border-bottom:1px solid #eee; font-size:16px;
    }
    #nearest-box .close-btn {
      float:right; cursor:pointer; font-size:20px; color:#888; line-height:1;
    }
    .candidate-row {
      padding:12px 16px; border-bottom:1px solid #f0f0f0; cursor:pointer;
      display:flex; align-items:center; gap:12px;
    }
    .candidate-row:hover { background:#f5f5f5; }
    .candidate-row:active { background:#e3f2fd; }
    .candidate-rank {
      background:#FF9800; color:white; border-radius:50%;
      width:28px; height:28px; display:flex; align-items:center;
      justify-content:center; font-weight:bold; font-size:13px; flex-shrink:0;
    }
    .candidate-info { flex:1; }
    .candidate-dist { font-size:15px; font-weight:600; color:#333; }
    .candidate-meta { font-size:12px; color:#888; margin-top:2px; }
    .candidate-gmaps {
      font-size:12px; color:#2196F3; text-decoration:none; margin-top:4px; display:block;
    }
    </style>

    <div id="nearest-modal">
      <div id="nearest-box">
        <h3>Nearest candidates <span class="close-btn" onclick="closeModal()">✕</span></h3>
        <div id="nearest-list"></div>
      </div>
    </div>

    <script>
    var CANDIDATES = """ + candidates_json + """;
    var _map = map_""" + map_name + """;
    var _userMarker = null;

    function closeModal() {
      document.getElementById('nearest-modal').classList.remove('open');
    }

    function focusCandidate(lat, lon) {
      closeModal();
      _map.setView([lat, lon], 15);
      // find and open the matching marker popup
      _map.eachLayer(function(layer) {
        if (layer instanceof L.CircleMarker) {
          var ll = layer.getLatLng();
          if (Math.abs(ll.lat - lat) < 0.0001 && Math.abs(ll.lng - lon) < 0.0001) {
            layer.openPopup();
          }
        }
      });
    }

    function findNearest() {
      if (!navigator.geolocation) { alert('Geolocation not supported'); return; }
      navigator.geolocation.getCurrentPosition(function(pos) {
        var lat = pos.coords.latitude;
        var lon = pos.coords.longitude;

        // Show user location
        if (_userMarker) _map.removeLayer(_userMarker);
        _userMarker = L.marker([lat, lon], {
          icon: L.divIcon({
            html: '<div style="background:#2196F3;width:14px;height:14px;border-radius:50%;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.4)"></div>',
            iconSize:[14,14], iconAnchor:[7,7]
          })
        }).addTo(_map).bindPopup('Your location').openPopup();
        _map.setView([lat, lon], 12);

        // Compute distances
        var dists = CANDIDATES.map(function(c) {
          var dlat = c.lat - lat, dlon = c.lon - lon;
          var km = Math.sqrt(dlat*dlat + dlon*dlon) * 111;
          return Object.assign({}, c, {dist: km});
        });
        dists.sort(function(a,b){ return a.dist - b.dist; });
        var nearest = dists.slice(0, 8);

        // Build modal list
        var html = '';
        nearest.forEach(function(c, i) {
          var elev = c.elevation ? c.elevation.toFixed(0) + 'm asl' : '';
          var gmaps = 'https://maps.google.com/?q=' + c.lat.toFixed(5) + ',' + c.lon.toFixed(5);
          html += '<div class="candidate-row" onclick="focusCandidate(' + c.lat + ',' + c.lon + ')">';
          html += '<div class="candidate-rank">' + (i+1) + '</div>';
          html += '<div class="candidate-info">';
          var pct = Math.round(c.score * 100);
          var label = pct >= 70 ? '🟢 High' : pct >= 45 ? '🟡 Medium' : '🔴 Low';
          html += '<div class="candidate-dist">' + c.dist.toFixed(1) + ' km away</div>';
          html += '<div class="candidate-meta">' + label + ' · ' + pct + '% · ' + (elev ? elev : '') + '</div>';
          html += '<a class="candidate-gmaps" href="' + gmaps + '" target="_blank" onclick="event.stopPropagation()">Open in Google Maps ↗</a>';
          html += '</div></div>';
        });
        document.getElementById('nearest-list').innerHTML = html;
        document.getElementById('nearest-modal').classList.add('open');

      }, function(e){ alert('Could not get location: ' + e.message); });
    }
    </script>

    <div style="position:fixed;bottom:30px;right:10px;z-index:1000">
      <button onclick="findNearest()" style="background:#2196F3;color:white;border:none;
        padding:14px 20px;border-radius:28px;font-size:15px;cursor:pointer;
        box-shadow:2px 2px 10px rgba(0,0,0,0.4);font-weight:600">
        📍 Nearest candidates
      </button>
    </div>
    """

    title_html = """
    <div style="position:fixed;top:10px;left:60px;z-index:1000;background:white;
                padding:10px;border-radius:8px;font-family:sans-serif;font-size:13px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.3);max-width:260px">
      <b>🪨 Rock Art Finder — Central Finland</b><br>
      <span style="color:red">●</span> Known sites &nbsp;
      <span style="color:orange">●</span> Candidates<br>
      <small>Heatmap = likelihood score<br>
      Elevation + cliff aspect + proximity</small>
    </div>
    """
    pwa_head = """
    <link rel="manifest" href="manifest.json">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Rock Art Finder">
    <meta name="theme-color" content="#2196F3">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
      html, body { margin:0; padding:0; height:100%; overflow:hidden; }
      .folium-map { position:fixed !important; top:0; left:0; width:100% !important; height:100% !important; }
    </style>
    """
    m.get_root().header.add_child(folium.Element(pwa_head))
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(gps_js))

    return m


def main():
    print("=== Rock Art Finder — Central Finland ===\n")

    known_sites = load_known_sites()
    grid = build_grid()

    print(f"\nReading elevations for {len(grid)} grid points...")
    all_elevations = load_all_elevations(grid)

    print("Computing slope and aspect from DEM...")
    slopes, aspects = load_slope_and_aspect(grid)
    valid_slopes = int((~np.isnan(slopes)).sum())
    print(f"  {valid_slopes}/{len(grid)} slope values computed")

    print(f"\nScoring {len(grid)} grid points...")
    scored = score_grid(grid, known_sites, all_elevations, slopes.tolist(), aspects.tolist())

    # Save candidates
    candidates = [r for r in scored if r["score"] > 0.2]
    candidates.sort(key=lambda x: x["score"], reverse=True)
    with open("candidates.json", "w") as f:
        json.dump(candidates[:200], f, indent=2)
    print(f"Saved {len(candidates[:200])} top candidates to candidates.json")

    print("\nBuilding map...")
    m = build_map(scored, known_sites)
    m.save("map.html")
    print("Map saved to map.html — open it in your browser!")

    print("\nTop 10 candidate locations:")
    print("-" * 72)
    for i, c in enumerate(candidates[:10], 1):
        elev = f"{c['elevation']:.0f}m" if c['elevation'] else "?"
        url = f"https://www.google.com/maps?q={c['lat']:.5f},{c['lon']:.5f}"
        print(f"#{i:>2}  score={c['score']:.3f}  elev={elev:<6}  {url}")


if __name__ == "__main__":
    main()
