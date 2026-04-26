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
    "max_lon": 28.0,
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
ELEV_ABOVE_LAKE_MIN = 0    # metres above local lake surface (Toussunlinna is ~0m)
ELEV_ABOVE_LAKE_MAX = 25   # metres above local lake surface (Pyhänpää ~24m above Päijänne)
ELEV_ABOVE_LAKE_PEAK = 7   # sweet spot — real site data averages ~7m above lake surface

# Approximate lake surface elevations (asl) for scoring
LAKE_SURFACES = [
    {"name": "Päijänne",    "elev": 77.5, "lat": 62.05, "lon": 25.55, "radius_km": 40},
    {"name": "Saraavesi",   "elev": 82.0, "lat": 62.48, "lon": 25.90, "radius_km": 10},
    {"name": "Hankavesi",   "elev": 82.0, "lat": 62.43, "lon": 26.24, "radius_km": 12},
    {"name": "Konnevesi",   "elev": 95.5, "lat": 62.57, "lon": 26.55, "radius_km": 15},
    {"name": "Keitele",     "elev": 99.0, "lat": 62.70, "lon": 26.20, "radius_km": 25},
    {"name": "Kynsivesi",   "elev": 82.0, "lat": 62.30, "lon": 26.10, "radius_km": 10},
    {"name": "Leppävesi",   "elev": 82.0, "lat": 62.37, "lon": 25.85, "radius_km": 8},
    {"name": "Suvasvesi",   "elev": 76.6, "lat": 62.45, "lon": 27.80, "radius_km": 20},
    {"name": "Juojärvi",    "elev": 77.0, "lat": 62.20, "lon": 28.00, "radius_km": 15},
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
        "note": "One of Finland's three major rock art sites and the largest painted surface in Scandinavia. Dozens of figures including elk, boats, humans and handprints spread across a wide terrace wall. Faces the lake, accessible only by water.",
        "elev_above_water": 2,
    },
    {
        "name": "Saraakallio II (Laukaa)",
        "lon": 25.9920, "lat": 62.4195,
        "access": "boat",
        "note": "~200m southeast of Saraakallio I. Low cliff set back ~40m from shore, partially hidden by trees. Figures badly weathered: two human figures in upper left corner, four overlapping animal figures to the right, fragmentary colour traces below. Accessible by water.",
        "elev_above_water": 2,
    },
    {
        "name": "Toussunlinna (Laukaa / Hankavesi)",
        "lon": 26.2413, "lat": 62.4332,
        "access": "boat",
        "note": "West shore of a long narrow south-facing bay on Hankavesi. Painting (~100×120cm) at the base of a cliff that drops straight into the water, above a rock hollow. Figures: a bent-knee human left, colour traces of deer right, small headless animal at the bottom, a long-necked animal above, and an unidentified mark above a crack. Inaccessible except by boat.",
        "elev_above_water": 5,
    },
    {
        "name": "Pyhänpää (Päijänne south)",
        "lon": 25.4745, "lat": 61.6418,
        "access": "mixed",
        "note": "Shore cliff rising from Päijänne with a wide ledge. Two painting panels above the ledge. Left panel: large outline elk with two boat figures above (lower boat forms a crown over the elk's head), human figure over the hindlegs, and two large X-marks at top. Right panel (~10m right): stronger colour but worn, no clear figures. Paintings notably high above current water level.",
        "elev_above_water": 15,
    },
    {
        "name": "Uittovuori / Paanalansaari (Hankavesi)",
        "lon": 26.2505, "lat": 62.3813,
        "access": "boat",
        "note": "NW end of Paanalansaari island between Kynsivesi and Leivonvesi, ~900m NE of Kaivanto canal. Painting on the west face of Uittovuori cliff, on a terrace 10–11m above water. Figures in a 36×60cm area within deep vertical cracks: three vertical lines above a circle, and to the right an unusual horned human figure (25cm tall) with curved legs touching heel-to-heel and arms bent like fists on hips. Rare style. Discovered 1999.",
        "elev_above_water": 10,
    },
    {
        "name": "Halsvuori (Jyväskylä)",
        "lon": 25.8124, "lat": 62.3699,
        "access": "land",
        "note": "Base of a ~500m long, ~20m high cliff face opening SW toward a now-overgrown pond (once drained to Hiidenjärvi and Iso-Kuukkanen). 5km from Leppävesi. NOT a boat-access site — proposed to be on a land portage route across the isthmus. Small 1m-wide painting panel: two human figures (35cm and 31cm tall) carrying objects in their hands, interpreted as prey animals (fox, beaver or squirrel). Damaged by campfire soot and climbing bolts nearby. Found 1979.",
        "elev_above_water": 5,
    },
    {
        "name": "Kumpusaari (Konnevesi)",
        "lon": 26.6699, "lat": 62.5411,
        "access": "boat",
        "note": "North shore of Kumpusaari island, Rajaniemi headland, Etelä-Konnevesi national park. Overhang ~5–6m above water, painting on pale cliff face under the lip. No clear figures — only vertical and diagonal colour patches. Authenticated by Kuopio Cultural History Museum in 2018.",
        "elev_above_water": 5,
    },
    {
        "name": "Hakavuori / Raidanlahti (Jyväskylä)",
        "lon": 25.7516, "lat": 61.9882,
        "access": "land",
        "note": "Painted on a erratic boulder (~3m high, 5m wide) on the NW slope of a rocky hill, ~250m from Päijänne shore. South-facing smooth side of the boulder, ~1m above ground. Badly faded and lichen-covered. Red pigment spread over an area containing 3 elk figures (identified digitally) walking left in single file; first two ~25cm long, third smaller. 5–6 elk reported by Lahelma. Near a Stone Age settlement site excavated in 1991.",
        "elev_above_water": 5,
    },
    {
        "name": "Avosaari (Päijänne / Luhanka)",
        "lon": 25.6913, "lat": 61.7253,
        "access": "boat",
        "note": "West shore of roadless Avosaari island, Päijänne east coast, between mainland and Judinsalo. Painting at S end of a multi-metre cliff that opens west toward Varpusenselä. Terrace 9.1m above water; paintings 10.46m above Päijänne surface (measured 1996). Panel 75×80cm: three elk facing left — only one clearly visible, painted solid with interior lines (rare type, otherwise only at Saraakallio in Finland). May show an elk-headed boat. Discovered 1976 by Timo Miettinen.",
        "elev_above_water": 10,
    },
    {
        "name": "Viherinkoski A+B (Joutsa)",
        "lon": 26.1862, "lat": 61.7286,
        "access": "mixed",
        "note": "Two sites at Viherinkoski rapids on the ancient outlet of Puulavesi to Päijänne — a portage/travel corridor. Site A: erratic boulder 10m from shore, 2.7m above water (8×5×4m); east-facing side with elk, 1–2 humans, partial handprint and fragments — heavily moss-covered. Site B: cliff ledge on opposite bank, only visible from a boat, ~2.2–3.0m above water; one boat figure and a long-tailed animal (wolf/dog?). Discovered 1990–1996.",
        "elev_above_water": 3,
    },
    {
        "name": "Hahlavuori (Hirvensalmi / Puulavesi)",
        "lon": 26.4559, "lat": 61.8306,
        "access": "boat",
        "note": "West shore of Kaiskonselkä, Puulavesi. Strongly fractured vertical cliff, NE-facing. Painting panel 6m above lake, 2.2×1.5m with 20+ figure fragments. Figures: 3 elk (1 outline left, 2 solid right), 3 adorant humans with arms raised (one 'sturdy' with triangle head holding a straight + curved stick), 1 upside-down 'diver' human (head downward, foot touching a deer — rare motif also at Pyhänpää, Verijärvi, Haukkavuori), and a 'calotte' geometric (semicircle bisected by vertical line, 20cm wide). Resembles Astuvansalmi. Discovered 1990.",
        "elev_above_water": 6,
    },
    {
        "name": "Karhuvuori (Leppävirta / Unnukka)",
        "lon": 27.8543, "lat": 62.4130,
        "access": "boat",
        "note": "West shore of Pöyhönsaari island in Unnukka lake, vertical cliff at a narrows on a water route. Nearly 2m wide painting. One stick-figure human: knees bent, one arm raised diagonally with a curved line attached. Dated 5200–3900 BCE (Stone Age). Very few rock paintings known in North Savo. Discovered by Anssi Toivanen in 2017.",
        "elev_above_water": 3,
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

    # ── Portage corridor score ────────────────────────────────────────────────
    # Points close to TWO different lake systems score high — likely on a land
    # carry route between water bodies (Halsvuori pattern).
    # For each grid point: find distance to each lake, take two closest *different*
    # lake systems (by name), score by sum of closeness.
    PORTAGE_MAX_KM = 8.0   # must be within this of each lake to count
    lake_dists_km = np.stack([
        vec_haversine(ll, lo, lats, lons)
        for ll, lo in zip(lake_lats, lake_lons)
    ], axis=1)   # shape (N, n_lakes)
    # Sort distances per point, take closest and 2nd closest lake distances
    sorted_dists = np.sort(lake_dists_km, axis=1)
    d1 = sorted_dists[:, 0]   # closest lake
    d2 = sorted_dists[:, 1]   # 2nd closest (different lake)
    portage_score = np.where(
        (d1 < PORTAGE_MAX_KM) & (d2 < PORTAGE_MAX_KM),
        (1.0 - d1 / PORTAGE_MAX_KM) * (1.0 - d2 / PORTAGE_MAX_KM),
        0.0
    )
    # Suppress portage bonus when point is also right on a single lake shore
    # (those are already rewarded by water route score)
    portage_score = np.where(d1 < 1.5, portage_score * 0.3, portage_score)

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
    # Portage corridor replaces part of approach weight — captures land-route sites
    total = np.clip(
        0.40*e_score + 0.20*c_score + 0.25*p_score + 0.07*a_score
        + 0.05*(1 + w_score) + 0.08*portage_score,
        0, 1
    )

    results = []
    for i in range(len(grid)):
        elev_val = float(elevs[i]) if not np.isnan(elevs[i]) else None
        results.append({
            "lat": float(lats[i]),
            "lon": float(lons[i]),
            "elevation": round(elev_val, 1) if elev_val else None,
            "above_lake": round(float(above[i]), 1) if not np.isnan(above[i]) else None,
            "elev_score": round(float(e_score[i]), 3),
            "cliff_score": round(float(c_score[i]), 3),
            "prox_score": round(float(p_score[i]), 3),
            "approach_score": round(float(a_score[i]), 3),
            "route_bonus": round(float(w_score[i]), 3),
            "portage_score": round(float(portage_score[i]), 3),
            "score": round(float(total[i]), 3),
        })
    return results


def build_map(scored, known_sites):
    center_lat = (BBOX["min_lat"] + BBOX["max_lat"]) / 2
    center_lon = (BBOX["min_lon"] + BBOX["max_lon"]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles=None)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Heatmap of candidate scores
    heat_data = [[r["lat"], r["lon"], r["score"]] for r in scored if r["score"] > 0.55]
    print(f"  Heatmap points: {len(heat_data)}")
    HeatMap(heat_data, radius=14, blur=22, min_opacity=0.1, max_opacity=0.6,
            gradient={0.0: "#440000", 0.5: "orange", 0.75: "yellow", 1.0: "red"},
            name="Likelihood heatmap", show=False).add_to(m)

    # Ancient shoreline band — tight 3–12m band above nearest lake surface.
    # Real confirmed sites cluster here (Toussunlinna ~0m, most 5–11m, peak ~7m).
    # Intensity = how close to the 7m sweet spot.
    def shore_intensity(above):
        if above is None: return 0
        if above < 3 or above > 12: return 0
        if above <= 7:
            return (above - 3) / 4      # ramps up 3→7m
        else:
            return (12 - above) / 5     # ramps down 7→12m
    shore_data = [
        [r["lat"], r["lon"], shore_intensity(r.get("above_lake"))]
        for r in scored if shore_intensity(r.get("above_lake")) > 0
    ]
    print(f"  Ancient shoreline points: {len(shore_data)}")
    HeatMap(shore_data, radius=8, blur=10, min_opacity=0.15, max_opacity=0.65,
            gradient={0.0: "#004488", 0.5: "#0099cc", 1.0: "#55eeff"},
            name="Ancient shoreline band (3–12m)", show=False).add_to(m)

    # Top candidates as markers
    top = sorted(scored, key=lambda x: x["score"], reverse=True)[:30]
    candidate_group = folium.FeatureGroup(name="Top 30 candidates")
    for r in top:
        elev_str = f"{r['elevation']:.0f}m" if r['elevation'] else "?"
        marker_radius = 4 + int(r["score"] * 10)   # 4–14px based on score
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=marker_radius,
            color="orange",
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Candidate #{top.index(r)+1}</b><br>"
                f"<b style='font-size:15px'>{round(r['score']*100)}% likelihood</b><br>"
                f"{'🟢 High' if r['score']>=0.7 else '🟡 Medium' if r['score']>=0.45 else '🔴 Low'}<br><br>"
                f"Elevation: {elev_str}<br>"
                f"Elev score: {r['elev_score']:.2f} · Cliff: {r.get('cliff_score',0):.2f}<br>"
                f"Proximity: {r['prox_score']:.2f} · Approach: {r.get('approach_score',0):.2f}<br>"
                f"Portage corridor: {r.get('portage_score',0):.2f}<br><br>"
                f"<a href='https://www.google.com/maps?q={r['lat']:.5f},{r['lon']:.5f}' target='_blank'>📍 Google Maps</a><br>"
                f"<a href='https://www.retkikartta.fi/?lat={r['lat']:.5f}&lng={r['lon']:.5f}&zoom=15' target='_blank'>🥾 Retkikartta</a><br>"
                f"<a href='https://asiointi.maanmittauslaitos.fi/karttapaikka/?lang=fi&share=customMarker&n={r['lat']:.5f}&e={r['lon']:.5f}&zoom=8' target='_blank'>🗺️ MML Maastokartta</a>",
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
    #panel {
      position:fixed; top:0; right:-320px; width:300px; height:100%;
      background:white; z-index:9999; transition:right .3s ease;
      display:flex; flex-direction:column; font-family:sans-serif;
      box-shadow:-4px 0 20px rgba(0,0,0,0.2);
    }
    #panel.open { right:0; }
    #panel-header {
      padding:14px 16px; background:#2196F3; color:white;
      display:flex; justify-content:space-between; align-items:center; flex-shrink:0;
    }
    #panel-header h3 { margin:0; font-size:15px; }
    #panel-close { cursor:pointer; font-size:22px; line-height:1; padding:0 4px; }
    #panel-list { overflow-y:auto; flex:1; }
    .crow {
      padding:13px 16px; border-bottom:1px solid #f0f0f0; cursor:pointer;
      display:flex; align-items:center; gap:12px; text-decoration:none; color:inherit;
    }
    .crow:hover, .crow:active { background:#e3f2fd; }
    .crank {
      background:#FF9800; color:white; border-radius:50%;
      width:30px; height:30px; min-width:30px;
      display:flex; align-items:center; justify-content:center;
      font-weight:bold; font-size:13px;
    }
    .cinfo { flex:1; min-width:0; }
    .cdist { font-size:14px; font-weight:600; color:#222; }
    .cmeta { font-size:12px; color:#777; margin-top:2px; }
    .cgmaps { font-size:12px; color:#2196F3; display:block; margin-top:3px; }
    </style>

    <div id="panel">
      <div id="panel-header">
        <h3>📍 Nearest candidates</h3>
        <span id="panel-close" onclick="document.getElementById('panel').classList.remove('open')">✕</span>
      </div>
      <div id="panel-list"></div>
    </div>

    <script>
    var CANDIDATES = """ + candidates_json + """;
    var _map = map_""" + map_name + """;
    var _userMarker = null;

    function focusCandidate(lat, lon) {
      document.getElementById('panel').classList.remove('open');
      _map.setView([lat, lon], 15);
    }

    function showNearest(lat, lon) {
      var dists = CANDIDATES.map(function(c) {
        var dlat = c.lat - lat, dlon = c.lon - lon;
        return Object.assign({}, c, {dist: Math.sqrt(dlat*dlat + dlon*dlon) * 111});
      });
      dists.sort(function(a,b){ return a.dist - b.dist; });
      var html = '';
      dists.slice(0, 10).forEach(function(c, i) {
        var pct = Math.round(c.score * 100);
        var label = pct >= 70 ? '🟢 High' : pct >= 45 ? '🟡 Medium' : '🔴 Low';
        var elev = c.elevation ? c.elevation.toFixed(0) + 'm' : '';
        var gmaps = 'https://maps.google.com/?q=' + c.lat.toFixed(5) + ',' + c.lon.toFixed(5);
        var retki = 'https://www.retkikartta.fi/?lat=' + c.lat.toFixed(5) + '&lng=' + c.lon.toFixed(5) + '&zoom=15';
        var mml = 'https://asiointi.maanmittauslaitos.fi/karttapaikka/?lang=fi&share=customMarker&n=' + c.lat.toFixed(5) + '&e=' + c.lon.toFixed(5) + '&zoom=8';
        html += '<div class="crow" onclick="focusCandidate(' + c.lat + ',' + c.lon + ')">';
        html += '<div class="crank">' + (i+1) + '</div>';
        html += '<div class="cinfo">';
        html += '<div class="cdist">' + c.dist.toFixed(1) + ' km &nbsp;' + label + ' ' + pct + '%' + (elev ? ' · ' + elev : '') + '</div>';
        html += '<a class="cgmaps" href="' + gmaps + '" target="_blank" onclick="event.stopPropagation()">📍 Google Maps</a> ';
        html += '<a class="cgmaps" href="' + retki + '" target="_blank" onclick="event.stopPropagation()">🥾 Retkikartta</a> ';
        html += '<a class="cgmaps" href="' + mml + '" target="_blank" onclick="event.stopPropagation()">🗺️ Maastokartta</a>';
        html += '</div></div>';
      });
      document.getElementById('panel-list').innerHTML = html;
      document.getElementById('panel').classList.add('open');
    }

    function findNearest() {
      var center = _map.getCenter();
      if (!navigator.geolocation) { showNearest(center.lat, center.lng); return; }
      navigator.geolocation.getCurrentPosition(function(pos) {
        var lat = pos.coords.latitude, lon = pos.coords.longitude;
        if (_userMarker) _map.removeLayer(_userMarker);
        _userMarker = L.marker([lat, lon], {
          icon: L.divIcon({
            html: '<div style="background:#2196F3;width:14px;height:14px;border-radius:50%;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,0.4)"></div>',
            iconSize:[14,14], iconAnchor:[7,7]
          })
        }).addTo(_map).bindPopup('Your location').openPopup();
        _map.setView([lat, lon], 12);
        showNearest(lat, lon);
      }, function() { showNearest(center.lat, center.lng); });
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
