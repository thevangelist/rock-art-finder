"""
Fetch known Finnish rock art sites from Museovirasto (Finnish Heritage Agency)
open API - Muinaisjäännösrekisteri
"""
import requests
import json
import sys

# Museovirasto open data WFS endpoint
WFS_URL = "https://kartta.nba.fi/arcgis/services/WFS/MV_KulttuuriymparistoSuojelu/MapServer/WFSServer"

# Bounding box for Central Finland / Jyväskylä-Hankasalmi area
# (roughly what's in the screenshot)
BBOX = {
    "min_lon": 25.5,
    "max_lon": 27.0,
    "min_lat": 61.8,
    "max_lat": 62.7,
}

def fetch_rock_art_sites():
    """Fetch kalliomaalaukset (rock paintings) from Museovirasto WFS."""
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": "MV_KulttuuriymparistoSuojelu:Muinaisjaannos_piste",
        "OUTPUTFORMAT": "application/json",
        "CQL_FILTER": (
            f"tyyppi='kalliomaalaus' AND "
            f"BBOX(shape,{BBOX['min_lon']},{BBOX['min_lat']},"
            f"{BBOX['max_lon']},{BBOX['max_lat']},'EPSG:4326')"
        ),
        "COUNT": "500",
    }

    print("Fetching rock art sites from Museovirasto...")
    try:
        r = requests.get(WFS_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        print(f"  Found {len(features)} sites via WFS")
        return features
    except Exception as e:
        print(f"  WFS failed: {e}")

    # Fallback: try the REST API
    return fetch_via_rest()


def fetch_via_rest():
    """Fallback: Museovirasto ArcGIS REST API."""
    url = (
        "https://kartta.nba.fi/arcgis/rest/services/WFS/"
        "MV_KulttuuriymparistoSuojelu/MapServer/0/query"
    )
    params = {
        "where": "tyyppi='kalliomaalaus'",
        "geometry": f"{BBOX['min_lon']},{BBOX['min_lat']},{BBOX['max_lon']},{BBOX['max_lat']}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": 500,
    }
    print("  Trying REST API fallback...")
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        print(f"  Found {len(features)} sites via REST")
        return features
    except Exception as e:
        print(f"  REST failed: {e}")
        return []


def fetch_all_finland_rock_art():
    """Fetch all Finnish rock art sites (for full dataset)."""
    url = (
        "https://kartta.nba.fi/arcgis/rest/services/WFS/"
        "MV_KulttuuriymparistoSuojelu/MapServer/0/query"
    )
    params = {
        "where": "tyyppi='kalliomaalaus'",
        "outSR": "4326",
        "outFields": "kohdenimi,tyyppi,ajoitus,kunta",
        "f": "geojson",
        "resultRecordCount": 2000,
    }
    print("Fetching ALL Finnish rock art sites...")
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        print(f"  Total Finnish rock art sites: {len(features)}")
        return features
    except Exception as e:
        print(f"  Failed: {e}")
        return []


if __name__ == "__main__":
    if "--all" in sys.argv:
        sites = fetch_all_finland_rock_art()
    else:
        sites = fetch_rock_art_sites()

    output = {"type": "FeatureCollection", "features": sites}
    with open("known_sites.geojson", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(sites)} sites to known_sites.geojson")
