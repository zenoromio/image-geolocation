"""
Useful functions to calculate neighbour rank and to visualize closest points between countries.

E.G.
neighbour_rank("Taiwan", "Trinidad and Tobago") -> 188

closest_points("Taiwan", "Trinidad and Tobago") -> {'true_point': (25.2874, 121.5397), 'pred_point': (11.3511, -60.5305), 'km': 15934.9}

"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Geod
from scipy.spatial import cKDTree

NE_PATH = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
TOUCH_KM = 5          # closer than this counts as a neighbour (Singapore-Malaysia: 1.9 km)
REMOTE_KM = 1000      # drop parts of a country farther than this from its mainland
STEP_DEG = 0.05       # border sampling step (~5 km)
R = 6371.0088         # mean Earth radius, km

NAME_MAP = {"Czech Republic": "Czechia", "United States": "United States of America",
            "Serbia": "Republic of Serbia", "Tanzania": "United Republic of Tanzania",
            "North Macedonia": "North Macedonia", "Eswatini": "eSwatini",
            "East Timor": "East Timor", "Bahamas": "The Bahamas"}
# pieces that belong to a country, and map entries that are not countries
MERGE = {"Northern Cyprus": "Cyprus", "N. Cyprus": "Cyprus", "Somaliland": "Somalia",
         "Aland": "Finland", "Åland": "Finland"}
DROP_IF_CONTAINS = ["Antarctic", "Sovereign Base", "Akrotiri", "Dhekelia", "No Mans", "Buffer Zone",
                    "Siachen", "Spratly", "Scarborough", "Bajo Nuevo", "Serranilla", "Clipperton",
                    "Coral Sea", "Ashmore", "Heard I", "South Georgia", "S. Geo.", "Indian Ocean",
                    "Minor Outlying", "Baykonur", "Baikonur", "Guantanamo", "Brazilian Island",
                    "Ice Field", "Bir Tawil"]

# Entries Natural Earth assigns to another state that you want to count as sovereign
# anyway (e.g. "Palestine", which Natural Earth lists under Israel). Your choice.
EXTRA_SOVEREIGN = []

_geod = Geod(ellps="WGS84")

def _xyz(lonlat):
    """lon/lat (degrees) -> points on the unit sphere."""
    lon, lat = np.radians(lonlat[:, 0]), np.radians(lonlat[:, 1])
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

def _km(chord):
    """straight-line distance between unit-sphere points -> great-circle km."""
    return 2 * R * np.arcsin(np.clip(chord / 2, 0, 1))

def _border_points(geom):
    return _xyz(shapely.get_coordinates(shapely.segmentize(geom, STEP_DEG)))

def _polygons(g):
    """All polygons inside any (multi)geometry or geometry collection."""
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        return [p for sub in g.geoms for p in _polygons(sub)]
    return []

# ---------- load once ----------
_w = gpd.read_file(NE_PATH).to_crs("EPSG:4326")
_name = "ADMIN" if "ADMIN" in _w.columns else "name"
_sov = "SOVEREIGNT" if "SOVEREIGNT" in _w.columns else _name
_w = _w[[_name, _sov, "geometry"]].set_axis(["country", "sovereign", "geometry"], axis=1)
_w = _w[_w.geometry.notna() & ~_w.country.str.contains("|".join(DROP_IF_CONTAINS))]
_w["country"] = _w.country.replace(MERGE)
_w["sovereign"] = _w.sovereign.replace(MERGE)

# Sovereign states = map entries that are their own sovereign in Natural Earth
# (France yes; French Polynesia, Aruba, Greenland, Hong Kong no).
SOVEREIGN = set(_w.loc[_w.country == _w.sovereign, "country"]) | set(EXTRA_SOVEREIGN)
_w["geometry"] = _w.geometry.apply(shapely.make_valid)

POINTS, TREES = {}, {}
for country, rows in _w.groupby("country"):
    parts = [p for g in rows.geometry for p in _polygons(g)]
    if not parts:                     # degenerate tiny shape (e.g. Vatican in coarse data)
        pts = _xyz(np.vstack([shapely.get_coordinates(g) for g in rows.geometry]))
        POINTS[country], TREES[country] = pts, cKDTree(pts)
        continue
    areas = [abs(_geod.geometry_area_perimeter(p)[0]) for p in parts]
    main = parts[int(np.argmax(areas))]
    main_tree = cKDTree(_border_points(main))
    # keep the mainland + parts within REMOTE_KM (Sicily yes, French Guiana no)
    keep = [p for p in parts
            if p is main or _km(main_tree.query(_border_points(p))[0].min()) <= REMOTE_KM]
    pts = np.vstack([_border_points(p) for p in keep])
    POINTS[country], TREES[country] = pts, cKDTree(pts)

_cache = {}

def _n(name):
    """Map a results-file name to the Natural Earth name."""
    for cand in (NAME_MAP.get(name), name):
        if cand in POINTS:
            return cand
    raise KeyError(f"{name!r} not in Natural Earth: add it to NAME_MAP")

def distances_from(country):
    """Shortest border-to-border great-circle distance (km) to every other country."""
    country = _n(country)
    if country not in _cache:
        pts, tree = POINTS[country], TREES[country]
        d = {}
        for other in POINTS:
            if other == country:
                continue
            # query the larger point set with the smaller one (faster, same result)
            if len(pts) <= len(POINTS[other]):
                d[other] = _km(TREES[other].query(pts)[0].min())
            else:
                d[other] = _km(tree.query(POINTS[other])[0].min())
        s = pd.Series(d)
        s[s < TOUCH_KM] = 0.0                     # neighbours (land border or < TOUCH_KM)
        _cache[country] = s
    return _cache[country]

def countries_closer(true_c, pred_c, candidates=None):
    """The sovereign countries strictly closer to true_c than pred_c is,
    i.e. the countries that push pred_c down the ranking. Sorted, in km."""
    d = distances_from(true_c)
    d_pred = d[_n(pred_c)]
    counted = d[d.index.isin(SOVEREIGN)]
    if candidates is not None:
        counted = counted[counted.index.isin({_n(c) for c in candidates})]
    return counted[counted < d_pred].sort_values().round(1)

def neighbour_rank(true_c, pred_c, candidates=None, verbose=False):
    """1 + number of SOVEREIGN countries strictly closer to true_c than pred_c (0 if correct).

    The distance to pred_c is always measured to pred_c itself, so a predicted
    territory (e.g. Aruba) is placed where it really is, but only sovereign
    countries are counted in the rank.
    candidates: optional list of names to restrict the counted countries further.
    verbose:    print the countries in between (those closer than pred_c)."""
    if true_c == pred_c:
        return 0
    closer = countries_closer(true_c, pred_c, candidates)
    rank = len(closer) + 1
    if verbose:
        print(f"{true_c} -> {pred_c}: rank {rank}, "
              f"{border_km(true_c, pred_c)} km; {len(closer)} countries closer:")
        for name, km in closer.items():
            print(f"  {name:<30} {km:>8.1f} km")
    return rank

def border_km(true_c, pred_c):
    return 0.0 if true_c == pred_c else round(float(distances_from(true_c)[_n(pred_c)]))

def _latlon(v):
    """unit-sphere point -> (lat, lon) in degrees."""
    return (round(float(np.degrees(np.arcsin(v[2]))), 4),
            round(float(np.degrees(np.arctan2(v[1], v[0]))), 4))

def closest_points(true_c, pred_c):
    """The two border points used for the distance: one on each country.

    Returns {"true_point": (lat, lon), "pred_point": (lat, lon), "km": distance}.
    "km" is the raw distance between the points, before TOUCH_KM sets
    near-neighbours to 0. Borders are sampled every ~5 km, so the points are
    within a few km of the exact closest pair."""
    a, b = _n(true_c), _n(pred_c)
    A, B = POINTS[a], POINTS[b]
    if len(A) <= len(B):
        d, i = TREES[b].query(A)
        j = int(np.argmin(d))
        pa, pb = A[j], B[i[j]]
    else:
        d, i = TREES[a].query(B)
        j = int(np.argmin(d))
        pa, pb = A[i[j]], B[j]
    return {"true_point": _latlon(pa), "pred_point": _latlon(pb),
            "km": round(float(_km(d[j])), 1)}
