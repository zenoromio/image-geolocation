"""
create_dataset.py

Downloads 500 Street View images per country for 50 countries.
- Uniform random sampling within country borders (Natural Earth polygons)
- Free metadata check before each paid image request
- Rejects snapped locations > 1km from sampled point
- Random heading per location
- 336x336px images
- Checkpoint CSV for safe resume across sessions

Usage:
    just run file. It can be stopped at any time and be resumed thanks to checkpoint system.
"""

import csv
import time
import random
import hashlib
import requests
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2

# ─── CONFIG ───────────────────────────────────────────────────────────────────

API_KEY        = "STREETVIEW_API_KEY"
OUTPUT_DIR     = Path("streetview_dataset")
CHECKPOINT_CSV = OUTPUT_DIR / "progress.csv"
IMAGES_DIR     = OUTPUT_DIR / "images"
IMAGES_PER_COUNTRY = 500
IMAGE_SIZE     = "336x336"
MAX_SNAP_KM    = 10.0      # reject if API snaps > 1km from sampled point
MAX_ATTEMPTS   = 1000       # metadata attempts per valid image before giving up
REQUEST_DELAY  = 0.05     # seconds between requests (stay under rate limits)

# Add full correct name of country or error will be thrown
COUNTRIES = [
    "Argentina", "Bangladesh", "Belgium", "Brazil", "Bulgaria",
    "Canada", "Chile", "Colombia", "Croatia", "Cyprus",
    "Czechia", "Denmark", "Ecuador", "Finland", "France",
    "Georgia", "Germany", "Ghana", "Greece", "Hungary",
    "India", "Indonesia", "Israel", "Italy", "Japan",
    "Kenya", "Lithuania", "Malaysia", "Mexico", "Netherlands",
    "New Zealand", "Nigeria", "Norway", "Peru", "Philippines",
    "Poland", "Portugal", "Romania", "Russia", "South Africa",
    "Spain", "Sri Lanka", "Sweden", "Switzerland", "Taiwan",
    "Thailand", "Turkey", "United States of America", "Uruguay", "Vietnam"
]


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def sample_point_in_polygon(polygon):
    """Uniform random sample within a (possibly multi-)polygon."""
    min_x, min_y, max_x, max_y = polygon.bounds
    while True:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            return point.y, point.x  # lat, lon


def check_metadata(lat, lon):
    """
    Call the free metadata endpoint.
    Returns (ok: bool, snapped_lat: float, snapped_lon: float).
    """
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "key": API_KEY}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        return False, None, None
    loc = data["location"]
    return True, loc["lat"], loc["lng"]


def download_image(lat, lon, heading, filepath):
    """Download a single Street View image and save to filepath."""
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size":     IMAGE_SIZE,
        "location": f"{lat},{lon}",
        "heading":  heading,
        "pitch":    0,
        "fov":      90,
        "key":      API_KEY,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    filepath.write_bytes(r.content)


def load_checkpoint():
    """
    Returns a set of already-completed (country, filename) pairs,
    and a dict of country -> count of completed images.
    """
    completed = {}  # country -> count
    if not CHECKPOINT_CSV.exists():
        return completed
    with open(CHECKPOINT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "ok":
                completed[row["country"]] = completed.get(row["country"], 0) + 1
    return completed


def append_checkpoint(row: dict):
    """Append a single row to the checkpoint CSV."""
    fieldnames = ["country", "lat", "lon", "snapped_lat", "snapped_lon",
                  "heading", "filename", "status"]
    write_header = not CHECKPOINT_CSV.exists()
    with open(CHECKPOINT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    # Load Natural Earth country polygons
    print("Loading country polygons...")
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
    # naturalearth_lowres has a 'name' column
    # For higher accuracy borders consider naturalearth_110m or a custom shapefile

    completed = load_checkpoint()
    print(f"Resuming — already completed: { {k: v for k, v in completed.items()} }\n")

    for country in COUNTRIES:
        already_done = completed.get(country, 0)
        if already_done >= IMAGES_PER_COUNTRY:
            print(f"[{country}] Already complete ({already_done} images), skipping.")
            continue

        need = IMAGES_PER_COUNTRY - already_done

        # Resolve country name for Natural Earth
        match = world[world["ADMIN"] == country]

        if match.empty:
            raise ValueError(f"[{country}] '{country}' not found in Natural Earth. "
                             f"Add the correct name to COUNTRY_NAME_MAP.")

        polygon = match.geometry.unary_union

        country_img_dir = IMAGES_DIR / country.replace(" ", "_")
        country_img_dir.mkdir(exist_ok=True)

        collected = 0
        attempts  = 0

        pbar = tqdm(total=need, desc=f"{country}", unit="img")

        while collected < need:
            attempts += 1

            # Safety valve: if we're burning too many attempts, warn and move on
            if attempts > need * MAX_ATTEMPTS:
                print(f"\n[{country}] WARNING: Exhausted {attempts} attempts for {need} images. "
                      f"Coverage may be too sparse. Got {collected}/{need}.")
                break

            # 1. Sample a random coordinate inside the country
            lat, lon = sample_point_in_polygon(polygon)

            # 2. Free metadata check
            try:
                ok, snapped_lat, snapped_lon = check_metadata(lat, lon)
            except Exception as e:
                time.sleep(1)
                continue

            time.sleep(REQUEST_DELAY)

            if not ok:
                continue

            # 3. Reject if API snapped too far from sampled point
            dist = haversine_km(lat, lon, snapped_lat, snapped_lon)
            if dist > MAX_SNAP_KM:
                continue

            # 4. Random heading
            heading = random.randint(0, 359)

            # 5. Unique filename based on coordinates
            uid = hashlib.md5(f"{snapped_lat}{snapped_lon}{heading}".encode()).hexdigest()[:8]
            filename = f"{country.replace(' ', '_')}_{uid}.jpg"
            filepath = country_img_dir / filename

            if filepath.exists():
                # Already downloaded in a previous run but checkpoint missed it
                collected += 1
                pbar.update(1)
                continue

            # 6. Download image
            try:
                download_image(snapped_lat, snapped_lon, heading, filepath)
            except Exception as e:
                time.sleep(1)
                continue

            time.sleep(REQUEST_DELAY)

            # 7. Checkpoint
            append_checkpoint({
                "country":     country,
                "lat":         round(lat, 6),
                "lon":         round(lon, 6),
                "snapped_lat": round(snapped_lat, 6),
                "snapped_lon": round(snapped_lon, 6),
                "heading":     heading,
                "filename":    filename,
                "status":      "ok",
            })

            collected += 1
            pbar.update(1)

        pbar.close()
        print(f"[{country}] Done: {collected} images in {attempts} attempts "
              f"({100*collected/max(attempts,1):.1f}% hit rate)\n")

    print("All countries processed.")
    print(f"Images saved to: {IMAGES_DIR.resolve()}")
    print(f"Checkpoint CSV:  {CHECKPOINT_CSV.resolve()}")


if __name__ == "__main__":
    main()