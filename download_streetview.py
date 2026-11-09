"""
download_streetview.py

Downloads Street View images for a fixed list of locations read from a CSV:

    country,lat,lon,heading
    Lithuania,54.905384,23.977194,44
    Lithuania,55.735107,21.888685,355

For each row:
  1. Free metadata check that searches for the nearest panorama around the
     point, widening the search radius step by step (SEARCH_RADII_M)
  2. Image download of exactly that panorama, at the given heading
  3. Result logged to a CSV (incl. panorama position and distance to the
     requested point), so interrupted runs resume where they stopped


Usage:
    python download_streetview.py dataset_coordinates.csv --out streetview_dataset_fixed --size 336x336

    It can be stopped at any time and be resumed thanks to checkpoint system.
"""

import argparse
import csv
import time
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import requests
from tqdm import tqdm

API_KEY = "STREETVIEW_API_KEY"
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
REQUEST_DELAY = 0.05  # seconds between requests (stay under rate limits)
MAX_RETRIES = 3
# Search radii (metres) tried in order until a panorama is found.
# The API's default is 50 m, which often misses when the point is off-road.
SEARCH_RADII_M = [50, 100, 250, 500, 1000]

LOG_FIELDS = ["row", "country", "lat", "lon", "heading", "filename",
              "pano_id", "pano_date", "pano_lat", "pano_lon",
              "distance_m", "status"]


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = radians(lat1), radians(lat2)
    a = (sin(radians(lat2 - lat1) / 2) ** 2
         + cos(p1) * cos(p2) * sin(radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def get_metadata(lat, lon, radius, api_key):
    """Metadata of the nearest outdoor panorama within `radius` metres."""
    params = {"location": f"{lat:.6f},{lon:.6f}", "radius": radius,
              "source": "outdoor", "key": api_key}
    r = requests.get(METADATA_URL, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def find_panorama(lat, lon, api_key):
    """Try increasing radii; return (metadata, radius) or (None, None)."""
    for radius in SEARCH_RADII_M:
        meta = with_retries(get_metadata, lat, lon, radius, api_key)
        time.sleep(REQUEST_DELAY)
        status = meta.get("status")
        if status == "OK":
            return meta, radius
        if status not in ("ZERO_RESULTS", "NOT_FOUND"):
            # e.g. REQUEST_DENIED (bad key / API not enabled / billing),
            # OVER_QUERY_LIMIT, INVALID_REQUEST: not a "no imagery" case.
            raise RuntimeError(f"API status {status}: {meta.get('error_message', '')}")
    return None, None


def download_image(pano_id, heading, size, filepath, api_key):
    # Request the exact panorama found by the metadata call, so the image
    # always matches the logged pano_id / position.
    params = {
        "size": size,
        "pano": pano_id,
        "heading": heading,
        "pitch": 0,
        "fov": 90,
        "key": api_key,
    }
    r = requests.get(IMAGE_URL, params=params, timeout=15)
    r.raise_for_status()
    # The API returns a grey placeholder with HTTP 200 instead of an error;
    # the metadata check above prevents that case.
    if not r.headers.get("Content-Type", "").startswith("image/"):
        raise ValueError(f"Unexpected response type: {r.headers.get('Content-Type')}")
    filepath.write_bytes(r.content)


def with_retries(fn, *args):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def read_locations(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"country", "lat", "lon", "heading"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        return list(reader)


def load_done_rows(log_path):
    """Row indices already handled (downloaded or confirmed unavailable)."""
    if not log_path.exists():
        return set()
    with open(log_path, newline="", encoding="utf-8") as f:
        return {int(r["row"]) for r in csv.DictReader(f)
                if r["status"] in ("ok", "no_imagery")}


def append_log(log_path, row):
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Download Street View images from a CSV of locations.")
    parser.add_argument("csv", type=Path, help="CSV with columns country,lat,lon,heading")
    parser.add_argument("--out", type=Path, default=Path("streetview_dataset_fixed"))
    parser.add_argument("--size", default="640x640",
                        help="Image size WxH (max 640x640 on the standard API plan)")
    args = parser.parse_args()

    
    if not API_KEY:
        raise SystemExit("Set the STREETVIEW_API_KEY environment variable first.")

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "download_log.csv"

    locations = read_locations(args.csv)
    done = load_done_rows(log_path)
    print(f"{len(locations)} locations in CSV, {len(done)} already handled.")

    counts = {"ok": 0, "no_imagery": 0, "error": 0}

    for i, loc in enumerate(tqdm(locations, unit="img")):
        if i in done:
            continue

        country = loc["country"].strip()
        lat, lon = float(loc["lat"]), float(loc["lon"])
        heading = int(float(loc["heading"]))

        country_dir = images_dir / country.replace(" ", "_")
        country_dir.mkdir(exist_ok=True)
        # Row index keeps names unique even if two rows share coordinates.
        filename = f"{country.replace(' ', '_')}_{i:05d}_{lat:.6f}_{lon:.6f}_{heading}.jpg"
        filepath = country_dir / filename

        log = {"row": i, "country": country, "lat": lat, "lon": lon,
               "heading": heading, "filename": filename,
               "pano_id": "", "pano_date": "", "pano_lat": "", "pano_lon": "",
               "distance_m": "", "status": ""}

        try:
            meta, _ = find_panorama(lat, lon, API_KEY)

            if meta is None:
                log["status"] = "no_imagery"
            else:
                plat, plon = meta["location"]["lat"], meta["location"]["lng"]
                log.update(pano_id=meta["pano_id"], pano_date=meta.get("date", ""),
                           pano_lat=round(plat, 6), pano_lon=round(plon, 6),
                           distance_m=round(haversine_m(lat, lon, plat, plon), 1))
                if not filepath.exists():
                    with_retries(download_image, meta["pano_id"], heading,
                                 args.size, filepath, API_KEY)
                    time.sleep(REQUEST_DELAY)
                log["status"] = "ok"
        except RuntimeError as e:
            # API-level failure affects every row: stop instead of logging 25k errors.
            raise SystemExit(f"[row {i}] {country} ({lat}, {lon}): {e}")
        except Exception as e:
            log["status"] = "error"
            tqdm.write(f"[row {i}] {country} ({lat}, {lon}): {e}")

        counts[log["status"]] += 1
        append_log(log_path, log)

    print(f"\nDone. Downloaded: {counts['ok']}, no imagery: {counts['no_imagery']}, "
          f"errors: {counts['error']} (rerun the script to retry errors).")
    print(f"Images: {images_dir.resolve()}")
    print(f"Log:    {log_path.resolve()}")


if __name__ == "__main__":
    main()