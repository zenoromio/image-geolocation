import os

# Must be set before importing transformers, otherwise it is ignored.
# Model is cached inside the project folder: stclip/hf_cache/
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
)

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from geopy.distance import geodesic
from transformers import CLIPModel, CLIPProcessor

# ── Paths (relative to this file, so it runs from any working directory) ──────
BASE_DIR      = Path(__file__).resolve().parent
IMAGES_DIR    = BASE_DIR / "images"
PROGRESS_DIR  = BASE_DIR / "progress"
RESULTS_DIR   = BASE_DIR / "results"
PROGRESS_FILE = PROGRESS_DIR / "progress.json"
RESULTS_FILE  = RESULTS_DIR / "results.csv"
SUMMARY_FILE  = RESULTS_DIR / "summary_table.csv"

PROGRESS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

GRID_SIZE  = 7
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# ── Country centroids ──────────────────────────────────────────────────────────
COUNTRY_CENTROIDS = {
    "Afghanistan": (33.93, 67.71), "Albania": (41.15, 20.17),
    "Algeria": (28.03, 1.66), "Angola": (-11.20, 17.87),
    "Argentina": (-38.42, -63.62), "Australia": (-25.27, 133.78),
    "Austria": (47.52, 14.55), "Belgium": (50.50, 4.47),
    "Bolivia": (-16.29, -63.59), "Brazil": (-14.24, -51.93),
    "Bulgaria": (42.73, 25.49), "Cambodia": (12.57, 104.99),
    "Canada": (56.13, -106.35), "Chile": (-35.68, -71.54),
    "China": (35.86, 104.20), "Colombia": (4.57, -74.30),
    "Croatia": (45.10, 15.20), "Czech Republic": (49.82, 15.47),
    "Czechia": (49.82, 15.47), "Denmark": (56.26, 9.50),
    "Ecuador": (-1.83, -78.18), "Egypt": (26.82, 30.80),
    "Eritrea": (15.18, 39.78), "Estonia": (58.60, 25.01),
    "Ethiopia": (9.15, 40.49), "Finland": (61.92, 25.75),
    "France": (46.23, 2.21), "Germany": (51.17, 10.45),
    "Ghana": (7.95, -1.02), "Greece": (39.07, 21.82),
    "Guatemala": (15.78, -90.23), "Hungary": (47.16, 19.50),
    "Iceland": (64.96, -19.02), "India": (20.59, 78.96),
    "Indonesia": (-0.79, 113.92), "Iran": (32.43, 53.69),
    "Iraq": (33.22, 43.68), "Ireland": (53.41, -8.24),
    "Israel": (31.05, 34.85), "Italy": (41.87, 12.57),
    "Japan": (36.20, 138.25), "Jordan": (30.59, 36.24),
    "Kenya": (-0.02, 37.91), "Kyrgyzstan": (41.20, 74.77),
    "Laos": (19.86, 102.50), "Latvia": (56.88, 24.60),
    "Lesotho": (-29.61, 28.23), "Lithuania": (55.17, 23.88),
    "Luxembourg": (49.82, 6.13), "Malaysia": (4.21, 108.96),
    "Mexico": (23.63, -102.55), "Moldova": (47.41, 28.37),
    "Mongolia": (46.86, 103.85), "Morocco": (31.79, -7.09),
    "Mozambique": (-18.67, 35.53), "Myanmar": (21.91, 95.96),
    "Namibia": (-22.96, 18.49), "Nepal": (28.39, 84.12),
    "Netherlands": (52.13, 5.29), "New Zealand": (-40.90, 174.89),
    "Nigeria": (9.08, 8.68), "North Macedonia": (41.61, 21.75),
    "Norway": (60.47, 8.47), "Pakistan": (30.38, 69.35),
    "Panama": (8.54, -80.78), "Paraguay": (-23.44, -58.44),
    "Peru": (-9.19, -75.02), "Philippines": (12.88, 121.77),
    "Poland": (51.92, 19.15), "Portugal": (39.40, -8.22),
    "Romania": (45.94, 24.97), "Russia": (61.52, 105.32),
    "Rwanda": (-1.94, 29.87), "Saudi Arabia": (23.89, 45.08),
    "Senegal": (14.50, -14.45), "Serbia": (44.02, 21.01),
    "Singapore": (1.35, 103.82), "Slovakia": (48.67, 19.70),
    "Slovenia": (46.15, 14.99), "South Africa": (-30.56, 22.94),
    "South Korea": (35.91, 127.77), "Spain": (40.46, -3.75),
    "Sri Lanka": (7.87, 80.77), "Sweden": (60.13, 18.64),
    "Switzerland": (46.82, 8.23), "Taiwan": (23.70, 121.00),
    "Tanzania": (-6.37, 34.89), "Thailand": (15.87, 100.99),
    "Tunisia": (33.89, 9.54), "Turkey": (38.96, 35.24),
    "Uganda": (1.37, 32.29), "Ukraine": (48.38, 31.17),
    "United Kingdom": (55.38, -3.44), "United States": (37.09, -95.71),
    "Uruguay": (-32.52, -55.77), "Vietnam": (14.06, 108.28),
    "Zambia": (-13.13, 27.85), "Zimbabwe": (-19.02, 29.15),
    "Côte d'Ivoire": (7.54, -5.55), "Eswatini": (-26.52, 31.47),
    "Andorra": (42.55, 1.57), "Antigua and Barbuda": (17.06, -61.80),
    "Armenia": (40.07, 45.04), "Aruba": (12.52, -69.97),
    "Azerbaijan": (40.14, 47.58), "Bahamas": (25.03, -77.40),
    "Bahrain": (26.02, 50.55), "Bangladesh": (23.68, 90.35),
    "Barbados": (13.19, -59.54), "Belarus": (53.71, 27.95),
    "Belize": (17.19, -88.49), "Benin": (9.31, 2.32),
    "Bhutan": (27.51, 90.43), "Bosnia and Herzegovina": (43.92, 17.68),
    "Botswana": (-22.33, 24.68), "Brunei": (4.54, 114.73),
    "Burkina Faso": (12.36, -1.53), "Burundi": (-3.37, 29.92),
    "Cabo Verde": (16.54, -23.04), "Cameroon": (7.37, 12.35),
    "Central African Republic": (6.61, 20.94), "Chad": (15.45, 18.73),
    "Comoros": (-11.64, 43.33), "Costa Rica": (9.75, -83.75),
    "Cuba": (21.52, -77.78), "Cyprus": (35.13, 33.43),
    "Democratic Republic of the Congo": (-4.04, 21.76), "Djibouti": (11.83, 42.59),
    "Dominica": (15.41, -61.37), "Dominican Republic": (18.74, -70.16),
    "East Timor": (-8.87, 125.73), "El Salvador": (13.79, -88.90),
    "Equatorial Guinea": (1.65, 10.27), "Fiji": (-17.71, 178.07),
    "Gabon": (-0.80, 11.61), "Gambia": (13.44, -15.31),
    "Georgia": (42.31, 43.36), "Greenland": (71.71, -42.60),
    "Grenada": (12.12, -61.68), "Guinea": (11.75, -15.45),
    "Guinea-Bissau": (11.80, -15.18), "Guyana": (4.86, -58.93),
    "Haiti": (18.97, -72.29), "Honduras": (15.20, -86.24),
    "Jamaica": (18.11, -77.30), "Kazakhstan": (48.02, 66.92),
    "Kuwait": (29.31, 47.48), "Lebanon": (33.85, 35.86),
    "Liberia": (6.43, -9.43), "Libya": (26.34, 17.23),
    "Liechtenstein": (47.14, 9.55), "Madagascar": (-18.77, 46.87),
    "Malawi": (-13.25, 34.30), "Mali": (17.57, -3.99),
    "Malta": (35.94, 14.38), "Marshall Islands": (7.13, 171.18),
    "Mauritania": (21.01, -10.94), "Mauritius": (-20.35, 57.55),
    "Micronesia": (7.43, 150.55), "Montenegro": (42.71, 19.37),
    "Nauru": (-0.53, 166.92), "Nicaragua": (12.87, -85.21),
    "Niger": (17.61, 8.08), "North Korea": (40.34, 127.51),
    "Oman": (21.51, 55.92), "Palau": (7.51, 134.58),
    "Palestine": (31.95, 35.23), "Papua New Guinea": (-6.31, 143.96),
    "Qatar": (25.35, 51.18), "Republic of the Congo": (-0.23, 15.83),
    "Saint Kitts and Nevis": (17.36, -62.78), "Saint Lucia": (13.91, -60.98),
    "Saint Vincent and the Grenadines": (13.25, -61.20), "Samoa": (-13.76, -172.10),
    "San Marino": (43.94, 12.46), "Seychelles": (-4.68, 55.49),
    "Sierra Leone": (8.46, -11.78), "Solomon Islands": (-9.64, 160.16),
    "Somalia": (5.15, 46.20), "South Sudan": (6.88, 31.31),
    "Sudan": (12.86, 30.22), "Suriname": (3.92, -56.03),
    "Syria": (34.80, 38.99), "Tajikistan": (38.86, 71.28),
    "Togo": (8.62, 0.82), "Tonga": (-21.18, -175.20),
    "Trinidad and Tobago": (10.69, -61.22), "Turkmenistan": (38.97, 59.56),
    "Tuvalu": (-7.11, 177.64), "United Arab Emirates": (23.42, 53.85),
    "Uzbekistan": (41.38, 64.59), "Vanuatu": (-15.38, 166.96),
    "Venezuela": (6.42, -66.59), "Yemen": (15.55, 48.52),
}

choices = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas",
    "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei",
    "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon",
    "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia",
    "Comoros", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic",
    "Côte d'Ivoire", "Democratic Republic of the Congo", "Denmark", "Djibouti",
    "Dominica", "Dominican Republic", "East Timor", "Ecuador", "Egypt", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Fiji", "Finland", "France",
    "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Greenland", "Grenada",
    "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary",
    "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein",
    "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Mali", "Malta",
    "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova",
    "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru",
    "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria",
    "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau",
    "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
    "Poland", "Portugal", "Qatar", "Republic of the Congo", "Romania", "Russia",
    "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines",
    "Samoa", "San Marino", "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
    "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia",
    "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan",
    "Suriname", "Eswatini", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan",
    "Tanzania", "Thailand", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia",
    "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam",
    "Yemen", "Zambia", "Zimbabwe", "United States"
]

# ── Model ──────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():  # Apple Silicon GPU
    device = "mps"
else:
    device = "cpu"
print(f"Loading model on {device}...")
model     = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(device).eval()
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
print("Model loaded.")

# ── Core functions ─────────────────────────────────────────────────────────────
def get_score(img_pil):
    inputs = processor(text=choices, images=img_pil, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
    return logits.softmax(dim=1)[0].cpu()

def gini(array):
    array = np.sort(np.abs(array.flatten()))
    n = len(array)
    if np.sum(array) == 0:
        return 0.0
    return (2 * np.sum(np.arange(1, n+1) * array) / (n * np.sum(array))) - (n+1)/n

def occlusion_heatmap(image_pil, target_idx):
    W, H = image_pil.size
    patch_w, patch_h = W // GRID_SIZE, H // GRID_SIZE
    importance = np.zeros((GRID_SIZE, GRID_SIZE))
    baseline = get_score(image_pil)[target_idx].item()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            img_copy = image_pil.copy()
            img_copy.paste((128, 128, 128),
                           (j*patch_w, i*patch_h, (j+1)*patch_w, (i+1)*patch_h))
            importance[i, j] = baseline - get_score(img_copy)[target_idx].item()
    return importance

def counterfactual_k(image_pil, target_idx, heatmap):
    W, H = image_pil.size
    patch_w, patch_h = W // GRID_SIZE, H // GRID_SIZE
    patch_coords = [
        (idx // GRID_SIZE, idx % GRID_SIZE)
        for idx in np.argsort(heatmap.flatten())[::-1]
    ]
    img_copy = image_pil.copy()
    for k, (i, j) in enumerate(patch_coords, start=1):
        img_copy.paste((128, 128, 128),
                       (j*patch_w, i*patch_h, (j+1)*patch_w, (i+1)*patch_h))
        if get_score(img_copy).argmax().item() != target_idx:
            return k
    return GRID_SIZE * GRID_SIZE

def geographic_error_km(true_country, predicted_country):
    if true_country == predicted_country:
        return 0.0
    c1 = COUNTRY_CENTROIDS.get(true_country)
    c2 = COUNTRY_CENTROIDS.get(predicted_country)
    if c1 is None or c2 is None:
        return None
    return round(geodesic(c1, c2).km, 1)

# ── Image discovery ────────────────────────────────────────────────────────────
def find_images():
    """All images anywhere under images/; the country is the parent folder name.
    Works for images/<batch>/<Country>/x.jpg and images/<Country>/x.jpg."""
    paths = sorted(
        p for p in IMAGES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return [(p.relative_to(BASE_DIR).as_posix(), p.parent.name.replace("_", " ")) for p in paths]

# ── Progress tracking ──────────────────────────────────────────────────────────
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()

def save_progress(done_set):
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(sorted(done_set), f)
    tmp.replace(PROGRESS_FILE)  # atomic: a crash mid-write can't corrupt the file

def append_result(row):
    pd.DataFrame([row]).to_csv(
        RESULTS_FILE, mode="a", header=not RESULTS_FILE.exists(), index=False
    )

def merge_old_batch_files():
    """One-off: fold existing progress_*.json / results_*.csv into the single files,
    so already-processed images are not recomputed."""
    if PROGRESS_FILE.exists():
        return
    old_progress = glob.glob(str(BASE_DIR / "progress" / "progress_*.json"))
    old_results  = glob.glob(str(BASE_DIR / "results" / "results_*.csv"))
    if not old_progress and not old_results:
        return
    done = set()
    for f in old_progress:
        with open(f) as fh:
            done |= set(json.load(fh))
    save_progress(done)
    if old_results and not RESULTS_FILE.exists():
        pd.concat([pd.read_csv(f) for f in old_results]).drop_duplicates("image") \
          .to_csv(RESULTS_FILE, index=False)
    print(f"Merged {len(old_progress)} old progress file(s), "
          f"{len(old_results)} old results file(s).")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    merge_old_batch_files()
    done   = load_progress()
    images = find_images()

    unknown = sorted({c for _, c in images if c not in choices})
    if unknown:
        print(f"WARNING: folder names not in `choices` (always counted wrong): {unknown}")

    print(f"Found {len(images)} images in {len({c for _, c in images})} countries.")
    print(f"Already processed: {len(done)} images.\n")

    for n, (img_path, country) in enumerate(images, start=1):
        if img_path in done:
            continue

        try:
            img   = Image.open(BASE_DIR / img_path).convert("RGB").resize((336, 336))
            probs = get_score(img)

            top_idx     = probs.argmax().item()
            top_country = choices[top_idx]
            top_score   = probs[top_idx].item()
            is_correct  = (top_country == country)

            heatmap = occlusion_heatmap(img, top_idx)
            g       = gini(heatmap)
            cf      = counterfactual_k(img, top_idx, heatmap)
            err_km  = geographic_error_km(country, top_country)

            append_result({
                "image":      img_path,
                "truth":      country,
                "predicted":  top_country,
                "correct":    is_correct,
                "confidence": round(top_score, 4),
                "gini":       round(g, 4),
                "cf_k":       cf,
                "error_km":   err_km if err_km is not None else "",
            })
            done.add(img_path)
            save_progress(done)

            err_str = f"{err_km:.0f}km" if err_km is not None else "n/a"
            print(f"[{n}/{len(images)}] {img_path} → {top_country} ({top_score:.3f}) "
                  f"{'✓' if is_correct else '✗'} | Gini: {g:.3f} | CF-k: {cf} | Error: {err_str}")

        except Exception as e:
            print(f"[error] {img_path}: {e}")

    # ── Final summary ──────────────────────────────────────────────────────────
    df = pd.read_csv(RESULTS_FILE)
    summary = (df.groupby("truth")
                 .agg(
                     Total           =("correct", "count"),
                     Accuracy        =("correct", "mean"),
                     Mean_Gini       =("gini",    "mean"),
                     Mean_CF_k       =("cf_k",    "mean"),
                     Mean_Confidence =("confidence", "mean"),
                     Mean_Error_km   =("error_km", "mean"),
                 )
                 .reset_index()
                 .rename(columns={"truth": "Country"}))

    summary["Accuracy"]        = summary["Accuracy"].map("{:.1%}".format)
    summary["Mean_Gini"]       = summary["Mean_Gini"].map("{:.3f}".format)
    summary["Mean_CF_k"]       = summary["Mean_CF_k"].map("{:.1f}".format)
    summary["Mean_Confidence"] = summary["Mean_Confidence"].map("{:.3f}".format)
    summary["Mean_Error_km"]   = summary["Mean_Error_km"].map("{:.0f}km".format)

    print("\n── Final Summary ──────────────────────────────────────────")
    print(summary.to_string(index=False))
    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSaved: {RESULTS_FILE.relative_to(BASE_DIR)}, {SUMMARY_FILE.relative_to(BASE_DIR)}")