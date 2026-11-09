# StreetCLIP occlusion & counterfactual analysis

Runs [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) zero-shot country classification on a folder of street-view images and, for each image, computes:

- **prediction & confidence**: top country out of 193 candidates
- **Gini**: how concentrated the occlusion heatmap is (7×7 grid); high = the decision relies on a few regions
- **CF-k**: how many of the most important patches must be greyed out before the prediction changes
- **error_km**: distance between the centroids of the true and predicted country (0 if correct)

## 1. Setup

Requires [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda).

```bash
git clone https://github.com/zenoromio/image-geolocation stclip
cd stclip
conda env create -f environment.yml
conda activate stclip
```

The same environment works on macOS (Apple Silicon GPU via MPS) and Linux (NVIDIA GPU via CUDA). Without a GPU it falls back to CPU, which works but is much slower.

## 2. Folder layout

```
stclip/
├── run_experiments.py         ← main experiment (whole dataset)
├── occlusion.py               ← single-image occlusion heatmap
├── counterfactual.py          ← single-image counterfactual analysis
├── neighbour_rank.py          ← geographic error as neighbour rank
├── create_dataset.py          ← how the dataset was sampled
├── download_streetview.py     ← downloads the exact dataset
├── dataset_coordinates.csv    ← locations and headings of all images
├── environment.yml
├── images/                    ← your dataset (not included)
│   ├── Chile/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── South_Africa/
│   └── ...
├── progress/                  ← created automatically
├── results/                   ← created automatically
└── hf_cache/                  ← model weights, downloaded automatically
```

**Dataset format:** one subfolder per country inside `images/`, containing `.jpg`, `.jpeg` or `.png` files. The folder name is the ground-truth label and must match a country name in `choices` in `run_experiments.py`; underscores are treated as spaces (`South_Africa` → `South Africa`). The script prints a warning for any folder name it does not recognise.

## 3. Dataset

### Download the exact dataset used in our experiments

`dataset_coordinates.csv` lists the location (`lat`, `lon`) and camera direction (`heading`) of every image in our dataset: 500 images for each of 50 countries. To download the same images:

```bash
python download_streetview.py dataset_coordinates.csv --out streetview_dataset_fixed --size 336x336
```

This requires a Google Maps Platform API key with the **Street View Static API** enabled.

The script is safe to interrupt: rerun the same command and it continues where it stopped thanks to a checkpoint system.

The images are saved in `streetview_dataset_fixed/images/<Country>/`. Move that folder into the project so `run_experiments.py` finds it:

```bash
mv streetview_dataset_fixed/images images
```

> **Note:** Google regularly replaces Street View imagery. A location may now show a newer panorama than the one we used, so a download made today can differ slightly from our dataset. The capture date in `download_log.csv` shows which panorama you received.

### How the dataset was sampled

`create_dataset.py` shows how the locations were chosen: for each country, random points are sampled uniformly inside the country borders (Natural Earth polygons), points without nearby Street View coverage are rejected, and each image gets a random camera heading. You do not need to run it to reproduce our results; it is included to document the sampling procedure. Running it again produces a *different* random dataset.

## 4. Run

```bash
conda activate stclip
python run_experiments.py
```

On the first run the model is downloaded into `hf_cache/` inside the project folder; later runs reuse it.

The first output line shows which device is used: `Loading model on cuda...`, `mps...` or `cpu...`.

## 5. Single-image analysis

To inspect one image in detail instead of running the whole dataset, use the two standalone scripts. Both load the same model (and reuse the cache in `hf_cache/`).

### Occlusion heatmap

```bash
python occlusion.py
```

Splits the image into a 7×7 grid and greys out each of the 49 patches one at a time. For each patch it records how much the confidence of the predicted country drops:

- **positive value** → the patch supports the prediction (hiding it lowers confidence)
- **negative value** → hiding the patch *raises* confidence

The output is the heatmap overlaid on the image, with the score of each patch.

### Counterfactual analysis

```bash
python counterfactual.py
```

Greys out patches one at a time, most important first (ranked by the occlusion heatmap), until the predicted country changes. It reports **k**, the number of patches that had to be removed. A small k means the prediction depends on a few regions; k = 49 means the prediction never changed.

## 6. Neighbour rank

`neighbour_rank.py` measures how far a wrong prediction is from the true country, as a **rank** instead of a distance in km: the neighbour rank of the predicted country is **1 + the number of countries closer to the true country than the predicted one is**.

- **0**: correct prediction
- **1**: a neighbour (shared land border, or less than 5 km away, e.g. Malaysia → Singapore)
- **higher**: more countries lie closer than the predicted one

How it works:

- Country borders come from [Natural Earth](https://www.naturalearthdata.com/) (1:10m Admin 0 – Countries).
- Parts of a country more than 1,000 km from its mainland are ignored (e.g. French Guiana for France), while nearby islands are kept (e.g. Sicily for Italy).
- Distances are the shortest **border-to-border** distances, measured on the globe (no map projection), so routes across oceans and the date line are correct.

Use it from Python:

```python
from neighbour_rank import neighbour_rank, border_km, closest_points, countries_closer

neighbour_rank("Colombia", "Honduras")                 # 7
neighbour_rank("Colombia", "Honduras", verbose=True)   # also prints the countries in between
border_km("Italy", "Tunisia")                          # distance in km
closest_points("Italy", "Tunisia")                     # the two border points used (lat, lon)
countries_closer("Japan", "Taiwan")                    # countries closer than the prediction
```

Loading the borders takes about a minute; each country's distances are computed once and then cached. Requires `geopandas`, `scipy` and `pyproj`.

## 7. Output

| File | Content |
| --- | --- |
| `results/results.csv` | One row per image: `image`, `truth`, `predicted`, `correct`, `confidence`, `gini`, `cf_k`, `error_km`, `importance` (the 49 occlusion values, row by row) |
| `results/summary_table.csv` | Per-country totals: accuracy, mean Gini, mean CF-k, mean confidence, mean error |
| `progress/progress.json` | List of images already processed |

### Resuming

Results are written after every image. If the run is interrupted, just run `python run_experiments.py` again: images listed in `progress/progress.json` are skipped.

To start from scratch, delete `progress/progress.json` and `results/results.csv`.