# StreetCLIP occlusion & counterfactual analysis

Runs [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) zero-shot country classification on a folder of street-view images and, for each image, computes:

- **prediction & confidence** — top country out of 196 candidates
- **Gini** — how concentrated the occlusion heatmap is (7×7 grid); high = the decision relies on a few regions
- **CF-k** — how many of the most important patches must be greyed out before the prediction changes
- **error_km** — distance between the centroids of the true and predicted country (0 if correct)

## 1. Setup

Requires [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda).

```bash
git clone <repo-url> stclip
cd stclip
conda env create -f environment.yml
conda activate stclip
```

The same environment works on macOS (Apple Silicon GPU via MPS) and Linux (NVIDIA GPU via CUDA). Without a GPU it falls back to CPU, which works but is much slower.

To update the environment after `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

## 2. Folder layout

```
stclip/
├── run.py
├── environment.yml
├── images/              ← your dataset (not included)
│   ├── Algeria/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── South_Africa/
│   └── ...
├── progress/            ← created automatically
├── results/             ← created automatically
└── hf_cache/            ← model weights, downloaded automatically (~1.7 GB)
```

**Dataset format:** one subfolder per country inside `images/`, containing `.jpg`, `.jpeg` or `.png` files. The folder name is the ground-truth label and must match a country name in `choices` in `run.py`; underscores are treated as spaces (`South_Africa` → `South Africa`). The script prints a warning for any folder name it does not recognise.

## 3. Run

```bash
conda activate stclip
python run.py
```

On the first run the model is downloaded into `hf_cache/` inside the project folder; later runs reuse it.

The first output line shows which device is used: `Loading model on cuda...`, `mps...` or `cpu...`.

### On a SLURM cluster (e.g. DAS)

- Put the project in scratch space (e.g. `/var/scratch/<user>/stclip`), not your home directory: the model cache is ~1.7 GB and home quotas are small.
- Do not run on the head node — request a GPU node through SLURM:

```bash
srun --gres=gpu:1 --time=12:00:00 python run.py
```

Exact flags (time limits, GPU type constraints) depend on the cluster; check its documentation. To check that the GPU is visible before a long run:

```bash
srun --gres=gpu:1 python -c "import torch; print(torch.cuda.is_available())"
```

## 4. Output

| File | Content |
| --- | --- |
| `results/results.csv` | One row per image: `image`, `truth`, `predicted`, `correct`, `confidence`, `gini`, `cf_k`, `error_km` |
| `results/summary_table.csv` | Per-country totals: accuracy, mean Gini, mean CF-k, mean confidence, mean error |
| `progress/progress.json` | List of images already processed |

### Resuming

Results are written after every image. If the run is interrupted, just run `python run.py` again: images listed in `progress/progress.json` are skipped.

To start from scratch, delete `progress/progress.json` and `results/results.csv`.

## 5. Troubleshooting

**Model download stuck at 0%**
Hugging Face's Xet downloader can hang on some networks/VPNs. Stop with Ctrl+C and run:

```bash
HF_HUB_DISABLE_XET=1 python run.py
```

**`Warning: You are sending unauthenticated requests to the HF Hub`**
Harmless. Optionally set a token (`huggingface-cli login`) for faster downloads.

**Prints `cpu` on a GPU machine**
On a cluster: make sure you are on a GPU node (see above). Otherwise check that `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.

**Store the model somewhere else**
Set `HF_HOME` before running, e.g. `HF_HOME=/path/to/cache python run.py`.