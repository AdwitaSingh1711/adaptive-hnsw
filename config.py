"""
Central configuration for the adaptive HNSW project.
Import from this in every script instead of hardcoding paths or seeds.
"""
import os

# ---- Master seed. Change this in exactly one place. ----
SEED = 42

N_SAMPLES = int(os.environ.get("HNSW_N_SAMPLES", 10_000))
N_FEATURES = int(os.environ.get("HNSW_N_FEATURES", 64))
N_CENTERS = int(os.environ.get("HNSW_N_CENTERS", 20))
CLUSTER_STD = float(os.environ.get("HNSW_CLUSTER_STD", 5))

# ---- DB/query split ----
# Fraction of samples that go into the HNSW database; the rest become queries.
DB_FRACTION = float(os.environ.get("HNSW_DB_FRACTION", 0.8))  # 80% DB / 20% query
 
# ---- HNSW build parameters ----
HNSW_M = int(os.environ.get("HNSW_M", 16))
HNSW_EF_CONSTRUCTION = int(os.environ.get("HNSW_EF_CONSTRUCTION", 200))
 
# ---- Run tagging ----
# Every distinct parameter combination gets its own output folder under runs/,
# so testing at a different scale (e.g. n_samples=50_000) never overwrites
# results from a previous scale (e.g. n_samples=10_000). Auto-derived from
# params unless you explicitly set HNSW_RUN_TAG (main.py's --tag flag).
_default_tag = f"n{N_SAMPLES}_d{N_FEATURES}_c{N_CENTERS}_std{CLUSTER_STD}"
RUN_TAG = os.environ.get("HNSW_RUN_TAG", _default_tag)

# ---- Directory structure ----
# ROOT is wherever THIS file lives, regardless of which script imports it
# or what directory you run python from — so this stays correct even
# though the pipeline scripts themselves now live in scripts/.
ROOT = os.path.dirname(os.path.abspath(__file__))

SCRIPTS_DIR = os.path.join(ROOT, "scripts")      # all pipeline .py files
# DATA_DIR = os.path.join(ROOT, "data")           # X_db, X_query, y_db, y_query
# GROUNDTRUTH_DIR = os.path.join(ROOT, "groundtruth")   # gt_indices*, gt_distances*
# INDEX_DIR = os.path.join(ROOT, "index")          # hnsw_index.bin
# RESULTS_DIR = os.path.join(ROOT, "results")       # gain.npy, recall_*.npy, baseline_results.json
# SPLITS_DIR = os.path.join(ROOT, "splits")         # dev_idx.npy, test_idx.npy
# PLOTS_DIR = os.path.join(ROOT, "plots")          # all .png outputs

RUN_ROOT = os.path.join(ROOT, "runs", RUN_TAG)  # everything below is per-run-tag
 
DATA_DIR = os.path.join(RUN_ROOT, "data")             # X_db, X_query, y_db, y_query
GROUNDTRUTH_DIR = os.path.join(RUN_ROOT, "groundtruth")  # gt_indices*, gt_distances*
INDEX_DIR = os.path.join(RUN_ROOT, "index")            # hnsw_index.bin
RESULTS_DIR = os.path.join(RUN_ROOT, "results")         # gain.npy, recall_*.npy, etc.
SPLITS_DIR = os.path.join(RUN_ROOT, "splits")           # dev_idx.npy, test_idx.npy
PLOTS_DIR = os.path.join(RUN_ROOT, "plots")   

ALL_DIRS = [DATA_DIR, GROUNDTRUTH_DIR, INDEX_DIR, RESULTS_DIR, SPLITS_DIR, PLOTS_DIR]


def ensure_dirs():
    """Call once at the start of any script that writes files."""
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)

def print_run_config():
    """Call at the start of a script for a quick sanity-check log line."""
    print(f"[config] RUN_TAG={RUN_TAG} | N_SAMPLES={N_SAMPLES} N_FEATURES={N_FEATURES} "
          f"N_CENTERS={N_CENTERS} CLUSTER_STD={CLUSTER_STD} DB_FRACTION={DB_FRACTION} "
          f"SEED={SEED}")


# ---- Convenience path helpers ----
def data_path(filename):
    return os.path.join(DATA_DIR, filename)


def groundtruth_path(filename):
    return os.path.join(GROUNDTRUTH_DIR, filename)


def index_path(filename):
    return os.path.join(INDEX_DIR, filename)


def results_path(filename):
    return os.path.join(RESULTS_DIR, filename)


def splits_path(filename):
    return os.path.join(SPLITS_DIR, filename)


def plots_path(filename):
    return os.path.join(PLOTS_DIR, filename)