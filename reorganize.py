"""
Run ONCE to move existing root-level output files and pipeline scripts
into the new data/groundtruth/index/results/splits/plots/scripts structure.

Safe to re-run — skips files that don't exist or are already moved.
"""
import os
import shutil
from config import (
    ensure_dirs, data_path, groundtruth_path, index_path,
    results_path, splits_path, plots_path, SCRIPTS_DIR, ROOT
)

# filename -> destination-path-function
FILE_MAP = {
    # data
    "X_db.npy": data_path,
    "X_query.npy": data_path,
    "y_db.npy": data_path,
    "y_query.npy": data_path,

    # groundtruth
    "gt_indices.npy": groundtruth_path,
    "gt_distances.npy": groundtruth_path,
    "gt_indices_k11.npy": groundtruth_path,
    "gt_distances_k11.npy": groundtruth_path,

    # index
    "hnsw_index.bin": index_path,

    # results
    "baseline_results.json": results_path,
    "recall_ef20.npy": results_path,
    "is_hard_ef20.npy": results_path,
    "recall_low_ef10.npy": results_path,
    "recall_high_ef100.npy": results_path,
    "gain.npy": results_path,
    "labels_low_ef10.npy": results_path,
    "distances_low_ef10.npy": results_path,
    "distances_low_ef10_k11.npy": results_path,
    "feat_margin.npy": results_path,
    "feat_density.npy": results_path,
    "feat_spread.npy": results_path,
    "cheap_margin.npy": results_path,
    "cheap_density.npy": results_path,
    "cheap_spread.npy": results_path,
    "mean_neighbor_distance.npy": results_path,

    # splits
    "dev_idx.npy": splits_path,
    "test_idx.npy": splits_path,

    # plots
    "recall_vs_efsearch.png": plots_path,
    "feature_scatter.png": plots_path,

    # pipeline scripts from main.py
    "generate_data_1.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
    "create_ground_truth_2.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
    "build_hnsw_3.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
    "evaluate_baseline_4.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
    "find_hard_queries_5.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
    "matrix_6.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
    "geometric_features_6.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
    "plots_6.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
    "gain_7.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
    "quintile_bucketing_8.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
    "test_evaluation_9.py": lambda filename: os.path.join(SCRIPTS_DIR, filename),
}


def main():
    ensure_dirs()
    os.makedirs(SCRIPTS_DIR, exist_ok=True)

    moved, skipped = 0, 0

    for filename, dest_fn in FILE_MAP.items():
        src = os.path.join(ROOT, filename)

        if os.path.exists(src):
            dest = dest_fn(filename)
            shutil.move(src, dest)
            print(f"Moved  {filename:40s} -> {dest}")
            moved += 1
        else:
            skipped += 1

    print(
        f"\nDone. Moved {moved} files, "
        f"{skipped} not found (already moved or never created)."
    )


if __name__ == "__main__":
    main()