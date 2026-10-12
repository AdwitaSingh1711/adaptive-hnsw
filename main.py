import argparse
import os
import subprocess
import sys
import time

from config import ensure_dirs, SCRIPTS_DIR, ROOT

PIPELINE = [
    "generate_data_1.py",
    "create_ground_truth_2.py",
    "build_hnsw_3.py",
    "evaluate_baseline_4.py",
    "find_hard_queries_5.py",
    "matrix_6.py",
    "geometric_features_6.py",
    "plots_6.py",
    "gain_7.py",
    "quintile_bucketing_8.py",
    "test_evaluation_9.py",
]


def parse_args():
    p = argparse.ArgumentParser(description="Run the adaptive HNSW pipeline.")
    p.add_argument("--n-samples", type=int, default=None, help="Total points (DB + query)")
    p.add_argument("--n-features", type=int, default=None)
    p.add_argument("--n-centers", type=int, default=None)
    p.add_argument("--cluster-std", type=float, default=None)
    p.add_argument("--db-fraction", type=float, default=None, help="Fraction of samples used as the HNSW DB (rest are queries)")
    p.add_argument("--m", type=int, default=None, help="HNSW M parameter")
    p.add_argument("--ef-construction", type=int, default=None)
    p.add_argument("--tag", type=str, default=None, help="Explicit run tag (folder name under runs/). Auto-derived from params if omitted.")
    return p.parse_args()


def build_env(args):
    """
    Overlay CLI args onto the current environment as HNSW_* variables.
    Each subprocess independently reads these via config.py's
    os.environ.get(...) fallbacks — this is how a value set here in
    main.py's process reaches the separate Python processes it spawns.
    """
    env = os.environ.copy()

    mapping = {
        "n_samples": "HNSW_N_SAMPLES",
        "n_features": "HNSW_N_FEATURES",
        "n_centers": "HNSW_N_CENTERS",
        "cluster_std": "HNSW_CLUSTER_STD",
        "db_fraction": "HNSW_DB_FRACTION",
        "m": "HNSW_M",
        "ef_construction": "HNSW_EF_CONSTRUCTION",
        "tag": "HNSW_RUN_TAG",
    }
    for arg_name, env_name in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            env[env_name] = str(value)

    return env


def run_script(script_name: str, env: dict) -> bool:
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('=' * 60)

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", f"scripts.{script_name[:-3]}"],
        capture_output=False,
        cwd=ROOT,
        env=env,
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"\n[FAILED] {script_name} exited with code {result.returncode}")
        return False

    print(f"[OK] {script_name} completed in {elapsed:.2f}s")
    return True


def main():
    args = parse_args()
    env = build_env(args)

    # Re-import config-derived values AFTER setting env, just for the banner —
    # subprocesses each compute their own RUN_TAG independently from the env.
    os.environ.update(env)
    from importlib import reload
    import config as _config
    reload(_config)  # pick up any env overrides set above for this banner print
    _config.ensure_dirs()

    print("Starting adaptive HNSW pipeline...")
    _config.print_run_config()
    print(f"Run output folder: {_config.RUN_ROOT}\n")

    overall_start = time.perf_counter()

    for script in PIPELINE:
        if not run_script(script, env):
            print(f"\nPipeline halted at '{script}'. Fix the error above and rerun.")
            sys.exit(1)

    total_elapsed = time.perf_counter() - overall_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete. Total time: {total_elapsed:.2f}s")
    print(f"Results in: {_config.RUN_ROOT}")
    print('=' * 60)


if __name__ == "__main__":
    main()