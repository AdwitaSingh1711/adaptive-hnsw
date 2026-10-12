from sklearn.datasets import make_blobs
import numpy as np
from config import (
    SEED, ensure_dirs, data_path, print_run_config,
    N_SAMPLES, N_FEATURES, N_CENTERS, CLUSTER_STD, DB_FRACTION
)

ensure_dirs()
print_run_config()

# Generate clustered data
X, y = make_blobs(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    centers=N_CENTERS,
    cluster_std=CLUSTER_STD,
    random_state=SEED  # was hardcoded to 42 before — now tracks the shared SEED
)

print(X.shape, y.shape)

# Shuffle before splitting so both sets have a mix of clusters
rng = np.random.default_rng(SEED)
indices = rng.permutation(N_SAMPLES)

n_db = int(N_SAMPLES * DB_FRACTION)  # was hardcoded to 8000 before
db_indices = indices[:n_db]
query_indices = indices[n_db:]

X_db, y_db = X[db_indices], y[db_indices]
X_query, y_query = X[query_indices], y[query_indices]

print("DB set:", X_db.shape)
print("Query set:", X_query.shape)

np.save(data_path('X_db.npy'), X_db)
np.save(data_path('X_query.npy'), X_query)
np.save(data_path('y_db.npy'), y_db)
np.save(data_path('y_query.npy'), y_query)