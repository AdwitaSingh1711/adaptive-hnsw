import hnswlib
import numpy as np
from config import data_path, index_path, ensure_dirs, HNSW_M, HNSW_EF_CONSTRUCTION

ensure_dirs()

# Load the DB set generated earlier
# X_db = np.load('X_db.npy')
X_db = np.load(data_path('X_db.npy'))

n_samples, dim = X_db.shape  # 8000, 64

# Initialize the index — 'l2' matches the euclidean distance used in NearestNeighbors
index = hnswlib.Index(space='l2', dim=dim)

# Initialize with construction params
# index.init_index(max_elements=n_samples, M=16, ef_construction=200)
index.init_index(max_elements=n_samples, M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, random_seed=42)
index.set_num_threads(1)   # forces single-threaded insertion → deterministic graph

# Add data — labels are just 0..7999, matching X_db's row order
# (this MUST match the indexing used in your ground truth, since gt_indices
# refers to positional indices into X_db)
index.add_items(X_db, np.arange(n_samples))

# Save the built index to disk
# index.save_index('hnsw_index.bin')
index.save_index(index_path('hnsw_index.bin'))

print(f"Index built with {index.get_current_count()} elements, dim={dim}")