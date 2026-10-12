# step6_label_hardness.py
import hnswlib
import numpy as np
from config import index_path, data_path, results_path, groundtruth_path

# X_query = np.load('X_query.npy')
# gt_indices = np.load('gt_indices_k11.npy')[:, :10] 
#  # top-10 for recall calc
X_query = np.load(data_path('X_query.npy'))
gt_indices = np.load(groundtruth_path('gt_indices_k11.npy'))[:, :10] 

dim = X_query.shape[1]
index = hnswlib.Index(space='l2', dim=dim)
# index.load_index('hnsw_index.bin')
index.load_index(index_path('hnsw_index.bin'))
index.set_ef(20)

labels, distances = index.knn_query(X_query, k=10)

recalls = np.array([
    len(set(gt_indices[i]) & set(labels[i])) / 10
    for i in range(len(X_query))
])

is_hard = recalls < 0.8

print(f"Hard queries: {is_hard.sum()} / {len(is_hard)} ({is_hard.mean():.1%})")

# np.save('recall_ef20.npy', recalls)
np.save(results_path('recall_ef20.npy'), recalls)
# np.save('is_hard_ef20.npy', is_hard)
np.save(results_path('is_hard_ef20.npy'), is_hard)