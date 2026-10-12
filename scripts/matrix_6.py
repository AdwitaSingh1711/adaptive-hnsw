# step_gain.py
import hnswlib
import numpy as np
from config import data_path, index_path, groundtruth_path, results_path, ensure_dirs
ensure_dirs()

# X_query = np.load('X_query.npy')
X_query = np.load(data_path('X_query.npy'))
# gt_indices = np.load('gt_indices_k11.npy')[:, :10]
gt_indices = np.load(groundtruth_path('gt_indices_k11.npy'))[:, :10]

dim = X_query.shape[1]
index = hnswlib.Index(space='l2', dim=dim)
# index.load_index('hnsw_index.bin')
index.load_index(index_path('hnsw_index.bin'))

def recall_at_ef(ef):
    index.set_ef(ef)
    labels, distances = index.knn_query(X_query, k=10)
    recalls = np.array([
        len(set(gt_indices[i]) & set(labels[i])) / 10
        for i in range(len(X_query))
    ])
    return recalls, labels, distances

recall_low, labels_low, dist_low = recall_at_ef(10)
recall_high, labels_high, dist_high = recall_at_ef(100)

gain = recall_high - recall_low

# np.save('recall_low_ef10.npy', recall_low)
# np.save('recall_high_ef100.npy', recall_high)
# np.save('gain.npy', gain)
# np.save('labels_low_ef10.npy', labels_low)
# np.save('distances_low_ef10.npy', dist_low)
np.save(results_path('recall_low_ef10.npy'), recall_low)
np.save(results_path('recall_high_ef100.npy'), recall_high)
np.save(results_path('gain.npy'), gain)
np.save(results_path('labels_low_ef10.npy'), labels_low)
np.save(results_path('distances_low_ef10.npy'), dist_low)

print(f"Mean recall @ ef=10:  {recall_low.mean():.4f}")
print(f"Mean recall @ ef=100: {recall_high.mean():.4f}")
print(f"Mean gain:            {gain.mean():.4f}")
print(f"Queries with gain > 0: {(gain > 0).sum()} / {len(gain)} ({(gain > 0).mean():.1%})")
print(f"Queries with gain >= 0.3: {(gain >= 0.3).sum()} / {len(gain)}")