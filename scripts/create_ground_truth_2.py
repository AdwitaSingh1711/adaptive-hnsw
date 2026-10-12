# rebuild_ground_truth_k11.py
from sklearn.neighbors import NearestNeighbors
from config import data_path, groundtruth_path, ensure_dirs
import numpy as np

ensure_dirs()

# X_db = np.load('X_db.npy')
# X_query = np.load('X_query.npy')

X_db = np.load(data_path('X_db.npy'))
X_query = np.load(data_path('X_query.npy'))

nn = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean', n_jobs=-1)
nn.fit(X_db)
distances, indices = nn.kneighbors(X_query)  # now shape (2000, 11)

# np.save('gt_indices_k11.npy', indices)
# np.save('gt_distances_k11.npy', distances)
np.save(groundtruth_path('gt_indices_k11.npy'), indices)
np.save(groundtruth_path('gt_distances_k11.npy'), distances)
print("Saved k=11 ground truth:", indices.shape)