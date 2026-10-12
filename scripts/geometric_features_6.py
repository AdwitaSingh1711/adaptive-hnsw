# step7_features.py
import numpy as np
from config import groundtruth_path, results_path

# gt_dist = np.load('gt_distances_k11.npy')  # (2000, 11) exact distances, sorted ascending
gt_dist = np.load(groundtruth_path('gt_distances_k11.npy'))

# Feature 1: margin between 10th and 11th neighbor
margin = gt_dist[:, 10] - gt_dist[:, 9]

# Feature 2: local density = avg distance to 10 nearest neighbors
density = gt_dist[:, :10].mean(axis=1)

# Feature 3: neighborhood spread = dist to 10th - dist to 1st
spread = gt_dist[:, 9] - gt_dist[:, 0]

# np.save('feat_margin.npy', margin)
# np.save('feat_density.npy', density)
# np.save('feat_spread.npy', spread)

np.save(results_path('feat_margin.npy'), margin)
np.save(results_path('feat_density.npy'), density)
np.save(results_path('feat_spread.npy'), spread)


print("margin:", margin[:5])
print("density:", density[:5])
print("spread:", spread[:5])