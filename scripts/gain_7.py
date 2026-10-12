# cheap_features_ef10.py
import numpy as np
from scipy.stats import pearsonr
from config import results_path, ensure_dirs
ensure_dirs()

# distances_low = np.load('distances_low_ef10.npy')  # (2000, 10) approx distances from ef=10
# gain = np.load('gain.npy')
distances_low = np.load(results_path('distances_low_ef10.npy'))
gain = np.load(results_path('gain.npy'))

# hnswlib space='l2' returns squared euclidean — fine, ranking/shape still valid

# Cheap Feature 1: margin between approx 10th and 9th neighbor
# (we only have 10 back from ef=10 query with k=10, not an 11th,
#  so use the last-vs-second-last gap as a boundary-sharpness proxy)
cheap_margin = distances_low[:, 9] - distances_low[:, 8]

# Cheap Feature 2: approx local density = mean of the 10 returned distances
# cheap_density = distances_low.mean(axis=1)
mean_neighbor_distance = distances_low.mean(axis=1)

# Cheap Feature 3: approx spread = 10th - 1st returned distance
cheap_spread = distances_low[:, 9] - distances_low[:, 0]

# np.save('cheap_margin.npy', cheap_margin)
# np.save('cheap_density.npy', mean_neighbor_distance)
# np.save('cheap_spread.npy', cheap_spread)
np.save(results_path('cheap_margin.npy'), cheap_margin)
np.save(results_path('cheap_density.npy'), mean_neighbor_distance)
np.save(results_path('cheap_spread.npy'), cheap_spread)

for name, feat in [('margin', cheap_margin), ('density', mean_neighbor_distance), ('spread', cheap_spread)]:
    r, p = pearsonr(feat, gain)
    print(f"Cheap {name:8s} vs gain: r={r:.3f}, p={p:.4g}")