# step8_quintile_and_captured_gain.py
import numpy as np
from config import SEED, results_path, splits_path, ensure_dirs

ensure_dirs()

# # ---- Load data ----
# gain = np.load('gain.npy')                          # (2000,)
# recall_low = np.load('recall_low_ef10.npy')          # (2000,)
# recall_high = np.load('recall_high_ef100.npy')       # (2000,)
# distances_low = np.load('distances_low_ef10.npy')    # (2000, 10) approx distances @ ef=10

gain = np.load(results_path('gain.npy'))
recall_low = np.load(results_path('recall_low_ef10.npy'))
recall_high = np.load(results_path('recall_high_ef100.npy'))
distances_low = np.load(results_path('distances_low_ef10.npy'))

# Rename per the earlier correction — this is mean neighbour distance, not density
mean_neighbor_distance = distances_low.mean(axis=1)

n = len(gain)
rng = np.random.default_rng(SEED)

# ---- Dev/test split (70/30), done ONCE, before any further analysis ----
perm = rng.permutation(n)
n_dev = int(0.7 * n)
dev_idx = perm[:n_dev]
test_idx = perm[n_dev:]

print(f"Dev set: {len(dev_idx)} queries | Test set: {len(test_idx)} queries\n")

# ============================================================
# Step 8A: Quintile bucket analysis (on DEV set only)
# ============================================================
dev_signal = mean_neighbor_distance[dev_idx]
dev_gain = gain[dev_idx]

quintile_edges = np.quantile(dev_signal, [0.2, 0.4, 0.6, 0.8])
quintile_labels = np.digitize(dev_signal, quintile_edges)  # 0..4

print("=== Step 8A: Quintile analysis (dev set) ===")
print(f"{'Quintile':<20}{'Mean signal':>15}{'Mean gain':>15}")
for q in range(5):
    mask = quintile_labels == q
    label = ["Lowest 20%", "20-40%", "40-60%", "60-80%", "Highest 20%"][q]
    print(f"{label:<20}{dev_signal[mask].mean():>15.4f}{dev_gain[mask].mean():>15.4f}")

# ============================================================
# Step 8B: Captured gain vs random baseline (on DEV set)
# ============================================================
print("\n=== Step 8B: Captured gain (dev set) ===")
total_dev_gain = dev_gain.sum()
percentages = [0.10, 0.20, 0.30, 0.40, 0.50]

# Sort dev queries by signal descending (higher mean_neighbor_distance -> escalate first)
sorted_dev_order = np.argsort(-dev_signal)  # indices into dev_idx/dev_signal/dev_gain

print(f"{'% escalated':<15}{'Captured gain (signal)':>25}{'Captured gain (random, avg of 100)':>38}")
n_random_trials = 100
for pct in percentages:
    k = int(pct * len(dev_idx))

    # Signal-based selection: top-k by mean_neighbor_distance
    top_k_dev_positions = sorted_dev_order[:k]
    captured_signal = dev_gain[top_k_dev_positions].sum() / total_dev_gain

    # Random baseline: average over many random k-subsets
    random_captures = []
    for _ in range(n_random_trials):
        rand_positions = rng.choice(len(dev_idx), size=k, replace=False)
        random_captures.append(dev_gain[rand_positions].sum() / total_dev_gain)
    captured_random_mean = np.mean(random_captures)

    print(f"{pct*100:>5.0f}%{'':<9}{captured_signal:>25.3f}{captured_random_mean:>38.3f}")

# ============================================================
# Step 8C: Adaptive recall vs % escalated (on DEV set, for policy selection)
# ============================================================
print("\n=== Step 8C: Adaptive recall curve (dev set) ===")
dev_recall_low = recall_low[dev_idx]
dev_recall_high = recall_high[dev_idx]

print(f"{'% escalated':<15}{'Adaptive Recall@10':>20}")
print(f"{'0% (always ef=10)':<15}{dev_recall_low.mean():>20.4f}")

for pct in percentages:
    k = int(pct * len(dev_idx))
    top_k_dev_positions = sorted_dev_order[:k]

    adaptive_recall = dev_recall_low.copy()
    adaptive_recall[top_k_dev_positions] = dev_recall_high[top_k_dev_positions]

    print(f"{pct*100:>5.0f}%{'':<9}{adaptive_recall.mean():>20.4f}")

print(f"{'100% (always ef=100)':<15}{dev_recall_high.mean():>20.4f}")

# Save split indices and chosen threshold info for the frozen-policy test evaluation
# np.save('dev_idx.npy', dev_idx)
# np.save('test_idx.npy', test_idx)
# np.save('mean_neighbor_distance.npy', mean_neighbor_distance)
np.save(splits_path('dev_idx.npy'), dev_idx)
np.save(splits_path('test_idx.npy'), test_idx)
np.save(results_path('mean_neighbor_distance.npy'), mean_neighbor_distance)

print("\nSaved dev_idx.npy, test_idx.npy, mean_neighbor_distance.npy")
print("Pick your escalation % from the curves above, then freeze it and evaluate on the TEST set only.")