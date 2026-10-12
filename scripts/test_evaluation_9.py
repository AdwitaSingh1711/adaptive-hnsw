# step9_test_evaluation.py
import numpy as np
from config import results_path, splits_path

# ---- Load everything ----
# gain = np.load('gain.npy')
# recall_low = np.load('recall_low_ef10.npy')
# recall_high = np.load('recall_high_ef100.npy')
# mean_neighbor_distance = np.load('mean_neighbor_distance.npy')
# dev_idx = np.load('dev_idx.npy')
# test_idx = np.load('test_idx.npy')
gain = np.load(results_path('gain.npy'))
recall_low = np.load(results_path('recall_low_ef10.npy'))
recall_high = np.load(results_path('recall_high_ef100.npy'))
mean_neighbor_distance = np.load(results_path('mean_neighbor_distance.npy'))
dev_idx = np.load(splits_path('dev_idx.npy'))
test_idx = np.load(splits_path('test_idx.npy'))

rng = np.random.default_rng(123)  # different seed from dev, doesn't matter which

# ============================================================
# FREEZE THE THRESHOLD using only the dev set
# ============================================================
dev_signal = mean_neighbor_distance[dev_idx]
FROZEN_PCT = 0.30
threshold = np.quantile(dev_signal, 1 - FROZEN_PCT)  # value cutting top 30% in dev
print(f"Frozen threshold (from dev set): {threshold:.4f}")

# ============================================================
# Apply frozen threshold to TEST set
# ============================================================
test_signal = mean_neighbor_distance[test_idx]
test_gain = gain[test_idx]
test_recall_low = recall_low[test_idx]
test_recall_high = recall_high[test_idx]

escalate_mask = test_signal >= threshold  # boolean, decided using ONLY the frozen number
n_test = len(test_idx)
actual_pct_escalated = escalate_mask.mean()
print(f"Test queries escalated: {escalate_mask.sum()} / {n_test} ({actual_pct_escalated:.1%})")

# ============================================================
# 1. TEST RECALL — four policies
# ============================================================
print("\n=== 1. Test Recall@10 ===")

recall_always_low = test_recall_low.mean()

adaptive_recall_per_query = np.where(escalate_mask, test_recall_high, test_recall_low)
recall_adaptive = adaptive_recall_per_query.mean()

recall_always_high = test_recall_high.mean()

# Random baseline at the SAME fixed budget (k = number escalated by signal), averaged over 100 trials
k = escalate_mask.sum()
n_random_trials = 100
random_recalls = []
random_captured = []
total_test_gain = test_gain.sum()

for _ in range(n_random_trials):
    rand_mask = np.zeros(n_test, dtype=bool)
    rand_positions = rng.choice(n_test, size=k, replace=False)
    rand_mask[rand_positions] = True

    rand_recall = np.where(rand_mask, test_recall_high, test_recall_low).mean()
    random_recalls.append(rand_recall)

    rand_captured_gain = test_gain[rand_mask].sum() / total_test_gain
    random_captured.append(rand_captured_gain)

recall_random = np.mean(random_recalls)

print(f"{'Policy':<25}{'Test Recall@10':>18}{'Queries escalated':>20}")
print(f"{'Always ef=10':<25}{recall_always_low:>18.4f}{'0%':>20}")
print(f"{'Random ' + str(int(FROZEN_PCT*100)) + '%':<25}{recall_random:>18.4f}{f'{actual_pct_escalated:.1%}':>20}")
print(f"{'Signal-guided adaptive':<25}{recall_adaptive:>18.4f}{f'{actual_pct_escalated:.1%}':>20}")
print(f"{'Always ef=100':<25}{recall_always_high:>18.4f}{'100%':>20}")

# ============================================================
# 2. TEST CAPTURED GAIN — signal vs random
# ============================================================
print("\n=== 2. Test Captured Gain ===")

captured_signal = test_gain[escalate_mask].sum() / total_test_gain
captured_random_mean = np.mean(random_captured)

print(f"Signal-guided captured gain: {captured_signal:.3f}")
print(f"Random captured gain (avg of {n_random_trials} trials): {captured_random_mean:.3f}")
print(f"Signal advantage over random: {captured_signal - captured_random_mean:+.3f}")

# ============================================================
# 3. BOOTSTRAP CI for (adaptive recall - random recall)
# ============================================================
print("\n=== 3. Bootstrap 95% CI: adaptive - random (per-query recall) ===")

# Per-query recall difference for ONE random draw isn't well-defined query-by-query
# since "random" picks different queries each trial. So we bootstrap over test queries
# instead: resample query indices with replacement, recompute both policies' mean recall
# on that resample, and look at the distribution of the difference.

n_bootstrap = 5000
diffs = []

for _ in range(n_bootstrap):
    boot_positions = rng.integers(0, n_test, size=n_test)  # resample WITH replacement

    boot_escalate_mask = escalate_mask[boot_positions]
    boot_recall_low = test_recall_low[boot_positions]
    boot_recall_high = test_recall_high[boot_positions]

    boot_adaptive_recall = np.where(boot_escalate_mask, boot_recall_high, boot_recall_low).mean()

    # random policy on this bootstrap sample: same k (proportional), one random draw per bootstrap
    boot_k = boot_escalate_mask.sum()
    if boot_k == 0 or boot_k == n_test:
        continue
    boot_rand_mask = np.zeros(n_test, dtype=bool)
    boot_rand_positions = rng.choice(n_test, size=boot_k, replace=False)
    boot_rand_mask[boot_rand_positions] = True
    boot_random_recall = np.where(boot_rand_mask, boot_recall_high, boot_recall_low).mean()

    diffs.append(boot_adaptive_recall - boot_random_recall)

diffs = np.array(diffs)
ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
mean_diff = diffs.mean()

print(f"Mean recall difference (adaptive - random): {mean_diff:+.4f}")
print(f"95% Bootstrap CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")

if ci_lower > 0:
    print(">> CI stays above zero: signal-guided policy reliably beats random at this budget.")
elif ci_upper < 0:
    print(">> CI stays below zero: signal-guided policy is reliably WORSE than random. Report this honestly.")
else:
    print(">> CI includes zero: cannot confidently distinguish signal-guided from random at this budget.")