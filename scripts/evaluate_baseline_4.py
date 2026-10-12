import hnswlib
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from config import index_path, data_path, plots_path, groundtruth_path, results_path

# Load data
# X_query = np.load('X_query.npy')
X_query = np.load(data_path('X_query.npy'))
# gt_indices = np.load(groundtruth_path('gt_indices.npy'))
# gt_indices = np.load('gt_indices.npy')  # shape (2000, 10), exact ground truth
gt_indices = np.load(groundtruth_path('gt_indices_k11.npy'))[:, :10]

# Load the pre-built HNSW index
dim = X_query.shape[1]
index = hnswlib.Index(space='l2', dim=dim)
# index.load_index('hnsw_index.bin')
index.load_index(index_path('hnsw_index.bin'))

ef_search_values = [10, 20, 50, 100, 200]
results = []

for ef in ef_search_values:
    index.set_ef(ef)

    # Time the queries - run one at a time to get realistic per-query latency
    start = time.perf_counter()
    labels, distances = index.knn_query(X_query, k=10)
    end = time.perf_counter()

    total_time = end - start
    avg_query_time_ms = (total_time / len(X_query)) * 1000

    # Compute Recall@10 for every query, then average
    recalls = []
    for i in range(len(X_query)):
        true_neighbors = set(gt_indices[i])
        found_neighbors = set(labels[i])
        overlap = len(true_neighbors & found_neighbors)
        recalls.append(overlap / 10)

    avg_recall = np.mean(recalls)

    results.append({
        'ef_search': ef,
        'recall@10': avg_recall,
        'avg_query_time_ms': avg_query_time_ms
    })

    print(f"ef_search={ef:>4} | Recall@10={avg_recall:.4f} | Avg query time={avg_query_time_ms:.4f} ms")

# Save results table
# with open('baseline_results.json', 'w') as f:
#     json.dump(results, f, indent=2)

with open(results_path('baseline_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Plot ef_search vs Recall@10
ef_vals = [r['ef_search'] for r in results]
recall_vals = [r['recall@10'] for r in results]

plt.figure(figsize=(7, 5))
plt.plot(ef_vals, recall_vals, marker='o', linewidth=2)
plt.xlabel('ef_search')
plt.ylabel('Recall@10')
plt.title('HNSW: ef_search vs Recall@10 (M=16, ef_construction=200)')
plt.grid(True, alpha=0.3)
plt.xscale('log')  # ef_search values span an order of magnitude, log scale reads better
plt.tight_layout()
# plt.savefig('recall_vs_efsearch.png', dpi=150)
plt.savefig(plots_path('recall_vs_efsearch.png'), dpi=150)
plt.show()

print("\nSaved: baseline_results.json, recall_vs_efsearch.png")