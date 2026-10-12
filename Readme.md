# Adaptive HNSW Query Escalation

## Overview

Approximate Nearest Neighbor (ANN) search involves a trade-off between search quality and computational cost. In HNSW, the `ef_search` parameter controls this trade-off: increasing it generally improves Recall@10 but requires more search work.

This project investigates whether HNSW search effort can be allocated adaptively at the query level.

Instead of running every query with a high `ef_search`, the approach is:

1. Run a cheap initial search with `ef_search = 10`.
2. Compute a query-level signal from the initial search results.
3. Use the signal to identify queries expected to benefit more from additional search.
4. Escalate only those queries to `ef_search = 100`.
5. Compare the adaptive policy against randomly escalating the same proportion of queries.

The central question here is:

> Can a cheap query-level signal identify which HNSW queries will benefit most from additional search effort?

---

## Key Result

Across three experimental configurations, the signal-guided policy consistently outperformed randomly escalating the same proportion of queries.

| Configuration      | Random Recall@10 | Adaptive Recall@10 | Adaptive − Random |
| ------------------ | ---------------: | -----------------: | ----------------: |
| 10K samples, 64D   |           0.8920 |         **0.9053** |       **+0.0133** |
| 50K samples, 64D   |           0.7551 |         **0.7757** |       **+0.0206** |
| 100K samples, 128D |           0.6219 |         **0.6389** |       **+0.0170** |

The bootstrap confidence interval for the adaptive policy's improvement over random remained entirely above zero in every experiment:

| Configuration      | 95% CI for Adaptive − Random |
| ------------------ | ---------------------------- |
| 10K samples, 64D   | [+0.0072, +0.0195]           |
| 50K samples, 64D   | [+0.0163, +0.0247]           |
| 100K samples, 128D | [+0.0140, +0.0199]           |

These results support the conclusion that, under the tested configurations, **signal-guided allocation of additional HNSW search effort is more effective than random allocation at the same search budget**.

---

## Motivation

A fixed HNSW search parameter treats every query equally.

However, queries can differ substantially in how much they benefit from additional search. Some already recover most of their true nearest neighbors using a low `ef_search`, while others gain significantly when given a larger search budget.

The objective of this project is therefore not simply to predict query recall. Instead, it focuses on predicting which queries are likely to gain the most if additional HNSW search effort is allocated to them.

This distinction is important. A query with low recall is not necessarily the same as a query that can be most effectively improved by increasing `ef_search`.

---

# Methodology

## 1. Dataset Generation

Synthetic clustered vector datasets are generated with configurable:

* Number of samples
* Number of features
* Number of cluster centers
* Cluster standard deviation
* Database/query split
* Random seed

The experiments use an 80/20 database/query split.

The configurations evaluated so far are:

| Run                      | Total Samples | Dimensions | Database | Queries |
| ------------------------ | ------------: | ---------: | -------: | ------: |
| Small                    |        10,000 |         64 |    8,000 |   2,000 |
| Medium                   |        50,000 |         64 |   40,000 |  10,000 |
| Large / high-dimensional |       100,000 |        128 |   80,000 |  20,000 |

Each run is stored separately using a configuration-based run tag, for example:

```text
runs/
├── n10000_d64_...
├── n50000_d64_c20_std5.0/
└── n100000_d128_c20_std5.0/
```

This prevents different experimental configurations from overwriting each other's results.

---

## 2. Exact Ground Truth

Exact nearest neighbors are computed using brute-force Euclidean search.

For each query, the project stores:

* Ground-truth neighbor indices
* Ground-truth neighbor distances

Ground truth is computed with `k=11`, allowing Recall@10 evaluation while also supporting distance-based feature calculations.

---

## 3. HNSW Index

An HNSW index is built over the database vectors using:

```text
M = 16
ef_construction = 200
```

Search quality is evaluated at multiple `ef_search` values:

```text
10, 20, 50, 100, 200
```

The adaptive policy uses:

```text
Cheap search:      ef_search = 10
Escalated search:  ef_search = 100
```

---

# Baseline HNSW Performance

## 10K samples, 64 dimensions

| `ef_search` | Recall@10 |
| ----------: | --------: |
|          10 |    0.9116 |
|          20 |    0.9876 |
|          50 |    1.0000 |
|         100 |    1.0000 |
|         200 |    1.0000 |

At this scale, HNSW reaches near-perfect recall relatively quickly.

---

## 50K samples, 64 dimensions

| `ef_search` | Recall@10 | Avg. Query Time |
| ----------: | --------: | --------------: |
|          10 |    0.6480 |       0.0055 ms |
|          20 |    0.8148 |       0.0083 ms |
|          50 |    0.9613 |       0.0144 ms |
|         100 |    0.9943 |       0.0250 ms |
|         200 |    0.9995 |       0.0426 ms |

The larger database creates a substantially stronger accuracy/search-effort trade-off.

---

## 100K samples, 128 dimensions

| `ef_search` | Recall@10 | Avg. Query Time |
| ----------: | --------: | --------------: |
|          10 |    0.4792 |       0.0137 ms |
|          20 |    0.6490 |       0.0213 ms |
|          50 |    0.8598 |       0.0410 ms |
|         100 |    0.9566 |       0.0654 ms |
|         200 |    0.9917 |       0.1130 ms |

This configuration represents a substantially more difficult retrieval setting, with both a larger database and higher dimensionality.

---

# Measuring the Benefit of Escalation

For every query, Recall@10 is measured at both search budgets:

```text
Recall_low  = Recall@10 at ef_search = 10
Recall_high = Recall@10 at ef_search = 100
```

The benefit available from escalation is defined as:

```text
Gain = Recall_high - Recall_low
```

A query with high gain benefits more from receiving the expensive search.

This allows the project to frame adaptive HNSW search as a resource-allocation problem:

> Given a limited budget for expensive searches, can the system allocate that budget to the queries with the largest expected gain?

---

# Query-Level Signals

The project explores geometric characteristics of the initial nearest-neighbor results, including:

* Margin
* Density
* Spread

The signals are evaluated both against raw recall and against **gain from escalation**.

The strongest and most consistent relationship is observed for density-derived signals.

### Correlation between cheap density and gain

| Configuration | Correlation with Gain |
| ------------- | --------------------: |
| 10K, 64D      |                 0.219 |
| 50K, 64D      |             **0.361** |
| 100K, 128D    |                 0.240 |

All reported density/gain relationships are statistically significant in the corresponding experiments.

The relationship is not identical across configurations, so the project does **not** assume that signal strength increases monotonically with dataset size. Instead, the results show that useful query-level information persists across multiple retrieval difficulty settings.

---

# Development/Test Protocol

To avoid selecting the policy directly on the test set, query indices are split into:

```text
70% Development
30% Test
```

The frozen policy is compared against:

1. **Always `ef=10`**
2. **Random escalation at the same budget**
3. **Signal-guided adaptive escalation**
4. **Always `ef=100`**

---

# Signal Ranking Analysis

The query signal generally ranks queries by expected gain.

For example, in the 100K/128D experiment:

| Signal Quintile |  Mean Gain |
| --------------- | ---------: |
| Lowest 20%      |     0.4095 |
| 20–40%          |     0.4544 |
| 40–60%          |     0.4799 |
| 60–80%          |     0.5033 |
| Highest 20%     | **0.5430** |

The ordering is monotonic: queries with higher signal values receive progressively greater average benefit from escalation.

A similar pattern appears in the 50K/64D experiment:

| Signal Quintile |  Mean Gain |
| --------------- | ---------: |
| Lowest 20%      |     0.2726 |
| 20–40%          |     0.3261 |
| 40–60%          |     0.3478 |
| 60–80%          |     0.3690 |
| Highest 20%     | **0.4223** |

This provides direct evidence that the signal contains information about the expected value of allocating additional search effort.

---

# Final Test Results

## 10K samples, 64 dimensions

The frozen policy escalated approximately 26.8% of held-out queries.

| Policy                     | Test Recall@10 | Queries Escalated |
| -------------------------- | -------------: | ----------------: |
| Always `ef=10`             |         0.8527 |                0% |
| Random escalation          |         0.8920 |             26.8% |
| **Signal-guided adaptive** |     **0.9053** |         **26.8%** |
| Always `ef=100`            |         0.9997 |              100% |

```text
Adaptive − Random: +0.0132 mean per-query recall
95% Bootstrap CI: [+0.0072, +0.0195]
```

The confidence interval remains above zero.

---

## 50K samples, 64 dimensions

The frozen policy escalated approximately 30.3% of held-out queries.

| Policy                     | Test Recall@10 | Queries Escalated |
| -------------------------- | -------------: | ----------------: |
| Always `ef=10`             |         0.6510 |                0% |
| Random escalation          |         0.7551 |             30.3% |
| **Signal-guided adaptive** |     **0.7757** |         **30.3%** |
| Always `ef=100`            |         0.9944 |              100% |

```text
Adaptive − Random: +0.0205 mean per-query recall
95% Bootstrap CI: [+0.0163, +0.0247]
```

Again, the confidence interval remains entirely above zero.

---

## 100K samples, 128 dimensions

The frozen policy escalated approximately 29.5% of held-out queries.

| Policy                     | Test Recall@10 | Queries Escalated |
| -------------------------- | -------------: | ----------------: |
| Always `ef=10`             |         0.4814 |                0% |
| Random escalation          |         0.6219 |             29.5% |
| **Signal-guided adaptive** |     **0.6389** |         **29.5%** |
| Always `ef=100`            |         0.9573 |              100% |

```text
Adaptive − Random: +0.0169 mean per-query recall
95% Bootstrap CI: [+0.0140, +0.0199]
```

The adaptive policy continues to outperform random escalation despite the substantially more difficult retrieval setting.

---

# Summary Across Experiments

| Configuration | Low Budget | Random |   Adaptive | High Budget |
| ------------- | ---------: | -----: | ---------: | ----------: |
| 10K, 64D      |     0.8527 | 0.8920 | **0.9053** |      0.9997 |
| 50K, 64D      |     0.6510 | 0.7551 | **0.7757** |      0.9944 |
| 100K, 128D    |     0.4814 | 0.6219 | **0.6389** |      0.9573 |

The main result is consistent:

> **At the same escalation budget, allocating expensive HNSW searches using a query-level signal outperforms randomly allocating the same expensive search budget.**

---

# Project Structure

```text
adaptivehnsw/
│
├── config.py
├── main.py
├── reorganize.py
│
├── scripts/
│   ├── generate_data_1.py
│   ├── create_ground_truth_2.py
│   ├── build_hnsw_3.py
│   ├── evaluate_baseline_4.py
│   ├── find_hard_queries_5.py
│   ├── matrix_6.py
│   ├── geometric_features_6.py
│   ├── plots_6.py
│   ├── gain_7.py
│   ├── quintile_bucketing_8.py
│   └── test_evaluation_9.py
│
└── runs/
    ├── <run_tag_1>/
    │   ├── data/
    │   ├── groundtruth/
    │   ├── index/
    │   ├── results/
    │   ├── splits/
    │   └── plots/
    │
    ├── n50000_d64_c20_std5.0/
    └── n100000_d128_c20_std5.0/
```

Each experiment writes its outputs to an independent run directory based on its configuration.

---

# Running the Project

Install the dependencies:

```bash
pip install numpy scikit-learn hnswlib scipy matplotlib
```

Configure the experiment parameters in `config.py`, including:

```python
N_SAMPLES
N_FEATURES
N_CENTERS
CLUSTER_STD
DB_FRACTION
SEED
```

Then run the complete pipeline from the project root:

```bash
python main.py
```

The pipeline executes:

```text
1. generate_data_1.py
2. create_ground_truth_2.py
3. build_hnsw_3.py
4. evaluate_baseline_4.py
5. find_hard_queries_5.py
6. matrix_6.py
7. geometric_features_6.py
8. plots_6.py
9. gain_7.py
10. quintile_bucketing_8.py
11. test_evaluation_9.py
```

---

# Limitations

This project is an experimental study rather than a production-ready adaptive ANN serving system.

Important limitations include:

* The datasets are synthetic and may not represent every real-world embedding distribution.
* Only selected HNSW configurations are evaluated.
* The adaptive policy uses fixed cheap and expensive search budgets (`ef=10` and `ef=100`).
* The current signal is heuristic and not a learned query-difficulty model.
* The 100K experiment changes both dataset size and dimensionality relative to the 50K experiment, so it should be interpreted as a larger, higher-dimensional stress test rather than a pure dataset-size comparison.
* End-to-end production latency would need to include the overhead of computing the query-level signal and potentially executing two searches for escalated queries.

---

# Future Work

Potential extensions include:

* Evaluating a clean `100K × 64D` configuration to isolate the effect of dataset size.
* Testing additional dataset sizes and dimensions.
* Evaluating real embedding datasets.
* Comparing alternative HNSW configurations.
* Learning the escalation policy with a supervised model.
* Incorporating latency directly into the optimization objective.
* Testing more than two search budgets.
* Extending the approach to other ANN indexes.

---

# Conclusion

The experiments demonstrate that HNSW search effort does not need to be allocated uniformly across all queries.

Across three configurations ranging from a small 10K-vector dataset to a 100K-vector, 128-dimensional retrieval problem, a cheap query-level signal consistently identified queries that benefited more from increased search effort.

At matched escalation budgets, the signal-guided adaptive policy outperformed random escalation in every test configuration, with bootstrap confidence intervals for the adaptive advantage remaining entirely above zero.

These results suggest that **query-aware allocation of ANN search effort can improve the accuracy obtained from a limited expensive-search budget compared with uniform or random allocation**.
