# step8_scatter.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, pearsonr
from config import plots_path, data_path, results_path

recalls = np.load(results_path('recall_ef20.npy'))
margin = np.load(results_path('feat_margin.npy'))
density = np.load(results_path('feat_density.npy'))
spread = np.load(results_path('feat_spread.npy'))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, feat, name in zip(axes, [margin, density, spread], ['Margin', 'Density', 'Spread']):
    ax.scatter(feat, recalls, alpha=0.3, s=10)
    ax.set_xlabel(name)
    ax.set_ylabel('Recall@10')
    r, p = pearsonr(feat, recalls)
    ax.set_title(f'{name} vs Recall\n r={r:.3f}, p={p:.4f}')

plt.tight_layout()
# plt.savefig('feature_scatter.png', dpi=150)
plt.savefig(plots_path('feature_scatter.png'), dpi=150)
plt.show()

print(f"Margin  correlation with recall: r={pearsonr(margin, recalls)[0]:.3f}")
print(f"Density correlation with recall: r={pearsonr(density, recalls)[0]:.3f}")
print(f"Spread  correlation with recall: r={pearsonr(spread, recalls)[0]:.3f}")