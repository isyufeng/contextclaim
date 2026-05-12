import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Read the three Excel files
file1 = pd.read_excel('/mnt/user-data/uploads/Quality_Assessment_CT22_n40_by_Annotator1.xlsx')
file2 = pd.read_excel('/mnt/user-data/uploads/Quality_Assessment_CT22_n40_by_Xiangyan.xlsx')
file3 = pd.read_excel('/mnt/user-data/uploads/Quality_Assessment_CT22_n40_by_YG.xlsx')

print("=" * 80)
print("CONTEXT QUALITY ANALYSIS")
print("=" * 80)

# Extract ratings
dimensions = ['Factual Accuracy', 'Relevance', 'Signal Clarity', 'Usefulness']
systems = ['GPT-4o', 'Mistral']

# Prepare data for both systems
gpt4o_ratings = {
    'Annotator1': file1[dimensions].values,
    'Annotator2': file2[dimensions].values,
    'YG': file3[dimensions].values
}

mistral_ratings = {
    'Annotator1': file1[[c + '.1' for c in dimensions]].values,
    'Annotator2': file2[[c + '.1' for c in dimensions]].values,
    'YG': file3[[c + '.1' for c in dimensions]].values
}

all_ratings = {
    'GPT-4o': gpt4o_ratings,
    'Mistral': mistral_ratings
}

# Calculate average ratings across annotators
results_summary = []

for system, ratings_dict in all_ratings.items():
    print(f"\n{'=' * 80}")
    print(f"SYSTEM: {system}")
    print(f"{'=' * 80}")

    # Stack all annotator ratings (40 items × 4 dimensions × 3 annotators)
    all_ann_ratings = np.stack([ratings_dict['Annotator1'],
                                ratings_dict['Annotator2'],
                                ratings_dict['YG']], axis=0)

    # Calculate mean across annotators (3 × 40 × 4) -> (40 × 4)
    mean_ratings = np.mean(all_ann_ratings, axis=0)
    std_ratings = np.std(all_ann_ratings, axis=0)

    print("\n1. AVERAGE RATINGS BY DIMENSION")
    print("-" * 80)
    print(f"{'Dimension':<20} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}")
    print("-" * 80)

    for i, dim in enumerate(dimensions):
        dim_mean = np.mean(mean_ratings[:, i])
        dim_std = np.mean(std_ratings[:, i])
        dim_min = np.min(mean_ratings[:, i])
        dim_max = np.max(mean_ratings[:, i])

        print(f"{dim:<20} {dim_mean:>8.2f} {dim_std:>8.2f} {dim_min:>6.2f} {dim_max:>6.2f}")

        results_summary.append({
            'System': system,
            'Dimension': dim,
            'Mean': dim_mean,
            'Std': dim_std,
            'Min': dim_min,
            'Max': dim_max
        })

    # Overall score
    overall_mean = np.mean(mean_ratings)
    overall_std = np.mean(std_ratings)
    print(f"{'OVERALL':<20} {overall_mean:>8.2f} {overall_std:>8.2f}")

    # Distribution analysis
    print("\n2. RATING DISTRIBUTION")
    print("-" * 80)
    for i, dim in enumerate(dimensions):
        print(f"\n{dim}:")
        all_dim_ratings = all_ann_ratings[:, :, i].flatten()
        counts = np.bincount(all_dim_ratings.astype(int), minlength=4)[1:]  # Ignore 0
        percentages = counts / len(all_dim_ratings) * 100
        print(f"  Poor (1):       {counts[0]:3d} ({percentages[0]:5.1f}%)")
        print(f"  Acceptable (2): {counts[1]:3d} ({percentages[1]:5.1f}%)")
        print(f"  Good (3):       {counts[2]:3d} ({percentages[2]:5.1f}%)")

    # Item-level analysis
    print("\n3. ITEM-LEVEL STATISTICS")
    print("-" * 80)

    # Items with highest consensus (lowest std)
    item_stds = np.mean(std_ratings, axis=1)
    item_means = np.mean(mean_ratings, axis=1)

    # Sort by mean rating (quality)
    sorted_indices = np.argsort(-item_means)

    print("\nTop 5 Highest Quality Items (by mean rating):")
    print(f"{'Rank':<6} {'Item ID':<22} {'Mean':>8} {'Std':>8}")
    print("-" * 50)
    for rank, idx in enumerate(sorted_indices[:5], 1):
        item_id = file1.iloc[idx]['ID']
        print(f"{rank:<6} {item_id:<22} {item_means[idx]:>8.2f} {item_stds[idx]:>8.2f}")

    print("\nTop 5 Lowest Quality Items (by mean rating):")
    print(f"{'Rank':<6} {'Item ID':<22} {'Mean':>8} {'Std':>8}")
    print("-" * 50)
    for rank, idx in enumerate(sorted_indices[-5:][::-1], 1):
        item_id = file1.iloc[idx]['ID']
        print(f"{rank:<6} {item_id:<22} {item_means[idx]:>8.2f} {item_stds[idx]:>8.2f}")

# Create visualizations
print("\n\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Figure 1: Mean ratings comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, system in enumerate(systems):
    ratings_dict = all_ratings[system]
    all_ann_ratings = np.stack([ratings_dict['Annotator1'],
                                ratings_dict['Annotator2'],
                                ratings_dict['YG']], axis=0)
    mean_ratings = np.mean(all_ann_ratings, axis=0)

    # Calculate means for each dimension
    dim_means = [np.mean(mean_ratings[:, i]) for i in range(4)]
    dim_stds = [np.std(mean_ratings[:, i]) for i in range(4)]

    x_pos = np.arange(len(dimensions))
    axes[ax_idx].bar(x_pos, dim_means, yerr=dim_stds, capsize=5, alpha=0.7,
                     color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
    axes[ax_idx].set_xlabel('Dimension', fontsize=11)
    axes[ax_idx].set_ylabel('Mean Rating', fontsize=11)
    axes[ax_idx].set_title(f'{system} - Mean Ratings by Dimension', fontsize=12, fontweight='bold')
    axes[ax_idx].set_xticks(x_pos)
    axes[ax_idx].set_xticklabels(dimensions, rotation=15, ha='right')
    axes[ax_idx].set_ylim([1, 3])
    axes[ax_idx].axhline(y=2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    axes[ax_idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/quality_comparison_bars.png', dpi=150, bbox_inches='tight')
print("✓ Saved: quality_comparison_bars.png")

# Figure 2: Distribution heatmaps
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for sys_idx, system in enumerate(systems):
    ratings_dict = all_ratings[system]
    all_ann_ratings = np.stack([ratings_dict['Annotator1'],
                                ratings_dict['Annotator2'],
                                ratings_dict['YG']], axis=0)

    for dim_idx, dim in enumerate(dimensions):
        all_dim_ratings = all_ann_ratings[:, :, dim_idx].flatten()
        counts = np.bincount(all_dim_ratings.astype(int), minlength=4)[1:]
        percentages = counts / len(all_dim_ratings) * 100

        ax = axes[sys_idx, dim_idx]
        bars = ax.bar(['1\n(Poor)', '2\n(Accept.)', '3\n(Good)'], percentages,
                      color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.7)
        ax.set_ylim([0, 100])
        ax.set_ylabel('Percentage (%)', fontsize=9)
        ax.set_title(f'{system}\n{dim}', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/rating_distributions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: rating_distributions.png")

# Figure 3: System comparison
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(dimensions))
width = 0.35

gpt4o_dict = all_ratings['GPT-4o']
mistral_dict = all_ratings['Mistral']

gpt4o_all = np.stack([gpt4o_dict['Annotator1'], gpt4o_dict['Annotator2'], gpt4o_dict['YG']], axis=0)
mistral_all = np.stack([mistral_dict['Annotator1'], mistral_dict['Annotator2'], mistral_dict['YG']], axis=0)

gpt4o_means = [np.mean(gpt4o_all[:, :, i]) for i in range(4)]
mistral_means = [np.mean(mistral_all[:, :, i]) for i in range(4)]

bars1 = ax.bar(x - width / 2, gpt4o_means, width, label='GPT-4o', alpha=0.8, color='#3498db')
bars2 = ax.bar(x + width / 2, mistral_means, width, label='Mistral', alpha=0.8, color='#9b59b6')

ax.set_xlabel('Dimension', fontsize=11)
ax.set_ylabel('Mean Rating', fontsize=11)
ax.set_title('GPT-4o vs Mistral: Context Quality Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dimensions, rotation=15, ha='right')
ax.set_ylim([1, 3])
ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/system_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: system_comparison.png")

# Statistical comparison
print("\n" + "=" * 80)
print("STATISTICAL COMPARISON (GPT-4o vs Mistral)")
print("=" * 80)

for dim_idx, dim in enumerate(dimensions):
    gpt4o_dim = gpt4o_all[:, :, dim_idx].flatten()
    mistral_dim = mistral_all[:, :, dim_idx].flatten()

    # Wilcoxon signed-rank test (paired)
    statistic, p_value = stats.wilcoxon(gpt4o_dim, mistral_dim)

    gpt4o_mean = np.mean(gpt4o_dim)
    mistral_mean = np.mean(mistral_dim)
    diff = gpt4o_mean - mistral_mean

    print(f"\n{dim}:")
    print(f"  GPT-4o mean:  {gpt4o_mean:.3f}")
    print(f"  Mistral mean: {mistral_mean:.3f}")
    print(f"  Difference:   {diff:+.3f}")
    print(
        f"  p-value:      {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)