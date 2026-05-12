"""
Entity density analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

def create_entity_analysis_figure(policlaim_entities, ct22_entities):
    """
    Create entity density vs performance gain comparison figure.

    Args:
        policlaim_entities: list of lists, entity lists for the PoliClaim dataset
        ct22_entities: list of lists, entity lists for the CT22 dataset
    """

    pc_avg = np.mean([len(ents) for ents in policlaim_entities])
    ct_avg = np.mean([len(ents) for ents in ct22_entities])

    pc_f1_improvement = 0.21
    ct_f1_improvement = 0.65

    fig, ax1 = plt.subplots(figsize=(10, 7))

    datasets = ['PoliClaim', 'CT22']
    entity_densities = [pc_avg, ct_avg]

    color = 'tab:blue'
    ax1.set_xlabel('Dataset', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Average Entities per Text', color=color, fontsize=14, fontweight='bold')
    bars = ax1.bar(datasets, entity_densities, alpha=0.6, color=color, width=0.5)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.set_ylim([0, max(entity_densities) * 1.3])

    for bar, val in zip(bars, entity_densities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{val:.2f}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('CC-G4o F1 Improvement', color=color, fontsize=14, fontweight='bold')

    f1_improvements = [pc_f1_improvement, ct_f1_improvement]
    line = ax2.plot(datasets, f1_improvements, color=color, marker='o',
                    markersize=15, linewidth=4, label='F1 Improvement')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2.set_ylim([0, max(f1_improvements) * 1.5])

    for i, (x, y) in enumerate(zip(datasets, f1_improvements)):
        ax2.text(i, y + 0.05, f'+{y:.2f}',
                 ha='center', va='bottom', fontsize=14,
                 color=color, fontweight='bold')

    ratio = ct_avg / pc_avg
    improvement_ratio = ct_f1_improvement / pc_f1_improvement

    textstr = f'Entity Density Ratio: {ratio:.2f}×\nPerformance Gain Ratio: {improvement_ratio:.2f}×'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.5, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='center', bbox=props)

    plt.title('Entity Density Drives Context Augmentation Effectiveness',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    return fig


def create_entity_distribution_figure(policlaim_entities, ct22_entities):
    """Create entity distribution histogram."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PoliClaim
    pc_counts = [len(e) for e in policlaim_entities]
    pc_mean = np.mean(pc_counts)
    pc_median = np.median(pc_counts)

    axes[0].hist(pc_counts, bins=range(0, max(pc_counts) + 2),
                 alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.5)
    axes[0].axvline(pc_mean, color='red', linestyle='--',
                    linewidth=3, label=f'Mean: {pc_mean:.2f}')
    axes[0].axvline(pc_median, color='orange', linestyle='--',
                    linewidth=3, label=f'Median: {pc_median:.1f}')
    axes[0].set_xlabel('Number of Entities per Text', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=14, fontweight='bold')
    axes[0].set_title('PoliClaim (Entity-Sparse)', fontsize=16, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].tick_params(labelsize=11)
    axes[0].grid(axis='y', alpha=0.3)

    # CT22
    ct_counts = [len(e) for e in ct22_entities]
    ct_mean = np.mean(ct_counts)
    ct_median = np.median(ct_counts)

    axes[1].hist(ct_counts, bins=range(0, max(ct_counts) + 2),
                 alpha=0.7, color='lightcoral', edgecolor='black', linewidth=1.5)
    axes[1].axvline(ct_mean, color='red', linestyle='--',
                    linewidth=3, label=f'Mean: {ct_mean:.2f}')
    axes[1].axvline(ct_median, color='orange', linestyle='--',
                    linewidth=3, label=f'Median: {ct_median:.1f}')
    axes[1].set_xlabel('Number of Entities per Text', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=14, fontweight='bold')
    axes[1].set_title('CT22 (Entity-Rich)', fontsize=16, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].tick_params(labelsize=11)
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle('Entity Distribution Across Datasets',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def create_statistics_table(policlaim_entities, ct22_entities, policlaim_pronouns, ct22_pronouns):
    """
    Generate statistics table data.

    Args:
        policlaim_pronouns: list of pronoun counts per text
        ct22_pronouns: list of pronoun counts per text
    """

    pc_entity_counts = [len(e) for e in policlaim_entities]
    ct_entity_counts = [len(e) for e in ct22_entities]

    stats_data = {
        'Metric': [
            'Avg entities per text',
            'Median entities per text',
            'Texts with ≥1 entity (%)',
            'Texts with 0 entities (%)',
            'Avg pronouns per text',
            'Pronoun/Entity ratio',
            'CC-G4o ΔF1',
            'CC-M ΔF1'
        ],
        'PoliClaim': [
            f'{np.mean(pc_entity_counts):.2f}',
            f'{np.median(pc_entity_counts):.1f}',
            f'{len([c for c in pc_entity_counts if c > 0]) / len(pc_entity_counts) * 100:.1f}%',
            f'{len([c for c in pc_entity_counts if c == 0]) / len(pc_entity_counts) * 100:.1f}%',
            f'{np.mean(policlaim_pronouns):.2f}',
            f'{np.mean(policlaim_pronouns) / np.mean(pc_entity_counts):.2f}' if np.mean(
                pc_entity_counts) > 0 else 'N/A',
            '+0.21',
            '-1.48'
        ],
        'CT22': [
            f'{np.mean(ct_entity_counts):.2f}',
            f'{np.median(ct_entity_counts):.1f}',
            f'{len([c for c in ct_entity_counts if c > 0]) / len(ct_entity_counts) * 100:.1f}%',
            f'{len([c for c in ct_entity_counts if c == 0]) / len(ct_entity_counts) * 100:.1f}%',
            f'{np.mean(ct22_pronouns):.2f}',
            f'{np.mean(ct22_pronouns) / np.mean(ct_entity_counts):.2f}' if np.mean(ct_entity_counts) > 0 else 'N/A',
            '+0.65',
            '-0.18'
        ]
    }

    df = pd.DataFrame(stats_data)

    pc_avg = np.mean(pc_entity_counts)
    ct_avg = np.mean(ct_entity_counts)
    df['CT22/PoliClaim Ratio'] = [
        f'{ct_avg / pc_avg:.2f}×' if pc_avg > 0 else 'N/A',
        f'{np.median(ct_entity_counts) / np.median(pc_entity_counts):.2f}×' if np.median(
            pc_entity_counts) > 0 else 'N/A',
        '-',
        '-',
        f'{np.mean(ct22_pronouns) / np.mean(policlaim_pronouns):.2f}×' if np.mean(policlaim_pronouns) > 0 else 'N/A',
        '-',
        f'{0.65 / 0.21:.2f}×',
        '-'
    ]

    return df


def perform_statistical_tests(policlaim_entities, ct22_entities):
    """Perform statistical tests on entity counts between datasets."""

    pc_counts = [len(e) for e in policlaim_entities]
    ct_counts = [len(e) for e in ct22_entities]

    results = {}

    # 1. Independent samples t-test
    t_stat, p_value = stats.ttest_ind(ct_counts, pc_counts)
    results['t_test'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

    # 2. Effect size (Cohen's d)
    mean_diff = np.mean(ct_counts) - np.mean(pc_counts)
    pooled_std = np.sqrt((np.std(ct_counts) ** 2 + np.std(pc_counts) ** 2) / 2)
    cohens_d = mean_diff / pooled_std
    results['cohens_d'] = cohens_d

    # 3. Mann-Whitney U test (non-parametric alternative)
    u_stat, p_value_mw = stats.mannwhitneyu(ct_counts, pc_counts, alternative='greater')
    results['mann_whitney'] = {
        'u_statistic': u_stat,
        'p_value': p_value_mw,
        'significant': p_value_mw < 0.05
    }

    return results


def print_results_summary(stats_df, test_results):
    """Print results summary."""

    print("=" * 80)
    print("ENTITY DENSITY ANALYSIS RESULTS")
    print("=" * 80)

    print("\nSTATISTICS TABLE:")
    print(stats_df.to_string(index=False))

    print("\n\nSTATISTICAL TESTS:")
    print(f"\n1. Independent Samples T-Test:")
    print(f"   H0: Entity density is equal across datasets")
    print(f"   t-statistic: {test_results['t_test']['t_statistic']:.4f}")
    print(f"   p-value: {test_results['t_test']['p_value']:.2e}")
    print(f"   Result: {'REJECT H0 (p < 0.05)' if test_results['t_test']['significant'] else 'FAIL TO REJECT H0'}")

    print(f"\n2. Effect Size (Cohen's d):")
    print(f"   d = {test_results['cohens_d']:.4f}")
    if abs(test_results['cohens_d']) < 0.2:
        interpretation = "negligible"
    elif abs(test_results['cohens_d']) < 0.5:
        interpretation = "small"
    elif abs(test_results['cohens_d']) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    print(f"   Interpretation: {interpretation.upper()} effect")

    print(f"\n3. Mann-Whitney U Test (non-parametric):")
    print(f"   U-statistic: {test_results['mann_whitney']['u_statistic']:.2f}")
    print(f"   p-value: {test_results['mann_whitney']['p_value']:.2e}")
    print(
        f"   Result: {'CT22 > PoliClaim (p < 0.05)' if test_results['mann_whitney']['significant'] else 'No significant difference'}")

    print("\n" + "=" * 80)

def load_entity_lists(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entity_lists = []
    for item in data:
        linked = item.get("linked_entities") or {}
        n_entities = len(linked)
        entity_lists.append(['entity'] * n_entities)
    return entity_lists

if __name__ == "__main__":
    np.random.seed(42)

    policlaim_entities = load_entity_lists("data/linked_entities/policlaim_linked_entities_test.json")

    ct22_entities = load_entity_lists("data/linked_entities/CT22_linked_entities_test_gold.json")
    # ct22_pronouns = [max(0, int(np.random.normal(1.5, 0.8))) for _ in range(500)]

    print("Creating visualizations...")

    fig1 = create_entity_analysis_figure(policlaim_entities, ct22_entities)
    fig1.savefig('output/visualizations/entity_density_vs_performance.png',
                 dpi=300, bbox_inches='tight')
    print("   Saved: entity_density_vs_performance.png")

    fig2 = create_entity_distribution_figure(policlaim_entities, ct22_entities)
    fig2.savefig('output/visualizations/entity_distribution.png',
                 dpi=300, bbox_inches='tight')
    print("   Saved: entity_distribution.png")

    # stats_df = create_statistics_table(policlaim_entities, ct22_entities,
    #                                    policlaim_pronouns, ct22_pronouns)
    # stats_df.to_csv('/mnt/user-data/outputs/entity_statistics.csv', index=False)
    # print("   Saved: entity_statistics.csv")
    #
    # test_results = perform_statistical_tests(policlaim_entities, ct22_entities)
    #
    # print_results_summary(stats_df, test_results)

    print("\nAnalysis complete!")