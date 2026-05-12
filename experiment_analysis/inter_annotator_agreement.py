import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


# Helper function for Fleiss' Kappa
def calculate_fleiss_kappa(rating_matrix):
    """
    Calculate Fleiss' Kappa for multiple raters.
    rating_matrix: (n_items × n_raters) array
    """
    n_items, n_raters = rating_matrix.shape
    categories = np.unique(rating_matrix)
    n_categories = len(categories)

    # Create frequency matrix (items × categories)
    freq_matrix = np.zeros((n_items, n_categories))
    for i in range(n_items):
        for j, cat in enumerate(categories):
            freq_matrix[i, j] = np.sum(rating_matrix[i, :] == cat)

    # Calculate P_i (agreement for each item)
    P_i = (np.sum(freq_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # Calculate P_j (proportion of all assignments to category j)
    P_j = np.sum(freq_matrix, axis=0) / (n_items * n_raters)
    P_e = np.sum(P_j ** 2)

    # Fleiss' Kappa
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


# Read the three Excel files
print("=" * 80)
print("CONTEXT SUMMARY QUALITY ASSESSMENT - INTER-ANNOTATOR AGREEMENT ANALYSIS")
print("=" * 80)

file1 = pd.read_excel('/mnt/user-data/uploads/Quality_Assessment_CT22_n40_by_Annotator1.xlsx')
file2 = pd.read_excel('/mnt/user-data/uploads/Quality_Assessment_CT22_n40_by_Xiangyan.xlsx')
file3 = pd.read_excel('/mnt/user-data/uploads/Quality_Assessment_CT22_n40_by_YG.xlsx')

print("\n1. DATA OVERVIEW")
print("-" * 80)
print(f"Number of items evaluated: {len(file1)}")
print(f"Number of annotators: 3")
print(f"Dimensions evaluated: 4 (Factual Accuracy, Relevance, Signal Clarity, Usefulness)")
print(f"Systems evaluated: 2 (GPT-4o, Mistral)")
print(f"Rating scale: 1-3 (1=Poor, 2=Acceptable, 3=Good)")

# Extract ratings for both systems
dimensions = ['Factual Accuracy', 'Relevance', 'Signal Clarity', 'Usefulness']
systems = ['GPT4o', 'Mistral']

# Prepare data structure
results = {}

for system_idx, system in enumerate(systems):
    print(f"\n{'=' * 80}")
    print(f"SYSTEM: {system.upper()}")
    print(f"{'=' * 80}")

    # Extract ratings for this system
    if system_idx == 0:  # GPT4o
        cols = ['Factual Accuracy', 'Relevance', 'Signal Clarity', 'Usefulness']
        a1_ratings = file1[cols].values
        a2_ratings = file2[cols].values
        a3_ratings = file3[cols].values
    else:  # Mistral
        cols = ['Factual Accuracy.1', 'Relevance.1', 'Signal Clarity.1', 'Usefulness.1']
        a1_ratings = file1[cols].values
        a2_ratings = file2[cols].values
        a3_ratings = file3[cols].values

    results[system] = {
        'annotator1': a1_ratings,
        'annotator2': a2_ratings,
        'annotator3': a3_ratings
    }

    # Calculate per-dimension metrics
    print("\n2. PER-DIMENSION INTER-ANNOTATOR AGREEMENT")
    print("-" * 80)

    for dim_idx, dim_name in enumerate(dimensions):
        print(f"\n{dim_name}:")
        print("-" * 40)

        # Extract ratings for this dimension
        a1_dim = a1_ratings[:, dim_idx]
        a2_dim = a2_ratings[:, dim_idx]
        a3_dim = a3_ratings[:, dim_idx]

        # Pairwise Cohen's Kappa
        kappa_12 = cohen_kappa_score(a1_dim, a2_dim)
        kappa_13 = cohen_kappa_score(a1_dim, a3_dim)
        kappa_23 = cohen_kappa_score(a2_dim, a3_dim)
        avg_kappa = np.mean([kappa_12, kappa_13, kappa_23])

        # Pairwise Agreement %
        agree_12 = np.mean(a1_dim == a2_dim) * 100
        agree_13 = np.mean(a1_dim == a3_dim) * 100
        agree_23 = np.mean(a2_dim == a3_dim) * 100
        avg_agreement = np.mean([agree_12, agree_13, agree_23])

        print(f"  Cohen's κ (pairwise):")
        print(f"    Annotator1 vs Annotator2: {kappa_12:.3f}")
        print(f"    Annotator1 vs YG:         {kappa_13:.3f}")
        print(f"    Annotator2 vs YG:         {kappa_23:.3f}")
        print(f"    Average:                  {avg_kappa:.3f}")

        print(f"\n  Agreement % (pairwise):")
        print(f"    Annotator1 vs Annotator2: {agree_12:.1f}%")
        print(f"    Annotator1 vs YG:         {agree_13:.1f}%")
        print(f"    Annotator2 vs YG:         {agree_23:.1f}%")
        print(f"    Average:                  {avg_agreement:.1f}%")

        # Fleiss' Kappa for all three annotators
        # Create rating matrix (items × annotators)
        rating_matrix = np.column_stack([a1_dim, a2_dim, a3_dim])
        fleiss_kappa = calculate_fleiss_kappa(rating_matrix)
        print(f"\n  Fleiss' κ (all annotators): {fleiss_kappa:.3f}")

        # Three-way exact agreement
        three_way = np.mean((a1_dim == a2_dim) & (a2_dim == a3_dim)) * 100
        print(f"  Three-way exact agreement:  {three_way:.1f}%")

    # Overall system-level metrics
    print(f"\n\n3. OVERALL SYSTEM-LEVEL AGREEMENT ({system.upper()})")
    print("-" * 80)

    # Flatten all ratings
    a1_all = a1_ratings.flatten()
    a2_all = a2_ratings.flatten()
    a3_all = a3_ratings.flatten()

    # Overall Cohen's Kappa
    kappa_12_all = cohen_kappa_score(a1_all, a2_all)
    kappa_13_all = cohen_kappa_score(a1_all, a3_all)
    kappa_23_all = cohen_kappa_score(a2_all, a3_all)
    avg_kappa_all = np.mean([kappa_12_all, kappa_13_all, kappa_23_all])

    print(f"\nCohen's κ (all dimensions combined):")
    print(f"  Annotator1 vs Annotator2: {kappa_12_all:.3f}")
    print(f"  Annotator1 vs YG:         {kappa_13_all:.3f}")
    print(f"  Annotator2 vs YG:         {kappa_23_all:.3f}")
    print(f"  Average:                  {avg_kappa_all:.3f}")

    # Overall Agreement %
    agree_12_all = np.mean(a1_all == a2_all) * 100
    agree_13_all = np.mean(a1_all == a3_all) * 100
    agree_23_all = np.mean(a2_all == a3_all) * 100
    avg_agreement_all = np.mean([agree_12_all, agree_13_all, agree_23_all])

    print(f"\nAgreement % (all dimensions combined):")
    print(f"  Annotator1 vs Annotator2: {agree_12_all:.1f}%")
    print(f"  Annotator1 vs YG:         {agree_13_all:.1f}%")
    print(f"  Annotator2 vs YG:         {agree_23_all:.1f}%")
    print(f"  Average:                  {avg_agreement_all:.1f}%")

    # Overall Fleiss' Kappa
    rating_matrix_all = np.column_stack([a1_all, a2_all, a3_all])
    fleiss_kappa_all = calculate_fleiss_kappa(rating_matrix_all)
    print(f"\nFleiss' κ (all dimensions, all annotators): {fleiss_kappa_all:.3f}")

    # Three-way exact agreement
    three_way_all = np.mean((a1_all == a2_all) & (a2_all == a3_all)) * 100
    print(f"Three-way exact agreement (all dimensions):  {three_way_all:.1f}%")

print("\n\n" + "=" * 80)
print("4. KAPPA INTERPRETATION GUIDE")
print("=" * 80)
print("""
According to Landis & Koch (1977):
  κ < 0.00:     Poor agreement
  0.00 - 0.20:  Slight agreement
  0.21 - 0.40:  Fair agreement
  0.41 - 0.60:  Moderate agreement
  0.61 - 0.80:  Substantial agreement
  0.81 - 1.00:  Almost perfect agreement
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)