"""
Error Analysis Framework for ContextClaim
Focus: RoBERTa + CC-M (Fine-tuning) on PoliClaim and CT22
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from entity_analyzer import EntityAnalyzer, heuristic_entity_detection


class ContextClaimErrorAnalyzer:
    """
    Comprehensive error analysis for ContextClaim system
    """

    def __init__(self,
                 baseline_predictions: List[int],
                 context_predictions: List[int],
                 true_labels: List[int],
                 claims: List[str],
                 contexts: List[str],
                 claim_ids: List[str] = None,
                 entity_analyzer: Optional[EntityAnalyzer] = None):
        """
        Initialize the analyzer

        Args:
            baseline_predictions: Predictions from baseline model (0/1)
            context_predictions: Predictions from ContextClaim model (0/1)
            true_labels: Ground truth labels (0/1)
            claims: Original claim texts
            contexts: Generated context summaries
            claim_ids: Optional claim IDs for tracking
            entity_analyzer: Optional EntityAnalyzer instance for entity analysis
        """
        self.baseline_preds = np.array(baseline_predictions)
        self.context_preds = np.array(context_predictions)
        self.true_labels = np.array(true_labels)
        self.claims = claims
        self.contexts = contexts
        self.claim_ids = claim_ids if claim_ids else list(range(len(claims)))
        self.entity_analyzer = entity_analyzer

        # Validate inputs
        assert len(self.baseline_preds) == len(self.context_preds) == len(self.true_labels), \
            "All inputs must have same length"

        # Compute correctness
        self.baseline_correct = (self.baseline_preds == self.true_labels)
        self.context_correct = (self.context_preds == self.true_labels)

    def categorize_predictions(self) -> Dict[str, np.ndarray]:
        """
        Categorize predictions into 4 groups:
        1. both_correct: Both models correct
        2. baseline_to_context: Baseline wrong, Context correct (ContextClaim helps)
        3. context_to_baseline: Baseline correct, Context wrong (ContextClaim hurts)
        4. both_wrong: Both models wrong
        """
        categories = {
            'both_correct': self.baseline_correct & self.context_correct,
            'baseline_to_context': (~self.baseline_correct) & self.context_correct,
            'context_to_baseline': self.baseline_correct & (~self.context_correct),
            'both_wrong': (~self.baseline_correct) & (~self.context_correct)
        }
        return categories

    def get_error_type(self, idx: int) -> str:
        """Get specific error type for an instance"""
        if self.context_correct[idx]:
            return "Correct"
        elif self.context_preds[idx] == 1 and self.true_labels[idx] == 0:
            return "False Positive (Non-verifiable → Verifiable)"
        else:
            return "False Negative (Verifiable → Non-verifiable)"

    def summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics table"""
        categories = self.categorize_predictions()

        stats = []
        for cat_name, mask in categories.items():
            count = mask.sum()
            percentage = (count / len(self.claims)) * 100

            # Get examples
            indices = np.where(mask)[0]

            stats.append({
                'Category': cat_name,
                'Count': count,
                'Percentage': f"{percentage:.1f}%",
                'Sample_Indices': indices[:5].tolist() if len(indices) > 0 else []
            })

        return pd.DataFrame(stats)

    def analyze_claim_characteristics(self, category_mask: np.ndarray) -> Dict:
        """
        Analyze characteristics of claims in a specific category
        """
        indices = np.where(category_mask)[0]

        if len(indices) == 0:
            return {}

        selected_claims = [self.claims[i] for i in indices]
        selected_contexts = [self.contexts[i] for i in indices]
        selected_claim_ids = [self.claim_ids[i] for i in indices]

        # Claim length analysis
        claim_lengths = [len(claim.split()) for claim in selected_claims]

        # Context length analysis
        context_lengths = [len(context.split()) for context in selected_contexts]

        # Entity detection - use EntityAnalyzer if available, otherwise use heuristic
        entity_counts = []
        entity_scores = []
        high_confidence_entity_counts = []

        if self.entity_analyzer and self.entity_analyzer.has_entities_data():
            # Use loaded entity data
            for claim_id in selected_claim_ids:
                entity_count = self.entity_analyzer.get_entity_count(claim_id)
                entity_counts.append(entity_count)

                # Get entity quality metrics
                quality = self.entity_analyzer.analyze_entity_quality(claim_id)
                entity_scores.append(quality['avg_score'])
                high_confidence_entity_counts.append(quality['high_confidence_count'])

            has_entity_data = True
        else:
            # Fallback to heuristic entity detection
            for claim in selected_claims:
                entity_count = heuristic_entity_detection(claim)
                entity_counts.append(entity_count)

            has_entity_data = False

        # Question detection
        is_question = [claim.strip().endswith('?') for claim in selected_claims]

        # Length categories
        length_dist = {
            'Short (<10)': sum(1 for l in claim_lengths if l < 10),
            'Medium (10-20)': sum(1 for l in claim_lengths if 10 <= l <= 20),
            'Long (>20)': sum(1 for l in claim_lengths if l > 20)
        }

        result = {
            'n_samples': len(indices),
            'claim_length': {
                'mean': np.mean(claim_lengths),
                'std': np.std(claim_lengths),
                'min': np.min(claim_lengths),
                'max': np.max(claim_lengths),
                'distribution': length_dist
            },
            'context_length': {
                'mean': np.mean(context_lengths),
                'std': np.std(context_lengths),
                'min': np.min(context_lengths),
                'max': np.max(context_lengths)
            },
            'entity_count': {
                'mean': np.mean(entity_counts),
                'std': np.std(entity_counts),
                'distribution': dict(Counter(entity_counts))
            },
            'question_percentage': sum(is_question) / len(is_question) * 100 if is_question else 0
        }

        # Add entity quality metrics if available
        if has_entity_data:
            result['entity_quality'] = {
                'avg_linking_score': np.mean(entity_scores) if entity_scores else 0.0,
                'avg_high_confidence_entities': np.mean(
                    high_confidence_entity_counts) if high_confidence_entity_counts else 0.0,
                'claims_with_entities': sum(1 for c in entity_counts if c > 0),
                'claims_with_high_confidence_entities': sum(1 for c in high_confidence_entity_counts if c > 0)
            }

        return result

    def extract_representative_cases(self, category_mask: np.ndarray,
                                     n_samples: int = 10,
                                     strategy: str = 'diverse') -> List[Dict]:
        """
        Extract representative cases from a category

        Args:
            category_mask: Boolean mask for the category
            n_samples: Number of samples to extract
            strategy: 'diverse' or 'random'
        """
        indices = np.where(category_mask)[0]

        if len(indices) == 0:
            return []

        # Select samples
        if strategy == 'diverse':
            # Try to get diverse claim lengths
            claim_lengths = np.array([len(self.claims[i].split()) for i in indices])

            # Sort by length
            sorted_indices = indices[np.argsort(claim_lengths)]

            # Sample evenly across length spectrum
            step = max(1, len(sorted_indices) // n_samples)
            selected_indices = sorted_indices[::step][:n_samples]
        else:
            # Random sampling
            selected_indices = np.random.choice(indices,
                                                min(n_samples, len(indices)),
                                                replace=False)

        cases = []
        for idx in selected_indices:
            case = {
                'index': int(idx),
                'claim_id': self.claim_ids[idx],
                'claim': self.claims[idx],
                'context': self.contexts[idx],
                'true_label': int(self.true_labels[idx]),
                'true_label_text': 'Verifiable' if self.true_labels[idx] == 1 else 'Non-verifiable',
                'baseline_pred': int(self.baseline_preds[idx]),
                'baseline_pred_text': 'Verifiable' if self.baseline_preds[idx] == 1 else 'Non-verifiable',
                'context_pred': int(self.context_preds[idx]),
                'context_pred_text': 'Verifiable' if self.context_preds[idx] == 1 else 'Non-verifiable',
                'baseline_correct': bool(self.baseline_correct[idx]),
                'context_correct': bool(self.context_correct[idx]),
                'error_type': self.get_error_type(idx),
                'claim_length': len(self.claims[idx].split()),
                'context_length': len(self.contexts[idx].split())
            }
            cases.append(case)

        return cases

    def analyze_error_patterns(self) -> Dict:
        """
        Comprehensive error pattern analysis
        """
        categories = self.categorize_predictions()

        analysis = {}

        for cat_name, mask in categories.items():
            analysis[cat_name] = {
                'statistics': self.analyze_claim_characteristics(mask),
                'representative_cases': self.extract_representative_cases(mask,
                                                                          n_samples=10,
                                                                          strategy='diverse')
            }

        return analysis

    def compare_false_positives_negatives(self) -> Dict:
        """
        Analyze False Positives vs False Negatives in context model
        """
        # False Positives: predict 1, truth 0
        fp_mask = (self.context_preds == 1) & (self.true_labels == 0)

        # False Negatives: predict 0, truth 1
        fn_mask = (self.context_preds == 0) & (self.true_labels == 1)

        return {
            'false_positives': {
                'count': fp_mask.sum(),
                'percentage': (fp_mask.sum() / len(self.claims)) * 100,
                'characteristics': self.analyze_claim_characteristics(fp_mask),
                'examples': self.extract_representative_cases(fp_mask, n_samples=5)
            },
            'false_negatives': {
                'count': fn_mask.sum(),
                'percentage': (fn_mask.sum() / len(self.claims)) * 100,
                'characteristics': self.analyze_claim_characteristics(fn_mask),
                'examples': self.extract_representative_cases(fn_mask, n_samples=5)
            }
        }

    def generate_report(self, output_dir: str = './error_analysis_results'):
        """
        Generate comprehensive error analysis report
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 1. Summary statistics
        summary = self.summary_statistics()
        summary.to_csv(f'{output_dir}/summary_statistics.csv', index=False)
        print("Summary Statistics:")
        print(summary.to_string(index=False))
        print("\n")

        # 2. Detailed error pattern analysis
        error_patterns = self.analyze_error_patterns()

        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        error_patterns = convert_to_native(error_patterns)
        with open(f'{output_dir}/error_patterns.json', 'w') as f:
            json.dump(error_patterns, f, indent=2)

        # 3. False Positive vs False Negative analysis
        fp_fn_analysis = self.compare_false_positives_negatives()
        fp_fn_analysis = convert_to_native(fp_fn_analysis)
        with open(f'{output_dir}/fp_fn_analysis.json', 'w') as f:
            json.dump(fp_fn_analysis, f, indent=2)

        # 4. Generate visualizations
        self.create_visualizations(output_dir)

        # 5. Generate LaTeX tables
        self.generate_latex_tables(output_dir)

        print(f"✓ Report generated in {output_dir}/")

        return {
            'summary': summary,
            'error_patterns': error_patterns,
            'fp_fn_analysis': fp_fn_analysis
        }

    def create_visualizations(self, output_dir: str):
        """Create visualization plots"""
        categories = self.categorize_predictions()

        # Figure 1: Category distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart
        cat_counts = {name: mask.sum() for name, mask in categories.items()}
        cat_labels = ['Both Correct', 'CC Helps\n(B→C)', 'CC Hurts\n(C→B)', 'Both Wrong']
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']

        axes[0].pie(cat_counts.values(), labels=cat_labels, autopct='%1.1f%%',
                    colors=colors, startangle=90)
        axes[0].set_title('Prediction Category Distribution', fontsize=14, fontweight='bold')

        # Bar chart with counts
        axes[1].bar(range(len(cat_counts)), list(cat_counts.values()), color=colors)
        axes[1].set_xticks(range(len(cat_counts)))
        axes[1].set_xticklabels(cat_labels, fontsize=10)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Prediction Counts by Category', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 2: Claim length distribution by category
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        data_for_plot = []
        labels_for_plot = []

        for name, mask in categories.items():
            if mask.sum() > 0:
                indices = np.where(mask)[0]
                lengths = [len(self.claims[i].split()) for i in indices]
                data_for_plot.append(lengths)
                labels_for_plot.append(name.replace('_', ' ').title())

        bp = ax.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Claim Length (words)', fontsize=12)
        ax.set_title('Claim Length Distribution by Category', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/claim_length_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 3: Error type analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Context model errors
        context_errors = ~self.context_correct
        if context_errors.sum() > 0:
            fp = ((self.context_preds == 1) & (self.true_labels == 0)).sum()
            fn = ((self.context_preds == 0) & (self.true_labels == 1)).sum()

            axes[0].bar(['False Positive\n(Non-ver → Ver)', 'False Negative\n(Ver → Non-ver)'],
                        [fp, fn], color=['#e74c3c', '#f39c12'])
            axes[0].set_ylabel('Count', fontsize=12)
            axes[0].set_title('ContextClaim Error Types', fontsize=14, fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)

        # Improvement analysis
        improved = (~self.baseline_correct & self.context_correct).sum()
        degraded = (self.baseline_correct & ~self.context_correct).sum()
        unchanged_correct = (self.baseline_correct & self.context_correct).sum()
        unchanged_wrong = (~self.baseline_correct & ~self.context_correct).sum()

        axes[1].bar(['Improved', 'Degraded', 'Unchanged\n(Correct)', 'Unchanged\n(Wrong)'],
                    [improved, degraded, unchanged_correct, unchanged_wrong],
                    color=['#3498db', '#e74c3c', '#2ecc71', '#95a5a6'])
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('ContextClaim Impact on Performance', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualizations saved to {output_dir}/")

    def generate_latex_tables(self, output_dir: str):
        """Generate LaTeX tables for paper"""

        # Table 1: Summary statistics
        categories = self.categorize_predictions()

        latex_summary = "\\begin{table}[t]\n\\centering\n"
        latex_summary += "\\caption{Error Analysis Summary: RoBERTa + CC-M}\n"
        latex_summary += "\\label{tab:error_summary}\n"
        latex_summary += "\\begin{tabular}{lcc}\n"
        latex_summary += "\\toprule\n"
        latex_summary += "\\textbf{Category} & \\textbf{Count} & \\textbf{Percentage} \\\\\n"
        latex_summary += "\\midrule\n"

        cat_names = {
            'both_correct': 'Both Correct',
            'baseline_to_context': 'ContextClaim Helps (B→CC)',
            'context_to_baseline': 'ContextClaim Hurts (CC→B)',
            'both_wrong': 'Both Wrong'
        }

        for cat_key, mask in categories.items():
            count = mask.sum()
            pct = (count / len(self.claims)) * 100
            latex_summary += f"{cat_names[cat_key]} & {count} & {pct:.1f}\\% \\\\\n"

        latex_summary += "\\bottomrule\n"
        latex_summary += "\\end{tabular}\n"
        latex_summary += "\\end{table}\n"

        with open(f'{output_dir}/latex_summary_table.tex', 'w') as f:
            f.write(latex_summary)

        # Table 2: Claim characteristics by category
        latex_chars = "\\begin{table}[t]\n\\centering\n"
        latex_chars += "\\caption{Claim Characteristics by Error Category}\n"
        latex_chars += "\\label{tab:claim_characteristics}\n"
        latex_chars += "\\small\n"
        latex_chars += "\\begin{tabular}{lcccc}\n"
        latex_chars += "\\toprule\n"
        latex_chars += "\\textbf{Category} & \\textbf{Avg Length} & \\textbf{Avg Entities} & \\textbf{\\% Questions} \\\\\n"
        latex_chars += "\\midrule\n"

        for cat_key, mask in categories.items():
            if mask.sum() > 0:
                chars = self.analyze_claim_characteristics(mask)
                avg_len = chars['claim_length']['mean']
                avg_ent = chars['entity_count']['mean']
                q_pct = chars['question_percentage']
                latex_chars += f"{cat_names[cat_key]} & {avg_len:.1f} & {avg_ent:.1f} & {q_pct:.1f}\\% \\\\\n"

        latex_chars += "\\bottomrule\n"
        latex_chars += "\\end{tabular}\n"
        latex_chars += "\\end{table}\n"

        with open(f'{output_dir}/latex_characteristics_table.tex', 'w') as f:
            f.write(latex_chars)

        print(f"✓ LaTeX tables saved to {output_dir}/")


def load_example_data():
    """
    Load example data for demonstration
    Replace this with your actual data loading
    """
    # This is just example structure - replace with your actual data
    np.random.seed(42)
    n_samples = 500

    data = {
        'claim_ids': [f"claim_{i}" for i in range(n_samples)],
        'claims': [f"Example claim {i} about topic X" for i in range(n_samples)],
        'contexts': [f"Context summary for claim {i} providing background" for i in range(n_samples)],
        'true_labels': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'baseline_preds': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),
        'context_preds': np.random.choice([0, 1], n_samples, p=[0.42, 0.58])
    }

    return data


# Example usage
if __name__ == "__main__":
    # Load your data
    print("Loading data...")
    data = load_example_data()  # Replace with your actual data loading

    # Initialize analyzer
    print("Initializing analyzer...")
    analyzer = ContextClaimErrorAnalyzer(
        baseline_predictions=data['baseline_preds'],
        context_predictions=data['context_preds'],
        true_labels=data['true_labels'],
        claims=data['claims'],
        contexts=data['contexts'],
        claim_ids=data['claim_ids']
    )

    # Generate comprehensive report
    print("\nGenerating error analysis report...")
    results = analyzer.generate_report(output_dir='./error_analysis_results')

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nFiles generated:")
    print("  - summary_statistics.csv")
    print("  - error_patterns.json")
    print("  - fp_fn_analysis.json")
    print("  - category_distribution.png")
    print("  - claim_length_by_category.png")
    print("  - error_type_analysis.png")
    print("  - latex_summary_table.tex")
    print("  - latex_characteristics_table.tex")

