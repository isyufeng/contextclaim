"""
Qualitative Analysis Helper
Generate formatted case studies for paper
"""

import json
from typing import Dict, List


class QualitativeAnalysisHelper:
    """
    Helper class to format case studies for paper inclusion
    """

    @staticmethod
    def format_case_for_paper(case: Dict, case_number: int, category: str) -> str:
        """
        Format a single case study for LaTeX inclusion

        Args:
            case: Dictionary containing case information
            case_number: Sequential number for the example
            category: Category name (e.g., "ContextClaim Helps")
        """
        latex = f"\\textbf{{Example {case_number}: {category}}}\n\n"
        latex += "\\begin{itemize}[leftmargin=*, itemsep=2pt]\n"
        latex += f"\\item \\textbf{{Claim:}} ``{case['claim']}''\n"
        latex += f"\\item \\textbf{{Ground Truth:}} {case['true_label_text']}\n"
        latex += f"\\item \\textbf{{Baseline Prediction:}} {case['baseline_pred_text']} "
        latex += f"({'✓' if case['baseline_correct'] else '✗'})\n"
        latex += f"\\item \\textbf{{ContextClaim Prediction:}} {case['context_pred_text']} "
        latex += f"({'✓' if case['context_correct'] else '✗'})\n"

        # Truncate context if too long
        context_preview = case['context'][:200] + "..." if len(case['context']) > 200 else case['context']
        latex += f"\\item \\textbf{{Context Summary:}} ``{context_preview}''\n"
        latex += "\\end{itemize}\n\n"

        return latex

    @staticmethod
    def generate_paper_examples(error_patterns: Dict,
                                categories_to_include: List[str],
                                n_per_category: int = 2) -> str:
        """
        Generate formatted examples for paper

        Args:
            error_patterns: Output from error_patterns.json
            categories_to_include: List of category names to include
            n_per_category: Number of examples per category
        """
        latex_output = ""
        example_counter = 1

        category_display_names = {
            'baseline_to_context': 'ContextClaim Helps (Baseline→ContextClaim)',
            'context_to_baseline': 'ContextClaim Hurts (ContextClaim→Baseline)',
            'both_wrong': 'Both Models Fail'
        }

        for category in categories_to_include:
            if category not in error_patterns:
                continue

            display_name = category_display_names.get(category, category)
            cases = error_patterns[category]['representative_cases'][:n_per_category]

            latex_output += f"\\subsection{{{display_name}}}\n\n"

            for case in cases:
                latex_output += QualitativeAnalysisHelper.format_case_for_paper(
                    case, example_counter, display_name
                )
                example_counter += 1

            latex_output += "\n"

        return latex_output

    @staticmethod
    def create_analysis_narrative(case: Dict) -> str:
        """
        Create narrative analysis text for a case

        Args:
            case: Dictionary containing case information
        """
        narrative = f"\\textbf{{Analysis:}} "

        if case['baseline_correct'] and not case['context_correct']:
            # ContextClaim hurts
            narrative += "The baseline model correctly classified this claim, but "
            narrative += "ContextClaim made an error. This suggests that the provided context "
            narrative += "may have introduced misleading information or noise. "

            if case['error_type'] == 'False Positive (Non-verifiable → Verifiable)':
                narrative += "Specifically, the context may have emphasized factual aspects "
                narrative += "that caused the model to incorrectly judge a subjective or "
                narrative += "unverifiable claim as verifiable."
            else:
                narrative += "The context may have obscured verifiable aspects of the claim "
                narrative += "or introduced ambiguity about documentability."

        elif not case['baseline_correct'] and case['context_correct']:
            # ContextClaim helps
            narrative += "The baseline model failed to correctly classify this claim, but "
            narrative += "ContextClaim succeeded. The context summary provided crucial information "

            if case['error_type'] == 'Correct':
                if case['true_label'] == 1:  # Verifiable
                    narrative += "confirming the existence of entities and documentability of events "
                    narrative += "mentioned in the claim, enabling correct verifiability assessment."
                else:  # Non-verifiable
                    narrative += "clarifying that the claim involves subjective opinions or "
                    narrative += "hypothetical scenarios that cannot be objectively fact-checked."

        elif not case['baseline_correct'] and not case['context_correct']:
            # Both wrong
            narrative += "Both models failed on this challenging case. "

            if case['claim_length'] > 30:
                narrative += "The claim's length and complexity may have contributed to the difficulty. "

            if case['error_type'] == 'False Positive (Non-verifiable → Verifiable)':
                narrative += "Both models struggled to recognize the subjective or unverifiable nature "
                narrative += "of this claim, possibly due to the presence of factual elements that "
                narrative += "obscured the overall non-verifiability."
            else:
                narrative += "Both models failed to recognize verifiable aspects, suggesting that "
                narrative += "either the claim requires specialized domain knowledge or the context "
                narrative += "lacked sufficient verifiability signals."

        return narrative + "\n\n"

    @staticmethod
    def generate_insights_summary(error_patterns: Dict) -> str:
        """
        Generate key insights summary from error patterns
        """
        insights = "\\section{Key Insights from Error Analysis}\n\n"

        # Improvement cases
        helps_cases = error_patterns.get('baseline_to_context', {}).get('statistics', {})
        if helps_cases and 'n_samples' in helps_cases:
            n_helps = helps_cases['n_samples']
            insights += f"\\textbf{{When ContextClaim Helps ({n_helps} cases):}}\n"
            insights += "\\begin{itemize}\n"

            if 'claim_length' in helps_cases:
                avg_len = helps_cases['claim_length']['mean']
                insights += f"\\item Average claim length: {avg_len:.1f} words\n"

            if 'entity_count' in helps_cases:
                avg_ent = helps_cases['entity_count']['mean']
                insights += f"\\item Average entity count: {avg_ent:.1f}\n"

            insights += "\\item Pattern: Context provides entity confirmation and event documentability\n"
            insights += "\\end{itemize}\n\n"

        # Degradation cases
        hurts_cases = error_patterns.get('context_to_baseline', {}).get('statistics', {})
        if hurts_cases and 'n_samples' in hurts_cases:
            n_hurts = hurts_cases['n_samples']
            insights += f"\\textbf{{When ContextClaim Hurts ({n_hurts} cases):}}\n"
            insights += "\\begin{itemize}\n"

            if 'claim_length' in hurts_cases:
                avg_len = hurts_cases['claim_length']['mean']
                insights += f"\\item Average claim length: {avg_len:.1f} words\n"

            insights += "\\item Pattern: Context introduces factual information that misleads model\n"
            insights += "\\item Issue: Difficulty distinguishing objective facts from subjective opinions\n"
            insights += "\\end{itemize}\n\n"

        # Both fail cases
        both_fail = error_patterns.get('both_wrong', {}).get('statistics', {})
        if both_fail and 'n_samples' in both_fail:
            n_fail = both_fail['n_samples']
            insights += f"\\textbf{{When Both Models Fail ({n_fail} cases):}}\n"
            insights += "\\begin{itemize}\n"
            insights += "\\item Complex claims requiring specialized knowledge\n"
            insights += "\\item Context lacks explicit verifiability signals\n"
            insights += "\\item Ambiguous boundary between verifiable and non-verifiable\n"
            insights += "\\end{itemize}\n\n"

        return insights


def generate_qualitative_report(error_patterns_file: str, output_file: str):
    """
    Generate complete qualitative analysis report

    Args:
        error_patterns_file: Path to error_patterns.json
        output_file: Path to save LaTeX output
    """
    # Load error patterns
    with open(error_patterns_file, 'r') as f:
        error_patterns = json.load(f)

    helper = QualitativeAnalysisHelper()

    # Generate document
    latex_content = "% Qualitative Error Analysis\n\n"

    # Add examples
    latex_content += "\\section{Representative Examples}\n\n"
    latex_content += helper.generate_paper_examples(
        error_patterns,
        categories_to_include=['baseline_to_context', 'context_to_baseline', 'both_wrong'],
        n_per_category=2
    )

    # Add detailed analysis for select cases
    latex_content += "\\section{Detailed Case Analysis}\n\n"

    # Analyze one case from each category in detail
    for category in ['baseline_to_context', 'context_to_baseline', 'both_wrong']:
        if category in error_patterns and error_patterns[category]['representative_cases']:
            case = error_patterns[category]['representative_cases'][0]
            latex_content += helper.format_case_for_paper(case, 1, category)
            latex_content += helper.create_analysis_narrative(case)

    # Add insights
    latex_content += helper.generate_insights_summary(error_patterns)

    # Save
    with open(output_file, 'w') as f:
        f.write(latex_content)

    print(f"✓ Qualitative analysis report saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    generate_qualitative_report(
        error_patterns_file='./error_analysis_results/error_patterns.json',
        output_file='./error_analysis_results/qualitative_analysis.tex'
    )
