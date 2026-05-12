"""
Usage Example: Error Analysis with Entity Analyzer
Demonstrates how to use the framework with linked entity data
"""

import sys
sys.path.append('/home/claude')

from error_analysis_framework import ContextClaimErrorAnalyzer
from entity_analyzer import EntityAnalyzer
import numpy as np

def example_with_entities():
    """
    Example workflow with entity analyzer
    """
    
    print("="*80)
    print("ERROR ANALYSIS WITH ENTITY ANALYZER - USAGE EXAMPLE")
    print("="*80)
    print()
    
    # Step 1: Load entity data
    print("Step 1: Loading entity data...")
    print()
    print("For your actual data, do:")
    print("  entity_analyzer = EntityAnalyzer('policlaim_linked_entities_test.json')")
    print()
    
    # For this example, we'll create mock data
    entity_analyzer = EntityAnalyzer()  # No file, will use heuristic
    
    # Or if you have the file:
    # entity_analyzer = EntityAnalyzer('/path/to/policlaim_linked_entities_test.json')
    
    # Step 2: Prepare your data
    print("Step 2: Preparing analysis data...")
    
    # Your actual data structure
    data = {
        'claim_ids': ['claim_001', 'claim_002', 'claim_003', 'claim_004', 'claim_005'],
        'claims': [
            "Senator John Smith voted against the healthcare bill.",
            "I believe the economy is getting worse.",
            "President Biden announced new climate initiatives.",
            "The stock market will crash soon.",
            "Lieutenant Governor Ainsworth supports education reform."
        ],
        'contexts': [
            "John Smith is a U.S. Senator who has served since 2019. Senate voting records are documented.",
            "Economic assessments vary by indicator and perspective.",
            "Joe Biden is the 46th President. Presidential announcements are official.",
            "Stock market predictions are speculative and based on analysis.",
            "Will Ainsworth is the Lieutenant Governor of Alabama since 2019."
        ],
        'true_labels': [1, 0, 1, 0, 1],  # 1=Verifiable, 0=Non-verifiable
        'baseline_preds': [0, 0, 1, 1, 1],  # Some errors
        'context_preds': [1, 0, 1, 0, 1]   # Better with context
    }
    
    print(f"  ✓ Loaded {len(data['claims'])} claims")
    print()
    
    # Step 3: Initialize error analyzer WITH entity analyzer
    print("Step 3: Initializing error analyzer with entity support...")
    
    analyzer = ContextClaimErrorAnalyzer(
        baseline_predictions=data['baseline_preds'],
        context_predictions=data['context_preds'],
        true_labels=data['true_labels'],
        claims=data['claims'],
        contexts=data['contexts'],
        claim_ids=data['claim_ids'],
        entity_analyzer=entity_analyzer  # ← KEY: Pass entity analyzer here
    )
    
    print("  ✓ Analyzer initialized with entity support")
    print()
    
    # Step 4: Run analysis
    print("Step 4: Running error analysis...")
    results = analyzer.generate_report('./example_with_entities')
    print()
    
    # Step 5: Show entity-enhanced results
    print("="*80)
    print("ENTITY-ENHANCED ANALYSIS RESULTS")
    print("="*80)
    print()
    
    # Load and display entity analysis
    import json
    with open('./example_with_entities/error_patterns.json', 'r') as f:
        patterns = json.load(f)
    
    # Show entity statistics for each category
    for category in ['baseline_to_context', 'context_to_baseline', 'both_correct']:
        if category in patterns and patterns[category]['statistics']:
            stats = patterns[category]['statistics']
            
            print(f"\n{category.replace('_', ' ').title()}:")
            print(f"  Number of claims: {stats.get('n_samples', 0)}")
            print(f"  Avg entities per claim: {stats['entity_count']['mean']:.2f}")
            
            if 'entity_quality' in stats:
                eq = stats['entity_quality']
                print(f"  Avg entity linking score: {eq['avg_linking_score']:.3f}")
                print(f"  Avg high-confidence entities: {eq['avg_high_confidence_entities']:.2f}")
                print(f"  Claims with entities: {eq['claims_with_entities']}/{stats['n_samples']}")
    
    print()
    print("="*80)
    print("COMPLETE!")
    print("="*80)
    print()
    print("Files generated in ./example_with_entities/")
    print("  - error_patterns.json now includes entity quality metrics")
    print("  - All visualizations and tables as before")
    print()


def real_usage_template():
    """
    Template for your actual usage with real data
    """
    
    template_code = '''
# ============================================================================
# YOUR ACTUAL USAGE CODE
# ============================================================================

from error_analysis_framework import ContextClaimErrorAnalyzer
from entity_analyzer import EntityAnalyzer
import pandas as pd
import json

# Step 1: Load entity data
print("Loading entity data...")
entity_analyzer = EntityAnalyzer('policlaim_linked_entities_test.json')

# Step 2: Load your test data
print("Loading test data...")
test_data = pd.read_csv('policlaim_test.csv')

# Load predictions
baseline_preds = pd.read_csv('roberta_baseline_predictions.csv')
context_preds = pd.read_csv('roberta_ccm_predictions.csv')

# Load generated contexts
with open('ccm_contexts_policlaim.json', 'r') as f:
    contexts_dict = json.load(f)

# Step 3: Prepare data
data = {
    'claim_ids': test_data['id'].tolist(),
    'claims': test_data['claim'].tolist(),
    'contexts': [contexts_dict[id] for id in test_data['id']],
    'true_labels': test_data['label'].tolist(),
    'baseline_preds': baseline_preds['prediction'].tolist(),
    'context_preds': context_preds['prediction'].tolist()
}

# Step 4: Run analysis WITH entity support
print("Running error analysis with entity support...")
analyzer = ContextClaimErrorAnalyzer(
    baseline_predictions=data['baseline_preds'],
    context_predictions=data['context_preds'],
    true_labels=data['true_labels'],
    claims=data['claims'],
    contexts=data['contexts'],
    claim_ids=data['claim_ids'],
    entity_analyzer=entity_analyzer  # ← Entity support enabled
)

# Generate comprehensive report
results = analyzer.generate_report('./policlaim_analysis')

print("✓ Analysis complete! Results in ./policlaim_analysis/")

# Step 5: Examine entity-related patterns
import json
with open('./policlaim_analysis/error_patterns.json', 'r') as f:
    patterns = json.load(f)

# Analyze: Do claims with more entities benefit more from context?
helps_stats = patterns['baseline_to_context']['statistics']
print(f"\\nClaims where ContextClaim helps:")
print(f"  Avg entities: {helps_stats['entity_count']['mean']:.2f}")
if 'entity_quality' in helps_stats:
    print(f"  Avg linking score: {helps_stats['entity_quality']['avg_linking_score']:.3f}")

hurts_stats = patterns['context_to_baseline']['statistics']
print(f"\\nClaims where ContextClaim hurts:")
print(f"  Avg entities: {hurts_stats['entity_count']['mean']:.2f}")
if 'entity_quality' in hurts_stats:
    print(f"  Avg linking score: {hurts_stats['entity_quality']['avg_linking_score']:.3f}")
    
# ============================================================================
# END OF TEMPLATE
# ============================================================================
'''
    
    print(template_code)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPTION 1: RUN EXAMPLE WITH MOCK DATA")
    print("="*80)
    example_with_entities()
    
    print("\n" + "="*80)
    print("OPTION 2: TEMPLATE FOR YOUR ACTUAL DATA")
    print("="*80)
    print()
    real_usage_template()

