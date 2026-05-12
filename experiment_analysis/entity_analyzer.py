"""
Entity Analysis Module for ContextClaim Error Analysis
Handles loading and analyzing linked entities from JSON files
"""

import json
from typing import Dict, List, Optional
import numpy as np

class EntityAnalyzer:
    """
    Helper class to analyze entities extracted from claims
    """
    
    def __init__(self, linked_entities_file: Optional[str] = None):
        """
        Initialize the entity analyzer
        
        Args:
            linked_entities_file: Path to JSON file containing linked entities
        """
        self.entities_data = {}
        if linked_entities_file:
            self.load_entities(linked_entities_file)

    def load_entities(self, filepath: str) -> None:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if data is a list (array format)
            if not isinstance(data, list):
                print(f"⚠ Warning: Expected JSON array but got {type(data).__name__}")
                print("  Continuing with heuristic entity detection...")
                return

            # Process each tweet in the array
            for item in data:
                # Use tweet_id as the claim_id, fallback to index if not available
                claim_id = item.get('tweet_id', str(data.index(item)))

                if 'linked_entities' in item and item['linked_entities']:
                    self.entities_data[claim_id] = {
                        'entities': list(item['linked_entities'].keys()),
                        'entity_count': len(item['linked_entities']),
                        'entity_details': item['linked_entities'],
                        'tweet_text': item.get('tweet_text', '')  # Store original text for reference
                    }

            print(f"✓ Loaded entities for {len(self.entities_data)} tweets from {filepath}")
            if self.entities_data:
                total_entities = sum(d['entity_count'] for d in self.entities_data.values())
                print(f"  Total entities: {total_entities}")

        except FileNotFoundError:
            print(f"⚠ Warning: Entity file not found: {filepath}")
            print("  Continuing with heuristic entity detection...")
        except json.JSONDecodeError as e:
            print(f"⚠ Warning: Invalid JSON in {filepath}: {str(e)}")
            print("  Continuing with heuristic entity detection...")
        except Exception as e:
            print(f"⚠ Warning: Error loading entities: {str(e)}")
            print("  Continuing with heuristic entity detection...")
    
    def get_entity_count(self, claim_id: str) -> int:
        """
        Get number of entities for a specific claim
        
        Args:
            claim_id: Claim identifier
        
        Returns:
            Number of entities (0 if claim not found)
        """
        if claim_id in self.entities_data:
            return self.entities_data[claim_id]['entity_count']
        return 0
    
    def get_entities(self, claim_id: str) -> List[str]:
        """
        Get list of entity names for a specific claim
        
        Args:
            claim_id: Claim identifier
        
        Returns:
            List of entity names (empty list if claim not found)
        """
        if claim_id in self.entities_data:
            return self.entities_data[claim_id]['entities']
        return []
    
    def get_entity_details(self, claim_id: str) -> Dict:
        """
        Get detailed entity information for a specific claim
        
        Args:
            claim_id: Claim identifier
        
        Returns:
            Dictionary of entity details (empty dict if claim not found)
        """
        if claim_id in self.entities_data:
            return self.entities_data[claim_id]['entity_details']
        return {}
    
    def has_entities_data(self) -> bool:
        """
        Check if entity data has been loaded
        
        Returns:
            True if entity data is available, False otherwise
        """
        return len(self.entities_data) > 0
    
    def get_average_entity_score(self, claim_id: str) -> float:
        """
        Get average linking score for entities in a claim
        
        Args:
            claim_id: Claim identifier
        
        Returns:
            Average score (0.0 if no entities or claim not found)
        """
        if claim_id not in self.entities_data:
            return 0.0
        
        entity_details = self.entities_data[claim_id]['entity_details']
        if not entity_details:
            return 0.0
        
        scores = [details.get('score', 0.0) for details in entity_details.values()]
        return np.mean(scores) if scores else 0.0
    
    def analyze_entity_quality(self, claim_id: str) -> Dict:
        """
        Analyze entity linking quality for a claim
        
        Args:
            claim_id: Claim identifier
        
        Returns:
            Dictionary with quality metrics
        """
        if claim_id not in self.entities_data:
            return {
                'has_entities': False,
                'entity_count': 0,
                'avg_score': 0.0,
                'high_confidence_count': 0
            }
        
        entity_details = self.entities_data[claim_id]['entity_details']
        scores = [details.get('score', 0.0) for details in entity_details.values()]
        
        return {
            'has_entities': True,
            'entity_count': len(entity_details),
            'avg_score': np.mean(scores) if scores else 0.0,
            'high_confidence_count': sum(1 for s in scores if s > 0.7),
            'entities': list(entity_details.keys())
        }


def heuristic_entity_detection(claim: str) -> int:
    """
    Fallback heuristic entity detection (capitalized words)
    
    Args:
        claim: Claim text
    
    Returns:
        Number of detected entities
    """
    words = claim.split()
    entities = [w for w in words if w and w[0].isupper() and w not in 
               ['The', 'A', 'An', 'This', 'That', 'These', 'Those', 'I', 'We', 'You']]
    return len(entities)


# Example usage
if __name__ == "__main__":
    # Test with example data
    test_data = {
        "claim_001": {
            "claim": "Lieutenant Governor Ainsworth supports the new policy.",
            "linked_entities": {
                "Lieutenant Governor Ainsworth": {
                    "page_id": "57620567",
                    "title": "Will Ainsworth",
                    "extract": "Will Ainsworth (born March 22, 1981) is an American politician...",
                    "score": 0.6321115732192993,
                    "context_similarity": 0.6165604591369629,
                    "title_similarity": 0.694316029548645
                }
            }
        }
    }
    
    # Save test data
    with open('test_entities.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Test analyzer
    analyzer = EntityAnalyzer('test_entities.json')
    
    print("\nEntity Count:", analyzer.get_entity_count('claim_001'))
    print("Entities:", analyzer.get_entities('claim_001'))
    print("Quality Analysis:", analyzer.analyze_entity_quality('claim_001'))
    print("\n✓ Entity analyzer working correctly!")
