import json
import logging
import os
import re

import pandas as pd
import numpy as np
from openai import OpenAI
from typing import Dict, List

import time
import random
from pathlib import Path

from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

timestamp = time.strftime("%Y%m%d_%H%M%S")
file_handler = logging.FileHandler(
    f"data/logs/gpt_experiment_{timestamp}.log",
    mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


class GPTExperimentConfig:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.temperature = 0.1
        self.max_tokens = 10
        self.top_p = 0.9

        self.system_message = """
                    Determine if the input text contains verifiable claims.
                    - If it clearly contains verifiable factual claims, respond "Yes".
                    - If it clearly contains only opinions or unverifiable statements, respond "No".
                    Note: When in doubt, choose "Yes". In the end, respond only with 'Yes' for verifiable claims or 'No' for non-verifiable claims.
                    """

        self.context_system_message = """
                    Determine if the input text contains verifiable claims.
                
                    Primary analysis:
                    - Analyze the input text first. If it clearly contains verifiable factual claims, respond "Yes".
                    - If it clearly contains only opinions or unverifiable statements, respond "No".
                
                    Secondary analysis (only if primary analysis is unclear):
                    - Reference the additional information to help clarify the nature of the claims in the input text.
                
                    Note: When in doubt, choose "Yes". In the end, respond only with 'Yes' for verifiable claims or 'No' for non-verifiable claims.
                    """

    def zero_shot_prompt(self, tweet_text: str) -> str:
        return f"""### Instruction:\n{self.system_message}\n\n### Input text:\n{tweet_text}\n\n### Response:"""

    def zero_shot_context_prompt(self, tweet_text: str, evidence: str) -> str:
        return f"""### Instruction:\n{self.context_system_message}\n\n### Input text:\n{tweet_text}\n\n### Additional information:\n{evidence}\n\n### Response:"""

    def few_shot_prompt(self, tweet_text: str, examples: List[Dict]) -> str:
        prompt = f"""### Instruction:\n{self.system_message}\n\n### Examples:\n"""
        for ex in examples:
            label = 'Yes' if ex['class_label'] == 1 else 'No'
            prompt += f"""
                      ### Input text:\n{ex['tweet_text']}\n\n### Response:\n{label}
                      """

        prompt += f"""
                    ### Input text:\n{tweet_text}\n\n### Response:"""
        return prompt

    def few_shot_context_prompt(self, tweet_text: str, evidence: str, examples: List[Dict]) -> str:
        prompt = f"""### Instruction:\n{self.context_system_message}\n\n### Examples:\n"""

        for ex in examples:
            label = 'Yes' if ex['class_label'] == 1 else 'No'
            prompt += f"""
                      ### Input text:\n{ex['tweet_text']}\n\n### Additional information:\n{ex['evidence']}\n\n ### Response:\n{label}
                      """

        prompt += f"""
                  ### Input text:\n{tweet_text}\n\n### Additional information:\n{evidence}\n\n### Response:"""
        return prompt


class ExperimentRunner:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.config = GPTExperimentConfig(model="gpt-4o")
        self.experiment_matrix = {
            "conditions": [
                ("zero_shot", "claim_only"),
                ("zero_shot", "claim_context"),
                ("few_shot", "claim_only"),
                ("few_shot", "claim_context"),
            ],
            "datasets": ["PoliClaim-GPT", "PoliClaim-M"],
        }

    def format_evidence_from_linked_entities(self, linked_entities: Dict) -> str:
        """
        Format linked_entities as title:extract pairs for evidence
        """
        if not linked_entities:
            return ""

        evidence_pairs = []
        for entity, data in linked_entities.items():
            title = data.get('title', '')
            extract = data.get('extract', '')
            if title and extract:
                evidence_pairs.append(f"{title}: {extract}")

        return "\n".join(evidence_pairs)

    def load_dataset(self, dataset_name: str, data_path: str) -> pd.DataFrame:
        """
        Load dataset from JSON file or CSV file
        """
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            processed_data = []
            for item in json_data:
                tweet_id = item.get('tweet_id', '')
                tweet_text = item.get('tweet_text', '')
                class_label = item.get('class_label', 0)
                linked_entities = item.get('linked_entities', {})

                evidence = item.get('generated_context', {})

                processed_data.append({
                    'tweet_id': tweet_id,
                    'tweet_text': tweet_text,
                    'class_label': class_label,
                    'evidence': evidence,
                    'linked_entities': linked_entities
                })

            df = pd.DataFrame(processed_data)

        else:
            if dataset_name.endswith('.tsv'):
                df = pd.read_csv(data_path, sep='\t', on_bad_lines='skip', dtype={"tweet_id": str})
            else:
                df = pd.read_csv(data_path, on_bad_lines='skip', dtype={"tweet_id": str})

        required_columns = ['tweet_id', 'tweet_text', 'class_label', 'evidence']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df

    def prepare_few_shot_examples(self, source_df: pd.DataFrame, n_examples: int = 5) -> List[Dict]:
        """
        Prepare few-shot examples from source_df
        Returns examples and their indices for removal if needed
        """
        # Filter out samples with empty or null evidence for context methods
        valid_df = source_df[
            (source_df['evidence'].notna()) &
            (source_df['evidence'].astype(str).str.strip() != '') &
            (source_df['evidence'].astype(str).str.strip() != 'nan')
            ].copy()

        logger.info(
            f"Filtered dataset for few-shot examples: {len(source_df)} -> {len(valid_df)} samples with valid evidence")

        # Check if we have enough samples for each class
        pos_count = len(valid_df[valid_df['class_label'] == 1])
        neg_count = len(valid_df[valid_df['class_label'] == 0])

        n_pos_needed = n_examples // 2
        n_neg_needed = n_examples - n_pos_needed  # Handle odd numbers properly

        logger.info(f"Available samples - Positive: {pos_count}, Negative: {neg_count}")
        logger.info(f"Needed samples - Positive: {n_pos_needed}, Negative: {n_neg_needed}")

        # Adjust if we don't have enough samples
        if pos_count < n_pos_needed:
            logger.warning(f"Not enough positive samples with evidence: {pos_count} < {n_pos_needed}")
            n_pos_needed = pos_count

        if neg_count < n_neg_needed:
            logger.warning(f"Not enough negative samples with evidence: {neg_count} < {n_neg_needed}")
            n_neg_needed = neg_count

        if n_pos_needed == 0 or n_neg_needed == 0:
            logger.error("Cannot create balanced few-shot examples - insufficient samples with evidence")
            # Fall back to using all available samples
            if pos_count > 0 and neg_count > 0:
                n_pos_needed = min(1, pos_count)
                n_neg_needed = min(1, neg_count)
            else:
                raise ValueError("No samples with valid evidence found for few-shot examples")

        # Sample examples
        pos_examples = valid_df[valid_df['class_label'] == 1].sample(n_pos_needed, random_state=42)
        neg_examples = valid_df[valid_df['class_label'] == 0].sample(n_neg_needed, random_state=42)

        # Combine and get indices
        selected_examples = pd.concat([pos_examples, neg_examples])
        examples = selected_examples.to_dict('records')
        example_indices = selected_examples.index.tolist()

        random.shuffle(examples)

        logger.info(f"Selected {len(examples)} few-shot examples ({n_pos_needed} positive, {n_neg_needed} negative)")

        # Verify examples have evidence
        for i, example in enumerate(examples):
            if not example.get('evidence') or str(example['evidence']).strip() == '':
                logger.warning(f"Example {i} has empty evidence: {example['tweet_id']}")

        return examples, example_indices

    def get_prompt(self, method: str, context_type: str, sample: Dict, examples: List[Dict] = None) -> str:
        tweet_text = sample['tweet_text']
        evidence = sample.get('evidence', '')

        if method == "zero_shot":
            if context_type == "claim_only":
                return self.config.zero_shot_prompt(tweet_text)
            else:  # claim_context
                return self.config.zero_shot_context_prompt(tweet_text, evidence)

        elif method == "few_shot":
            if context_type == "claim_only":
                return self.config.few_shot_prompt(tweet_text, examples)
            else:  # claim_context
                return self.config.few_shot_context_prompt(tweet_text, evidence, examples)

        elif method == "cot":
            if context_type == "claim_only":
                return self.config.cot_prompt(tweet_text)
            else:  # claim_context
                return self.config.cot_context_prompt(tweet_text, evidence)

    def query_gpt(self, prompt: str, method_type: str, max_retries: int = 3) -> str:
        current_max_tokens = self.config.max_tokens
        if "cot" in method_type:
            current_max_tokens = 256
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "system", "content": "You are a claim detection specialist."},
                              {"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=current_max_tokens,
                    top_p=self.config.top_p
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return "Error"

    def parse_response_with_json(self, response: str, method_type: str = "") -> tuple:
        if not response or response == "Error":
            logger.error("Empty or error response received - stopping execution")
            raise ValueError("Cannot parse empty or error response")

        # Check if this is a context-based method (has relation field)
        has_context = "context" in method_type.lower()

        try:
            # Clean the response
            response = response.strip()
            logger.debug(f"Original response: {response[:200]}...")

            # Remove code block markers if present
            if response.startswith('```json'):
                response = response[7:-3].strip()
            elif response.startswith('```'):
                response = response[3:-3].strip()

            # Method 1: Try to parse as direct JSON
            try:
                parsed_json = json.loads(response)

                # Extract prediction
                prediction_raw = parsed_json.get('predict_label', '').strip().lower()
                if prediction_raw in ['yes', 'no']:
                    prediction = 1 if prediction_raw == 'yes' else 0
                    # Extract relation if available
                    relation = parsed_json.get('relation', '') if has_context else ""
                    logger.debug(f"Method 1 success: prediction={prediction}, relation={relation[:50]}")
                    return prediction, relation
                else:
                    logger.warning(f"Invalid predict_label value: {prediction_raw}")

            except json.JSONDecodeError as e:
                logger.debug(f"Method 1 failed (JSON decode): {e}")

            # Method 2: Look for JSON patterns in the response
            json_patterns = [
                r'\{[^}]*"predict_label"[^}]*\}',  # Standard JSON pattern
                r'\{[^}]*\'predict_label\'[^}]*\}',  # Single quotes
                r'"predict_label":\s*"(Yes|No)"',  # Extract just the label
                r"'predict_label':\s*'(Yes|No)'"  # Single quotes version
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)
                if matches:
                    try:
                        if pattern.startswith('"predict_label"') or pattern.startswith("'predict_label'"):
                            # Direct label extraction
                            prediction = 1 if matches[0].lower() == 'yes' else 0
                            relation = ""
                            if has_context:
                                # Try to find relation in the response
                                relation_patterns = [
                                    r'"relation":\s*"([^"]*)"',
                                    r"'relation':\s*'([^']*)'",
                                    r'relation["\']?\s*:\s*["\']?([^"\'}\n,]*)["\']?'
                                ]
                                for rel_pattern in relation_patterns:
                                    rel_match = re.search(rel_pattern, response, re.IGNORECASE)
                                    if rel_match:
                                        relation = rel_match.group(1).strip()
                                        break
                            logger.debug(f"Method 2a success: prediction={prediction}, relation={relation[:50]}")
                            return prediction, relation
                        else:
                            # Full JSON extraction
                            json_str = matches[0]
                            parsed_json = json.loads(json_str)
                            prediction_raw = parsed_json.get('predict_label', '').strip().lower()
                            if prediction_raw in ['yes', 'no']:
                                prediction = 1 if prediction_raw == 'yes' else 0
                                relation = parsed_json.get('relation', '') if has_context else ""
                                logger.debug(f"Method 2b success: prediction={prediction}, relation={relation[:50]}")
                                return prediction, relation
                    except (json.JSONDecodeError, IndexError) as e:
                        logger.debug(f"Method 2 pattern failed: {e}")
                        continue

            # Method 3: Structured pattern matching (key: value format)
            structured_patterns = [
                r"predict_label['\"]?\s*:\s*['\"]?(yes|no)['\"]?",
                r"label['\"]?\s*:\s*['\"]?(yes|no)['\"]?",
                r"answer['\"]?\s*:\s*['\"]?(yes|no)['\"]?",
                r"prediction['\"]?\s*:\s*['\"]?(yes|no)['\"]?"
            ]

            for pattern in structured_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    prediction = 1 if match.group(1).lower() == 'yes' else 0

                    # Try to extract relation if context method
                    relation = ""
                    if has_context:
                        relation_patterns = [
                            r"relation['\"]?\s*:\s*['\"]?([^'\"}\n,]*)['\"]?",
                            r"context['\"]?\s*:\s*['\"]?([^'\"}\n,]*)['\"]?"
                        ]
                        for rel_pattern in relation_patterns:
                            relation_match = re.search(rel_pattern, response, re.IGNORECASE)
                            if relation_match:
                                relation = relation_match.group(1).strip()
                                break

                    logger.debug(f"Method 3 success: prediction={prediction}, relation={relation[:50]}")
                    return prediction, relation

            # Method 4: Look for standalone Yes/No (last resort)
            yes_patterns = [r'\byes\b', r'\bYes\b', r'\bYES\b']
            no_patterns = [r'\bno\b', r'\bNo\b', r'\bNO\b']

            yes_found = any(re.search(pattern, response) for pattern in yes_patterns)
            no_found = any(re.search(pattern, response) for pattern in no_patterns)

            if yes_found and not no_found:
                logger.debug("Method 4 success: found Yes")
                return 1, ""
            elif no_found and not yes_found:
                logger.debug("Method 4 success: found No")
                return 0, ""
            elif yes_found and no_found:
                # Both found, check which comes last
                last_yes_pos = max([response.rfind(pattern.strip('\\b')) for pattern in yes_patterns
                                    if response.rfind(pattern.strip('\\b')) != -1] + [-1])
                last_no_pos = max([response.rfind(pattern.strip('\\b')) for pattern in no_patterns
                                   if response.rfind(pattern.strip('\\b')) != -1] + [-1])

                if last_yes_pos > last_no_pos:
                    logger.debug("Method 4 success: Yes came last")
                    return 1, ""
                else:
                    logger.debug("Method 4 success: No came last")
                    return 0, ""

            # All methods failed - stop execution
            logger.error(f"All parsing methods failed for response: {response[:300]}...")
            logger.error("Stopping execution due to unparseable response")
            raise ValueError(f"Cannot parse response after trying all methods: {response[:100]}...")

        except Exception as e:
            logger.error(f"Critical error in response parsing: {e}")
            logger.error(f"Response content: {response}...")
            raise ValueError(f"Fatal parsing error: {e}")

    def run_single_experiment(self, method: str, context_type: str, dataset_name: str,
                              test_df: pd.DataFrame, train_df: pd.DataFrame = None) -> Dict:
        logger.info(f"Running experiment: {method} + {context_type} on {dataset_name}")

        examples = None
        examples_indices = []
        current_test_df = test_df.copy()

        if method == "few_shot":
            if train_df is not None and len(train_df) > 0:
                examples, _ = self.prepare_few_shot_examples(train_df)
                logger.info("Using training data for few-shot examples")
            else:
                logger.info("No training data available, selecting few-shot examples from test set")
                examples, examples_indices = self.prepare_few_shot_examples(test_df)
                current_test_df = test_df.drop(examples_indices).reset_index(drop=True)
                logger.info(
                    f"Removed {len(examples_indices)} examples from test set. Remaining: {len(current_test_df)} samples")

                removed_tweet_ids = [test_df.iloc[idx]['tweet_id'] for idx in examples_indices]
                logger.info(f"Removed tweet IDs for few-shot examples: {removed_tweet_ids}")

        results = []
        predictions = []
        true_labels = []

        for idx, sample in current_test_df.iterrows():
            logger.info(f"Processing sample {idx + 1}/{len(current_test_df)}: {sample['tweet_id']}")
            prompt = self.get_prompt(method, context_type, sample, examples)

            try:
                response = self.query_gpt(prompt, method, 1)
                prediction, relation = self.parse_response_with_json(response, f"{method}_{context_type}")
                print(f"Tweet ID: {sample['tweet_id']}, prediction: {prediction}, relation: {relation[:50]}")

                results.append({
                    'tweet_id': sample['tweet_id'],
                    'true_label': int(sample['class_label']),
                    'prediction': int(prediction),
                    'relation': relation,  # Will be empty string if no context method
                    'response': response,
                    'prompt': prompt
                })

                predictions.append(prediction)
                true_labels.append(sample['class_label'])

            except ValueError as e:
                logger.error(f"Failed to process sample {sample['tweet_id']}: {e}")
                logger.error("Stopping experiment due to parsing failure")
                raise e

            time.sleep(0.5)

        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        precision = precision_score(true_labels, predictions, pos_label=1)
        recall = recall_score(true_labels, predictions, pos_label=1)
        f1 = f1_score(true_labels, predictions, pos_label=1)

        accuracy = float(accuracy)
        precision = float(precision)
        recall = float(recall)
        f1 = float(f1)

        print(f"{method} + {context_type} on {dataset_name}:\n"
              f"accuracy = {accuracy:.4f},"
              f"\nprecision = {precision:.4f},"
              f"\nrecall = {recall:.4f},"
              f"\nf1 = {f1:.4f}")

        experiment_result = {
            'method': method,
            'context_type': context_type,
            'dataset': dataset_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'results': results,
            'summary': {
                'total_samples': len(current_test_df),
                'correct_predictions': int(sum(np.array(predictions) == np.array(true_labels))),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }

        if method == "few_shot" and examples_indices:
            experiment_result['few_shot_info'] = {
                'examples_from_test_set': True,
                'examples_count': len(examples_indices),
                'removed_tweet_ids': [test_df.iloc[idx]['tweet_id'] for idx in examples_indices],
                'original_test_size': len(test_df),
                'actual_test_size': len(current_test_df)
            }
        else:
            experiment_result['few_shot_info'] = {
                'examples_from_test_set': False,
                'examples_count': len(examples) if examples else 0,
                'original_test_size': len(test_df),
                'actual_test_size': len(current_test_df)
            }

        return experiment_result

    def save_results(self, result: Dict, output_path: str, filename: str = "detailed_results.json") -> None:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Single Experiment Results are written into : {filename}")

    def run_full_experiment(self, data_paths: Dict[str, str],
                            train_data_paths: Dict[str, str] = None) -> Dict:
        all_results = {}
        print(f"Total experiment datasets: {self.experiment_matrix['datasets']}")
        for dataset_name in self.experiment_matrix["datasets"]:
            logger.info(f"\n=== Processing dataset: {dataset_name} ===")
            if not os.path.exists(data_paths[dataset_name]):
                logger.info(f"Dataset {data_paths[dataset_name]} is not exist!")
                continue

            test_df = self.load_dataset(dataset_name, data_paths[dataset_name])
            logger.info(f"Loaded {len(test_df)} samples from {dataset_name} dataset.")
            train_df = None
            # if train_data_paths and dataset_name in train_data_paths:
            #     train_df = self.load_dataset(f"{dataset_name}_train", train_data_paths[dataset_name])

            dataset_results = {}
            for method, context_type in self.experiment_matrix["conditions"]:
                logger.info(f"Running condition: {method} + {context_type}")
                condition_key = f"{method}_{context_type}"

                try:
                    result = self.run_single_experiment(
                        method, context_type, dataset_name, test_df, train_df
                    )

                    dataset_results[condition_key] = result
                    logger.info(
                        f"{method}+{condition_key}:\nAccuracy = {result['accuracy']:.4f},\nPrecision = {result['precision']:.4f},"
                        f"\nRecall = {result['recall']:.4f},\nF1 = {result['f1']:.4f}")

                    self.save_results(result,
                                      "data/PoliClaim/experiment_results_web4good",
                                      filename=f"{dataset_name}_{method}_{context_type}_results.json")

                except ValueError as e:
                    logger.error(f"Experiment failed for {method} + {context_type} on {dataset_name}: {e}")
                    logger.error("Stopping full experiment due to parsing failure")
                    raise e

            all_results[dataset_name] = dataset_results
        return all_results

def run_multiple_exp():
    api_key = os.environ.get("OPENAI_API_KEY")
    runner = ExperimentRunner(api_key=api_key)

    data_paths = {
        "PoliClaim-GPT": "data/PoliClaim/evidence/policlaim_context_test.json",
        "PoliClaim-M": "data/PoliClaim/evidence/policlaim_mistral_context_test.json",
    }

    train_data_paths = {
        "PoliClaim-GPT": "data/PoliClaim/evidence/policlaim_context_test.json",
        "PoliClaim-M": "data/PoliClaim/evidence/policlaim_mistral_context_test.json",
    }

    try:
        all_results = runner.run_full_experiment(data_paths, train_data_paths)

        summary_data = []
        for dataset_name, dataset_results in all_results.items():
            for condition_key, result in dataset_results.items():
                summary_data.append({
                    'dataset': dataset_name,
                    'method': result['method'],
                    'context_type': result['context_type'],
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1': result['f1'],
                    'total_samples': result['summary']['total_samples'],
                    'correct_predictions': result['summary']['correct_predictions'],
                    'examples_from_test': result['few_shot_info']['examples_from_test_set'],
                    'examples_count': result['few_shot_info']['examples_count']
                })

        print("\nExperiment Summary:")
        print(summary_data)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(
            "data/PoliClaim/experiment_results_web4good"
            "/experiment_summary.csv", index=False)
        print(summary_df.to_string(index=False))

    except ValueError as e:
        logger.error(f"Multiple experiments failed: {e}")
        raise


if __name__ == "__main__":
    run_multiple_exp()
    # run_single_exp()