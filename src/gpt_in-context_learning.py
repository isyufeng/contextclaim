import json
import logging
import os

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
timestamp = time.strftime("%Y%m%d%H%M%S")
file_handler = logging.FileHandler(f"data/logs/gpt_experiment_{timestamp}.log",
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

        self.system_message = """Determine if this tweet contains verifiable claims. if it contains claims that can be verified, respond "Yes". Otherwise, respond "No".
                        Note: When in doubt, choose "Yes". In the end, respond only with 'Yes' for verifiable claims or 'No' for non-verifiable claims."""

        self.context_system_message = """Determine if this tweet contains verifiable claims.
                                      A tweet contains verifiable claims if it makes specific factual statements that can be checked against evidence.
                                      The additional information is provided for context but should not be the main basis for your decision.
                                      If the tweet contains claims that can be verified, respond "Yes". Otherwise, respond "No".
                                      Note: When in doubt, choose "Yes". In the end, respond only with 'Yes' for verifiable claims or 'No' for non-verifiable claims."""

    def zero_shot_prompt(self, tweet_text: str) -> str:
        return f"""### Instruction:\n{self.system_message}\n\n### Input tweet:\n{tweet_text}\n\n### Response:"""

    def zero_shot_context_prompt(self, tweet_text: str, evidence: str) -> str:
        return f"""### Instruction:\n{self.context_system_message}\n\n### Input tweet:\n{tweet_text}\n\n### Additional information:\n{evidence}\n\n### Response:"""

    def few_shot_prompt(self, tweet_text: str, examples: List[Dict]) -> str:
        prompt = f"""### Instruction:\n{self.system_message}\n\n### Examples:\n"""
        for ex in examples:
            label = 'Yes' if ex['class_label'] == 1 else 'No'
            prompt += f"""
                      ### Input tweet:\n{ex['tweet_text']}\n\n### Response:\n{label}
                      """

        prompt += f"""
                    ### Input tweet:\n{tweet_text}\n\n### Response:"""
        return prompt

    def few_shot_context_prompt(self, tweet_text: str, evidence: str, examples: List[Dict]) -> str:
        prompt = f"""### Instruction:\n{self.context_system_message}\n\n### Examples:\n"""

        for ex in examples:
            label = 'Yes' if ex['class_label'] == 1 else 'No'
            prompt += f"""
                      ### Input tweet:\n{ex['tweet_text']}\n\n### Additional information:\n{ex['evidence']}\n\n ### Response:\n{label}
                      """

        prompt += f"""
                  ### Input tweet:\n{tweet_text}\n\n### Additional information:\n{evidence}\n\n### Response:"""
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
            "datasets": ["CT22_test", "CT22_dev_test"],
            "sample_sizes": {
                "CT22": {"dev_test": 911, "test": 251},
                "PoliClaim": {"subset": 500}
            }
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

                original_entities = item.get('original_entities', [])
                if original_entities:
                    entity_words = [entity.get('word', '') for entity in original_entities]
                    evidence = ', '.join(entity_words)
                else:
                    evidence = ''

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

    def prepare_few_shot_examples(self, train_df: pd.DataFrame, n_examples: int = 5) -> List[Dict]:
        pos_examples = train_df[train_df['class_label'] == 1].sample(n_examples // 2, random_state=42)
        neg_examples = train_df[train_df['class_label'] == 0].sample(n_examples // 2, random_state=42)

        examples = pd.concat([pos_examples, neg_examples]).to_dict('records')
        random.shuffle(examples)
        return examples

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
                    messages=[{"role": "user", "content": prompt}],
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

    def parse_response(self, response: str) -> int:
        response = response.lower().strip()

        if "yes" in response and "no" in response:
            logger.info(f"!!! The response contains both Yes and No: {response}")
            last_yes = response.rfind("yes")
            last_no = response.rfind("no")
            if last_yes > last_no:
                return 1
            else:
                return 0
        elif "yes" in response:
            return 1
        elif "no" in response:
            return 0
        else:
            return 1

    def run_single_experiment(self, method: str, context_type: str, dataset_name: str,
                              test_df: pd.DataFrame, train_df: pd.DataFrame = None) -> Dict:
        logger.info(f"Running experiment: {method} + {context_type} on {dataset_name}")

        examples = None
        if method == "few_shot" and train_df is not None:
            examples = self.prepare_few_shot_examples(train_df)

        results = []
        predictions = []
        true_labels = []

        for idx, sample in test_df.iterrows():
            logger.info(f"Processing sample {idx + 1}/{len(test_df)}: {sample['tweet_id']}")
            prompt = self.get_prompt(method, context_type, sample, examples)

            response = self.query_gpt(prompt, method, 1)
            prediction = self.parse_response(response)

            results.append({
                'tweet_id': sample['tweet_id'],
                'true_label': int(sample['class_label']),
                'prediction': int(prediction),
                'response': response,
                'prompt': prompt
            })

            predictions.append(prediction)
            true_labels.append(sample['class_label'])

            time.sleep(0.2)

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

        return {
            'method': method,
            'context_type': context_type,
            'dataset': dataset_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'results': results,
            'summary': {
                'total_samples': len(test_df),
                'correct_predictions': int(sum(np.array(predictions) == np.array(true_labels))),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }

    def save_results(self, result: Dict, output_path: str, filename: str = "detailed_results.json") -> None:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Single Experiment Results are written into : {filename}")

    def run_full_experiment(self, data_paths: Dict[str, str],
                            train_data_paths: Dict[str, str] = None) -> Dict:
        all_results = {}
        for dataset_name in self.experiment_matrix["datasets"]:
            logger.info(f"\n=== Processing dataset: {dataset_name} ===")
            if not os.path.exists(data_paths[dataset_name]):
                logger.info(f"Dataset {data_paths[dataset_name]} is not exist!")
                continue

            test_df = self.load_dataset(dataset_name, data_paths[dataset_name])
            # sample_test_df = test_df.sample(n=10, random_state=42)
            logger.info(f"Loaded {len(test_df)} samples from {dataset_name} dataset.")
            train_df = None
            if train_data_paths and dataset_name in train_data_paths:
                train_df = self.load_dataset(f"{dataset_name}_train", train_data_paths[dataset_name])

            dataset_results = {}
            for method, context_type in self.experiment_matrix["conditions"]:
                logger.info(f"Running condition: {method} + {context_type}")
                condition_key = f"{method}_{context_type}"

                result = self.run_single_experiment(
                    method, context_type, dataset_name, test_df, train_df
                )

                dataset_results[condition_key] = result
                logger.info(
                    f"{method}+{condition_key}:\nAccuracy = {result['accuracy']:.4f},\nPrecision = {result['precision']:.4f},"
                    f"\nRecall = {result['recall']:.4f},\nF1 = {result['f1']:.4f}")

                self.save_results(result, "data/in-context_learning/experiment_results",
                                  filename=f"{dataset_name}_{method}_{context_type}_CC-Entities_results.json")

            all_results[dataset_name] = dataset_results
        return all_results


def run_single_experiment():
    api_key = os.environ.get("OPENAI_API_KEY")
    runner = ExperimentRunner(api_key=api_key)

    # Now supports both JSON and CSV files
    data_paths = {
        # JSON file example:
        "CT22": "data/in-context_learning/CT22_dataset/CT22_gpt4o_generated_context_test_gold.json",
        # "CT22": "data/evidence/CT22_claim/CT22_gpt4o_generated_context_test_gold.csv",
        "PoliClaim": "path/to/PoliClaim.json"
    }

    train_data_paths = {
        "CT22": "data/in-context_learning/CT22_dataset/CT22_gpt4o_generated_context_train.json",
        # "CT22": "data/evidence/CT22_claim/CT22_gpt4o_generated_context_train.csv",
        "PoliClaim": "path/to/PoliClaim_train.json"
    }

    test_df = runner.load_dataset("CT22", data_paths["CT22"])
    # sample_test_df = test_df.sample(n=10, random_state=42)
    train_df = runner.load_dataset("CT22_train", train_data_paths["CT22"])
    results = runner.run_single_experiment("few_shot", "claim_context", "CT22", test_df, train_df)
    print(results['summary'])
    filename = f"CT22_few_shot_claim_context_CC-Entities_results.json"
    output_path = "data/in-context_learning/experiment_results"
    runner.save_results(results, output_path, filename)


def run_multiple_exp():
    api_key = os.environ.get("OPENAI_API_KEY")
    runner = ExperimentRunner(api_key=api_key)

    data_paths = {
        # JSON file example:
        "CT22_test": "data/in-context_learning/CT22_dataset/CT22_generated_context_test_gold.json",
        "CT22_dev_test": "data/in-context_learning/CT22_dataset/CT22_generated_context_dev_test.json",
        # "PoliClaim": "data/PoliClaim/evidence/policlaim_context_test.json",
    }

    train_data_paths = {
        "CT22_test": "data/in-context_learning/CT22_dataset/CT22_generated_context_train.json",
        "CT22_dev_test": "data/in-context_learning/CT22_dataset/CT22_generated_context_train.json",
        # "PoliClaim": "data/PoliClaim/evidence/policlaim_context_train.json",
    }
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
                'correct_predictions': result['summary']['correct_predictions']
            })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("data/in-context_learning/experiment_results/CC-Mistral_experiment_summary.csv", index=False)
    print("\nExperiment Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    # run_multiple_exp()
    run_single_experiment()