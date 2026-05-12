import re
import json
import warnings
import torch
from typing import List, Dict, Optional, Union
from pathlib import Path

import pandas as pd
from nltk import WordNetLemmatizer
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

warnings.filterwarnings('ignore')


def preprocess_claim(text: str) -> str:
    """
    Preprocess FEVER claim text. Lighter than tweet preprocessing
    since FEVER claims are well-formed sentences.
    """
    text = re.sub(r'&amp;', '&', text)
    text = ' '.join(text.split())
    return text


def process_keywords(entities: List[Dict]) -> List[str]:
    """
    Process keywords by lemmatizing and removing duplicates while handling acronyms.
    """
    lemmatizer = WordNetLemmatizer()
    processed_keywords = {}
    for entity in entities:
        keyword = entity['word']
        if entity['entity_group'] in ['LOC', 'PER', 'ORG']:
            processed_keywords[keyword] = keyword
            continue

        lemmatized = lemmatizer.lemmatize(keyword.lower())

        if lemmatized not in processed_keywords or \
                len(keyword) > len(processed_keywords[lemmatized]):
            processed_keywords[lemmatized] = keyword
    return list(processed_keywords.values())


def process_entities(entities):
    """
    Process entities by merging adjacent tokens.
    """
    processed_entities = []
    current_entity = None

    for entity in entities:
        if current_entity is None:
            current_entity = entity
        elif ((entity['entity_group'] == current_entity['entity_group'] and entity['start'] == current_entity['end'])
              or (entity['start'] == current_entity['end'] and entity['word'].startswith('##'))):
            current_entity['word'] += entity['word'].replace('##', '')
            current_entity['end'] = entity['end']
            current_entity['score'] = (current_entity['score'] + entity['score']) / 2
        else:
            processed_entities.append(current_entity)
            current_entity = entity

    if current_entity is not None:
        processed_entities.append(current_entity)

    return processed_entities


class FEVERNERExtractor:
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the NER extractor for FEVER claims.
        No COVID-specific keyword matching — FEVER covers general topics.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            grouped_entities=True,
            device=0 if self.device == "cuda" else -1
        )

    def extract_entities(self, text: str) -> List[Dict]:
        try:
            with torch.no_grad():
                model_predictions = self.ner_pipeline(text)
            model_predictions = process_entities(model_predictions)

            # Deduplicate by lowercase word and convert numpy types for JSON serialization
            seen = {}
            for item in model_predictions:
                item['score'] = float(item['score'])
                word_lower = item['word'].lower()
                if word_lower not in seen:
                    seen[word_lower] = item
            return list(seen.values())

        except Exception as e:
            print(f"Error in NER processing: {e}")
            return []

    def process_claim(self, claim: str) -> Dict:
        processed_claim = preprocess_claim(claim)
        entities = self.extract_entities(processed_claim)

        return {
            'claim': claim,
            'keywords': process_keywords(entities),
            'keyword_str': ', '.join(process_keywords(entities)),
            'entities': entities,
        }


def extract_keywords_from_fever(input_file, model_name='dslim/bert-base-NER',
                                 output_dir=None, batch_size=32, device=None):
    """
    Extract entities from sampled FEVER jsonl file.
    Preserves all original FEVER fields (id, verifiable, label, claim, evidence)
    and adds keyword extraction fields (keywords, entities) matching existing output format.
    """
    extractor = FEVERNERExtractor(model_name, device=device)

    # Setup output path
    if output_dir is None:
        output_dir = './data/keywords'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_name = Path(input_file).stem
    output_csv_path = Path(output_dir) / f"entities_{file_name}.csv"
    output_json_path = Path(output_dir) / f"entities_{file_name}.json"

    # Load FEVER data
    fever_data = []
    with open(input_file, 'r') as f:
        for line in f:
            fever_data.append(json.loads(line))

    print(f"Loaded {len(fever_data)} claims from {input_file}")

    results = []
    total = len(fever_data)
    report_interval = max(1, min(100, total // 10))

    for i, item in enumerate(fever_data):
        if i > 0 and i % report_interval == 0:
            print(f"  Progress: {i}/{total} claims ({i / total:.1%})")

        try:
            extraction = extractor.process_claim(item['claim'])

            result = {
                # --- FEVER original fields ---
                'id': item['id'],
                'verifiable': item['verifiable'],
                'label': item['label'],           # SUPPORTS / REFUTES (for verification experiment)
                'claim': item['claim'],
                'evidence': json.dumps(item['evidence']),  # serialize for CSV compatibility
                # --- Extraction fields (matching existing output format) ---
                'tweet_id': item['id'],          # map to tweet_id for pipeline compatibility
                'tweet_text': item['claim'],     # map to tweet_text for pipeline compatibility
                'keywords': extraction['keyword_str'],
                'entities': extraction['entities'],
                'hashtags': '',                  # no hashtags in FEVER, keep for format consistency
                'class_label': 1,                # all sampled claims are VERIFIABLE (0=non-verifiable, 1=verifiable)
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing claim {i} (id={item.get('id', 'unknown')}): {e}")
            results.append({
                'id': item.get('id'),
                'verifiable': item.get('verifiable'),
                'label': item.get('label'),
                'claim': item.get('claim'),
                'evidence': json.dumps(item.get('evidence', [])),
                'tweet_id': item.get('id'),
                'tweet_text': item.get('claim'),
                'keywords': '',
                'entities': '[]',
                'hashtags': '',
                'class_label': 1,
            })

    results_df = pd.DataFrame(results)

    # Save CSV (with columns matching existing pipeline output)
    csv_columns = ['tweet_id', 'tweet_text', 'keywords', 'hashtags', 'class_label']
    results_df.to_csv(output_csv_path, columns=csv_columns, index=False)
    print(f"Saved CSV (pipeline-compatible) to: {output_csv_path}")

    # Save JSON (with all fields including FEVER originals)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_df.to_dict(orient='records'), f, ensure_ascii=False, indent=4)
    print(f"Saved JSON (full fields) to: {output_json_path}")

    print(f"Total processed: {len(results)} claims")
    return output_json_path, results_df


if __name__ == "__main__":
    # input_file = "data/CIKM/fever_sampled_1000.jsonl"  # output from sample_fever.py
    input_file = "data/CIKM/paper_test.jsonl"
    output_dir = "data/CIKM/keywords"
    model_name = 'dslim/bert-base-NER'

    extract_keywords_from_fever(
        input_file,
        model_name=model_name,
        output_dir=output_dir,
        batch_size=100,
        device='cpu'
    )