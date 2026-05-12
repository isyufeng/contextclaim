import re
import emoji
import warnings
import torch
from typing import List, Dict, Set, Optional, Union
from pathlib import Path

import pandas as pd
from nltk import WordNetLemmatizer
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

warnings.filterwarnings('ignore')


def preprocess_tweet(tweet: str) -> str:
    """
    Preprocess tweet with COVID-19 specific handling.
    """
    tweet = emoji.demojize(tweet)
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'&amp;', '&', tweet)
    tweet = re.sub(r'covid-19|covid19|covid', 'COVID-19', tweet, flags=re.IGNORECASE)
    tweet = re.sub(r'coronavirus', 'Coronavirus', tweet, flags=re.IGNORECASE)
    tweet = re.sub(r'sars-cov-2|sarscov2', 'SARS-CoV-2', tweet, flags=re.IGNORECASE)

    tweet = ' '.join(tweet.split())

    return tweet


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


class CovidNERExtractor:
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the COVID-19 NER extractor with GPU support.

        Args:
            model_name: The name of the HuggingFace model to use
            device: Device to use ('cuda', 'cpu', or None to auto-detect)
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

        self.covid_keywords = {
            'DISEASE': {
                'covid-19', 'coronavirus', 'corona virus', 'sars-cov-2', 'covid'
            },
            'ORG': {
                'cdc', 'fda', 'nih', 'pfizer', 'moderna', 'astrazeneca',
                'johnson & johnson', 'j & j', 'world health organization',
                'centers for disease control'
            }
        }
        # 'MEDICAL_TERM': {
        #     'vaccines', 'quarantine', 'hospitalization', 'icu', 'immune system', 'antibodies', 'ppe'
        # },
        self.keyword_patterns = self._compile_keyword_patterns()

    def _compile_keyword_patterns(self) -> Dict[str, List[str]]:
        """
        Compile regex patterns for each category of keywords, handling case and plurals
        """
        patterns = {}
        for category, terms in self.covid_keywords.items():
            normalized_terms = set()

            for term in terms:
                normalized_terms.add(term.lower())

            filtered_terms = self._filter_redundant_terms(normalized_terms)
            sorted_terms = sorted(filtered_terms, key=len, reverse=True)
            patterns[category] = [r'\b' + re.escape(term) + r'\b'
                                  for term in sorted_terms]

        return patterns

    def _filter_redundant_terms(self, terms: set) -> set:
        """
        Remove redundant singular/plural forms, keeping the more general form
        """
        filtered = set(terms)

        for term in list(terms):
            if term.endswith('s') and len(term) > 1:
                singular = term[:-1]
                if singular in terms:
                    filtered.discard(singular)

            elif term + 's' in terms:
                filtered.discard(term + 's')

        return filtered

    def _find_keyword_matches(self, text: str) -> List[Dict]:
        matches = []

        for category, patterns in self.keyword_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matches.append({
                        'word': text[match.start():match.end()],
                        'entity_group': category,
                        'score': 1.0,
                        'start': match.start(),
                        'end': match.end()
                    })

        return sorted(matches, key=lambda x: x['start'])

    def extract_covid_entities(self, tweet: str) -> List[Dict]:
        try:
            with torch.no_grad():
                model_predictions = self.ner_pipeline(tweet)
            model_predictions = process_entities(model_predictions)

            keyword_matches = self._find_keyword_matches(tweet)

            combined_entities = sorted(
                model_predictions + keyword_matches,
                key=lambda x: x.get('start', float('inf'))
            )

            filtered_matches = []
            current_end = -1

            for match in combined_entities:
                if match['start'] >= current_end:
                    filtered_matches.append(match)
                    current_end = match['end']
                elif filtered_matches and match['start'] == filtered_matches[-1]['start'] \
                        and match['end'] > filtered_matches[-1]['end']:
                    filtered_matches[-1] = match
                    current_end = match['end']

            seen = {}
            for item in filtered_matches:
                word_lower = item['word'].lower()
                if word_lower not in seen:
                    seen[word_lower] = item
            return list(seen.values())

        except Exception as e:
            print(f"Error in NER processing: {e}")
            return []

    def extract_hashtag_keywords(self, tweet: str) -> List[str]:
        """
        Extract COVID-19 related hashtags.
        """
        hashtags = re.findall(r'#(\w+)', tweet)
        covid_related_hashtags = []

        for tag in hashtags:
            words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', tag)
            tag_lower = ' '.join(words).lower()

            if any(keyword.lower() in tag_lower for keyword_list in self.covid_keywords.values()
                   for keyword in keyword_list) or \
                    any(char.isdigit() for char in tag):
                covid_related_hashtags.append(tag)

        return covid_related_hashtags

    def process_tweet(self, tweet: str) -> Dict:
        processed_tweet = preprocess_tweet(tweet)

        entities = self.extract_covid_entities(processed_tweet)
        hashtags = self.extract_hashtag_keywords(tweet)

        return {
            'tweet': tweet,
            'keywords': process_keywords(entities),
            'keyword_str': ', '.join(process_keywords(entities)),
            'entities': entities,
            'hashtags': hashtags
        }

    def batch_process_tweets(self, tweets: List[str], tweet_ids: Union[List[str], pd.Series] = None,
                             labels: Union[List[str], pd.Series] = None, batch_size: int = 16) -> pd.DataFrame:
        results = []
        total = len(tweets)

        report_interval = max(1, min(100, total // 10))
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_tweets = tweets[i:batch_end]

            if i > 0 and i % report_interval == 0:
                print(f"  Progress: {i}/{total} tweets ({i / total:.1%})")

            for j, tweet in enumerate(batch_tweets):
                idx = i + j
                try:
                    result = self.process_tweet(tweet)
                    data = {
                        'tweet_text': tweet,
                        'entities': result['entities'],
                        'keywords': result['keyword_str'],
                        'hashtags': ','.join(result['hashtags']) if result['hashtags'] else ''
                    }

                    if tweet_ids is not None:
                        data['tweet_id'] = tweet_ids.iloc[idx] if hasattr(tweet_ids, 'iloc') else tweet_ids[idx]

                    if labels is not None:
                        data['class_label'] = labels.iloc[idx] if hasattr(labels, 'iloc') else labels[idx]

                    results.append(data)

                except Exception as e:
                    print(f"Error processing tweet {idx}: {e}")
                    minimal_data = {'tweet_text': tweet}
                    if tweet_ids is not None:
                        minimal_data['tweet_id'] = tweet_ids.iloc[idx] if hasattr(tweet_ids, 'iloc') else tweet_ids[idx]
                    results.append(minimal_data)

        return pd.DataFrame(results)


def extract_keywords_from_file(file_path, model_name='dslim/bert-base-NER', output_dir=None,
                               batch_size=32, device=None):
    extractor = CovidNERExtractor(model_name, device=device)

    if output_dir is None:
        output_dir = './data/keywords'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_name = Path(file_path).stem.replace("english_1B_claim", "entities_claim")
    output_path = Path(output_dir) / f"{file_name}.csv"

    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path)
            processed_ids = set(existing_df['tweet_id'].astype(str))
            print(f"Found existing file with {len(processed_ids)} processed tweets")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            processed_ids = set()
    else:
        processed_ids = set()

    chunks = pd.read_csv(
        file_path,
        sep='\t',
        on_bad_lines='skip',
        dtype={"tweet_id": str},
        chunksize=batch_size * 10  # Read larger chunks but process in smaller batches
    )

    all_results = []
    total_processed = 0

    for i, chunk in enumerate(chunks):
        if processed_ids:
            chunk = chunk[~chunk['tweet_id'].astype(str).isin(processed_ids)]
            if len(chunk) == 0:
                continue

        tweet_ids = chunk['tweet_id']
        tweets = chunk['tweet_text']

        labels = None
        column_names = ['tweet_id', 'tweet_text', 'keywords', 'hashtags']
        if 'class_label' in chunk.columns:
            labels = chunk['class_label']
            column_names.append('class_label')

        print(f"Processing batch {i + 1} with {len(tweets)} tweets...")
        result_df = extractor.batch_process_tweets(tweets, tweet_ids, labels, batch_size=batch_size)
        all_results.append(result_df)

        for col in column_names:
            if col not in result_df.columns:
                print(f"Warning: Column '{col}' not found in results, skipping it.")
                column_names.remove(col)

        mode = 'a' if output_path.exists() else 'w'
        header = not output_path.exists()
        result_df.to_csv(output_path, columns=column_names, mode=mode, header=header, index=False)

        total_processed += len(result_df)
        print(f"Processed {total_processed} tweets so far...")

    print(f"Saved all keywords to {output_path} (Total: {total_processed} tweets)")
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)

        json_output_path = output_path.with_suffix('.json')
        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                combined_results.to_json(f, orient='records', force_ascii=False, indent=4)
            print(f"Saved all results to JSON: {json_output_path}")
        except Exception as e:
            print(f"Error saving results to JSON file: {e}")

    final_df = pd.read_csv(output_path)
    return output_path.with_suffix('.json'), final_df


def main():
    model_name = 'dslim/bert-base-NER'
    # input_path = "data/CT22_english_1B_claim/CT22_english_1B_claim_test_gold.tsv"
    # output_dir = "data/in-context_learning"
    # extract_keywords_from_file(input_path, model_name=model_name, output_dir=output_dir, batch_size=32, device="cpu")

    input_dir = "data/CT22_english_1B_claim"
    output_dir = "data/keywords"
    input_dir = Path(input_dir)
    file_pattern = '*.tsv'

    print(f"Processing files in directory: {input_dir} with pattern: {file_pattern}")
    for file_path in input_dir.glob(file_pattern):
        print(f"Processing file: {file_path}")
        keywords_path, keywords_df = extract_keywords_from_file(file_path,
                                                                  model_name=model_name,
                                                                  output_dir=output_dir,
                                                                  batch_size=100,
                                                                  device="cpu")
        print(f"Keywords extracted and saved to: {keywords_path}")


if __name__ == "__main__":
    # main()

    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Extract COVID-19 related keywords from tweets")
    # parser.add_argument("--input", type=str, help="Path to input file", required=True)
    # parser.add_argument("--output_dir", type=str, help="Output directory", default=None)
    # parser.add_argument("--model", type=str, help="HuggingFace model name", default="dslim/bert-base-NER")
    # parser.add_argument("--batch_size", type=int, help="Batch size for GPU processing", default=32)
    # parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use (default: auto-detect)",
    #                     default=None)
    #
    # args = parser.parse_args()
    #
    # extract_keywords_from_file(
    #     args.input,
    #     model_name=args.model,
    #     output_dir=args.output_dir,
    #     batch_size=args.batch_size,
    #     device=args.device
    # )

    input_file = "data/CT22_english_1B_claim/CT22_english_1B_claim_test_gold.tsv"
    output_dir = "data/keywords"

    model_name = 'dslim/bert-base-NER'
    extract_keywords_from_file(
        input_file,
        model_name=model_name,
        output_dir=output_dir,
        batch_size=100,
        device='cpu'
    )