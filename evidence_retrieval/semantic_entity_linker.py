import os
import json
import pickle
import requests
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Union, Any
from collections import defaultdict

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def check_cuda_availability():
    """Safely check CUDA availability without crashing if torch wasn't compiled with CUDA support"""
    try:
        return torch.cuda.is_available()
    except (AssertionError, RuntimeError):
        # This happens when PyTorch was not compiled with CUDA support
        return False


class NumpyJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that can handle NumPy types by converting them to Python native types
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def json_serialize(obj: Any) -> Any:
    """
    Recursively convert an object with numpy values to JSON-serializable types

    Args:
        obj: Object to convert (dict, list, numpy value, etc.)

    Returns:
        Object with all numpy types converted to standard Python types
    """
    if isinstance(obj, dict):
        return {key: json_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(json_serialize(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class SemanticEntityLinker:
    def __init__(self,
                 cache_dir: str = 'cache',
                 model_name: str = 'all-MiniLM-L6-v2',
                 embedding_cache_size: int = 1000,
                 device: Optional[str] = None):
        """
        Initialize entity linker with semantic matching capabilities and device selection

        Args:
            cache_dir: Directory to store cache files
            model_name: Sentence transformer model name for text embeddings
            embedding_cache_size: Maximum number of embeddings to keep in memory
            device: Device to use ('cuda', 'cpu', or None to auto-detect)
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if check_cuda_availability() else "cpu"
        else:
            self.device = device

        print(f"Entity Linker using device: {self.device}")

        # Create cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load Sentence Transformer model with device handling
        print(f"Loading Sentence Transformer model '{model_name}'...")
        try:
            if self.device == "cuda":
                self.sentence_model = SentenceTransformer(model_name, device="cuda")
            else:
                self.sentence_model = SentenceTransformer(model_name, device="cpu")
        except Exception as e:
            print(f"Error loading model on {self.device}: {e}")
            print("Falling back to CPU")
            self.device = "cpu"
            self.sentence_model = SentenceTransformer(model_name, device="cpu")

        # Cache files
        self.wiki_search_cache_file = self.cache_dir / 'wiki_search_results.pkl'
        self.embedding_cache_file = self.cache_dir / 'embeddings.pkl'

        # Load caches from disk
        self.wiki_search_cache = self._load_cache(self.wiki_search_cache_file)
        self.embedding_cache = self._load_cache(self.embedding_cache_file)

        # In-memory LRU-like cache for embeddings
        self.embedding_cache_size = embedding_cache_size
        self.embedding_cache_access = defaultdict(int)  # Track access count

        # API request rate limiting
        self.last_api_call = 0
        self.min_api_call_interval = 5  # seconds between API calls

        # Batch processing settings for GPU optimization
        self.batch_size = 16  # Default batch size for embedding

    def _load_cache(self, cache_file: Path) -> Dict:
        """
        Load cache from pickle file if it exists

        Args:
            cache_file: Path to the cache file

        Returns:
            Loaded cache dictionary or empty dict if file doesn't exist
        """
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache {cache_file}: {e}")
                # Backup the corrupted cache file
                if cache_file.exists():
                    backup_file = cache_file.with_suffix('.pkl.bak')
                    try:
                        os.rename(cache_file, backup_file)
                        print(f"Backed up possibly corrupted cache to {backup_file}")
                    except Exception:
                        pass
        return {}

    def _save_cache(self, cache: Dict, cache_file: Path):
        """
        Save cache to pickle file

        Args:
            cache: Dictionary to save
            cache_file: Path to the cache file
        """
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Error saving cache {cache_file}: {e}")

    def _rate_limit_api_call(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call

        if time_since_last_call < self.min_api_call_interval:
            sleep_time = self.min_api_call_interval - time_since_last_call
            time.sleep(sleep_time)

        self.last_api_call = time.time()

    def _manage_embedding_cache(self):
        """
        Manage the embedding cache size by removing least used entries
        """
        if len(self.embedding_cache) > self.embedding_cache_size:
            # Sort by access count and keep only most accessed entries
            items_to_keep = sorted(
                self.embedding_cache_access.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.embedding_cache_size]

            # Create new cache with only the kept items
            new_cache = {}
            for key, _ in items_to_keep:
                if key in self.embedding_cache:
                    new_cache[key] = self.embedding_cache[key]

            # Update caches
            self.embedding_cache = new_cache
            self.embedding_cache_access = defaultdict(int, {k: v for k, v in self.embedding_cache_access.items() if
                                                            k in new_cache})

            # Save updated cache
            self._save_cache(self.embedding_cache, self.embedding_cache_file)

    def get_text_embedding(self, text: str) -> np.ndarray:
        cache_key = text[:100]  # Use first 100 chars as key to avoid very long keys

        # Check cache
        if cache_key in self.embedding_cache:
            self.embedding_cache_access[cache_key] += 1
            return self.embedding_cache[cache_key]

        # Compute embedding with GPU acceleration
        try:
            with torch.no_grad():  # Disable gradient calculation for inference
                embedding = self.sentence_model.encode(text)

            # Cache the result
            self.embedding_cache[cache_key] = embedding
            self.embedding_cache_access[cache_key] = 1

            # Manage cache size
            self._manage_embedding_cache()

            return embedding
        except Exception as e:
            print(f"Error generating embedding for text: {e}")
            # Return a zero vector as fallback
            return np.zeros(self.sentence_model.get_sentence_embedding_dimension())

    def get_batch_embeddings(self, texts: List[str]) -> Dict[str, np.ndarray]:
        # Get unique texts that aren't in cache
        unique_texts = []
        cache_keys = []
        result_dict = {}

        for text in texts:
            cache_key = text[:100]
            cache_keys.append(cache_key)

            if cache_key in self.embedding_cache:
                self.embedding_cache_access[cache_key] += 1
                result_dict[cache_key] = self.embedding_cache[cache_key]
            else:
                unique_texts.append(text)

        # If there are texts not in cache, compute their embeddings
        if unique_texts:
            try:
                with torch.no_grad():
                    embeddings = self.sentence_model.encode(unique_texts, batch_size=self.batch_size)

                # Add to cache and result dict
                for i, text in enumerate(unique_texts):
                    cache_key = text[:100]
                    self.embedding_cache[cache_key] = embeddings[i]
                    self.embedding_cache_access[cache_key] = 1
                    result_dict[cache_key] = embeddings[i]
            except Exception as e:
                print(f"Error in batch embedding: {e}")
                # Create fallback embeddings
                embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
                for text in unique_texts:
                    cache_key = text[:100]
                    result_dict[cache_key] = np.zeros(embedding_dim)

            # Manage cache size
            self._manage_embedding_cache()

        return result_dict

    def is_disambiguation_page(self, title: str, extract: str) -> bool:
        disambiguation_indicators = [
            'disambiguation',
            'may refer to',
            'can refer to',
            'may mean',
            'refers to',
            'is the name of'
        ]

        for indicator in disambiguation_indicators:
            if indicator in title or indicator in extract:
                return True

        return False

    def search_wikipedia(self, query: str, limit: int = 5) -> Dict[str, Dict]:
        # Normalize query for caching
        query_key = query.lower().strip()

        # Check cache first
        if query_key in self.wiki_search_cache:
            return self.wiki_search_cache[query_key]

        try:
            self._rate_limit_api_call()
            wiki_response = requests.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query',
                    'generator': 'search',
                    'gsrsearch': query,
                    'gsrlimit': limit,
                    'prop': 'extracts',
                    'exlimit': 'max',
                    'exintro': True,
                    'exsentences': 3,
                    'explaintext': True,
                    'format': 'json'
                },
                headers={'User-Agent': 'ContextClaim/1.0 (research project)'}
            )

            if wiki_response.status_code != 200:
                print(f"Wikipedia API returned status {wiki_response.status_code} for '{query}', retrying...")
                time.sleep(5)
                wiki_response = requests.get(
                    'https://en.wikipedia.org/w/api.php',
                    params={
                        'action': 'query',
                        'generator': 'search',
                        'gsrsearch': query,
                        'gsrlimit': limit,
                        'prop': 'extracts',
                        'exlimit': 'max',
                        'exintro': True,
                        'exsentences': 3,
                        'explaintext': True,
                        'format': 'json'
                    },
                    headers={'User-Agent': 'ContextClaim/1.0 (research project)'}
                )
                if wiki_response.status_code != 200:
                    print(f"Retry failed for '{query}', skipping")
                    self.wiki_search_cache[query_key] = {}
                    return {}

            wiki_data = wiki_response.json()
            pages = wiki_data.get('query', {}).get('pages', {})

            # Cache and return results
            self.wiki_search_cache[query_key] = pages
            self._save_cache(self.wiki_search_cache, self.wiki_search_cache_file)

            return pages

        except Exception as e:
            print(f"Error searching Wikipedia for '{query}': {e}")

            # Cache empty result
            self.wiki_search_cache[query_key] = {}
            self._save_cache(self.wiki_search_cache, self.wiki_search_cache_file)

            return {}

    def link_entity(self, mention: str, context: str) -> Optional[Dict]:
        # Search Wikipedia
        pages = self.search_wikipedia(mention)

        if not pages:
            return None

        # Get context and mention embeddings
        context_embedding = self.get_text_embedding(context)
        mention_embedding = self.get_text_embedding(mention)

        best_score = -1
        best_match = None

        # Prepare batch for title and extract embeddings
        titles = []
        extracts = []
        page_info = []

        for page_id, page_data in pages.items():
            title = page_data.get('title', '')
            extract = page_data.get('extract', '')

            if self.is_disambiguation_page(title, extract):
                continue

            if extract:
                titles.append(title)
                extracts.append(extract)
                page_info.append((page_id, title, extract))

        # Get embeddings in batch for GPU efficiency
        if not titles:
            return None

        # Get embeddings for titles and extracts in batches
        title_embeddings = self.get_batch_embeddings(titles)
        extract_embeddings = self.get_batch_embeddings(extracts)

        # Process each candidate
        for i, (page_id, title, extract) in enumerate(page_info):
            try:
                # Get the embeddings
                title_key = title[:100]
                extract_key = extract[:100]

                title_embedding = title_embeddings[title_key]
                extract_embedding = extract_embeddings[extract_key]

                # Calculate similarities using vectorized operations
                context_similarity = float(cosine_similarity(
                    context_embedding.reshape(1, -1),
                    extract_embedding.reshape(1, -1)
                )[0][0])

                title_similarity = float(cosine_similarity(
                    mention_embedding.reshape(1, -1),
                    title_embedding.reshape(1, -1)
                )[0][0])

                # Combined score (weights can be adjusted)
                final_score = (0.8 * context_similarity) + (0.2 * title_similarity)

                if final_score > best_score:
                    best_score = final_score
                    best_match = {
                        'page_id': page_id,
                        'title': title,
                        'extract': extract,
                        'score': float(final_score),
                        'context_similarity': context_similarity,
                        'title_similarity': title_similarity
                    }
            except Exception as e:
                print(f"Error processing entity match: {e}")
                continue

        return best_match

    def process_keywords_batch(self,
                               keyword_groups: List[List[Tuple[str, str]]],
                               contexts: List[str],
                               min_score: float = 0.5) -> List[Dict[str, Dict]]:
        results = []

        # Process each context's keywords
        for keywords, context in zip(keyword_groups, contexts):
            context_results = {}

            if not keywords:
                results.append({})
                continue

            for keyword_tuple in keywords:
                # Handle both tuple format and string format for backward compatibility
                if isinstance(keyword_tuple, tuple):
                    keyword, entity_group = keyword_tuple
                else:
                    # If it's just a string, use default entity group
                    keyword = keyword_tuple
                    entity_group = "UNK"

                linked_entity = self.link_entity(keyword, context)

                # Apply different scoring thresholds based on entity group
                if linked_entity and (linked_entity['score'] >= min_score or entity_group in ['LOC', 'PER', 'ORG']):
                    context_results[keyword] = linked_entity

            results.append(context_results)

        return results

    def process_json_file(self, json_file: str, output_dir: str = None, batch_size: int = 50) -> str:
        print(f"Processing JSON file: {json_file}")

        # Setup output directory
        if output_dir is None:
            output_dir = 'data/linked_entities'

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Prepare output path
        file_name = Path(json_file).stem.replace("entities_claim", "linked_entities")
        output_path = Path(output_dir) / f"{file_name}.json"
        temp_output_path = Path(output_dir) / f"{file_name}_temp.json"

        # Check if we have a temp file to resume from
        all_results = []
        processed_ids = set()

        if temp_output_path.exists():
            try:
                with open(temp_output_path, 'r') as f:
                    all_results = json.load(f)
                processed_ids = {str(result.get('tweet_id', '')) for result in all_results}
                print(f"Resuming from temp file with {len(all_results)} processed tweets")
            except Exception as e:
                print(f"Error loading temp file: {e}")
                all_results = []

        # Load the JSON file
        try:
            with open(json_file, 'r') as f:
                all_tweets = json.load(f)
            print(f"Loaded {len(all_tweets)} tweets from JSON file")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return str(temp_output_path)

        # Process JSON in batches
        try:
            # Process in batches to manage memory
            for i in range(0, len(all_tweets), batch_size):
                batch_start_time = time.time()

                # Get current batch
                batch_tweets = all_tweets[i:i + batch_size]

                # Skip already processed tweets
                if processed_ids:
                    batch_tweets = [tweet for tweet in batch_tweets
                                    if str(tweet.get('tweet_id', '')) not in processed_ids]

                if len(batch_tweets) == 0:
                    continue

                print(f"Processing batch {i // batch_size + 1} with {len(batch_tweets)} tweets...")

                # Prepare batch data for parallel processing
                tweet_texts = []
                tweet_ids = []
                keyword_lists = []
                class_labels = []
                entities_lists = []

                for tweet in batch_tweets:
                    tweet_text = tweet.get('tweet_text', '')
                    tweet_id = str(tweet.get('tweet_id', ''))
                    print(f"Processing tweet ID: {tweet_id} | Text: {tweet_text[:10]}...")

                    # Extract keywords with their entity groups from entities
                    keywords = []
                    if 'entities' in tweet and tweet['entities']:
                        # Extract (word, entity_group) tuples from entities
                        keywords = [(entity.get('word', '').strip(), entity.get('entity_group', 'UNK'))
                                    for entity in tweet['entities']
                                    if entity.get('word', '').strip()]

                    tweet_texts.append(tweet_text)
                    tweet_ids.append(tweet_id)
                    keyword_lists.append(keywords)
                    entities_lists.append(tweet.get('entities', []))

                    if 'class_label' in tweet:
                        class_labels.append(tweet['class_label'])
                    else:
                        class_labels.append(None)

                # Process in GPU-optimized batches
                batch_results = []

                # Use smaller batch size for entity linking due to multiple API calls per tweet
                linking_batch_size = min(16, batch_size)

                for j in range(0, len(tweet_texts), linking_batch_size):
                    end_idx = min(j + linking_batch_size, len(tweet_texts))

                    # Process this mini-batch
                    mini_batch_texts = tweet_texts[j:end_idx]
                    mini_batch_keywords = keyword_lists[j:end_idx]
                    mini_batch_entities = entities_lists[j:end_idx]

                    # Track which tweets have keywords (but process all tweets)
                    mini_batch_has_keywords = [bool(kws) for kws in mini_batch_keywords]

                    # Process all tweets, but pass empty keywords for those without keywords
                    linked_entities_results = [{} for _ in range(len(mini_batch_texts))]

                    try:
                        # Process each tweet individually to handle empty keywords
                        for idx, (text, keywords, has_keywords) in enumerate(zip(mini_batch_texts,
                                                                                 mini_batch_keywords,
                                                                                 mini_batch_has_keywords)):
                            if has_keywords:
                                # Only process with actual keywords
                                try:
                                    # Process keywords as (word, entity_group) tuples
                                    result = self.process_keywords_batch([keywords], [text], min_score=0.5)
                                    if result and len(result) > 0:
                                        linked_entities_results[idx] = result[0]
                                except Exception as e:
                                    print(f"Error processing tweet {idx} in mini-batch: {e}")
                            # If no keywords, the default empty dict is already set
                    except Exception as e:
                        print(f"Error in batch entity linking: {e}")

                    # Construct results for all tweets
                    for k in range(j, end_idx):
                        tweet_id = tweet_ids[k]
                        tweet_text = tweet_texts[k]
                        original_entities = mini_batch_entities[k - j]
                        relative_idx = k - j

                        # Determine status based on keywords and results
                        status = 'success'
                        if not mini_batch_has_keywords[relative_idx]:
                            status = 'no_keywords'
                        elif not linked_entities_results[relative_idx]:
                            status = 'no_linked_entities'

                        # Create result for this tweet
                        result = {
                            'tweet_text': tweet_text,
                            'original_entities': original_entities,
                            'linked_entities': linked_entities_results[relative_idx],
                            'status': status
                        }

                        if tweet_id:
                            result['tweet_id'] = tweet_id
                            processed_ids.add(tweet_id)

                        if class_labels[k] is not None:
                            result['class_label'] = class_labels[k]

                        batch_results.append(result)

                # Update results and save temp file
                all_results.extend(batch_results)

                # Convert NumPy types to Python native types for JSON serialization
                json_safe_results = json_serialize(all_results)

                with open(temp_output_path, 'w') as f:
                    json.dump(json_safe_results, f)

                # Save caches periodically
                if i % 5 == 0:
                    self._save_cache(self.wiki_search_cache, self.wiki_search_cache_file)
                    self._save_cache(self.embedding_cache, self.embedding_cache_file)

                batch_time = time.time() - batch_start_time
                print(
                    f"Batch {i // batch_size + 1} completed in {batch_time:.2f}s ({len(batch_results)} tweets, {len(all_results)} total)")

            # Save final results - ensure all NumPy types are converted to Python types
            json_safe_results = json_serialize(all_results)

            with open(output_path, 'w') as f:
                json.dump(json_safe_results, f, indent=2)

            # Remove temp file
            if temp_output_path.exists():
                os.remove(temp_output_path)

            print(f"Saved {len(all_results)} linked entities to {output_path}")

        except Exception as e:
            print(f"Error processing JSON file: {e}")
            # Save progress before exiting - ensure all NumPy types are converted
            if all_results:
                json_safe_results = json_serialize(all_results)
                with open(temp_output_path, 'w') as f:
                    json.dump(json_safe_results, f)
                print(f"Saved partial results to {temp_output_path}")

            # Save caches
            self._save_cache(self.wiki_search_cache, self.wiki_search_cache_file)
            self._save_cache(self.embedding_cache, self.embedding_cache_file)

        return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Link entities in tweets using semantic matching")
    parser.add_argument('--input_dir', required=True, help='Input dir with keywords file')
    parser.add_argument('--output_dir', default='data/linked_entities', help='Output directory')
    parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence transformer model')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--cache_dir', default='cache', help='Cache directory')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='Device to use (default: auto-detect)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    processed_files = []
    file_pattern = '*.json'
    sorted_files = sorted(input_dir.glob(file_pattern), key=lambda x: x.stat().st_size)

    for file_path in sorted_files:
        print(f"Processing file: {file_path}")
        # if 'policlaim' not in str(file_path):
        #     print(f"Skipping file {file_path} as it does not match the expected pattern")
        #     continue
        #
        # if 'test' in str(file_path) in str(file_path):
        #     print(f"Skipping test file {file_path}")
        #     continue

        linker = SemanticEntityLinker(
            cache_dir=args.cache_dir,
            model_name=args.model,
            device=args.device
        )

        output_path = linker.process_json_file(
            str(file_path),
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
        print(f"Entity linking complete. Results saved to {output_path}")
