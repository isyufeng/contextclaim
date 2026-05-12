import json
import os
import re
import time
from pathlib import Path
from openai import OpenAI
from collections import deque
from datetime import datetime, timedelta


class RateLimiter:
    """Rate limiter to respect OpenAI API limits"""

    def __init__(self, rpm_limit=500, tpm_limit=30000, avg_tokens_per_request=500):
        # Request limits
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.avg_tokens_per_request = avg_tokens_per_request

        # Request tracking
        self.request_timestamps = deque()
        self.token_usage_timestamps = deque()
        self.estimated_token_usage = deque()

        # Time window (1 minute)
        self.time_window = 60  # seconds

    def _clean_old_timestamps(self, queue):
        """Remove timestamps older than the time window"""
        current_time = time.time()
        while queue and current_time - queue[0] > self.time_window:
            queue.popleft()

    def wait_if_needed(self, estimated_tokens=None):
        """Wait if we're approaching rate limits"""
        current_time = time.time()

        # Clean up old timestamps
        self._clean_old_timestamps(self.request_timestamps)
        self._clean_old_timestamps(self.token_usage_timestamps)

        # Calculate current rates
        current_rpm = len(self.request_timestamps)

        # Token estimation
        if estimated_tokens is None:
            estimated_tokens = self.avg_tokens_per_request

        # Calculate token usage rate
        total_tokens = sum(self.estimated_token_usage)
        if len(self.token_usage_timestamps) > 0:
            self._clean_old_timestamps(self.token_usage_timestamps)
            total_tokens = sum(self.estimated_token_usage)
        current_tpm = total_tokens

        # Check if we need to wait for RPM
        if current_rpm >= self.rpm_limit * 0.95:  # Using 95% as a safety buffer
            sleep_time = max(0, self.time_window - (current_time - self.request_timestamps[0]))
            print(f"RPM limit approaching ({current_rpm}/{self.rpm_limit}), waiting {sleep_time:.2f}s")
            time.sleep(sleep_time)

        # Check if we need to wait for TPM
        if current_tpm >= self.tpm_limit * 0.95:  # Using 95% as a safety buffer
            sleep_time = max(0, self.time_window - (current_time - self.token_usage_timestamps[0]))
            print(f"TPM limit approaching ({current_tpm}/{self.tpm_limit}), waiting {sleep_time:.2f}s")
            time.sleep(sleep_time)

        # Register this request
        self.request_timestamps.append(time.time())
        self.token_usage_timestamps.append(time.time())
        self.estimated_token_usage.append(estimated_tokens)

    def update_actual_token_usage(self, tokens_used, request_index=-1):
        """Update token usage with actual count after response"""
        if self.estimated_token_usage:
            # Update the most recent request with actual token count
            self.estimated_token_usage[-1] = tokens_used


class TweetContextGenerator:
    def __init__(self, api_key=None, model_name="gpt-4o", threshold=0.5,
                 rpm_limit=500, tpm_limit=30000):
        self.threshold = threshold
        self.model_name = model_name

        # Initialize OpenAI client
        print(f"Setting up OpenAI client with model: {model_name}")
        self.client = OpenAI(api_key=api_key)
        print(f"OpenAI client setup successfully...")

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(rpm_limit=rpm_limit, tpm_limit=tpm_limit)

        # For logging and tracking
        self.requests_made = 0
        self.total_tokens_used = 0
        self.start_time = time.time()

    def filter_relevant_entities(self, linked_entities):
        filtered = {}
        seen_page_ids = set()

        # filter score less than threshold and items with the same articles
        for k, v in linked_entities.items():
            # if v['score'] >= self.threshold and v['page_id'] not in seen_page_ids:
            if v['page_id'] not in seen_page_ids:
                filtered[k] = v
                seen_page_ids.add(v['page_id'])

        return filtered

    def generate_context(self, tweet_text, linked_entities):
        # Filter relevant entities -- now only filter duplicates based on page_id, not score
        relevant_entities = self.filter_relevant_entities(linked_entities)

        if not relevant_entities:
            return "No relevant context found."

        # Sort by score in descending order
        sorted_entities = sorted(relevant_entities.items(), key=lambda x: x[1]['score'], reverse=True)
        if len(sorted_entities) > 5:
            sorted_entities = sorted_entities[:5]

        # Prepare extracts for the LLM
        def format_extracts(entities):
            formatted = []
            for i, (entity_id, entity_data) in enumerate(entities):
                formatted.append(f"{i + 1}. {entity_data['extract']} (Relevance Score: {entity_data['score']:.2f})")
            return "\n".join(formatted)

        all_extracts = format_extracts(sorted_entities)

        # Clean tweet text (remove URLs)
        clean_tweet = re.sub(r'https?://\S+', '', tweet_text).strip()

        # Create a prompt for GPT-4o
        # system_message = "You are a helpful assistant. Provide only a factual summarization (under 100 words) of the most relevant information that gives important context for the tweet. Do not include any reasoning or additional information."
        # user_prompt = f"""Tweet: "{clean_tweet}"
        # Relevant Context:
        # {all_extracts}
        # Summarization:"""
        system_message = "You are a helpful assistant. Provide only a factual summarization (under 100 words) of the most relevant information that gives important context for the input text. Do not include any reasoning or additional information."
        user_prompt = f"""Input text: "{clean_tweet}"
        Relevant Context: 
        {all_extracts}
        Summarization:"""


        try:
            # Build API call
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Extract response content
            response = completion.choices[0].message.content
            time.sleep(1)  # Sleep to avoid hitting rate limits too quickly, adjust as needed


            # Update stats and rate limiter with actual usage
            total_tokens = completion.usage.total_tokens
            self.rate_limiter.update_actual_token_usage(total_tokens)
            self.requests_made += 1
            self.total_tokens_used += total_tokens

            # Log progress
            elapsed = time.time() - self.start_time
            requests_per_minute = self.requests_made / (elapsed / 60) if elapsed > 0 else 0
            tokens_per_minute = self.total_tokens_used / (elapsed / 60) if elapsed > 0 else 0

            print(f"Request {self.requests_made}: {total_tokens} tokens used")
            print(f"Current rates: {requests_per_minute:.1f} RPM, {tokens_per_minute:.1f} TPM")

        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            response = ""

        return response.strip()

    def process_tweet_json(self, tweet_json_data):
        # Extract tweet text and linked entities
        tweet_text = tweet_json_data.get('tweet_text', '')
        linked_entities = tweet_json_data.get('linked_entities', {})

        # Generate context
        context = self.generate_context(tweet_text, linked_entities)
        print(f"context: {context}")

        # Add context to the result
        result = tweet_json_data.copy()
        result['generated_context'] = context

        return result

    def process_file(self, input_file, output_dir=None, batch_size=10, start_from=None):
        """
        Process a linked-entities JSON file and generate GPT-4o context summaries.

        Args:
            input_file  : Path to linked entities JSON.
            output_dir  : Output directory.
            batch_size  : Number of tweets per batch.
            start_from  : Tweet index to start from (0-based). If None, auto-detects
                          the latest interim file and resumes from there.
        """
        if output_dir is None:
            output_dir = '../data/kw_entity_linking/evidence'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        file_name = Path(input_file).stem.replace("linked_entities", "gpt4o_context_taslp")
        output_file = Path(output_dir) / f"{file_name}.json"
        output_file = str(output_file)

        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process single tweet
        if isinstance(data, dict):
            results = [self.process_tweet_json(data)]

        # Process list of tweets in batches to show progress
        elif isinstance(data, list):
            total_tweets = len(data)

            # ── Resume logic ───────────────────────────────────────────────
            results = []
            resume_index = 0  # tweet index to start from

            if start_from is not None:
                # Manual override: load the interim file for that index if it exists
                interim_candidate = output_file.replace('.json', f'_interim_{start_from}.json')
                if Path(interim_candidate).exists():
                    with open(interim_candidate, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    resume_index = start_from
                    print(f"Loaded {len(results)} results from {interim_candidate}")
                else:
                    # No matching interim; start from the requested index with empty results
                    resume_index = start_from
                    print(f"No interim file found for index {start_from}. Starting from tweet {start_from} with empty results.")
            else:
                # Auto-detect the latest interim file
                import glob
                pattern = output_file.replace('.json', '_interim_*.json')
                interim_files = sorted(
                    glob.glob(pattern),
                    key=lambda p: int(re.search(r'_interim_(\d+)\.json$', p).group(1))
                )
                if interim_files:
                    latest_interim = interim_files[-1]
                    latest_index = int(re.search(r'_interim_(\d+)\.json$', latest_interim).group(1))
                    with open(latest_interim, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    resume_index = latest_index
                    print(f"Resuming from interim file: {latest_interim}")
                    print(f"  Already processed: {len(results)} tweets (up to index {resume_index})")
                    print(f"  Remaining: {total_tweets - resume_index} tweets")
                else:
                    print(f"No interim file found. Starting from the beginning.")
            # ── End resume logic ───────────────────────────────────────────

            for i in range(resume_index, total_tweets, batch_size):
                batch = data[i:i + batch_size]
                print(f"Processing batch {i // batch_size + 1}/{(total_tweets + batch_size - 1) // batch_size} "
                      f"(tweets {i + 1}-{min(i + batch_size, total_tweets)} of {total_tweets})...")

                # Process each tweet in the current batch
                batch_results = [self.process_tweet_json(tweet) for tweet in batch]
                results.extend(batch_results)

                # Optionally save intermediate results
                if output_file and i + batch_size < total_tweets:
                    interim_file = output_file.replace('.json', f'_interim_{i + batch_size}.json')
                    with open(interim_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"Saved interim results to {interim_file}")

                # Display rate statistics
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    print(f"Overall statistics:")
                    print(f"  - Requests: {self.requests_made}")
                    print(f"  - Tokens: {self.total_tokens_used}")
                    print(f"  - Avg RPM: {self.requests_made / (elapsed / 60):.1f}")
                    print(f"  - Avg TPM: {self.total_tokens_used / (elapsed / 60):.1f}")

        else:
            raise ValueError("Input JSON must be a dictionary or list of dictionaries")

        # Write final results to output file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved final results to {output_file}")

        # Print final stats
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            print(f"\nFinal statistics:")
            print(f"  - Total requests: {self.requests_made}")
            print(f"  - Total tokens: {self.total_tokens_used}")
            print(f"  - Average RPM: {self.requests_made / (elapsed / 60):.1f}")
            print(f"  - Average TPM: {self.total_tokens_used / (elapsed / 60):.1f}")
            print(f"  - Total runtime: {elapsed:.1f} seconds")

        return results


def run_dir(api_key="YOUR_API_KEY"):
    try:
        # Create a GPT-4o context generator with rate limits
        print("Creating TweetContextGenerator with GPT-4o...")
        generator = TweetContextGenerator(
            api_key=api_key,
            model_name="gpt-4o",
            threshold=0.5,
            rpm_limit=500,  # GPT-4o limit of 500 requests per minute
            tpm_limit=30000  # GPT-4o limit of 30,000 tokens per minute
        )

        # input_dir = "data/kw_entity_linking/linked_entities"
        # input_dir = Path(input_dir)
        # file_pattern = "*.json"
        # output_dir = "data/kw_entity_linking/evidence"
        # input_dir = "data/kw_entity_linking/linked_entities/linked_entities_v4_taslp"
        input_dir = "data/CIKM/linked_entities"
        input_dir = Path(input_dir)
        file_pattern = "*.json"
        # file_pattern = "policlaim_*.json"
        # output_dir = "data/evidence/CT22_claim/taslp"
        output_dir = "data/CIKM/summarized_evidence"
        sorted_files = sorted(input_dir.glob(file_pattern), key=lambda x: x.stat().st_size)

        processed_files = []
        start_time = time.time()
        for file_path in sorted_files:
            print(f"Processing file: {file_path}")
            tweets_with_context = generator.process_file(str(file_path), output_dir)
            processed_files.append((file_path, len(tweets_with_context)))

        time_cost = (time.time() - start_time) / 360  # Convert to minutes
        print(f"\nProcessing complete:")
        print(f"Total time: {time_cost:.1f} minutes")
        print(f"  Input directory: {input_dir}")
        print(f"  Files processed: {len(processed_files)}")
        for input_file, processed_tweets_size in processed_files:
            print(f"\n  - {input_file.name}")
            print(f"    Keywords extracted to: {processed_tweets_size}")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()


def run_single_file(api_key="YOUR_API_KEY"):
    try:
        # Create a GPT-4o context generator with rate limits
        print("Creating TweetContextGenerator with GPT-4o...")
        generator = TweetContextGenerator(
            api_key=api_key,
            model_name="gpt-4o",
            threshold=0.5,
            rpm_limit=500,  # GPT-4o limit of 500 requests per minute
            tpm_limit=30000  # GPT-4o limit of 30,000 tokens per minute
        )

        input_file_1 = "data/PoliClaim/linked_entities/policlaim_linked_entities_train.json"
        input_file_2 = "data/kw_entity_linking/linked_entities/CT22_linked_entities_dev.json"
        file_list = [input_file_1]
        output_dir = "data/PoliClaim/evidence/"

        for input_file in file_list:
            print(f"Processing file: {input_file}")
            tweets_with_context = generator.process_file(input_file, output_dir)

            print(f"\nProcessing complete:")
            print(f"  Input file: {input_file}")
            print(f"    Keywords extracted to: {len(tweets_with_context)}")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()


# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key here - replace with your actual key
    api_key = os.environ.get("OPENAI_API_KEY")
    # run_single_file(api_key)
    run_dir(api_key)