import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TweetContextGenerator:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", threshold=0.5, access_token=None,
                 device=None, offload_folder="./offload"):
        self.threshold = threshold

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading Mistral model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

        from transformers import BitsAndBytesConfig
        import os

        os.makedirs(offload_folder, exist_ok=True)

        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=access_token,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        print(f"Mistral model loaded successfully...")

    def get_llm_response(self, prompt):
        """Get response from Mistral LLM with memory optimization"""
        try:
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            input_length = inputs["input_ids"].shape[1]

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generation_config = {
                    "max_new_tokens": 256,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "do_sample": False,
                    "use_cache": True,
                }

                outputs = self.model.generate(
                    **inputs,
                    **generation_config,
                    return_dict_in_generate=False
                )

                outputs = outputs.cpu()

                full_sequence = outputs[0]
                new_tokens = full_sequence[input_length:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                if self.device == 'cuda':
                    torch.cuda.empty_cache()

            return response

        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def filter_relevant_entities(self, linked_entities):
        filtered = {}
        seen_page_ids = set()

        for k, v in linked_entities.items():
            if v['page_id'] not in seen_page_ids:
                filtered[k] = v
                seen_page_ids.add(v['page_id'])

        return filtered

    def generate_context(self, tweet_text, linked_entities):
        relevant_entities = self.filter_relevant_entities(linked_entities)

        if not relevant_entities:
            return "No relevant context found."

        sorted_entities = sorted(relevant_entities.items(), key=lambda x: x[1]['score'], reverse=True)
        if len(sorted_entities) > 5:
            sorted_entities = sorted_entities[:5]

        def format_extracts(entities):
            formatted = []
            for i, (entity_id, entity_data) in enumerate(entities):
                formatted.append(f"{i + 1}. {entity_data['extract']} (Relevance Score: {entity_data['score']:.2f})")
            return "\n".join(formatted)

        all_extracts = format_extracts(sorted_entities)
        clean_tweet = re.sub(r'https?://\S+', '', tweet_text).strip()

        # Create a prompt for Mistral in the instruct format
        # prompt = f"""<s>[INST] <<SYS>>
        # You are a helpful assistant. Provide only a factual summarization (under 100 words) of the most relevant information that gives important context for the tweet. Do not include any reasoning or additional information.
        # <</SYS>>
        # Tweet: "{clean_tweet}"
        # Relevant Context:
        # {all_extracts}
        # Summarization:"""

        prompt = f"""<s>[INST] <<SYS>>
        You are a helpful assistant. Provide only a factual summarization (under 100 words) of the most relevant information that gives important context for the input text. Do not include any reasoning or additional information.
        <</SYS>>
        Input text: "{clean_tweet}"
        Relevant Context: 
        {all_extracts}
        Summarization:"""

        response = self.get_llm_response(prompt)

        return response.replace(prompt, "").strip()

    def process_tweet_json(self, tweet_json_data):
        tweet_text = tweet_json_data.get('tweet_text', '')
        linked_entities = tweet_json_data.get('linked_entities', {})

        context = self.generate_context(tweet_text, linked_entities)
        print(f"context: {context}")

        result = tweet_json_data.copy()
        result['generated_context'] = context

        return result

    def process_file(self, input_file, output_dir=None, batch_size=10):

        if output_dir is None:
            output_dir = '../data/kw_entity_linking/evidence'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        file_name = Path(input_file).stem.replace("linked_entities", "mistral_context_taslp")
        output_file = Path(output_dir) / f"{file_name}.json"
        output_file = str(output_file)

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            results = [self.process_tweet_json(data)]

        elif isinstance(data, list):
            results = []
            total_tweets = len(data)

            for i in range(0, total_tweets, batch_size):
                batch = data[i:i + batch_size]
                print(f"Processing batch {i // batch_size + 1}/{(total_tweets + batch_size - 1) // batch_size} "
                      f"(tweets {i + 1}-{min(i + batch_size, total_tweets)} of {total_tweets})...")

                batch_results = [self.process_tweet_json(tweet) for tweet in batch]
                results.extend(batch_results)

                if output_file and i + batch_size < total_tweets:
                    interim_file = output_file.replace('.json', f'_interim_{i + batch_size}.json')
                    with open(interim_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"Saved interim results to {interim_file}")
        else:
            raise ValueError("Input JSON must be a dictionary or list of dictionaries")

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved final results to {output_file}")

        return results


def optimize_cuda_memory():
    """Configure CUDA memory settings to avoid OOM errors."""
    import torch
    import gc
    import os

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True

    gc.collect()
    torch.cuda.empty_cache()

    free = 0
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
        allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
        free = total_memory - allocated

        print(f"Total GPU memory: {total_memory:.2f} GB")
        print(f"Reserved memory: {reserved:.2f} GB")
        print(f"Allocated memory: {allocated:.2f} GB")
        print(f"Free memory: {free:.2f} GB")
    else:
        print("CUDA not available, using CPU")

    return free


if __name__ == "__main__":
    free_memory = optimize_cuda_memory()

    access_token = os.environ.get("HF_ACCESS_TOKEN")
    try:
        print("Creating TweetContextGenerator with memory-efficient settings...")
        generator = TweetContextGenerator(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            threshold=0.48,
            access_token=access_token,
            offload_folder="data/offload_folder"
        )


        input_dir = "data/kw_entity_linking/linked_entities/linked_entities_v4_taslp"
        input_dir = Path(input_dir)
        # file_pattern = "*.json"
        file_pattern = "policlaim_*.json"
        output_dir = "data/evidence/CT22_claim/taslp"

        processed_files = []
        start_time = time.time()
        for file_path in input_dir.glob(file_pattern):
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
        print(f"Error during model loading or inference: {e}")
        import traceback

        traceback.print_exc()


    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Memory usage after processing:")
    optimize_cuda_memory()