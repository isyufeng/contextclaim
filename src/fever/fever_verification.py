import json
import logging
import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
timestamp = time.strftime("%Y%m%d%H%M%S")

file_handler = logging.FileHandler(
    f"../data/logs/fever_verification_experiment_{timestamp}.log", mode='w'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Wiki-pages index for extracting gold evidence sentences
# ─────────────────────────────────────────────────────────────────────────────

class WikiPageIndex:
    """
    Loads the pre-processed Wikipedia pages (wiki-pages.zip / *.jsonl) into
    an in-memory dict keyed by page title for fast lookup.

    Each wiki-pages line looks like:
        {
            "id": "Oliver_Reed",
            "text": "...",
            "lines": "0\tOliver Reed was a British film actor...\n1\t...\n"
        }

    Usage:
        index = WikiPageIndex("/path/to/wiki-pages/")
        sentences = index.get_sentences("Oliver_Reed", [0, 3])
    """

    def __init__(self, wiki_pages_dir: str):
        self.index: dict[str, dict[int, str]] = {}
        self._load(wiki_pages_dir)

    def _load(self, wiki_pages_dir: str):
        wiki_dir = Path(wiki_pages_dir)
        files = sorted(wiki_dir.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"No .jsonl files found in {wiki_pages_dir}. "
                "Make sure you extracted wiki-pages.zip."
            )

        logger.info(f"Loading wiki-pages from {wiki_pages_dir} ({len(files)} files)...")
        total = 0
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    page = json.loads(line)
                    page_id = page.get("id", "")
                    lines_raw = page.get("lines", "")
                    sentence_map: dict[int, str] = {}
                    for row in lines_raw.split("\n"):
                        row = row.strip()
                        if not row:
                            continue
                        parts = row.split("\t", 1)
                        if len(parts) == 2:
                            try:
                                sid = int(parts[0])
                                sentence_map[sid] = parts[1].strip()
                            except ValueError:
                                continue
                    self.index[page_id] = sentence_map
                    total += 1

        logger.info(f"Wiki-pages index built: {total} pages loaded.")

    def get_sentences(self, page_title: str, sentence_ids: list[int]) -> list[str]:
        """
        Returns the requested sentences from a Wikipedia page.
        page_title: e.g. "Oliver_Reed" (as stored in FEVER evidence)
        sentence_ids: list of integer sentence IDs
        """
        page = self.index.get(page_title, {})
        sentences = []
        for sid in sentence_ids:
            text = page.get(sid)
            if text:
                sentences.append(text)
            else:
                logger.warning(f"Sentence {sid} not found in page '{page_title}'")
        return sentences


def extract_gold_sentences(fever_evidence_raw, wiki_index: WikiPageIndex) -> str:
    """
    Parse the fever_evidence field (JSON string or list) and return
    the concatenated gold evidence sentences as a single string.

    FEVER evidence format:
        [[[ann_id, ev_id, "Page_Title", sentence_id], ...], ...]

    Strategy: use the *first* complete evidence set (annotation set)
    that has non-null page references, which is the minimal sufficient evidence.
    """
    if isinstance(fever_evidence_raw, str):
        evidence_sets = json.loads(fever_evidence_raw)
    else:
        evidence_sets = fever_evidence_raw

    # Collect (page_title, sentence_id) pairs from the first valid annotation set
    for annotation_set in evidence_sets:
        pairs = []
        for item in annotation_set:
            if len(item) >= 4 and item[2] is not None and item[3] is not None:
                pairs.append((item[2], int(item[3])))

        if not pairs:
            continue  # NOT ENOUGH INFO annotation, skip

        # Group by page, then fetch sentences
        from collections import defaultdict
        page_to_sids: dict[str, list[int]] = defaultdict(list)
        for page_title, sid in pairs:
            page_to_sids[page_title].append(sid)

        all_sentences = []
        for page_title, sids in page_to_sids.items():
            sids_sorted = sorted(set(sids))
            sentences = wiki_index.get_sentences(page_title, sids_sorted)
            all_sentences.extend(sentences)

        if all_sentences:
            return " ".join(all_sentences)

    return ""  # fallback: no valid gold sentences found


# ─────────────────────────────────────────────────────────────────────────────
# Original classes (unchanged except for new condition support)
# ─────────────────────────────────────────────────────────────────────────────

class FEVERVerificationConfig:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.temperature = 0.0
        self.max_tokens = 10
        self.top_p = 1.0
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.seed = 42

        self.system_message_claim_only = (
            "You are a fact-checking assistant. Given a claim, determine whether "
            "the claim is factually correct or incorrect based on your knowledge. "
            "Respond only with 'SUPPORTS' if the claim is correct, "
            "or 'REFUTES' if the claim is incorrect."
        )

        self.system_message_claim_context = (
            "You are a fact-checking assistant. Given a claim and additional context "
            "about the entities mentioned in the claim, determine whether the claim "
            "is factually correct or incorrect. "
            "Respond only with 'SUPPORTS' if the claim is correct, "
            "or 'REFUTES' if the claim is incorrect."
        )

        # Gold condition uses the same prompt template as claim+context
        self.system_message_claim_gold = self.system_message_claim_context

    def build_messages_claim_only(self, claim: str) -> list:
        return [
            {"role": "system", "content": self.system_message_claim_only},
            {"role": "user", "content": claim},
        ]

    def build_messages_claim_context(self, claim: str, context: str) -> list:
        return [
            {"role": "system", "content": self.system_message_claim_context},
            {"role": "user", "content": f"{claim}\n\nAdditional context:\n{context}"},
        ]

    def build_messages_claim_gold(self, claim: str, gold_context: str) -> list:
        return [
            {"role": "system", "content": self.system_message_claim_gold},
            {"role": "user", "content": f"{claim}\n\nAdditional context:\n{gold_context}"},
        ]


def load_results(output_dir: str, filename: str) -> dict:
    filepath = os.path.join(output_dir, filename)
    if not os.path.exists(filepath):
        logger.error(f"Results file not found: {filepath}")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class FEVERVerificationRunner:
    def __init__(self, api_key: str, wiki_index: WikiPageIndex | None = None):
        self.client = OpenAI(api_key=api_key)
        self.config = FEVERVerificationConfig(model="gpt-4o")
        self.wiki_index = wiki_index  # only needed for claim_gold condition

    def load_dataset(self, data_path: str) -> pd.DataFrame:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed = []
        for item in data:
            label = item.get('label', '').upper().strip()
            if label not in ['SUPPORTS', 'REFUTES']:
                continue

            processed.append({
                'tweet_id': item.get('tweet_id', ''),
                'claim': item.get('tweet_text', ''),
                'label': label,
                'generated_context': item.get('generated_context', ''),
                'fever_evidence': item.get('fever_evidence', ''),
            })

        df = pd.DataFrame(processed)
        logger.info(f"Loaded {len(df)} samples (SUPPORTS: {(df['label']=='SUPPORTS').sum()}, "
                     f"REFUTES: {(df['label']=='REFUTES').sum()})")
        return df

    def query_gpt(self, messages: list, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty,
                    seed=self.config.seed,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return "Error"

    def parse_response(self, response: str) -> str:
        response = response.upper().strip()
        if "SUPPORTS" in response and "REFUTES" in response:
            if response.rfind("SUPPORTS") > response.rfind("REFUTES"):
                return "SUPPORTS"
            else:
                return "REFUTES"
        elif "SUPPORTS" in response:
            return "SUPPORTS"
        elif "REFUTES" in response:
            return "REFUTES"
        else:
            logger.warning(f"Unparseable response: {response}")
            return "UNKNOWN"

    def run_experiment(self, df: pd.DataFrame, condition: str,
                       output_dir: str = None, checkpoint_interval: int = 50) -> dict:
        """
        Run verification for one condition with checkpoint saving and resume support.

        Args:
            df                  : DataFrame of claims to verify.
            condition           : "claim_only" | "claim_context" | "claim_gold"
            output_dir          : Directory for checkpoint files. If None, checkpointing
                                  is disabled.
            checkpoint_interval : Save a checkpoint every N samples (default: 50).
        """
        assert condition in ("claim_only", "claim_context", "claim_gold"), \
            f"Unknown condition: {condition}"

        if condition == "claim_gold" and self.wiki_index is None:
            raise ValueError(
                "WikiPageIndex must be provided to run 'claim_gold' condition. "
                "Pass wiki_index= when constructing FEVERVerificationRunner."
            )

        # ── Checkpoint setup ──────────────────────────────────────────────────
        checkpoint_path = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            checkpoint_path = Path(output_dir) / f"fever_verification_{condition}_checkpoint.json"

        # Load existing checkpoint if present
        results = []
        processed_ids = set()
        if checkpoint_path and checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                processed_ids = {str(r['tweet_id']) for r in results}
                logger.info(f"Resumed checkpoint: {len(results)} samples already done "
                            f"for condition '{condition}'")
            except Exception as e:
                logger.warning(f"Could not load checkpoint ({e}). Starting fresh.")
                results = []
        # ── End checkpoint setup ──────────────────────────────────────────────

        logger.info(f"=== Running verification: {condition} "
                    f"({len(df)} samples, {len(processed_ids)} already done) ===")

        gold_missing = 0

        for idx, row in df.iterrows():
            tweet_id = str(row['tweet_id'])

            # Skip already-processed samples
            if tweet_id in processed_ids:
                continue

            claim = row['claim']
            label = row['label']

            logger.info(f"Processing sample {len(results) + 1}/{len(df)}: tweet_id={tweet_id}")

            if condition == "claim_only":
                messages = self.config.build_messages_claim_only(claim)

            elif condition == "claim_context":
                context = row.get('generated_context', '')
                if not context:
                    messages = self.config.build_messages_claim_only(claim)
                    logger.warning(f"No generated_context for {tweet_id}, falling back to claim_only")
                else:
                    messages = self.config.build_messages_claim_context(claim, context)

            else:  # claim_gold
                gold_context = extract_gold_sentences(
                    row.get('fever_evidence', '[]'), self.wiki_index
                )
                if not gold_context:
                    gold_missing += 1
                    messages = self.config.build_messages_claim_only(claim)
                    logger.warning(f"No gold sentences found for {tweet_id}, falling back to claim_only")
                else:
                    messages = self.config.build_messages_claim_gold(claim, gold_context)

            response = self.query_gpt(messages)
            prediction = self.parse_response(response)

            results.append({
                'tweet_id': tweet_id,
                'claim': claim,
                'true_label': label,
                'prediction': prediction,
                'response': response,
                'condition': condition,
            })
            processed_ids.add(tweet_id)

            # Periodic checkpoint save
            if checkpoint_path and len(results) % checkpoint_interval == 0:
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False)
                logger.info(f"Checkpoint saved: {len(results)}/{len(df)} samples")

            time.sleep(0.2)

        if condition == "claim_gold":
            logger.info(f"Gold sentences missing (fell back to claim_only): {gold_missing}/{len(df)}")

        # ── Compute metrics from all results ──────────────────────────────────
        label_map = {"SUPPORTS": 1, "REFUTES": 0}
        predictions = [r['prediction'] for r in results if r['prediction'] != "UNKNOWN"]
        true_labels = [r['true_label'] for r in results if r['prediction'] != "UNKNOWN"]

        y_true = [label_map[l] for l in true_labels]
        y_pred = [label_map[p] for p in predictions]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        metrics = {
            'condition': condition,
            'total_samples': len(df),
            'valid_predictions': len(predictions),
            'unknown_predictions': len(df) - len(predictions),
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'macro_f1': round(macro_f1, 4),
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"Results for {condition}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1:        {metrics['f1']:.4f}")
        logger.info(f"  Macro-F1:  {metrics['macro_f1']:.4f}")
        logger.info(f"  Unknown:   {metrics['unknown_predictions']}")
        logger.info(f"{'='*50}\n")

        # Remove checkpoint on successful completion
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Checkpoint removed: {checkpoint_path}")

        return {
            'metrics': metrics,
            'results': results,
        }

    def save_results(self, result: dict, output_dir: str, filename: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filepath}")

    def run_full_experiment(self, data_path: str, output_dir: str):
        df = self.load_dataset(data_path)

        if len(df) == 0:
            logger.error("No valid samples found.")
            return

        # --- Condition 1: Claim only ---
        result_claim_only = self.run_experiment(df, "claim_only", output_dir=output_dir)
        self.save_results(result_claim_only, output_dir, "fever_verification_claim_only.json")

        # --- Condition 2: Claim + generated context ---
        result_claim_context = self.run_experiment(df, "claim_context", output_dir=output_dir)
        self.save_results(result_claim_context, output_dir, "fever_verification_claim_context.json")

        # --- Condition 3: Claim + FEVER gold sentences (oracle upper bound) ---
        result_claim_gold = self.run_experiment(df, "claim_gold", output_dir=output_dir)
        self.save_results(result_claim_gold, output_dir, "fever_verification_claim_gold.json")

        # --- Comparison Summary ---
        m1 = result_claim_only['metrics']
        m2 = result_claim_context['metrics']
        m3 = result_claim_gold['metrics']

        comparison = {
            'claim_only':    m1,
            'claim_context': m2,
            'claim_gold':    m3,
            'improvement_context_vs_claim_only': {
                metric: round(m2[metric] - m1[metric], 4)
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'macro_f1']
            },
            'improvement_gold_vs_claim_only': {
                metric: round(m3[metric] - m1[metric], 4)
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'macro_f1']
            },
        }

        self.save_results(comparison, output_dir, "fever_verification_comparison.json")

        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 70)
        logger.info(f"{'Metric':<15} {'Claim Only':<15} {'Claim+Context':<18} {'Claim+Gold':<15}")
        logger.info("-" * 63)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'macro_f1']:
            v1 = m1[metric]
            v2 = m2[metric]
            v3 = m3[metric]
            logger.info(f"{metric:<15} {v1:<15.4f} {v2:<18.4f} {v3:<15.4f}")
        logger.info("=" * 70)

        return comparison

    def run_gold_only(self, data_path: str, output_dir: str):
        """
        Entry point to run ONLY the claim_gold condition independently.
        Useful when claim_only and claim_context results already exist.
        """
        df = self.load_dataset(data_path)

        if len(df) == 0:
            logger.error("No valid samples found.")
            return

        result_claim_gold = self.run_experiment(df, "claim_gold", output_dir=output_dir)
        self.save_results(result_claim_gold, output_dir, "fever_verification_claim_gold.json")
        return result_claim_gold

    def run_context_claim_only(self, data_path: str, output_dir: str):
        df = self.load_dataset(data_path)

        if len(df) == 0:
            logger.error("No valid samples found.")
            return

        # --- Condition 1: Claim only ---
        # result_claim_only = self.run_experiment(df, "claim_only", output_dir=output_dir)
        # self.save_results(result_claim_only, output_dir, "fever_verification_claim_only.json")

        result_claim_context = self.run_experiment(df, "claim_context", output_dir=output_dir)
        self.save_results(result_claim_context, output_dir, "fever_verification_claim_context.json")

        # --- Comparison Summary ---
        result_claim_only = load_results(output_dir, "fever_verification_claim_only.json")
        m1 = result_claim_only['metrics']
        m2 = result_claim_context['metrics']

        comparison = {
            'claim_only':    m1,
            'claim_context': m2,
            'improvement_context_vs_claim_only': {
                metric: round(m2[metric] - m1[metric], 4)
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'macro_f1']
            },
        }

        self.save_results(comparison, output_dir, "fever_verification_comparison_2.json")

        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 70)
        logger.info(f"{'Metric':<15} {'Claim Only':<15} {'Claim+Context':<18}")
        logger.info("-" * 63)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'macro_f1']:
            v1 = m1[metric]
            v2 = m2[metric]
            logger.info(f"{metric:<15} {v1:<15.4f} {v2:<18.4f}")
        logger.info("=" * 70)

        return comparison


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    API_KEY = os.environ.get("OPENAI_API_KEY")

    DATA_PATH = "data/CIKM/summarized_evidence/gpt4o_context_taslp_fever_paper_test.json"
    OUTPUT_DIR = "data/CIKM/fever_verification_results"

    # Path to the extracted wiki-pages directory (contains wiki-00.jsonl, wiki-01.jsonl, ...)
    WIKI_PAGES_DIR = "data/CIKM/wiki-pages"

    # Build wiki index once (takes ~1-2 min depending on machine)
    # wiki_index = WikiPageIndex(WIKI_PAGES_DIR)

    runner = FEVERVerificationRunner(api_key=API_KEY)

    # ── Option A: Run all three conditions ──────────────────────────────────
    # runner.run_full_experiment(DATA_PATH, OUTPUT_DIR)

    # ── Option B: Run ONLY claim_gold (if claim_only/claim_context are done) ─
    runner.run_context_claim_only(DATA_PATH, OUTPUT_DIR)