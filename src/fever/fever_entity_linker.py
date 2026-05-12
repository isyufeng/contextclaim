"""
src/fever/fever_entity_linker.py

Entity linker for FEVER claims using the local wiki-pages dump.

Replaces Wikipedia API calls in SemanticEntityLinker with in-memory lookup over the
FEVER Wikipedia snapshot (data/CIKM/wiki-pages/), ensuring the retrieval is grounded
in the same snapshot used to construct the FEVER labels.

Pipeline position:
    evidence_retrieval/fever_keyword_extractor.py
        → [this file]
        → evidence_generator_gpt4o.py / evidence_generator_mistral.py
        → src/fever/fever_verification.py

Output format is compatible with SemanticEntityLinker.process_json_file() so all
downstream scripts work without modification.

Memory: loading 109 wiki-pages files (~5.4M articles) requires ~3-4 GB RAM.
Time:   index build ~1-2 min; linking 9999 claims ~5-10 min on CPU.
"""

import json
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STOP_WORDS = {
    'the', 'of', 'in', 'and', 'a', 'an', 'to', 'is', 'was', 'are', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'for', 'from',
    'by', 'with', 'at', 'on', 'it', 'its', 'this', 'that', 'which', 'who',
    'whom', 'whose', 'as', 'or', 'not', 'but', 'if', 'about', 'into', 'than',
    'then', 'so', 'also', 'up', 'out', 'no', 'new', 'one', 'two', 'three',
}

# Cap postings per word to avoid dominant high-frequency terms swamping results
MAX_POSTINGS_PER_WORD = 50_000

# Max chars of article text to store (intro paragraph only, sufficient for semantic matching)
TEXT_LIMIT = 600


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def decode_fever_text(text: str) -> str:
    """Decode FEVER bracket encodings (-LRB-, -RRB-, etc.) for cleaner text."""
    text = text.replace('-LRB-', '(').replace('-RRB-', ')')
    text = text.replace('-LSB-', '[').replace('-RSB-', ']')
    text = text.replace('-LCB-', '{').replace('-RCB-', '}')
    return text


def normalize_title(title: str) -> str:
    """Convert a keyword/mention to wiki-pages id format (spaces → underscores)."""
    return title.strip().replace(' ', '_')


def json_serialize(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_serialize(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Wiki index
# ─────────────────────────────────────────────────────────────────────────────

class FEVERWikiIndex:
    """
    In-memory index over the FEVER Wikipedia dump (wiki-pages/*.jsonl).

    Provides two access modes:
      - get_page(title)   : exact title lookup, O(1)
      - search(query, k)  : keyword search over article titles via inverted index

    The `text` field (intro paragraph) is stored truncated to TEXT_LIMIT chars and
    used as the candidate extract during semantic entity linking.

    Example:
        index = FEVERWikiIndex("data/CIKM/wiki-pages")
        pages = index.search("Albert Einstein", limit=5)
        page  = index.get_page("Albert_Einstein")
    """

    def __init__(self, wiki_pages_dir: str, verbose: bool = True):
        self.verbose = verbose
        # page_id (e.g. "Albert_Einstein") → intro text (truncated)
        self._pages: Dict[str, str] = {}
        # lowercase word → list of page_ids (capped at MAX_POSTINGS_PER_WORD)
        self._word_index: Dict[str, List[str]] = defaultdict(list)
        self._load(wiki_pages_dir)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _load(self, wiki_pages_dir: str):
        wiki_dir = Path(wiki_pages_dir)
        files = sorted(wiki_dir.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"No .jsonl files found in '{wiki_pages_dir}'. "
                "Ensure the wiki-pages directory contains wiki-001.jsonl ... wiki-109.jsonl."
            )

        self._log(f"[FEVERWikiIndex] Loading {len(files)} files from {wiki_pages_dir} ...")
        t0 = time.time()
        total = 0

        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    page = json.loads(line)
                    page_id: str = page.get('id', '')
                    if not page_id:
                        continue

                    # Store truncated intro text
                    raw_text = page.get('text', '')
                    self._pages[page_id] = decode_fever_text(raw_text)[:TEXT_LIMIT]

                    # Index title words (split on underscores, hyphens, spaces)
                    words = re.split(r'[_\-\s]+', page_id.lower())
                    for word in words:
                        if len(word) >= 2 and word not in STOP_WORDS:
                            postings = self._word_index[word]
                            if len(postings) < MAX_POSTINGS_PER_WORD:
                                postings.append(page_id)

                    total += 1

        elapsed = time.time() - t0
        self._log(
            f"[FEVERWikiIndex] Built index: {total:,} pages, "
            f"{len(self._word_index):,} index terms, {elapsed:.1f}s"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def get_page(self, title: str) -> Optional[Dict]:
        """
        Direct lookup by exact page title (as stored in wiki-pages 'id' field).
        Matches the format of FEVER evidence annotations (e.g. 'Mike_Huckabee').
        """
        text = self._pages.get(title)
        if text is None:
            return None
        return {'page_id': title, 'title': title.replace('_', ' '), 'extract': text}

    def search(self, query: str, limit: int = 5) -> Dict[str, Dict]:
        """
        Search for candidate pages matching the query keyword.

        Returns a dict of {page_id: {page_id, title, extract}} — the same shape
        as SemanticEntityLinker.search_wikipedia() so the semantic re-ranking
        logic in FEVEREntityLinker is identical.
        """
        candidates = self._rank_candidates(query, top_n=limit * 6)
        result = {}
        for page_id in candidates:
            text = self._pages.get(page_id, '')
            result[page_id] = {
                'page_id': page_id,
                'title': page_id.replace('_', ' '),
                'extract': text,
            }
            if len(result) >= limit:
                break
        return result

    # ── Internal search logic ─────────────────────────────────────────────────

    def _rank_candidates(self, query: str, top_n: int) -> List[str]:
        """
        Score candidate page_ids for a query using:
          1. Exact normalised title match (highest priority)
          2. Title-cased variant match
          3. Word overlap from inverted index
        """
        scored: Dict[str, float] = defaultdict(float)

        # Priority 1: exact normalised match
        norm = normalize_title(query)
        if norm in self._pages:
            scored[norm] += 100.0

        # Priority 2: title-cased variant
        title_cased = normalize_title(query.title())
        if title_cased in self._pages and title_cased != norm:
            scored[title_cased] += 90.0

        # Priority 3: word overlap in inverted index
        query_words = re.split(r'[\s_\-]+', query.lower())
        meaningful_words = [w for w in query_words if len(w) >= 2 and w not in STOP_WORDS]
        for word in meaningful_words:
            for page_id in self._word_index.get(word, []):
                scored[page_id] += 1.0

        sorted_candidates = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_candidates[:top_n]]

    def __len__(self) -> int:
        return len(self._pages)


# ─────────────────────────────────────────────────────────────────────────────
# Entity linker
# ─────────────────────────────────────────────────────────────────────────────

class FEVEREntityLinker:
    """
    Entity linker for FEVER claims using FEVERWikiIndex instead of the Wikipedia API.

    The semantic re-ranking logic is identical to SemanticEntityLinker:
      final_score = 0.8 * cosine(context, extract) + 0.2 * cosine(mention, title)

    This ensures results are directly comparable to the existing pipeline.

    Args:
        wiki_index : Pre-built FEVERWikiIndex.
        model_name : Sentence Transformer model (default: all-MiniLM-L6-v2).
        min_score  : Minimum score to accept a linked entity (relaxed for PER/ORG/LOC).
        cache_dir  : Optional directory for embedding cache (speeds up reruns).
        device     : 'cuda' | 'cpu' | None (auto-detect).
    """

    def __init__(self,
                 wiki_index: FEVERWikiIndex,
                 model_name: str = 'all-MiniLM-L6-v2',
                 min_score: float = 0.5,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        self.wiki_index = wiki_index
        self.min_score = min_score

        # Device selection
        if device is None:
            try:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                self.device = 'cpu'
        else:
            self.device = device

        print(f"[FEVEREntityLinker] Device: {self.device}")
        print(f"[FEVEREntityLinker] Loading sentence transformer '{model_name}' ...")
        self.sentence_model = SentenceTransformer(model_name, device=self.device)

        # Embedding cache (keyed by first 100 chars of text)
        self._emb_cache: Dict[str, np.ndarray] = {}
        self._cache_path: Optional[Path] = None
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self._cache_path = Path(cache_dir) / 'fever_linker_embeddings.pkl'
            if self._cache_path.exists():
                try:
                    with open(self._cache_path, 'rb') as f:
                        self._emb_cache = pickle.load(f)
                    print(f"[FEVEREntityLinker] Loaded {len(self._emb_cache):,} cached embeddings")
                except Exception as e:
                    print(f"[FEVEREntityLinker] Warning: embedding cache load failed: {e}")

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        key = text[:100]
        if key not in self._emb_cache:
            with torch.no_grad():
                self._emb_cache[key] = self.sentence_model.encode(text)
        return self._emb_cache[key]

    def _embed_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        to_encode: List[str] = []
        keys: List[str] = []

        for text in texts:
            key = text[:100]
            if key in self._emb_cache:
                result[key] = self._emb_cache[key]
            else:
                to_encode.append(text)
                keys.append(key)

        if to_encode:
            with torch.no_grad():
                embeddings = self.sentence_model.encode(to_encode, batch_size=32)
            for key, emb in zip(keys, embeddings):
                self._emb_cache[key] = emb
                result[key] = emb

        return result

    def _save_cache(self):
        if self._cache_path:
            try:
                with open(self._cache_path, 'wb') as f:
                    pickle.dump(self._emb_cache, f)
            except Exception as e:
                print(f"[FEVEREntityLinker] Warning: could not save embedding cache: {e}")

    # ── Core linking logic ────────────────────────────────────────────────────

    @staticmethod
    def _is_disambiguation(title: str, extract: str) -> bool:
        indicators = [
            'disambiguation', 'may refer to', 'can refer to',
            'may mean', 'refers to', 'is the name of',
        ]
        combined = (title + ' ' + extract).lower()
        return any(ind in combined for ind in indicators)

    def link_entity(self, mention: str, context: str,
                    entity_group: str = 'UNK') -> Optional[Dict]:
        """
        Link a mention string to the best matching Wikipedia page.

        Scoring (identical to SemanticEntityLinker):
            final_score = 0.8 * cosine(context_emb, extract_emb)
                        + 0.2 * cosine(mention_emb, title_emb)

        Returns a dict with keys: page_id, title, extract, score,
        context_similarity, title_similarity — or None if no good match.
        """
        pages = self.wiki_index.search(mention, limit=5)
        if not pages:
            return None

        context_emb = self._embed(context)
        mention_emb = self._embed(mention)

        # Collect valid candidates (skip empty or disambiguation pages)
        candidates = []
        titles_display = []
        extracts = []

        for page_id, page_data in pages.items():
            extract = page_data['extract']
            title_display = page_data['title']  # underscores already replaced
            if not extract:
                continue
            if self._is_disambiguation(title_display, extract):
                continue
            candidates.append((page_id, title_display, extract))
            titles_display.append(title_display)
            extracts.append(extract)

        if not candidates:
            return None

        # Batch-encode titles and extracts
        title_embs = self._embed_batch(titles_display)
        extract_embs = self._embed_batch(extracts)

        best_score = -1.0
        best_match: Optional[Dict] = None

        for page_id, title_display, extract in candidates:
            t_key = title_display[:100]
            e_key = extract[:100]
            if t_key not in title_embs or e_key not in extract_embs:
                continue

            context_sim = float(cosine_similarity(
                context_emb.reshape(1, -1),
                extract_embs[e_key].reshape(1, -1)
            )[0][0])

            title_sim = float(cosine_similarity(
                mention_emb.reshape(1, -1),
                title_embs[t_key].reshape(1, -1)
            )[0][0])

            final_score = 0.8 * context_sim + 0.2 * title_sim

            if final_score > best_score:
                best_score = final_score
                best_match = {
                    'page_id': page_id,
                    'title': title_display,
                    'extract': extract,
                    'score': float(final_score),
                    'context_similarity': context_sim,
                    'title_similarity': title_sim,
                }

        if best_match is None:
            return None

        # Apply threshold — relax for named entities (PER/ORG/LOC)
        if best_match['score'] < self.min_score and entity_group not in ('PER', 'ORG', 'LOC'):
            return None

        return best_match

    def link_claim(self, claim_text: str, entities: List[Dict]) -> Dict[str, Dict]:
        """Link all NER entities from a single claim. Returns keyword → page dict."""
        linked: Dict[str, Dict] = {}
        for entity in entities:
            word = entity.get('word', '').strip()
            entity_group = entity.get('entity_group', 'UNK')
            if not word:
                continue
            result = self.link_entity(word, claim_text, entity_group)
            if result:
                linked[word] = result
        return linked

    # ── Batch processing ──────────────────────────────────────────────────────

    def process_json_file(self,
                          input_file: str,
                          output_dir: str,
                          batch_size: int = 100,
                          save_interval: int = 500) -> str:
        """
        Process a keywords JSON file (output of fever_keyword_extractor.py) and
        write linked entities JSON compatible with downstream scripts.

        Preserves FEVER-specific fields (label, fever_evidence) so the output can be
        used directly by evidence_generator_*.py and fever_verification.py without
        re-joining the original FEVER annotations.

        Supports resuming: if a _temp.json file exists, already-processed claims
        are skipped.

        Args:
            input_file    : Path to keywords JSON (entities_*.json).
            output_dir    : Output directory.
            batch_size    : Claims per progress log line.
            save_interval : Save temp checkpoint every N claims.

        Returns:
            Path to final output JSON.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        stem = Path(input_file).stem.replace('entities_', 'linked_entities_fever_')
        output_path = Path(output_dir) / f"{stem}.json"
        temp_path = Path(output_dir) / f"{stem}_temp.json"

        # Load input
        with open(input_file, 'r', encoding='utf-8') as f:
            all_claims: List[Dict] = json.load(f)
        print(f"[FEVEREntityLinker] Loaded {len(all_claims):,} claims from {input_file}")

        # Resume support
        results: List[Dict] = []
        processed_ids: set = set()
        if temp_path.exists():
            try:
                with open(temp_path, 'r') as f:
                    results = json.load(f)
                processed_ids = {str(r.get('tweet_id', '')) for r in results}
                print(f"[FEVEREntityLinker] Resuming: {len(results):,} already done, "
                      f"{len(all_claims) - len(results):,} remaining")
            except Exception as e:
                print(f"[FEVEREntityLinker] Could not load temp file ({e}). Starting fresh.")
                results = []

        t0 = time.time()
        n_done = len(results)

        for i, item in enumerate(all_claims):
            tweet_id = str(item.get('tweet_id', item.get('id', i)))
            if tweet_id in processed_ids:
                continue

            claim_text = item.get('tweet_text', item.get('claim', ''))

            # Deserialise entities field (may be a JSON string from CSV round-trip)
            entities = item.get('entities', [])
            if isinstance(entities, str):
                try:
                    entities = json.loads(entities)
                except Exception:
                    entities = []

            keywords = item.get('keywords', '')

            # Link entities
            if entities:
                linked = self.link_claim(claim_text, entities)
                if linked:
                    status = 'success'
                elif not keywords:
                    status = 'no_keywords'
                else:
                    status = 'no_linked_entities'
            else:
                linked = {}
                status = 'no_keywords'

            result: Dict = {
                'tweet_id': tweet_id,
                'tweet_text': claim_text,
                'original_entities': entities,
                'linked_entities': linked,
                'status': status,
                'class_label': item.get('class_label', 1),
                # FEVER-specific fields — preserved for downstream use
                'label': item.get('label', ''),
                'fever_evidence': item.get('evidence', ''),
            }

            results.append(result)
            processed_ids.add(tweet_id)
            n_done += 1

            # Progress log
            if n_done % batch_size == 0:
                elapsed = time.time() - t0
                processed_this_run = n_done - len(processed_ids) + len(results) - len(all_claims) + len(all_claims)
                rate = n_done / elapsed if elapsed > 0 else 1
                remaining_claims = len(all_claims) - n_done
                eta_min = remaining_claims / rate / 60 if rate > 0 else 0
                print(f"  [{n_done:,}/{len(all_claims):,}]  "
                      f"{elapsed:.0f}s elapsed  |  ETA ~{eta_min:.1f} min  |  "
                      f"{self._status_summary(results[-batch_size:])}")

            # Periodic checkpoint
            if n_done % save_interval == 0:
                self._write_json(json_serialize(results), temp_path)
                self._save_cache()

        # Final output
        self._write_json(json_serialize(results), output_path)
        if temp_path.exists():
            temp_path.unlink()
        self._save_cache()

        elapsed_total = time.time() - t0
        print(f"\n[FEVEREntityLinker] Finished {len(results):,} claims in "
              f"{elapsed_total/60:.1f} min")
        print(f"  {self._status_summary(results)}")
        print(f"  Output → {output_path}")
        return str(output_path)

    @staticmethod
    def _status_summary(results: List[Dict]) -> str:
        counts: Dict[str, int] = defaultdict(int)
        for r in results:
            counts[r.get('status', 'unknown')] += 1
        return '  '.join(f"{k}={v}" for k, v in sorted(counts.items()))

    @staticmethod
    def _write_json(data: Any, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FEVER entity linker — local wiki-pages lookup + semantic re-ranking"
    )
    parser.add_argument(
        '--input', required=True,
        help='Keywords JSON from fever_keyword_extractor.py (e.g. entities_paper_test.json)'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Directory for linked entities output'
    )
    parser.add_argument(
        '--wiki_pages_dir', required=True,
        help='Path to wiki-pages directory (wiki-001.jsonl … wiki-109.jsonl)'
    )
    parser.add_argument(
        '--model', default='all-MiniLM-L6-v2',
        help='Sentence Transformer model name (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--min_score', type=float, default=0.5,
        help='Minimum semantic similarity to accept a link (default: 0.5)'
    )
    parser.add_argument(
        '--cache_dir', default=None,
        help='Directory for embedding cache (optional, speeds up reruns)'
    )
    parser.add_argument(
        '--device', choices=['cuda', 'cpu'], default=None,
        help='Device (default: auto-detect)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=100,
        help='Claims per progress log line (default: 100)'
    )
    parser.add_argument(
        '--save_interval', type=int, default=500,
        help='Save checkpoint every N claims (default: 500)'
    )

    args = parser.parse_args()

    # Build wiki index once (one-time cost, ~1-2 min)
    wiki_index = FEVERWikiIndex(args.wiki_pages_dir)

    linker = FEVEREntityLinker(
        wiki_index=wiki_index,
        model_name=args.model,
        min_score=args.min_score,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    linker.process_json_file(
        input_file=args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
    )
