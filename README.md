# ContextClaim: Context-Augmented Claim Detection

This repository contains the code and supplementary material for **ContextClaim**, a paradigm for automated claim detection that enhances verifiability classification by integrating retrieved evidence as context.

## Overview

ContextClaim addresses a key challenge in automated fact-checking: determining whether a claim in social media is verifiable. The framework retrieves relevant evidence from Wikipedia via named entity linking and uses it as additional context to improve classification accuracy.

**Pipeline:**

```
Tweet --> Keyword & Entity Extraction --> Context Retirieval --> Context Summarization --> Classification
```

Two model families are supported:
- **BERT/RoBERTa + Cross-Attention**: Dual encoders fuse tweet and evidence representations via multi-head cross-attention
- **LLM In-Context Learning**: Zero-shot and few-shot prompting with GPT-4o, LLaMA, and Mistral

## Repository Structure

```
src/                        # Training scripts (RoBERTa, LLM-based models, hyperparameter optimization)
models/                     # PyTorch model definitions and dataset classes
evidence_retrieval/         # Evidence pipeline: keyword extraction, entity linking, evidence generation
experiment_analysis/        # Error analysis, entity analysis, visualization
shell_files/                # Example SLURM scripts for HPC execution
utils/                      # Utilities (WandB logging, file I/O)
data/                       # Datasets and intermediate outputs (not included in repo)
```

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `sentence-transformers`, `openai`, `peft`, `trl`, `wandb`, `spacy`

## Usage

### Evidence Retrieval

```bash
# Extract keywords and named entities
python evidence_retrieval/keyword_extractor.py

# Link entities to Wikipedia
python evidence_retrieval/semantic_entity_linker.py \
    --input_dir data/keywords \
    --output_dir data/linked_entities \
    --cache_dir data/cache \
    --device cuda

# Generate evidence summaries
python evidence_retrieval/evidence_generator_gpt4o.py
python evidence_retrieval/evidence_generator_mistral.py
```

### Model Training

```bash
# RoBERTa with cross-attention (tweet + evidence)
python src/roberta_tc_cross_refined_ct22.py \
    --model_id FacebookAI/roberta-large \
    --learning_rate 3e-05 \
    --batch_size 32 \
    --num_epochs 8 \
    --dropout_rate 0.23 \
    --warmup_ratio 0.15 \
    --num_runs 5 \
    --stability_test \
    --experiment_name "experiment_name" \
    --prefix "CT22_claim/CT22_gpt4o_context_claim"

# LLM in-context learning
python src/llm_verifiable_tweet_context_ct22.py \
    --model_id meta-llama/Meta-Llama-3-8B-Instruct \
    --num_epochs 3 \
    --num_runs 3 \
    --stability_test \
    --experiment_name "experiment_name" \
    --prefix "CT22_claim/CT22_gpt4o_context_claim"
```

### Datasets

- **CT22** (CheckThat! 2022 Task1B): Tweet-level claim verifiability
- **PoliClaim**: Political claim detection
- **FEVER**: Wikipedia-based fact verification

## Environment Variables

```bash
export OPENAI_API_KEY="your-key"       # For GPT-4o evidence generation and in-context learning
export HF_ACCESS_TOKEN="your-token"    # For gated HuggingFace models (LLaMA, Mistral)
```
