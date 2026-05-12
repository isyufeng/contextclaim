import argparse

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import torch
from peft import LoraConfig
# TRL 0.15+: SFTConfig replaces SFTTrainer's TrainingArguments integration;
# SFTTrainer still exists but dataset_text_field / max_seq_length moved to SFTConfig
from trl import SFTTrainer, SFTConfig
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datetime import datetime
import gc
import os
import statistics
import csv
import random
import numpy as np
from collections import Counter
from contextlib import contextmanager
import time

gc.collect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


# ── Memory helpers ────────────────────────────────────────────────────────────

def clear_memory():
    """Aggressively clear GPU and RAM memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def track_memory():
    """Context manager to track peak memory usage"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            max_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(f"Peak memory usage: {max_memory:.2f} MB")
    else:
        yield


def optimize_model_memory(model):
    """Apply memory optimizations to the model"""
    if hasattr(model, "config"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model


# ── Prompt generators ─────────────────────────────────────────────────────────

# [FIX 1] Extract system message as a module-level constant to avoid
# duplication and prevent inconsistencies between train/test prompts.
_SYSTEM_MESSAGE = """Determine if the input text contains verifiable claims.
    The input text contains verifiable claims if it makes specific factual statements that can be checked against evidence.

    Additional information may help clarify what the claim refers to, but base your decision primarily on whether the tweet makes specific factual statements.

    If the input text contains claims that can be verified, respond "Yes". Otherwise, respond "No".
    Note: When in doubt, choose "Yes". In the end, respond only with 'Yes' for verifiable claims or 'No' for non-verifiable claims."""


def generate_prompt(sample):
    tweet_text = sample["tweet_text"]
    evidence = sample["evidence"]
    class_label = 'Yes' if sample["class_label"] != 0 else 'No'

    full_prompt = (
            "### Instruction:\n" + _SYSTEM_MESSAGE +
            # [FIX 2] Unified label: use "Input text" consistently in both
            # generate_prompt and generate_test_prompt. The old code used
            # "Input text" here but "Input tweet" in the test prompt.
            "\n\n### Input text:\n" + tweet_text +
            "\n\n### Additional information:\n" + evidence +
            "\n\n### Response:\n" + class_label
    )
    sample['prompt'] = full_prompt
    return sample


def generate_test_prompt(sample):
    tweet_text = sample["tweet_text"]
    evidence = sample["evidence"]

    full_prompt = (
            "### Instruction:\n" + _SYSTEM_MESSAGE +
            # [FIX 2] Was "Input tweet" — now "Input text" to match training prompt.
            "\n\n### Input text:\n" + tweet_text +
            "\n\n### Additional information:\n" + evidence +
            "\n\n### Response:"
    )
    sample['prompt'] = full_prompt
    return sample


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_response(sample, model, tokenizer, j):
    prompt = sample['prompt']
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = {k: v.to(device) for k, v in encoded_input.items()}

    # [FIX 3] Add torch.no_grad() to disable gradient computation during
    # inference — reduces memory usage and matches reference code.
    with torch.no_grad():
        output = model.generate(
            **model_inputs,
            max_new_tokens=3,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded_output.replace(prompt, "").strip()
    sample["response" + str(j)] = response

    if 'No' in response:
        sample["prediction" + str(j)] = 0
    elif 'Yes' in response:
        sample["prediction" + str(j)] = 1
    else:
        print("Unexpected response: ", response)
        exit(0)
    return sample


def getMajority(sample, iterations):
    predictions = [sample["prediction" + str(i)] for i in range(iterations)]
    counter = Counter(predictions)
    majority, _ = counter.most_common()[0]
    sample['prediction'] = majority
    return sample


def getConsistency(predictions):
    transpose = list(zip(*predictions))
    consistent_count = sum(
        1 for sub_list in transpose if all(sub_list[0] == e for e in sub_list)
    )
    return consistent_count / len(predictions)


# ── File utilities ────────────────────────────────────────────────────────────

@contextmanager
def file_lock(filename):
    lockfile = f"{filename}.lock"
    while True:
        try:
            fd = open(lockfile, 'x')
            break
        except FileExistsError:
            time.sleep(0.1)
    try:
        yield
    finally:
        fd.close()
        try:
            os.remove(lockfile)
        except OSError:
            pass


def write_to_csv(data, csv_filename, column_names):
    with file_lock(csv_filename):
        file_exists = os.path.exists(csv_filename)
        mode = 'a' if file_exists else 'w'
        with open(csv_filename, mode, newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(column_names)
            writer.writerow(data)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a fine-tuned language model with different hyperparameters.'
    )
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=str, default='constant',
                        choices=['constant', 'linear', 'cosine', 'cosine_with_restarts'])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--output_dir', type=str,
                        default='/gpfs/scratch/acw760/taslp/ckpt/')
    parser.add_argument('--prefix', type=str, default='CT22_claim/policlaim_context')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--stability_test', action='store_true')
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def process_evidence_field(df, batch_size=1000):
    df = df.copy()
    for col in ['evidence']:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataframe")
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            df.loc[batch.index, col] = batch[col].fillna('').astype(str).replace('nan', '')
            if i % (batch_size * 10) == 0:
                clear_memory()
    return df


def load_and_process_data(prefix, chunk_size=1000):
    def load_csv_in_chunks(file_path):
        chunks = pd.read_csv(
            file_path,
            on_bad_lines='skip',
            dtype={"tweet_id": str},
            chunksize=chunk_size,
        )
        return pd.concat(list(chunks), ignore_index=True)

    print("Loading data from policlaim_context files...")

    data_files = {
        'train_full': f"data/evidence/{prefix}_train.csv",
        'test': f"data/evidence/{prefix}_test.csv",
    }

    datasets = {}
    for name, file_path in data_files.items():
        print(f"Processing {name}({file_path}) dataset...")
        data = load_csv_in_chunks(file_path)
        print(f"Initial {name} dataset size:", data.shape)

        processed_chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size].copy()
            chunk = process_evidence_field(chunk)
            processed_chunks.append(chunk)
            if i % (chunk_size * 5) == 0:
                clear_memory()

        data = pd.concat(processed_chunks, ignore_index=True)
        datasets[name] = data
        print(f"Processed {name} dataset size:", data.shape)
        clear_memory()

    # Split train_full into train and dev (90%-10% split)
    train_full = datasets['train_full']
    print(f"Splitting training data: {len(train_full)} samples")

    if 'class_label' in train_full.columns:
        train_data, dev_data = train_test_split(
            train_full, test_size=0.1, random_state=42,
            stratify=train_full['class_label'],
        )
    else:
        train_data, dev_data = train_test_split(
            train_full, test_size=0.1, random_state=42,
        )

    datasets['train'] = train_data.reset_index(drop=True)
    datasets['validation'] = dev_data.reset_index(drop=True)

    print(f"Train dataset size: {len(datasets['train'])}")
    print(f"Validation dataset size: {len(datasets['validation'])}")
    print(f"Test dataset size: {len(datasets['test'])}")

    del datasets['train_full']
    clear_memory()

    return datasets


def prepare_datasets(datasets, tokenizer, chunk_size=1000):
    prepared_datasets = {}

    for name in ['train', 'validation']:
        print(f"Preparing {name} dataset...")
        chunks = []
        for i in range(0, len(datasets[name]), chunk_size):
            chunk = datasets[name].iloc[i:i + chunk_size].copy()
            chunk = chunk.apply(generate_prompt, axis=1)
            chunks.append(chunk)
            if i % (chunk_size * 5) == 0:
                clear_memory()

        dataset = Dataset.from_pandas(pd.concat(chunks, ignore_index=True))
        dataset = dataset.map(
            lambda samples: tokenizer(
                samples["prompt"],
                padding=True,
                truncation=True,
                max_length=512,
            ),
            batched=True,
            batch_size=32,
        )
        prepared_datasets[name] = dataset
        clear_memory()

    # Process test dataset
    print("Preparing test dataset...")
    test_chunks = []
    for i in range(0, len(datasets['test']), chunk_size):
        chunk = datasets['test'].iloc[i:i + chunk_size].copy()
        chunk = chunk.apply(generate_test_prompt, axis=1)
        test_chunks.append(chunk)
        if i % (chunk_size * 5) == 0:
            clear_memory()

    prepared_datasets['test'] = pd.concat(test_chunks, ignore_index=True)

    return prepared_datasets


def setup_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir='/gpfs/scratch/acw760/hf_cache',
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def process_all_data(model_id, prefix, chunk_size=1000):
    print('Processing policlaim_context data...')
    tokenizer = setup_tokenizer(model_id)

    with track_memory():
        datasets = load_and_process_data(prefix, chunk_size)

    with track_memory():
        prepared_datasets = prepare_datasets(datasets, tokenizer, chunk_size)

    for name, dataset in prepared_datasets.items():
        if name in ['train', 'validation']:
            print(f"{name.capitalize()} dataset size:", dataset.shape)
        else:
            print(f"{name.capitalize()} dataset size:", len(dataset))

    return prepared_datasets, tokenizer


# [FIX 4] Removed unused get_peft_config() function — its logic was
# duplicated inside setup_training() and the two copies had diverged
# (e.g. different target_modules for Mistral). Keeping only
# setup_training() as the single source of truth.


# ── Training setup ────────────────────────────────────────────────────────────

def setup_training(args, model_id):
    # ------------------------------------------------------------------ #
    # [FIX 5] Use bfloat16 instead of float16 throughout.                #
    # Transformers 5 + bitsandbytes 0.49: from_pretrained() now respects #
    # the model's own config.json dtype. Modern LLMs ship with           #
    # torch_dtype=bfloat16 in config, so non-quantised layers land in    #
    # bf16 regardless. PyTorch's CUDA GradScaler (fp16=True) does NOT    #
    # support bf16 tensors → crash.                                      #
    # Fix: bf16=True in training args (no GradScaler), bfloat16 compute. #
    # ------------------------------------------------------------------ #
    compute_dtype = torch.bfloat16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model_name = model_id.lower()
    base_peft_config = {
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    # [FIX 6] Removed fan_in_fan_out=True from all LoRA configs.
    # PEFT 0.14+ removed this parameter for non-Conv1D layers — passing
    # it causes a warning or error on newer PEFT versions.
    if "mistral" in model_name:
        peft_config = LoraConfig(**base_peft_config, r=48,
                                 target_modules=["q_proj", "v_proj"])
    elif "llama" in model_name:
        peft_config = LoraConfig(**base_peft_config, r=64,
                                 target_modules=["q_proj", "v_proj", "o_proj"])
    elif "gemma" in model_name:
        peft_config = LoraConfig(**base_peft_config, r=64,
                                 target_modules=["q_proj", "v_proj"])
    else:
        peft_config = LoraConfig(**base_peft_config, r=32,
                                 target_modules=["q_proj", "v_proj", "o_proj"])

    if "llama-3" in model_name:
        learning_rate = 1e-5
    elif "gemma" in model_name:
        learning_rate = 3e-5
    elif "mistral" in model_name:
        learning_rate = 1e-4
    else:
        learning_rate = args.learning_rate

    batch_size = 16 if "instruct" in model_name else 8

    optim = "paged_adamw_8bit" if "mistral" in model_name else "adamw_torch"

    # [FIX 7] Use SFTConfig instead of plain TrainingArguments.
    # SFTConfig (TRL 0.15+) is the preferred way to pass SFT-specific
    # args (dataset_text_field, max_length, packing). Using plain
    # TrainingArguments generates a deprecation warning and requires
    # passing these to SFTTrainer directly.
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, args.experiment_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=2 if batch_size >= 16 else 4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25,
        optim=optim,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=300,
        learning_rate=learning_rate,
        # [FIX 5 cont.] bf16=True + fp16=False
        bf16=True,
        fp16=False,
        weight_decay=0.001,
        max_grad_norm=1.0,
        warmup_ratio=0.1 if "llama" in model_name else 0.05,
        group_by_length=True,
        lr_scheduler_type="cosine_with_restarts",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["none"],
        ddp_find_unused_parameters=False,
        # [FIX 7 cont.] SFTConfig-specific fields
        dataset_text_field="prompt",
        max_length=512,
        packing=False,
    )

    return quant_config, peft_config, training_args


# ── Train & evaluate ──────────────────────────────────────────────────────────

def train_and_evaluate(experiment_name, model, trainer, test_data, i, tokenizer):
    """Run training and evaluation with memory optimization for test dataset only"""
    with track_memory():
        start_time = datetime.now()
        print(f"Training iteration {i} started at: {start_time.strftime('%Y-%m-%d-%H-%M')}")

        trainer.train()

        end_time = datetime.now()
        print(f"Training ended at: {end_time.strftime('%Y-%m-%d-%H-%M')}")
        time_taken = (end_time - start_time).total_seconds() / 3600

        print("Evaluating on test dataset...")

        model.eval()
        predictions = []

        all_preds = {k: [] for k in range(5)}

        batch_size = 32
        for j in range(0, len(test_data), batch_size):
            batch = test_data.iloc[j:j + batch_size].copy()
            batch_predictions = []

            for run_idx in range(5):
                batch = batch.apply(
                    lambda sample: generate_response(sample, model, tokenizer, run_idx),
                    axis=1,
                )
                preds = batch[f"prediction{run_idx}"].tolist()
                batch_predictions.append(preds)
                all_preds[run_idx].extend(preds)

            predictions.extend(zip(*batch_predictions))

            if j % (batch_size * 10) == 0:
                clear_memory()

        majority_predictions = [Counter(pred).most_common(1)[0][0] for pred in predictions]
        actual = test_data['class_label']

        test_data_copy = test_data.copy()
        for run_idx in range(5):
            test_data_copy[f"prediction{run_idx}"] = all_preds[run_idx]
        test_data_copy['majority_prediction'] = majority_predictions

        output_filename = (
            f"prediction_results/taslp_policlaim/"
            f"{experiment_name}_test_{i}.csv"
        )
        test_data_copy.to_csv(output_filename, index=False)
        print(f"Predictions saved to {output_filename}")

        # ── Compute metrics ───────────────────────────────────────────
        # [FIX 8] Added macro-averaged precision, recall, F1 to match
        # reference code. Binary (pos_label=1) metrics kept for backward
        # compatibility.
        metrics = {
            'accuracy': accuracy_score(actual, majority_predictions),
            # Binary (positive-class) metrics
            'precision': precision_score(actual, majority_predictions, pos_label=1),
            'recall': recall_score(actual, majority_predictions, pos_label=1),
            'f1': f1_score(actual, majority_predictions, pos_label=1),
            # Macro-averaged metrics
            'macro_precision': precision_score(actual, majority_predictions, average='macro'),
            'macro_recall': recall_score(actual, majority_predictions, average='macro'),
            'macro_f1': f1_score(actual, majority_predictions, average='macro'),
            'consistency': getConsistency(list(zip(*predictions))),
            'time': time_taken,
        }

        print("Test evaluation metrics:", metrics)

        clear_memory()
        return metrics


# ── Metrics summary ───────────────────────────────────────────────────────────

def calculate_and_save_metrics(metrics_list, args):
    try:
        running_times = [m['time'] for m in metrics_list]
        accuracy_t = [m['accuracy'] for m in metrics_list]
        precision_t = [m['precision'] for m in metrics_list]
        recall_t = [m['recall'] for m in metrics_list]
        f1_score_t = [m['f1'] for m in metrics_list]
        macro_precision_t = [m['macro_precision'] for m in metrics_list]
        macro_recall_t = [m['macro_recall'] for m in metrics_list]
        macro_f1_t = [m['macro_f1'] for m in metrics_list]
        consistency_t = [m['consistency'] for m in metrics_list]

        def _safe_stdev(lst):
            return statistics.stdev(lst) if len(lst) > 1 else 0

        mean_running_time = statistics.mean(running_times)
        metrics_summary = {
            'accuracy': (statistics.mean(accuracy_t), _safe_stdev(accuracy_t)),
            'precision': (statistics.mean(precision_t), _safe_stdev(precision_t)),
            'recall': (statistics.mean(recall_t), _safe_stdev(recall_t)),
            'f1': (statistics.mean(f1_score_t), _safe_stdev(f1_score_t)),
            'macro_precision': (statistics.mean(macro_precision_t), _safe_stdev(macro_precision_t)),
            'macro_recall': (statistics.mean(macro_recall_t), _safe_stdev(macro_recall_t)),
            'macro_f1': (statistics.mean(macro_f1_t), _safe_stdev(macro_f1_t)),
            'consistency': (statistics.mean(consistency_t), _safe_stdev(consistency_t)),
        }

        print(f"\nExperiment: {args.experiment_name}")
        print(f"Model: {args.model_id}")
        print(f"Dataset: policlaim_context_test")
        print(f"Runtime: {mean_running_time:.2f} hours")
        print("\nIteration test (binary, pos_label=1):")
        print(f"Accuracy list: {accuracy_t}")
        print(f"Precision list: {precision_t}")
        print(f"Recall list: {recall_t}")
        print(f"F1-score list: {f1_score_t}")
        print("\nIteration test (macro-averaged):")
        print(f"Macro Precision list: {macro_precision_t}")
        print(f"Macro Recall list: {macro_recall_t}")
        print(f"Macro F1-score list: {macro_f1_t}")
        print("****************************")
        print("\nTest Summary:")
        metric_string = ", ".join([
            f"{metric.title()}: {mean:.4f} ± {std:.4f}"
            for metric, (mean, std) in metrics_summary.items()
        ])
        print(metric_string)

        # [FIX 8 cont.] CSV columns updated with macro metrics
        evaluation_file = "../evaluation_results/taslp_policlaim_tc_evaluation.csv"
        columns = [
            'Experiment Name', 'Model', 'Dataset',
            'Accuracy-T', 'Accuracy-STD-T',
            'Precision-T', 'Precision-STD-T',
            'Recall-T', 'Recall-STD-T',
            'F1-T', 'F1-STD-T',
            'Macro-Precision-T', 'Macro-Precision-STD-T',
            'Macro-Recall-T', 'Macro-Recall-STD-T',
            'Macro-F1-T', 'Macro-F1-STD-T',
            'Consistency-T', 'Consistency-STD-T',
            'Timestamp',
        ]

        row_data = [
            args.experiment_name,
            args.model_id,
            'policlaim_context_test',
            metrics_summary['accuracy'][0], metrics_summary['accuracy'][1],
            metrics_summary['precision'][0], metrics_summary['precision'][1],
            metrics_summary['recall'][0], metrics_summary['recall'][1],
            metrics_summary['f1'][0], metrics_summary['f1'][1],
            metrics_summary['macro_precision'][0], metrics_summary['macro_precision'][1],
            metrics_summary['macro_recall'][0], metrics_summary['macro_recall'][1],
            metrics_summary['macro_f1'][0], metrics_summary['macro_f1'][1],
            metrics_summary['consistency'][0], metrics_summary['consistency'][1],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ]

        write_to_csv(row_data, evaluation_file, columns)
        print(f"\nResults saved to {evaluation_file}")

    except Exception as e:
        print(f"Error calculating or saving metrics: {str(e)}")
        raise


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    datasets, tokenizer = process_all_data(args.model_id, args.prefix)

    train_data = datasets['train']
    validation_data = datasets['validation']
    test_data = datasets['test']

    quant_config, peft_config, training_args = setup_training(args, args.model_id)

    test_metrics_list = []

    num_runs = args.num_runs if args.stability_test else 1
    random_seeds = ([42, 123, 456, 789, 1024][:num_runs]
                    if args.stability_test else [42] * num_runs)

    if args.stability_test:
        print(f"Running stability test with {num_runs} seeds: {random_seeds}")

    for i in range(args.num_runs):
        clear_memory()
        set_random_seed(random_seeds[i])

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map='auto',
            quantization_config=quant_config,
            cache_dir='/gpfs/scratch/acw760/hf_cache',
            # [FIX 5 cont.] Use bfloat16 to match training config
            torch_dtype=torch.bfloat16,
        )
        model = optimize_model_memory(model)

        # [FIX 9] Use processing_class= instead of deprecated tokenizer=
        # kwarg. TRL 0.13+ renamed it; the old name still works via a
        # deprecation shim but processing_class is forward-compatible.
        trainer = SFTTrainer(
            model=model,
            args=training_args,  # SFTConfig (subclass of TrainingArguments)
            train_dataset=train_data,
            eval_dataset=validation_data,
            peft_config=peft_config,
            processing_class=tokenizer,
        )

        metrics = train_and_evaluate(
            args.experiment_name, model, trainer,
            test_data, i, tokenizer,
        )
        test_metrics_list.append(metrics)

        del model, trainer
        clear_memory()

    print("\nResults for test dataset:")
    calculate_and_save_metrics(test_metrics_list, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)