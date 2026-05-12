import argparse

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForSeq2Seq
import torch
from peft import LoraConfig
from trl import SFTTrainer
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    # torch.mps.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

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
        # Disable gradient checkpointing if not needed
        model.config.use_cache = False
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model

def generate_prompt(sample):
    system_message = """Determine if the input tweet contains verifiable claims.
The tweet contains verifiable claims if it makes specific factual statements that can be checked against evidence.

If the tweet contains claims that can be verified, respond "Yes". Otherwise, respond "No".
Note: When in doubt, choose "Yes". In the end, respond only with 'Yes' for verifiable claims or 'No' for non-verifiable claims."""

    tweet_text = sample["tweet_text"]
    evidence = sample["evidence"]
    class_label = 'Yes'
    if sample["class_label"] == 0:
        class_label = 'No'

    full_prompt = ""
    full_prompt += "### Instruction:"
    full_prompt += "\n" + system_message
    full_prompt += "\n\n### Input tweet:"
    full_prompt += "\n" + tweet_text
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + class_label

    sample['prompt'] = full_prompt
    return sample


def generate_test_prompt(sample):
    system_message = """Determine if the input tweet contains verifiable claims.
The tweet contains verifiable claims if it makes specific factual statements that can be checked against evidence.

If the tweet contains claims that can be verified, respond "Yes". Otherwise, respond "No".
Note: When in doubt, choose "Yes". In the end, respond only with 'Yes' for verifiable claims or 'No' for non-verifiable claims."""

    tweet_text = sample["tweet_text"]
    evidence = sample["evidence"]

    full_prompt = ""
    full_prompt += "### Instruction:"
    full_prompt += "\n" + system_message
    full_prompt += "\n\n### Input tweet:"
    full_prompt += "\n" + tweet_text
    full_prompt += "\n\n### Response:"

    sample['prompt'] = full_prompt
    return sample


def generate_response(sample, model, tokenizer, j):
    prompt = sample['prompt']
    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    # model_inputs = encoded_input.to('cuda')
    model_inputs = {k: v.to(device) for k, v in encoded_input.items()}
    # model_inputs = {k: v.to(device).to(model.dtype) for k, v in encoded_input.items()}
    output = model.generate(**model_inputs, max_new_tokens=3,
                            pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded_output.replace(prompt, "").strip()
    sample["response" + str(j)] = response

    if 'No' in response:
        sample["prediction" + str(j)] = 0
    elif 'Yes' in response:
        sample["prediction" + str(j)] = 1
    else:
        print("Error: ", response)
        exit(0)
    return sample


def getMajority(sample, iterations):
    predictions = []

    for i in range(iterations):
        predictions.append(sample["prediction" + str(i)])

    counter = Counter(predictions)
    majority, count = counter.most_common()[0]
    sample['prediction'] = majority
    return sample


def getConsistency(predictions):
    transpose = list(zip(*predictions))

    consistent_count = 0

    for sub_list in transpose:
        if all(sub_list[0] == element for element in sub_list):
            consistent_count += 1

    consistency = consistent_count / (len(predictions))
    return consistency


@contextmanager
def file_lock(filename):
    """File lock context manager."""
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
        except:
            pass


def write_to_csv(data, csv_filename, column_names):
    """Write data to a CSV file with file locking."""
    with file_lock(csv_filename):
        file_exists = os.path.exists(csv_filename)

        mode = 'a' if file_exists else 'w'
        with open(csv_filename, mode, newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(column_names)
            writer.writerow(data)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned language model with different hyperparameters.')
    parser.add_argument('--model_id', type=str, required=True, help='Path to the fine-tuned model directory')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    # parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    # parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    # parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio')
    # parser.add_argument('--max_grad_norm', type=float, default=0.3, help='Maximum gradient norm')
    parser.add_argument('--lr_scheduler', type=str, default='constant',
                        choices=['constant', 'linear', 'cosine', 'cosine_with_restarts'],
                        help='Learning rate scheduler type')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--output_dir', type=str, default='/gpfs/scratch/acw760/phd_afc/verifiable_only_tweet/ckpt/',
                        help='Output directory for checkpoints')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name for this experiment run')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs for stability testing')
    parser.add_argument('--stability_test', action='store_true', help='Run stability test with multiple seeds')


    return parser.parse_args()

def process_evidence_field(df, batch_size=1000):
    """Process dataframe in batches to reduce memory usage"""
    df = df.copy()
    required_columns = ['evidence']

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the dataframe")

    # Process in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        for col in required_columns:
            df.loc[batch.index, col] = batch[col].fillna('').astype(str).replace('nan', '')

        if i % (batch_size * 10) == 0:
            clear_memory()

    return df


def load_and_process_data(prefix, chunk_size=1000):
    """
    Load and process data in chunks to minimize memory usage
    """

    def load_csv_in_chunks(file_path):
        chunks = pd.read_csv(
            file_path,
            on_bad_lines='skip',
            dtype={"tweet_id": str},
            chunksize=chunk_size
        )
        return pd.concat([chunk for chunk in chunks], ignore_index=True)

    print(f"Loading data from prefix: {prefix}")

    # Load data files in chunks - UPDATED to include test_gold.csv
    data_files = {
        'train': f"../data/evidence/{prefix}_train.csv",
        'validation': f"../data/evidence/{prefix}_dev.csv",
        'dev_test': f"../data/evidence/{prefix}_dev_test.csv",
        'test': f"../data/evidence/{prefix}_test_gold.csv"
    }

    datasets = {}
    for name, file_path in data_files.items():
        print(f"Processing {name} dataset...")

        # Load and process in chunks
        data = load_csv_in_chunks(file_path)
        print(f"Initial {name} dataset size:", data.shape)

        # Process evidence field in chunks
        processed_chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size].copy()
            chunk = process_evidence_field(chunk)
            processed_chunks.append(chunk)

            # Clear memory after processing each chunk
            if i % (chunk_size * 5) == 0:
                clear_memory()

        data = pd.concat(processed_chunks, ignore_index=True)
        datasets[name] = data

        print(f"Processed {name} dataset size:", data.shape)
        clear_memory()

    return datasets


def prepare_datasets(datasets, tokenizer, chunk_size=1000):
    """
    Prepare datasets with prompts and tokenization in a memory-efficient way
    """
    prepared_datasets = {}

    # Process train and validation data
    for name in ['train', 'validation']:
        print(f"Preparing {name} dataset...")

        # Generate prompts in chunks
        chunks = []
        for i in range(0, len(datasets[name]), chunk_size):
            chunk = datasets[name].iloc[i:i + chunk_size].copy()
            # Apply generate_prompt to chunk
            chunk = chunk.apply(generate_prompt, axis=1)
            chunks.append(chunk)

            if i % (chunk_size * 5) == 0:
                clear_memory()

        # Combine chunks and convert to Dataset
        dataset = Dataset.from_pandas(pd.concat(chunks, ignore_index=True))

        # Tokenize in batches
        dataset = dataset.map(
            lambda samples: tokenizer(
                samples["prompt"],
                padding=True,
                truncation=True,
                max_length=512  # Adjust based on your needs
            ),
            batched=True,
            batch_size=32  # Smaller batch size for tokenization
        )

        prepared_datasets[name] = dataset
        clear_memory()

    # Process both test datasets separately since they use a different prompt generator
    for name in ['dev_test', 'test']:
        print(f"Preparing {name} dataset...")
        test_chunks = []
        for i in range(0, len(datasets[name]), chunk_size):
            chunk = datasets[name].iloc[i:i + chunk_size].copy()
            chunk = chunk.apply(generate_test_prompt, axis=1)
            test_chunks.append(chunk)

            if i % (chunk_size * 5) == 0:
                clear_memory()

        prepared_datasets[name] = pd.concat(test_chunks, ignore_index=True)

    return prepared_datasets


def setup_tokenizer(model_id):
    """
    Setup tokenizer with proper configuration
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir='/gpfs/scratch/acw760/hf_cache',
        use_fast=True  # Use fast tokenizer for better performance
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def process_all_data(model_id, prefix, chunk_size=1000):
    """
    Main function to process all data with memory optimization
    """
    print(f'Processing data with prefix: {prefix}')

    # Setup tokenizer first
    tokenizer = setup_tokenizer(model_id)

    # Load and process raw data
    with track_memory():
        datasets = load_and_process_data(prefix, chunk_size)

    # Prepare datasets with prompts and tokenization
    with track_memory():
        prepared_datasets = prepare_datasets(datasets, tokenizer, chunk_size)

    # Print final dataset sizes
    for name, dataset in prepared_datasets.items():
        if name in ['train', 'validation']:
            print(f"{name.capitalize()} dataset size:", dataset.shape)

    return prepared_datasets, tokenizer


def setup_training(args, model_id):
    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype
    )

    model_name = model_id.lower()
    base_peft_config = {
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }

    if "mistral" in model_name:
        peft_config = LoraConfig(
            **base_peft_config,
            r=48,
            target_modules=["q_proj", "v_proj"],
            fan_in_fan_out=True
        )
    elif "llama" in model_name:
        peft_config = LoraConfig(
            **base_peft_config,
            r=64,
            target_modules=["q_proj", "v_proj", "o_proj"],
            fan_in_fan_out=True
        )
    elif "gemma" in model_name:
        peft_config = LoraConfig(
            **base_peft_config,
            r=64,
            target_modules=["q_proj", "v_proj"]
        )
    else:
        peft_config = LoraConfig(
            **base_peft_config,
            r=32,
            target_modules=["q_proj", "v_proj", "o_proj"]
        )

    if "llama-3" in model_name:
        learning_rate = 1e-5
    elif "gemma" in model_name:
        learning_rate = 3e-5
    elif "mistral" in model_name:
        learning_rate = 1e-4
    else:
        learning_rate = args.learning_rate

    batch_size = 16 if "instruct" in model_name else 8

    if "mistral" in model_name:
        optim = "paged_adamw_8bit"
    else:
        optim = "adamw_torch"

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.experiment_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=2 if batch_size >= 16 else 4,
        gradient_checkpointing=True,
        logging_steps=25,
        optim=optim,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=300,
        learning_rate=learning_rate,
        bf16=False,
        fp16=True,
        weight_decay=0.001,
        max_grad_norm=1.0,
        warmup_ratio=0.1 if "llama" in model_name else 0.05,
        group_by_length=True,
        lr_scheduler_type="cosine_with_restarts",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=['none'],
        ddp_find_unused_parameters=False,
    )

    return quant_config, peft_config, training_args


def train_and_evaluate(experiment_name, model, trainer, dev_test_data, test_data, i, tokenizer):
    """Run training and evaluation with memory optimization for both test datasets"""
    with track_memory():
        start_time = datetime.now()
        print(f"Training iteration {i} started at: {start_time.strftime('%Y-%m-%d-%H-%M')}")

        trainer.train()

        end_time = datetime.now()
        print(f"Training ended at: {end_time.strftime('%Y-%m-%d-%H-%M')}")
        time_taken = (end_time - start_time).total_seconds() / 3600

        metrics = {}

        # Evaluate on both test datasets
        for dataset_name, dataset in [("dev_test", dev_test_data), ("test", test_data)]:
            print(f"Evaluating on {dataset_name} dataset...")

            # Evaluation
            model.eval()
            predictions = []

            # Lists to store individual predictions for each sample
            all_pred0 = []
            all_pred1 = []
            all_pred2 = []
            all_pred3 = []
            all_pred4 = []

            # Process test data in smaller batches
            batch_size = 32
            for j in range(0, len(dataset), batch_size):
                batch = dataset.iloc[j:j + batch_size]
                with torch.no_grad():
                    batch_predictions = []
                    for _ in range(5):  # 5 predictions per sample
                        batch = batch.apply(lambda sample: generate_response(sample, model, tokenizer, _), axis=1)
                        batch_predictions.append(batch['prediction' + str(_)].tolist())

                        # Collect predictions for this batch to add to respective lists
                        if _ == 0:
                            all_pred0.extend(batch['prediction0'].tolist())
                        elif _ == 1:
                            all_pred1.extend(batch['prediction1'].tolist())
                        elif _ == 2:
                            all_pred2.extend(batch['prediction2'].tolist())
                        elif _ == 3:
                            all_pred3.extend(batch['prediction3'].tolist())
                        elif _ == 4:
                            all_pred4.extend(batch['prediction4'].tolist())

                predictions.extend(zip(*batch_predictions))

                if j % (batch_size * 10) == 0:
                    clear_memory()

            # Calculate metrics
            majority_predictions = [Counter(pred).most_common(1)[0][0] for pred in predictions]
            actual = dataset['class_label']

            # Add the 5 individual predictions and majority prediction to dataset
            dataset_copy = dataset.copy()
            dataset_copy['prediction0'] = all_pred0
            dataset_copy['prediction1'] = all_pred1
            dataset_copy['prediction2'] = all_pred2
            dataset_copy['prediction3'] = all_pred3
            dataset_copy['prediction4'] = all_pred4
            dataset_copy['majority_prediction'] = majority_predictions

            # Save predictions to CSV
            output_filename = f"prediction_results/only_tweet/{experiment_name}_{dataset_name}_{i}.csv"
            dataset_copy.to_csv(output_filename, index=False)
            print(f"Predictions saved to {output_filename}")

            # Store metrics for this dataset
            metrics[dataset_name] = {
                'accuracy': accuracy_score(actual, majority_predictions),
                'precision': precision_score(actual, majority_predictions, pos_label=1),
                'recall': recall_score(actual, majority_predictions, pos_label=1),
                'f1': f1_score(actual, majority_predictions, pos_label=1),
                'consistency': getConsistency(list(zip(*predictions))),
                'time': time_taken
            }

            print(f"{dataset_name} evaluation metrics:", metrics[dataset_name])

        clear_memory()
        return metrics


def calculate_and_save_metrics(metrics_list, args, dataset_name):
    try:
        # Extract individual metric lists
        running_times = [m['time'] for m in metrics_list]
        accuracy_t = [m['accuracy'] for m in metrics_list]
        precision_t = [m['precision'] for m in metrics_list]
        recall_t = [m['recall'] for m in metrics_list]
        f1_score_t = [m['f1'] for m in metrics_list]
        consistency_t = [m['consistency'] for m in metrics_list]

        # Calculate means and standard deviations
        mean_running_time = statistics.mean(running_times)
        metrics_summary = {
            'accuracy': (statistics.mean(accuracy_t), statistics.stdev(accuracy_t)),
            'precision': (statistics.mean(precision_t), statistics.stdev(precision_t)),
            'recall': (statistics.mean(recall_t), statistics.stdev(recall_t)),
            'f1': (statistics.mean(f1_score_t), statistics.stdev(f1_score_t)),
            'consistency': (statistics.mean(consistency_t), statistics.stdev(consistency_t))
        }

        # Print detailed results
        print(f"\nExperiment: {args.experiment_name}")
        print(f"Model: {args.model_id}")
        print(f"Dataset: {dataset_name}")
        print(f"Runtime: {mean_running_time:.2f} hours")

        # Print iteration details
        print("\nIteration test:")
        print(f"Accuracy list: {accuracy_t}")
        print(f"Precision list: {precision_t}")
        print(f"Recall list: {recall_t}")
        print(f"F1-score list: {f1_score_t}")
        print("****************************")

        # Print summary statistics
        print("\nTest Summary:")
        metric_string = ", ".join([
            f"{metric.title()}: {mean:.4f} ± {std:.4f}"
            for metric, (mean, std) in metrics_summary.items()
        ])
        print(metric_string)

        # Save results to CSV
        evaluation_file = "../evaluation_results/web4good_llm_t_evaluation_ct22.csv"
        columns = [
            'Experiment Name', 'Model', 'Dataset',
            'Accuracy-T', 'Accuracy-STD-T',
            'Precision-T', 'Precision-STD-T',
            'Recall-T', 'Recall-STD-T',
            'F1-T', 'F1-STD-T',
            'Consistency-T', 'Consistency-STD-T',
            'Timestamp'
        ]

        # Prepare row data
        row_data = [
            args.experiment_name,
            args.model_id,
            dataset_name,
            metrics_summary['accuracy'][0],
            metrics_summary['accuracy'][1],
            metrics_summary['precision'][0],
            metrics_summary['precision'][1],
            metrics_summary['recall'][0],
            metrics_summary['recall'][1],
            metrics_summary['f1'][0],
            metrics_summary['f1'][1],
            metrics_summary['consistency'][0],
            metrics_summary['consistency'][1],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]

        write_to_csv(row_data, evaluation_file, columns)
        print(f"\nResults saved to {evaluation_file}")

    except Exception as e:
        print(f"Error calculating or saving metrics: {str(e)}")
        raise


def main(args):
    # Load and process data
    # prefix = 'CT22_claim/CT22_top3evidence_claim'
    prefix = "CT22_claim/CT22_gpt4o_context_web_new"
    datasets, tokenizer = process_all_data(args.model_id, prefix)

    train_data = datasets['train']
    validation_data = datasets['validation']
    dev_test_data = datasets['dev_test']
    test_data = datasets['test']

    # Setup training
    quant_config, peft_config, training_args = setup_training(args, args.model_id)

    dev_test_metrics_list = []
    test_metrics_list = []

    num_runs = args.num_runs if args.stability_test else 1
    if args.stability_test:
        random_seeds = [42, 123, 456, 789, 1024][:num_runs]
        print(f"Running stability test with {num_runs} different random seeds: {random_seeds}")
    else:
        random_seeds = [42] * num_runs

    for i in range(num_runs):
        clear_memory()
        current_seed = random_seeds[i]
        set_random_seed(current_seed)

        # Initialize model with memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map='auto',
            quantization_config=quant_config,
            cache_dir='/gpfs/scratch/acw760/hf_cache',
            torch_dtype=torch.float16
        )
        model = optimize_model_memory(model)

        trainer = SFTTrainer(
            model=model,
            args=training_args,  # Use TrainingArguments
            train_dataset=train_data,
            eval_dataset=validation_data,
            peft_config=peft_config,
        )

        # Train and evaluate
        metrics = train_and_evaluate(args.experiment_name, model, trainer, dev_test_data, test_data, i, tokenizer)
        dev_test_metrics_list.append(metrics['dev_test'])
        test_metrics_list.append(metrics['test'])

        # Clean up
        del model
        del trainer
        clear_memory()

    # Calculate and save final metrics for both test datasets
    print("\nResults for dev_test dataset:")
    calculate_and_save_metrics(dev_test_metrics_list, args, prefix + "_dev_test")

    print("\nResults for test_gold dataset:")
    calculate_and_save_metrics(test_metrics_list, args, prefix + "_test_gold")


if __name__ == "__main__":
    args = parse_args()
    main(args)