# coding=utf-8
import sys
import argparse
import csv
import gc
import os
import random
import re
import statistics
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime

import emoji
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wandb import WandbLogger

'''
model: roberta
best_model_metrics: f1
attention mechanism: self-attention and cross-attention
'''

gc.collect()


class TweetEvidenceDataset(Dataset):
    def __init__(self, tweets, evidences, labels, tokenizer, max_length=200):
        self.tweets = tweets
        self.evidences = evidences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = str(self.tweets[idx])
        evidence = str(self.evidences[idx]) if self.evidences[idx] else ""

        tweet_encoding = self.tokenizer(
            tweet,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )

        evidence_encoding = self.tokenizer(
            evidence,
            add_special_tokens=True,
            max_length=200,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )

        return {
            'tweet_input_ids': torch.tensor(tweet_encoding['input_ids'], dtype=torch.long),
            'tweet_attention_mask': torch.tensor(tweet_encoding['attention_mask'], dtype=torch.long),
            'evidence_input_ids': torch.tensor(evidence_encoding['input_ids'], dtype=torch.long),
            'evidence_attention_mask': torch.tensor(evidence_encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class EvidenceEnhancedBERTOnlyCross(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.tweet_bert = AutoModel.from_pretrained(bert_model_name)
        self.evidence_bert = AutoModel.from_pretrained(bert_model_name)

        self.dropout = nn.Dropout(dropout_rate)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.tweet_bert.config.hidden_size,
            num_heads=8,
            batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.tweet_bert.config.hidden_size * 2, self.tweet_bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.classifier = nn.Linear(self.tweet_bert.config.hidden_size, num_classes)

    def forward(self, tweet_input_ids, tweet_attention_mask,
                evidence_input_ids, evidence_attention_mask):
        tweet_outputs = self.tweet_bert(
            input_ids=tweet_input_ids,
            attention_mask=tweet_attention_mask
        )
        tweet_hidden = tweet_outputs.last_hidden_state
        tweet_pooled = tweet_hidden[:, 0, :]

        evidence_outputs = self.evidence_bert(
            input_ids=evidence_input_ids,
            attention_mask=evidence_attention_mask
        )
        evidence_hidden = evidence_outputs.last_hidden_state
        evidence_pooled = evidence_hidden[:, 0, :]

        tweet_query = tweet_pooled.unsqueeze(1)

        key_padding_mask = (evidence_attention_mask == 0)

        evidence_context, _ = self.cross_attention(
            query=tweet_query,
            key=evidence_hidden,
            value=evidence_hidden,
            key_padding_mask=key_padding_mask
        )
        evidence_context = evidence_context.squeeze(1)

        combined_features = torch.cat([tweet_pooled, evidence_context], dim=1)
        fused_features = self.fusion(combined_features)

        logits = self.classifier(fused_features)

        return logits


def clean_tweet(tweet):
    tweet = emoji.demojize(tweet)
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'[@#]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub(r'&amp;', '&', tweet)
    return tweet


def process_evidence_field(df):
    df = df.copy()

    if 'evidence' not in df.columns:
        raise ValueError("Column 'evidence' not found in the dataframe")

    df['evidence'] = df['evidence'].fillna('')
    df['evidence'] = df['evidence'].astype(str)
    df['evidence'] = df['evidence'].replace('nan', '')

    return df


def getMajority(sample, iterations):
    predictions = [sample["prediction" + str(i)] for i in range(iterations)]
    counter = Counter(predictions)
    majority, _ = counter.most_common()[0]
    sample['prediction'] = majority
    return sample


# [FIX 1] Added macro-averaged metrics to compute_metrics.
# Binary (pos_label=1) metrics kept for backward compatibility.
def compute_metrics(predictions, labels):
    return {
        'accuracy': accuracy_score(labels, predictions),
        # Binary (positive-class) metrics
        'precision': precision_score(labels, predictions, average='binary', zero_division=0, pos_label=1),
        'recall': recall_score(labels, predictions, average='binary', zero_division=0, pos_label=1),
        'f1': f1_score(labels, predictions, average='binary', zero_division=0, pos_label=1),
        # Macro-averaged metrics
        'macro_precision': precision_score(labels, predictions, average='macro', zero_division=0),
        'macro_recall': recall_score(labels, predictions, average='macro', zero_division=0),
        'macro_f1': f1_score(labels, predictions, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(labels, predictions)
    }


def train_model(model, train_loader, val_loader, warmup_ratio, criterion, optimizer,
                device, outputdir, iteration, global_step, wandbLogger,
                num_epochs=3, early_stopping_patience=3, gradient_accumulation_steps=2):
    model.to(device)

    best_f1_score = 0
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1_score': [],
        'epochs': []
    }

    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            tweet_input_ids = batch['tweet_input_ids'].to(device)
            tweet_attention_mask = batch['tweet_attention_mask'].to(device)
            evidence_input_ids = batch['evidence_input_ids'].to(device)
            evidence_attention_mask = batch['evidence_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                tweet_input_ids=tweet_input_ids,
                tweet_attention_mask=tweet_attention_mask,
                evidence_input_ids=evidence_input_ids,
                evidence_attention_mask=evidence_attention_mask
            )
            loss = criterion(outputs, labels)

            loss = loss / gradient_accumulation_steps
            loss.backward()

            total_train_loss += loss.item() * gradient_accumulation_steps

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        val_results = evaluate_model(model, val_loader, criterion, device)
        avg_val_loss = val_results['avg_val_loss']
        val_metrics = compute_metrics(val_results['predictions'], val_results['labels'])
        current_f1_score = val_metrics['f1']

        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_f1_score'].append(current_f1_score)
        training_history['epochs'].append(epoch + 1)

        epoch_metrics = {
            "current_epoch": epoch,
            "current_iteration": iteration,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_metrics['accuracy'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1": val_metrics['f1'],
            "val_macro_precision": val_metrics['macro_precision'],
            "val_macro_recall": val_metrics['macro_recall'],
            "val_macro_f1": val_metrics['macro_f1'],
        }
        wandbLogger.log_metrics(epoch_metrics, step=global_step, commit=True)
        global_step += 1

        print(f"Epoch {epoch + 1}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Metrics:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}"
              f", Precision: {val_metrics['precision']:.4f}"
              f", Recall: {val_metrics['recall']:.4f}"
              f", F1 Score: {current_f1_score:.4f}"
              f", Macro F1: {val_metrics['macro_f1']:.4f}"
              f", Validation Loss: {avg_val_loss:.4f}")
        print("  Confusion Matrix:")
        print(val_metrics['confusion_matrix'])
        print("-" * 50)

        best_model_path = outputdir + "best_model.pth"
        if current_f1_score > best_f1_score:
            print(f"F1 score increased from {best_f1_score:.4f} to {current_f1_score:.4f}")
            best_f1_score = current_f1_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_f1_score': current_f1_score,
                'training_history': training_history
            }, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"F1 score did not improve. Counter: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print("Best F1 score: {:.4f}".format(best_f1_score))
            break

    return training_history


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_val_loss = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            tweet_input_ids = batch['tweet_input_ids'].to(device)
            tweet_attention_mask = batch['tweet_attention_mask'].to(device)
            evidence_input_ids = batch['evidence_input_ids'].to(device)
            evidence_attention_mask = batch['evidence_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                tweet_input_ids=tweet_input_ids,
                tweet_attention_mask=tweet_attention_mask,
                evidence_input_ids=evidence_input_ids,
                evidence_attention_mask=evidence_attention_mask
            )
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()

            _, predictions = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'avg_val_loss': total_val_loss / len(data_loader)
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned language model with different hyperparameters.')
    parser.add_argument('--model_id', type=str, required=True, help='Path to the fine-tuned model directory')
    parser.add_argument('--learning_rate', type=float, default=1.3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.011, help='Warmup ratio')
    parser.add_argument('--dropout_rate', type=float, default=0.22, help='Dropout rate for the model')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs for stability testing')
    parser.add_argument('--stability_test', action='store_true', help='Run stability test with multiple seeds')
    parser.add_argument('--prefix', type=str, default='CT22_claim/CT22_gpt4o_generated_context',
                        help='Data path prefix')
    parser.add_argument('--output_dir', type=str, default='/gpfs/scratch/acw760/phd_afc/best_param/',
                        help='Output directory for checkpoints')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name for this experiment run')
    return parser.parse_args()


def load_data(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip', dtype={"tweet_id": str})
    df['tweet_text'] = df['tweet_text'].apply(clean_tweet)
    df = process_evidence_field(df)

    tweet_ids = df['tweet_id']
    tweets = df['tweet_text']
    evidences = df['evidence']
    labels = df['class_label']
    return tweet_ids, tweets, evidences, labels


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    config = {
        "model_name": args.model_id,
        "iterations": 5,
        "epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "warmup_ratio": args.warmup_ratio,
        "dropout_rate": args.dropout_rate,
        "early_stopping_patience": 2,
        "optimizer": "AdamW",
        "stability_test": args.stability_test,
        "num_runs": args.num_runs
    }

    experiment_name = args.experiment_name
    if args.stability_test:
        experiment_name = f"StabilityTest_{experiment_name}"

    wandbLogger = WandbLogger(
        project="AFC_CD_TASLP",
        name=experiment_name,
        config=config,
        tags=["tweet_context", "roberta", "best_model_f1", "cross",
              "stability_test" if args.stability_test else "single_run"],
        notes=f"Roberta with context {'(stability test)' if args.stability_test else ''}",
        group="BERT-based-experiments"
    )

    global_step = 0

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # [FIX 2] Removed dev_test data loading — evaluation on dev_test
    # is no longer performed.
    prefix = args.prefix
    train_ids, train_tweets, train_evidences, train_labels = load_data(f"../data/evidence/{prefix}_train.csv")
    val_ids, val_tweets, val_evidences, val_labels = load_data(f"../data/evidence/{prefix}_dev.csv")
    test_ids, test_tweets, test_evidences, test_labels = load_data(f"../data/evidence/{prefix}_test_gold.csv")

    train_dataset = TweetEvidenceDataset(train_tweets, train_evidences, train_labels, tokenizer)
    val_dataset = TweetEvidenceDataset(val_tweets, val_evidences, val_labels, tokenizer)
    test_dataset = TweetEvidenceDataset(test_tweets, test_evidences, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    num_runs = args.num_runs if args.stability_test else 1

    if args.stability_test:
        random_seeds = [42, 123, 456, 789, 1024][:num_runs]
        print(f"Running stability test with {num_runs} seeds: {random_seeds}")
    else:
        random_seeds = [42] * num_runs

    # Initialize metrics lists for test_gold only
    test_accuracy_list = []
    test_precision_list = []
    test_recall_list = []
    test_f1_list = []
    test_macro_precision_list = []
    test_macro_recall_list = []
    test_macro_f1_list = []

    all_runs_results = []

    try:
        for run_id in range(num_runs):
            current_seed = random_seeds[run_id]
            set_random_seed(current_seed)
            print(datetime.now().strftime('%Y-%m-%d %H:%M'),
                  f"Run {run_id + 1}/{num_runs} with seed {current_seed} ===================")

            run_suffix = f"_run{run_id}_seed{current_seed}" if args.stability_test else ""
            outputdir = args.output_dir + args.experiment_name + run_suffix + "/"
            if not os.path.exists(outputdir):
                os.makedirs(outputdir, exist_ok=True)

            model = EvidenceEnhancedBERTOnlyCross(args.model_id, num_classes=2, dropout_rate=args.dropout_rate)

            optimizer = AdamW(model.parameters(), lr=args.learning_rate)

            criterion = nn.CrossEntropyLoss()

            train_model(model, train_loader, val_loader, args.warmup_ratio, criterion, optimizer,
                        device, outputdir, run_id, global_step, wandbLogger,
                        num_epochs=args.num_epochs)

            print("Start evaluating at: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            best_checkpoint = torch.load(outputdir + 'best_model.pth')
            model.load_state_dict(best_checkpoint['model_state_dict'])

            # Evaluate on test_gold dataset only
            test_pred_dic = {}
            for j in range(5):
                test_results = evaluate_model(model, test_loader, criterion, device)
                test_pred_dic[j] = test_results['predictions']

            test_pred_df = pd.DataFrame({
                "tweet_id": test_ids,
                "tweet_text": test_tweets,
                "evidence": test_evidences,
                "class_label": test_labels,
                "prediction0": test_pred_dic[0],
                "prediction1": test_pred_dic[1],
                "prediction2": test_pred_dic[2],
                "prediction3": test_pred_dic[3],
                "prediction4": test_pred_dic[4],
            })
            test_pred_df = test_pred_df.apply(lambda sample: getMajority(sample, 5), axis=1)
            test_prediction_results = test_pred_df['prediction']

            test_pred_file = f'../prediction_results/roberta_taslp/test_gold_{args.experiment_name}{run_suffix}.csv'
            test_pred_df.to_csv(test_pred_file, index=False)

            print("Write inference csv files at: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            # Calculate metrics for test_gold (now includes macro)
            test_metrics = compute_metrics(test_prediction_results, test_labels)
            print("Test Gold Results:")
            print(f"Accuracy: {test_metrics['accuracy']:.4f}"
                  f", Precision: {test_metrics['precision']:.4f}"
                  f", Recall: {test_metrics['recall']:.4f}"
                  f", F1 Score: {test_metrics['f1']:.4f}"
                  f", Macro P: {test_metrics['macro_precision']:.4f}"
                  f", Macro R: {test_metrics['macro_recall']:.4f}"
                  f", Macro F1: {test_metrics['macro_f1']:.4f}")
            print("Confusion Matrix:")
            print(test_metrics['confusion_matrix'])
            print("=" * 50)

            run_result = {
                'run_id': run_id,
                'seed': current_seed,
                'test_gold_accuracy': test_metrics['accuracy'],
                'test_gold_precision': test_metrics['precision'],
                'test_gold_recall': test_metrics['recall'],
                'test_gold_f1': test_metrics['f1'],
                'test_gold_macro_precision': test_metrics['macro_precision'],
                'test_gold_macro_recall': test_metrics['macro_recall'],
                'test_gold_macro_f1': test_metrics['macro_f1'],
            }
            all_runs_results.append(run_result)

            # Log test metrics to wandb
            run_metrics = {
                f"run{run_id}_test_gold_accuracy": test_metrics['accuracy'],
                f"run{run_id}_test_gold_precision": test_metrics['precision'],
                f"run{run_id}_test_gold_recall": test_metrics['recall'],
                f"run{run_id}_test_gold_f1": test_metrics['f1'],
                f"run{run_id}_test_gold_macro_precision": test_metrics['macro_precision'],
                f"run{run_id}_test_gold_macro_recall": test_metrics['macro_recall'],
                f"run{run_id}_test_gold_macro_f1": test_metrics['macro_f1'],
            }
            wandbLogger.log_metrics(run_metrics, step=global_step, commit=True)

            # Append metrics to lists
            test_accuracy_list.append(test_metrics['accuracy'])
            test_precision_list.append(test_metrics['precision'])
            test_recall_list.append(test_metrics['recall'])
            test_f1_list.append(test_metrics['f1'])
            test_macro_precision_list.append(test_metrics['macro_precision'])
            test_macro_recall_list.append(test_metrics['macro_recall'])
            test_macro_f1_list.append(test_metrics['macro_f1'])

            model = None
            gc.collect()
            torch.cuda.empty_cache()

        # ── Summary statistics ────────────────────────────────────────
        def _safe_stdev(lst):
            return statistics.stdev(lst) if len(lst) > 1 else 0

        mean_accuracy_tg = statistics.mean(test_accuracy_list)
        stdev_accuracy_tg = _safe_stdev(test_accuracy_list)
        mean_precision_tg = statistics.mean(test_precision_list)
        stdev_precision_tg = _safe_stdev(test_precision_list)
        mean_recall_tg = statistics.mean(test_recall_list)
        stdev_recall_tg = _safe_stdev(test_recall_list)
        mean_f1_score_tg = statistics.mean(test_f1_list)
        stdev_f1_score_tg = _safe_stdev(test_f1_list)
        mean_macro_precision_tg = statistics.mean(test_macro_precision_list)
        stdev_macro_precision_tg = _safe_stdev(test_macro_precision_list)
        mean_macro_recall_tg = statistics.mean(test_macro_recall_list)
        stdev_macro_recall_tg = _safe_stdev(test_macro_recall_list)
        mean_macro_f1_tg = statistics.mean(test_macro_f1_list)
        stdev_macro_f1_tg = _safe_stdev(test_macro_f1_list)

        if args.stability_test:
            cv_accuracy_tg = stdev_accuracy_tg / mean_accuracy_tg if mean_accuracy_tg != 0 else float('nan')
            cv_precision_tg = stdev_precision_tg / mean_precision_tg if mean_precision_tg != 0 else float('nan')
            cv_recall_tg = stdev_recall_tg / mean_recall_tg if mean_recall_tg != 0 else float('nan')
            cv_f1_tg = stdev_f1_score_tg / mean_f1_score_tg if mean_f1_score_tg != 0 else float('nan')
            cv_macro_f1_tg = stdev_macro_f1_tg / mean_macro_f1_tg if mean_macro_f1_tg != 0 else float('nan')

            print(f"\nCoefficient of Variation (CV = std/mean):")
            print(f"Test Gold CV: Accuracy={cv_accuracy_tg:.4f}, Precision={cv_precision_tg:.4f}, "
                  f"Recall={cv_recall_tg:.4f}, F1={cv_f1_tg:.4f}, Macro F1={cv_macro_f1_tg:.4f}")

        print(f"\nExperiment: {args.experiment_name}:\n"
              f"Test Gold Metrics (binary, pos_label=1):\n"
              f"Accuracy: {mean_accuracy_tg:.4f} ± {stdev_accuracy_tg:.4f}, "
              f"Precision: {mean_precision_tg:.4f} ± {stdev_precision_tg:.4f}, "
              f"Recall: {mean_recall_tg:.4f} ± {stdev_recall_tg:.4f}, "
              f"F1 Score: {mean_f1_score_tg:.4f} ± {stdev_f1_score_tg:.4f}\n"
              f"Test Gold Metrics (macro-averaged):\n"
              f"Macro Precision: {mean_macro_precision_tg:.4f} ± {stdev_macro_precision_tg:.4f}, "
              f"Macro Recall: {mean_macro_recall_tg:.4f} ± {stdev_macro_recall_tg:.4f}, "
              f"Macro F1: {mean_macro_f1_tg:.4f} ± {stdev_macro_f1_tg:.4f}")

        # Save stability test details
        if args.stability_test:
            runs_df = pd.DataFrame(all_runs_results)
            stability_csv = f"../evaluation_results/statistics/taslp_{args.experiment_name}_results.csv"
            runs_df.to_csv(stability_csv, index=False)
            print(f"Stability test results saved to: {stability_csv}")

        # [FIX 3] CSV columns updated with macro metrics
        evaluation_file = "../evaluation_results/taslp_tc_roberta_CT22_evaluation.csv"
        columns = [
            'Experiment Name',
            'Accuracy-TG', 'Accuracy-STD-TG',
            'Precision-TG', 'Precision-STD-TG',
            'Recall-TG', 'Recall-STD-TG',
            'F1-TG', 'F1-STD-TG',
            'Macro-Precision-TG', 'Macro-Precision-STD-TG',
            'Macro-Recall-TG', 'Macro-Recall-STD-TG',
            'Macro-F1-TG', 'Macro-F1-STD-TG',
            'Timestamp',
        ]
        write_to_csv([
            args.experiment_name,
            mean_accuracy_tg, stdev_accuracy_tg,
            mean_precision_tg, stdev_precision_tg,
            mean_recall_tg, stdev_recall_tg,
            mean_f1_score_tg, stdev_f1_score_tg,
            mean_macro_precision_tg, stdev_macro_precision_tg,
            mean_macro_recall_tg, stdev_macro_recall_tg,
            mean_macro_f1_tg, stdev_macro_f1_tg,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ], evaluation_file, columns)

        # Log final stats to wandb
        final_metrics = {
            "final_test_gold_accuracy_mean": mean_accuracy_tg,
            "final_test_gold_accuracy_std": stdev_accuracy_tg,
            "final_test_gold_precision_mean": mean_precision_tg,
            "final_test_gold_precision_std": stdev_precision_tg,
            "final_test_gold_recall_mean": mean_recall_tg,
            "final_test_gold_recall_std": stdev_recall_tg,
            "final_test_gold_f1_mean": mean_f1_score_tg,
            "final_test_gold_f1_std": stdev_f1_score_tg,
            "final_test_gold_macro_precision_mean": mean_macro_precision_tg,
            "final_test_gold_macro_precision_std": stdev_macro_precision_tg,
            "final_test_gold_macro_recall_mean": mean_macro_recall_tg,
            "final_test_gold_macro_recall_std": stdev_macro_recall_tg,
            "final_test_gold_macro_f1_mean": mean_macro_f1_tg,
            "final_test_gold_macro_f1_std": stdev_macro_f1_tg,
        }

        if args.stability_test:
            final_metrics.update({
                "final_test_gold_accuracy_cv": cv_accuracy_tg,
                "final_test_gold_precision_cv": cv_precision_tg,
                "final_test_gold_recall_cv": cv_recall_tg,
                "final_test_gold_f1_cv": cv_f1_tg,
                "final_test_gold_macro_f1_cv": cv_macro_f1_tg,
            })

        wandbLogger.log_metrics(final_metrics, commit=True)

    except Exception as e:
        wandbLogger.log_metrics({"error": str(e)}, commit=True)
        raise e
    finally:
        wandbLogger.finish()


if __name__ == "__main__":
    main()