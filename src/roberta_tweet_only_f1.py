import argparse
import csv
import gc
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime

import emoji
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wandb import WandbLogger


class TweetOnlyDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_length=128):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = str(self.tweets[idx])

        encoding = self.tokenizer(
            tweet,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class TweetAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, hidden_states, attention_mask):
        attention_weights = self.attention(hidden_states)
        bool_attention_mask = attention_mask.bool().unsqueeze(-1)
        attention_weights = attention_weights.masked_fill(~bool_attention_mask, float('-inf'))
        attention_weights = torch.softmax(attention_weights, dim=1)
        weighted_sum = torch.sum(attention_weights * hidden_states, dim=1)
        return weighted_sum


class TweetOnlyBERT(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.attention = TweetAttention(self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden = outputs.last_hidden_state
        pooled = self.attention(hidden, attention_mask)
        logits = self.classifier(pooled)
        return logits


def clean_tweet(tweet):
    # convert emojis from the text
    tweet = emoji.demojize(tweet)
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'[@#]', '', tweet)
    # matches one or more whitespace characters (spaces, tabs, newlines, etc.)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub(r'&amp;', '&', tweet)
    return tweet


def getMajority(sample, iterations):
    predictions = []

    for i in range(iterations):
        predictions.append(sample["prediction" + str(i)])

    counter = Counter(predictions)
    majority, count = counter.most_common()[0]
    sample['prediction'] = majority
    return sample


def compute_metrics(predictions, labels):
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='binary', zero_division=0, pos_label=1),
        'recall': recall_score(labels, predictions, average='binary', zero_division=0, pos_label=1),
        'f1': f1_score(labels, predictions, average='binary', zero_division=0, pos_label=1),
        'confusion_matrix': confusion_matrix(labels, predictions)
    }


def train_model(model, train_loader, val_loader, warmup_ratio, criterion, optimizer, device, outputdir,
                iteration, global_step, wandbLogger, num_epochs=3,
                early_stopping_patience=3, gradient_accumulation_steps=2):
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
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
        training_history['val_f1_score'].append(val_metrics['f1'])
        training_history['epochs'].append(epoch + 1)

        epoch_metrics = {
            "current_epoch": epoch,
            "current_iteration": iteration,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_metrics['accuracy'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1": val_metrics['f1']
        }
        wandbLogger.log_metrics(epoch_metrics, step=global_step, commit=True)
        global_step += 1

        print(f"Epoch {epoch + 1}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Metrics:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}"
              f", Precision: {val_metrics['precision']:.4f}"
              f", Recall: {val_metrics['recall']:.4f}"
              f", F1 Score: {val_metrics['f1']:.4f}"
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
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


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    # torch.mps.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


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
    """Thread-safe CSV writer."""
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
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio')
    parser.add_argument('--lr_scheduler', type=str, default='linear',
                        choices=['constant', 'linear', 'cosine', 'cosine_with_restarts'],
                        help='Learning rate scheduler type')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs for stability testing')
    parser.add_argument('--stability_test', action='store_true', help='Run stability test with multiple seeds')
    parser.add_argument('--output_dir', type=str, default='/gpfs/scratch/acw760/phd_afc/best_param/',
                        help='Output directory for checkpoints')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name for this experiment run')
    return parser.parse_args()


def load_data(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip', dtype={"tweet_id": str})
    df['tweet_text'] = df['tweet_text'].apply(clean_tweet)

    tweet_ids = df['tweet_id']
    tweets = df['tweet_text']
    labels = df['class_label']
    return tweet_ids, tweets, labels


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
        "early_stopping_patience": 3,
        "optimizer": "AdamW"
    }

    wandbLogger = WandbLogger(
        project="ft_no_evidence",
        name=args.experiment_name,
        config=config,
        tags=["tweet_only", "roberta", "best_model_f1"],
        notes="Roberta without evidence model",
        group="BERT-based-experiments"
    )

    global_step = 0

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    prefix = 'CT22_claim/CT22_gpt4o_context_web_new'
    train_ids, train_tweets, train_labels = load_data(f"../data/evidence/{prefix}_train.csv")
    val_ids, val_tweets, val_labels = load_data(f"../data/evidence/{prefix}_dev.csv")
    dev_test_ids, dev_test_tweets, dev_test_labels = load_data(f"../data/evidence/{prefix}_dev_test.csv")
    test_ids, test_tweets, test_labels = load_data(f"../data/evidence/{prefix}_test_gold.csv")

    train_dataset = TweetOnlyDataset(train_tweets, train_labels, tokenizer)
    val_dataset = TweetOnlyDataset(val_tweets, val_labels, tokenizer)
    dev_test_dataset = TweetOnlyDataset(dev_test_tweets, dev_test_labels, tokenizer)
    test_dataset = TweetOnlyDataset(test_tweets, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    dev_test_loader = DataLoader(dev_test_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


    num_runs = args.num_runs if args.stability_test else 1

    if args.stability_test:
        random_seeds = [42, 123, 456, 789, 1024][:num_runs]
        print(f"Running stability test with {num_runs} different random seeds: {random_seeds}")
    else:
        random_seeds = [42] * num_runs

    dev_test_accuracy_list = []
    dev_test_precision_list = []
    dev_test_recall_list = []
    dev_test_f1_list = []

    test_accuracy_list = []
    test_precision_list = []
    test_recall_list = []
    test_f1_list = []

    evaluation_file = "../evaluation_results/t_best_param_roberta_CT22_evaluation.csv"
    columns = ['Experiment Name',
               'Accuracy-DT', 'Accuracy-STD-DT', 'Precision-DT',
               'Precision-STD-DT', 'Recall-DT', 'Recall-STD-DT', 'F1-DT', 'F1-STD-DT',
               'Accuracy-TG', 'Accuracy-STD-TG', 'Precision-TG',
               'Precision-STD-TG', 'Recall-TG', 'Recall-STD-TG', 'F1-TG', 'F1-STD-TG',
               'Timestamp']

    try:
        for i in range(num_runs):
            current_seed = random_seeds[i]
            set_random_seed(current_seed)
            print(datetime.now().strftime('%Y-%m-%d %H:%M'), f"Iterate training {i}===================")

            model = TweetOnlyBERT(args.model_id, num_classes=2)
            optimizer = AdamW(model.parameters(), lr=args.learning_rate)
            criterion = nn.CrossEntropyLoss()
            outputdir = args.output_dir + args.experiment_name + "/"
            if not os.path.exists(outputdir):
                os.makedirs(outputdir, exist_ok=True)
            train_model(model, train_loader, val_loader, args.warmup_ratio, criterion, optimizer, device,
                        outputdir, i, global_step, wandbLogger, num_epochs=args.num_epochs)

            print("Start evaluating at: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            best_checkpoint = torch.load(outputdir + 'best_model.pth')
            model.load_state_dict(best_checkpoint['model_state_dict'])

            dev_test_pred_dic = {}
            for j in range(5):
                dev_test_results = evaluate_model(model, dev_test_loader, criterion, device)
                dev_test_pred_dic[j] = dev_test_results['predictions']

            dev_test_pred_df = pd.DataFrame({
                "tweet_id": dev_test_ids,
                "tweet_text": dev_test_tweets,
                "class_label": dev_test_labels,
                "prediction0": dev_test_pred_dic[0],
                "prediction1": dev_test_pred_dic[1],
                "prediction2": dev_test_pred_dic[2],
                "prediction3": dev_test_pred_dic[3],
                "prediction4": dev_test_pred_dic[4],
            })
            dev_test_pred_df = dev_test_pred_df.apply(lambda sample: getMajority(sample, 5), axis=1)
            dev_test_prediction_results = dev_test_pred_df['prediction']
            dev_test_pred_df.to_csv(
                '../prediction_results/roberta/no_evidence/dev_test_' + args.experiment_name + "_" + str(i) + ".csv",
                index=False)

            test_pred_dic = {}
            for j in range(5):
                test_results = evaluate_model(model, test_loader, criterion, device)
                test_pred_dic[j] = test_results['predictions']

            test_pred_df = pd.DataFrame({
                "tweet_id": test_ids,
                "tweet_text": test_tweets,
                "class_label": test_labels,
                "prediction0": test_pred_dic[0],
                "prediction1": test_pred_dic[1],
                "prediction2": test_pred_dic[2],
                "prediction3": test_pred_dic[3],
                "prediction4": test_pred_dic[4],
            })
            test_pred_df = test_pred_df.apply(lambda sample: getMajority(sample, 5), axis=1)
            test_prediction_results = test_pred_df['prediction']
            test_pred_df.to_csv(
                '../prediction_results/roberta/no_evidence/test_gold_' + args.experiment_name + "_" + str(i) + ".csv",
                index=False)

            print("Write inference csv files at: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            dev_test_metrics = compute_metrics(dev_test_prediction_results, dev_test_labels)
            print("Dev Test Results:")
            print(f"Accuracy: {dev_test_metrics['accuracy']:.4f}"
                  f", Precision: {dev_test_metrics['precision']:.4f}"
                  f", Recall: {dev_test_metrics['recall']:.4f}"
                  f", F1 Score: {dev_test_metrics['f1']:.4f}")
            print("Confusion Matrix:")
            print(dev_test_metrics['confusion_matrix'])

            test_metrics = compute_metrics(test_prediction_results, test_labels)
            print("Test Gold Results:")
            print(f"Accuracy: {test_metrics['accuracy']:.4f}"
                  f", Precision: {test_metrics['precision']:.4f}"
                  f", Recall: {test_metrics['recall']:.4f}"
                  f", F1 Score: {test_metrics['f1']:.4f}")
            print("Confusion Matrix:")
            print(test_metrics['confusion_matrix'])
            print("=" * 50)

            iteration_summary = {
                "dev_test_accuracy": dev_test_metrics['accuracy'],
                "dev_test_precision": dev_test_metrics['precision'],
                "dev_test_recall": dev_test_metrics['recall'],
                "dev_test_f1": dev_test_metrics['f1'],
                "test_gold_accuracy": test_metrics['accuracy'],
                "test_gold_precision": test_metrics['precision'],
                "test_gold_recall": test_metrics['recall'],
                "test_gold_f1": test_metrics['f1']
            }
            wandbLogger.log_metrics(iteration_summary, step=global_step, commit=True)

            dev_test_accuracy_list.append(dev_test_metrics['accuracy'])
            dev_test_precision_list.append(dev_test_metrics['precision'])
            dev_test_recall_list.append(dev_test_metrics['recall'])
            dev_test_f1_list.append(dev_test_metrics['f1'])

            test_accuracy_list.append(test_metrics['accuracy'])
            test_precision_list.append(test_metrics['precision'])
            test_recall_list.append(test_metrics['recall'])
            test_f1_list.append(test_metrics['f1'])

            model = None
            gc.collect()
            torch.cuda.empty_cache()

        mean_accuracy_dt = statistics.mean(dev_test_accuracy_list)
        stdev_accuracy_dt = statistics.stdev(dev_test_accuracy_list)
        mean_precision_dt = statistics.mean(dev_test_precision_list)
        stdev_precision_dt = statistics.stdev(dev_test_precision_list)
        mean_recall_dt = statistics.mean(dev_test_recall_list)
        stdev_recall_dt = statistics.stdev(dev_test_recall_list)
        mean_f1_score_dt = statistics.mean(dev_test_f1_list)
        stdev_f1_score_dt = statistics.stdev(dev_test_f1_list)

        mean_accuracy_tg = statistics.mean(test_accuracy_list)
        stdev_accuracy_tg = statistics.stdev(test_accuracy_list)
        mean_precision_tg = statistics.mean(test_precision_list)
        stdev_precision_tg = statistics.stdev(test_precision_list)
        mean_recall_tg = statistics.mean(test_recall_list)
        stdev_recall_tg = statistics.stdev(test_recall_list)
        mean_f1_score_tg = statistics.mean(test_f1_list)
        stdev_f1_score_tg = statistics.stdev(test_f1_list)

        print(f"Experiment: {args.experiment_name}: \n"
              f"Dev Test Metrics:\n"
              f"Accuracy: {mean_accuracy_dt:.4f} ± {stdev_accuracy_dt:.4f}, "
              f"Precision: {mean_precision_dt:.4f} ± {stdev_precision_dt:.4f}, "
              f"Recall: {mean_recall_dt:.4f} ± {stdev_recall_dt:.4f}, "
              f"F1 Score: {mean_f1_score_dt:.4f} ± {stdev_f1_score_dt:.4f}\n"
              f"Test Gold Metrics:\n"
              f"Accuracy: {mean_accuracy_tg:.4f} ± {stdev_accuracy_tg:.4f}, "
              f"Precision: {mean_precision_tg:.4f} ± {stdev_precision_tg:.4f}, "
              f"Recall: {mean_recall_tg:.4f} ± {stdev_recall_tg:.4f}, "
              f"F1 Score: {mean_f1_score_tg:.4f} ± {stdev_f1_score_tg:.4f}")

        write_to_csv([args.experiment_name,
                      mean_accuracy_dt, stdev_accuracy_dt,
                      mean_precision_dt, stdev_precision_dt,
                      mean_recall_dt, stdev_recall_dt,
                      mean_f1_score_dt, stdev_f1_score_dt,
                      mean_accuracy_tg, stdev_accuracy_tg,
                      mean_precision_tg, stdev_precision_tg,
                      mean_recall_tg, stdev_recall_tg,
                      mean_f1_score_tg, stdev_f1_score_tg,
                      datetime.now().strftime('%Y-%m-%d %H:%M:%S')], evaluation_file, columns)

        final_metrics = {
            "final_dev_test_accuracy_mean": mean_accuracy_dt,
            "final_dev_test_accuracy_std": stdev_accuracy_dt,
            "final_dev_test_precision_mean": mean_precision_dt,
            "final_dev_test_precision_std": stdev_precision_dt,
            "final_dev_test_recall_mean": mean_recall_dt,
            "final_dev_test_recall_std": stdev_recall_dt,
            "final_dev_test_f1_mean": mean_f1_score_dt,
            "final_dev_test_f1_std": stdev_f1_score_dt,
            "final_test_gold_accuracy_mean": mean_accuracy_tg,
            "final_test_gold_accuracy_std": stdev_accuracy_tg,
            "final_test_gold_precision_mean": mean_precision_tg,
            "final_test_gold_precision_std": stdev_precision_tg,
            "final_test_gold_recall_mean": mean_recall_tg,
            "final_test_gold_recall_std": stdev_recall_tg,
            "final_test_gold_f1_mean": mean_f1_score_tg,
            "final_test_gold_f1_std": stdev_f1_score_tg
        }
        wandbLogger.log_metrics(final_metrics, commit=True)

    except Exception as e:
        wandbLogger.log_metrics({"error": str(e)}, commit=True)
        raise e
    finally:
        wandbLogger.finish()


if __name__ == "__main__":
    main()
