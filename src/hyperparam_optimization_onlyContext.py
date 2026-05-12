import argparse
import optuna
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import random
import sys
import gc
import emoji
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wandb import WandbLogger
from models.evidence_enhanced_bert import TweetOnlyDataset, TweetOnlyBERT


def set_random_seed(random_seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

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

def load_data(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip', dtype={"tweet_id": str})
    df['tweet_text'] = df['tweet_text'].apply(clean_tweet)
    # Fall back to tweet text when evidence is missing
    mask = (df['evidence'] == '') | (df['evidence'] == ' ') | df['evidence'].isna()
    df.loc[mask, 'evidence'] = df.loc[mask, 'tweet_text']

    return df['tweet_id'], df['evidence'], df['class_label']


def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            tweet_input_ids = batch['input_ids'].to(device)
            tweet_attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=tweet_input_ids,
                attention_mask=tweet_attention_mask
            )

            _, predictions = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0, pos_label=1)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0, pos_label=1)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0, pos_label=1)

    return f1, precision, recall


def calculate_combined_score(f1, precision, recall, weights=None):
    """Calculate weighted combined score from multiple metrics."""
    if weights is None:
        weights = {"f1": 0.6, "precision": 0.2, "recall": 0.2}

    return weights["f1"] * f1 + weights["precision"] * precision + weights["recall"] * recall


def objective(trial):
    """Optuna optimization objective function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")

    learning_rate = trial.suggest_float('learning_rate', 5e-6, 2e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 12, 16, 32])
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.1, 0.18)
    dropout_rate = trial.suggest_float('dropout_rate', 0.15, 0.25)
    num_epochs = trial.suggest_int('num_epochs', 5, 15)

    metric_weights = {
        "f1": args.f1_weight,
        "precision": args.precision_weight,
        "recall": args.recall_weight
    }

    model_id = args.model_id
    set_random_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prefix = args.prefix
    train_ids, train_tweets, train_labels = load_data(f"../data/evidence/{prefix}_train.csv")
    val_ids, val_tweets, val_labels = load_data(f"../data/evidence/{prefix}_dev.csv")

    train_dataset = TweetOnlyDataset(train_tweets, train_labels, tokenizer)
    val_dataset = TweetOnlyDataset(val_tweets, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = TweetOnlyBERT(args.model_id, num_classes=2)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    gradient_accumulation_steps = 2
    total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_combined_score = 0
    best_epoch_metrics = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            tweet_input_ids = batch['input_ids'].to(device)
            tweet_attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=tweet_input_ids,
                attention_mask=tweet_attention_mask
            )

            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            total_train_loss += loss.item() * gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        f1, precision, recall = evaluate_model(model, val_loader, device)
        combined_score = calculate_combined_score(f1, precision, recall, metric_weights)

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_epoch_metrics = {
                "epoch": epoch,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "combined": combined_score,
                "f1_weight": metric_weights["f1"],
                "precision_weight": metric_weights["precision"],
                "recall_weight": metric_weights["recall"]
            }

        trial.report(combined_score, epoch)

        print(
            f"Epoch {epoch}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Combined={combined_score:.4f}")

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    trial.set_user_attr("best_f1", best_epoch_metrics["f1"])
    trial.set_user_attr("best_precision", best_epoch_metrics["precision"])
    trial.set_user_attr("best_recall", best_epoch_metrics["recall"])
    trial.set_user_attr("best_epoch", best_epoch_metrics["epoch"])
    trial.set_user_attr("best_combined_score", best_epoch_metrics["combined"])
    trial.set_user_attr("f1_weight", best_epoch_metrics["f1_weight"])
    trial.set_user_attr("precision_weight", best_epoch_metrics["precision_weight"])
    trial.set_user_attr("recall_weight", best_epoch_metrics["recall_weight"])

    return best_combined_score


def parse_args():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters with multiple metrics using Optuna.')
    parser.add_argument('--model_id', type=str, default="FacebookAI/roberta-large",
                        help='Path to the pre-trained model')
    parser.add_argument('--n_trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default="roberta_evidence_optuna", help='Name for the Optuna study')
    parser.add_argument('--storage', type=str, default=None, help='Database URL for Optuna storage')
    parser.add_argument('--prefix', type=str, default='CT22_claim/CT22_gpt4o_generated_context',
                        help='Data path prefix')

    parser.add_argument('--f1_weight', type=float, default=0.6, help='Weight for F1 score (default: 0.6)')
    parser.add_argument('--precision_weight', type=float, default=0.2, help='Weight for precision (default: 0.2)')
    parser.add_argument('--recall_weight', type=float, default=0.2, help='Weight for recall (default: 0.2)')

    return parser.parse_args()


def main():
    global args
    args = parse_args()

    total_weight = args.f1_weight + args.precision_weight + args.recall_weight
    if abs(total_weight - 1.0) > 1e-5:
        print(f"Warning: Metric weights sum to {total_weight}, not 1.0. Normalizing weights.")
        args.f1_weight /= total_weight
        args.precision_weight /= total_weight
        args.recall_weight /= total_weight

    print(f"Using metric weights: F1={args.f1_weight}, Precision={args.precision_weight}, Recall={args.recall_weight}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        load_if_exists=True
    )

    study.optimize(objective, n_trials=args.n_trials)

    print("\nBest parameters:")
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    best_trial = study.best_trial
    print("\nBest trial metrics:")
    print(f"  Combined score: {best_trial.value:.4f}")
    print(f"  F1 score: {best_trial.user_attrs['best_f1']:.4f} (weight: {best_trial.user_attrs['f1_weight']:.2f})")
    print(
        f"  Precision: {best_trial.user_attrs['best_precision']:.4f} (weight: {best_trial.user_attrs['precision_weight']:.2f})")
    print(
        f"  Recall: {best_trial.user_attrs['best_recall']:.4f} (weight: {best_trial.user_attrs['recall_weight']:.2f})")
    print(f"  Best epoch: {best_trial.user_attrs['best_epoch']}")

    df = study.trials_dataframe()


    for trial in study.trials:
        for key, value in trial.user_attrs.items():
            df.loc[trial.number, key] = value

    if "best_combined_score" not in df.columns:
        for trial in study.trials:
            if hasattr(trial, "value") and trial.value is not None:
                df.loc[trial.number, "best_combined_score"] = trial.value

    if "f1_weight" not in df.columns:
        df["f1_weight"] = args.f1_weight

    if "precision_weight" not in df.columns:
        df["precision_weight"] = args.precision_weight

    if "recall_weight" not in df.columns:
        df["recall_weight"] = args.recall_weight

    metric_cols = ["best_f1", "best_precision", "best_recall", "best_combined_score",
                   "f1_weight", "precision_weight", "recall_weight", "best_epoch"]
    avail_metric_cols = [col for col in metric_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in avail_metric_cols]

    df = df[avail_metric_cols + other_cols]

    df.to_csv(f"../data/hyperparameter/optuna_results_{args.study_name}.csv", index=False)

    best_metrics_df = pd.DataFrame([{
        "study_name": args.study_name,
        "model_id": args.model_id,
        "prefix": args.prefix,
        "best_f1": best_trial.user_attrs["best_f1"],
        "best_precision": best_trial.user_attrs["best_precision"],
        "best_recall": best_trial.user_attrs["best_recall"],
        "best_combined_score": best_trial.value,
        "f1_weight": best_trial.user_attrs["f1_weight"],
        "precision_weight": best_trial.user_attrs["precision_weight"],
        "recall_weight": best_trial.user_attrs["recall_weight"],
        "best_epoch": best_trial.user_attrs["best_epoch"],
    }])

    for param, value in best_params.items():
        best_metrics_df[param] = value

    best_metrics_df.to_csv(f"../data/hyperparameter/best_result_{args.study_name}.csv", index=False)

    importance = optuna.importance.get_param_importances(study)
    print("\nParameter importance:")
    for param, score in importance.items():
        print(f"  {param}: {score:.4f}")


if __name__ == "__main__":
    main()