import argparse
import optuna
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import random
import sys
import gc
import emoji
import re
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wandb import WandbLogger
from models.evidence_enhanced_bert import EvidenceEnhancedBERTOnlyCross, TweetEvidenceDataset


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def clean_tweet(tweet):
    tweet = emoji.demojize(tweet)
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'[@#]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub(r'&amp;', '&', tweet)
    return tweet


def load_data(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip', dtype={"tweet_id": str})
    df['tweet_text'] = df['tweet_text'].apply(clean_tweet)
    df['evidence'] = df['evidence'].fillna('')
    df['evidence'] = df['evidence'].astype(str)
    df['evidence'] = df['evidence'].replace('nan', '')

    return df['tweet_id'], df['tweet_text'], df['evidence'], df['class_label']


def evaluate_model(model, data_loader, device):
    model.eval()
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

            _, predictions = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Binary (positive-class) metrics
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0, pos_label=1)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0, pos_label=1)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0, pos_label=1)

    # [FIX 1] Added macro-averaged metrics for consistency with all other scripts
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    macro_precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
    }


def calculate_combined_score(metrics, weights=None):
    if weights is None:
        weights = {"f1": 0.6, "precision": 0.2, "recall": 0.2}

    return (weights["f1"] * metrics['f1']
            + weights["precision"] * metrics['precision']
            + weights["recall"] * metrics['recall'])


def make_objective(args):
    """
    [FIX 2] Use a factory function instead of relying on a global `args`.
    This returns a closure that captures `args`, which is safer for
    potential distributed Optuna usage and avoids global state.
    """

    def objective(trial):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"using device: {device}")

        # Hyperparameter search space
        learning_rate = trial.suggest_float('learning_rate', 5e-6, 2e-5, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        warmup_ratio = 0.15
        # [FIX 3] Pass dropout_rate directly to model constructor instead of
        # post-hoc module traversal
        dropout_rate = trial.suggest_float('dropout_rate', 0.15, 0.25)
        num_epochs = trial.suggest_int('num_epochs', 5, 15)

        metric_weights = {
            "f1": args.f1_weight,
            "precision": args.precision_weight,
            "recall": args.recall_weight,
        }

        model_id = args.model_id

        set_random_seed(42)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        prefix = args.prefix
        train_ids, train_tweets, train_evidences, train_labels = load_data(
            f"../data/evidence/{prefix}_train.csv")
        val_ids, val_tweets, val_evidences, val_labels = load_data(
            f"../data/evidence/{prefix}_dev.csv")

        train_dataset = TweetEvidenceDataset(train_tweets, train_evidences, train_labels, tokenizer)
        val_dataset = TweetEvidenceDataset(val_tweets, val_evidences, val_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # [FIX 3] Pass dropout_rate directly — no need for post-hoc module traversal
        model = EvidenceEnhancedBERTOnlyCross(
            model_id, num_classes=2, dropout_rate=dropout_rate
        )
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        gradient_accumulation_steps = 2
        total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
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
                tweet_input_ids = batch['tweet_input_ids'].to(device)
                tweet_attention_mask = batch['tweet_attention_mask'].to(device)
                evidence_input_ids = batch['evidence_input_ids'].to(device)
                evidence_attention_mask = batch['evidence_attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    tweet_input_ids=tweet_input_ids,
                    tweet_attention_mask=tweet_attention_mask,
                    evidence_input_ids=evidence_input_ids,
                    evidence_attention_mask=evidence_attention_mask,
                )

                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
                loss.backward()

                total_train_loss += loss.item() * gradient_accumulation_steps

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # [FIX 4] Flush remaining gradients if last batch didn't align
            # with gradient_accumulation_steps boundary.
            if (batch_idx + 1) % gradient_accumulation_steps != 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Evaluate current epoch
            metrics = evaluate_model(model, val_loader, device)
            combined_score = calculate_combined_score(metrics, metric_weights)

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_epoch_metrics = {
                    "epoch": epoch,
                    "f1": metrics['f1'],
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "macro_f1": metrics['macro_f1'],
                    "macro_precision": metrics['macro_precision'],
                    "macro_recall": metrics['macro_recall'],
                    "combined": combined_score,
                    "f1_weight": metric_weights["f1"],
                    "precision_weight": metric_weights["precision"],
                    "recall_weight": metric_weights["recall"],
                }

            trial.report(combined_score, epoch)

            print(
                f"Epoch {epoch}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, Macro F1={metrics['macro_f1']:.4f}, "
                f"Combined={combined_score:.4f}")

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Save all metrics as trial user_attrs
        trial.set_user_attr("best_f1", best_epoch_metrics["f1"])
        trial.set_user_attr("best_precision", best_epoch_metrics["precision"])
        trial.set_user_attr("best_recall", best_epoch_metrics["recall"])
        trial.set_user_attr("best_macro_f1", best_epoch_metrics["macro_f1"])
        trial.set_user_attr("best_macro_precision", best_epoch_metrics["macro_precision"])
        trial.set_user_attr("best_macro_recall", best_epoch_metrics["macro_recall"])
        trial.set_user_attr("best_epoch", best_epoch_metrics["epoch"])
        trial.set_user_attr("best_combined_score", best_epoch_metrics["combined"])
        trial.set_user_attr("f1_weight", best_epoch_metrics["f1_weight"])
        trial.set_user_attr("precision_weight", best_epoch_metrics["precision_weight"])
        trial.set_user_attr("recall_weight", best_epoch_metrics["recall_weight"])

        return best_combined_score

    return objective


def parse_args():
    parser = argparse.ArgumentParser(
        description='Optimize hyperparameters with multiple metrics using Optuna.')
    parser.add_argument('--model_id', type=str, default="FacebookAI/roberta-large")
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--study_name', type=str, default="roberta_evidence_optuna")
    parser.add_argument('--storage', type=str, default=None)
    parser.add_argument('--prefix', type=str, default='CT22_claim/CT22_gpt4o_generated_context')
    parser.add_argument('--f1_weight', type=float, default=0.6)
    parser.add_argument('--precision_weight', type=float, default=0.2)
    parser.add_argument('--recall_weight', type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    # Normalize weights to sum to 1
    total_weight = args.f1_weight + args.precision_weight + args.recall_weight
    if abs(total_weight - 1.0) > 1e-5:
        print(f"Warning: Metric weights sum to {total_weight}, not 1.0. Normalizing weights.")
        args.f1_weight /= total_weight
        args.precision_weight /= total_weight
        args.recall_weight /= total_weight

    print(f"Using metric weights: F1={args.f1_weight}, Precision={args.precision_weight}, "
          f"Recall={args.recall_weight}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        load_if_exists=True,
    )

    # [FIX 2] Pass args via closure instead of global variable
    study.optimize(make_objective(args), n_trials=args.n_trials)

    # Print best hyperparameters
    print("\nBest parameters:")
    best_params = study.best_params
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Print best trial metrics
    best_trial = study.best_trial
    print("\nBest trial metrics:")
    print(f"  Combined score: {best_trial.value:.4f}")
    print(f"  F1 score: {best_trial.user_attrs['best_f1']:.4f} "
          f"(weight: {best_trial.user_attrs['f1_weight']:.2f})")
    print(f"  Precision: {best_trial.user_attrs['best_precision']:.4f} "
          f"(weight: {best_trial.user_attrs['precision_weight']:.2f})")
    print(f"  Recall: {best_trial.user_attrs['best_recall']:.4f} "
          f"(weight: {best_trial.user_attrs['recall_weight']:.2f})")
    print(f"  Macro F1: {best_trial.user_attrs['best_macro_f1']:.4f}")
    print(f"  Macro Precision: {best_trial.user_attrs['best_macro_precision']:.4f}")
    print(f"  Macro Recall: {best_trial.user_attrs['best_macro_recall']:.4f}")
    print(f"  Best epoch: {best_trial.user_attrs['best_epoch']}")

    # Save full results to CSV
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

    metric_cols = [
        "best_f1", "best_precision", "best_recall",
        "best_macro_f1", "best_macro_precision", "best_macro_recall",
        "best_combined_score",
        "f1_weight", "precision_weight", "recall_weight", "best_epoch",
    ]
    avail_metric_cols = [col for col in metric_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in avail_metric_cols]

    df = df[avail_metric_cols + other_cols]

    df.to_csv(f"../data/hyperparameter/taslp/optuna_results_{args.study_name}.csv", index=False)

    # Save simplified best-result CSV
    best_metrics_df = pd.DataFrame([{
        "study_name": args.study_name,
        "model_id": args.model_id,
        "prefix": args.prefix,
        "best_f1": best_trial.user_attrs["best_f1"],
        "best_precision": best_trial.user_attrs["best_precision"],
        "best_recall": best_trial.user_attrs["best_recall"],
        "best_macro_f1": best_trial.user_attrs["best_macro_f1"],
        "best_macro_precision": best_trial.user_attrs["best_macro_precision"],
        "best_macro_recall": best_trial.user_attrs["best_macro_recall"],
        "best_combined_score": best_trial.value,
        "f1_weight": best_trial.user_attrs["f1_weight"],
        "precision_weight": best_trial.user_attrs["precision_weight"],
        "recall_weight": best_trial.user_attrs["recall_weight"],
        "best_epoch": best_trial.user_attrs["best_epoch"],
    }])

    for param, value in best_params.items():
        best_metrics_df[param] = value

    best_metrics_df.to_csv(
        f"../data/hyperparameter/taslp/best_result_{args.study_name}.csv", index=False)

    # Print parameter importance
    importance = optuna.importance.get_param_importances(study)
    print("\nParameter importance:")
    for param, score in importance.items():
        print(f"  {param}: {score:.4f}")


if __name__ == "__main__":
    main()