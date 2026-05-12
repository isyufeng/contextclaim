# coding=utf-8

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModel

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


class TweetEvidenceDataset(Dataset):
    def __init__(self, tweets, evidences, labels, tokenizer, max_length=128):
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
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )

        evidence_encoding = self.tokenizer(
            evidence,
            add_special_tokens=True,
            max_length=self.max_length,
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