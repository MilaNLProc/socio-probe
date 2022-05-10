import os
import pandas as pd
from torch.utils import data as dst

class ProbingDataset(dst.Dataset):
    """
    Simple dataset wrapper to support the probing pipeline
    """
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):

        return self.embeddings[idx], self.labels[idx]


def tt(tokenizer):
    def tokenize_function(examples):
        # Remove empty lines
        examples["texts"] = [
            line
            for line in examples["texts"]]

        return tokenizer(
            examples["texts"],
            padding="max_length")
    return tokenize_function

def prepare_dataset(dataset, tokenizer):

    dataset = dataset.map(
        tt(tokenizer),
        batched=True,
        remove_columns=["texts"],

    )

    dataset.set_format("torch")

    return dataset