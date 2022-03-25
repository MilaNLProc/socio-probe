import os
import pandas as pd
from torch.utils.data import Dataset

class ProbingDataset(Dataset):
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