import torch
from tqdm import tqdm
from typing import List
from os.path import exists
from transformers import *
from torch.utils.data import DataLoader
from datasets import Dataset
from custom_dataset import *
import pickle
import pandas as pd
from collections import defaultdict

class Embedder:

    def __init__(self, embedding_model, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model, output_hidden_states=True).to(device)
        self.device = device

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def create_embeddings(self, texts, labels, layers: List, embedding_path, batch_size=32):

        file_exists = exists(embedding_path)

        if file_exists:
            raise Exception("File Already Exists")

        df = pd.DataFrame({"texts": texts})

        train_dataset = Dataset.from_pandas(df)
        train_dataset = prepare_dataset(train_dataset, self.tokenizer)

        saving_dict = defaultdict(list)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        pbar = tqdm(total=len(train_loader), position=0)

        with torch.no_grad():
            self.model.eval()
            for batch in train_loader:
                pbar.update(1)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                preds = self.model(**batch)

                for layer in layers:
                    mean_pooling = self.mean_pooling(preds["hidden_states"][layer], batch["attention_mask"]).detach()
                    mean_pooling = mean_pooling.cpu()
                    mean_pooling = mean_pooling.numpy()
                    saving_dict[layer].extend(mean_pooling)

        pbar.close()

        saving_dict["labels"] = labels
        saving_dict = dict(saving_dict)
        with open(f"{embedding_path}", "wb") as filino:
            pickle.dump(saving_dict, filino)