import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch, numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from tqdm import tqdm

from utils.seed import set_seed
from utils.data_prep import download_and_prepare_datasets
from models.base import MultilingualEmotionDetector


class EmotionDataset(Dataset):
    def __init__(self, hf_dataset, emotions, tokenizer):
        self.data = hf_dataset
        self.emotions = emotions
        self.tok = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        vec = np.zeros(len(self.emotions) * 2, dtype=np.float32)
        if row['emotion'].lower() in self.emotions:
            idx_e = self.emotions.index(row['emotion'].lower())
            vec[idx_e] = 1.0
            vec[idx_e + len(self.emotions)] = 1.0
        tokens = self.tok(
            row['text'],
            truncation=True,
            padding='max_length',
            max_length=64,
            return_tensors='pt'
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(vec)
        }


def train_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultilingualEmotionDetector(device=device)

    dataset = download_and_prepare_datasets()
    loader = DataLoader(
        EmotionDataset(dataset, model.emotions, model.tokenizer),
        batch_size=16,
        shuffle=True
    )

    opt = AdamW(
        list(model.base_model.parameters()) + list(model.classifier.parameters()),
        lr=5e-5
    )
    loss_fn = BCEWithLogitsLoss()

    for epoch in range(1):
        model.train()
        total_loss = 0
        for batch in tqdm(loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            opt.zero_grad()
            out = model.forward(ids, mask)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print("Epoch loss:", total_loss / len(loader))
    return model
