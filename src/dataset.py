#importing packages
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from interface import TOKEN_PADDING, MAX_LENGTH
from sampler import negative_sample
class SASRecDataset(Dataset):
    # We are generating samples for SASRec training
    def __init__(self, train_data, num_items):
        self.num_items = num_items
        self.samples = []

        # We built training samples from user interaction sequences
        for user, seq in train_data.items():
            h = set(seq)  
            for i in range(1, len(seq)):
                self.samples.append((h, seq[:i + 1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve one training example
        h, seq = self.samples[idx]
        seq1 = seq[:-1]
        target = seq[-1]
        
        # perform Truncate
        seq1 = seq1[-MAX_LENGTH:]

        # doing TOKEN_PADDING
        pad_len = MAX_LENGTH - len(seq1)
        seq1 = [TOKEN_PADDING] * pad_len + seq1
        pos = seq1[1:] + [target]
        # Sampling one negative item per position
        neg = [
            negative_sample(h, self.num_items)
            if p != TOKEN_PADDING else TOKEN_PADDING
            for p in pos
        ]
        return (
            torch.LongTensor(seq1),
            torch.LongTensor(pos),
            torch.LongTensor(neg),
        )


def get_loader(data_path="../data/processed/data.pkl", batch_size=128):
    # Loading preprocessed data and creation of  DataLoader
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    dataset = SASRecDataset(data["train"], data["num_items"])

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, data["val"], data["test"], data["num_items"]
