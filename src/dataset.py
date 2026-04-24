# importing packages
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from interface import TOKEN_PADDING, MAX_LENGTH
from sampler import negative_sample


class SASRecDataset(Dataset):
    """
    Generates training samples for SASRec.

    For each user sequence of length N, we produce N-1 training samples —
    one for every prefix length from 1 to N-1. This matches the reference
    implementation (kang205/SASRec) and the assignment requirement to
    "construct input-target pairs for next-item prediction based on user
    sequences" (plural pairs per user, not one).

    Each sample is a sliding window over the user's history:
        seq[:i+1] → predict seq[i] from seq[:i]

    Example for sequence [A, B, C, D, E]:
        sample 1: input=[A],       target=B
        sample 2: input=[A, B],    target=C
        sample 3: input=[A, B, C], target=D
        sample 4: input=[A,B,C,D], target=E
    """
    def __init__(self, train_data, num_items):
        self.num_items = num_items
        self.samples = []

        # Build all N-1 prefix samples per user
        for user, seq in train_data.items():
            h = set(seq)                            # full user history for negative sampling
            for i in range(1, len(seq)):
                self.samples.append((h, seq[:i + 1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h, seq = self.samples[idx]

        seq1   = seq[:-1]       # input sequence (all but last item)
        target = seq[-1]        # next item to predict

        # Truncate to MAX_LENGTH, keeping most recent items
        seq1 = seq1[-MAX_LENGTH:]

        # Left-pad with TOKEN_PADDING (0) to fixed length MAX_LENGTH
        pad_len = MAX_LENGTH - len(seq1)
        seq1 = [TOKEN_PADDING] * pad_len + seq1

        # Positive targets: shift seq1 left by one, append actual target
        pos = seq1[1:] + [target]

        # Sample one negative item per position; pad positions get TOKEN_PADDING
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
    """Load preprocessed data and return a DataLoader for training."""
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