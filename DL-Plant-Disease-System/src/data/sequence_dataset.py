import torch
from torch.utils.data import Dataset


class SequenceFromFeatures(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, seq_len: int = 5):
        assert features.shape[0] == labels.shape[0]
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.features) - self.seq_len + 1)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return x, y
