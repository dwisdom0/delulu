import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MoonsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_moons_datasets():
    X, Y = make_moons(n_samples=1500, shuffle=True, noise=0.1)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
    return MoonsDataset(x_train, y_train), MoonsDataset(x_test, y_test)
