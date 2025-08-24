import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
import plotly.graph_objects as go
from torchvision import datasets
from torch.utils.data import DataLoader
from multiprocessing import Pool, cpu_count


class ReLU(nn.Module):
    @staticmethod
    def name():
        return "ReLU"

    def forward(self, x):
        return torch.where(x >= 0, x, 0)


class Adonis(nn.Module):
    @staticmethod
    def name():
        return "Adonis"

    def forward(self, x):
        return x.abs()


class DeluLU(nn.Module):
    @staticmethod
    def name():
        return "DeluLU"

    def forward(self, x):
        # if x >= 0 -> x
        # if x < 0  -> 1 - (alpha / (alpha - x))
        alpha = 0.2
        return torch.where(x >= 0, x, 1 - (alpha / (alpha - x)))


class DeluLUv2(nn.Module):
    @staticmethod
    def name():
        return "DeluLUv2"

    def forward(self, x):
        # if x >= 0 -> x
        # if x < 0  -> (alpha / (alpha - x)) - 1
        alpha = 0.2
        return torch.where(x >= 0, x, (alpha / (alpha - x)) - 1)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation_func: nn.Module,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation_func())
        for i, h in enumerate(hidden_dims[1:]):
            layers.append(nn.Linear(hidden_dims[i], h))
            layers.append(activation_func())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    epochs: int = 5,
) -> list[float]:
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    mean_train_losses = []
    test_accys = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            ouputs = model(images)
            loss = loss_func(ouputs, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}], Mean Loss: {epoch_loss / len(train_dataloader):.4f}"
        )
        mean_train_losses.append(epoch_loss / len(train_dataloader))
        test_accys.append(evaluate(model, test_dataloader))
    return mean_train_losses, test_accys


def evaluate(model: nn.Module, dataloader: DataLoader):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    # LLMs want me to noramlize as well
    # transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std (qwen)
    # transforms.Normalize((0.5,), (0.5,)) (chatgpt)

    # seems to be the well-known way to do things
    # https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset

    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # 28 x 28 pixel images
    input_dim = 28 * 28
    hidden_dims = [128, 64]
    # 10 digits (0- 9)
    output_dim = 10

    func_names = []
    train_args_list = []
    for act_func in [ReLU, Adonis, DeluLU, DeluLUv2]:
        try:
            func_name = act_func.name()
        except AttributeError:
            func_name = str(act_func().__class__)
        func_names.append(func_name)
        model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation_func=act_func,
        )
        train_args_list.append((model, train_loader, test_loader, 10))

    with Pool(cpu_count()) as pool:
        results = pool.starmap(train, train_args_list)

    train_fig = go.Figure()
    train_fig.update_layout(
        yaxis_title="Mean Training Loss",
        xaxis_title="Epoch",
    )
    train_fig.update_layout(
        title_text="Training loss on MNIST for different activation functions"
    )

    test_fig = go.Figure()
    test_fig.update_layout(
        yaxis_title="Accuracy",
        xaxis_title="Epoch",
    )
    test_fig.update_layout(
        title_text="Test accuracy on MNIST for different activation functions"
    )
    test_fig.update
    for result, func_name in zip(results, func_names):
        train_losses, test_accys = result
        train_fig.add_trace(
            go.Scatter(x=list(range(len(train_losses))), y=train_losses, name=func_name)
        )
        test_fig.add_trace(
            go.Scatter(x=list(range(len(test_accys))), y=test_accys, name=func_name)
        )
    train_fig.show()
    test_fig.show()

