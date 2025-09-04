import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
import plotly.graph_objects as go
import plotly.express as px
from torchvision import datasets
from torch.utils.data import DataLoader
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.stats import bootstrap

from load_penguins import load_penguins_datasets
from load_moons import load_moons_datasets

import warnings

warnings.simplefilter("error")


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    epochs: int


class ReLU(nn.Module):
    @staticmethod
    def name():
        return "ReLU"

    def forward(self, x):
        return torch.where(x >= 0, x, 0)


class GELU(nn.Module):
    @staticmethod
    def name():
        return "GELU"

    def forward(self, x):
        return nn.functional.gelu(x)


class SiLU(nn.Module):
    @staticmethod
    def name():
        return "SiLU"

    def forward(self, x):
        return nn.functional.silu(x)


class Adonis(nn.Module):
    @staticmethod
    def name():
        return "Adonis"

    def forward(self, x):
        return x.abs()


class Spongebob(nn.Module):
    @staticmethod
    def name():
        return "Spongebob"

    def forward(self, x):
        return x**2


class Spongebobv2(nn.Module):
    @staticmethod
    def name():
        return "Spongebob V2"

    def forward(self, x):
        return torch.sqrt(x.abs())


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
        return "DeluLU V2"

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
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    mean_train_losses = []
    test_accys = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data, labels in train_dataloader:
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            if torch.isnan(outputs).any():
                print(
                    "NaNs detected in model output. Searching for NaNs in model weights"
                )
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"{name}: NaNs detected")
                        print(param, end="\n\n")
                raise ValueError("NaNs detected in model output")
            # labels are 1D batch_size but torch CrossEntropyLoss automatically
            # interprets them as class indicies
            loss = loss_func(outputs, labels)
            if torch.isnan(loss).any():
                breakpoint()
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
            # clip gradients to an L2 norm of 1.0 to prevent NaNs (exploding gradients)
            # still getting some NaNs though
            # might have to clip individual values
            # instead of the norm
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

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
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def run_experiment(
    funcs: list[nn.Module],
    mlp_config: MLPConfig,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    name: str,
    num_trials: int = 5,
):
    train_fig = go.Figure()
    train_fig.update_layout(
        yaxis_title="Mean Training Loss",
        xaxis_title="Epoch",
    )
    train_fig.update_layout(
        title_text=f"Training loss on {name} for different activation functions ({num_trials=})"
    )

    test_fig = go.Figure()
    test_fig.update_layout(
        yaxis_title="Test Accuracy",
        xaxis_title="Epoch",
    )
    test_fig.update_layout(
        title_text=f"Test accuracy on {name} for different activation functions ({num_trials=})"
    )

    print(f"\nRunning {name}\n")
    for i, act_func in tqdm(enumerate(funcs), ascii=True, total=len(funcs)):
        train_losses = []
        test_accys = []
        train_args_list = []
        for trial in tqdm(range(num_trials), ascii=True, leave=False):
            model = MLP(
                input_dim=mlp_config.input_dim,
                hidden_dims=mlp_config.hidden_dims,
                output_dim=mlp_config.output_dim,
                activation_func=act_func,
            )
            train_args_list.append(
                (model, train_dataloader, test_dataloader, mlp_config.epochs)
            )
        with Pool(cpu_count()) as pool:
            results = pool.starmap(train, train_args_list)

        for result in results:
            train_losses.append(result[0])
            test_accys.append(result[1])

        train_losses = np.array(train_losses)
        test_accys = np.array(test_accys)

        mean_train, hi_train, lo_train = many_runs_to_mean_plus_ci(train_losses)
        mean_test, hi_test, lo_test = many_runs_to_mean_plus_ci(test_accys)

        try:
            func_name = act_func.name()
        except AttributeError:
            func_name = str(act_func().__class__)

        func_color = px.colors.qualitative.Plotly[i]
        train_losses, test_accys = result
        train_fig.add_trace(
            go.Scatter(
                x=list(range(len(mean_train))),
                y=mean_train.tolist(),
                name=func_name,
                line=dict(color=func_color),
            )
        )
        train_fig.add_trace(
            go.Scatter(
                x=list(range(len(hi_train))) + list(range(len(hi_train)))[::-1],
                y=hi_train.tolist() + lo_train.tolist()[::-1],
                name=f"{func_name} CI",
                fill="toself",
                fillcolor=hex_to_rgba(func_color, 0.2),
                line=dict(color=hex_to_rgba(func_color, 0)),
                hoverinfo="skip",
                showlegend=True,
            )
        )
        test_fig.add_trace(
            go.Scatter(
                x=list(range(len(mean_test))),
                y=mean_test.tolist(),
                name=func_name,
                line=dict(color=func_color),
            )
        )
        test_fig.add_trace(
            go.Scatter(
                x=list(range(len(hi_test))) + list(range(len(hi_test)))[::-1],
                y=hi_test.tolist() + lo_test.tolist()[::-1],
                name=f"{func_name} CI",
                fill="toself",
                fillcolor=hex_to_rgba(func_color, 0.2),
                line=dict(color=hex_to_rgba(func_color, 0)),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    train_fig.show()
    test_fig.show()


def many_runs_to_mean_plus_ci(data: np.array):
    """ "
    expects an array with shape (num_runs, run)
    """
    if np.isnan(data).any():
        raise ValueError("NaNs detected in the loss or accuracy data.")
    mean = np.mean(data, axis=0)
    bca_ci = bootstrap(
        (data,), statistic=np.mean, vectorized=True, axis=0, method="BCa"
    )
    hi = bca_ci.confidence_interval.high
    lo = bca_ci.confidence_interval.low
    return mean, hi, lo


def hex_to_rgba(h: str, alpha: float = 0.2):
    r = int(h[1:3], base=16)
    g = int(h[3:5], base=16)
    b = int(h[5:7], base=16)
    return f"rgba({r}, {g}, {b}, {alpha})"


if __name__ == "__main__":
    funcs = [ReLU, GELU, SiLU, Adonis, Spongebob, Spongebobv2, DeluLU, DeluLUv2]

    # scikit-learn moons
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
    # TODO: scale the input data and see how that impacts results
    train_moons, test_moons = load_moons_datasets()
    train_moons = DataLoader(train_moons, batch_size=64, shuffle=True)
    test_moons = DataLoader(test_moons, batch_size=1024)

    moons_model_config = MLPConfig(
        input_dim=2, hidden_dims=[64, 16], output_dim=2, epochs=10
    )

    run_experiment(
        funcs=funcs,
        mlp_config=moons_model_config,
        train_dataloader=train_moons,
        test_dataloader=test_moons,
        name="SKLearn Moons",
    )

    # palmer penguins
    # TODO: scale the input data and see how that impacts results
    train_penguins, test_penguins = load_penguins_datasets()
    train_penguins = DataLoader(train_penguins, batch_size=64, shuffle=True)
    test_penguins = DataLoader(test_penguins, batch_size=1024)

    penguins_model_config = MLPConfig(
        input_dim=7, hidden_dims=[16, 64, 16], output_dim=3, epochs=10
    )

    run_experiment(
        funcs=funcs,
        mlp_config=penguins_model_config,
        train_dataloader=train_penguins,
        test_dataloader=test_penguins,
        name="Palmer Penguins",
    )

    # MNIST
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            # seems to be the well-known way to do things
            # https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset
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
    test_loader = DataLoader(test_dataset, batch_size=1024)

    # input is 28 x 28 pixel images
    mnist_model_config = MLPConfig(
        input_dim=28 * 28, hidden_dims=[64, 16], output_dim=10, epochs=5
    )

    run_experiment(
        funcs=funcs,
        mlp_config=mnist_model_config,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        name="MNIST Handwritten Digits",
    )
