import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class Delulu(nn.Module):
    def forward(self, x):
        return x.abs()


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


def train(model: nn.Module, epochs: int, dataloader: DataLoader):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            ouputs = model(images)
            loss = loss_func(ouputs, labels)
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}], Mean Loss: {epoch_loss / len(train_loader):.4f}"
        )


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

    print(f"Accuracy: {100 * correct / total:.2f}%")


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

    old_and_busted = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation_func=nn.ReLU,
    )
    new_hotness = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation_func=Delulu,
    )

    print(old_and_busted)
    print(new_hotness)

    print("OLD AND BUSTED")
    train(old_and_busted, 5, train_loader)
    evaluate(old_and_busted, test_loader)

    print("\nNEW HOTNESS")
    train(new_hotness, 5, train_loader)
    evaluate(new_hotness, test_loader)
