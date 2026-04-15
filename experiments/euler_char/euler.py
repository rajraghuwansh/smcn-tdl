# experiments/euler_char/euler.py
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from dataset import EulerCharacteristicDataset
from model import EulerSMCN


class Config:
    def __init__(self):
        self.hidden_dim = 64
        self.num_layers = 4
        self.number_of_mlp_layers = 2
        self.dropout = 0.0
        self.in_dropout = 0.0
        self.residual = False
        self.activation = "relu"
        self.high_rank = 2
        self.output_ranks = [0, 2]
        self.lr = 0.001
        self.batch_size = 32
        self.epochs = 100
        self.seed = 0
        self.num_pairs = 1000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.y.size(0)

    return total_loss / len(loader.dataset)


def test(model, loader, device):
    model.eval()
    total_error = 0.0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            total_error += F.l1_loss(out, data.y).item() * data.y.size(0)

    return total_error / len(loader.dataset)


def split_pairwise(dataset, train_ratio: float = 0.8):
    num_pairs = len(dataset) // 2
    train_pairs = int(train_ratio * num_pairs)
    train_cutoff = 2 * train_pairs
    return dataset[:train_cutoff], dataset[train_cutoff:]


if __name__ == "__main__":
    config = Config()
    torch.manual_seed(config.seed)

    dataset = EulerCharacteristicDataset(
        root="./data/EulerChar",
        num_pairs=config.num_pairs,
        seed=config.seed,
    )
    train_dataset, test_dataset = split_pairwise(dataset)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "follow_batch": ["x_0", "x_1", "x_2"],
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = EulerSMCN(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print("Starting training for Euler characteristic prediction...")
    for epoch in range(1, config.epochs + 1):
        loss = train(model, train_loader, optimizer, config.device)
        test_mae = test(model, test_loader, config.device)
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d}, Train MSE Loss: {loss:.4f}, "
                f"Test Mean Absolute Error: {test_mae:.4f}"
            )
