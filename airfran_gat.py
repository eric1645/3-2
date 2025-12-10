import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from airfrans.simulation import Simulation
import numpy as np
import json
from torch_geometric.data import Data
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import math
import copy
import wandb

class AirfRANSGATDataset(Dataset):
    def __init__(self, root, task='scarce', train=True, k=6):
        self.root = root
        self.task = task
        self.train = train
        self.k = k

        tasks = ['full', 'scarce', 'reynolds', 'aoa']
        if task not in tasks:
            raise ValueError(f"Expected 'task' to be in {tasks} "
                             f"got '{task}'")
        taskk = 'full' if task == 'scarce' and not train else task
        split = 'train' if train else 'test'

        with open(os.path.join(root, 'manifest.json'), 'r') as f:
            manifest = json.load(f)[f"{taskk}_{split}"]

        self.names = manifest
        self.graphs = []
        self.raw_graphs = []  # 원본 보관

        transform = T.KNNGraph(k)

        for name in tqdm(manifest, desc=f'Loading AirfRANS ({taskk}, {split})'):
            simulation = Simulation(root=root, name=name)
            inlet_velocity = (np.array([np.cos(simulation.angle_of_attack),
                                        np.sin(simulation.angle_of_attack)]) *
                              simulation.inlet_velocity).reshape(1, 2) \
                             * np.ones_like(simulation.sdf)

            attribute = np.concatenate([
                simulation.position,
                inlet_velocity,
                simulation.sdf,
                simulation.normals,
                simulation.velocity,
                simulation.pressure,
                simulation.nu_t,
                simulation.surface.reshape(-1, 1)
            ], axis=-1)

            data = torch.tensor(attribute, dtype=torch.float32)
            x = data[:, :7]
            y = data[:, 7:11]

            graph = Data(x=x, y=y, pos=torch.tensor(simulation.position, dtype=torch.float32))
            graph = transform(graph)

            self.graphs.append(graph)
            self.raw_graphs.append(copy.deepcopy(graph))  # 원본 저장

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        graph.name = self.names[idx]
        return graph

    def restore(self):
        """run_training 시작할 때 원본으로 완전 복구"""
        self.graphs = [copy.deepcopy(g) for g in self.raw_graphs]


# ==========================================================
# Normalizer
# ==========================================================
class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.count = 0

    def partial_fit(self, tensor):
        if self.mean is None:
            self.mean = tensor.sum(dim=0)
            self.sq_sum = (tensor ** 2).sum(dim=0)
        else:
            self.mean += tensor.sum(dim=0)
            self.sq_sum += (tensor ** 2).sum(dim=0)

        self.count += tensor.size(0)

    def finalize(self):
        self.mean = self.mean / self.count
        var = self.sq_sum / self.count - self.mean**2
        self.std = torch.sqrt(var + 1e-8)

    def encode(self, tensor):
        return (tensor - self.mean) / self.std

    def decode(self, tensor):
        return tensor * self.std + self.mean


# ==========================================================
# GAT 모델
# ==========================================================
class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                             concat=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_channels * heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads,
                             concat=True, dropout=dropout)
        
        self.conv3 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, concat=False, dropout=dropout)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return x


# ==========================================================
# Training utils
# ==========================================================
def create_dataloaders(train_dataset, val_dataset, batch_size=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    mse_list = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        mse = criterion(out, batch.y).item()
        mse_list.append(mse)
    return sum(mse_list) / len(mse_list)


# ==========================================================
# Main training (Sweep 반복에도 데이터셋 재사용 가능)
# ==========================================================
def run_training(train_dataset, val_dataset, save_path="gat.pt"):

    # ★ 원본 그래프를 복구 (정규화 다시 적용 가능)
    train_dataset.restore()
    val_dataset.restore()

    wandb.init(
        project="airfrans_gat",
        config={
            "hidden_channels": 16,
            "num_heads": 4,
            "dropout": 0.1,
            "lr": 1e-3,
            "batch_size": 4,
        }
    )
    config = wandb.config

    hidden_channels = config.hidden_channels
    num_heads = config.num_heads
    dropout = config.dropout
    lr = config.lr
    batch_size = config.batch_size

    x_norm = Normalizer()
    y_norm = Normalizer()

    for g in train_dataset:
        x_norm.partial_fit(g.x)
        y_norm.partial_fit(g.y)

    x_norm.finalize()
    y_norm.finalize()

    for g in train_dataset:
        g.x = x_norm.encode(g.x)
        g.y = y_norm.encode(g.y)
    for g in val_dataset:
        g.x = x_norm.encode(g.x)
        g.y = y_norm.encode(g.y)

    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGAT(
        in_channels=7,
        hidden_channels=hidden_channels,
        out_channels=4,
        heads=num_heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss()
    best_val = math.inf

    for epoch in range(1, 31):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mse = evaluate(model, val_loader, criterion, device)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_mse": val_mse})
        print(f"[{epoch:03d}] Train {train_loss:.5f} | Val MSE {val_mse:.5f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), save_path)

    print("학습 완료:", save_path)
    return model


# ==========================================================
# 실행 (Dataset은 1번만 로드)
# ==========================================================
if __name__ == "__main__":
    root = r"C:\airfran\Dataset"
    train_dataset = AirfRANSGATDataset(root, task='full', train=True,  k=6)
    val_dataset   = AirfRANSGATDataset(root, task='full', train=False, k=6)
    run_training(train_dataset, val_dataset)

