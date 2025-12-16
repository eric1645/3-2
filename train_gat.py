import os
import math
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from dataset_gat import AirfRANSGATDataset
from model_gat import SimpleGAT
from dataset_gat import Normalizer

def train():
    root = r"C:\airfran\Dataset"
    task = "scarce"         
    batch_size = 4
    epochs = 50
    lr = 5e-4

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "gat_best.pt")
    norm_path  = os.path.join(save_dir, "normalizer.pt")

    train_ds = AirfRANSGATDataset(root, task=task, train=True)
    val_ds   = AirfRANSGATDataset(root, task=task, train=False)

    train_ds.restore()
    val_ds.restore()

    x_norm = Normalizer()
    y_norm = Normalizer()

    for g in train_ds:
        x_norm.partial_fit(g.x)
        y_norm.partial_fit(g.y)

    x_norm.finalize()
    y_norm.finalize()

    for g in train_ds:
        g.x = x_norm.encode(g.x)
        g.y = y_norm.encode(g.y)

    for g in val_ds:
        g.x = x_norm.encode(g.x)
        g.y = y_norm.encode(g.y)

    torch.save(
        {
            "x_mean": x_norm.mean,
            "x_std":  x_norm.std,
            "y_mean": y_norm.mean,
            "y_std":  y_norm.std,
        },
        norm_path
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGAT(
        in_channels=7,
        hidden_channels=16,
        out_channels=4,
        heads=4,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss()

    best_val = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index)
                val_loss += criterion(pred, batch.y).item()
        val_loss /= len(val_loader)

        print(f"[{epoch:03d}] Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)

    print("Model saved to:", model_path)
    print("Normalizer saved to:", norm_path)

if __name__ == "__main__":
    train()
