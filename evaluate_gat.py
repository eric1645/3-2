import torch
from torch import nn
from torch_geometric.loader import DataLoader

from dataset_gat import AirfRANSGATDataset
from model_gat import SimpleGAT

def evaluate():
    root = r"C:\airfran\Dataset"
    task = "scarce"

    model_path = "checkpoints/gat_best.pt"
    norm_path  = "checkpoints/normalizer.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = AirfRANSGATDataset(root, task=task, train=False)
    test_ds.restore()

    loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    norm = torch.load(norm_path, map_location=device)
    x_mean = norm["x_mean"].to(device)
    x_std  = norm["x_std"].to(device)
    y_mean = norm["y_mean"].to(device)
    y_std  = norm["y_std"].to(device)

    def encode_x(x):
        return (x - x_mean) / x_std

    def decode_y(y):
        return y * y_std + y_mean

    model = SimpleGAT(
        in_channels=7,
        hidden_channels=16,
        out_channels=4,
        heads=4,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.MSELoss()
    mse = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x = encode_x(batch.x)

            pred = model(batch.x, batch.edge_index)
            pred_phys = decode_y(pred)

            mse += criterion(pred, batch.y).item()

    mse /= len(loader)
    print("Test MSE:", mse)


if __name__ == "__main__":
    evaluate()
