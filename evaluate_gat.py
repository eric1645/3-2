import torch
import time
from torch import nn
from torch_geometric.loader import DataLoader

from dataset_gat import AirfRANSGATDataset
from model_gat import SimpleGAT

def evaluate():
    root = r"C:\airfran\Dataset"
    task = "full"

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
        hidden_channels=8,
        out_channels=4,
        heads=4,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.MSELoss()
    mse = 0.0

    total_infer_time = 0.0

    all_pred = []
    all_gt   = []

    with torch.no_grad():
        if device.type == "cuda":
            dummy = next(iter(loader)).to(device)
            dummy.x = encode_x(dummy.x)
            for _ in range(20):
                _ = model(dummy.x, dummy.edge_index)
            torch.cuda.synchronize()

        for batch in loader:
            batch = batch.to(device)
            batch.x = encode_x(batch.x)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred = model(batch.x, batch.edge_index)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_infer_time += (t1 - t0)

            pred_phys = decode_y(pred)          
            gt_phys   = batch.y                 

            all_pred.append(pred_phys.cpu())
            all_gt.append(gt_phys.cpu())

            mse += criterion(pred, (gt_phys - y_mean) / y_std).item() 
        


    mse /= len(loader)
    avg_infer_time = total_infer_time / len(loader)

    save_path = "checkpoints/gat_test_predictions.pt"
    torch.save(
        {"pred": all_pred, "gt": all_gt},
        save_path
    )
    print("Saved GAT predictions to:", save_path)


    print(f"Test MSE            : {mse:.6e}")
    print(f"Avg inference time  : {avg_infer_time*1000:.3f} ms / sample")
    print(f"Total inference time: {total_infer_time:.3f} s")


if __name__ == "__main__":
    evaluate()
