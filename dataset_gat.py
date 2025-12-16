import os, json, copy
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
from airfrans.simulation import Simulation
from tqdm import tqdm

class AirfRANSGATDataset(Dataset):
    def __init__(self, root, task='scarce', train=True, k=6):
        self.root = root
        self.task = task
        self.train = train
        self.k = k

        tasks = ['full', 'scarce', 'reynolds', 'aoa']
        if task not in tasks:
            raise ValueError(f"Expected 'task' to be in {tasks}, got '{task}'")

        taskk = 'full' if task == 'scarce' and not train else task
        split = 'train' if train else 'test'

        with open(os.path.join(root, 'manifest.json'), 'r') as f:
            self.names = json.load(f)[f"{taskk}_{split}"]

        self.graphs = []
        self.raw_graphs = []
        transform = T.KNNGraph(k)

        for name in tqdm(self.names, desc=f'Loading {taskk}-{split}'):
            sim = Simulation(root=root, name=name)

            inlet = (
                np.array([np.cos(sim.angle_of_attack),
                          np.sin(sim.angle_of_attack)])
                * sim.inlet_velocity
            ).reshape(1, 2) * np.ones_like(sim.sdf)

            attr = np.concatenate([
                sim.position,
                inlet,
                sim.sdf,
                sim.normals,
                sim.velocity,
                sim.pressure,
                sim.nu_t,
                sim.surface.reshape(-1, 1)
            ], axis=-1)

            data = torch.tensor(attr, dtype=torch.float32)
            x = data[:, :7]
            y = data[:, 7:11]

            graph = Data(
                x=x,
                y=y,
                pos=torch.tensor(sim.position, dtype=torch.float32),
                name=name
            )
            graph = transform(graph)

            self.graphs.append(graph)
            self.raw_graphs.append(copy.deepcopy(graph))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def restore(self):
        self.graphs = [copy.deepcopy(g) for g in self.raw_graphs]

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        self.count = 0
        self.sq_sum = None

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
        var = self.sq_sum / self.count - self.mean ** 2
        self.std = torch.sqrt(var + 1e-8)

    def encode(self, tensor):
        return (tensor - self.mean) / self.std

    def decode(self, tensor):
        return tensor * self.std + self.mean

