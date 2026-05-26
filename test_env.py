import torch, torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from uci_dataset_loader import load_uci_batch, build_physical_adjacency, convert_to_pyg_graphs, convert_to_pyg_graphs_pure_attention
from model import RobustDriftGNN
from losses import jda_loss_function, multi_scale_jda_loss, multi_scale_wasserstein_loss, wasserstein_jda_loss, sinkhorn_distance

dev = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("All imports OK")

source = "Dataset/batch1.dat"
target = "Dataset/batch2.dat"
Xs, ys, scaler = load_uci_batch(source)
Xt, yt, _ = load_uci_batch(target, scaler=scaler)
Rs = torch.tensor(build_physical_adjacency(Xs), dtype=torch.float32)
print(f"Source: {Xs.shape}, Target: {Xt.shape}")

model = RobustDriftGNN(num_node_features=24, hidden_dim=64, embed_dim=32, num_classes=6, heads_layer1=8).to(dev)
ds = convert_to_pyg_graphs(Xs, ys, Rs, domain_id=0)
loader = DataLoader(ds, batch_size=32, shuffle=True)
batch = next(iter(loader)).to(dev)
logits, emb, _ = model(batch)
print(f"Forward pass: logits {logits.shape}, emb {emb.shape}")
print("All systems GO!")
