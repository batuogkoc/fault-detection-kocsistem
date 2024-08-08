import pyreadr
import torch
import numpy as np

name = "TEP_Faulty_Training"

data = pyreadr.read_r(f"dataset/RData/{name}.RData")
df = data[list(data.keys())[0]]

data_np = df.to_numpy()
data_torch = torch.tensor(data_np)
torch.save(data_torch, f"dataset/Torch/{name}.torch")