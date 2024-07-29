import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyreadr

class TEPSingleSampleDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = torch.load(file_path, weights_only=True)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx,3:], self.dataset[idx,0]
    

if __name__ == "__main__":
    dataset = TEPSingleSampleDataset("dataset\Torch\TEP_Faulty_Testing.torch")

    for x, y in dataset:
        pass