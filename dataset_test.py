import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyreadr

# class TEPSingleSampleDatasetR(Dataset):
#     def __init__(self, file_path):
#         self.file = pyreadr.read_r(file_path)
#         self.df = self.file[list(self.keys())[0]]

#     def __len__(self):
#         return self.df.shape[0]

#     def __getitem__(self, idx):
#         return torch.tensor(self.df.iloc[idx,3:], dtype=torch.float32), torch.tensor(self.df.iloc[idx,0], dtype=torch.float32)

class TEPSingleSampleDatasetTorch(Dataset):
    def __init__(self, file_path):
        self.dataset = torch.load(file_path, weights_only=True)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx,3:], self.dataset[idx,0]
    

if __name__ == "__main__":
    dataset = TEPSingleSampleDatasetTorch("dataset\Torch\TEP_Faulty_Testing.torch")

    for x, y in dataset:
        pass