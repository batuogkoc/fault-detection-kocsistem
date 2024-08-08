import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pyreadr
# import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

def split_by_run(dataset):
    # dataset = dataset.T
    sample_count = int(torch.max(dataset[:,2]).item())
    print(sample_count)
    dataset = torch.reshape(dataset, (-1, sample_count, dataset.shape[-1]))
    return dataset

class TEPSingleSampleDataset(Dataset):
    def __init__(self, path):
        self.dataset = torch.load(path, weights_only=True)
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.scaler.fit(self.dataset[:,3:])
        self.one_hot_encoder.fit(self.dataset[:,0].reshape(-1,1))

        self.x = torch.FloatTensor(self.scaler.transform(self.dataset[:,3:]))
        self.y = torch.FloatTensor(self.one_hot_encoder.transform(self.dataset[:,0].reshape(-1,1)))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class TEPSlidingWindowDataset(Dataset):
    def __init__(self, stride, width, faultfree_path, faulty_path):
        self.stride = stride
        self.width = width
        faultfree = torch.load(faultfree_path, weights_only=True)
        faulty = torch.load(faulty_path, weights_only=True)
        self.dataset = torch.concat((faultfree, faulty))
        self._range = range(0,len(self.dataset)-width, stride)

        self.scaler = StandardScaler()
        self.scaler.fit(self.dataset[:,3:])
    
    def __len__(self):
        return len(self._range)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.scaler.transform(self.dataset[self._range[idx]:self._range[idx]+self.width,3:])), F.one_hot(self.dataset[self._range[idx]:self._range[idx]+self.width,0].type(torch.LongTensor), 21).type(torch.FloatTensor)

# class TEPSlidingWindowWaveletDataset(TEPSlidingWindowDataset):
#     def __getitem__(self, idx):
#         x, y = super(TEPSlidingWindowDataset)[idx]
#         wavelet = "cmor1.5-1.5"
#         return x 

if __name__ == "__main__":
    dataset = TEPSingleSampleDataset("dataset/train_reduced.torch")
    # dataset = TEPSlidingWindowDataset(100, 1000, "dataset/Torch/TEP_FaultFree_Training.torch", "dataset/Torch/TEP_Faulty_Training.torch")
    print(len(dataset))
    # x, y = dataset[0]

    with open("dataset/dataset.npy", "rb") as f:
        x_train = torch.Tensor(np.load(f))
        y_train = torch.Tensor(np.load(f))
        x_val = torch.Tensor(np.load(f))
        y_val = torch.Tensor(np.load(f))
        
    train_set = TensorDataset(x_train, y_train)
    val_set = TensorDataset(x_val, y_val)
    print(len(train_set))
    # print(x)
    # print(x.dtype)
    # print(y.dtype)
    # print(x.shape)
    # print(y.shape)
    # for x, y in dataset:
    #     pass