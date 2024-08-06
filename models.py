import torch.nn as nn
import torch.nn.functional as F
import torch
import torchsummary
import numpy as np
import math

def gen_pe(max_length, d_model, n=10000):

  # generate an empty matrix for the positional encodings (pe)
  pe = np.zeros(max_length*d_model).reshape(max_length, d_model) 

  # for each position
  for k in np.arange(max_length):

    # for each dimension
    for i in np.arange(d_model//2):

      # calculate the internal value for sin and cos
      theta = k / (n ** ((2*i)/d_model))       

      # even dims: sin   
      pe[k, 2*i] = math.sin(theta) 

      # odd dims: cos               
      pe[k, 2*i+1] = math.cos(theta)

  return pe

class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.linear_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(len(sizes)-1):
            linear = nn.Linear(sizes[i], sizes[i+1])
            nn.init.xavier_uniform_(linear.weight)
            self.linear_layers.append(linear)
            self.batch_norms.append(nn.BatchNorm1d(sizes[i+1]))

    def forward(self, x):
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            if i == len(self.linear_layers)-1:
                break
            x = F.selu(x)
            # x = self.batch_norms[i](x)
        return F.softmax(x, -1)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_1 = nn.LSTM(input_size=52, hidden_size=128, batch_first=True, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.linear_1 = nn.Linear(128, 300)
        self.dropout = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(300, 18)
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=1)
        # for name, param in self.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_uniform_(param)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        # x = F.tanh(x)
        x, _ = self.lstm_2(x)
        # x = F.tanh(x)
        x = x[:,-1,:]
        x = self.linear_1(x)
        x = self.selu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return self.softmax(x)
    
class TEPTransformer(nn.Module):
    def __init__(self, input_size, num_classes, sequence_length, embedding_dim, nhead, num_layers):
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.num_classes = num_classes
        self.pe_matrix = nn.Parameter(torch.Tensor(gen_pe(sequence_length, embedding_dim)), requires_grad=False)
        self.cls_token = nn.Parameter(torch.Tensor(np.random.default_rng(seed=42).standard_normal((embedding_dim,))), requires_grad=False)
        self.fc_encoding = nn.Linear(input_size, embedding_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_dim, nhead=nhead, dim_feedforward=embedding_dim*4, batch_first=True), num_layers=num_layers)
        self.fc_decoding = nn.Linear(embedding_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.fc_encoding(x) + self.pe_matrix
        shape = list(x.shape)
        shape[-2] = 1
        x = torch.concat((self.cls_token.reshape(1,1,-1).broadcast_to(shape), x), dim=-2)
        x = self.encoder(x)
        x = self.fc_decoding(x[:,0,:])
        return self.softmax(x)
    
if __name__ == "__main__":
    model = TEPTransformer(52, 18, 20, 512, 8, 6)
    # torchsummary.summary(model, (32,20,52))

    pred = model(torch.randn([32,20,52]))
    print(pred)
    print(torch.sum(pred, dim=-1))
    print(pred.shape)