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
    def __init__(self, 
                 input_size:int=1,
                 hidden_size:int=128,
                 num_classes:int=1,
                 bidirectional_layers_num:int=1,
                 unidirectional_layers_num:int=1,
                 custom_classification_head:nn.Sequential=None):
        super().__init__()        
        # save hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.bidirectional_layers_num = bidirectional_layers_num
        self.unidirectional_layers_num = unidirectional_layers_num

        # generate lstm layers
        if bidirectional_layers_num == -1:
            self.bidirectional_lstm = None
            self.unidirectional_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=unidirectional_layers_num, batch_first=True)
        else:
            self.bidirectional_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=bidirectional_layers_num, batch_first=True, bidirectional=True)
            self.unidirectional_lstm = nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, num_layers=unidirectional_layers_num, batch_first=True)

        # if custom classification head isn't specified, use default head
        if custom_classification_head:
            self.classification_head = custom_classification_head
        else:
            self.classification_head = nn.Sequential(
                nn.Linear(hidden_size, 300),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Linear(300, num_classes),
            )
        if num_classes == 1:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)
        # for name, param in self.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        #     elif 'weight' in name:
        #         nn.init.xavier_uniform_(param)

    def forward(self, x):
        x, _ = self.bidirectional_lstm(x)
        x, _ = self.unidirectional_lstm(x)
        x = x[:,-1,:]
        x = self.classification_head(x)
        return self.final_activation(x)
    
class TEPTransformer(nn.Module):
    def __init__(self, input_size, num_classes, sequence_length, embedding_dim, nhead, num_layers):
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.num_classes = num_classes
        self.pe_matrix = nn.Parameter(torch.Tensor(gen_pe(sequence_length, embedding_dim)), requires_grad=False)
        # self.pe_matrix = nn.Parameter(torch.randn(sequence_length, embedding_dim))
        self.cls_token = nn.Parameter(torch.Tensor(np.random.default_rng(seed=42).standard_normal((embedding_dim,))), requires_grad=True)
        self.fc_encoding = nn.Linear(input_size, embedding_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embedding_dim, nhead=nhead, dim_feedforward=embedding_dim*4, batch_first=True), num_layers=num_layers)
        self.fc_decoding = nn.Linear(embedding_dim, num_classes)
        # self.fc_decoding = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(sequence_length*embedding_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_classes)
        # )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.fc_encoding(x) + self.pe_matrix
        # x = torch.concat((x,self.pe_matrix.broadcast_to(*x.shape[:-2], *self.pe_matrix.shape)), dim=-1)
        # x = self.fc_encoding(x)
        shape = list(x.shape)
        shape[-2] = 1
        x = torch.concat((self.cls_token.reshape(1,1,-1).broadcast_to(shape), x), dim=-2)
        x = self.encoder(x)
        x = self.fc_decoding(x[:,0,:])
        # print(x.shape)
        return self.softmax(x)
    
if __name__ == "__main__":
    model = TEPTransformer(52, 18, 20, 512, 8, 6)
    # torchsummary.summary(model, (32,20,52))

    pred = model(torch.randn([32,20,52]))
    # print(pred)
    print(torch.sum(pred, dim=-1))
    print(pred.shape)