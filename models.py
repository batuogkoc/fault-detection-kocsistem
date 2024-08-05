import torch.nn as nn
import torch.nn.functional as F
import torch
import torchsummary

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

        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        x, _ = self.lstm_1(x)
        x = F.tanh(x)
        x, _ = self.lstm_2(x)
        x = F.tanh(x)
        x = x[:,-1,:]
        x = self.linear_1(x)
        x = F.selu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return F.softmax(x, -1)
    
if __name__ == "__main__":
    model = LSTM()
    # torchsummary.summary(model, (32,20,52))

    pred = model(torch.randn([32,20,52]))
    print(pred)
    print(torch.sum(pred, dim=-1))
    print(pred.shape)