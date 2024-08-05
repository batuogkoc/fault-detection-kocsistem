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
    
if __name__ == "__main__":
    model = MLP([52,100, 100,10])
    torchsummary.summary(model, (1,52))

    pred = model(torch.randn([10,52]))
    print(pred)
    print(torch.sum(pred, dim=-1))
    print(pred.shape)