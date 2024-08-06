import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from train import *
from models import *
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        
        # Bidirectional LSTM layer
        self.bidirectional_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=1, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 300)
        self.fc2 = nn.Linear(300, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Activation functions
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Bidirectional LSTM layer
        x, _ = self.bidirectional_lstm(x)
        
        # LSTM layer
        x, _ = self.lstm(x)
        
        # Take the output of the last LSTM cell
        x = x[:, -1, :]
        
        # Fully connected layers with activation and dropout
        x = self.selu(self.fc1(x))
        x = self.dropout(x)
        x = self.softmax(self.fc2(x))
        
        return x

with open("dataset/dataset_pm_benchmark.npy", "rb") as f:
    x_train = torch.from_numpy(np.load(f, allow_pickle=True).astype(np.float32))
    y_train = torch.from_numpy(np.load(f, allow_pickle=True).astype(np.float32))
    x_val = torch.from_numpy(np.load(f, allow_pickle=True).astype(np.float32))
    y_val = torch.from_numpy(np.load(f, allow_pickle=True).astype(np.float32))


# Instantiate the model, define the loss function and the optimizer
input_dim = x_train.shape[2]
hidden_dim = 128
output_dim = y_train.shape[1]

# model = LSTMModel(input_dim, hidden_dim, output_dim)
model = LSTM()
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming you have DataLoader for your training data
train_set = TensorDataset(x_train, y_train)
val_set = TensorDataset(x_val, y_val)
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
acc_logger = RunningAverageLogger() 
for epoch in range(num_epochs):
    acc_logger.reset()
    for i, (inputs, labels) in enumerate(train_loader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc_logger.add_value(float(torch.argmax(labels) == torch.argmax(outputs)))
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:  # Print every 100 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, {acc_logger.get_avg()}')
