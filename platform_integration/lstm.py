import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import pandas as pd
import torch.utils
from base import BaseDetector
from typing import Union
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from py_utils import *
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class LSTM(BaseDetector):
    """LSTM based 1d anomaly detector"""
    def __init__(self,
                 contamination: float = 0.1,
                 anomaly_threshold: Union[float, str] = "auto",
                 model_mp_quantile: float = 0.02,
                 feature_rp: str = None,
                 model_rp: str = "60min",
                 range_filter: bool = True,
                 range_rp: str = None,
                 raw_risk_rp: str = None,
                 risk_rp: str = None,
                 run_idx_column:str="Run",
                 timestep_column:str="Time",
                 feature_column:str="Value",
                 y_label_column:str="YLabel",
                 train_fraction:float=0.8,
                 batch_size:int=128,
                 sequence_len:int=20,
                 stride:int=10,
                 hidden_size=128,
                 max_epoch_num:int=25,
                 device=None,
                 loss_fn=None,
                 optimizer:torch.optim.Optimizer=None,
                 scheduler:torch.optim.lr_scheduler.LRScheduler=None
                 ):
        
        #hyperparameters
        self.run_idx_column = run_idx_column
        self.timestep_column = timestep_column
        self.feature_column = feature_column
        self.y_label_column = y_label_column
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.stride = stride
        self.max_epoch_num = max_epoch_num

        self.contamination = contamination
        self.anomaly_threshold = anomaly_threshold
        self.model_mp_quantile = model_mp_quantile
        self.model_rp = pd.Timedelta(model_rp)
        self.feature_rp = pd.Timedelta(feature_rp) if feature_rp is not None else self.model_rp / 2
        self.range_filter = range_filter
        self.range_rp = pd.Timedelta(range_rp) if range_rp is not None else self.model_rp * 3 / 4
        self.raw_risk_rp = pd.Timedelta(raw_risk_rp) if raw_risk_rp is not None else self.model_rp * 2
        self.risk_rp = pd.Timedelta(risk_rp) if risk_rp is not None else self.model_rp * 6

        self.model_mp: int  
        self.feature_mp: int
        self.raw_risk_mp: int 
        self.range_limit: float

        self.model = _LSTM(input_size=1,
                           hidden_size=hidden_size,
                           num_classes=1,
                           bidirectional_layers_num=1,
                           unidirectional_layers_num=1,
                           custom_classification_head=None)
        self.loss_fn = loss_fn if loss_fn else nn.BCELoss()
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = scheduler

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        
    def fit(self, data:pd.DataFrame):
        self.train_set, self.val_set = self.generate_train_val_sets(data)

        train_loader = DataLoader(self.train_set, self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_set, self.batch_size, shuffle=True)

        LSTM.train_classifier(model=self.model,
                              optimizer=self.optimizer,
                              scheduler=self.scheduler,
                              loss_fn=self.loss_fn,
                              epoch_range=range(self.max_epoch_num),
                              train_loader=train_loader,
                              val_loader=val_loader,
                              device=self.device,
                              printing=False,
                              checkpoint_folder=None,
                              tensorboard_writer=None,
                              )

        return self
        
    def anomaly_score(self, data:pd.DataFrame):
        x_set = torch.Tensor(data.set_axis(self.timestep_column)[self.feature_column].to_numpy(), device=self.device)
        data["Raw_Anomaly_Score"] = 0
        with torch.inference_mode():
            self.model.eval()
            self.model.to(self.device)
            x_set = x_set.to(self.device)
            for start_idx in range(0, len(x_set)-self.sequence_len):
                x = x_set[start_idx:start_idx+self.sequence_len]
                data.loc[start_idx+self.sequence_len-1, "Raw_Anomaly_Score"] = self.model(x)
        
        # Apply range_filter
        if self.range_filter: data = self._range_filter(data, fit = False)

        # Apply rolling mean to get anomaly scores
        data["Anomaly_Score"] = data['Raw_Anomaly_Score'].rolling(self.model_rp, min_periods=self.model_mp).mean()

        df_anomaly = data["Anomaly_Score"].reset_index()
        return data["Anomaly_Score"].reset_index()
    

    def fit_min_periods(self, data: pd.DataFrame):
        """
        """
        data = data.set_index(self.timestep_column).sort_index()        
        data['count'] = data.rolling(self.model_rp).count()
        data['count'] = np.where(data['count'] == 0, np.nan, data['count'])
        self.model_mp = int(data['count'].quantile(self.model_mp_quantile))
        self.raw_risk_mp = self.model_mp 
        self.raw_risk_mp = int(self.model_mp * (self.raw_risk_rp / self.model_rp))

        return self


    def generate_x_y_sets(self, data:pd.DataFrame, stride, return_indices=False):
        data = data.dropna()
        x_segments = []
        y_segments = []

        for run_idx in data[self.run_idx_column].unique():
            data_run = data[data[self.run_idx_column] == run_idx].set_index(self.timestep_column)
            for start_time in range(0, len(data_run)-self.sequence_len, stride):
                x_segments.append(data_run.loc[start_time:start_time+self.sequence_len, self.feature_column].to_numpy().astype(np.float32))
                y_segments.append(data_run.loc[start_time+self.sequence_len, self.y_label_column].to_numpy().astype(np.float32))
        x_dataset = np.stack(x_segments, axis=0)
        y_dataset = np.stack(y_segments, axis=0)

        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x_dataset.reshape(-1, x_dataset.shape[-1])).reshape(x_dataset.shape)

        x_train, x_val, y_train, y_val = train_test_split(x_scaled, y_dataset, train_size=self.train_fraction)

        train_set = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        val_set = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))

        return train_set, val_set

    @staticmethod
    def train_classifier(model:nn.Module, 
                        optimizer:torch.optim.Optimizer, 
                        scheduler:torch.optim.lr_scheduler.LRScheduler, 
                        loss_fn,
                        epoch_range:range, 
                        train_loader:DataLoader, 
                        val_loader:DataLoader,
                        device:torch.device, 
                        printing:bool=True,
                        checkpoint_folder:None|str=None,
                        tensorboard_writer:None|SummaryWriter=None):
        
        running_average_training_loss_logger = RunningAverageLogger()
        running_average_training_accuracy_logger = RunningAverageLogger()   

        test_loss_logger = RunningAverageLogger()
        test_accuracy_logger = RunningAverageLogger()

        printer = InplacePrinter(2)
        
        model.to(device)
        for epoch in epoch_range:
            if printing:
                printer.reset()
                print("-"*5 + f"EPOCH: {epoch}" + "-"*5)
            start = time.time()

            running_average_training_loss_logger.reset()
            running_average_training_accuracy_logger.reset()
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                model.train()

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                
                loss.backward()
                optimizer.step()
                
                running_average_training_loss_logger.add_value(loss.item())
                running_average_training_accuracy_logger.add_value(float(torch.argmax(y)==torch.argmax(y_pred)))

                if i % 10 == 0 and i != 0:
                    fraction_done = max(i/len(train_loader), 1e-6)
                    time_taken = (time.time()-start)
                    if printing:
                        printer.print(f"e: {epoch} | i: {i} | loss: {loss:2.3f} | ratl: {running_average_training_loss_logger.get_avg():2.3f} | rata: {running_average_training_accuracy_logger.get_avg():.3f}")
                        printer.print(f"{fraction_done*100:2.2f}% | est time left: {time_taken*(1-fraction_done)/fraction_done:.1f} s | est total: {time_taken/fraction_done:.1f} s")
                    if tensorboard_writer:
                        tensorboard_writer.add_scalar("running_average_training_loss", running_average_training_loss_logger.get_avg(), epoch*len(train_loader) + i)
                        tensorboard_writer.add_scalar("running_average_training_accuracy", running_average_training_accuracy_logger.get_avg(), epoch*len(train_loader) + i)
                    
                if checkpoint_folder and i%10000 == 0 and i!=0:
                    torch.save({
                        "epoch": epoch,
                        "epoch_progress": i,
                        "epoch_size": len(train_loader),
                        "model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                        "running_average_training_loss": running_average_training_loss_logger.get_avg(),
                        "running_average_training_accuracy": running_average_training_accuracy_logger.get_avg(),
                    }, os.path.join(checkpoint_folder, f"e-{epoch}-i-{i}-mbtl-{loss.item()}-rata-{running_average_training_accuracy_logger.get_avg()}.pt"))

            test_loss_logger.reset()
            test_accuracy_logger.reset()
            with torch.inference_mode():
                for i, (x, y) in enumerate(val_loader):
                    x, y = x.to(device), y.to(device)

                    model.eval()
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    test_loss_logger.add_value(loss.item())
                    test_accuracy_logger.add_value(int(torch.argmax(y)==torch.argmax(y_pred)))

            if printing:
                print(f"train loss: {running_average_training_loss_logger.get_avg()} | test loss: {test_loss_logger.get_avg()}")
                print(f"train acc: {running_average_training_accuracy_logger.get_avg()} | test acc: {test_accuracy_logger.get_avg()}")
            if scheduler:
                scheduler.step()

            if tensorboard_writer:
                tensorboard_writer.add_scalar("train_loss", running_average_training_loss_logger.get_avg(), epoch)
                tensorboard_writer.add_scalar("test_loss", test_loss_logger.get_avg(), epoch)
                tensorboard_writer.add_scalar("train_accuracy", running_average_training_accuracy_logger.get_avg(), epoch)
                tensorboard_writer.add_scalar("test_accuracy", test_accuracy_logger.get_avg(), epoch)

            if checkpoint_folder:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "training_loss": running_average_training_loss_logger.get_avg(),
                    "test_loss": test_loss_logger.get_avg(),
                    "training_accuracy": running_average_training_accuracy_logger.get_avg(),
                    "test_accuracy" : test_accuracy_logger.get_avg(),
                }, os.path.join(checkpoint_folder, f"e-{epoch}-train_l-{running_average_training_loss_logger.get_avg()}-test_l-{test_loss_logger.get_avg()}-train_a-{running_average_training_accuracy_logger.get_avg()}-test_a-{test_accuracy_logger.get_avg()}.pt"))
        
        return model, running_average_training_accuracy_logger.get_avg(), test_accuracy_logger.get_avg()

class _LSTM(nn.Module):
    def __init__(self, 
                 input_size:int=1,
                 hidden_size:int=128,
                 num_classes:int=1,
                 bidirectional_layers_num:int=1,
                 unidirectional_layers_num:int=1,
                 custom_classification_head:nn.Sequential=None):
        super().__init__()
        # assert num_classes > 1, "Num classes must be 2 or greater. If you want to use this model as a binary fault detector, use 2 classes: \'fault\', \'no fault\' "
        
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
    
if __name__ == "__main__":
    model_lstm = LSTM()    