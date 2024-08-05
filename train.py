import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import time
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from models import *
from datasets import *

class RunningAverageLogger():
    def __init__(self, initial_value=0):
        self.initial_value = initial_value
        self.reset()
    
    def reset(self):
        self._count = 0
        self._running_sum = self.initial_value
    
    def get_avg(self):
        if self._count == 0:
            return self._running_sum
        return float(self._running_sum) / float(self._count)

    def add_value(self, value):
        self._running_sum += float(value)
        self._count += 1
    
    def count(self):
        return self._count

def train_classifier(model, 
                     optimizer, 
                     scheduler, 
                     loss_fn,
                     epoch_range, 
                     train_loader, 
                     val_loader, 
                     printing=True,
                     logging=True):
    
    running_average_training_loss_logger = RunningAverageLogger()
    running_average_training_accuracy_logger = RunningAverageLogger()   

    test_loss_logger = RunningAverageLogger()
    test_accuracy_logger = RunningAverageLogger()
    
    model.to(device)
    for epoch in epoch_range:
        if printing:
            print("-"*5 + f"EPOCH: {epoch}" + "-"*5)
        start = time.time()

        running_average_training_loss_logger.reset()
        running_average_training_accuracy_logger.reset()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            model.train()

            y_pred = model(x)

            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_average_training_loss_logger.add_value(loss.item())
            running_average_training_accuracy_logger.add_value(int(torch.argmax(y)==torch.argmax(y_pred)))

            if i % 200 == 0 and i != 0:
                fraction_done = max(i/len(train_loader), 1e-6)
                time_taken = (time.time()-start)
                if printing:
                    print(f"i: {i}| loss: {loss} | ratl: {running_average_training_loss_logger.get_avg()} | rata: {running_average_training_accuracy_logger.get_avg()}")
                    print(f"{fraction_done*100}% | est time left: {time_taken*(1-fraction_done)/fraction_done} s | est total: {time_taken/fraction_done} s")
                if logging:
                    writer.add_scalar("running_average_training_loss", running_average_training_loss_logger.get_avg(), epoch*len(train_loader) + i)
                    writer.add_scalar("running_average_training_accuracy", running_average_training_accuracy_logger.get_avg(), epoch*len(train_loader) + i)
                
            if logging and i%10000 == 0 and i!=0:
                torch.save({
                    "epoch": epoch,
                    "epoch_progress": i,
                    "epoch_size": len(train_loader),
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "running_average_training_loss": running_average_training_loss_logger.get_avg(),
                    "running_average_training_accuracy": running_average_training_accuracy_logger.get_avg(),
                }, os.path.join(CHECKPOINT_FOLDER, f"e-{epoch}-i-{i}-mbtl-{loss.item()}-rata-{running_average_training_accuracy_logger.get_avg()}.pt"))

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

        if logging:
            writer.add_scalar("train_loss", running_average_training_loss_logger.get_avg(), epoch)
            writer.add_scalar("test_loss", test_loss_logger.get_avg(), epoch)
            writer.add_scalar("train_accuracy", running_average_training_accuracy_logger.get_avg(), epoch)
            writer.add_scalar("test_accuracy", test_accuracy_logger.get_avg(), epoch)
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "training_loss": running_average_training_loss_logger.get_avg(),
                "test_loss": test_loss_logger.get_avg(),
                "training_accuracy": running_average_training_accuracy_logger.get_avg(),
                "test_accuracy" : test_accuracy_logger.get_avg(),
            }, os.path.join(CHECKPOINT_FOLDER, f"e-{epoch}-train_l-{running_average_training_loss_logger.get_avg()}-test_l-{test_loss_logger.get_avg()}-train_a-{running_average_training_accuracy_logger.get_avg()}-test_a-{test_accuracy_logger.get_avg()}.pt"))
    
    return model, running_average_training_accuracy_logger.get_avg(), test_accuracy_logger.get_avg()

def hyperparameter_search(train_loader, val_loader):
    params = [[256, 256],
              [256, 128, 128],
              [1024, 512, 256]]
    for i, param in enumerate(params):
        model = MLP([52] + param + [21])
        loss_fn = nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.3)
        _, train_acc, test_acc = train_classifier(model, optimizer, scheduler, loss_fn, range(0,1), train_loader, val_loader, printing=True, logging=False)
        writer.add_scalar("train_accuracy", train_acc, i)
        writer.add_scalar("test_accuracy", test_acc, i)


if __name__ == "__main__":
    KUACC=True
    RECORD=True
    if KUACC:
        print("-"*10 + "~KUACC~" + "-"*10)
        device = torch.device("cuda")
    else:
        torch.manual_seed(42)
        device = torch.device("cpu")

    print(f"Using device: {device.type}")

    print("Loading datasets...")
    DATASET_FOLDER_PATH = "dataset"

    with open("dataset/dataset.npy", "rb") as f:
        x_train = torch.Tensor(np.load(f))
        y_train = torch.Tensor(np.load(f))
        x_val = torch.Tensor(np.load(f))
        y_val = torch.Tensor(np.load(f))
        
    train_set = TensorDataset(x_train, y_train)
    val_set = TensorDataset(x_val, y_val)
    # train_set = TEPSingleSampleDataset(os.path.join(DATASET_FOLDER_PATH, "train_reduced.torch"))
    # val_set = TEPSingleSampleDataset(os.path.join(DATASET_FOLDER_PATH, "val_reduced.torch"))

    NUM_WORKERS = 1
    SHUFFLE = True
    train_loader = DataLoader(train_set, batch_size=128, num_workers=NUM_WORKERS, shuffle=SHUFFLE)
    val_loader = DataLoader(val_set, batch_size=128, num_workers=NUM_WORKERS, shuffle=SHUFFLE)

    START_EPOCH = 0
    print("Setting up model, optim, etc...")
    model = MLP([52, 100, 100, 18])
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.3)
    scheduler = None
    
    # LOAD_PROGRESS_PATH = "runs/2024-06-20T15:23:33/e-8-train_l-813.9820446734548-test_loss-869.9771018757278.pt"

    LOAD_PROGRESS_PATH = None
    if LOAD_PROGRESS_PATH:
        print("Loading checkpoint...")
        CHECKPOINT_FOLDER, _ = os.path.split(LOAD_PROGRESS_PATH)
        path_components = os.path.normpath(LOAD_PROGRESS_PATH).split(os.sep)
        EXPERIMENT_DATE_TIME = path_components[1]
        state = torch.load(LOAD_PROGRESS_PATH, map_location=device)
        if "epoch" in state and not "epoch_progress" in state:
            START_EPOCH = state["epoch"] + 1
        else:
            assert False, "Must start progress from a finished epoch"
        
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optim_state_dict"])
        if state["scheduler_state_dict"]:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        print(state.keys())
    else:
        print("Starting fresh run...")
        EXPERIMENT_DATE_TIME = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        CHECKPOINT_FOLDER = f"runs/{EXPERIMENT_DATE_TIME}"

    if RECORD:
        writer = SummaryWriter(f'runs_tensorboard/{EXPERIMENT_DATE_TIME}')
        os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

    # hyperparameter_search(train_loader, val_loader)
    train_classifier(model, optimizer, scheduler, loss_fn, range(START_EPOCH, 200), train_loader, val_loader, logging=RECORD)
