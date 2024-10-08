import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchmetrics
import torch.nn as nn
import time
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from models import *
from py_utils import *

def train(
        model:nn.Module, 
        optimizer:torch.optim.Optimizer, 
        scheduler:torch.optim.lr_scheduler.LRScheduler, 
        loss_fn,
        epoch_range:range, 
        train_loader:DataLoader, 
        val_loader:DataLoader,
        device:torch.device, 
        printing:bool=True,
        metrics:dict[str, torchmetrics.Metric]={},
        checkpoint_folder:None|str=None,
        tensorboard_writer:None|SummaryWriter=None):
    
    running_average_training_loss_logger = RunningAverageLogger()
    val_loss_logger = RunningAverageLogger()

    printer = InplacePrinter(2+len(metrics))
    
    model.to(device)

    for metric in metrics.values():
        metric.to(device)
    
    for epoch in epoch_range:
        if printing:
            printer.reset()
            print("-"*5 + f"EPOCH: {epoch}" + "-"*5)
        start = time.time()

        running_average_training_loss_logger.reset()

        for metric in metrics.values():
            metric.reset()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            model.train()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            epoch_metric_values = {}
            for metric_name, metric in metrics.items():
                epoch_metric_values[metric_name] = metric(y_pred, y).item()

            loss.backward()
            optimizer.step()
            
            running_average_training_loss_logger.add_value(loss.item())

            if i % 10 == 0 and i != 0:
                fraction_done = max(i/len(train_loader), 1e-6)
                time_taken = (time.time()-start)
                if printing:
                    for metric_name, metric in metrics.items():
                        printer.print(f"{metric_name} : {epoch_metric_values[metric_name]}")
                    printer.print(f"e: {epoch} | i: {i} | loss: {loss.item():2.3f} | ratl: {running_average_training_loss_logger.get_avg():2.3f}")
                    printer.print(f"{fraction_done*100:2.2f}% | est time left: {time_taken*(1-fraction_done)/fraction_done:.1f} s | est total: {time_taken/fraction_done:.1f} s")
                if tensorboard_writer:
                    tensorboard_writer.add_scalar("running_average_training_loss", running_average_training_loss_logger.get_avg(), epoch*len(train_loader) + i)
                    for metric_name, metric in metrics.items():
                        tensorboard_writer.add_scalar(f"train_{metric_name}", epoch_metric_values[metric_name], epoch*len(train_loader) + i)

            if checkpoint_folder and i%10000 == 0 and i!=0:
                torch.save({
                    "epoch": epoch,
                    "epoch_progress": i,
                    "epoch_size": len(train_loader),
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "running_average_training_loss": running_average_training_loss_logger.get_avg(),
                    # "running_average_training_accuracy": running_average_training_accuracy_logger.get_avg() if is_classifier else None,
                    "epoch_metric_values": epoch_metric_values,
                }, os.path.join(checkpoint_folder, f"e-{epoch}-i-{i}-mbtl-{loss.item()}.pt"))

        if printing:
            print("--Train--")
            for metric_name, metric in metrics.items():
                print(f"{metric_name} : {metric.compute()}")
                metric.reset()

        val_loss_logger.reset()
        # if is_classifier:
        #     test_accuracy_logger.reset()
        with torch.inference_mode():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)

                model.eval()
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                val_loss_logger.add_value(loss.item())

                for metric_name, metric in metrics.items():
                    metric(y_pred, y)
                # if is_classifier:
                #     test_accuracy_logger.add_value(int(torch.argmax(y)==torch.argmax(y_pred)))


        if printing:
            print("--Val--")
            for metric_name, metric in metrics.items():
                print(f"{metric_name} : {metric.compute().item()}")
            print(f"train loss: {running_average_training_loss_logger.get_avg()} | test loss: {val_loss_logger.get_avg()}")
            # if is_classifier:
            #     print(f"train acc: {running_average_training_accuracy_logger.get_avg()} | test acc: {test_accuracy_logger.get_avg()}")
            # else:
            #     print(metric.compute())
        if scheduler:
            scheduler.step()

        if tensorboard_writer:
            tensorboard_writer.add_scalar("train_loss", running_average_training_loss_logger.get_avg(), epoch)
            tensorboard_writer.add_scalar("test_loss", val_loss_logger.get_avg(), epoch)
            for metric_name, metric in metrics.items():
                tensorboard_writer.add_scalar(f"val_{metric_name}", metric.compute().item(), epoch)
            # if is_classifier:
            #     tensorboard_writer.add_scalar("train_accuracy", running_average_training_accuracy_logger.get_avg(), epoch)
            #     tensorboard_writer.add_scalar("test_accuracy", test_accuracy_logger.get_avg(), epoch)

        if checkpoint_folder:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "training_loss": running_average_training_loss_logger.get_avg(),
                "val_loss": val_loss_logger.get_avg(),
                "metrics": {metric_name:metric.compute().item() for metric_name, metric in metrics.items()},
                # "training_accuracy": running_average_training_accuracy_logger.get_avg(),
                # "test_accuracy" : test_accuracy_logger.get_avg(),
            }, os.path.join(checkpoint_folder, f"e-{epoch}-train_l-{running_average_training_loss_logger.get_avg()}-test_l-{val_loss_logger.get_avg()}.pt"))
    
    # return model, running_average_training_accuracy_logger.get_avg(), test_accuracy_logger.get_avg()
    return model, metrics
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
    RECORD=False
    print("-"*10 + "~TRAIN~" + "-"*10)
    print(f"RECORD: {RECORD}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        

    print(f"Using device: {device.type}")

    print("Loading datasets...")
    DATASET_FOLDER_PATH = "dataset"

    with open("nasa/dataset/dataset_pm_nasa.npy", "rb") as f:
        x_train_np = np.load(f, allow_pickle=True)
        y_train_np = np.load(f, allow_pickle=True)
        x_val_np = np.load(f, allow_pickle=True)
        y_val_np = np.load(f, allow_pickle=True)    

        x_train = torch.from_numpy(x_train_np.astype(np.float32))
        y_train = torch.from_numpy(y_train_np.astype(np.float32))
        x_val = torch.from_numpy(x_val_np.astype(np.float32))
        y_val = torch.from_numpy(y_val_np.astype(np.float32))
    
    train_set = TensorDataset(x_train, y_train)
    val_set = TensorDataset(x_val, y_val)
    # scaling_factor = 64
    # train_set = TensorDataset(x_train[:len(x_train)//scaling_factor], y_train[:len(x_train)//scaling_factor])
    # val_set = TensorDataset(x_val[:len(x_val)//scaling_factor], y_val[:len(x_val)//scaling_factor])
    print(len(train_set))
    print(len(val_set))
    assert len(train_set) > len(val_set), f"Sanity check failed, size of train set ({len(train_set)}) must be greater than size of val set ({len(val_set)})"
    NUM_WORKERS = 1
    SHUFFLE = True
    BATCH_SIZE=16
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=SHUFFLE)

    START_EPOCH = 0
    print("Setting up model, optim, etc...")
    model = MLP([train_set[0][0].shape[-1], 100, 100, train_set[0][1].shape[-1]])
    model = LSTM(input_size=train_set[0][0].shape[-1],
                 hidden_size=256,
                 output_size=train_set[0][1].shape[-1],
                 bidirectional_layers_num=1,
                 unidirectional_layers_num=1,
                 is_classifier=False,
                 custom_head=nn.Sequential( 
                nn.Linear(256, 512),
                nn.SELU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1)))
    # model = LSTM(input_size=train_set[0][0].shape[-1],
    #              hidden_size=128,
    #              output_size=train_set[0][1].shape[-1],
    #              bidirectional_layers_num=1,
    #              unidirectional_layers_num=1,
    #              is_classifier=True,
    #              custom_head=None)

    # model = TEPTransformer(input_size=train_set[0][0].shape[-1], num_classes=train_set[0][1].shape[-1],
    #                        sequence_length=train_set[0][0].shape[-2], embedding_dim=128, nhead=4, num_layers=4)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000003)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.3)
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
        EXPERIMENT_DATE_TIME = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        CHECKPOINT_FOLDER = f"runs/{EXPERIMENT_DATE_TIME}"

    if RECORD:
        writer = SummaryWriter(f'runs_tensorboard\\{EXPERIMENT_DATE_TIME}')
        os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
    else:
        # shutil.rmtree(f'runs_tensorboard\\temp')
        writer = SummaryWriter(os.path.join("runs_tensorboard", str(EXPERIMENT_DATE_TIME)))
        CHECKPOINT_FOLDER = None


    train(
        model=model, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        loss_fn=loss_fn,
        epoch_range=range(START_EPOCH, 50), 
        train_loader=train_loader, 
        val_loader=val_loader,
        device=device, 
        printing=True,
        # is_classifier=False,
        metrics={
            "mae": torchmetrics.MeanAbsoluteError(),
            "mse": torchmetrics.MeanSquaredError()
            # "acc" : torchmetrics.Accuracy(task="multiclass", num_classes=18)
        },
        checkpoint_folder=CHECKPOINT_FOLDER,
        tensorboard_writer=writer)