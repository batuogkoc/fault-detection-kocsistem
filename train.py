import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from dataset_single import *


if __name__ == "__main__":
    KUACC=True
    RECORD=True
    if KUACC:
        print("-"*10 + "~KUACC~" + "-"*10)
        device = torch.device("cuda")
    else:
        torch.manual_seed(42)
        device = torch.device("cpu")

    # TORCH_DATASET_FOLDER_PATH = "dataset/Torch"

    # faulty_train = TEPSingleSampleDataset(os.path.join(TORCH_DATASET_FOLDER_PATH, "TEP_Faulty_Training.hdf5"))
    # faulty_test = TEPSingleSampleDataset(os.path.join(TORCH_DATASET_FOLDER_PATH, "TEP_Faulty_Test.hdf5"))

    # faultfree_train = TEPSingleSampleDataset(os.path.join(TORCH_DATASET_FOLDER_PATH, "TEP_FaultFree_Training.hdf5"))
    # faultfree_test = TEPSingleSampleDataset(os.path.join(TORCH_DATASET_FOLDER_PATH, "TEP_FaultFree_Test.hdf5"))
    DATASET_FOLDER_PATH = "dataset"
    train_set = torch.load(os.path.join(DATASET_FOLDER_PATH, "train.pt"))
    val_set = torch.load(os.path.join(DATASET_FOLDER_PATH, "val.pt"))

    print(len(train_set))
    print(len(val_set))
    NUM_WORKERS = 1
    SHUFFLE = False
    train_loader = DataLoader(train_set, batch_size=64, num_workers=NUM_WORKERS, shuffle=SHUFFLE)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=NUM_WORKERS, shuffle=SHUFFLE)

    START_EPOCH = 0
    print(train_set[0][0].shape[-1])
    model = torch.nn.Sequential([
        nn.Linear(train_set[0][0].shape[-1], )
    ])
    exit()
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.3)
    
    # LOAD_PROGRESS_PATH = "runs/2024-06-20T15:23:33/e-8-train_l-813.9820446734548-test_loss-869.9771018757278.pt"
    LOAD_PROGRESS_PATH = None
    if LOAD_PROGRESS_PATH:
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
        scheduler.load_state_dict(state["scheduler_state_dict"])
        print(state.keys())
    else:
        EXPERIMENT_DATE_TIME = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        CHECKPOINT_FOLDER = f"runs/{EXPERIMENT_DATE_TIME}"

    if RECORD:
        writer = SummaryWriter(f'runs_tensorboard/{EXPERIMENT_DATE_TIME}')
        os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

    for epoch in range(START_EPOCH, 25):
        print("-"*5 + f"EPOCH: {epoch}" + "-"*5)
        start = time.time()
        train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            model.train()
            y_pred = model(x)

            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            running_average_training_loss = train_loss/max((i), 1)

            if i % 20 == 0 and i != 0:
                fraction_done = max(i/len(train_loader), 1e-6)
                time_taken = (time.time()-start)
                print(f"i: {i}| loss: {loss} | ratl: {running_average_training_loss}")
                print(f"{fraction_done*100}% | est time left: {time_taken*(1-fraction_done)/fraction_done} s | est total: {time_taken/fraction_done} s")
                if RECORD:
                    writer.add_scalar("running_average_training_loss", running_average_training_loss, epoch*len(train_loader) + i)
                
            if i%500 == 0 and i!=0 and RECORD:
                torch.save({
                    "epoch": epoch,
                    "epoch_progress": i,
                    "epoch_size": len(train_loader),
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "running_average_training_loss": running_average_training_loss,
                }, os.path.join(CHECKPOINT_FOLDER, f"e-{epoch}-i-{i}-mbtl-{loss.item()}-ratl-{running_average_training_loss}.pt"))

        train_loss /= len(train_loader)

        test_loss = 0
        metrics = {
            "ade": 0,
            "de_1": 0,
            "de_2": 0,
            "de_3": 0,
        }
        with torch.inference_mode():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)

                model.eval()
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()

                batch_metrics = calculate_metrics(y, y_pred)
                metrics["ade"] += batch_metrics["ade"]
                metrics["de_1"] += batch_metrics["de_1"]
                metrics["de_2"] += batch_metrics["de_2"]
                metrics["de_3"] += batch_metrics["de_3"]

        test_loss /= len(val_loader)
        metrics["ade"] /= len(val_loader)
        metrics["de_1"] /= len(val_loader)
        metrics["de_2"] /= len(val_loader)
        metrics["de_3"] /= len(val_loader)

        print(f"train loss: {train_loss} test loss: {test_loss}")
        print(f"metrics: {metrics}")
        scheduler.step()

        if RECORD:
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("ade", metrics["ade"], epoch)
            writer.add_scalar("de_1", metrics["de_1"], epoch)
            writer.add_scalar("de_2", metrics["de_2"], epoch)
            writer.add_scalar("de_3", metrics["de_3"], epoch)

            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "training_loss": train_loss,
                "test_loss": test_loss,
                "metrics": metrics,
            }, os.path.join(CHECKPOINT_FOLDER, f"e-{epoch}-train_l-{train_loss}-test_l-{test_loss}-ade-{metrics['ade']}.pt"))
        

