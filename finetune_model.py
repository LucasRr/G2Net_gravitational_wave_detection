import numpy as np
import matplotlib.pyplot as plt
import time, os, sys
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

import utils
import utils.ML, utils.models
from utils.data import G2NetDataset


if __name__ == "__main__":

    ############ load model:
    model_name = "CNN_all.pt"

    model = torch.jit.load('saved_models/'+model_name, map_location="cpu")

    device = "mps"
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    ########### Load data:

    path = '/Users/lucas/Documents/Data/G2Net'

    dataset_list = ["train",
        "data_synth_sensitivity_30",
        "data_synth_sensitivity_40",
        "data_synth_realistic_noise"]

    normalize = True  # normalize all spectrograms to zero-mean unit variance
    augment = True    # data augmentation

    data = torch.utils.data.ConcatDataset(
        [G2NetDataset(path, dataset_name, normalize=normalize, augment=augment) for dataset_name in dataset_list]
        )

    num_data = len(data)

    train_val_split = 0.9
    num_train = int(num_data * train_val_split)
    num_val = num_data - num_train

    print(f"dataset length: {num_data}")
    print(f"train/val split: {num_train}/{num_val} ({train_val_split*100}%)")

    # split training data into train and validation sets:
    train_split, val_split = torch.utils.data.random_split(data, [num_train, num_val], generator=torch.Generator().manual_seed(42))

    batch_size = 32
    num_workers = 1

    # Create data loaders.
    train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_split, batch_size=batch_size)


    ######## fine tune model:

    print(f"\nFine-tuning model on {dataset_list}:")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10

    train_loss_log, val_loss_log = utils.ML.train_model(model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
        num_epochs,
        save_model=True,
        model_name="CNN_all_finetuned3",
        device=device,
        verbose=True)

