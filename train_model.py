import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, os, sys
import copy
import h5py
import sklearn
import sklearn.metrics

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import timm

import utils
import utils.ML, utils.models
from utils.data import G2NetDataset

if __name__ == "__main__":

    # Train metadata
    path = '/Users/lucas/Documents/Data/G2Net'

    # dataset_list = ["data_synth_sensitivity_5",
    # "data_synth_sensitivity_10",
    # "data_synth_sensitivity_15",
    # "data_synth_sensitivity_20",
    # "data_synth_sensitivity_25",
    # "data_synth_sensitivity_30",
    # "data_synth_sensitivity_35",
    # "data_synth_sensitivity_40",
    # "data_synth_realistic_noise",
    # "data_synth_realistic_noise_1_5",
    # "train",
    # "train"]

    dataset_list = ["data_synth_sensitivity_5",
    "data_synth_sensitivity_10",
    "data_synth_sensitivity_15",
    "data_synth_sensitivity_20",
    "data_synth_sensitivity_25",
    "data_synth_sensitivity_30",
    "data_synth_sensitivity_35",
    "data_synth_sensitivity_40",
    "data_synth_realistic_noise_5_20",
    "data_synth_realistic_noise_1_5",
    "train",
    "train",
    "data_synth_sensitivity_5",
    "data_synth_sensitivity_10",
    "data_synth_sensitivity_15",
    "data_synth_sensitivity_20",
    "data_synth_realistic_noise_1_5",
    "train",
    "train"]

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

    # sys.exit()

    # split training data into train and validation sets:
    train_split, val_split = torch.utils.data.random_split(data, [num_train, num_val], generator=torch.Generator().manual_seed(42))

    batch_size = 32
    num_workers = 1

    # Create data loaders.
    train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_split, batch_size=batch_size)

    device = "mps"

    loss_fn = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()

    # model = utils.models.CNN().to(device)
    # model = utils.models.CNN_v2().to(device)
    model = utils.models.CNN_v2_dropout().to(device)
    # model = utils.models.EfficientNet(freeze_blocks=False).to(device)

    # Freeze parameters:
    model.moving_average.weight.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10

    model_name = "CNN_v2_dropout_17400_augment"

    train_loss_log, val_loss_log = utils.ML.train_model(model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
        num_epochs,
        save_model=True,
        model_name=model_name,
        device=device,
        verbose=True)


    # save model last epoch:
    utils.ML.save_model_func(model, 'saved_models/'+model_name+'_last.pt')


    ############## Evaluate on kaggle data:

    print("Evaluation on kaggle data:")

    data = G2NetDataset(path, "train", normalize=normalize)
    dataloader = DataLoader(data, batch_size=batch_size)
    average_loss, accuracy, auc = utils.ML.evaluate_model(model, dataloader, loss_fn, device)
    print(f"loss {average_loss:.3f}, accuracy {accuracy:.3f}, auc {auc:.3f}")

    plt.figure()
    plt.plot(train_loss_log)
    plt.plot(val_loss_log)
    plt.legend(["train loss", "validation loss"])
    plt.show()
