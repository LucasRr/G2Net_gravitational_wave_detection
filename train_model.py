import os, sys
import copy
import h5py
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import utils
import utils.ML, utils.models
from utils.data import G2NetDataset

if __name__ == "__main__":

    # Read arguments:
    msg = "Train a model using dataset present in data/\n"
    msg += "Example usage:\n"
    msg += "> python train_model.py --model_name CNN --num_epochs 15 --batch_size 32 --device mps"
    parser = argparse.ArgumentParser(description=msg, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--model_name', required=True)  
    parser.add_argument('--num_epochs', default='10', type=int)
    parser.add_argument('--batch_size', default='32', type=int)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--num_workers', default='1', type=int)

    args = parser.parse_args()

    model_name = args.model_name
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    device = args.device
    num_workers = args.num_workers

    # Load data:
    data_path = './data/'

    # List all datasets in data/:
    dataset_list = [d for d in os.listdir('data/') if os.path.isdir('data/'+d)]
    print(f"datasets found for train/validation/test:\n{dataset_list}")

    normalize = True  # normalize all spectrograms to zero-mean unit variance
    augment = True    # data augmentation

    data = torch.utils.data.ConcatDataset(
        [G2NetDataset(data_path, dataset_name, normalize=normalize, augment=augment) for dataset_name in dataset_list]
        )

    train_val_test_split = [0.6, 0.2, 0.2]

    # split training data into train and validation sets:
    train_split, val_split, test_split = torch.utils.data.random_split(data, train_val_test_split, generator=torch.Generator().manual_seed(42))

    num_data = len(data)
    num_train, num_val, num_test = len(train_split), len(val_split), len(test_split)

    print(f"\nDataset size: {num_data}")
    print(f"train/val/test split: {train_val_test_split[0]}/{train_val_test_split[1]}/{train_val_test_split[2]}")
    print(f"Number of train/val/test samples: {num_train}/{num_val}/{num_test}")

    # Create data loaders.
    train_dataloader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load model:
    if model_name == "CNN":
        model = utils.models.CNN().to(device)
    elif model_name == "CNN_2":
        model = utils.models.CNN_v2().to(device)
    elif model_name == "EfficientNet":
        model = utils.models.EfficientNet(freeze_blocks=False).to(device)
    else:
        raise NotImplementedError(f"{model_name} not implemented. Choose among 'CNN', 'CNN_2', or 'EfficientNet'.")
    print(f"\nTraining model {model_name}:")

    # Freeze moving average parameters:
    model.moving_average.weight.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

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

    # Evaluate model:

    # Load the model with best validation AUC:
    model = torch.jit.load('saved_models/'+model_name+'.pt', map_location="cpu")
    model.to(device)

    test_loss, test_acc, test_auc = utils.ML.evaluate_model(model, test_dataloader, loss_fn, device=device)
    print(f"Test loss: {test_loss:.3f}, test accuracy: {test_acc:.3f}, test AUC: {test_auc:.3f}")

