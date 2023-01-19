import time, copy
import numpy as np
import sklearn
import sklearn.metrics

import torch
from torch import nn
from torch.nn import functional as F

sigmoid = nn.Sigmoid()

def train_model(model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    num_epochs,
    device,
    save_model=False,
    model_name="saved_model",
    verbose=True):
    '''
        Trains a model, given an optimizer and a number of epochs.
        Computes validation loss and accuracy after each epoch, and prints train/validation metrics.
        Returns per-iteration train and validation losses, for plotting.
    '''
    
    model.train()

    train_loss_log = []
    val_loss_log = []

    best_auc = 0.0

    num_batches = len(train_dataloader)

    t = time.time()

    for i_epoch in range(num_epochs):
        
        epoch_loss = 0
        num_samples = 0

        for i_batch, (X, y) in enumerate(train_dataloader):
            y = torch.unsqueeze(y, 1)
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss * len(y)
            num_samples += len(y)

            if verbose == "batch_metrics":
                print(f"batch {i_batch+1}/{num_batches}, loss: {batch_loss:.3f}")

        # calculate average loss of this epoch:
        mean_epoch_loss = epoch_loss/num_samples

        # calculate validation loss and accuracy:
        val_loss, val_acc, val_auc = evaluate_model(model, val_dataloader, loss_fn, device=device)
        model.train()
        
        # print and save metrics:
        if verbose:
            print(f"epoch: {i_epoch+1:>2}, training loss: {mean_epoch_loss:.3f}, validation loss {val_loss:.3f}, validation accuracy {val_acc:.3f}, validation auc {val_auc:.3f}")

        if save_model:
            if val_auc > best_auc:
                save_model_func(model, 'saved_models/'+model_name+'.pt')
                best_auc = val_auc

        train_loss_log.append(mean_epoch_loss)
        val_loss_log.append(val_loss)

    # Calculate average time per epoch:
    time_per_epoch = (time.time()-t)/num_epochs
    
    if verbose:
        print(f'\nAverage time per epoch: {time_per_epoch:.3f}s')
        
    return train_loss_log, val_loss_log



def evaluate_model(model, dataloader, loss_fn, device):
    ''' Calculates average loss and accuracy over a dataset'''
    model.eval()
    
    num_correct = 0
    total_loss = 0
    total_auc = 0
    num_samples, num_batches = 0, 0
    predictions, labels = [], []

    for i_batch, (X, y) in enumerate(dataloader):
        y = torch.unsqueeze(y, 1)
        X, y = X.to(device), y.to(device)

        # Prediction on batch X:
        with torch.no_grad():
            pred = model(X)
            
        # convert back to cpu:
        # pred = pred.cpu()
        pred_sigmoid = sigmoid(pred)
            
        # Predicted class indexes:
        pred_idx = pred_sigmoid >= 0.5
        
        # Batch loss:
        batch_loss = loss_fn(pred, y).item()

        # store results:
        predictions.append(pred_sigmoid.cpu())
        labels.append(y.cpu())

        total_loss += batch_loss * len(y)
        num_correct += torch.sum(pred_idx == y).item()
        num_samples += len(y)
        num_batches += 1

    average_loss = total_loss/num_samples
    accuracy = num_correct/num_samples

    labels = torch.concatenate(labels)
    predictions = torch.concatenate(predictions)
    auc = sklearn.metrics.roc_auc_score(labels, predictions)
        
    return average_loss, accuracy, auc


def save_model_func(model, path, verbose=True):

    model_cpu = copy.deepcopy(model)
    model_cpu.to('cpu')  # in place
    model_scripted = torch.jit.script(model_cpu) # Export to TorchScript
    model_scripted.save(path) # Save
    if verbose:
        print(f"Model saved to {path}")

