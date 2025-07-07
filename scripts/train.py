import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import trange
import evaluate
from tqdm import tqdm
import random


def estimate_loss(model, eval_interval, get_batch, train_data, val_data, batch_size, block_size, device):
    """Function to estimate loss on train/val dataset"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            if split == "train":
                X, Y = get_batch(train_data, batch_size, block_size, device)
            else:
                X, Y = get_batch(val_data, batch_size, block_size, device)

            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(model, max_iterations, eval_interval, optimizer, get_batch, train_data, val_data, batch_size, block_size, device):
    """Function to train the model and evaluate the validation loss"""
    history = {
        'step': [],
        'train': [],
        'val': []
    }
    pbar = trange(max_iterations, desc="Training", leave=True)
    for iter in pbar:
        if iter % eval_interval == 0 or iter == max_iterations - 1:
            losses = estimate_loss(model, eval_interval, get_batch, train_data, val_data, batch_size, block_size, device)
            pbar.set_description(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            history['step'].append(iter)
            history['train'].append(losses['train'])
            history['val'].append(losses['val'])

        x, y = get_batch(train_data, batch_size, block_size, device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training Finished!")
    return model, history
