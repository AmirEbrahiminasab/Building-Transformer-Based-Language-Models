import os
import pandas as pd
import numpy as np
import sys
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
import random
from collections import Counter
import string
from utils import visualization
import pandas as pd
os.chdir(os.path.dirname(__file__))


np.random.seed(38)


def encode(s: str, stoi: dict) -> list:
    """Function to encode the given input"""
    return [stoi[c] for c in s]


def decode(ls: list, itos: dict) -> str:
    """Function to encode the given input"""
    return ''.join([itos[i] for i in ls])


def load_dataset() -> tuple:
    """Function to load our dataset"""
    df = pd.read_csv("dataset/friends.csv")
    text = ' '.join(df['text'].dropna().astype(str).tolist())
    print(f"Total length of dataset in characters: {len(text)}")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    print(''.join(chars))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    sz = int(0.9 * len(data))
    train_data, val_data = data[:sz], data[sz:]

    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    return train_data, val_data, vocab_size, itos, stoi


def get_batch(data, batch_size, block_size, device) -> tuple:
    """Function to return a batch from our corpus based on the split type."""
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


