import sys
import os
import numpy as np
from multiprocessing import freeze_support
import pickle
import torch.optim as optim
import yaml
import torch
import torch.nn as nn

np.random.seed(38)
import train
import evaluate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import model
from data import data_loader
from utils import visualization


with open('../config/config.yaml', "r") as f:
    config = yaml.safe_load(f)

if __name__ == '__main__':
    freeze_support()
    train_data, val_data, vocab_size, itos, stoi = data_loader.load_dataset()
    os.chdir(os.path.dirname(__file__))
    print("Loaded Data Successfully!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gpt_model = model.GPTLanguageModel(int(config["n_embd"]), vocab_size, int(config["n_head"]), int(config["n_layer"]), int(config["block_size"]), int(config["dropout"]), device)
    gpt_model = gpt_model.to(device)
    print(f"{sum(p.numel() for p in gpt_model.parameters()) / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=float(config["lr"]))
    best_model, history = train.train_model(gpt_model, int(config["max_iterations"]), int(config["eval_interval"]), optimizer, data_loader.get_batch, train_data, val_data, int(config["batch_size"]), int(config["block_size"]), device)

    visualization.visualize_loss(history)
    torch.save(best_model.state_dict(), "../models/saved_models/model.pth")

    # Load best model
    best_model = model.GPTLanguageModel(int(config["n_embd"]), vocab_size, int(config["n_head"]), int(config["n_layer"]), int(config["block_size"]), int(config["dropout"]), device)
    best_model.load_state_dict(torch.load("../models/saved_models/model.pth"))
    best_model.to(device)
    best_model.eval()

    # Evaluate
    evaluate.generate_text(best_model, torch.zeros((1, 1), dtype=torch.long, device=device), data_loader.decode, itos)

    encoded_prompt = data_loader.encode("Monica:", stoi)
    context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
    evaluate.generate_text(best_model, context, data_loader.decode, itos)

    encoded_prompt = data_loader.encode("Chandler:", stoi)
    context = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
    evaluate.generate_text(best_model, context, data_loader.decode, itos)








