import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def generate_text(model, start_token, decode, itos):
    """Function to generate text based on the given prompt(input)"""
    generated_indices = model.generate(start_token, max_new_tokens=500)
    generated_text = decode(generated_indices[0].tolist(), itos)

    print(generated_text)
