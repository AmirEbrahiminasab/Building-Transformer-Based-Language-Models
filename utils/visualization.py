import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from skimage.transform import resize
import numpy as np
from PIL import Image
import os
import random
matplotlib.use('Agg')


def visualize_loss(loss_history) -> None:
    """Function to visualize loss during training"""
    plt.plot(loss_history['step'], loss_history['train'], label='Train Loss')
    plt.plot(loss_history['step'], loss_history['val'], label='Val Loss')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    plt.savefig(f'../utils/loss.png', dpi=350)



