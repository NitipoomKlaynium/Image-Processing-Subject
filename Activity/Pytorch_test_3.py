import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import numpy as np
from zmq import device

device = torch.device('cuda' if torch.cuda.is_available() else 'epu')

input_size = 784
hidden_size = 100
num_classes = 10
num_classes = 2
batch_size = 100
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='./data')