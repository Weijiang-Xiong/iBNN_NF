import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from models import LeNet, StoLeNet

# setup device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# prepare data
data_dir = "./data"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=True)
testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=transform, download=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']