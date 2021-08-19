import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F

from torch.utils.data import DataLoader
from models import LeNet, StoLeNet
from utils import compute_accuracy, compute_ece_loss

train_deterministic = False # train a deterministic model as starting point 

# setup device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# prepare data
data_dir = "./codes/data"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=False)
testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=transform, download=False)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=16, shuffle=False)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ===================================================== #
# ========= train a deterministic model =============== #
# ===================================================== #
if train_deterministic:
    num_epochs = 10
    base_model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(base_model.parameters(), lr=0.001, momentum=0.9)

    loss_list, acc_list, ece_list = [[] for _ in range(3)]
    for epoch in range(num_epochs):
        base_model.train()
        batch_loss = []
        for img, label in trainloader:
            img, label = img.to(device), label.to(device)
            pred = base_model(img)
            loss = criterion(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
            
        avg_loss = sum(batch_loss)/len(batch_loss)
        base_acc = compute_accuracy(base_model, testloader, device=device)
        base_ece = compute_ece_loss(base_model, testloader, device=device)
        print("Base Model Epoch {} Avg Loss {} Acc {} base ECE {}".format(epoch, avg_loss, base_acc, base_ece))
        loss_list.append(avg_loss)
        acc_list.append(base_acc)
        ece_list.append(base_ece)

# ===================================================== #
# == migrate from base model, finetune and train flow = #
# ===================================================== #

sto_model_cfg = [
            ("normal", {"loc":1.0, "scale":0.5},  # the name of base distribution and parameters for that distribution
                [("affine", 1, {"learnable":True}), # the first stack of flows (type, depth, params)
                ("planar2d", 2, {"init_sigma":0.01}), # the second stack of flows (type, depth, params)
                ("flowstep", 2, {"width":6,"keepdim":True}),
                ("planar2d", 2, {"init_sigma":0.01})] 
            ),
            (
                "normal", {"loc":1.0, "scale":0.5}, 
                [("affine", 1), 
                 ("planar", 6)]
            )
            ]
sto_model = StoLeNet(sto_cfg=sto_model_cfg).to(device)

if train_deterministic:
    sto_model.migrate_from_det_model(base_model)
    
det_params, sto_params = sto_model.det_and_sto_params()
optimizer = optim.Adam([
                {'params': det_params, 'lr': 1e-4},
                {'params': sto_params, 'lr': 1e-3}
            ])

num_epochs = 10
loss_list, ll_list, kl_list, acc_list, ece_list = [[] for _ in range(5)]
for epoch in range(num_epochs):
    sto_model.train()
    batch_loss, batch_ll, batch_kl = [[] for _ in range(3)]
    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        pred = sto_model(img)
        log_likelihood, kl = sto_model.calc_loss(pred, label)
        loss = -log_likelihood + kl 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())
        batch_ll.append(log_likelihood.item()) 
        batch_kl.append(kl.item())
    avg = lambda l: sum(l)/len(l)
    avg_loss, avg_ll, avg_kl = avg(batch_loss), avg(batch_ll), avg(batch_kl)
    sto_acc = compute_accuracy(sto_model, testloader, device=device)
    sto_ece = compute_ece_loss(sto_model, testloader, device=device)
    print("Sto Model Epoch {} Avg Loss {} Likelihood {} KL {} Acc {} base ECE {}".format(
                        epoch, avg_loss, avg_ll, avg_kl,sto_acc, sto_ece))
    loss_list.append(avg_loss)
    ll_list.append(avg_ll)
    kl_list.append(avg_kl)
    acc_list.append(sto_acc)
    ece_list.append(sto_ece)
    