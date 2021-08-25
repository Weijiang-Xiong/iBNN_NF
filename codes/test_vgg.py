import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from models import vgg16, sto_vgg16
from utils import compute_accuracy, compute_ece_loss
from flows import NormalAffine, NormalGlowStep, NormalPlanar1d, NormalInvConv

from utils import train_sto_model, plot_multiple_results, plot_results, prepare_dataset

train_deterministic = True # train a deterministic model as starting point 

# setup device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# define data folder
data_dir = "./data"
fig_dir = "./figs"
weight_dir = "./models/trained/"

trainloader, testloader = prepare_dataset(data_dir, "cifar10", 128, 16)

if train_deterministic:
    num_epochs = 30
    base_model = vgg16().to(device)
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
        torch.save(base_model.state_dict(), "{}/{}".format(weight_dir, "base_vgg_weights.pth"))
        
if train_deterministic:
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1,3,1)
    plt.plot(loss_list)
    plt.title("Training Loss")
    plt.subplot(1,3,2)
    plt.plot(acc_list)
    plt.title("Test Accuracy")
    plt.subplot(1,3,3)
    plt.plot(ece_list)
    plt.title("ECE on Test Set")
    plt.show()
    fig.tight_layout()
    fig.savefig(fig_dir + "/" + "VGG16_CIFAR10.jpg")
    
# ===================================================== #
# =  migrate from base model, finetune and train flow = #
# ===================================================== #

feature_flow = [NormalAffine]*3 + [NormalInvConv] + \
                [NormalAffine]*3 + [NormalInvConv] + \
                [NormalAffine]*2 + [NormalInvConv] + [NormalAffine]*2
classifier_flow = [NormalAffine] + [NormalPlanar1d] + [NormalAffine]
sto_model_cfg = feature_flow + classifier_flow

sto_epochs = 80
if train_deterministic == False:
    try:
        base_model = vgg16().load_state_dict(torch.load("{}/{}".format(weight_dir, "base_vgg_weights.pth")))
        print("Loaded weights from pretrained model")
    except:
        base_model = vgg16()

sto_model = sto_vgg16(sto_cfg=sto_model_cfg).to(device)
result1 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=sto_epochs, device=device)


plot_results(result1, anno="VGG16_flow_CIFAR10")

sto_model_cfg = [NormalAffine]*16
sto_model = sto_vgg16(sto_cfg=sto_model_cfg).to(device)
result2 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=sto_epochs, device=device)
plot_results(result2, anno="no_flow")


result_list = [result1, result2]
anno_list = ["VGG_flow", "VGG_no_flow"]
plot_multiple_results(result_list, anno_list, fig_dir=fig_dir, save_name="all_results_vgg")