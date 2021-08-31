import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from torch.utils.data import DataLoader, Subset
from models import LeNet, StoLeNet
from utils import compute_accuracy, compute_ece_loss
from flows import NormalAffine, NormalGlowStep, NormalPlanar1d

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

print("Experiments with FashionMNIST")
trainloader, testloader = prepare_dataset(data_dir, "FashionMNIST", 128, 64)
# ===================================================== #
# ========= train a deterministic model =============== #
# ===================================================== #
if train_deterministic:
    num_epochs = 10
    base_model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(base_model.parameters(), lr=0.002, momentum=0.9)

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
        print("Base Model Epoch {} Avg Loss {:.4f} Acc {:.4f} ECE {:.4f}".format(epoch, avg_loss, base_acc, base_ece))
        loss_list.append(avg_loss)
        acc_list.append(base_acc)
        ece_list.append(base_ece)


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
    # plt.show()
    fig.tight_layout()
    fig.savefig(fig_dir + "/" + "LeNet_FMNIST.jpg")


# ===================================================== #
# =  migrate from base model, finetune and train flow = #
# ===================================================== #
# flow config for all layers in the model  
sto_model_cfg = [NormalAffine, NormalGlowStep, NormalAffine, NormalPlanar1d, NormalAffine]

sto_epochs = 50
sto_model = StoLeNet(sto_cfg=sto_model_cfg, colored=False).to(device)
result1 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=sto_epochs, device=device)
torch.save(sto_model.state_dict(), "{}/{}".format(weight_dir, "StoLeNet_flow_FMNIST.pth"))
plot_results(result1, anno="StoLeNet_flow_FMNIST", fig_dir=fig_dir)

sto_model_cfg = [NormalAffine, NormalAffine, NormalAffine, NormalAffine, NormalAffine]
sto_model = StoLeNet(sto_cfg=sto_model_cfg, colored=False).to(device)
result2 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=sto_epochs, device=device)
plot_results(result2, anno="StoLeNet_no_flow_FMNIST", fig_dir=fig_dir)
torch.save(sto_model.state_dict(), "{}/{}".format(weight_dir, "StoLeNet_no_flow_FMNIST.pth"))

print("Experiments with CIFAR10")
# transforms adopted from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if train_deterministic:
    num_epochs = 20
    base_model = LeNet(colored=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(base_model.parameters(), lr=0.002, momentum=0.9)

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
        print("Base Model Epoch {} Avg Loss {:.4f} Acc {:.4f} ECE {:.4f}".format(epoch, avg_loss, base_acc, base_ece))
        loss_list.append(avg_loss)
        acc_list.append(base_acc)
        ece_list.append(base_ece)


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
    # plt.show()
    fig.tight_layout()
    fig.savefig(fig_dir + "/" + "LeNet_CIFAR10.jpg")


sto_model_cfg = [NormalAffine, NormalGlowStep, NormalAffine, NormalPlanar1d, NormalAffine]
sto_model = StoLeNet(sto_cfg=sto_model_cfg, colored=True).to(device)
result3 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=sto_epochs, device=device)
plot_results(result3, anno="StoLeNet_flow_CIFAR10", fig_dir=fig_dir)
torch.save(sto_model.state_dict(), "{}/{}".format(weight_dir, "StoLeNet_flow_CIFAR10.pth"))


sto_model_cfg = [NormalAffine, NormalAffine, NormalAffine, NormalAffine, NormalAffine]
sto_model = StoLeNet(sto_cfg=sto_model_cfg, colored=True).to(device)
result4 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=sto_epochs, device=device)
plot_results(result4, anno="StoLeNet_no_flow_CIFAR10", fig_dir=fig_dir)
torch.save(sto_model.state_dict(), "{}/{}".format(weight_dir, "StoLeNet_no_flow_CIFAR10.pth"))

result_list = [result1, result2, result3, result4]
anno_list = ["FMNIST Flow", "FMNIST no Flow", "CIFAR Flow", "CIFAR no Flow"]
plot_multiple_results(result_list, anno_list, fig_dir, "all_results_lenet")





