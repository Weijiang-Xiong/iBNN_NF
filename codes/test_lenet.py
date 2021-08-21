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

from torch.utils.data import DataLoader, Subset
from models import LeNet, StoLeNet
from utils import compute_accuracy, compute_ece_loss
from flows import NormalAffine, NormalGlowStep, NormalPlanar1d

train_deterministic = True # train a deterministic model as starting point 

# setup device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# define data folder
data_dir = "./data"
fig_dir = "./figs"

print("Experiments with FashionMNIST")
# compose transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=True)
testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


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

# parameters for base distribution 
NormalParams = lambda scale: {"loc":1.0, "scale":scale}
# flow configurations, List of tuple (type, depth, params)
AffineLayer = [("affine", 1, {"learnable":True})]
GlowStep =  lambda depth, width:[
            ("affine", 1, {"learnable":True}), # the first stack of flows (type, depth, params)
            ("planar2d", 1, {"init_sigma":0.01}),# the second stack of flows (type, depth, params)
            ("flowstep", depth, {"width":width,"keepdim":True}),
            ("planar2d", 1, {"init_sigma":0.01})] 
Planar1d = lambda depth: [("affine", 1), 
            ("planar", depth),
            ("element", 1, {"act":"tanh"})]
# stochastic part for a layer, base distribution name, distribution parameters, flow config 
NormalAffine = ("normal", NormalParams(0.5), AffineLayer)
NormalGlowStep = ("normal", NormalParams(0.5), GlowStep(3, 0.3))
NormalPlanar1d = ("normal", NormalParams(0.5), Planar1d(2))
# flow config for all layers in the model  
sto_model_cfg = [NormalAffine, NormalGlowStep, NormalAffine, NormalPlanar1d, NormalAffine]


def train_sto_model(sto_model:nn.Module, trainloader=None, testloader=None, base_model=None, num_epochs=30, device=None):
    
    if isinstance(base_model, sto_model.DET_MODEL_CLASS):
        sto_model.migrate_from_det_model(base_model)

    det_params, sto_params = sto_model.det_and_sto_params()
    optimizer = optim.Adam([
                    {'params': det_params, 'lr': 2e-4},
                    {'params': sto_params, 'lr': 2e-3}
                ])

    loss_list, ll_list, kl_list, acc_list, ece_list = [[] for _ in range(5)]
    for epoch in range(num_epochs):
        sto_model.train()
        batch_loss, batch_ll, batch_kl = [[] for _ in range(3)]
        for img, label in trainloader:
            img, label = img.to(device), label.to(device)
            pred = sto_model(img)
            log_likelihood, kl = sto_model.calc_loss(pred, label)
            loss = -log_likelihood + kl / len(trainloader.dataset)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
            batch_ll.append(log_likelihood.item()) 
            batch_kl.append(kl.item()/ len(trainloader.dataset))
        avg = lambda l: sum(l)/len(l)
        avg_loss, avg_ll, avg_kl = avg(batch_loss), avg(batch_ll), avg(batch_kl)
        sto_acc = compute_accuracy(sto_model, testloader)
        sto_ece = compute_ece_loss(sto_model, testloader)
        print("Sto Model Epoch {} Avg Loss {:.4f} Likelihood {:.4f} KL {:.4f} Acc {:.4f} ECE {:.4f}".format(
                            epoch, avg_loss, avg_ll, avg_kl,sto_acc, sto_ece))
        loss_list.append(avg_loss)
        ll_list.append(avg_ll)
        kl_list.append(avg_kl)
        acc_list.append(sto_acc)
        ece_list.append(sto_ece)

    return loss_list, ll_list, kl_list, acc_list, ece_list


sto_model = StoLeNet(sto_cfg=sto_model_cfg, colored=False).to(device)
result1 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=30, device=device)

def plot_results(results, anno=""):
    loss_list, ll_list, kl_list, acc_list, ece_list = results 
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2,3,1)
    plt.plot(loss_list)
    plt.title("Negative ELBO")
    plt.subplot(2,3,2)
    plt.plot(ll_list)
    plt.title("Log Likelihood")
    plt.subplot(2,3,3)
    plt.plot(kl_list)
    plt.title("KL Divergence")
    plt.subplot(2,3,4)
    plt.plot(acc_list)
    plt.title("Test Accuracy")
    plt.subplot(2,3,5)
    plt.plot(ece_list)
    plt.title("ECE on testset")
    # plt.show()
    fig.tight_layout()
    fig.savefig(fig_dir + "/" + "{}.jpg".format(anno))


plot_results(result1, anno="StoLeNet_flow_FMNIST")

sto_model_cfg = [NormalAffine, NormalAffine, NormalAffine, NormalAffine, NormalAffine]
sto_model = StoLeNet(sto_cfg=sto_model_cfg, colored=False).to(device)
result2 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=30, device=device)
plot_results(result2, anno="StoLeNet_no_flow_FMNIST")

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

sto_epochs = 50
sto_model_cfg = [NormalAffine, NormalGlowStep, NormalAffine, NormalPlanar1d, NormalAffine]
sto_model = StoLeNet(sto_cfg=sto_model_cfg, colored=True).to(device)
result3 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=sto_epochs, device=device)
plot_results(result3, anno="StoLeNet_flow_CIFAR10")


sto_model_cfg = [NormalAffine, NormalAffine, NormalAffine, NormalAffine, NormalAffine]
sto_model = StoLeNet(sto_cfg=sto_model_cfg, colored=True).to(device)
result4 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=sto_epochs, device=device)
plot_results(result4, anno="StoLeNet_no_flow_CIFAR10")


def plot_multiple_results(result_list, anno_list, fig_dir=None, save_name=None):
    fig = plt.figure(figsize=(15, 10))
    for result in result_list:
        loss_list, ll_list, kl_list, acc_list, ece_list = result 
        plt.subplot(2,3,1)
        plt.plot(loss_list)
        plt.title("Negative ELBO")
        plt.legend(anno_list)
        plt.subplot(2,3,2)
        plt.plot(ll_list)
        plt.title("Log Likelihood")
        plt.legend(anno_list)
        plt.subplot(2,3,3)
        plt.plot(kl_list)
        plt.title("KL Divergence")
        plt.legend(anno_list)
        plt.subplot(2,3,4)
        plt.plot(acc_list)
        plt.title("Test Accuracy")
        plt.legend(anno_list)
        plt.subplot(2,3,5)
        plt.plot(ece_list)
        plt.title("ECE on testset")
        plt.legend(anno_list)

    # plt.show()
    fig.tight_layout()
    if os.path.exists(fig_dir) and save_name!=None:
        fig.savefig(fig_dir + "/" + "{}.jpg".format(save_name))


result_list = [result1, result2, result3, result4]
anno_list = ["FMNIST Flow", "FMNIST no Flow", "CIFAR Flow", "CIFAR no Flow"]
plot_multiple_results(result_list, anno_list, fig_dir, "all_results_lenet")





