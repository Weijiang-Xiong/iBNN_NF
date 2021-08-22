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
from models import vgg16, sto_vgg16
from utils import compute_accuracy, compute_ece_loss
from flows import NormalAffine, NormalGlowStep, NormalPlanar1d

train_deterministic = True # train a deterministic model as starting point 

# setup device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# prepare data
data_dir = "./data"
fig_dir = "./figs"
weight_dir = "./models/trained/"
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
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

sto_model_cfg = [NormalAffine, NormalGlowStep, NormalAffine, NormalPlanar1d, NormalAffine]
feature_flow = [NormalAffine]*3 + [NormalGlowStep] + \
                [NormalAffine]*3 + [NormalGlowStep] + \
                [NormalAffine]*2 + [NormalGlowStep] + [NormalAffine]*2
classifier_flow = [NormalAffine] + [NormalPlanar1d] + [NormalAffine]
sto_model_cfg = feature_flow + classifier_flow

def train_sto_model(sto_model, trainloader=None, testloader=None, base_model=None, num_epochs=30, device=None):

    if isinstance(base_model, sto_model.DET_MODEL_CLASS):
        sto_model.migrate_from_det_model(base_model)
        print("Loaded weights from a base model")

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
            if np.isnan(pred.detach().cpu()[0,0]):
                continue
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
        sto_acc = compute_accuracy(sto_model, testloader, n_samples=64)
        sto_ece = compute_ece_loss(sto_model, testloader, n_samples=64)
        print("Sto Model Epoch {} Avg Loss {:.4f} Likelihood {:.4f} KL {:.4f} Acc {:.4f} ECE {:.4f}".format(
                            epoch, avg_loss, avg_ll, avg_kl,sto_acc, sto_ece))
        loss_list.append(avg_loss)
        ll_list.append(avg_ll)
        kl_list.append(avg_kl)
        acc_list.append(sto_acc)
        ece_list.append(sto_ece)

    return loss_list, ll_list, kl_list, acc_list, ece_list

sto_epochs = 80
if train_deterministic == False:
    try:
        base_model = vgg16().load_state_dict(torch.load("{}/{}".format(weight_dir, "base_vgg_weights.pth")))
        print("Loaded weights from pretrained model")
    except:
        base_model = vgg16()

sto_model = sto_vgg16(sto_cfg=sto_model_cfg).to(device)
result1 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=sto_epochs, device=device)

from test_lenet import plot_results, plot_multiple_results

plot_results(result1, anno="VGG16_flow_CIFAR10")

sto_model_cfg = [NormalAffine]*16
sto_model = sto_vgg16(sto_cfg=sto_model_cfg).to(device)
result2 = train_sto_model(sto_model, trainloader, testloader, base_model, num_epochs=sto_epochs, device=device)
plot_results(result2, anno="no_flow")


result_list = [result1, result2]
anno_list = ["VGG_flow", "VGG_no_flow"]
plot_multiple_results(result_list, anno_list, fig_dir=fig_dir, save_name="all_results_vgg")