import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def prepare_dataset(data_dir:str, data_name:str, train_bs:int, test_bs:int, subset:int=None):
    """ prepare trainloader, testloader
    Args:
        data_dir (str): the directory to store dataset
        data_name (str): the name of dataset, "fashionmnist", "cifar10"
        train_bs (int): batch size for training
        test_bs (int): batch size for testing
        subset (int, optional): slice a subset of original, good for debugging. Defaults to None.
    """
    if data_name.lower()=="fashionmnist":
        trainloader, testloader = _fashion_mnist(data_dir, train_bs, test_bs, subset)
    elif data_name.lower() == "cifar10":
        trainloader, testloader = _cifar10(data_dir, train_bs, test_bs, subset)
        
    return trainloader, testloader

def _cifar10(data_dir, train_bs, test_bs, subset:int=None):
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
    if isinstance(subset,int) and subset > 0:
         trainset, testset = Subset(trainset, range(subset)), Subset(testset, range(subset))
    trainloader = DataLoader(trainset, batch_size=train_bs, shuffle=True)
    testloader = DataLoader(testset, batch_size=test_bs, shuffle=False)
    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader

def _fashion_mnist(data_dir, train_bs, test_bs, subset:int=None):
    # compose transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=True)
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=transform, download=True)
    trainloader = DataLoader(trainset, batch_size=train_bs, shuffle=True)
    testloader = DataLoader(testset, batch_size=test_bs, shuffle=False)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                    'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return trainloader, testloader