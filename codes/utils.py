import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F 

from models import StoModel
EPS:float = np.finfo(np.float32).eps

""" contains utility functions, like metrics and plotting helper
"""


class ECELoss(nn.Module):
    """
    Ported from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        # # accuracy, confidence, sample percentage for each bin 
        self.acc_list = torch.zeros(n_bins)
        self.conf_list = torch.zeros(n_bins)
        self.sample_per = torch.zeros(n_bins)
        self.n_samples = torch.zeros(n_bins) # number of processed samples in each bin, for batched forward 

    def forward(self, probs, labels, is_logit=False):
        """ compute the expected calibration error of a classification output

        Args:
            probs ([torch.Tensor]): the probability of each class (after softmax), typically shape (n_samples, n_classes)
            labels ([torch.Tensor]):  the true class label, typically shape (n_samples, 1)
            is_logit (bool, optional): the input to `probs` will be regarded as logits (before softmax) if this is true
        """
        if is_logit:
            probs = F.softmax(probs, dim=-1)
            
        confidences, predictions = torch.max(probs, 1)
        correct_samples = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        for idx, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0: # if no sample assigned in this bin, we get NaN from indexing
                accuracy_in_bin = correct_samples[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece = ece + torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                self.acc_list[idx] = accuracy_in_bin.item()
                self.conf_list[idx] = avg_confidence_in_bin.item()
                self.sample_per[idx] = prop_in_bin.item()

        return ece

    def add_batch(self, probs, labels, is_logit=False):
        """ gather the confidence and count the number of correct samples for a batch
            works similar to forward, use summarize_batch to get the ECE for the whole dataset 
            this implementation does not support back propagation, unless store a lot of computation maps
            as many as the number of batches in the dataset 
        """
        if is_logit:
            probs = F.softmax(probs, dim=-1)
        
        confidences, predictions = torch.max(probs, 1)
        correct_samples = predictions.eq(labels)
        
        for idx, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            num_in_bin, prop_in_bin = in_bin.sum(), in_bin.float().mean()
            n, k = self.n_samples[idx], num_in_bin
            
            if prop_in_bin > 0.0:
                weights = torch.tensor([n, k])/torch.tensor([n+k])
                accuracy_in_bin = correct_samples[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                # suppose we've already calculated the mean of n previous samples in a bin, i.e., self.acc_list[idx]
                # now we are adding k new samples to that bin, i.e., accuracy_in_bin, and want to average over n+k samples
                # we can weight the previous mean by n/(n+k), and the new batch by k/(n+k) and compute weighted mean of the two 
                self.acc_list[idx] = torch.sum(torch.tensor([self.acc_list[idx], accuracy_in_bin])*weights).item()
                self.conf_list[idx] = torch.sum(torch.tensor([self.conf_list[idx], avg_confidence_in_bin])*weights).item()
                
            self.n_samples[idx] = self.n_samples[idx] + k

    def summarize_batch(self):
        self.sample_per = self.n_samples/torch.norm(self.n_samples, p=1)
        return torch.sum(torch.abs(self.acc_list - self.conf_list)*self.sample_per)

    def plot_acc_conf_gap(self):
        """ plot the confidence-accuracy gap
            must run a forward pass first (or summarize_batch for batched ece)
        """
        if len(self.acc_list) == 0 or len(self.conf_list) == 0:
            print("run forward pass before plotting ece")
            return
        
        xs = (self.bin_uppers + self.bin_lowers) / 2
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.grid(True)
        width = 0.05
        plt.bar(xs, np.array(self.sample_per), width, label="Sample Percentage")
        plt.xlabel("Confidence")
        plt.legend()

        plt.subplot(1,2,2)
        plt.grid(True)
        plt.bar(xs, np.array(self.acc_list), width, label="Accuracy")
        plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), label="Ideal", color="r")
        plt.xlabel("Confidence")
        plt.legend()
        plt.show()
        
def compute_accuracy(model, dataloader, device=None):
    """ compute the classification accuracy of a model on a test set
    """
    if device==None:
        device = next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, StoModel):
                prob, _ = model.make_prediction(images)
            else:
                prob = model(images)
            _, predicted = torch.max(prob.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def compute_ece_loss(model, dataloader, device=None, n_bins=15):
    """ compute the batched expected calibration loss 
    """ 
    if device==None:
        device = next(model.parameters()).device
    model.eval() 
    batch_ece = ECELoss(n_bins=n_bins)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if isinstance(model, StoModel):
                probs, _ = model.make_prediction(images)
            else:
                probs = F.softmax(model(images), dim=-1) 
            batch_ece.add_batch(probs, labels)
    
    return batch_ece.summarize_batch().item() 

def classification_accuracy(prob, y):
    """ compute classification accuracy given the probability of each class 
        prob: size (n_samples, n_class)
        y: size (n_samples)
    """
    _, idx = torch.max(prob, dim=1)
    acc = torch.sum(idx == y)/y.numel()
    return acc      