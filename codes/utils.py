import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F 

""" contains utility functions, like metrics and plotting helper
"""

def classification_accuracy(prob, y):
    """ compute classification accuracy given the probability of each class 
        prob: size (n_samples, n_class)
        y: size (n_samples)
    """
    _, idx = torch.max(prob, dim=1)
    acc = torch.sum(idx == y)/y.numel()
    return acc

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
        self.acc_list = []
        self.conf_list = []
        self.perc_samples_list = []

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
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                self.acc_list.append(accuracy_in_bin.item())
                self.conf_list.append(avg_confidence_in_bin.item())
                self.perc_samples_list.append(prop_in_bin.item())
            else:
                self.acc_list.append(0.0)
                self.conf_list.append(0.0)
                self.perc_samples_list.append(0.0)

        return ece

    def plot_acc_conf_gap(self):
        """ plot the confidence-accuracy gap
            must run a forward pass first 
        """
        if len(self.acc_list) == 0 or len(self.conf_list) == 0:
            print("run forward pass before plotting ece")
            return
        
        xs = (self.bin_uppers + self.bin_lowers) / 2
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.grid(True)
        width = 0.05
        plt.bar(xs, np.array(self.perc_samples_list), width, label="Sample Percentage")
        plt.xlabel("Confidence")
        plt.legend()

        plt.subplot(1,2,2)
        plt.grid(True)
        plt.bar(xs, np.array(self.acc_list), width, label="Accuracy")
        plt.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), label="Ideal", color="r")
        plt.xlabel("Confidence")
        plt.legend()
        plt.show()