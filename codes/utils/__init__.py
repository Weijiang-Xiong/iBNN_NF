""" utility functions, including metrics, helper functions, plotting functions
"""
import os, sys
import numpy as np 

from .metrics import compute_ece_loss, compute_accuracy, ECELoss
from .train_helper import train_sto_model
from .plot_helper import plot_multiple_results, plot_results
from .data_helper import prepare_dataset

EPS:float = np.finfo(np.float32).eps