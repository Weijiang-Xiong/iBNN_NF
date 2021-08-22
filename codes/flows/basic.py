import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as D

import numpy as np 
from typing import List, Tuple, Dict, Set

EPS = 1e-8

class PlanarFlow(nn.Module):
    """ modified based on https://github.com/kamenbliznashki/normalizing_flows/blob/master/planar_flow.py
    """
    def __init__(self, vec_len, init_sigma=0.01):
        super(PlanarFlow, self).__init__()
        self.v = nn.Parameter(torch.randn(1, vec_len).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, vec_len).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))

    def forward(self, x, normalize_u=True):
        # allow for a single forward pass over all the transforms in the flows with a Sequential container
        z = x
        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        v_hat = self.v
        if normalize_u:
            wtu = (self.w @ self.v.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            v_hat = self.v + (m_wtu - wtu) * self.w / (self.w @ self.w.t() + EPS)

        # compute transform
        wtz_plus_b = z @ self.w.t() + self.b
        f_z = z + v_hat * torch.tanh(wtz_plus_b)
        # compute log_abs_det_jacobian
        psi = (1 - torch.tanh(wtz_plus_b)**2) @ self.w
        det = 1 + psi @ v_hat.t()
        log_abs_det_jacobian = torch.log(torch.abs(det) + EPS).squeeze()

        return f_z, log_abs_det_jacobian

class AffineTransform(nn.Module):
    """ will keep the input unchanged if not learnable 
    """
    def __init__(self, vec_len=2, learnable=True):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(vec_len)).requires_grad_(learnable)
        self.logsigma = nn.Parameter(torch.zeros(vec_len)).requires_grad_(learnable)

    def forward(self, x):
        # apply mu and sigma to different channels
        if len(x.shape) == 4: # when x has shape (N, C, H, W) in conv layer
            z = self.mu.view(1, -1, 1, 1) + self.logsigma.exp().view(1, -1, 1, 1) * x
        elif len(x.shape) == 2: # when x has shape (N, C) in linear layer
            z = self.mu.view(1, -1) + self.logsigma.exp().view(1, -1) * x
        
        # z = self.mu + self.logsigma.exp() * x
        log_abs_det_jacobian = self.logsigma.sum()
        return z, log_abs_det_jacobian
    
class PlanarFlow2d(nn.Module):
    
    def __init__(self, in_channel, init_sigma=0.01, keepdim=True):
        """
        Args:
            in_channel ([int]): [number of input channel]
            init_sigma (float, optional): [ standard deviation used to initialize weights]. Defaults to 0.01.
            keepdim (bool, optional): [Suppose the input feature map has shape (N, C, H, W), if set to true, the flow will regard the tensor as N*H*W (batch and image dimensions) independent length-C vectors. If false, the image dimensions (H, W) will be counted into a vector, resulting in N independent vectors with size C*H*W]. Defaults to True.
        """
        super().__init__()
        self.in_channel = in_channel
        self.v = nn.Parameter(torch.randn(1, in_channel).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, in_channel).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))
        self.keepdim = keepdim
        
    def forward(self, x, normalize_u=True):
        """
        Args:
            x ([tuple]): consists of two tensors 
                        1. input image/feature z of size (N, C, H, W)
                        2. sum_log_abs_det_jacobians of size (N, H, W)
        returns: f_z: transformed samples  
                 log_abs_det_jacobian for this stack of flow 
        """
        z = x    
            
        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        v_hat = self.v
        if normalize_u:
            wtu = (self.w @ self.v.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            v_hat = self.v + (m_wtu - wtu) * self.w / (self.w @ self.w.t() + EPS)
        
        # treat z as N*H*W different length-C vectors, apply planar transform to each of them
        # which is equivalent to a 2d convolution
        wtz_plus_b = F.conv2d(z, self.w.view(1, self.in_channel, 1, 1)) + self.b
        f_z = z + v_hat.view(1, self.in_channel, 1, 1) * torch.tanh(wtz_plus_b)
        det = 1 + (1-torch.tanh(wtz_plus_b)**2) * (self.w @ v_hat.t())
        
        log_abs_det_jacobian = torch.log(torch.abs(det) + EPS).squeeze()
        
        if not self.keepdim:
            log_abs_det_jacobian = log_abs_det_jacobian.sum(dim=[1,2])
            
        return f_z, log_abs_det_jacobian
class ElementFlow(nn.Module):
    """ apply element wise transformation to the samples, similar to activation 
        idea from Lecture 3 of Deep Unsupervised Learning at UC Berkeley 
        https://drive.google.com/file/d/1j-3ErOVr8gPLEbN6J4jBeO84I7CqQdde/view
    """
    act_fun = { # activation functions 
        "tanh": torch.tanh,
    } 
    der_fun = { # derivatives 
        "tanh": lambda x: 1 - torch.tanh(x)**2
    }
    
    def __init__(self, in_channel=None, act="tanh"):
        """
        Args:
            in_channel : [input channel, doesn't really needed for element-wise flow, put it here for consistent interface]. Defaults to None.
            act : [type of activation function]. Defaults to "tanh".
        """
        super().__init__()
        self.act = ElementFlow.act_fun.get(act, "tanh")
        self.der = ElementFlow.der_fun.get(act, "tanh")
    
    def forward(self, x):
        """ x (Tensor of size (N, C, H, W) )
        """
        
        f_z = self.act(x)
        log_abs_det_jacobian = torch.sum(torch.log(torch.abs(self.der(x)) + EPS), dim=1)
        return f_z, log_abs_det_jacobian
    


        