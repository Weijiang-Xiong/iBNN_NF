"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2

code credit to https://github.com/kamenbliznashki/normalizing_flows
actually only the "step of flow" is ported here
"""
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributions as D 
from torch.utils.checkpoint import checkpoint

EPS = 1e-8

class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,3,1,1), keepdim=True):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())
        self.keepdim = keepdim
    def forward(self, x):
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, False) + EPS).view_as(self.bias)
            self.initialized += 1

        z = (x - self.bias) / (self.scale + EPS)
        logdet = - self.scale.abs().log().sum()
        if not self.keepdim:
            logdet = logdet * z.shape[2] * z.shape[3]
        return z, logdet

    def inverse(self, z):
        f_z = z * self.scale + self.bias
        logdet = self.scale.abs().log().sum()
        if not self.keepdim:
            logdet = logdet * z.shape[2] * z.shape[3]
        return f_z, logdet


class Invertible1x1Conv(nn.Module):
    """ Invertible 1x1 convolution layer; cf Glow section 3.2 """
    def __init__(self, n_channels=3, lu_factorize=False, keepdim=True):
        super().__init__()
        self.lu_factorize = lu_factorize    
        self.keepdim = keepdim
        # initiaize a 1x1 convolution weight matrix
        w = torch.randn(n_channels, n_channels)
        w = torch.qr(w)[0]  # note: nn.init.orthogonal_ returns orth matrices with dets +/- 1 which complicates the inverse call below

        if lu_factorize:
            # compute LU factorization
            p, l, u = torch.lu_unpack(*w.unsqueeze(0).lu())
            # initialize model parameters
            self.p, self.l, self.u = nn.Parameter(p.squeeze()), nn.Parameter(l.squeeze()), nn.Parameter(u.squeeze())
            s = self.u.diag()
            self.log_s = nn.Parameter(s.abs().log())
            self.register_buffer('sign_s', s.sign())  # note: not optimizing the sign; det W remains the same sign
            self.register_buffer('l_mask', torch.tril(torch.ones_like(self.l), -1))  # store mask to compute LU in forward/inverse pass
        else:
            self.w = nn.Parameter(w)

    def forward(self, x):
        B,C,H,W = x.shape
        if self.lu_factorize:
            l = self.l * self.l_mask + torch.eye(C).to(self.l.device)
            u = self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp())
            self.w = self.p @ l @ u
            logdet = self.log_s.sum() 
        else:
            logdet = torch.slogdet(self.w)[-1] 
        
        if not self.keepdim:
            logdet = logdet * H * W
            
        return F.conv2d(x, self.w.view(C,C,1,1)), logdet
    
    def inverse(self, z):
        B,C,H,W = z.shape
        if self.lu_factorize:
            l = torch.inverse(self.l * self.l_mask + torch.eye(C).to(self.l.device))
            u = torch.inverse(self.u * self.l_mask.t() + torch.diag(self.sign_s * self.log_s.exp()))
            w_inv = u @ l @ self.p.inverse()
            logdet = - self.log_s.sum()
        else:
            w_inv = self.w.inverse()
            logdet = - torch.slogdet(self.w)[-1]
            
        if not self.keepdim:
            logdet = logdet * H * W
            
        return F.conv2d(z, w_inv.view(C,C,1,1)), logdet


class AffineCoupling(nn.Module):
    """ Affine coupling layer; cf Glow section 3.3; RealNVP figure 2 """
    def __init__(self, n_channels, width, keepdim=True):
        super().__init__()
        # width proportional to channel number, minimal 3
        n_width = torch.clamp(torch.tensor([n_channels*width]), min=3).type(torch.int)
        # network layers;
        # per realnvp, network splits input, operates on half of it, and returns shift and scale of dim = half the input channels
        self.conv1 = nn.Conv2d(n_channels//2, n_width, kernel_size=3, padding=1, bias=False)  # input is split along channel dim
        self.actnorm1 = Actnorm(param_dim=(1, n_width, 1, 1))
        self.conv2 = nn.Conv2d(n_width, n_width, kernel_size=1, padding=1, bias=False)
        self.actnorm2 = Actnorm(param_dim=(1, n_width, 1, 1))
        self.conv3 = nn.Conv2d(n_width, n_channels, kernel_size=3)            # output is split into scale and shift components
        # learned scale (cf RealNVP sec 4.1 / Glow official code
        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels,1,1))   
        self.keepdim = keepdim

        # initialize last convolution with zeros, such that each affine coupling layer performs an identity function
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()

    def forward(self, x:torch.Tensor):
        x_a, x_b = x.chunk(2, 1)  # split along channel dim

        h = F.relu(self.actnorm1(self.conv1(x_b))[0])
        h = F.relu(self.actnorm2(self.conv2(h))[0])
        h = self.conv3(h) * self.log_scale_factor.exp()
        t = h[:,0::2,:,:]  # shift; take even channels
        s = h[:,1::2,:,:]  # scale; take odd channels
        s = torch.sigmoid(s + 2.)  # at initalization, s is 0 and sigmoid(2) is near identity

        z_a = s * x_a + t
        z_b = x_b
        z = torch.cat([z_a, z_b], dim=1)  # concat along channel dim
        
        if self.keepdim:
            logdet = s.log().sum([1])
        else:
            logdet = s.log().sum([1, 2, 3])

        return z, logdet



# --------------------
# Container layers
# --------------------

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def __init__(self, *args, **kwargs):
        self.checkpoint_grads = kwargs.pop('checkpoint_grads', None)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        sum_logdets = 0.
        for module in self:
            x, logdet = module(x) if not self.checkpoint_grads else checkpoint(module, x)
            sum_logdets = sum_logdets + logdet
        return x, sum_logdets

    def inverse(self, z):
        sum_logdets = 0.
        for module in reversed(self):
            z, logdet = module.inverse(z)
            sum_logdets = sum_logdets + logdet
        return z, sum_logdets


class FlowStep(FlowSequential):
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """
    def __init__(self, n_channels, width, lu_factorize=False, keepdim=True):
        super().__init__(Actnorm(param_dim=(1,n_channels,1,1), keepdim=keepdim),
                         Invertible1x1Conv(n_channels, lu_factorize, keepdim),
                         AffineCoupling(n_channels, width, keepdim))
