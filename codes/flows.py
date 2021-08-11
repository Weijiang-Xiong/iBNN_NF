import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as D

import numpy as np 

EPS = np.finfo(np.float32).eps
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
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0

        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        v_hat = self.v
        if normalize_u:
            wtu = (self.w @ self.v.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            v_hat = self.v + (m_wtu - wtu) * self.w / (self.w @ self.w.t())

        # compute transform
        wtz_plus_b = z @ self.w.t() + self.b
        f_z = z + v_hat * torch.tanh(wtz_plus_b)
        # compute log_abs_det_jacobian
        psi = (1 - torch.tanh(wtz_plus_b)**2) @ self.w
        det = 1 + psi @ v_hat.t()
        log_abs_det_jacobian = torch.log(torch.abs(det) + EPS).squeeze()
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian

        return f_z, sum_log_abs_det_jacobians

class AffineTransform(nn.Module):
    """ will keep the input unchanged if not learnable 
    """
    def __init__(self, vec_len=2, learnable=True):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(vec_len)).requires_grad_(learnable)
        self.logsigma = nn.Parameter(torch.zeros(vec_len)).requires_grad_(learnable)

    def forward(self, x):
        z = self.mu + self.logsigma.exp() * x
        sum_log_abs_det_jacobians = self.logsigma.sum()
        return z, sum_log_abs_det_jacobians
    
class PlanarFlow2d(nn.Module):
    
    def __init__(self, in_channel, init_sigma=0.01):
        super().__init__()
        self.in_channel = in_channel
        self.v = nn.Parameter(torch.randn(1, in_channel).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, in_channel).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))
        
    def forward(self, x, normalize_u=True):
        """
        Args:
            x ([tuple]): consists of two tensors 
                        1. input image/feature z of size (N, C, H, W)
                        2. sum_log_abs_det_jacobians of size (N, H, W)
        """
        
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0    
            
        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        v_hat = self.v
        if normalize_u:
            wtu = (self.w @ self.v.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            v_hat = self.v + (m_wtu - wtu) * self.w / (self.w @ self.w.t())
        
        # treat z as N*H*W different length-C vectors, apply planar transform to each of them
        # which is equivalent to a 2d convolution
        wtz_plus_b = F.conv2d(z, self.w.view(1, self.in_channel, 1, 1)) + self.b
        f_z = z + v_hat.view(1, self.in_channel, 1, 1) * F.tanh(wtz_plus_b)
        det = 1 + (1-F.tanh(wtz_plus_b)**2) * (self.w @ v_hat.t())
        
        log_abs_det_jacobian = torch.log(torch.abs(det) + EPS).squeeze()
        sum_log_abs_det_jacobians += log_abs_det_jacobian
        
        return f_z, sum_log_abs_det_jacobians

class ElementFlow(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass 
    

class NF_Block(nn.Module):

    flow_types = {
        "affine": AffineTransform,
        "planar": PlanarFlow,
        "planar2d": PlanarFlow2d,
    }
    
    def __init__(self, vec_len=2, flow_cfg=None):
        super(NF_Block, self).__init__()
        # self.forward_block = nn.Sequential(AffineTransform(vec_len=vec_len, learnable=True), 
        #                             *[PlanarFlow(vec_len=vec_len, init_sigma=0.01) for _ in range(flow_cfg)])
        self.forward_block = nn.Sequential()
        for cfg in flow_cfg:
            if len(cfg)==2:
                name, depth = cfg
                params = {} # use defaults 
            elif len(cfg) == 3:
                name, depth, params = cfg
            else:
                raise NotImplementedError("unknown configuration format")
            
            for idx in range(depth):
                flow_name = "{}_{}".format(name, idx)
                flow_layer = self.flow_types[name](vec_len, **params)
                self.forward_block.add_module(flow_name, flow_layer)
        
        # print("")
    
    def forward(self, samples):

        transformed_samples, log_det_jacobian = self.forward_block(samples)

        return transformed_samples, log_det_jacobian
    
def test_2d_planar_flow():

    data = torch.randn(4,3,8,8)
    planar_1d = PlanarFlow(vec_len=3)
    planar_2d = PlanarFlow2d(in_channel=3)
    
    # copy the weights from 1d flow to 2d flow 
    planar_2d.w.data.copy_(planar_1d.w.data)
    planar_2d.v.data.copy_(planar_1d.v.data)
    planar_2d.b.data.copy_(planar_1d.b.data) 
    
    # use the 1d flow to iterate over height and weight dimension 
    out_1d = torch.zeros_like(data)
    ldj_1d = torch.zeros(4,8,8)
    for i in range(data.shape[-2]):
        for j in range(data.shape[-1]):
            f_z_ij, ldj_ij = planar_1d(data[:,:,i,j])
            out_1d[:,:,i,j] = f_z_ij
            ldj_1d[:,i,j] = ldj_ij 
    
    # use 2d flow to transform the data from beginning and compare two results
    out_2d, ldj_2d = planar_2d(data)
    
    if torch.allclose(out_1d, out_2d) and torch.allclose(ldj_1d, ldj_2d):
        print("Test Pass: 2D flow is consistent with iteratively applying 1D flow")
    else:
        print("Test Fail: 2D flow is NOT consistent with iteratively applying 1D flow")

def test_flow_cfg_format():
    flow_cfg = [("affine", 1, {"learnable":True}), # the first stack of flows (type, depth, params)
                ("planar2d", 8, {"init_sigma":0.01})] # the second stack of flows (type, depth, params)
    norm_flow = NF_Block(vec_len=16, flow_cfg=flow_cfg)
    print(norm_flow)

if __name__ == "__main__":
    test_2d_planar_flow()
    test_flow_cfg_format()

        
        
        
        