import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as D

import numpy as np 
from typing import List, Tuple, Dict, Set

from .basic import AffineTransform, PlanarFlow, PlanarFlow2d, ElementFlow

class NF_Block(nn.Module):
    """ a container class for a series of flows in a stochastic layer 
    """
    flow_types = {
        "affine": AffineTransform,
        "planar": PlanarFlow,
        "planar2d": PlanarFlow2d,
        "element": ElementFlow,
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