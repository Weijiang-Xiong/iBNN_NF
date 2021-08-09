import torch 
import torch.nn as nn 
import torch.distributions as D

from flows import NF_Block
    
class StoLayer(nn.Module):
    
    dist_types = {
        "normal": D.Normal
    }

    def __init__(self):
        super(StoLayer, self).__init__()
        self.det_compo = None # base on a deterministic component
        self.base_dist = None # a distribution to draw samples  
        self.norm_flow = None # a flow to transform the samples
        self.mean_log_det_jacobian = None # to compute KL divergence
    
    # def kl_div(self):
    #     return - self.mean_log_det_jacobian

class StoLinear(StoLayer):
    
    DET_CLASS = nn.Linear
    
    def __init__(self, in_features, out_features, bias=True, dist_name=None, dist_params=None, flow_cfg=None):
        """
        Args:
            in_features, out_features, bias: same as nn.Linear
            dist_name (str): name of distribution, check StoLayer.dist_types, for example "normal"
            dist_params (dict): parameters to initialize the distribution, for example {"loc":0, "scale":1}
            flow_cfg (??): configuration of the normalizing flow 
        """
        super().__init__()
        self.det_compo = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        # unpack `dist_params` as a dict, the keys have to be the same as input arguments accepted by the distribution
        self.base_dist = self.__class__.dist_types[dist_name](**dist_params)
        self.norm_flow = NF_Block(flow_cfg)
        
    def forward(self, x:torch.Tensor):
        """ generate and transform stochastic samples, and calculate log_det_jacobian
            x : size (n_samples, n_features)
        """
        mult_noise = self.base_dist.sample(x.shape)
        transformed_noise, log_det_jacobian = self.norm_flow(mult_noise)
        out = self.det_compo(x*transformed_noise)
        # store mean instead of the whole tensor 
        self.mean_log_det_jacobian = log_det_jacobian.mean()
        
        return out
    
    def kl_div(self):
        return - self.mean_log_det_jacobian
    
    def migrate_from_det_layer(self, base_layer:nn.Linear):
        # copy the weight and bias from a deterministic Linear layer
        if isinstance(base_layer, self.__class__.DET_CLASS):
            self.det_compo.weight.data.copy_(base_layer.weight.data)
            self.det_compo.bias.data.copy_(base_layer.bias.data)
        
class StoConv2d(StoLayer):
    
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None,
                 dist_name=None, dist_params=None, flow_cfg=None):
        super().__init__()
        self.det_compo = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                        dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.base_dist = self.__class__.dist_types[dist_name](**dist_params)
        self.norm_flow = NF_Block(flow_cfg)
    
    def forward(self, x):

        """ generate and transform stochastic samples, and calculate log_det_jacobian
            x ([Tensor]): size (N, C, H, W)
        """
        
        mult_noise = self.base_dist.sample(x.shape)
        transformed_noise, log_det_jacobian = self.norm_flow(mult_noise)
        out = self.det_compo(x*transformed_noise)
        # store mean instead of the whole tensor 
        self.mean_log_det_jacobian = log_det_jacobian.mean()
        
        return out
    
    def kl_div(self):
        return - self.mean_log_det_jacobian

def test_linear_migration():
    
    # type and parameters of the distribution, 
    # TODO parse configuration of the flow
    dist_name = "normal"
    dist_params = {"loc":0, "scale":1}
    flow_cfg=8
    
    base_layer = nn.Linear(2,2)
    sto_layer = StoLinear(2,2, dist_name=dist_name, dist_params=dist_params, flow_cfg=flow_cfg)
    sto_layer.migrate_from_det_model(base_layer)
    print(list(base_layer.named_parameters()))
    print(list(sto_layer.det_compo.named_parameters()))  
    
    base_weight = base_layer.weight.data
    base_bias = base_layer.bias.data
    
    sto_weight = sto_layer.det_compo.weight.data 
    sto_bias = sto_layer.det_compo.bias.data 
    
    if torch.allclose(base_weight, sto_weight) and torch.allclose(base_bias, sto_bias):
        print("Linear Layer: Weight Migration Successful")
    else:
        print("Linear Layer: Weight Migration Failed")

        
if __name__ == "__main__":
    test_linear_migration()
    pass 