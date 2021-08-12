import torch
import torch.nn as nn
import torch.distributions as D

from flows import NF_Block


class StoLayer(nn.Module):

    dist_types = {
        None: None,
        "normal": D.Normal
    }

    def __init__(self):
        super(StoLayer, self).__init__()
        self.det_compo = None
        self.base_dist = None
        self.norm_flow = None
        self.mean_log_det_jacobian = None  # to compute KL divergence
        # is_stochastic indicates if a proper stochastic part has been initialized
        # and it can be used to disable stochastic part (maybe useful?)
        self.is_stochastic = False

    def build_flow(self, vec_len, dist_name:str, dist_params:dict, flow_cfg):
        """
        Args:
            in_features, out_features, bias: same as nn.Linear
            dist_name (str): name of distribution, check StoLayer.dist_types, for example "normal"
            dist_params (dict): parameters to initialize the distribution, for example {"loc":0, "scale":1}
            flow_cfg (??): configuration of the normalizing flow 
        """
        # if the flow was not initialized when creating this layer
        if not self.is_stochastic:
            # unpack `dist_params` as a dict, therefore the keys have to be the
            # same as input arguments accepted by the distribution class
            self.base_dist = self.__class__.dist_types[dist_name.lower()](**dist_params)
            self.norm_flow = NF_Block(vec_len, flow_cfg)
            self.is_stochastic = True

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
        if all([dist_name, dist_params, flow_cfg]):
            self.build_flow(in_features, dist_name, dist_params, flow_cfg)

    def forward(self, x: torch.Tensor):
        """ generate and transform stochastic samples, and calculate log_det_jacobian
            x : size (n_samples, n_features)
        """
        if self.is_stochastic:
            mult_noise = self.base_dist.sample(x.shape)
            transformed_noise, log_det_jacobian = self.norm_flow(mult_noise)
            out = self.det_compo(x*transformed_noise)
            # store mean instead of the whole tensor
            self.mean_log_det_jacobian = log_det_jacobian.mean()
        else:
            out = self.det_compo(x)

        return out

    def kl_div(self):
        return - self.mean_log_det_jacobian if self.is_stochastic else 0

    def migrate_from_det_layer(self, base_layer: nn.Linear):
        # copy the weight and bias from a deterministic Linear layer
        if isinstance(base_layer, self.__class__.DET_CLASS):
            self.det_compo.weight.data.copy_(base_layer.weight.data)
            self.det_compo.bias.data.copy_(base_layer.bias.data)


class StoConv2d(StoLayer):

    DET_CLASS = nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', dist_name=None, dist_params=None, flow_cfg=None):
        super().__init__()
        self.det_compo = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        # to correctly configure the stochastic part, all these has to be set
        if all([dist_name, dist_params, flow_cfg]):
            self.build_flow(in_channels, dist_name, dist_params, flow_cfg)

    def forward(self, x):
        """ generate and transform stochastic samples, and calculate log_det_jacobian
            x ([Tensor]): size (N, C, H, W)
        """
        if self.is_stochastic:
            mult_noise = self.base_dist.sample(x.shape)
            transformed_noise, log_det_jacobian = self.norm_flow(mult_noise)
            out = self.det_compo(x*transformed_noise)
            # store mean instead of the whole tensor
            self.mean_log_det_jacobian = log_det_jacobian.mean()
        else:
            out = self.det_compo(x)
        return out

    def kl_div(self):
        return - self.mean_log_det_jacobian if self.is_stochastic else 0

    def migrate_from_det_layer(self, det_layer: nn.Conv2d):
        if isinstance(det_layer, self.__class__.DET_CLASS):
            self.det_compo.weight.data.copy_(det_layer.weight.data)
            self.det_compo.bias.data.copy_(det_layer.bias.data)


def test_linear_migration():

    # type and parameters of the distribution,
    # TODO parse configuration of the flow
    dist_name = "normal"
    dist_params = {"loc": 0, "scale": 1}
    flow_cfg = [("affine", 1, {"learnable": True}),  # the first stack of flows (type, depth, params)
                ("planar2d", 8, {"init_sigma": 0.01})]

    base_layer = nn.Linear(2, 2)
    sto_layer = StoLinear(2, 2, dist_name=dist_name, dist_params=dist_params, flow_cfg=flow_cfg)
    sto_layer.migrate_from_det_layer(base_layer)
    print(list(base_layer.named_parameters()))
    print(list(sto_layer.det_compo.named_parameters()))

    base_weight = base_layer.weight.data
    base_bias = base_layer.bias.data

    sto_weight = sto_layer.det_compo.weight.data
    sto_bias = sto_layer.det_compo.bias.data

    # disable the stochastic part
    sto_layer.is_stochastic = False
    in_data = torch.randn(5, 2)
    base_out = base_layer(in_data)
    sto_out = sto_layer(in_data)

    cond1 = torch.allclose(base_weight, sto_weight)
    cond2 = torch.allclose(base_bias, sto_bias)
    cond3 = torch.allclose(base_out, sto_out)

    if all([cond1, cond2, cond3]):
        print("Linear Layer: Weight Migration Successful")
    else:
        print("Linear Layer: Weight Migration Failed")

def test_conv_migration():
    dist_name = "normal"
    dist_params = {"loc": 0, "scale": 1}
    flow_cfg = [("affine", 1, {"learnable": True}),  # the first stack of flows (type, depth, params)
                ("planar2d", 8, {"init_sigma": 0.01})]

    base_layer = nn.Conv2d(3,4,2)
    sto_layer = StoConv2d(3,4,2, dist_name=dist_name, dist_params=dist_params, flow_cfg=flow_cfg)
    sto_layer.migrate_from_det_layer(base_layer)
    print(list(base_layer.named_parameters()))
    print(list(sto_layer.det_compo.named_parameters()))
    
    base_weight = base_layer.weight.data
    base_bias = base_layer.bias.data

    sto_weight = sto_layer.det_compo.weight.data
    sto_bias = sto_layer.det_compo.bias.data
    
    sto_layer.is_stochastic = False
    in_data = torch.randn(5, 3, 10, 10)
    base_out = base_layer(in_data)
    sto_out = sto_layer(in_data)
    
    cond1 = torch.allclose(base_weight, sto_weight)
    cond2 = torch.allclose(base_bias, sto_bias)
    cond3 = torch.allclose(base_out, sto_out)
    
    if all([cond1, cond2, cond3]):
        print("StoConv2d: Weight Migration Successful")
    else:
        print("StoConv2d: Weight Migration Failed")
if __name__ == "__main__":
    test_linear_migration()
    test_conv_migration()
    pass
