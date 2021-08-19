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
        self.base_dist:D.Distribution = None
        self.norm_flow:NF_Block = None
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
        device = x.device
        if self.is_stochastic:
            base_samples = self.base_dist.sample(x.shape).to(device)
            transformed_samples, log_det_jacobian = self.norm_flow(base_samples)
            out = self.det_compo(x*transformed_samples)
            # store mean instead of the whole tensor
            self.mean_log_det_jacobian = log_det_jacobian.mean()
            # mean probability of base and transformed samples over the base distribution
            # average over n_samples samples, sum over n_features (independent) elements in a sample vector 
            self.mean_sample_prob = self.base_dist.log_prob(base_samples).sum(dim=1).mean()
            self.mean_transf_prob = self.base_dist.log_prob(transformed_samples).sum(dim=1).mean() 
        else:
            out = self.det_compo(x)

        return out.to(device)

    def kl_div(self):
        if self.is_stochastic:
            kl = self.mean_sample_prob - self.mean_log_det_jacobian - self.mean_transf_prob
        else:
            kl = 0
        return kl

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
        device = x.device   
        if self.is_stochastic:
            base_samples = self.base_dist.sample(x.shape).to(device)
            transformed_samples, log_det_jacobian = self.norm_flow(base_samples)
            out = self.det_compo(x*transformed_samples)
            # store mean instead of the whole tensor
            self.mean_log_det_jacobian = log_det_jacobian.mean()
            # average over different samples
            # sum over elements in a sample vector of a random variable
            self.mean_sample_prob = self.base_dist.log_prob(base_samples).sum(dim=1).mean()
            self.mean_transf_prob = self.base_dist.log_prob(transformed_samples).sum(dim=1).mean() 
        else:
            out = self.det_compo(x)
        return out.to(device)

    def kl_div(self):
        if self.is_stochastic:
            kl = self.mean_sample_prob - self.mean_log_det_jacobian - self.mean_transf_prob
        else:
            kl = 0
        return kl

    def migrate_from_det_layer(self, det_layer: nn.Conv2d):
        if isinstance(det_layer, self.__class__.DET_CLASS):
            self.det_compo.weight.data.copy_(det_layer.weight.data)
            self.det_compo.bias.data.copy_(det_layer.bias.data)
