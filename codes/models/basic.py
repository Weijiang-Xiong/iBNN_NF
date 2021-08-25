import torch
import torch.nn as nn
import torch.distributions as D 
import torch.nn.functional as F 

from .layers import StoLayer, StoLinear, StoConv2d
from typing import List, Tuple, Dict, Set

__all__ = ["StoModel", "LogisticRegression", "StoLogisticRegression", "MLP", "StoMLP", "LeNet", "StoLeNet"]

class StoModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.sto_layers: List[StoLayer] = []

    def forward(self, x):
        raise NotImplementedError

    def kl_div(self):
        return sum([m.kl_div() for m in self.sto_layers]) 
    
    def make_prediction(self, x, n_samples=128, return_all_probs=False):
        """ output normalized class probability and variances
        """
        # draw `n_samples` stochastic noise for each input and treat them as batches 
        x = x.repeat_interleave(n_samples, dim=0) 
        log_prob = self.forward(x) # don't expect the model to output normalized (log) probability
        log_prob = log_prob.reshape(-1, n_samples, log_prob.size(-1)) # result size (batch_size, n_samples, n_classes)
        probs = F.softmax(log_prob, dim=-1) # works even if the model outputs un-normalized log probability  
        # average over the samples, result size (batch_size, n_classes)
        variances, mean_prob = torch.var_mean(probs, dim=1, unbiased=False)
        if return_all_probs:
            return mean_prob, variances, probs
        else:
            return mean_prob, variances
          
    def build_all_flows(self, sto_cfg=None):
        
        # build the stochastic part for the stochastic layers (if not initialized before)
        if sto_cfg == None:
            return 
        for layer, cfg in zip(self.sto_layers, sto_cfg):
            if isinstance(layer, StoLinear):
                vec_len = layer.det_compo.in_features
            elif isinstance(layer, StoConv2d):
                vec_len = layer.det_compo.in_channels
            else:
                raise NotImplementedError
            dist_name, dist_params, flow_cfg=cfg
            layer.build_flow(vec_len, dist_name, dist_params, flow_cfg)    
            
    def calc_loss(self, log_probs, label):
        """
            log_probs : unnormalized log probability
        """
        # elbo =  log_likelihood - kl_divergence (both should be averaged over samples)
        # but log_likelihood is averaged over samples AND data points (cleaner code)
        log_likelihood = D.Categorical(logits=log_probs).log_prob(label).mean() 
        # log_likelihood = - F.cross_entropy(log_probs, label) # an alternative
        # so remember to divide kl by the number of data points 
        # maybe len(dataset), or len(dataloader)*dataloader.batch_size
        kl_divergence = self.kl_div()
        return log_likelihood, kl_divergence
    
    def det_and_sto_params(self):
        """ return the deterministic and stochastic parameters in the model
            both det_params and sto_params are list of nn.Parameters
            this implementation works for simple feed forward netwoks, where
            the layers are directly defined like `self.conv1 = nn.Conv2d(1,6,5)`
        """
        det_params, sto_params = [], []
        # _modules won't recursively find all submodules
        for name, layer in self._modules.items():
            if isinstance(layer, StoLayer):
                det_params.extend(list(layer.det_compo.parameters()))
                sto_params.extend(list(layer.norm_flow.parameters()))
            else:
                det_params.extend(list(layer.parameters()))
        return det_params, sto_params
    
    def migrate_from_det_model(self):
        raise NotImplementedError
    
    # disable stochastic parts in all layers
    def no_stochastic(self):
        for layer in self.sto_layers:
            layer.is_stochastic = False
    
    def stochastic(self):
        for layer in self.sto_layers:
            layer.is_stochastic = True
    
    # use fixed samples for inference or not 
    def use_fixed_samples(self):
        for layer in self.sto_layers:
            layer.use_fixed_samples = True
            
    def no_fixed_samples(self):
        for layer in self.sto_layers:
            layer.use_fixed_samples = False
    
    def clear_stored_samples(self):
        """ remember to clear the precomptued samples whenever the model updates
        """
        for layer in self.sto_layers:
            layer.stored_samples = None
      
class LogisticRegression(nn.Module):
    
    """ a logistic regression model for demonstration
    """
    
    def __init__(self, in_features:int, out_features:int, bias:bool=True):
        super(LogisticRegression, self).__init__()
        self.lin_layer = nn.Linear(in_features, out_features, bias)
        self.out_layer = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """ calculate log probability 
        """
        log_prob = self.out_layer(self.lin_layer(x))
        return log_prob

    def get_decision_boundary(self):
        return self._get_decision_boundary(self.lin_layer)
    
    @classmethod
    def _get_decision_boundary(cls, lin_layer):
        
        """ return a function for the decision boundary (a line on 2D plane)
        """
 
        assert lin_layer.in_features==2 and lin_layer.out_features==2

        # equation y = Ax + b, and y_0 = y_1, solve for x_0 = f(x_1)
        A = lin_layer.weight.data
        b = lin_layer.bias.data
        c = A[0,:] - A[1,:]
        d = b[0] - b[1]
        # then the decision boundary is c[0] * x_0 + c[1] * x_1 + d = 0, i.e., np.dot(c, x) + d = 0
        # general solution for higher dimension data could be found by scipy.linalg.null_space
        # but hard to visualize anyway 

        return lambda x_1: -1.0 * (d+c[1]*x_1) / c[0]    
    
class StoLogisticRegression(StoModel):
    
    # class of corresponding deterministic model
    DET_MODEL_CLASS = LogisticRegression 
    
    def __init__(self, in_features:int, out_features:int, bias:bool=True, sto_cfg=None):
        """ 
           
        """
        super(StoLogisticRegression, self).__init__()
        self.lin_layer = StoLinear(in_features, out_features, bias)
        self.out_layer = nn.LogSoftmax(dim=-1)
        self.sto_layers = [m for m in self.modules() if isinstance(m, StoLayer)]
        self.build_all_flows(sto_cfg=sto_cfg)

    def forward(self, x):
        """ calculate log probability of each class, and the log_det_jacobian for the stochastic samples 
        """
        log_prob = self.out_layer(self.lin_layer(x))
        # batched class probability and batched log_det_jacobian
        return log_prob

    def kl_div(self):
        
        return self.lin_layer.kl_div()
    
    def calc_loss(self, log_probs, label):
        # elbo =  log_likelihood - kl_divergence (both should be averaged over samples)
        # but log_likelihood is averaged over samples AND data points (cleaner code)
        log_likelihood = D.Categorical(logits=log_probs).log_prob(label).mean()
        # so divide kl by the number of data points
        kl_divergence = self.kl_div() / label.size(0)
        # minimize negative elbo
        return - log_likelihood + kl_divergence, log_likelihood, kl_divergence

    def get_decision_boundary(self):
        
        return self.__class__.DET_MODEL_CLASS._get_decision_boundary(self.lin_layer.det_compo)

    def migrate_from_det_model(self, det_model):
        det_class = self.DET_MODEL_CLASS
        if isinstance(det_model, det_class):
            self.lin_layer.migrate_from_det_layer(det_model.lin_layer)
            # self.lin_layer.det_compo.weight.data.copy_(det_model.lin_layer.weight.data)
            # self.lin_layer.det_compo.bias.data.copy_(det_model.lin_layer.bias.data)

class MLP(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features, bias):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias)
    
    def forward(self, x):
        return F.log_softmax(self.fc2(torch.sigmoid(self.fc1(x))), dim=-1)
    
class StoMLP(StoModel):
    
    DET_MODEL_CLASS = MLP
    
    def __init__(self, in_features, hidden_features, out_features, bias=True, sto_cfg:List[Tuple]=None):
        super().__init__()
        self.fc1 = StoLinear(in_features, hidden_features, bias)
        self.fc2 = StoLinear(hidden_features, out_features, bias)
        self.sto_layers = [m for m in self.modules() if isinstance(m, StoLayer)]
        self.build_all_flows(sto_cfg)

    def forward(self, x):
        return F.log_softmax(self.fc2(torch.sigmoid(self.fc1(x))), dim=-1)
            
    def migrate_from_det_model(self, det_model:MLP):
        det_class = self.DET_MODEL_CLASS
        if isinstance(det_model, det_class):
            self.fc1.migrate_from_det_layer(det_model.fc1)
            self.fc2.migrate_from_det_layer(det_model.fc2)

class LeNet(nn.Module):
    
    def __init__(self, colored=False):
        super(LeNet, self).__init__()
        # use 3 dimensions for colored images
        self.conv1 = nn.Conv2d(3, 6, 5) if colored else nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.ada_pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x:torch.Tensor):
        """
        Args:
          x of shape (batch_size, 1, H, W): Grey scale Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        device = x.device
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2), 2) # cell size (2,2), stride 2
        x = self.ada_pool(F.relu(self.conv2(x)))
        # flatten the features, but keep the batch dimension
        x = x.flatten(start_dim=1, end_dim=-1)
        # x = x.view(-1, int(torch.prod(torch.Tensor(list(x.size())[1:])))) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # use F.log_softmax() depending on needs
        return x.to(device)
    
class StoLeNet(StoModel):

    DET_MODEL_CLASS = LeNet
    
    def __init__(self, sto_cfg:List[Tuple], colored=False):
        super().__init__()
        # use 3 dimensions for colored images
        self.conv1 = StoConv2d(3, 6, 5) if colored else StoConv2d(1, 6, 5)
        self.conv2 = StoConv2d(6, 16, 5)
        self.ada_pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = StoLinear(16*4*4, 120)
        self.fc2 = StoLinear(120, 84)
        self.fc3 = StoLinear(84, 10)
        self.sto_layers = [m for m in self.modules() if isinstance(m, StoLayer)]
        self.build_all_flows(sto_cfg)
            
    def forward(self, x:torch.Tensor):
        """
        Args:
          x of shape (batch_size, 1, H, W): Grey scale Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        device = x.device
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2), 2) # cell size (2,2), stride 2
        x = self.ada_pool(F.relu(self.conv2(x)))
        # flatten the features, but keep the batch dimension
        x = x.flatten(start_dim=1, end_dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # use F.log_softmax() depending on needs
        return x.to(device)

    def kl_div(self):
        return sum([m.kl_div() for m in self.sto_layers]) 
    
    def migrate_from_det_model(self, det_model):
        # TODO try to monitor how the deterministic weights changes 
        det_class = self.DET_MODEL_CLASS
        if isinstance(det_model, det_class):
            # _modules won't recursively find all submodules
            # but named_modules() will, and thus it includes the flows (should not happen) 
            for name, layer in self._modules.items():
                if name == "": # skip empty names 
                    continue
                if isinstance(layer, StoLayer):
                    layer.migrate_from_det_layer(getattr(det_model, name))
                elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    layer.weight.data.copy_(getattr(det_model, name).weight.data)
                    layer.bias.data.copy_(getattr(det_model, name).bias.data)

