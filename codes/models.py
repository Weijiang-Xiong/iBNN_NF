import torch
import torch.nn as nn
import torch.distributions as D 
import torch.nn.functional as F 

from layers import StoLayer, StoLinear, StoConv2d
from typing import List, Tuple, Dict, Set

class StoModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.sto_layers: List[StoLayer] = []

    def forward(self, x):
        raise NotImplementedError

    def kl_div(self):
        return sum([m.kl_div() for m in self.sto_layers]) 
    
    def make_prediction(self, x, n_samples=128, return_all_probs=False):
        # draw `n_samples` stochastic noise for each input and treat them as batches 
        x = x.repeat_interleave(n_samples, dim=0) 
        log_prob = self.forward(x)
        log_prob = log_prob.reshape(-1, n_samples, x.size(1)) # result size (batch_size, n_samples, n_classes)
        probs = log_prob.exp()
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
        # elbo =  log_likelihood - kl_divergence (both should be averaged over samples)
        # but log_likelihood is averaged over samples AND data points (cleaner code)
        log_likelihood = D.Categorical(logits=log_probs).log_prob(label).mean()
        # so remember to divide kl by the number of data points 
        # maybe len(dataset), or len(dataloader)*dataloader.batch_size
        kl_divergence = self.kl_div()
        return log_likelihood, kl_divergence    
    
    def migrate_from_det_model(self):
        raise NotImplementedError
    
    def no_stochastic(self):
        for layer in self.sto_layers:
            layer.is_stochastic = False
    
    def stochastic(self):
        for layer in self.sto_layers:
            layer.is_stochastic = True
      
      
class LogisticRegression(nn.Module):
    
    """ a logistic regression model for demonstration
    """
    
    def __init__(self, in_features:int, out_features:int, bias:bool=True):
        super(LogisticRegression, self).__init__()
        self.lin_layer = nn.Linear(in_features, out_features, bias)
        self.out_layer = nn.LogSoftmax()

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
        self.out_layer = nn.LogSoftmax()
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
        return F.log_softmax(self.fc2(F.sigmoid(self.fc1(x))), dim=-1)
    
class StoMLP(StoModel):
    
    DET_MODEL_CLASS = MLP
    
    def __init__(self, in_features, hidden_features, out_features, bias=True, sto_cfg:List[Tuple]=None):
        super().__init__()
        self.fc1 = StoLinear(in_features, hidden_features, bias)
        self.fc2 = StoLinear(hidden_features, out_features, bias)
        self.sto_layers = [m for m in self.modules() if isinstance(m, StoLayer)]
        self.build_all_flows(sto_cfg)

    def forward(self, x):
        return F.log_softmax(self.fc2(F.sigmoid(self.fc1(x))), dim=-1)
            
    def migrate_from_det_model(self, det_model:MLP):
        det_class = self.DET_MODEL_CLASS
        if isinstance(det_model, det_class):
            self.fc1.migrate_from_det_layer(det_model.fc1)
            self.fc2.migrate_from_det_layer(det_model.fc2)
class LeNet(nn.Module):
    
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2), 2) # cell size (2,2), stride 2
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2), 2)
        # flatten the features, but keep the batch dimension
        x = x.view(-1, int(torch.prod(torch.Tensor(list(x.size())[1:])))) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class StoLeNet(StoModel):

    DET_MODEL_CLASS = LeNet
    
    def __init__(self, sto_cfg:List[Tuple]):
        super().__init__()
        self.conv1 = StoConv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = StoLinear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.sto_layers = [m for m in self.modules() if isinstance(m, StoLayer)]
        self.build_all_flows(sto_cfg)
            
    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2), 2) # cell size (2,2), stride 2
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2), 2)
        # flatten the features, but keep the batch dimension
        x = x.view(-1, int(torch.prod(torch.Tensor(list(x.size())[1:])))) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

def test_model_initialization():
    sto_model_cfg = [
                ("normal", {"loc":1.0, "scale":0.5},  # the name of base distribution and parameters for that distribution
                    [("affine", 1, {"learnable":True}), # the first stack of flows (type, depth, params)
                    ("planar2d", 8, {"init_sigma":0.01})] # the second stack of flows (type, depth, params)
                )]
    base_logistic = LogisticRegression(2,2,True)
    sto_logistic = StoLogisticRegression(2,2,True, sto_cfg=sto_model_cfg)
    sto_logistic.migrate_from_det_model(base_logistic)
    fn = sto_logistic.get_decision_boundary()
    
    data = torch.randn(8, 2)
    base_out1 = base_logistic(data)
    sto_logistic.no_stochastic()
    sto_out1 = sto_logistic(data)
    cond1 = torch.allclose(base_out1, sto_out1)
    # print(sto_model)
    
    # looks really ugly, 
    sto_model_cfg = [
                ("normal", {"loc":1.0, "scale":0.5},  # the name of base distribution and parameters for that distribution
                    [("affine", 1, {"learnable":True}), # the first stack of flows (type, depth, params)
                    ("planar2d", 6, {"init_sigma":0.01})] # the second stack of flows (type, depth, params)
                ),
                (
                    "normal", {"loc":1.0, "scale":0.5}, 
                    [("affine", 1), 
                     ("planar", 6)]
                )
                ]
    base_mlp = MLP(2, 10, 2, True)
    sto_mlp = StoMLP(2, 10, 2, True, sto_cfg=sto_model_cfg)
    sto_mlp.migrate_from_det_model(base_mlp)
    
    data = torch.randn(8, 2)
    base_out2 = base_mlp(data)
    sto_mlp.no_stochastic()
    sto_out2 = sto_mlp(data)
    cond2 = torch.allclose(base_out2, sto_out2)
    
    base_lenet = LeNet()
    sto_lenet = StoLeNet(sto_model_cfg)
    sto_lenet.migrate_from_det_model(base_lenet)
    
    data = torch.randn(8,1,28,28)
    base_out3 = base_lenet(data)
    sto_lenet.no_stochastic()
    sto_out3 = sto_lenet(data)
    cond3 = torch.allclose(base_out3, sto_out3)
    
    if all([cond1, cond2, cond3]):
        print("Test Pass: Model Initialization and Migration")
    else:
        print("Test Fail: Model Initialization and Migration")
    
if __name__ == "__main__":
    
    test_model_initialization()
    pass 
