import torch
import torch.nn as nn
import torch.distributions as D 
import torch.nn.functional as F 

from layers import StoLayer, StoLinear, StoConv2d

class LogisticRegression(nn.Module):
    
    """ a logistic regression model for demonstration
    """
    
    def __init__(self, in_features, out_features, bias):
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
    
class StoLogisticRegression(nn.Module):
    
    # class of corresponding deterministic model
    DET_MODEL_CLASS = LogisticRegression 
    
    def __init__(self, in_features, out_features, bias, flow_cfg=16):
        """ 
           
        """
        super(StoLogisticRegression, self).__init__()
        self.lin_layer = StoLinear(in_features, out_features, bias, 
                                   dist_name="normal", dist_params= {"loc":1.0, "scale":0.5},
                                   flow_cfg=flow_cfg)
        self.out_layer = nn.LogSoftmax()
        self.sto_layers = [m for m in self.modules() if isinstance(m, StoLayer)]

    def forward(self, x):
        """ calculate log probability of each class, and the log_det_jacobian for the stochastic samples 
        """
        log_prob = self.out_layer(self.lin_layer(x))
        # batched class probability and batched log_det_jacobian
        return log_prob
    
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

    def kl_div(self):
        
        return self.lin_layer.kl_div()
    
    def calc_loss(self, log_probs, label):
        # elbo =  log_likelihood - kl_divergence (both averaged over samples)
        # log_likelihood is averaged over samples and data points 
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

class LeNet(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass 
    
class StoLeNet(nn.Module):

    DET_MODEL_CLASS = LeNet
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass 
    
    def migrate_from_det_model(self, det_model):
        pass 
    
    def calc_loss(self, x):
        pass 
    

if __name__ == "__main__":
    
    model = LogisticRegression(2,2,True)
    sto_model = StoLogisticRegression(2,2,True)
    sto_model.migrate_from_det_model(model)
    fn = sto_model.get_decision_boundary()
    print(model)
