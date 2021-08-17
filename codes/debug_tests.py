import torch 
import torch.nn as nn 

from typing import List, Tuple, Dict, Set
from layers import StoLayer, StoLinear, StoConv2d
from models import StoModel, MLP, StoMLP, LeNet, StoLeNet, LogisticRegression, StoLogisticRegression
from flows import NF_Block, PlanarFlow, AffineTransform, PlanarFlow2d, ElementFlow
from utils import ECELoss
 
def test_linear_migration():

    # type and parameters of the distribution,
    dist_name = "normal"
    dist_params = {"loc": 0, "scale": 1}
    flow_cfg = [("affine", 1, {"learnable": True}),  # the first stack of flows (type, depth, params)
                ("planar2d", 8, {"init_sigma": 0.01})]

    base_layer = nn.Linear(2, 2)
    sto_layer = StoLinear(2, 2, dist_name=dist_name, dist_params=dist_params, flow_cfg=flow_cfg)
    sto_layer.migrate_from_det_layer(base_layer)
    # print(list(base_layer.named_parameters()))
    # print(list(sto_layer.det_compo.named_parameters()))

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
        raise ValueError("Linear Layer: Weight Migration Failed")

def test_conv_migration():
    dist_name = "normal"
    dist_params = {"loc": 0, "scale": 1}
    flow_cfg = [("affine", 1, {"learnable": True}),  # the first stack of flows (type, depth, params)
                ("planar2d", 8, {"init_sigma": 0.01})]

    base_layer = nn.Conv2d(3,4,2)
    sto_layer = StoConv2d(3,4,2, dist_name=dist_name, dist_params=dist_params, flow_cfg=flow_cfg)
    sto_layer.migrate_from_det_layer(base_layer)
    # print(list(base_layer.named_parameters()))
    # print(list(sto_layer.det_compo.named_parameters()))
    
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
        raise ValueError("StoConv2d: Weight Migration Failed")
        

def test_optimizer():
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
    sto_model = StoLeNet(sto_cfg=sto_model_cfg)
    det_params, sto_params = sto_model.det_and_sto_params()
    optimizer = torch.optim.Adam([
                    {'params': det_params, 'lr': 1e-4},
                    {'params': sto_params, 'lr': 1e-3}
                ])
    print("Test Pass: Parameters accepted by optimizer")

def test_cuda_forward():
    device = torch.device("cuda")
    base_model = LeNet().to(device)
    
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
    sto_model = StoLeNet(sto_cfg=sto_model_cfg).to(device)
    data = torch.randn(10, 1, 28, 28).to(device)
    base_out = base_model(data)
    sto_out = sto_model(data)
    print("Test Pass: Forward pass runs on CUDA")
    
def test_model_initialization():
    sto_model_cfg = [
                ("normal", {"loc":1.0, "scale":0.5},  # the name of base distribution and parameters for that distribution
                    [("affine", 1, {"learnable":True}), # the first stack of flows (type, depth, params)
                    ("planar", 8, {"init_sigma":0.01})] # the second stack of flows (type, depth, params)
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
                    ("planar", 6, {"init_sigma":0.01})] # the second stack of flows (type, depth, params)
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
    
    sto_model_cfg = [
            ("normal", {"loc":1.0, "scale":0.5},  
                [("affine", 1, {"learnable":True}), 
                ("planar2d", 6, {"init_sigma":0.01})] # use planar2d for image data 
            ),
            (
                "normal", {"loc":1.0, "scale":0.5}, 
                [("affine", 1), 
                    ("planar", 6)]
            )
            ]
    base_lenet = LeNet()
    sto_lenet = StoLeNet(sto_model_cfg)
    sto_lenet.migrate_from_det_model(base_lenet)
    
    data = torch.randn(8,1,28,28)
    base_out3 = base_lenet(data)
    test_out = sto_lenet(data) # see if the forward pass runs 
    sto_lenet.no_stochastic()
    sto_out3 = sto_lenet(data)
    cond3 = torch.allclose(base_out3, sto_out3)
    
    if all([cond1, cond2, cond3]):
        print("Test Pass: Migrated model shares the deterministic part with base model")
    else:
        raise ValueError("Test Fail: Migrated model doesn't share the deterministic part with base model")

def test_acc():
    from utils import compute_accuracy, compute_ece_loss
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    data_dir = "./codes/data"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    device = torch.device("cuda")
    trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=False)
    testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=transform, download=False)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=16, shuffle=False)
    sto_model_cfg = [
        ("normal", {"loc":1.0, "scale":0.5},  
            [("affine", 1, {"learnable":True}), 
            ("planar2d", 6, {"init_sigma":0.01})] # use planar2d for image data 
        ),
        (
            "normal", {"loc":1.0, "scale":0.5}, 
            [("affine", 1), 
                ("planar", 6)]
        )
        ]
    base_lenet = LeNet().to(device)
    sto_lenet = StoLeNet(sto_model_cfg).to(device)
    base_acc = compute_accuracy(base_lenet, testloader)
    sto_acc = compute_accuracy(sto_lenet, testloader)
    print("Test Pass: compute accuracy runs for StoModel")
    
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
        raise ValueError("Test Fail: 2D flow is NOT consistent with iteratively applying 1D flow")

def test_flow_cfg_format():
    flow_cfg: List[Tuple] = [ # the first stack of flows (type, depth, params)
                             ("affine", 1, {"learnable":True}), 
                              # keys of params must be consistent with the arguments in the flow
                             ("planar2d", 8, {"init_sigma":0.01})] 
    norm_flow = NF_Block(vec_len=16, flow_cfg=flow_cfg)
    print(norm_flow)
    
def test_batched_ece():
    
    from models import LogisticRegression
    
    full_ece = ECELoss(n_bins=15)
    log_reg = LogisticRegression(2, 2, True)
    data = torch.randn(1000, 2)
    label = torch.randint(0,2,(1000,))
    pred_prob = log_reg(data).exp()
    full_loss = full_ece.forward(pred_prob, label)
    
    list_batch_pred = []
    batch_ece = ECELoss(n_bins=15)
    batch_size = 100
    for idx in range(10):
        batch_X = data[idx*batch_size:(idx+1)*batch_size, :]
        batch_Y = label[idx*batch_size:(idx+1)*batch_size]
        b_pred_prob = log_reg.forward(batch_X).exp()
        list_batch_pred.append(b_pred_prob)
        batch_ece.add_batch(b_pred_prob, batch_Y)
    cat_batch_result = torch.cat(list_batch_pred, dim=0)
    batch_loss = batch_ece.summarize_batch()
    
    conditions = [] 
    conditions.append(torch.allclose(cat_batch_result, pred_prob))
    conditions.append(torch.allclose(batch_ece.acc_list, full_ece.acc_list))
    conditions.append(torch.allclose(batch_ece.conf_list, full_ece.conf_list))
    conditions.append(torch.allclose(batch_ece.sample_per, full_ece.sample_per))
    conditions.append(torch.allclose(batch_loss, full_loss))
    
    if all(conditions):
        print("Test Pass: Batched ECE loss is equivalent to full ECE")
    else:
        raise ValueError("Test Fail: Batched ECE loss is NOT equivalent to full ECE") 
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    test_2d_planar_flow()
    test_flow_cfg_format()
    test_linear_migration()
    test_conv_migration()
    test_acc()
    test_model_initialization()
    test_cuda_forward()
    test_optimizer()
    test_batched_ece()
    pass