"""
    ported from torchvision.models.vgg, use batch normalization for all models 
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast
from .basic import StoModel
from .layers import StoLayer, StoConv2d, StoLinear

__all__ = ["VGG", "StoVGG", 'vgg16', 'vgg19', "sto_vgg16", "sto_vgg19"]


# ============================ # 
# ===  original vgg setup  === # 
# ============================ # 

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


# ============================ # 
# === stochastic vgg setup === # 
# ============================ # 

class StoVGG(StoModel):
    
    DET_MODEL_CLASS = VGG
    
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 10,
        init_weights: bool = True,
        sto_cfg=None
    ) -> None:
        super(StoVGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            StoLinear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            StoLinear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            StoLinear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        self.sto_layers = [m for m in self.features if isinstance(m, StoLayer)] + [m for m in self.classifier if isinstance(m, StoLayer)]
        self.build_all_flows(sto_cfg=sto_cfg)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def det_and_sto_params(self):
        
        det_params, sto_params = [], []

        for name, layer in self.features._modules.items():
            if isinstance(layer, StoLayer):
                det_params.extend(list(layer.det_compo.parameters()))
                sto_params.extend(list(layer.norm_flow.parameters()))
            else:
                det_params.extend(list(layer.parameters()))
                
        for name, layer in self.classifier._modules.items():
            if isinstance(layer, StoLayer):
                det_params.extend(list(layer.det_compo.parameters()))
                sto_params.extend(list(layer.norm_flow.parameters()))
            else:
                det_params.extend(list(layer.parameters()))   
                
        return det_params, sto_params
    
    def migrate_from_det_model(self, det_model):
        det_class = self.DET_MODEL_CLASS
        if isinstance(det_model, det_class):
            for name, layer in self.features._modules.items():
                if name == "": # skip empty names 
                    continue
                if isinstance(layer, StoLayer):
                    layer.migrate_from_det_layer(getattr(det_model.features, name))
                elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    layer.weight.data.copy_(getattr(det_model, name).weight.data)
                    layer.bias.data.copy_(getattr(det_model, name).bias.data)
            for name, layer in self.classifier._modules.items():
                if name == "": # skip empty names 
                    continue
                if isinstance(layer, StoLayer):
                    layer.migrate_from_det_layer(getattr(det_model.classifier, name))
                elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    layer.weight.data.copy_(getattr(det_model, name).weight.data)
                    layer.bias.data.copy_(getattr(det_model, name).bias.data)
        
                    
def make_sto_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = StoConv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _sto_vgg(cfg: str, batch_norm: bool, 
            num_classes=10,init_weights=True,sto_cfg=None) -> StoVGG:

    model = StoVGG(make_sto_layers(cfgs[cfg], batch_norm=batch_norm), 
                num_classes= num_classes,init_weights= init_weights,sto_cfg=sto_cfg)

    return model

# ============================ # 
# === model initialization === # 
# ============================ # 


def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 16, True, pretrained, progress, **kwargs)

def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 19, True, pretrained, progress, **kwargs)

def sto_vgg16(num_classes=10,init_weights=True,sto_cfg=None) -> StoVGG:
    """
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _sto_vgg(16, True, num_classes, init_weights, sto_cfg)

def sto_vgg19(num_classes=10,init_weights=True,sto_cfg=None) -> StoVGG:
    """
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _sto_vgg(19, True, num_classes, init_weights, sto_cfg)
