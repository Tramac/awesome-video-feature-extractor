"""Model store which handles pretrained models """
from .alexnet import *
from .vgg import *
from .googlenet import *
from .inception import *
from .inceptionv4 import *
from .squeezenet import *
from .resnet import *
from .densenet import *
from .mobilenet import *
from .shufflenetv2 import *
from .c3d import *

__all__ = ['get_model', 'get_model_list']

_models = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'c3d': c3d,
}

def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net

def get_model_list():
    return _models.keys()

