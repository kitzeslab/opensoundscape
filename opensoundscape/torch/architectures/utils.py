# copying from https://github.com/zhmiao/BirdMultiLabel/blob/b31edf022e5c54a5d7ebe994460fec1579e90e96/src/models/utils.py

import torch
import torch.nn as nn

models = {}


def register_model(name):

    """
    Model register
    """

    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def get_model(name, **args):

    """
    Model getter
    """

    net = models[name](**args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net


class BaseModule(nn.Module):

    """
    Base module for reference.
    """

    name = None

    def __init__(self):
        super(BaseModule, self).__init__()

    def setup_net(self):
        pass

    def setup_critera(self):
        pass

    def load(self, init_path):
        pass

    def save(self, out_path):
        pass

    def update_best(self):
        pass
