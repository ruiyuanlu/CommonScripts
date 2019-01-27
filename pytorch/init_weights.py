# coding=utf8

from torch import nn
from torch.nn.init import zeros_, ones_
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.init import kaiming_normal_, kaiming_uniform_

_Init_Func = {
    'zeros': zeros_,
    'ones': ones_,
    'he_normal': kaiming_normal_,
    'he_uniform': kaiming_uniform_,
    'glorot_normal': xavier_normal_,
    'glorot_uniform': xavier_uniform_,
}

def get_initializer(method='he', dist='normal', bias=0, *, modules):
    """
    Return initializer for specific module types.

    Args:
        method (str, int): weihgts initialization method.
            'he' -> He Kaiming initialization: Better for relu based net.
                https://arxiv.org/pdf/1502.01852.pdf
            'glorot' -> xavier initialization: Better for linear
                based net and sigmoid based activation.
                http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            'ones' or 1 -> all ones initialization. Not recommended.
            'zeros' or 0 -> all zeros initialization. Not recommended.
        dist (str): distribution of method. (normal | uniform).
            Only valid for 'he' and 'glorot' method.
        bias (str, int): 'ones' or 1 -> all ones initialization.
            'zeros' or 0 -> all zeros initialization.
        modules (Iterable): Specify types of torch.nn.Module instances to initialize.

    Returns:
        initializer (Function): Used by 'apply' method of torch.nn.Module to initialize weights.
    """
    # 'method' check
    method_err = "'method' must be one of (he, glorot, ones, zeros, 1, 0), %s found"
    if isinstance(method, str):
        assert method.lower() in ('he', 'glorot', 'zeros', 'ones'), method_err % method
        method = method.lower()
    elif isinstance(method, (int, float)):
        assert method in (1, 0), method_err % method
        method = 'zeros' if method == 0 else 'ones'
    else:
        raise ValueError(method_err % type(method))

    # 'dist' check
    dist_err = "distribution must be either 'normal' or 'uniform', %s found"
    if isinstance(dist, str):
        assert dist.lower() in ('normal', 'uniform'), dist_err % dist
        dist = dist.lower()
    else:
        raise ValueError(dist_err % type(dist))

    # 'bias' check
    bias_err = "'bias' must be one of (ones, zeros, 1, 0), %s found"
    if isinstance(bias, str):
        assert bias.lower() in ('ones', 'zeros'), bias_err % bias
        bias = bias.lower()
    elif isinstance(bias, (int, float)):
        assert bias in (1, 0), bias_err % bias
        bias = 'zeros' if bias == 0 else 'ones'
    else:
        raise ValueError(bias_err % type(bias))

    # 'module_type' check
    type_err = "'modules' must be iterable of types of torch.nn.Module instances and not empty, %s found"
    modules = tuple(modules)
    assert modules, type_err % modules
    pos = nn.Module.__module__.rfind('.')
    prefix = nn.Module.__module__[:pos] # common prefix 'torch.nn.modules'
    for m in modules:
        assert m.__module__.startswith(prefix), type_err % type_err % m.__module__

    # weights initializer
    if method not in ('ones', 'zeros'):
        weights_init = _Init_Func['%s_%s' % (method, dist)]
    else:
        weights_init = _Init_Func[method]

    # bias initializer
    bias_init = _Init_Func[bias]

    # using closure function as real initializer
    def initializer(m):
        """
        Initialize weights and bias.
        """
        if isinstance(m, modules):
            weights_init(m.weight)
            if m.bias is not None:
                bias_init(m.bias)

    return initializer


if __name__ == '__main__':
    # Init weights in the __init__ function.
    from torch.nn import functional as F
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2)
            self.fc = nn.Linear(14 * 14 * 64, 10)
            # init weights
            initer = get_initializer(method='he', dist='normal',
                        bias=1, modules=[nn.Conv2d, nn.Linear])
        
        def forward(self, x):
            x = F.dropout(F.relu(self.conv(x)), p=0.5, inplace=True)
            return F.log_softmax(self.fc(x.view(14 * 14 * 64)))
