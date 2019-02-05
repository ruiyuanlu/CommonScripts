# coding=utf8

from functools import partial

from torch import nn
from torch.nn.init import zeros_, ones_, eye_
from torch.nn.init import constant_, normal_, uniform_, dirac_
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.init import kaiming_normal_, kaiming_uniform_

_Init_Func = {
    'eye': eye_,
    'zeros': zeros_,
    'ones': ones_,
    'dirac': dirac_,
    'normal': normal_,
    'uniform': uniform_,
    'he_normal': kaiming_normal_,
    'he_uniform': kaiming_uniform_,
    'glorot_normal': xavier_normal_,
    'glorot_uniform': xavier_uniform_,
}

def get_initializer(method='he', dist='normal', bias=0, *, modules):
    """
    Return initializer for specific module types.

    Args:
        method (str, int, float): weihgts initialization method.
            'he' -> He Kaiming initialization: Better for relu based net.
                https://arxiv.org/pdf/1502.01852.pdf
            'glorot' -> xavier initialization: Better for linear
                based net and sigmoid based activation.
                http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            'normal' -> normal distribution.
            'uniform' -> uniform distribution.
            'eye' -> eye initialization
            'dirac' -> dirac initialization
            'ones' -> all 0
            'zeros' -> all 1
            Other int or float for constant initialization.
        
        dist (str): distribution of method. (normal | uniform).
            Only valid for 'he' and 'glorot' method.
        bias (str, int, float): int or float for constant initialization.
            'normal' -> normal distribution.
            'uniform' -> uniform distribution.
            'ones' -> all 0
            'zeros' -> all 1
            Other int or float for constant initialization.

        modules (Iterable): Specify types of torch.nn.Module instances to initialize.

    Returns:
        initializer (Function): Used by 'apply' method of torch.nn.Module to initialize weights.
    """
    # 'method' check
    method_err = ("'method' must be one of ('he', 'glorot', 'zeros', 'ones', " +
            "'eye', 'dirac', 'normal', 'uniform' or float for constant init), %r found")
    if isinstance(method, str):
        assert method.lower() in ('he', 'glorot', 'zeros', 'ones', 
            'eye', 'dirac', 'normal', 'uniform'), method_err % method
        method = method.lower()
    elif isinstance(method, (int, float)):
        if method in (1, 0):
            method = 'zeros' if method == 0 else 'ones'
        else:
            method = partial(constant_, val=method) # weight constant initializer
    else:
        raise ValueError(method_err % type(method))

    # 'dist' check
    dist_err = "distribution must be either 'normal' or 'uniform', %r found"
    if isinstance(dist, str):
        assert dist.lower() in ('normal', 'uniform'), dist_err % dist
        dist = dist.lower()
    else:
        raise ValueError(dist_err % type(dist))

    # 'bias' check
    bias_err = ("'bias' must be one of (ones, zeros, 1, 0, 'normal', " +
        "'uniform' or float for constant init), %r found")
    if isinstance(bias, str):
        assert bias.lower() in ('ones', 'zeros', 'normal', 'uniform'), bias_err % bias
        bias = bias.lower()
    elif isinstance(bias, (int, float)):
        if bias in (1, 0):
            bias = 'zeros' if bias == 0 else 'ones'
        else:
            bias = partial(constant_, val=bias) # bias constant initializer
    else:
        raise ValueError(bias_err % type(bias))

    # 'module_type' check
    type_err = "'modules' must be iterable of types of torch.nn.Module instances and not empty, %r found"
    modules = tuple(modules)
    assert modules, type_err % modules
    pos = nn.Module.__module__.rfind('.')
    prefix = nn.Module.__module__[:pos] # common prefix 'torch.nn.modules'
    for m in modules:
        assert m.__module__.startswith(prefix), type_err % type_err % m.__module__

    # weights initializer
    if isinstance(method, partial):
        weights_init = method # already constant initializer
    elif method in _Init_Func:
        weights_init = _Init_Func[method]
    else:
        weights_init = _Init_Func['%s_%s' % (method, dist)]

    # bias initializer. bias dim < 2, thus can not use
    # Kaiming He's initialization nor Glorot's initialization
    if isinstance(bias, partial):
        bias_init = bias # already constant_ initializer
    else:
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
    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    # Init weights demos
    # 
    # The first demo initialize weights in the __init__ function.
    # The second demo initialize weights outside the 'class' definition
    #
    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------

    from torch.nn import functional as F
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2)
            self.bn = nn.BatchNorm2d(num_features=64) # num_features means channels
            self.fc = nn.Linear(14 * 14 * 64, 10)
            # get weights initializer
            initializer = get_initializer(method=0, dist='normal',
                            bias=1, modules=[nn.Conv2d, nn.Linear])
            # init weights recursively
            self.apply(initializer)
        
        def forward(self, x):
            x = F.dropout(F.relu(self.conv(x)), p=0.5, inplace=True)
            return F.log_softmax(self.fc(x.view(14 * 14 * 64)))
    
    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    # 
    # check weights. 
    # Fist case all weights == 0.
    # Second case all weights ==1.
    # 
    # ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------
    net = Net()
    print((net.conv.weight == 0).numpy().all()) # case 1, weights init in the __init__.

    ones_initializer = get_initializer(method=0.3, bias='normal', modules=[nn.Conv2d])
    net.apply(ones_initializer)
    print(net.conv.bias)
