import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

# Reference
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>
    

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids) # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class ResBlock(nn.Module):
    def __init__(self, filters, bias=True):
        super(ResBlock, self).__init__()
        self.bn = nn.BatchNorm2d(filters)
        self.activ = nn.LeakyReLU()
        
        layers  = [nn.Conv2d(filters, filters, 3, padding=1, bias=bias)]
        layers += [self.activ]
        layers += [nn.Conv2d(filters, filters, 3, padding=1, bias=bias)]
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        bn  = self.bn(x)
        res = self.block(bn)
        out = self.activ(bn+res)
        return out
    
    
class ResBlockEnsemble(nn.Module):
    def __init__(self, filters:int, blocks:int, bias=True):
        super(ResBlockEnsemble, self).__init__()
        self.blocks = [ResBlock(filters, bias=bias) for _ in range(blocks)]
        
        layers  = [nn.Conv2d(filters*(blocks+1), filters, 1, bias=False)]
        #layers += [nn.LeakyReLU()]
        self.ensemble = nn.Sequential(*layers)
        
        self.dummy = nn.Sequential(*self.blocks) # so pytorch can see the layers
        
    def forward(self, x):
        res = [x]
        for block in self.blocks:
            res += [block(res[-1])]
        cat = torch.cat(res, dim=1)
        out = self.ensemble(cat)
        return out
    
    
class ContextBlock(nn.Module):
    def __init__(self, in_filters:int, out_filters:int, bias=True):
        super(ContextBlock, self).__init__()
        relu_and_norm = lambda: [nn.LeakyReLU(), nn.BatchNorm2d(out_filters)]
        
        layers  = [nn.BatchNorm2d(in_filters)]
        layers += [nn.Conv2d(in_filters, out_filters, 5, padding=2, bias=bias)]
        layers += [nn.LeakyReLU()]
        self.layer1 = nn.Sequential(*layers)
        
        layers  = [nn.Conv2d(out_filters, out_filters, 3, dilation=2, padding=2, bias=bias)]
        #layers += [nn.LeakyReLU()]
        self.layer2 = nn.Sequential(*layers)
        
        layers  = [nn.Conv2d(out_filters, out_filters, 2, dilation=2, padding=1, bias=bias)]
        layers += [nn.LeakyReLU()]
        self.layer3 = nn.Sequential(*layers)
        
        layers  = [nn.Conv2d(2*out_filters, out_filters, 1, bias=False)]
        self.attention = nn.Sequential(*layers)
        
        self.layers = [self.layer1, self.layer2]#, self.layer3]

    def forward(self, x):
        out = [x]
        for layer in self.layers:
            out += [layer(out[-1])]
        cat = torch.cat(out[1:], dim=1)
        out = self.attention(cat)
        return out
    
    
class ResContextBlock(nn.Module):
    def __init__(self, in_filters, bias=True):
        super(ResContextBlock, self).__init__()
        out_filters = in_filters
        self.activ = nn.LeakyReLU()
        activ_and_norm = lambda: [self.activ, nn.BatchNorm2d(out_filters)]
        
        layers  = [nn.Conv2d(in_filters, out_filters, 1, stride=1, bias=bias)]
        layers += [self.activ]
        self.skip_path = nn.Sequential(*layers)
        
        layers  = [nn.Conv2d(out_filters, out_filters, 5, padding=2, bias=bias)]
        layers += activ_and_norm()
        layers += [nn.Conv2d(out_filters, out_filters, 3, dilation=3, padding=3, bias=bias)]
        #layers += activ_and_norm()
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        shortcut = self.skip_path(x)
        resA = self.block(shortcut)
        out = self.activ(shortcut + resA)
        return out
    

class ResContextBlock(nn.Module):
    def __init__(self, filters, bias=True):
        super(ResContextBlock, self).__init__()     
        self.layers = ContextBlock(filters, filters, bias=bias)
        self.activ  = nn.LeakyReLU()

    def forward(self, x):
        res = self.layers(x)
        out = self.activ(x + res)
        return out

    
class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ConvBlock, self).__init__()
        layers  = [nn.BatchNorm2d(in_filters)]
        layers += [nn.Conv2d(in_filters, out_filters, 7, padding=3)]
        layers += [nn.LeakyReLU()]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, drop_out:float=None):
        super(UpBlock, self).__init__()
        layers  = [nn.BatchNorm2d(in_filters)]
        layers += [nn.Dropout2d(p=drop_out)] if drop_out else []
        #layers += [ResContextBlock(in_filters)]
        layers += [nn.PixelShuffle(2)]
        layers += [nn.Conv2d(in_filters>>2, out_filters, 3, padding=1)]
        layers += [nn.LeakyReLU()]
        self.up = nn.Sequential(*layers)
        
        layers  = [nn.Conv2d(2*out_filters, out_filters, 5, padding=2)]
        layers += [nn.LeakyReLU()]
        layers += [ResContextBlock(out_filters, bias=True)]
        self.block = nn.Sequential(*layers)
        
        layers  = [nn.Conv2d(2*out_filters, out_filters, 1, bias=False)]
        layers += [nn.LeakyReLU()]
        self.linear = nn.Sequential(*layers)

    def forward(self, x, skip):
        up  = self.up(x)
        cat = torch.cat((up, skip), dim=1)
        out = self.block(cat)
        #out = self.linear(cat)
        return out
    
    
class DownBlock(nn.Module):
    def __init__(self, in_filters, out_filters, drop_out:float=None):
        super(DownBlock, self).__init__()
        layers  = [ResContextBlock(in_filters, bias=True)]
        self.pre = nn.Sequential(*layers)
        
        layers  = [nn.Dropout2d(p=drop_out)] if drop_out else []
        layers += [nn.Conv2d(in_filters, out_filters, 5, stride=2, padding=2)]
        layers += [nn.LeakyReLU()]
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        pre = self.pre(x)
        down = self.down(pre)
        return down, pre
    
############## Salsa blocks ##########
class SalsaResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, bias=True):
        super(SalsaResContextBlock, self).__init__()
        relu_and_norm = lambda: [nn.LeakyReLU(), nn.BatchNorm2d(out_filters)]
        self.activ = nn.LeakyReLU()
        
        layers  = [nn.Conv2d(in_filters, out_filters, 1, bias=bias)]
        layers += [nn.LeakyReLU()]
        self.skip_path = nn.Sequential(*layers)
        
        layers  = [nn.Conv2d(in_filters, out_filters, 5, padding=2, bias=bias)]
        layers += relu_and_norm()
        layers += [nn.Conv2d(out_filters, out_filters, 3, dilation=3, padding=3, bias=bias)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        shortcut = self.skip_path(x)
        resA = self.block(x)
        out = self.activ(shortcut + resA)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, stride=1,
                 pooling=True, drop_out:float=0.2, bias=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        relu_and_norm = lambda: [nn.LeakyReLU(), nn.BatchNorm2d(out_filters)]
        
        self.activ = nn.LeakyReLU()

        layers  = [nn.Conv2d(in_filters, out_filters, 1, stride=stride, bias=bias)]
        layers += [nn.LeakyReLU()]
        self.skip_path = nn.Sequential(*layers)

        layers  = [nn.Conv2d(in_filters, out_filters, 3, padding=1, bias=bias)]
        layers += relu_and_norm()
        self.path1 = nn.Sequential(*layers)

        layers  = [nn.Conv2d(out_filters, out_filters, 3, dilation=2, padding=2, bias=bias)]
        layers += relu_and_norm()
        self.path2 = nn.Sequential(*layers)

        layers  = [nn.Conv2d(out_filters, out_filters, 2, dilation=2, padding=1, bias=bias)]
        layers += [nn.LeakyReLU()]
        self.path3 = nn.Sequential(*layers)

        layers  = [nn.Conv2d(3*out_filters, out_filters, 1, bias=bias)]
        self.path4 = nn.Sequential(*layers)

        self.dropout = nn.Dropout2d(p=drop_out)
        if pooling:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
            #layers  = [nn.PixelUnshuffle(2)]
            #layers += [nn.Conv2d(4*out_filters, out_filters, 1, bias=bias)]
            #layers += [nn.LeakyReLU()]
            #self.pool = nn.Sequential(*layers)

    def forward(self, x):
        shortcut = self.skip_path(x)

        resA1 = self.path1(x)
        resA2 = self.path2(resA1)
        resA3 = self.path3(resA2)

        concat = torch.cat((resA1, resA2, resA3),dim=1)
        resA = self.path4(concat)
        resA += shortcut
        resA = self.activ(resA)

        resB = self.dropout(resA) if self.drop_out else resA

        if self.pooling:
            resB = self.pool(resB)
            return resB, resA
        return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, drop_out:float=None):
        super(UpBlock, self).__init__()
        relu_and_norm = lambda: [nn.LeakyReLU(), nn.BatchNorm2d(out_filters)]
        
        layers  = [nn.PixelShuffle(2)]
        layers += [nn.Dropout2d(p=drop_out)] if drop_out else []
        #layers += [nn.Conv2d(in_filters>>2, out_filters, 1, padding=0)]
        self.up_path = nn.Sequential(*layers)

        layers  = [nn.Dropout2d(p=drop_out)] if drop_out else []
        layers += [nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, 1, padding=0)]
        layers += relu_and_norm()
        self.path1 = nn.Sequential(*layers)
        
        layers  = [nn.Conv2d(out_filters, out_filters, 3, dilation=1, padding=1)]
        layers += [nn.LeakyReLU()]
        self.path2 = nn.Sequential(*layers)
        
        layers  = [nn.Conv2d(2*out_filters, out_filters, 1)]
        layers += relu_and_norm()
        layers += [nn.Dropout2d(p=drop_out)] if drop_out else []
        self.path4 = nn.Sequential(*layers)

    def forward(self, x, skip):
        upA = self.up_path(x) 
        upB = torch.cat((upA,skip),dim=1)
    
        upE1 = self.path1(upB)
        upE2 = self.path2(upE1)

        concat = torch.cat((upE1, upE2),dim=1)
        upE = self.path4(concat)
        return upE
    
############## Combined ##############   
class SalsaNext(nn.Module):
    def __init__(self, _):
        super(SalsaNext, self).__init__()
        in_chnl = 5
        
        name2block = {
            'conv':lambda i,o: ConvBlock(i, o),
            'res': lambda i,o: ResBlock(i,bias=True), 
            'eres':lambda i,o: ResBlockEnsemble(i, 10, bias=True),
            'cres':lambda i,o: ResContextBlock(i, bias=True),
            #'cntx':lambda i,o: ContextBlock(i, o, bias=True),
            'down':lambda i,o: DownBlock(i, o, drop_out=False),
            'up':  lambda i,o: UpBlock(i, o, drop_out=False),
            'c'  : lambda i,o: nn.Conv2d(i, o, 5, padding=2),
            'sup' :lambda i,o: UpBlock(i, o, drop_out=False),
            'sdwn':lambda i,o: ResBlock(i, o, pooling=True,  drop_out=False),
            'sres':lambda i,o: ResBlock(i, o, pooling=False, drop_out=False),
            'cntx':lambda i,o: SalsaResContextBlock(i, o),
        }
        
        
        layer_list = [
            'cntx-32', 'cntx-32', 'marker', # Pre process
            'sdwn-64', 'sdwn-128', 'sdwn-256', 'marker', # Encode
            'sres-256', 'marker', # Process features
            'sup-128', 'sup-64', 'sup-32', 'marker', # Decode
            'cntx-32', 'c-1', # Post process
        ]
        # Extract markers so we know where up/down layers start
        markers = [i for i,x in enumerate(layer_list) if x=='marker']
        markers = [x-i for i,x in enumerate(markers)]
        layer_list = [l for l in layer_list if l != 'marker']
        
        layers = []
        for idx, layer in enumerate(layer_list):
            block_name, out_chnl = layer.split('-')
            out_chnl = int(out_chnl)
            layers += [name2block[block_name](in_chnl, out_chnl)]
            in_chnl = out_chnl
        
        m = markers
        self.pre  = nn.Sequential(*layers[   0:m[0]])
        self.down =                layers[m[0]:m[1]]
        self.mid  = nn.Sequential(*layers[m[1]:m[2]])
        self.up   =                layers[m[2]:m[3]]
        self.post = nn.Sequential(*layers[m[3]:])
        
        self.dummy = nn.Sequential(*layers) # so pytorch can see the layers
        self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        # Encode
        pre  = self.pre(x)
        skips = [pre]
        down = pre
        for d in self.down:
            down, skip = d(down)
            skips += [skip]
        mid = self.mid(down)
        #skips = skips[:-1] # because last "skip" is not a skip
        skips = [*reversed(skips)] # decoders want reverse order
        
        # Decode
        up  = mid
        for u,s in zip(self.up, skips):
            up = u(up,s)
        m = self.post(up)
        m = m - torch.finfo(torch.float32).max * (x[:,0] == -1).reshape(m.shape) # leverage that -1 -> -inf
        m = self.sigm(m)
        return m
