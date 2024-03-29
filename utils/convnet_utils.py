import torch.nn as nn
from block.dac import DACBlock

CONV_BN_IMPL = 'DACB'


def conv(in_channels, out_channels, kernel_size, stride, padding, bias, groups=1, dilation=1):
    if CONV_BN_IMPL == 'base':
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)
    elif CONV_BN_IMPL == 'ACB':
        blk_type = ACBlock
    elif CONV_BN_IMPL == 'DBB':
        blk_type = DiverseBranchBlock
    elif CONV_BN_IMPL == 'DCDB':
        blk_type = DCDBlock
    elif CONV_BN_IMPL == 'DACB-S':
        blk_type = DACBlockS
    elif CONV_BN_IMPL == 'DACB':
        blk_type = DACBlock
    elif CONV_BN_IMPL == 'DACB-C':
        blk_type = DACBlockC
    else:
        blk_type = Dynamic_conv2d
    return blk_type(in_planes=in_channels, out_planes=out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, grounps=groups, bias=bias)


def switch_conv_bn_impl(block_type):
    assert block_type in ['base', 'DCDB', 'DCB', 'ACB', 'DACB-S', 'DACB', 'DACB-C',
                          'DBB']
    global CONV_BN_IMPL
    CONV_BN_IMPL = block_type
    if CONV_BN_IMPL == 'ACB':
        print("使用非对称卷积")
    elif CONV_BN_IMPL == 'DBB':
        print("使用多分支卷机卷积")
    elif CONV_BN_IMPL == 'DCDB':
        print("使用分解卷积")
    elif CONV_BN_IMPL == 'DACB-S':
        print("空间注意力")
    elif CONV_BN_IMPL == 'DACB':
        print("使用注意力之和为1")
    elif CONV_BN_IMPL == 'DACB-C':
        print("通道注意力")


def build_model(data):
    if data == 'imagenet':
        num_classes = 1000
    elif data == 'cifar10':
        num_classes = 10
    elif data == 'Cub':
        num_classes = 200
    else:
        num_classes = 100
    from model.RWFFNet import RWFFNet
    model = RWFFNet(num_classes=num_classes)
    model.resnet.load_state_dict(torch.load(''), strict=False)
    return model
