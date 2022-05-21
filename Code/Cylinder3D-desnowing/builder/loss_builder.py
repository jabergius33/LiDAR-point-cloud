# -*- coding:utf-8 -*-
# author: Xinge
# @file: loss_builder.py 

import torch
from utils.lovasz_losses import lovasz_softmax
from utils.lovasz_losses import binary_xloss


def build(wce=True, lovasz=True, num_class=2, ignore_label=200):

    loss_funs = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    if wce and lovasz:
        return loss_funs, lovasz_softmax, binary_xloss #ADDED binary
    elif wce and not lovasz:
        return wce
    elif not wce and lovasz:
        return lovasz_softmax
    else:
        raise NotImplementedError
