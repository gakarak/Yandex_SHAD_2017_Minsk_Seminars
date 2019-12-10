#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import torch
from torch import nn

import segmentation_models_pytorch as smp


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main_test():
    model = smp.Unet('resnet34', encoder_weights=None)

    print('-')


if __name__ == '__main__':
    main_test()