# -*- coding: utf-8 -*-
"""Utility code.

* Author: Minseong Kim(tyui592@gmail.com)
"""

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from typing import List


def get_normalizer(denormalize=False):
    """Get normalizer."""
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    if denormalize:
        MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
        STD = [1/std for std in STD]

    return T.Normalize(mean=MEAN, std=STD)


def get_transformer(imsize=None, cropsize=None, cencrop=False):
    """Get a data transformer."""
    transformer = []
    if imsize:
        transformer.append(T.Resize(imsize))
    if cencrop:
        transformer.append(T.CenterCrop(cropsize))
    else:
        transformer.append(T.RandomCrop(cropsize))

    transformer.append(T.ToTensor())
    transformer.append(get_normalizer())
    return T.Compose(transformer)


def imload(path, imsize=None, cropsize=None, cencrop=False):
    """Load a Image."""
    transformer = get_transformer(imsize=imsize,
                                  cropsize=cropsize,
                                  cencrop=cencrop)
    return transformer(Image.open(path).convert("RGB")).unsqueeze(0)


def maskload(path):
    """Load a Mask."""
    mask = Image.open(path).convert('L')
    return T.functional.to_tensor(mask).unsqueeze(0)


def ten2pil(tensor, nrow=8):
    """Change a tensor to a pil image."""
    denormalize = get_normalizer(denormalize=True)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    grid = torchvision.utils.make_grid(tensor, nrow=nrow)
    pil = TF.to_pil_image(denormalize(grid).clamp_(0.0, 1.0))
    return pil


def imsave(tensor, path, nrow=8):
    """Save a tensor to image file."""
    pil = ten2pil(tensor, nrow=nrow)
    pil.save(path)
    return None


def avg_values(values, length=100):
    """Calculate a average of lastest n values."""
    if len(values) < length:
        length = len(values)
    return sum(values[-length:])/length


def calc_num_params(network):
    """Calculate the number of parameters."""
    return sum(p.numel() for p in network.parameters())


def calc_nonzero_channel(features: List[torch.Tensor]) -> float:
    """Calculate the number of nonzero channels in the feature map."""
    nonzeros = []
    for feature in features:
        batch_size = feature.shape[0]

        vector = feature.flatten(2).detach()
        abssum = torch.sum(torch.abs(vector), dim=2)

        nonzero = torch.nonzero(abssum, as_tuple=False)
        num_nonzero_channels = nonzero.shape[0] / batch_size

        nonzeros.append(num_nonzero_channels)

    return sum(nonzeros) / len(nonzeros)
