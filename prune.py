# -*- coding: utf-8 -*-
"""Prune the network.

* Author: Minseong Kim(tyui592@gmail.com)
"""
import torch
from typing import List


def get_pruned_tensor(tensor: torch.Tensor,
                      dim: int,
                      indices: List[int]) -> torch.Tensor:
    """Get a pruned tensor."""
    survive_indices = list(set(range(tensor.size(dim))) - set(indices))
    pruned_tensor = torch.index_select(input=tensor,
                                       dim=dim,
                                       index=torch.tensor(survive_indices))

    return pruned_tensor


def get_pruned_conv(conv: torch.nn.Conv2d,
                    dim: int,
                    indices: List[int]) -> torch.nn.Conv2d:
    """Get a convolutional layer with pruned channels.

    Conv2d has weight and bias tensor.
    - weight tensor has 4 dims with [out-ch, int-ch, kernel, kernel].
    - bias tensor has 1 dim with [out-ch]

    * dim: 0 or 1
        - set '0' to prune output channel and '1' for input channel.
    """
    if dim == 0:
        out_ch = int(conv.out_channels - len(indices))
        pruned_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                      out_channels=out_ch,
                                      kernel_size=conv.kernel_size,
                                      stride=conv.stride,
                                      padding=conv.padding,
                                      dilation=conv.dilation)
        pruned_conv.weight.data = get_pruned_tensor(tensor=conv.weight.data,
                                                    dim=dim,
                                                    indices=indices)
        pruned_conv.bias.data = get_pruned_tensor(tensor=conv.bias.data,
                                                  dim=dim,
                                                  indices=indices)

        return pruned_conv

    elif dim == 1:
        in_ch = int(conv.in_channels - len(indices))
        pruned_conv = torch.nn.Conv2d(in_channels=in_ch,
                                      out_channels=conv.out_channels,
                                      kernel_size=conv.kernel_size,
                                      stride=conv.stride,
                                      padding=conv.padding,
                                      dilation=conv.dilation)

        pruned_conv.weight.data = get_pruned_tensor(tensor=conv.weight.data,
                                                    dim=dim,
                                                    indices=indices)
        pruned_conv.bias.data = conv.bias.data

        return pruned_conv


def prune_style_transfer_network(model, channel_index, backbone='vgg19'):
    """Prune style transfer network."""
    decoder_index = 1
    if backbone == 'vgg16':
        encoder_index = 14
    elif backbone == 'vgg19':
        encoder_index = 19

    # conv of encoder last layer
    encoder_conv = model.encoder.layers[0][encoder_index]

    # conv of decoder first layer
    decoder_conv = model.decoder.layers[0][decoder_index]

    # prune the conv layers
    pruned_encoder_conv = get_pruned_conv(encoder_conv, 0, channel_index)
    pruned_decoder_conv = get_pruned_conv(decoder_conv, 1, channel_index)

    # swap the target layers
    model.encoder.layers[0][encoder_index] = pruned_encoder_conv
    model.decoder.layers[0][decoder_index] = pruned_decoder_conv
    return None
