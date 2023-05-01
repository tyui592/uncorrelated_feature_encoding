# -*- coding: utf-8 -*-
"""Optimization function.

* Author: Minseong Kim(tyui592@gmail.com)
"""

from torch.optim import Adam


# optimizer
def get_optimizer(network, lr, encoder_lr, decoder_lr):
    """Get optimizer."""
    optim_params = []
    encoder_params = network.encoder.parameters()
    if encoder_lr > 0:
        optim_params.append({'params': encoder_params,
                             'lr': encoder_lr})
    else:
        for param in encoder_params:
            param.requires_grad = False

    decoder_params = network.decoder.parameters()
    if decoder_lr > 0:
        optim_params.append({'params': decoder_params,
                             'lr': decoder_lr})
    else:
        for param in decoder_params:
            param.requires_grad = False

    optimizer = Adam(optim_params, lr=lr)
    return optimizer
