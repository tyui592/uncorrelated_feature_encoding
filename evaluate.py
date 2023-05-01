# -*- coding: utf-8 -*-
"""Evaluation code.

* Author: Minseong Kim(tyui592@gmail.com)
"""

import torch
from utils import imload, imsave
from network import get_networks


def evaluate_network(args):
    """Generate a stylized image."""
    device = torch.device('cuda' if args.gpu_no >= 0 else 'cpu')

    ckpt = torch.load(args.load_path, map_location='cpu')

    network, loss_network = get_networks(backbone=args.backbone)
    network.load_state_dict(ckpt['state_dict'])
    network.to(device)
    network.eval()

    content = imload(path=args.content,
                     imsize=args.imsize,
                     cropsize=args.cropsize,
                     cencrop=args.cencrop).to(device)

    style = imload(path=args.style,
                   imsize=args.imsize,
                   cropsize=args.cropsize,
                   cencrop=args.cencrop).to(device)

    with torch.inference_mode():
        output_img = network(content=content,
                             style=style,
                             style_strength=args.style_strength)
    imsave(output_img, args.save_path)

    return None
