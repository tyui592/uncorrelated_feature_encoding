# -*- coding: utf-8 -*-
"""Parameters for execution.

* Author: Minseong Kim
"""

import argparse
from pathlib import Path


def build_parser():
    """Get arguments."""
    psr = argparse.ArgumentParser()

    psr.add_argument('--gpu_no', type=int, default=0,
                     help="Device: cpu = -1, gpu = 0 ~ n")

    psr.add_argument('--mode', type=str, default='train',
                     help="train or eval")

    psr.add_argument('--content', type=str, default=None,
                     help="Content image for evaluation")

    psr.add_argument('--style', type=str, default=None,
                     help="Style images for evaluation")

    psr.add_argument('--content_dir', type=str, default=None,
                     help="Content data dir path for training")

    psr.add_argument('--style_dir', type=str, default=None,
                     help="Style data dir path for traiing")

    psr.add_argument('--backbone', type=str, default='vgg19',
                     help="Backbone network, vgg16 or vgg19")

    psr.add_argument('--swap_max_pool', action='store_true', default=False,
                     help=("Swap the max pooling in vgg "
                           "to avg pooling for a smooth result image"))

    psr.add_argument('--style_strength', type=float, default=1.0,
                     help="Interpolation factor 1.0: style, 0.0: content")

    psr.add_argument('--imsize', type=int, default=None,
                     help="Image size (shorter side)")

    psr.add_argument('--cropsize', type=int, default=None,
                     help="Crop to (cropsize x cropsize) pixels")

    psr.add_argument('--cencrop', action='store_true', default=False,
                     help="Flag to crop center region else randomly crop")

    psr.add_argument('--lr', type=float, default=1e-4,
                     help="Learning rate")

    psr.add_argument('--encoder_lr', type=float, default=1e-4,
                     help="Learning rate of encoder network")

    psr.add_argument('--decoder_lr', type=float, default=1e-4,
                     help="Learning rate of decoder network")

    psr.add_argument('--max_iter', type=int, default=80_000,
                     help="Nubmer of iterations for training")

    psr.add_argument('--batch_size', type=int, default=16,
                     help="Batch size")

    psr.add_argument('--style_loss', type=str, default='meanstd',
                     help="Style loss: meanstd, gram")

    psr.add_argument('--style_loss_weight', type=float, default=100,
                     help="Style loss weight")

    psr.add_argument('--uncorrelation_loss_weight', type=float, default=0.01,
                     help="Uncorrelation loss weight")

    psr.add_argument('--check_iter', type=int, default=500,
                     help=("Interval of iterations "
                           "to save model and training images"))

    psr.add_argument('--load_path', type=str, default=None,
                     help="Mode load path")

    psr.add_argument('--save_path', type=str, default=None,
                     help="Mode save path")

    psr.add_argument('--wb_name', type=str, default=None,
                     help="Run name for wandb")

    psr.add_argument('--wb_notes', type=str, default=None,
                     help="Notes for wandb")

    psr.add_argument('--wb_tags', type=str, nargs='+', default=None,
                     help="Tags for wandb")

    return psr


def get_args():
    """Get arguments to run app."""
    psr = build_parser()
    args = psr.parse_args()

    # make a save dir.
    args.save_path = Path(args.save_path)
    if args.save_path.is_dir():
        args.save_path.mkdir(exist_ok=True)

    return args
