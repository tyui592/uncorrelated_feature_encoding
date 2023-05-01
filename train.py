# -*- coding: utf-8 -*-
"""Train code.

* Author: Minseong Kim(tyui592@gmail.com)
"""

import wandb
import torch
from data import get_dataloader
from optim import get_optimizer
from network import get_networks
from collections import defaultdict
from utils import avg_values, ten2pil, calc_nonzero_channel
from loss import (calc_l2_loss,
                  calc_uncorrelation_loss,
                  calc_gram_loss,
                  calc_meanstd_loss)


def train_network(args):
    """Training network."""
    # set device
    device = torch.device('cuda' if args.gpu_no >= 0 else 'cpu')

    # get style transfer network and loss network
    network, loss_network = get_networks(backbone=args.backbone,
                                         swap_max_pool=args.swap_max_pool)
    network.to(device)
    loss_network.to(device)

    # data loader
    content_loader = get_dataloader(path=args.content_dir,
                                    imsize=args.imsize,
                                    cropsize=args.cropsize,
                                    cencrop=args.cencrop,
                                    max_iter=args.max_iter,
                                    batch_size=args.batch_size)

    style_loader = get_dataloader(path=args.style_dir,
                                  imsize=args.imsize,
                                  cropsize=args.cropsize,
                                  cencrop=args.cencrop,
                                  max_iter=args.max_iter,
                                  batch_size=args.batch_size)

    # optimizer
    optimizer = get_optimizer(network=network,
                              lr=args.lr,
                              encoder_lr=args.encoder_lr,
                              decoder_lr=args.decoder_lr)

    # training log
    logs = defaultdict(list)
    for i, (content, style) in enumerate(zip(content_loader, style_loader)):
        content = content.to(device)
        style = style.to(device)

        # get a output image and features
        output, cf_from_encoder, sf_from_encoder = network(content, style)

        # features from loss network
        sf_from_loss_network = loss_network(style)
        cf_from_loss_network = loss_network(content)
        of_from_loss_network = loss_network(output)

        # content loss
        content_loss = calc_l2_loss(features=of_from_loss_network[-1:],
                                    targets=cf_from_loss_network[-1:])

        # style loss
        if args.style_loss == 'meanstd':
            style_loss = calc_meanstd_loss(features=of_from_loss_network,
                                           targets=sf_from_loss_network)

        elif args.style_loss == 'gram':
            style_loss = calc_gram_loss(features=of_from_loss_network,
                                        targetrs=sf_from_loss_network)

        else:
            raise NotImplementedError(f"Not expected loss: {args.style_loss}")

        # uncorrelation loss
        content_uncorrelation_loss = calc_uncorrelation_loss(cf_from_encoder)
        style_uncorrelation_loss = calc_uncorrelation_loss(sf_from_encoder)
        uncorrelation_loss = (content_uncorrelation_loss
                              + style_uncorrelation_loss) * 0.5

        total_loss = content_loss \
            + style_loss * args.style_loss_weight \
            + uncorrelation_loss * args.uncorrelation_loss_weight

        # optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # save logs
        logs['content_loss'].append(content_loss.item())
        logs['style_loss'].append(style_loss.item())
        logs['uncorr_loss'].append(uncorrelation_loss.item())
        logs['total_loss'].append(total_loss.item())
        logs['nzch'].append(calc_nonzero_channel(cf_from_encoder
                                                 + sf_from_encoder))

        # check training
        if (i + 1) % args.check_iter == 0:
            # calcuate the avg with recent values.
            c_loss_avg = avg_values(logs['content_loss'])
            s_loss_avg = avg_values(logs['style_loss'])
            u_loss_avg = avg_values(logs['uncorr_loss'])
            t_loss_avg = avg_values(logs['total_loss'])
            nzch_avg = avg_values(logs['nzch'])

            # Save current input and output images
            training_img = ten2pil(torch.cat([content, style, output], dim=0),
                                   nrow=content.shape[0])
            wb_img = wandb.Image(training_img,
                                 caption=f"Iteration: {i + 1}")

            wandb.log({'Content Loss': c_loss_avg,
                       'Style Loss': s_loss_avg,
                       'Uncorr Loss': u_loss_avg,
                       'Total Loss': t_loss_avg,
                       'Nonzero Ch': nzch_avg,
                       'Training Image': [wb_img]},
                      step=i+1)

            # Save current check point
            torch.save({'iteration': i + 1,
                        'state_dict': network.state_dict(),
                        'logs': logs},
                       args.save_path / 'check_point.pth')

    return None
