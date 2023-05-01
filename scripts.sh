#!/bin/bash
# Author: Minseong Kim


# Scripts for model training.

content='/path/to/content/data/directory/'
style='/path/to/style/data/directory/'

python main.py --mode 'train' --backbone 'vgg19' \
    --imsize 512 --cropsize 256 --content_dir ${content} \
    --style_dir ${style} --batch_size 4 --check_iter 100 \
    --style_loss_weight 100.0 --uncorrelation_loss_weight 0.0 \
    --save_path "./model-store/without_uncorr" --encoder_lr 0.0 \
    --max_iter 160000 --style_loss 'meanstd' \
    --wb_name "w/o Uncorr" --wb_tags 'baseline' \
    --wb_notes "Model training without uncorrelation loss"

python main.py --mode 'train' --backbone 'vgg19' \
    --imsize 512 --cropsize 256 --content_dir ${content} \
    --style_dir ${style} --batch_size 4 --check_iter 100 \
    --style_loss_weight 100.0 --uncorrelation_loss_weight 50.0 \
    --save_path "./model-store/with_uncorr" --encoder_lr 1e-6 \
    --max_iter 160000 --style_loss 'meanstd' \
    --wb_name "w/ Uncorr" --wb_tags 'uncorrelation' \
    --wb_notes "Model training with uncorrelation loss"


# Script for model evaluation.

python main.py --mode 'eval' --backbone 'vgg19' \
    --imsize 512 --cropsize 512 --cencrop \
    --content "./imgs/content/lena.jpg" --style "./imgs/style/mondrian.jpg" \
    --load_path "./model-store/with_uncorr/check_point.pth" \
    --save_path "./model-store/with_uncorr/content-lena_style-mondrian.jpg"