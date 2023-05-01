# -*- coding: utf-8 -*-
"""Network code.

* Author: Minseong Kim(tyui592@gmail.com)
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights, vgg19, VGG19_Weights


class Model(nn.Module):
    """Style Transfer Network."""

    def __init__(self, backbone='vgg19', swap_max_pool=False):
        """Init with layer indices."""
        super(Model, self).__init__()
        if backbone == 'vgg16':
            layer_index = [15]

        elif backbone == 'vgg19':
            layer_index = [20]

        else:
            raise NotImplementedError(f"Not expected backbone: {backbone}")

        self.encoder = Encoder(backbone, layer_index, swap_max_pool)
        self.decoder = Decoder(backbone, layer_index)
        self.adain = AdaIN()

    def forward(self, content, style, style_strength=1.0):
        """Forward pass."""
        # encode the content image
        content_feature = self.encoder(content)

        # encode multiple style images
        style_feature = self.encoder(style)

        # style transform
        transformed_feature = self.adain(content_feature[0],
                                         style_feature[0],
                                         style_strength)

        # generate the stylized feature
        stylized_image = self.decoder(transformed_feature)

        # get output image and features to reduce correlation
        if self.training:
            return (stylized_image,
                    content_feature,
                    style_feature)
        else:
            return stylized_image


class AdaIN(nn.Module):
    """Adaptive instance normalization."""

    def __init__(self):
        """init."""
        super(AdaIN, self).__init__()

    def forward(self, content, style, style_strength=1.0):
        """forward."""
        # flatten a feature map to the vector
        cv = content.flatten(2)
        sv = style.flatten(2)

        c_std, c_mean = torch.std_mean(cv, dim=2, keepdim=True)
        s_std, s_mean = torch.std_mean(sv, dim=2, keepdim=True)

        # normalize the content feature
        n_cv = (cv - c_mean) / (c_std + 1e-5)

        # inverse normalization with the statistics of style feature
        s_cv = (n_cv * s_std) + s_mean

        # interpolatation with the stylized feature and content feature
        out = (1 - style_strength) * content \
            + style_strength * s_cv.view_as(content)

        return out


class Encoder(nn.Module):
    """Encoder network."""

    def __init__(self,
                 backbone='vgg16',
                 layer_index=[3, 8, 15],
                 swap_max_pool=False):
        """Init with layer indicies."""
        super(Encoder, self).__init__()

        if backbone == 'vgg16':
            network = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        elif backbone == 'vgg19':
            network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        else:
            raise NotImplementedError(f"Not expected backbone: {backbone}")

        self.layers = nn.ModuleList()

        temp_seq = nn.Sequential()
        for i in range(max(layer_index)+1):
            if isinstance(network[i], nn.MaxPool2d) and swap_max_pool:
                _module = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            else:
                _module = network[i]
            temp_seq.add_module(name=str(i), module=_module)

            if i in layer_index:
                self.layers.append(temp_seq)
                temp_seq = nn.Sequential()

    def forward(self, x):
        """Forward."""
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class Decoder(nn.Module):
    """Decoder network."""

    def __init__(self,
                 backbone='vgg16',
                 layer_index=[15]):
        """Init with layer indices."""
        super(Decoder, self).__init__()

        if backbone == 'vgg16':
            network = vgg16(weights=None).features

        elif backbone == 'vgg19':
            network = vgg19(weights=None).features

        else:
            raise NotImplementedError(f"Not expected backbone: {backbone}")

        self.layers = nn.ModuleList()

        temp_seq = nn.Sequential()
        count = 0
        for i in range(max(layer_index)-1, -1, -1):
            if isinstance(network[i], nn.Conv2d):
                # get number of in/out channels
                out_channels = network[i].in_channels
                in_channels = network[i].out_channels
                kernel_size = network[i].kernel_size

                # make a [reflection pad + convolution + relu] layer
                temp_seq.add_module(name=str(count),
                                    module=nn.ReflectionPad2d((1, 1, 1, 1)))
                count += 1
                temp_seq.add_module(name=str(count),
                                    module=nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size))
                count += 1
                temp_seq.add_module(name=str(count),
                                    module=nn.ReLU())
                count += 1

            # change down-sampling(MaxPooling) --> upsampling
            elif isinstance(network[i], nn.MaxPool2d):
                temp_seq.add_module(name=str(count),
                                    module=nn.Upsample(scale_factor=2))
                count += 1

            if i in layer_index:
                self.layers.append(temp_seq)
                temp_seq = nn.Sequential()

        # append last conv layers without ReLU activation
        self.layers.append(temp_seq[:-1])

    def forward(self, x):
        """Forward."""
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


def get_networks(backbone, swap_max_pool=False):
    """Get model and loss network."""
    network = Model(backbone=backbone, swap_max_pool=swap_max_pool)

    if backbone == 'vgg16':
        layer_index = [3, 6, 15, 22]
    elif backbone == 'vgg19':
        layer_index = [1, 6, 11, 20]
    else:
        raise RuntimeError(f"Not expected backbone: {backbone}")

    loss_network = Encoder(backbone=backbone,
                           layer_index=layer_index,
                           swap_max_pool=False)
    loss_network.eval()
    for param in loss_network.parameters():
        param.requires_grad = False

    return network, loss_network
