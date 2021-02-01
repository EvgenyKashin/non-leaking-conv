from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from non_leaking_conv import Conv2dFactorized, Conv2dHamming, \
                             Conv2dHamingFactorized


def modify_model(model, kernel=3, padding=1, factorized=False, hamming=False,
                 big_first=False):
    def _replace_conv3x3(modules, parent_name, parent_module):
        for cur_name, cur_module in modules.items():
            if isinstance(cur_module, nn.Conv2d) \
                    and cur_module.kernel_size == (3, 3):
                if factorized and hamming:
                    ConvClass = Conv2dHamingFactorized
                elif factorized:
                    ConvClass = Conv2dFactorized
                elif hamming:
                    ConvClass = Conv2dHamming
                else:
                    ConvClass = nn.Conv2d

                if big_first and cur_module.in_channels == 3:
                    new_kernel = 9
                    new_padding = 5
                else:
                    new_kernel = kernel
                    new_padding = padding

                new_conv = ConvClass(cur_module.in_channels,
                                     cur_module.out_channels,
                                     new_kernel,
                                     cur_module.stride,
                                     new_padding,
                                     cur_module.dilation,
                                     cur_module.groups,
                                     cur_module.bias)
                setattr(parent_module, cur_name, new_conv)

            if len(cur_module._modules) > 0:
                _replace_conv3x3(cur_module._modules, cur_name, cur_module)
    _replace_conv3x3(model._modules, 'base', model)


def decrease_channel_width(model, coeff):
    def _decrease_func(modules, parent_name, parent_module):
        for cur_name, cur_module in modules.items():
            if isinstance(cur_module, nn.Conv2d) \
                    or isinstance(cur_module, Conv2dHamming):
                ConvClass = type(cur_module)
                in_channels = int(cur_module.in_channels * coeff)
                if cur_module.in_channels == 3:
                    in_channels = 3
                new_conv = ConvClass(in_channels,
                                     int(cur_module.out_channels * coeff),
                                     cur_module.kernel_size,
                                     cur_module.stride,
                                     cur_module.padding,
                                     cur_module.dilation,
                                     cur_module.groups,
                                     cur_module.bias)
                setattr(parent_module, cur_name, new_conv)

            elif isinstance(cur_module, nn.BatchNorm2d):
                new_bn = nn.BatchNorm2d(int(cur_module.num_features * coeff))
                setattr(parent_module, cur_name, new_bn)

            elif isinstance(cur_module, nn.Linear):
                new_ln = nn.Linear(int(cur_module.in_features * coeff),
                                   cur_module.out_features)
                setattr(parent_module, cur_name, new_ln)

            if len(cur_module._modules) > 0:
                _decrease_func(cur_module._modules, cur_name, cur_module)

    _decrease_func(model._modules, 'base', model)


def create_model(first_downsampling=True, num_classes=10):
    model = torchvision.models.resnet18(pretrained=False,
                                        num_classes=num_classes)
    if not first_downsampling:
        model.conv1 = nn.Conv2d(model.conv1.in_channels,
                                model.conv1.out_channels, kernel_size=(3, 3),
                                stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
    return model


def create_custom_model(kernel=3, padding=1, factorized=False, hamming=False,
                        big_first=False, wm=1.0, first_downsampling=False,
                        num_classes=10):
    model = create_model(first_downsampling, num_classes)
    modify_model(model, kernel, padding, factorized, hamming, big_first)
    decrease_channel_width(model, wm)
    return model


def calc_params(model):
    total_params = 0
    per_layer_params = {}
    for name, module in model.named_modules():
        layer_params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            layer_params += np.product(list(module.weight.size()))
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            layer_params += np.product(list(module.bias.size()))

        if layer_params != 0:
            per_layer_params[name] = layer_params
            total_params += layer_params

    return total_params, per_layer_params


class LitResnet(pl.LightningModule):
    def __init__(self, kernel=3, padding=1, factorized=False, hamming=False,
                 big_first=False, wm=1, first_downsampling=False,
                 num_classes=10, lr=0.05, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_custom_model(kernel, padding, factorized, hamming,
                                         big_first, wm, first_downsampling,
                                         num_classes)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                    momentum=0.9, weight_decay=5e-4)
        scheduler_dict = {
            'scheduler': OneCycleLR(optimizer, 0.1,
                                    epochs=self.trainer.max_epochs,
                                    steps_per_epoch=len(
                                        self.datamodule.train_dataloader())),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--kernel', default=3, type=int,
                            help='kernel_size for all conv2d layers')
        parser.add_argument('--padding', default=1, type=int,
                            help='padding for all conv2d layers')
        parser.add_argument('--factorized', action='store_true',
                            help='Use model with spatial factorized '
                                 'conv2d layers')
        parser.add_argument('--hamming', action='store_true',
                            help='Use model with non leaking hamming '
                                 'conv2d layers')
        parser.add_argument('--big_first', action='store_true',
                            help='Use model with big first conv2d '
                                 'layer with kernel_size 9x9')
        parser.add_argument('--wm', default=1.0, type=float,
                            help='channel width multiplier for all '
                            'conv2d layers')
        parser.add_argument('--first_downsampling', action='store_true',
                            help='Use model with first downsampling ('
                                 'stride and maxpool)')
        parser.add_argument('--num_classes', default=10, type=int,
                            help='output size of last layer')
        parser.add_argument('--lr', default=0.05, type=float,
                            help='learning rate')
        return parser
