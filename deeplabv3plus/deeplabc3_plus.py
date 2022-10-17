import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import resnet


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.depthwise_activate = nn.ReLU()
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.pointwise_activate = nn.ReLU()

    '''forward'''
    def forward(self, x):
        x = self.depthwise_conv(x)
        if hasattr(self, 'depthwise_bn'): x = self.depthwise_bn(x)
        if hasattr(self, 'depthwise_activate'): x = self.depthwise_activate(x)
        x = self.pointwise_conv(x)
        if hasattr(self, 'pointwise_bn'): x = self.pointwise_bn(x)
        if hasattr(self, 'pointwise_activate'): x = self.pointwise_activate(x)
        return x


class DepwiseSeparableASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, **kwargs):
        super(DepwiseSeparableASPP, self).__init__()
        self.parallel_branches = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            if dilation == 1:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            else:
                branch = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
            self.parallel_branches.append(branch)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        size = x.size()
        outputs = []
        for branch in self.parallel_branches:
            outputs.append(branch(x))
        global_features = self.global_branch(x)
        global_features = F.interpolate(global_features, size=(size[2], size[3]), mode='bilinear',
                                        align_corners=False)
        outputs.append(global_features)
        features = torch.cat(outputs, dim=1)
        features = self.bottleneck(features)
        return features


class DeeplabV3Plus(nn.Module):
    def __init__(self, config, mode="train", pretrain_base=False):
        super(DeeplabV3Plus, self).__init__()
        self.config = config
        self.mode = mode
        assert self.mode.upper() in ['TRAIN', 'TEST']
        self.backbone = resnet.__dict__[config.deeplab.backbone](pretrained=pretrain_base, outstride=self.config.deeplab.outstride)
        aspp_config = {
            'in_channels': config.deeplab.aspp["in_channels"],
            'out_channels': config.deeplab.aspp["out_channels"],
            'dilations': config.deeplab.aspp["dilations"],
        }
        self.aspp_net = DepwiseSeparableASPP(**aspp_config)
        shortcut_config = config.deeplab.shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(shortcut_config['in_channels'], shortcut_config['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(shortcut_config['out_channels']),
            nn.ReLU()
        )
        decoder_cfg = config.deeplab.decoder
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            DepthwiseSeparableConv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], config.data.num_classes, kernel_size=1, stride=1, padding=0)
        )

    def load_backbone_checkpoint(self, state_dict, strict=True):
        self.backbone.load_state_dict(state_dict, strict=strict)

    def forward(self, x):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.backbone(x)
        # feed to aspp
        aspp_out = self.aspp_net(backbone_outputs[-1])
        aspp_out = F.interpolate(aspp_out, size=backbone_outputs[0].shape[2:], mode='bilinear',
                                 align_corners=True)
        # feed to shortcut
        shortcut_out = self.shortcut(backbone_outputs[0])
        # feed to decoder
        feats = torch.cat([aspp_out, shortcut_out], dim=1)
        predictions = self.decoder(feats)
        predictions = F.interpolate(predictions, size=img_size, mode='bilinear', align_corners=True)
        return predictions