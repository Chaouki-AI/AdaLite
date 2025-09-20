import torch
import torch.nn as nn
from torchvision import models
from .efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class DenseDecoderBlock(nn.Module):
    """DenseNet-like block with concatenation of features"""
    def __init__(self, in_channels, growth_rate=32, upsample=True):
        super(DenseDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(),
            nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if upsample else None
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out = torch.cat([x, x1, x2], dim=1)  # Dense concatenation
        
        if self.upsample is not None:
            out = self.upsample(out)
        return out

class Generator_ResNet_v2(nn.Module):
    def __init__(self, args):
        super(Generator_ResNet_v2, self).__init__()
        efficientnet_version=args.backbone
        self.args = args
        # Load pretrained EfficientNet as encoder
        encoder = EfficientNet.from_pretrained(f'efficientnet-{efficientnet_version}')
        
        encoder._fc = nn.Identity()
        encoder._avg_pooling = nn.Identity()
        encoder._dropout = nn.Identity()

        # Replace the conv head with a conv that outputs desired channels
        # Example: 256 output channels
        desired_channels = 320
        encoder._conv_head = nn.Conv2d(
            in_channels=encoder._conv_head.in_channels,
            out_channels=desired_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
                )

        # Keep batchnorm if you want (optional)
        encoder._bn1 = nn.BatchNorm2d(desired_channels)
        self.encoder = encoder



        # Get encoder output channels based on version
        encoder_channels = {'b1': 320}[efficientnet_version]
        
        # Decoder with DenseNet blocks
        self.decoder = nn.Sequential(
            # Initial projection
            nn.Conv2d(encoder_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # Dense Block 1 (with upsampling)
            DenseDecoderBlock(512, growth_rate=64),
            
            # Dense Block 2
            DenseDecoderBlock(512 + 64*2, growth_rate=64),
            
            # Dense Block 3
            DenseDecoderBlock(512 + 64*4, growth_rate=32),
            
            # Dense Block 4
            DenseDecoderBlock(512 + 64*4 + 32*2, growth_rate=32),
            
            # Final layers
            nn.Conv2d(512 + 64*4 + 32*4, 256, kernel_size=1),
        )
        self.out = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.ReLU())
        self.intermediate = None
            
    def forward(self, x):
        # Encoder forward pass
        x = self.encoder.extract_features(x)
        x = self.decoder(x) 
        self.intermediate = x
        return  self.out(x) #* self.args.max_depth

    def get_1x_lr_params(self):
        modules = [self.encoder]
        for m in modules:
            yield from m.parameters()

    def get_10x_lr_params(self):
        modules = [self.decoder]
        for m in modules:
            yield from m.parameters()
            
    def prune(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                filter_norms = module.weight.abs().sum(dim=(1,2,3))
                prune_indices = filter_norms.argsort()[:int(0.1 * len(filter_norms))]
                module.weight.data[prune_indices] = 0
        return model

    def __init__(self, args=None):
        super().__init__()
        self.block1 = DenseDecoderBlock2(1280, 112, 256)
        self.block2 = DenseDecoderBlock2(256, 40, 256)
        self.block3 = DenseDecoderBlock2(256, 24, 256)
        self.block4 = DenseDecoderBlock2(256, 16, 256)
        self.output = ParallelDilatedConvBlock(256)
        
    def forward(self, out):
        res1 = self.block1(out[-1], out[-2])
        res2 = self.block2(res1, out[-3])
        res3 = self.block3(res2, out[-4])
        res4 = self.block4(res3, out[-5])
        return self.output(res4), res4