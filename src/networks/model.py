from src.efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from torch import nn
import torch

class EfficientNetEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Load pretrained EfficientNet
        self.backbone = EfficientNet.from_name(f'efficientnet-{args.backbone}', in_channels=3)
        self.args = args

        # Extract the "stages" of EfficientNet
        # stem conv + bn
        self.backbone._conv_head = nn.Identity()
        self.backbone._fc = nn.Identity()

        self.stem = nn.Sequential(
            self.backbone._conv_stem,
            self.backbone._bn0,
            self.backbone._swish
        )

        # Sequential blocks — EfficientNet has several "stages" of MBConv blocks
        self.blocks = self.backbone._blocks
        
    def forward(self, x):
        features = []

        # Stem
        x = self.stem(x)
        features.append(x)   # [B, 32, H/2, W/2]

        block_id_to_feature = [2, 4, 6, 11] if self.args.backbone == 'b0' else [2, 6, 11, 17] # pick indices where resolution changes

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in block_id_to_feature:
                features.append(x)
        return features  # list of tensors at multiple scales


class DenseDecoderBlock(nn.Module):
    """DenseNet-like block with concatenation of features"""
    def __init__(self, in_channels1, in_channels2,  growth_rate, upsample=True):
        super(DenseDecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels1),
            nn.ReLU(),
            nn.Conv2d(in_channels1, growth_rate, kernel_size=3, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(in_channels2),
            nn.ReLU(),
            nn.Conv2d(in_channels2, growth_rate, kernel_size=3, padding=1)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if upsample else nn.Identity()
        
    def forward(self, x, x2):
        x1 = F.interpolate(self.conv1(x), x2.shape[2:], mode='bilinear', align_corners=True)
        x2 = self.conv2(x2)
        return self.upsample(x1 + x2)


class Decoder(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        channels = [192, 80, 40, 24]  # Example channels from EfficientNet-B1
        self.decoderblock1 = DenseDecoderBlock(channels[0], channels[1], 72, upsample=False)
        self.decoderblock2 = DenseDecoderBlock(channels[1] , 72, 60, upsample=True)
        self.decoderblock3 = DenseDecoderBlock(channels[2] , 60, 48, upsample=True)
        self.decoderblock4 = DenseDecoderBlock(channels[3] , 48, args.dim_out, upsample=True)
        #self.final_conv = nn.Conv2d(36, 1, kernel_size=1)
        self.act = nn.Softmax(dim=1) if args.dim_out != 1 else nn.ReLU(inplace=True)
        
    def forward(self, features):
        out1 = self.decoderblock1(features[-1], features[-2] )
        out2 = self.decoderblock2(features[-2], out1)
        out3 = self.decoderblock3(features[-3], out2)
        out4 = self.decoderblock4(features[-4], out3)
        return self.act(out4)


class GlobalFeatureAggregator(nn.Module):
    def __init__(self, channels, hidden_dim=128, activation="relu"):
        """
        channels: list of input channels [C1, C2, C3, C4]
        hidden_dim: output dimension of Conv1d
        activation: activation function ("relu", "gelu", "sigmoid", etc.)
        """
        super(GlobalFeatureAggregator, self).__init__()
        
        self.total_channels = sum(channels)
        self.activation = activation
        
        # Conv1d expects input (B, C, L)
        #self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        
        # activation
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "softmax":
            self.act = nn.Softmax(1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # projection after conv
        self.proj = nn.Sequential( #nn.LayerNorm(self.total_channels),
                                   nn.Linear(self.total_channels, 128, bias=False), 
                                   nn.LeakyReLU(inplace=False), 
                                   nn.Linear(128, hidden_dim))
        self.drop = nn.Dropout(0.2)


    def forward(self, features):
        """
            features: list of tensors [(B, C1, H, W), (B, C2, H/2, W/2), ...]
        """
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in features]  # [(B, Ci)]
        x = torch.cat(pooled, dim=1)  # (B, sum(Ci))
        x = x.unsqueeze(1)  # (B, 1, total_channels)
        x = self.drop(x)
        x = x.squeeze(1)  # (B, total_channels)
        x = self.proj(x)  # (B, hidden_dim)
        x = self.act(x) + 0.1 if self.activation == 'relu' else self.act(x)
        return x


class Generator_ResNet(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.dim_out = 128 if not hasattr(self.args, 'dim_out') else self.args.dim_out
        self.encoder = EfficientNetEncoder(args)
        self.decoder = Decoder(args)
        self.bins = GlobalFeatureAggregator([args.dim_out, 24, 40, 80, 192], hidden_dim=args.dim_out, activation="relu") if args.dim_out != 1 else nn.Identity()
        print(self.bins)
        
    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        features[0] = out
        bins = self.bins(features)
        return self.results(out, bins) if self.args.dim_out != 1 else out 

    def results(self, feats, bins): 
        if self.bins.activation == 'softmax':
                y = bins
        else: 
                y = bins / bins.sum(dim=1, keepdim=True)
            
        bin_widths = (self.args.max_depth - self.args.min_depth) * y  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.args.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(feats * centers, dim=1, keepdim=True)

        return pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.bins]
        for m in modules:
            yield from m.parameters()

    
    @classmethod
    def build(cls, args, **kwargs):
        print('Building Encoder-Decoder model..', end='')
        m = cls(args=args, **kwargs)
        print(f'Done with {sum(p.numel() for p in m.parameters() if p.requires_grad)} trainable parameters.')
        return m
        
    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_parameters(self):  # lr learning rate
        modules = [self.decoder, self.bins, self.encoder]
        for m in modules:
            yield from m.parameters()
    
    
    def prune(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                filter_norms = module.weight.abs().sum(dim=(1,2,3))
                prune_indices = filter_norms.argsort()[:int(0.1 * len(filter_norms))]
                module.weight.data[prune_indices] = 0
        return model


    
    