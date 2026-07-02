import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    One convolutional block from the paper's Convolutional Stage:
    Conv2d -> BatchNorm -> ReLU
    (paper section 2, "Convolutional Stage" paragraph)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ShiftMLP(nn.Module):
    """
    Shifted MLP block (paper section 2, 'Shifted MLP' + equations 1-2).
    Shifts feature channels along one spatial axis before tokenizing,
    so the MLP that follows sees a locally-shifted window instead of
    the whole feature map at once.
    """
    def __init__(self, dim, shift_size=5):
        super().__init__()
        self.shift_size = shift_size
        self.mlp = nn.Linear(dim, dim)

    def shift(self, x, axis):
        B, C, H, W = x.shape
        pad = self.shift_size // 2
        x = nn.functional.pad(x, (pad, pad, pad, pad), mode='constant', value=0)
        x = torch.roll(x, shifts=self.shift_size // 2, dims=axis)
        x = x[:, :, pad:pad+H, pad:pad+W]
        return x

    def forward(self, x, axis):
        x = self.shift(x, axis)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x
    
class TokMLPBlock(nn.Module):
    """
    Full Tokenized MLP block (paper equations 1-4).
    Shift width -> MLP -> DWConv -> GELU -> Shift height -> MLP -> residual -> LayerNorm
    """
    def __init__(self, dim):
        super().__init__()
        self.shift_w = ShiftMLP(dim)
        self.shift_h = ShiftMLP(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depth-wise
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # Eq 1-2: shift width -> tokenize (handled inside ShiftMLP) -> MLP -> DWConv
        x = self.shift_w(x, axis=3)   # axis=3 is width
        x = self.dwconv(x)
        x = self.gelu(x)

        # Eq 3-4: shift height -> MLP -> add residual -> LayerNorm
        x = self.shift_h(x, axis=2)   # axis=2 is height
        x = x + residual

        # LayerNorm expects channels last, so permute -> normalize -> permute back
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
    
class UNeXt(nn.Module):
    def __init__(self, num_classes=1, in_ch=3):
        super().__init__()

        # Encoder: conv levels 1-3, tokmlp levels 4-5 (bottleneck)
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc3_to_4 = nn.Conv2d(128, 160, kernel_size=3, padding=1)  # channel bump before tokmlp
        self.enc4 = TokMLPBlock(160)
        self.enc4_to_5 = nn.Conv2d(160, 256, kernel_size=3, padding=1)  # channel bump for bottleneck
        self.bottleneck = TokMLPBlock(256)

        self.pool = nn.MaxPool2d(2)  # halves H and W, stride=2, same mechanism as Andrew Ng's max pooling

        # Decoder: tokmlp levels 4-3, conv levels 2-1, mirrored channel counts
        self.dec5_to_4 = nn.Conv2d(256, 160, kernel_size=3, padding=1)
        self.dec4 = TokMLPBlock(160)
        self.dec4_to_3 = nn.Conv2d(160, 128, kernel_size=3, padding=1)
        self.dec3 = TokMLPBlock(128)
        self.dec2 = ConvBlock(128, 64)
        self.dec1 = ConvBlock(64, 32)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # doubles H and W

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)  # 1x1 conv: squash to mask channels

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                 # (B, 32, 256, 256)
        p1 = self.pool(e1)                # (B, 32, 128, 128)

        e2 = self.enc2(p1)                # (B, 64, 128, 128)
        p2 = self.pool(e2)                # (B, 64, 64, 64)

        e3 = self.enc3(p2)                # (B, 128, 64, 64)
        p3 = self.pool(e3)                # (B, 128, 32, 32)

        e4_in = self.enc3_to_4(p3)        # (B, 160, 32, 32)
        e4 = self.enc4(e4_in)             # (B, 160, 32, 32)
        p4 = self.pool(e4)                # (B, 160, 16, 16)

        b_in = self.enc4_to_5(p4)         # (B, 256, 16, 16)
        b = self.bottleneck(b_in)         # (B, 256, 16, 16)

        # Decoder (with skip connections: add matching encoder output back in)
        d4_in = self.dec5_to_4(self.up(b))    # (B, 160, 32, 32)
        d4 = self.dec4(d4_in + e4)             # skip: add e4 back
        d3_in = self.dec4_to_3(self.up(d4))    # (B, 128, 64, 64)
        d3 = self.dec3(d3_in + e3)             # skip: add e3 back
        d2 = self.dec2(self.up(d3) )           # (B, 64, 128, 128)
        d2 = d2 + e2                            # skip: add e2 back
        d1 = self.dec1(self.up(d2))            # (B, 32, 256, 256)
        d1 = d1 + e1                            # skip: add e1 back

        out = self.final(d1)              # (B, num_classes, 256, 256)
        return out