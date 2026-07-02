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