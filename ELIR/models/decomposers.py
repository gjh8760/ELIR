"""Retinex decomposers. Always frozen."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    return g2d


class MaxChannelDecomposer(nn.Module):
    """Retinex decomposition without learned weights.

    I(x) = GaussianBlur(max_c(x)),  R(x) = clip(x / I, 0, 1)
    """

    def __init__(self, kernel_size: int = 15, sigma: float = 3.0, eps: float = 1e-4):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.eps = eps
        kernel = _build_gaussian_kernel(kernel_size, sigma)[None, None]  # [1,1,k,k]
        self.register_buffer("gauss_kernel", kernel, persistent=False)
        self._pad = kernel_size // 2

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self._pad,) * 4, mode="reflect")
        return F.conv2d(x, self.gauss_kernel)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        I = x.max(dim=1, keepdim=True)[0]          # [B,1,H,W]
        I = self._blur(I).clamp(min=self.eps)
        R = (x / I).clamp(0.0, 1.0)                # [B,3,H,W]
        return R, I


class DecomNet(nn.Module):
    """RetinexNet Decom-Net (weichen582 / aasharma90 PyTorch port).

    Input : [B, 3, H, W] in [0,1]
    Output: [B, 4, H, W] = [R(3) | I(1)], each in (0,1) via sigmoid
    """

    def __init__(self, channel: int = 64, kernel_size: int = 3, num_blocks: int = 5):
        super().__init__()
        pad = kernel_size // 2
        # Input head takes [x_max, x] -> 4ch (RetinexNet paper's "image_max" trick)
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode="replicate")
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(channel, channel, kernel_size, padding=pad, padding_mode="replicate"))
            layers.append(nn.ReLU(inplace=True))
        self.net1_convs = nn.Sequential(*layers)
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=pad, padding_mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = x.max(dim=1, keepdim=True)[0]
        h = torch.cat([x_max, x], dim=1)
        h = self.net1_conv0(h)
        h = self.net1_convs(h)
        h = self.net1_recon(h)
        R = torch.sigmoid(h[:, :3])
        I = torch.sigmoid(h[:, 3:4])
        return torch.cat([R, I], dim=1)


class RetinexNetDecomposer(nn.Module):
    """Wraps RetinexNet DecomNet with pretrained weights, frozen."""

    def __init__(self, pretrained_path: str):
        super().__init__()
        self.net = DecomNet()
        state_dict = torch.load(pretrained_path, map_location="cpu")
        # Tolerate minor key mismatches (e.g., ported checkpoints may prefix with "module.")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict, strict=True)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # force eval
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        out = self.net(x)                 # [B, 4, H, W]
        R = out[:, :3]
        I = out[:, 3:4]
        return R, I
