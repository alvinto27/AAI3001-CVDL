"""Shared XuNet models. Layer names and operation order preserve old checkpoints."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KV_ARRAY = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2],
                     [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
                     [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0


class KVFilter(nn.Module):
    """Frozen depthwise KV residual filter from notebook 04 cell 14."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=5, padding=2, groups=3, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor(KV_ARRAY).view(1, 1, 5, 5).repeat(3, 1, 1, 1))
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


def make_norm(channels, norm):
    if norm == 'none':
        return nn.Identity()
    if norm == 'batch':
        return nn.BatchNorm2d(channels)
    if norm == 'group':
        return nn.GroupNorm(8, channels)
    raise ValueError(f'unsupported checkpoint norm: {norm}')


class XuNet(nn.Module):
    """Same layers and forward order as notebook 04 cell 14; expose pre-mean features."""
    def __init__(self, clip_t=3.0, norm='group'):
        super().__init__()
        self.clip_t = clip_t
        self.kv = KVFilter()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.bn1 = make_norm(16, norm)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn2 = make_norm(16, norm)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=1)
        self.bn3 = make_norm(32, norm)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn4 = make_norm(64, norm)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=1)
        self.bn5 = make_norm(128, norm)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(128, 2)

    def features(self, x):
        x = self.kv(x)
        if self.clip_t is not None:
            x = torch.clamp(x, -self.clip_t, self.clip_t)
        x = self.pool(torch.tanh(self.bn1(torch.abs(self.conv1(x)))))
        x = self.pool(torch.tanh(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(-2, -1))
        return self.classifier(x)

    def cell_logits(self, x):
        return F.conv2d(self.features(x), self.classifier.weight[:, :, None, None],
                        self.classifier.bias)


class XuNetWide(XuNet):
    """Notebook 05 wide model; same 32 px output cells and linear cell logits."""
    def __init__(self, clip_t=3.0, norm='group', width=4):
        super().__init__(clip_t=clip_t, norm=norm)
        a, b, c, d, e = (16, 16*width, 32*width, 64*width, 128*width)
        for name, ni, no, kernel in ((1,3,a,5),(2,a,b,5),(3,b,c,1),
                                     (4,c,d,1),(5,d,e,1),(6,e,e,1),(7,e,d,1)):
            setattr(self, f'conv{name}', nn.Conv2d(ni, no, kernel, padding=kernel//2))
            setattr(self, f'bn{name}', make_norm(no, norm))
        self.classifier = nn.Linear(d, 2)

    def features(self, x):
        x = self.kv(x)
        if self.clip_t is not None:
            x = torch.clamp(x, -self.clip_t, self.clip_t)
        x = self.pool(torch.tanh(self.bn1(torch.abs(self.conv1(x)))))
        x = self.pool(torch.tanh(self.bn2(self.conv2(x))))
        for name in (3, 4, 5):
            x = self.pool(F.relu(getattr(self, f'bn{name}')(getattr(self, f'conv{name}')(x))))
        for name in (6, 7):
            x = F.relu(getattr(self, f'bn{name}')(getattr(self, f'conv{name}')(x)))
        return x


def model_from_hparams(hparams):
    model = hparams.get('model', 'xunet')
    clip_t, norm = hparams.get('clip_t', 3.0), hparams.get('norm', 'group')
    if model == 'xunet':
        return XuNet(clip_t=clip_t, norm=norm)
    if model == 'xunet_wide':
        return XuNetWide(clip_t=clip_t, norm=norm, width=int(hparams.get('width', 4)))
    raise ValueError(f'unsupported checkpoint model: {model}')


