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


# Fixed 30-filter spatial rich model (SRM) high-pass bank: Fridrich & Kodovsky,
# IEEE TIFS 7(3) 2012, as used by Ye-Net (Ye, Ni, Yi, IEEE TIFS 12(11) 2017).
# Each row is (divisor, integer 5x5 kernel). The kernel is integer/divisor, so
# its largest coefficient has magnitude 1. Order: 8 first-order, 4 second-order,
# 8 third-order, SQUARE3x3, 4 EDGE3x3, SQUARE5x5, 4 EDGE5x5.
SRM_KERNEL_TABLE = (
    (1, [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (1, [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (1, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (1, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]),
    (1, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]),
    (1, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (1, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (1, [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (2, [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]),
    (2, [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -2, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (2, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (2, [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -2, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]),
    (3, [[0, 0, -1, 0, 0], [0, 0, 3, 0, 0], [0, 0, -3, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]),
    (3, [[0, 0, 0, 0, -1], [0, 0, 0, 3, 0], [0, 0, -3, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (3, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -3, 3, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (3, [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -3, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, -1]]),
    (3, [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -3, 0, 0], [0, 0, 3, 0, 0], [0, 0, -1, 0, 0]]),
    (3, [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -3, 0, 0], [0, 3, 0, 0, 0], [-1, 0, 0, 0, 0]]),
    (3, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 3, -3, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (3, [[-1, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, -3, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]),
    (4, [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]),
    (4, [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (4, [[0, 0, 0, 0, 0], [0, 0, 2, -1, 0], [0, 0, -4, 2, 0], [0, 0, 2, -1, 0], [0, 0, 0, 0, 0]]),
    (4, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]),
    (4, [[0, 0, 0, 0, 0], [0, -1, 2, 0, 0], [0, 2, -4, 0, 0], [0, -1, 2, 0, 0], [0, 0, 0, 0, 0]]),
    (12, [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]),
    (12, [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    (12, [[0, 0, -2, 2, -1], [0, 0, 8, -6, 2], [0, 0, -12, 8, -2], [0, 0, 8, -6, 2], [0, 0, -2, 2, -1]]),
    (12, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]),
    (12, [[-1, 2, -2, 0, 0], [2, -6, 8, 0, 0], [-2, 8, -12, 0, 0], [2, -6, 8, 0, 0], [-1, 2, -2, 0, 0]]),
)
SRM_ARRAY = np.stack([np.array(m, dtype=np.float64)/d for d, m in SRM_KERNEL_TABLE]).astype(np.float32)


class SRMFilter(nn.Module):
    """Frozen depthwise SRM bank: 30 filters per colour channel -> 90 channels.

    Output channel 30*c + f is filter f applied to colour channel c.
    The attribute is named ``conv`` like KVFilter; weights stay out of training.
    """
    channels = 90

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 90, kernel_size=5, padding=2, groups=3, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor(SRM_ARRAY).repeat(3, 1, 1).unsqueeze(1))
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


def clipped_residuals(front, x, clip_t):
    """Protect fixed SRM residuals and clamp from outer AMP rounding.

    KV retains its historical autocast behavior. FP32 inputs outside AMP keep
    the old operations and checkpoint state exactly.
    """
    if isinstance(front, SRMFilter):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = front(x.float())
            return torch.clamp(x, -clip_t, clip_t) if clip_t is not None else x
    x = front(x)
    return torch.clamp(x, -clip_t, clip_t) if clip_t is not None else x


FRONT_ENDS = {'kv': (KVFilter, 3), 'srm': (SRMFilter, 90)}


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
    def __init__(self, clip_t=3.0, norm='group', front_end='kv'):
        super().__init__()
        if front_end not in FRONT_ENDS:
            raise ValueError(f'unsupported front end: {front_end}')
        self.clip_t = clip_t
        self.front_end = front_end
        front, channels = FRONT_ENDS[front_end]
        self.kv = front()  # historical attribute name; kept for checkpoint keys
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding=2)
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
        x = clipped_residuals(self.kv, x, self.clip_t)
        x = self.pool(torch.tanh(self.bn1(torch.abs(self.conv1(x)))))
        x = self.pool(torch.tanh(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        return x

    def forward(self, x, *, return_aux=False):
        x = self.features(x)
        if return_aux:
            return self.classifier(x.mean(dim=(-2, -1))), self.cell_head(x)
        x = x.mean(dim=(-2, -1))
        return self.classifier(x)

    def cell_logits(self, x):
        return F.conv2d(self.features(x), self.classifier.weight[:, :, None, None],
                        self.classifier.bias)


class XuNetWide(XuNet):
    """Notebook 05 wide model; same 32 px output cells and linear cell logits."""
    def __init__(self, clip_t=3.0, norm='group', width=4, front_end='kv'):
        super().__init__(clip_t=clip_t, norm=norm, front_end=front_end)
        a, b, c, d, e = (16, 16*width, 32*width, 64*width, 128*width)
        for name, ni, no, kernel in ((1,FRONT_ENDS[front_end][1],a,5),(2,a,b,5),(3,b,c,1),
                                     (4,c,d,1),(5,d,e,1),(6,e,e,1),(7,e,d,1)):
            setattr(self, f'conv{name}', nn.Conv2d(ni, no, kernel, padding=kernel//2))
            setattr(self, f'bn{name}', make_norm(no, norm))
        self.classifier = nn.Linear(d, 2)

    def features(self, x):
        x = clipped_residuals(self.kv, x, self.clip_t)
        x = self.pool(torch.tanh(self.bn1(torch.abs(self.conv1(x)))))
        x = self.pool(torch.tanh(self.bn2(self.conv2(x))))
        for name in (3, 4, 5):
            x = self.pool(F.relu(getattr(self, f'bn{name}')(getattr(self, f'conv{name}')(x))))
        for name in (6, 7):
            x = F.relu(getattr(self, f'bn{name}')(getattr(self, f'conv{name}')(x)))
        return x


class XuNetWideDelayed(XuNetWide):
    """Arm A: SRM XuNetWide with only the first two pool calls removed.

    Conv1/conv2 stay at full resolution; conv3/4/5 retain their pools, and
    conv6/7, the mean and classifier are unchanged. State keys and shapes
    exactly match XuNetWide with the same width and front end.
    """
    def __init__(self, clip_t=3.0, norm='none', width=4, front_end='srm'):
        if front_end != 'srm':
            raise ValueError('delayed-pooling arm requires the SRM front end')
        super().__init__(clip_t=clip_t, norm=norm, width=width, front_end=front_end)

    def features(self, x):
        x = clipped_residuals(self.kv, x, self.clip_t)
        x = torch.tanh(self.bn1(torch.abs(self.conv1(x))))
        x = torch.tanh(self.bn2(self.conv2(x)))
        for name in (3, 4, 5):
            x = self.pool(F.relu(getattr(self, f'bn{name}')(getattr(self, f'conv{name}')(x))))
        for name in (6, 7):
            x = F.relu(getattr(self, f'bn{name}')(getattr(self, f'conv{name}')(x)))
        return x


def model_from_hparams(hparams):
    model = hparams.get('model', 'xunet')
    clip_t, norm = hparams.get('clip_t', 3.0), hparams.get('norm', 'group')
    front_end = hparams.get('front_end', 'kv')
    if model == 'xunet':
        network = XuNet(clip_t=clip_t, norm=norm, front_end=front_end)
    elif model == 'xunet_wide':
        network = XuNetWide(clip_t=clip_t, norm=norm, width=int(hparams.get('width', 4)), front_end=front_end)
    elif model == 'xunet_wide_delayed':
        network = XuNetWideDelayed(clip_t=clip_t, norm=norm, width=int(hparams.get('width', 4)), front_end=front_end)
    elif model == 'gbras':
        from .gbras import GBRASNet
        network = GBRASNet(clip_t=clip_t, norm=norm, front_end=front_end)
    else:
        raise ValueError(f'unsupported checkpoint model: {model}')
    supervision = hparams.get('cell_supervision', 'off')
    if supervision != 'off':
        if (supervision != 'changed_cells_v1' or model != 'xunet_wide' or
                front_end != 'srm' or int(hparams.get('width',4)) != 4 or norm != 'none'):
            raise ValueError('cell head requires changed_cells_v1 with SRM wide-w4, norm none')
        # The new head must not perturb the historical backbone/training RNG.
        with torch.random.fork_rng(devices=[]):
            network.cell_head = nn.Conv2d(network.classifier.in_features, 1, 1)
    return network


