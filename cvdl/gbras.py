"""GBRAS-Net arm G: pinned author body with our SRM90 -> learned 1x1 adapter.

Reference: Tabares-Soto (Reinel) et al., "GBRAS-Net: A Convolutional Neural
Network Architecture for Spatial Image Steganalysis", IEEE Access 9,
14340-14350 (2021), doi:10.1109/ACCESS.2021.3052494.
Source: https://github.com/BioAITeam/Steganalysis/blob/
0b40ab046116beb64dc1b0973546cf9a197b883e/GBRAS-Net.ipynb,
GBRAS_Net() (cell 9) and its saved model.summary() (cell 20).

Layer audit (NCHW here; authors use NHWC). All convolutions stride 1 and
have zero bias initially; regular/separable convolutions use Glorot uniform.
DW=depthwise; Sep=DW3x3 multiplier 3 then pointwise1x1, bias only at output.
BN=beta only (no gamma), eps=.001, Keras momentum=.2 (Torch momentum=.8).
Activations precede BN. 'same' is zero padding; all pools are valid.
Counts below are the authors' Keras counts: learned values + moving mean/var.

Author layer  Operation                          C in->out  Size    Count
1 (body)      BN after adapted front end          30->30     256      90
2             DW1x1, bias, no activation          30->30     256      60
              Sep3x3 same, ELU(alpha=1)           30->30     256    3540
              BN                                 30->30     256      90
3             DW1x1, bias, no activation          30->30     256      60
              Sep3x3 same, ELU(alpha=1)           30->30     256    3540
              BN                                 30->30     256      90
skip1         add layer1 BN + layer3 BN           30->30     256       0
4             Conv3x3 same, ELU, BN               30->30     256    8220
5             Conv3x3 same, ELU, BN               30->30     256    8220
6             AvgPool2x2 stride2                  30->30     128       0
7             Conv3x3 same, ELU, BN               30->60     128   16440
8             DW1x1, bias, no activation          60->60     128     120
              Sep3x3 same, ELU, BN                60->60     128   12660
9             DW1x1, bias, no activation          60->60     128     120
              Sep3x3 same, ELU, BN                60->60     128   12660
skip2         add layer7 BN + layer9 BN           60->60     128       0
10            Conv3x3 same, ELU, BN               60->60     128   32640
11            AvgPool2x2 stride2                  60->60      64       0
12            Conv3x3 same, ELU, BN               60->60      64   32640
13            AvgPool2x2 stride2                  60->60      32       0
14            Conv3x3 same, ELU, BN               60->60      32   32640
15            AvgPool2x2 stride2                  60->60      16       0
16            Conv1x1, ELU, BN                    60->30      16    1920
17            Conv1x1, ELU, BN                    30->2       16      68
18            Global mean                         2->2        1       0
19            Softmax (loss/evaluator boundary)   2->2        1       0

Body: 165,818 Keras-count values = 164,734 learned + 1,084 moving statistics.
Author total 166,598 adds their frozen grayscale SRM (750 weights + 30 biases).
Our adaptation instead adds frozen SRM90 (2,250 weights, no bias), the unchanged
[-3,3] clamp, and learned 90->30 Conv1x1 (2,730 values, no activation). No
3*tanh, colour averaging, widened skips, extra body pool or Dense layer.
Our whole model: 167,464 learned; 169,714 Torch parameters including fixed SRM.
Torch additionally has 13 integer BN batch counters, not Keras parameters.

Paper/code comparison: the paper PDF could not be retrieved (IEEE HTTP 418)
so no unverified paper-text discrepancies are asserted. The pinned executable
code is authoritative: DW1x1 (not a guessed DW3x3), four pools (not five),
ELU before BN, and Conv2+ELU+BN BEFORE GAP/Softmax, not GAP->Dense. The final
softmax is supplied by the existing cross-entropy/evaluation contract; forward
returns its inputs, without changing the authors' predicted probabilities.

BN follows the authors' TensorFlow 2.2 fused default: normalization uses batch
population variance, running variance uses Bessel correction, as in Torch BN.
See tensorflow/v2.2.0 tensorflow/python/keras/layers/normalization.py,
_fused_batch_norm and _bessels_correction_test_only=True.

Batch composition: author notebook cells 18/23 use shuffled individual cover/
stego arrays, batch32 (Keras fit's default shuffle=True), not guaranteed pairs.
Our notebook keeps SRM-recipe unpaired OTF batches of128 images. If paired
training is enabled, pair_logits_targets concatenates B clean+B stego images:
BN then sees 2B correlated samples together, changing sampled statistics and
batch size, not the BN equation. This is NOT the authors' batch composition.
Validation concatenates pairs too, but Lightning eval mode uses fixed running
statistics; it must not normalize on the validation/test/deployment batch.

The final two score maps have stride16 and RF68 in eval mode. They are not a
linear-classifier CAM of the preceding 30 features (ELU+BN intervene). A
DIAGNOSTIC-ONLY 2x2 average yields stride32/RF84 class-score cells, preserving
mean logits exactly in real arithmetic. It does not insert a fifth body pool
or claim exact payload locations. BN training statistics are not local maps.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import SRMFilter


class GBRASBatchNorm(nn.BatchNorm2d):
    """Author BN: learned beta, no gamma; fused TF2.2 variance convention."""
    def __init__(self, channels):
        super().__init__(channels, eps=.001, momentum=.8, affine=False)
        self.bias = nn.Parameter(torch.zeros(channels))


def _conv(ni, no, kernel, *, groups=1, bias=True, depth_multiplier=None):
    layer = nn.Conv2d(ni, no, kernel, padding=kernel//2, groups=groups, bias=bias)
    if depth_multiplier is None:
        nn.init.xavier_uniform_(layer.weight)
    else:
        # Keras DW kernel shape is [k,k,C,m], not Torch [C*m,1,k,k].
        bound = math.sqrt(6 / (kernel*kernel*(ni+depth_multiplier)))
        nn.init.uniform_(layer.weight, -bound, bound)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class GBRASSeparable(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = _conv(channels, channels*3, 3, groups=channels,
                               bias=False, depth_multiplier=3)
        self.pointwise = _conv(channels*3, channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class GBRASBody(nn.Module):
    """Exact body after the authors' frozen SRM activation; see module audit."""
    def __init__(self):
        super().__init__()
        self.bn1 = GBRASBatchNorm(30)
        for number, channels in ((2,30),(3,30),(8,60),(9,60)):
            setattr(self, f'dw{number}', _conv(channels, channels, 1, groups=channels,
                                               depth_multiplier=1))
            setattr(self, f'sep{number}', GBRASSeparable(channels))
            setattr(self, f'bn{number}', GBRASBatchNorm(channels))
        for number, ni, no, kernel in ((4,30,30,3),(5,30,30,3),(7,30,60,3),
                (10,60,60,3),(12,60,60,3),(14,60,60,3),(16,60,30,1),(17,30,2,1)):
            setattr(self, f'conv{number}', _conv(ni, no, kernel))
            setattr(self, f'bn{number}', GBRASBatchNorm(no))
        for number in (6,11,13,15):
            setattr(self, f'pool{number}', nn.AvgPool2d(2,2))

    def forward(self, x):
        skip = self.bn1(x)
        x = skip
        for number in (2,3):
            x = getattr(self, f'bn{number}')(F.elu(getattr(self, f'sep{number}')(
                getattr(self, f'dw{number}')(x))))
        x = x+skip
        for number in (4,5):
            x = getattr(self, f'bn{number}')(F.elu(getattr(self, f'conv{number}')(x)))
        x = self.pool6(x)
        skip = self.bn7(F.elu(self.conv7(x)))
        x = skip
        for number in (8,9):
            x = getattr(self, f'bn{number}')(F.elu(getattr(self, f'sep{number}')(
                getattr(self, f'dw{number}')(x))))
        x = x+skip
        x = self.pool11(self.bn10(F.elu(self.conv10(x))))
        x = self.pool13(self.bn12(F.elu(self.conv12(x))))
        x = self.pool15(self.bn14(F.elu(self.conv14(x))))
        x = self.bn16(F.elu(self.conv16(x)))
        return self.bn17(F.elu(self.conv17(x)))


class GBRASNet(nn.Module):
    """Arm G adaptation only: our frozen/clamped SRM90 -> learned1x1 -> body."""
    cell_receptive_field = 84
    cell_map_label = 'GBRAS class-score margin (16 to 32 px)'

    def __init__(self, clip_t=3.0, norm='batch', front_end='srm'):
        super().__init__()
        if front_end!='srm' or clip_t!=3.0 or norm!='batch':
            raise ValueError('GBRAS arm requires front_end=srm, clip_t=3 and intrinsic norm=batch')
        self.clip_t, self.front_end = clip_t, front_end
        self.kv = SRMFilter()
        self.adapter = _conv(90,30,1)
        self.body = GBRASBody()

    def features(self, x):
        return self.body(self.adapter(torch.clamp(self.kv(x), -self.clip_t, self.clip_t)))

    def forward(self, x):
        return self.features(x).mean((-2,-1))

    def class_score_cells(self, x):
        """32px eval score cells and the UNCHANGED native16px pooled logits."""
        if self.training:
            raise ValueError('GBRAS cell maps require eval mode (fixed BN statistics)')
        if x.shape[-2]%32 or x.shape[-1]%32 or min(x.shape[-2:])<32:
            raise ValueError('GBRAS score cells require complete 32px cells')
        scores = self.features(x)
        return F.avg_pool2d(scores,2,2), scores.mean((-2,-1))

    def cell_logits(self, x):
        return self.class_score_cells(x)[0]
