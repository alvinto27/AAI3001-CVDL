"""ROCm-only workaround for the MIOpen 3.6.1 conv1 weight-gradient slowdown.

See rocm-libraries#12222. At batch >=15, Xu-Net's 512x512 conv1 falls back
from Winograd to per-image rocBLAS GEMM. Remove this workaround when
MIOpen/rocBLAS are fixed. It uses about 1 GiB extra memory.

Conv2d replacement with an unfold plus split-K bmm weight gradient.

The forward and data-gradient paths use MIOpen through convolution_backward.
The weight gradient reduces each image chunk with F.unfold and a large-batch
FP32 torch.bmm. The default chunk and piece sizes keep the unfold buffer near
1 GiB for the Xu-Net conv1 shape.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_IMAGE_CHUNK = 4
_DEFAULT_PIECE = 4096


def _unfold_weight_grad(x, gy, weight, image_chunk, piece):
    n, _, h, width = x.shape
    kout, cin, kh, kw = weight.shape
    l = h * width
    if l % piece:
        raise ValueError(f"piece={piece} must divide H*W={l}")
    npiece = l // piece
    out = torch.zeros_like(weight)
    for start in range(0, n, image_chunk):
        stop = min(start + image_chunk, n)
        bc = stop - start
        cols = F.unfold(
            x[start:stop],
            kernel_size=(kh, kw),
            padding=(weight.shape[-2] // 2, weight.shape[-1] // 2),
        )
        xp = cols.reshape(bc, cin * kh * kw, npiece, piece)
        xp = xp.permute(0, 2, 1, 3).reshape(
            bc * npiece, cin * kh * kw, piece
        )
        gp = gy[start:stop].reshape(bc, kout, npiece, piece)
        gp = gp.permute(0, 2, 3, 1).reshape(bc * npiece, piece, kout)
        products = torch.bmm(xp, gp).reshape(bc, npiece, cin * kh * kw, kout)
        out += products.sum(dim=(0, 1)).transpose(0, 1).reshape_as(weight)
    return out


class _UnfoldWgradConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, padding, groups, image_chunk, piece):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.padding = padding
        ctx.groups = groups
        ctx.image_chunk = image_chunk
        ctx.piece = piece
        return F.conv2d(x, weight, bias, padding=padding, groups=groups)

    @staticmethod
    def backward(ctx, gy):
        x, weight = ctx.saved_tensors
        pad = list(ctx.padding)
        groups = ctx.groups
        need_x = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]
        need_b = ctx.has_bias and ctx.needs_input_grad[2]
        gx = gw = gb = None
        common = dict(
            stride=[1, 1],
            padding=pad,
            dilation=[1, 1],
            transposed=False,
            output_padding=[0, 0],
            groups=groups,
        )
        if need_x:
            gx = torch.ops.aten.convolution_backward(
                gy,
                x,
                weight,
                None,
                output_mask=[True, False, False],
                **common,
            )[0]
        if need_w:
            gw = _unfold_weight_grad(
                x, gy, weight, ctx.image_chunk, ctx.piece
            )
        if need_b:
            gb = gy.sum(dim=(0, 2, 3))
        return gx, gw, gb, None, None, None, None


class UnfoldWgradConv2d(nn.Conv2d):
    """nn.Conv2d with an unfold/split-K bmm weight gradient."""

    def __init__(
        self,
        *args,
        image_chunk=_DEFAULT_IMAGE_CHUNK,
        piece=_DEFAULT_PIECE,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.stride != (1, 1) or self.dilation != (1, 1):
            raise ValueError("UnfoldWgradConv2d requires stride=1 and dilation=1")
        if not isinstance(self.padding, tuple):
            raise ValueError("UnfoldWgradConv2d requires tuple padding")
        self.image_chunk = image_chunk
        self.chunk = image_chunk
        self.piece = piece

    @classmethod
    def from_conv(
        cls,
        conv: nn.Conv2d,
        chunk=_DEFAULT_IMAGE_CHUNK,
        piece=_DEFAULT_PIECE,
    ):
        new = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            padding=conv.padding,
            groups=conv.groups,
            bias=conv.bias is not None,
            image_chunk=chunk,
            piece=piece,
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(conv.weight)
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        return new

    def forward(self, x):
        return _UnfoldWgradConvFn.apply(
            x,
            self.weight,
            self.bias,
            self.padding,
            self.groups,
            self.image_chunk,
            self.piece,
        )


def _check(device="cuda", n=32):
    """Check dW and db against a plain nn.Conv2d at the conv1 shape."""
    torch.manual_seed(0)
    ref = nn.Conv2d(3, 16, 5, padding=2).to(device)
    new = UnfoldWgradConv2d.from_conv(ref)
    x = torch.randn(n, 3, 512, 512, device=device)
    gy = torch.randn(n, 16, 512, 512, device=device)
    ref(x).backward(gy)
    new(x).backward(gy)
    for name in ("weight", "bias"):
        a = getattr(ref, name).grad
        b = getattr(new, name).grad
        rel = ((a - b).abs().max() / a.abs().max()).item()
        print(f"{name}: max rel diff {rel:.3e}")


if __name__ == "__main__":
    _check()
