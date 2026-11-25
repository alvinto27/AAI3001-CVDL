"""ROCm-only workaround for the MIOpen 3.6.1 conv1 weight-gradient slowdown.

At batch >= 15, MIOpen 3.6.1 on gfx1201 cannot use Winograd for the 512x512
conv1 weight gradient. It falls back to GemmWrwUniversal: one long rocBLAS GEMM
per image, which rocBLAS splits into many launches (see rocm-libraries#12222).
Remove this workaround when MIOpen/rocBLAS are fixed.

This module keeps the forward pass and the data gradient in MIOpen. It computes
only the weight gradient itself: F.unfold per chunk of images, then one
strided-batched FP32 GEMM per image over pieces of the spatial axis, summed at
the end. The GEMM operands are strided views of the unfold buffer, so no extra
copy of that buffer is made. Extra memory: about 1.3 GiB at the default chunk
size (8 images) for the Xu-Net conv1 shape.

Do NOT set MIOPEN_GEMM_ENFORCE_BACKEND=5 with this module. In MIOpen 3.6.1 that
setting makes the 1x1 weight-gradient solver GemmWrw1x1_stride1 run on
hipBLASLt, which returns wrong weight gradients for batch > 1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_IMAGE_CHUNK = 8
_DEFAULT_PIECE = 1024


def _unfold_weight_grad(x, gy, weight, image_chunk, piece):
    """Return dW for a stride-1, dilation-1, ungrouped conv with 'same' padding."""
    n, _, height, width = x.shape
    kout, cin, kh, kw = weight.shape
    spatial = height * width
    if spatial % piece:
        raise ValueError(f"piece={piece} must divide H*W={spatial}")
    pieces = spatial // piece
    columns = cin * kh * kw
    out = torch.zeros((kout, columns), device=x.device, dtype=x.dtype)
    for start in range(0, n, image_chunk):
        stop = min(start + image_chunk, n)
        cols = F.unfold(x[start:stop], kernel_size=(kh, kw), padding=(kh // 2, kw // 2))
        for b in range(stop - start):
            # gy_v: [P, kout, piece], cols_v: [P, piece, columns]. Both are strided
            # views (unit stride in one matrix dim), so bmm reads them without a copy.
            gy_v = gy[start + b].reshape(kout, pieces, piece).permute(1, 0, 2)
            cols_v = cols[b].view(columns, pieces, piece).permute(1, 2, 0)
            out += torch.bmm(gy_v, cols_v).sum(dim=0)
    return out.reshape_as(weight)


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
        need_x = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]
        need_b = ctx.has_bias and ctx.needs_input_grad[2]
        gx = gw = gb = None
        if need_x:
            gx = torch.ops.aten.convolution_backward(
                gy, x, weight, None,
                stride=[1, 1], padding=list(ctx.padding), dilation=[1, 1],
                transposed=False, output_padding=[0, 0], groups=ctx.groups,
                output_mask=[True, False, False],
            )[0]
        if need_w:
            gw = _unfold_weight_grad(x, gy, weight, ctx.image_chunk, ctx.piece)
        if need_b:
            gb = gy.sum(dim=(0, 2, 3))
        return gx, gw, gb, None, None, None, None


class UnfoldWgradConv2d(nn.Conv2d):
    """nn.Conv2d with an unfold + strided-batched GEMM weight gradient."""

    def __init__(self, *args, image_chunk=_DEFAULT_IMAGE_CHUNK, piece=_DEFAULT_PIECE, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stride != (1, 1) or self.dilation != (1, 1):
            raise ValueError("UnfoldWgradConv2d requires stride=1 and dilation=1")
        if self.groups != 1:
            raise ValueError("UnfoldWgradConv2d requires groups=1")
        if not isinstance(self.padding, tuple) or self.padding != (
            self.kernel_size[0] // 2, self.kernel_size[1] // 2
        ):
            raise ValueError("UnfoldWgradConv2d requires 'same' padding (kernel_size // 2)")
        self.image_chunk = image_chunk
        self.chunk = image_chunk
        self.piece = piece

    @classmethod
    def from_conv(cls, conv: nn.Conv2d, chunk=_DEFAULT_IMAGE_CHUNK, piece=_DEFAULT_PIECE):
        new = cls(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            padding=conv.padding, groups=conv.groups, bias=conv.bias is not None,
            image_chunk=chunk, piece=piece,
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(conv.weight)
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        return new

    def forward(self, x):
        return _UnfoldWgradConvFn.apply(
            x, self.weight, self.bias, self.padding, self.groups, self.image_chunk, self.piece
        )


def _check(device="cuda"):
    """Check dW and dB against an fp64 CPU reference at the conv1 shape."""
    torch.manual_seed(0)
    for n in (1, 20, 32):
        ref = nn.Conv2d(3, 16, 5, padding=2)
        x = torch.randn(n, 3, 512, 512)
        gy = torch.randn(n, 16, 512, 512)
        ref64 = nn.Conv2d(3, 16, 5, padding=2).double()
        ref64.load_state_dict({k: v.double() for k, v in ref.state_dict().items()})
        ref64(x.double()).backward(gy.double())
        new = UnfoldWgradConv2d.from_conv(ref.to(device))
        new(x.to(device)).backward(gy.to(device))
        for name in ("weight", "bias"):
            a = getattr(ref64, name).grad
            b = getattr(new, name).grad.double().cpu()
            rel = ((a - b).abs().max() / a.abs().max()).item()
            print(f"N={n:2d} {name}: max rel diff vs fp64 {rel:.3e}")


if __name__ == "__main__":
    _check()
