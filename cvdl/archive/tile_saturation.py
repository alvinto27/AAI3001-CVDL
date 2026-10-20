"""Opt-in per-tile first-tanh saturation; existing cvdl.saturation is unchanged.

The definition matches saturation.Stats.add: the share of inputs with |x| > 3,
strictly. First-tanh inputs are bn1(abs(conv1(clipped_front_end(image)))), exactly
as in XuNet.features. Use the evaluated model's own front end, clipping and bn1.
No classifier or later network layer is called and no probabilities are inferred.
"""
import torch


def first_tanh_inputs(network, images):
    """Return actual first-tanh inputs; caller controls eval/inference mode."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError('expected N x 3 x H x W images')
    x = network.kv(images.to(dtype=torch.float32))
    if network.clip_t is not None:
        x = torch.clamp(x, -network.clip_t, network.clip_t)
    return network.bn1(torch.abs(network.conv1(x)))


def tile_saturated_share(inputs):
    """One float64 fraction per tile, over all channels and spatial positions."""
    if inputs.ndim != 4 or min(inputs.shape) < 1 or not inputs.is_floating_point():
        raise ValueError('expected non-empty floating NCHW first-tanh inputs')
    values = inputs.detach().abs().flatten(1)
    return (values > 3).sum(1).to(torch.float64)/values.shape[1]


def first_tanh_share(network, images):
    """Only front-end/clip/conv1/abs/bn1; no score inference or model changes."""
    return tile_saturated_share(first_tanh_inputs(network, images))
