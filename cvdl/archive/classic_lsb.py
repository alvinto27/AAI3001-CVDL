"""Opt-in, CPU-only quantitative LSB-replacement detectors (no model imports).

This is a NumPy translation of ws.m and sp.m published by Jessica Fridrich:
https://dde.binghamton.edu/download/structural_lsb_detectors/ (2011 code).
WS: Fridrich & Goljan, 'On estimation of secret message length in LSB
steganography in spatial domain', SPIE 5306, 2004; moderated weights, KB
predictor and parity-bias correction follow Ker & Boehme, 'Revisiting Weighted
Stego-Image Steganalysis', SPIE 6819, 2008 (the reference ws.m, bias='yes').
SPA: Dumitrescu, Wu & Wang, 'Detection of LSB Steganography via Sample Pair
Analysis', IEEE TSP 51(7), 1995-2007, 2003. The reference sp.m credits the
preceding Dumitrescu, Wu & Memon ICIP 2002 formulation; we use its horizontal
sample pairs and quadratic, not a trained or fitted statistic.

Both return the estimated fraction of channel samples carrying random message
bits (NOT the fraction actually changed). WS's reference change-rate beta is
multiplied by two; SPA's quadratic already estimates the replacement rate p.
Unbounded estimates are retained for ranking, including negative cover estimates.
SPA follows the reference's real part for a negative discriminant and returns -1
when its quadratic has no coefficient (no same-upper-bits pairs). No test tuning.
Each RGB channel is independent; combined score is their arithmetic mean.
These statistics assume random LSB replacement, not matching or higher planes.
"""
import numpy as np


def _channel(image):
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) < 3 or image.dtype != np.uint8:
        raise ValueError('expected a uint8 channel at least 3 by 3')
    return image.astype(np.float64)


def _predict(x):
    return (-x[:-2, :-2]-x[2:, :-2]-x[2:, 2:]-x[:-2, 2:]
            + 2*(x[1:-1, :-2]+x[1:-1, 2:]+x[:-2, 1:-1]+x[2:, 1:-1])) / 4


def weighted_stego(image):
    """WS payload-rate estimate with the published moderated/bias correction."""
    x = _channel(image)
    # Only interiors are used, so boundary padding in localvar is irrelevant.
    windows = [x[i:i+x.shape[0]-2, j:j+x.shape[1]-2] for i in range(3) for j in range(3)]
    mean = sum(windows)/9
    variance = np.maximum(0, sum(v*v for v in windows)/9-mean*mean)
    weights = 1/(5+variance); weights /= weights.sum()
    flipped = x + 1 - 2*(x % 2)
    sign = x[1:-1, 1:-1]-flipped[1:-1, 1:-1]
    beta = np.sum(weights*(x[1:-1, 1:-1]-_predict(x))*sign)
    beta += beta*np.sum(weights*_predict(flipped-x)*sign)
    return float(2*beta)


def sample_pairs(image):
    """SPA payload-rate estimate, horizontal adjacent pairs as in sp.m."""
    x = _channel(image); u, v = x[:, :-1], x[:, 1:]
    xc = np.count_nonzero(((v % 2 == 0) & (u < v)) | ((v % 2 == 1) & (u > v)))
    zc = np.count_nonzero(u == v)
    wc = np.count_nonzero((u//2 == v//2) & (u != v))
    n = u.size; vc = n-xc-zc-wc
    a, b, c = (wc+zc)/2, 2*xc-n, vc+wc-xc
    if a == 0:
        return -1.0
    discriminant = b*b-4*a*c
    return float((-b-np.sqrt(max(0, discriminant)))/(2*a))


def scores(rgb):
    """R/G/B estimates and the pre-specified mean; never fit or calibrate."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('expected RGB pixels')
    result = {}
    for name, detector in (('ws', weighted_stego), ('spa', sample_pairs)):
        values = [detector(rgb[:, :, i]) for i in range(3)]
        result.update({f'{name}_{channel}': value for channel, value in zip('rgb', values)})
        result[name] = float(np.mean(values))
    return result
