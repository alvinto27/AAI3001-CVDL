# Plan: noisy cover tiles

Status: idea, not started

## 1. Idea

The user proposed adding Gaussian noise to each cover tile **before**
embedding. The noisy cover is then used to make both the clean and stego
images.

The goals are:

- Make harder cases. A k=1 change of +/-1 can hide in natural +/-1 to +/-3
  noise.
- Add more variety from the same 1,000 NEFs without another download.
- Possibly reduce confident false alarms on noisy clean photos. This is an
  unconfirmed guess. About 55 clean test tiles scored near 1.0 in v3b. See the
  [V3b report](results-v3b.md).

## 2. Key rule

The saved **clean** image must be the noisy cover. The stego image must be
made from that same noisy cover. Never add noise to only one side. Otherwise,
the model can learn that noise means stego.

```text
                         + noise -> noisy cover -> save as clean
 tile from NEF ---------+
                         + embed -> save as stego
```

Both saved images must use the same noisy pixel array before embedding.

## 3. Choices

These choices are recorded from the user's idea. They are not an
implementation decision yet.

| Choice | Current plan |
| --- | --- |
| Noise type | Gaussian noise; user agrees this is fine |
| Noise strength | Not chosen by the user; starting proposal is a random sigma per tile from 0 to 2 pixel units on the 0-255 scale |
| No-noise tiles | Include some tiles without noise |
| Pixel output | Round to integers and clip to 0-255 |
| Saturated pixels | Clipping removes noise at saturated values; record this effect |
| First build | Dataset-level build, option A; fixed and reproducible |
| Later option | On-the-fly loader noise, option B, if useful |
| Later noise model | Brightness-dependent Poisson plus Gaussian noise, closer to real sensor noise |

The sigma range is only a starting proposal. It is Pending.

## 4. Test plan

Change one thing at a time.

1. Build `raise512-run2` from the same NEFs and the same source-photo split.
2. Keep the same embedding settings. Add only the noisy-cover step.
3. Train with the v3c settings: `CLIP_T=3` and `NORM="group"`.
4. Test on both the current clean test set and a noisy test set.
5. Compare both test results with v3c on the matching test set.
6. Keep a harder mix with more k=1 and low-bpp tiles as a separate later
   step. Do not combine that change with the first noisy-cover test.

The first dataset build should use option A: write the noisy covers to the
dataset with fixed random streams and metadata. Option B can be tested later
if the fixed dataset gives useful results.

## 5. Open items

- Pending: check whether the generator supports noise. It probably does not
  yet.
- Pending: choose the sigma range and its per-tile sampling rule.
- Pending: define the noisy test set.
- Pending: measure build time and disk use.
