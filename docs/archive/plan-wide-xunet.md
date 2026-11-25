> Archived 2026-10-16: done; the wide-model run completed. See [findings](../findings.md).

# Wide XuNet: a model-size test against R2

## Change and limits

`V3_MODEL=xunet` is the unchanged notebook 05 model and default. `V3_MODEL=xunet_wide` uses `V3_WIDTH=4` by default. The two models use the same frozen per-channel KV 5×5 filter, residual clip at ±3, `abs(conv1)`, `tanh`, five 2×2 average pools, global average pool, and two-class linear output. **Conv1 stays 3→16**, with the same weights and shape as XuNet: the wider path starts at conv2. This keeps the front end exact. Two 1×1 blocks after pool 5 add depth without another pool. Both models produce a 16×16 logit grid for a 512×512 tile: one cell per 32×32 pixels. The map code uses this grid and the final linear weights. The model stays fully convolutional before global pooling.

| Stage on a 512 tile | XuNet channels | Wide channels (`width=4`) | Wide kernel / output grid |
| --- | --- | --- | --- |
| KV + clip; conv1, absolute value, norm, tanh, pool | 3→16 | **3→16 (unchanged)** | 5×5 / 256×256 |
| conv2, norm, tanh, pool | 16→16 | 16→64 | 5×5 / 128×128 |
| conv3, norm, ReLU, pool | 16→32 | 64→128 | 1×1 / 64×64 |
| conv4, norm, ReLU, pool | 32→64 | 128→256 | 1×1 / 32×32 |
| conv5, norm, ReLU, pool | 64→128 | 256→512 | 1×1 / 16×16 |
| conv6, norm, ReLU; no pool | — | 512→512 | 1×1 / 16×16 |
| conv7, norm, ReLU; no pool | — | 512→256 | 1×1 / 16×16 |
| Global mean + classifier | 128→2 | 256→2 | 1×1 |

Parameters include 75 frozen KV coefficients. With `V3_NORM=none` (R2), XuNet has **18,941** and wide has **594,381** parameters. With `V3_NORM=group`, they have **19,453** and **597,869**. The 512-tile counts are about **1.56** and **5.03 billion FLOPs**, respectively. This counts each convolution multiply-add as two FLOPs, includes the KV filter and classifier, and excludes nonlinear, norm, pool, and transfer costs. The wide model has about 31× the R2 parameters and 3.23× its counted FLOPs. FLOPs do not predict training wall time directly: the new 1×1 layers use small grids, while the first and second 5×5 layers dominate the work.

## Comparison recipe and checks

The proposed job `scripts/g4/jobs/wide-1.env` changes only the model and run name from R2: no norm, clip 3, LR 3e-4, one warm-up epoch, gradient clip 1, drops at epochs 5 and 20, 30 epochs, `configs/train-k1to4.yaml`, decode, batch 32, 32 train and 16 eval workers, FP32, default compile, next-epoch prime, and extra bpp-1 test. The standard validation and test inputs stay unchanged. Old checkpoint hyperparameters have no `model` field; `model_from_hparams` uses XuNet for them. Strict load of a wide QUICK checkpoint with that factory passed. With `V3_MODEL=xunet`, notebook 05 has the same **13 state-dict keys and shapes** as R2's no-norm `last.ckpt`. The default XuNet run-name prefix is unchanged; wide names add `_wide-w4`.

The PC QUICK check used the R2 recipe with PC-safe overrides: 6 train workers, 4 eval workers, batch 8, and QUICK's small photo set. All 16 code cells passed in 367.313 s. Its 60 CSV metric rows (30 epochs, train and validation) were finite. This checks execution and not convergence. A prior PC QUICK attempt with **32 workers and batch 32** was killed by the container (`memory.events` recorded `oom_kill=1`); those settings are for the G4, not the PC. A synthetic, compiled batch-32 loop had finite loss and sampled unclipped gradient norms **0.00021–0.27224** for wide FP32 over 200 measured steps. The synthetic batches are not a model-quality result or a full gradient survey.

| PC RX 9070 XT, batch 32, 512 tile, default compile; 20 warm-up + 200 measured steps | FP32 steps/s | FP32 peak allocated MiB | BF16 steps/s | BF16 peak allocated MiB |
| --- | ---: | ---: | ---: | ---: |
| XuNet, no norm | 6.706 | 1,808 | 16.134 | 1,144 |
| Wide, no norm | 5.115 | 2,762 | 13.259 | 1,592 |

This loop reuses a synthetic GPU input, so it **does not** time photo decoding, validation, worker lifetime, checkpoints, or the late R2 slowdown. Its FP32 model-time ratio is 6.706 / 5.115 = **1.31×**. If that ratio transfers to G4, R2's observed **2,868 s** fit becomes **3,761 s (~62.7 min)**. Add R2's ~250 s outside that fit cell for a total near **67 min (~$1.82 at $1.63/h)**; both are **estimates**, not G4 measurements. R2 slowed markedly in epochs 24–29, and wide can slow differently. A **95-minute** job timeout leaves about 28 minutes above this estimate and is below the 105-minute permitted maximum. Do not launch without approval. BF16 speed is only exploratory: changing precision can change training results, so it is not the R2 comparison job.
