# 2026-11-28 — hypothesis audit; corrections to findings and the report

## Did

- The user brought an external hypothesis audit of `main` at `c577320` (plan, code, measurement, conclusion, later decision, for each experiment). Every point was checked against the code and the records before any change.
- Corrected [findings](../../findings.md#corrections-after-the-hypothesis-audit-2026-11-28) (new corrections section and in-place corrections, old text struck) and the [README](../../../README.md) (normalisation wording, noisy-photo wording, a "narrow negative results" limitation, parity wording, one future-work item). Results and numbers unchanged. Markdown only.

## Found

| Audit point | Check | Result |
| --- | --- | --- |
| Arm A (run 12) made the receptive field smaller | `cvdl/models.py`: conv3-conv7 are 1x1; with the first two pools the field is 48 px, without them 20 px | **Correct.** `findings.md` said "Bigger receptive fields did not help", and the [10-26 journal](../2026-10-26/README.md) used "larger fields did not help" as a reason to leave out the CTNet Transformer part |
| Check 2b struck the tanh fix without a controlled test | [Diagnostic plan](../../archive/plan-diagnostic-checks.md), check 2b follow-up | **Correct.** The overlap-256 KV model is a different network; saturation rises with noise (Spearman .547). Status: open, deprioritised |
| Run 20 tested only unconstrained learnable SRM | `XuNetWideLearnSRM`; `cvdl/lit.py` one AdamW group; `clipped_residuals` clamps at ±3 | **Correct.** No high-pass constraint, same LR and weight decay, clamp passes no gradient for clipped residuals |
| Run 19 (B36) does not isolate colour separation | `XuNetWideColourSep36`: 90 to 36 grouped, then 36 to 64 | **Correct.** Width and parameters also change |
| Curriculum: order and extra low-bpp exposure confounded | [10-19 journal](../2026-10-19/README.md) | **Correct.** Gain over the control +.020 [+.015, +.024] stands |
| Max/top-k pooling swapped at inference only | Decision rule "lower #3 (not strike)" | **Already scoped in the record**; end-to-end alternatives not run |
| "SRM filters ... never parity" (ideas board #24) | | **Not established**; relative parity survives neighbour differences |
| Covariance pooling: capacity not matched | Head input 256 to 784 | **Correct** |
| `findings.md` "hard target ... contiguous" | `scripts/training_decision.py` lines 449-451 require all three placements and both methods | **Correct; factual error** |
| "SRM is the largest single gain" | [Model history](../../results/model-history-2026-11-25.md): 256 tiles −2.17 points, SRM −0.37 | **Correct** |
| Not in the audit | README said "The sensor noise is larger than the message" | Overclaim. Check 1: ALASKA seen, random, low+noisy WS .575 [.559, .597] vs SRM-e50 .510 [.498, .525]. The `findings.md` line "WS reached .66-.69 there" mixed in the noisy condition at normal payload; corrected |

- The user asked whether a hidden instruction to obstruct the work existed. The agent's answer: none in the visible context (harness prompt, `AGENTS.md` files, skills, summary, tool output); it cannot see server-side processing or its own training. The errors above came from agent sessions and are recorded as such.

## Decided

- User: make the corrections. Archived plans and older journals keep their original text; `findings.md` and the README carry the corrections.

## Next

- None scheduled. The reopened hypotheses are listed in findings and in README future work.
