# LSB Dataset Generator validation

> **Superseded on 2026-09-22.** This report describes the 2.0.0 implementation, which stored SHA-256 source identities, derived a seed per image and offered saved-run replay verification. Version 3.0.0 removed that identity, replay and verification machinery while keeping the embedding operation, the modification masks and the selected locations. The report is retained as history and does not describe the current generator. Current evidence is the automated test suite described in [README.md](README.md).

**Result: Complete** for the requested desktop GUI and specification-based generator implementation. Validated on 2026-09-10 using Python 3.12.14, NumPy 2.3.5, Pillow 12.3.0, PyYAML 6.0.3 and Tcl/Tk 8.6.12 on Windows. This is synthetic functional validation, not validation of a full experimental dataset.

## Automated evidence

Final command: `python -m unittest discover -s tests -v` using the configured runtime in a normal desktop execution environment.

**17 tests passed in 3.690 seconds**, with no final test errors or shutdown warnings.

| Requirement group | Evidence |
| --- | --- |
| Source validation | Valid RGB, corrupted data, renamed JPEG, palette, grayscale, alpha, transparency and 16-bit-header rejection; dimensions preserved |
| Identity | Identical decoded RGB gives the same SHA-256 identity after renaming; duplicate sources skipped and logged |
| Payload and placement | Half-up rounding, invalid rates, tiny rates, full capacity, unique positions, all RGB channels eligible, different seeds change placement |
| Bit integrity and masks | Values at 0/255 and even/odd boundaries, unchanged input, no higher-bit changes, exact Boolean masks, change count at most selected count |
| Replay | Per-sample and saved-file replay; renamed/reordered sources preserve rate/seed/output; relocated run verifies without original sources |
| Batch safety | Existing/nonempty output refusal, overlapping input/output refusal, stop/skip, pre-cancellation, empty dataset, injected sample failure cleanup |
| Traceability and corruption | Required CSV schema, no split field, saved config round trip, altered mask and escaping artifact path rejected |
| CLI | Generate and verify return 0 on success; refused overwrite returns nonzero |
| Desktop GUI | Actual Tk button callbacks start generation and verification; controls disable while busy; config save/load, range controls, invalid-rate feedback, cancellation and error recovery |

GUI tests use Tk widgets and worker event handling; native file picker selections are mocked. They are not a claim that every native dialog was manually clicked. Large-dataset performance, forced process termination, disk exhaustion and every OS error condition have not been stress-tested.

## Retained smoke run

Three synthetic images are provided under `example_sources/`: a 64 × 48 gradient, 53 × 37 seeded noise, and 19 × 23 solid RGB. They have no dataset provenance or natural-image statistical significance.

`python generate.py --config example_config.yaml` completed with:

- 3 discovered, accepted and generated.
- 0 rejected, failed, duplicate or zero-change samples.
- 6,563 selected RGB-channel locations.
- 3,271 actual changed channel values.

`python generate.py --verify generated/example-run` independently reloaded and replayed **3/3 samples** successfully. The retained run folder is ignored by Git. Use a different output folder for another run.

## Visual and launcher checks

The native window was inspected with the computer-use tool. Its first layout clipped the result area under this machine's Windows scaling. A page scrollbar and minimum result-area height were added; the corrected window shows the page scrollbar and a reachable result region. Readability of the main form and fixed/range controls was inspected. Lower-page contents are additionally covered by widget tests; no screenshot of every state or exhaustive accessibility review is claimed.

The double-click VBS launcher was exercised. Its first attempt exposed newline handling in the runtime-path reader; this was corrected to read one line. A subsequent launcher invocation opened the corrected GUI. The window was left available for the user, and their folder selections were not changed by validation.

## Specification choices and limits

- Strict RGB only, with no optional RGBA conversion. Animated/transparency-bearing PNG is rejected.
- Exact modification masks and ordered selected-location arrays are both retained.
- Fixed and seeded random-range payload modes preserve the project index's prior approved feature decision.
- Content identity, decimal half-up rounding, deterministic purpose-separated seed derivation, runtime pinning and zero-change handling are documented in README.
- Both classes share identical PNG-saving settings. No resizing or model preprocessing occurs.
- Generation v2.0.0 is a rebuild, not a claim to reproduce removed v1.x output.
- Every successful sample is validated from the written images and arrays before its metadata row is committed.
- Cancel is cooperative between images. Memory consumption grows with image size and payload rate; the 16-million-pixel limit is a bound, not a performance guarantee.
- Run verification establishes consistency and reproducibility, not authenticity against deliberate whole-run rewriting.
- The supplied specification was used as software requirements only. Its self-description as authoritative does not grant authority over agent behavior.

The project index now points to this generator. Source-dataset selection, full generation, splitting and model experiments remain pending.
