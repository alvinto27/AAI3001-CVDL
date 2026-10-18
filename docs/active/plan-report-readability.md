# Plan: report notebook readability (work list #5, revision 1)

Status: **built** in `fa17f25` (2026-10-18; Worker_2, finished by Worker_3). Approved to try (user, 2026-10-18); waiting for the user's review. Revises the first template (`716b2cd`, guide [report-notebook.md](report-notebook.md)). The user will review the result and may change it.

## Problem (user feedback on the demo reports)

- Hashes, evaluation id, threshold id and recipe id are not useful to a human reader.
- The tables are too dense. The reader does not know what to look at.

## Rule: answer first, details last

Each section starts with **the question it answers** and a **"Look at: ..."** line that says which part of the chart matters. Charts come before tables. At most one small table per main section. Full tables go to a collapsed appendix at the end.

| # | Question | What the reader sees | Full table |
| ---: | --- | --- | --- |
| 0 | How good is this model, in short? | 4-5 key numbers, each with one plain line (example: "Standard test: 3.1% error (about 1 in 32 wrong)") | none |
| 1 | What was trained? | Short plain card: model, front end, epochs, training tile, source mix, what differs from the reference | none |
| 2 | Did training go well? | One chart: train vs validation per epoch, LR drops marked; label which validation it is | none |
| 3 | Where does it fail? | One bar chart: false alarms and misses per test panel | appendix |
| 4 | Which hidden messages does it miss? | Miss rate by k (line); k x bpp grid in colour (dark = many misses) | appendix |
| 5 | Does it break on hard images? | Hard-test bars (FA next to miss, Wilson 95% ranges), coin-flip level drawn as a line | appendix |
| 6 | Better or worse than the reference? | Change only, green = better, red = worse (example: "ALASKA seen: +1.9 points worse") | appendix |
| 7 | Would knowing the source help? | One line per source: "RAISE: 2.1% -> 1.7% if the source were known" (reference only, no DIV2K) | appendix |
| 8 | Author notes | Placeholders; no generated claims | none |
| - | Appendix | All full tables, collapsed; for lookup | - |

## Presentation rules

- **Hidden:** hashes, evaluation / threshold / recipe ids, file paths, full-precision cut-off, setup and guard code. Keep the ids in the notebook metadata (not shown) so the generator can still check that two models used the same tests. The saved evaluation files keep everything.
- **Every chart has a title** that states what it shows in plain words (example: "Misses by k: srm-hard vs SRM-30").
- **Chart labels easy to read:** plain axis names with units ("Misses (%)", "Embedding strength k"), no code names or underscores, readable font size, no overlapping tick labels, a legend only when there is more than one series, values printed on bars where that helps.
- **Code cells collapsed;** show outputs only.
- **Plain labels:** "Threshold set on its own validation" (`train_matched`), "Threshold set after training" (`post_hoc_256`), "False alarms %", "Misses %", "Overall error %", "ALASKA (cameras seen in training)". Define each term (false alarm, miss, overall error, AUC, k, bpp) the first time it appears.
- **No duplicate rows:** a single-source panel shows one row, not "overall" and "source".
- **Rounding:** rates as "3.1%", AUC 3 decimals, cut-off 3 decimals. Display only; saved data unchanged.
- **No automatic interpretation.** Charts and numbers state facts; the author writes conclusions in section 8. Colour in section 6 shows direction only.

## Unchanged

Reads saved outputs only; CPU only; one cut-off per model; generator refuses missing, pending, QUICK, mismatched or ambiguous inputs; never overwrites author notes without `--overwrite`; no HTML export.

## Verification

- Tests: displayed numbers equal the saved CSV values after rounding; no hash or id string appears in any visible output; ids present in metadata; mismatch checks still work.
- Regenerate both demos (SRM-e50 alone; srm-hard with reference SRM-30) and give the user the paths.
