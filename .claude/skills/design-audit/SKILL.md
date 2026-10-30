---
name: design-audit
description: Run the full UI/UX design audit for the app/ front end (Playwright at desktop, tablet and mobile, axe-core accessibility, screenshots, report). Use at the end of any UI change, or when asked to audit, QA or check the design.
---

# Design audit

Run this once when a UI task is finished, not after every edit.

1. Make sure nothing else is using ports 4173 or 8765. The tests start their own mock API and a production preview.
2. Run the audit:
   ```
   cd app && npm run audit
   ```
   This type-checks nothing by itself; run `npm run typecheck` first if you have not. The audit wipes `app/audit/`, builds the app against the mock API, runs Playwright for three viewports, and writes `app/audit/report.md`. A pass also writes the stamp that releases the Stop hook.
3. Read `app/audit/report.md`. Open the screenshots in `app/audit/screens/` (`desktop-`, `tablet-`, `mobile-` for landing, verdict-negative, verdict-positive, error-unsupported). Judge them against the rules in `CLAUDE.md`, and for the visual critique use `ui-ux-pro-max:ui-ux-pro-max` (UX, accessibility, responsive and interaction review) and `taste-skill:design-taste-frontend` (visual taste), when available. Their suggestions never override the rules in `CLAUDE.md`.
4. Fix failures in the UI layer only (`app/src/components`, `app/src/index.css`). Never edit `app/src/lib/api.ts`, validation, interpretation wording, or the Python service to make a check pass. If a test is wrong because the design deliberately changed, update the test and say why.
5. Re-run until it passes. Axe findings of impact serious or critical fail the run; minor and moderate ones are listed as advisory, so mention them in your summary.
6. Report: pass/fail counts, any advisory axe findings, and the paths of the report and screenshots. CI uploads the same files as the `design-audit` artifact.

Scenarios come from the upload's file name via the mock API: `clean` gives a negative verdict, `stego` a positive one.
