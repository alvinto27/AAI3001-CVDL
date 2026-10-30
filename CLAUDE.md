# AAI3001-CVDL

Python/ML code lives in `cvdl/`. The Image Integrity Inspector front end lives in `app/` (React 19, Vite, Tailwind 4, TypeScript). The Python service owns all inference; the web app only calls `GET /health` and `POST /analyze`.

## Web design quality (permanent rules for `app/`)

- Keep the dark graphite workstation look: graphite surfaces, muted teal for the primary action, amber/pink heatmap colours reserved for heat and flagged regions. Tokens live in `app/src/index.css`.
- No gradients, glows, resting shadows or large border radii. No generic SaaS card grids or oversized hero sections.
- Text is at least 13px; secondary text uses `text-ink-2` or brighter. Body copy is 15px where space allows.
- Anything illustrative (such as the landing preview) is labelled as an example and never shows invented model scores.
- Desktop-first at 1920x1080 with no vertical scroll on the landing page; tablet (820) and mobile (390) stack and scroll without clipping or horizontal overflow.
- Do not change `app/src/lib/api.ts`, `/health` or `/analyze` behaviour, validation rules (`app/src/lib/file.ts`) or inference wording (`app/src/lib/interpret.ts`) as part of a visual change.
- Design skills: `taste-skill:design-taste-frontend` and `ui-ux-pro-max:ui-ux-pro-max` (both plugins). Use them for the visual critique at the end of a UI task, within the rules above.

## Verification workflow for UI changes

1. After each edit under `app/src/`, a PostToolUse hook runs `tsc --noEmit`. Fix what it reports. Do not run the browser suite per edit.
2. When a UI task is complete, run `/design-audit` once (`cd app && npm run audit`). It builds the app, starts a deterministic mock API, and runs Playwright at desktop, tablet and mobile with axe-core, then writes `app/audit/report.md` and screenshots.
3. A Stop hook blocks finishing while UI files are newer than the last passing audit. CI (`.github/workflows/web-design-qa.yml`) runs the same suite on pushes and pull requests touching `app/`.

## Testing without a GPU

`app/scripts/mock-api.mjs` mimics `/health` and `/analyze` with seeded scores. The uploaded file name picks the scenario: `*clean*` is a negative verdict, `*stego*` a positive one, `*border*` inconclusive, anything else a localised positive. Tests use it; the real service is never required.
