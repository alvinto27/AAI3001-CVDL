# Image Integrity Inspector

React + Vite + Tailwind front end for the inference API in `cvdl/pipeline/`. It shows the verdict, where the model scores pixel behaviour as abnormal, and the evidence behind it. The Python service does all inference. This app only calls `GET /health` and `POST /analyze`.

## Run

Start the API with this app's origin allowed (see `cvdl/pipeline/README.md`):

```sh
docker run --rm --name cvdl-pipeline --memory=4g -p 127.0.0.1:8000:8000 cvdl-pipeline:cpu --allow-origin http://127.0.0.1:3000 --allow-origin http://localhost:3000
```

```sh
cd web
npm install
npm run dev        # http://127.0.0.1:3000
```

`VITE_API_URL` overrides the API address (default `http://127.0.0.1:8000`).

Without the model, `npm run mock-api` serves the same routes with synthetic scores. The file name picks the scenario: `*clean*`, `*stego*`, `*border*`, anything else gives a localised cluster.

`npm test` runs the interpretation tests, including one against a captured real `/analyze` response. `npm run build` type-checks and builds.

## Structure

| Path | Role |
| --- | --- |
| `src/lib/api.ts` | The only code that talks to the service. |
| `src/lib/interpret.ts` | Wire response to the `Analysis` model: verdict wording, regions, categories. |
| `src/lib/categories.ts` | Registry of anomaly categories. Add a row to support a new one. |
| `src/lib/regions.ts` | Connected regions from flagged heatmap cells. |
| `src/hooks/useSession.ts` | Upload, staged, analysing, results state machine. |
| `src/components/` | `ImageInspector` (zoom, pan, overlays, local readout) and the evidence panels. |

Optional fields the engine may add (`anomaly_category`, `anomaly_categories`, `regions`, `explanation`, `uncertainty`, `recommended_next_step`) are already read by `interpret.ts` and rendered when present.

## Interpretation rules

The model is a forensic anomaly detector. The verdict label is chosen in `decideVerdict`: a detection near the rule cutoff is `INCONCLUSIVE`, a detection spread over half the heatmap or more is `POSSIBLE HIDDEN DATA`, a localised one is `SUSPICIOUS MODIFICATION DETECTED`. That split is a wording heuristic, not a model output. The UI never names an editing tool.
