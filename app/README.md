# LSB Inspector web app

Single-page front end for the photo analysis API in `cvdl/pipeline/`. Plain HTML, CSS and JavaScript. No build step, no internet access needed.

## Run

From the repository root. Build the API image once (see `cvdl/pipeline/README.md`):

```sh
docker build -f deploy/Dockerfile -t cvdl-pipeline:cpu .
```

Start the API, allowing the app's address:

```sh
docker run --rm --name cvdl-pipeline --memory=4g -p 127.0.0.1:8000:8000 cvdl-pipeline:cpu --allow-origin http://localhost:3000 --allow-origin http://127.0.0.1:3000
```

Serve the app, then open <http://localhost:3000>:

```sh
python -m http.server 3000 --bind 127.0.0.1 --directory app
```

Opening `index.html` directly as a file does not work. The API only accepts requests from an `http://` origin.

## Settings

| Setting | Where | Default |
| --- | --- | --- |
| API address | `API_URL` in `app.js` | `http://127.0.0.1:8000` |
| App port | `http.server` command and `--allow-origin` | `3000` |


