# ftmi web UI

A single-process FastAPI app (`server.py`) + single HTML page (`index.html`) wrapping the
existing `ftmi` pipeline. No build step, no new heavy deps (FastAPI/uvicorn/sse-starlette
are already in `.venv`).

## Run it (on the cluster)

```bash
./webui/run.sh                 # GPU 3 (free), port 8000, bound to localhost
GPU=2 PORT=8123 ./webui/run.sh # override
```

## Reach it (from your laptop)

```bash
ssh -L 8000:localhost:8000 <user>@<cluster>
# open http://localhost:8000
```

(VS Code Remote-SSH forwards the port automatically — just open the URL.)

## Tabs

- **Dashboard** — base→final eval drift per app, read from `data/<app>/results/summary.json`.
- **Live steering** — pick a concept vector, set a coefficient, generate base vs steered
  output. Loads the base model (Qwen2.5-7B) once onto the pinned GPU (~10–60s), cached after.
  Mirrors `vectors/validate.py`: `add_steering(model, layer, v̂, coef)`, greedy decode.
- **Concept vectors** — per-domain vector/probe metadata from `data/<domain>/vectors/*.json`.
- **Launch runs** — spawn `ftmi run|train|eval --app …` as a subprocess; logs stream over SSE.
- **Full report** — embeds the static `data/report.html`.

## API

`GET /api/overview` · `GET /api/vectors` · `GET /api/steer/concepts` · `POST /api/steer`
· `GET /api/configs` · `POST /api/run` · `GET /api/runs` · `GET /api/runs/{id}/stream` (SSE)
· `POST /api/runs/{id}/stop` · `GET /report` · `GET /api/health`
