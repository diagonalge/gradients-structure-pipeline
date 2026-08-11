# ds-structure-server

Internal Gradients worker that turns unstructured documents into instruct JSONL datasets.

## Role

- Receives jobs from the main Gradients API (`STRUCTURE_SERVICE_TOKEN` auth)
- Runs the structure pipeline (OpenRouter) with on-disk job folders (no DB)
- Uploads train JSONL to S3/MinIO and returns URLs in job status
- Main API owns billing + Postgres; FE polls main API; main API lazily refreshes from this service on GET

## Endpoints

| Method | Path | Notes |
|--------|------|--------|
| GET | `/healthz` | Public health |
| POST | `/v1/jobs` | Start job (`job_id`, `sources`, `num_rows`, …) |
| GET | `/v1/jobs/{job_id}` | Compact status + unique errors + single progress line |
| POST | `/v1/jobs/{job_id}/cancel` | **Internal only** — hidden from Swagger |
| POST | `/v1/suggest-rows` | Capacity / persona suggest |

## Env

```bash
STRUCTURE_SERVICE_TOKEN=...          # required shared secret
STRUCTURE_DATA_DIR=/data/ds-structure
STRUCTURE_MAX_JOBS=3
OPENROUTER_API_KEY=...
DS_STRUCTURE_WORKERS=100             # LLM concurrency (default 100)
S3_BUCKET_NAME=gradients
S3_COMPATIBLE_ENDPOINT=...
S3_COMPATIBLE_ACCESS_KEY=...
S3_COMPATIBLE_SECRET_KEY=...
S3_REGION=...
S3_SECURE=true
```

## Run locally

```bash
cd ds-structure-server
pip install -e .
export STRUCTURE_SERVICE_TOKEN=dev
export OPENROUTER_API_KEY=...
uvicorn ds_structure_server.main:app --host 0.0.0.0 --port 8080
```

## Job disk layout (while running)

```
$STRUCTURE_DATA_DIR/{job_id}/
  request.json
  status.json          # overwritten counters + progress line
  errors.jsonl         # unique errors only
  work/                # pipeline scratch
```

## Main Gradients API env

On the main API service, set:

```bash
STRUCTURE_SERVICE_URL=https://structure.example.com
STRUCTURE_SERVICE_TOKEN=...   # same shared secret
STRUCTURE_SERVICE_TIMEOUT=60
```

Upload + billing stay on the main API. Structure generation and suggest-rows are proxied here.
