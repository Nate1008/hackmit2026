# ShapeR Studio full-stack demo

The demo keeps uploads and generated artifacts only under `/tmp/shaper-video-jobs`.
Completed jobs expire after one hour by default; restarting the backend clears the
in-memory job index. There is no database or object storage.

## Start

Open two terminals from the repository root:

```bash
./scripts/start_backend.sh
```

```bash
./scripts/start_frontend.sh
```

Then open <http://localhost:3000>. FastAPI documentation is available at
<http://localhost:8000/docs>.

The frontend uses `http://localhost:8000` by default. To point it at another GPU
backend, set `NEXT_PUBLIC_SHAPER_API_URL` before starting the frontend.

## API

- `POST /api/jobs` — multipart video upload plus `prompts`, `max_frames`,
  `max_objects`, and `preset` (`speed`, `balance`, or `quality`).
- `GET /api/jobs/{job_id}/events` — Server-Sent Events for frames, SAM3,
  geometry, per-object ShapeR, room reconstruction, and completion.
- `GET /api/jobs/{job_id}` — current status/result snapshot.
- `GET /api/jobs/{job_id}/result` — final scene manifest.
- `GET /api/jobs/{job_id}/files/{path}` — previews and GLB artifacts.
- `DELETE /api/jobs/{job_id}` — cancel and delete temporary artifacts.
- `GET /api/health` — backend and single-GPU queue status.

Environment controls:

- `SHAPER_JOB_ROOT` (default `/tmp/shaper-video-jobs`)
- `SHAPER_JOB_TTL_SECONDS` (default `3600`)
- `SHAPER_MAX_UPLOAD_BYTES` (default 1 GiB)
- `SHAPER_CORS_ORIGINS` (default local frontend origins)

Only one reconstruction enters the GPU pipeline at a time. The loaded ShapeR
runtime is reused between jobs, so later jobs avoid model warm-up.
