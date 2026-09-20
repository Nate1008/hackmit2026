# HonkPack — Scan. Pack. Honk your way out.

<p align="center">
  <img src="web/public/brand/honkpack-hero.png" alt="HonkPack goose filming packed items" width="360">

  <img width="1461" height="720" alt="Screenshot 2026-09-20 at 9 52 28 AM" src="https://github.com/user-attachments/assets/4a19958d-54bb-4de3-b16c-31cfd7a51071" />
  <img width="1461" height="720" alt="Screenshot 2026-09-20 at 9 50 00 AM" src="https://github.com/user-attachments/assets/17091b9c-3775-46d4-897a-a61768ddf1a7" />
  <img width="1461" height="720" alt="Screenshot 2026-09-20 at 9 48 54 AM" src="https://github.com/user-attachments/assets/98db0a08-2228-4996-8fbe-68b4a962577c" />

</p>

HonkPack turns a quick walk-around video of your packed belongings into an editable 3D
scene, then figures out how to fit everything into a rental vehicle, how many trips it
will take, and what the move will cost. Built at **HackMIT 2026** on top of Meta's
[ShapeR](https://github.com/facebookresearch/ShapeR) shape-generation model.

**How it works**

1. **Capture** — drag a 10–30 s MP4/MOV/WebM onto the site, or scan a QR code and record
   straight from your phone; the desktop session picks the upload up automatically.
2. **Reconstruct** — a GPU backend samples frames, detects and tracks each item with SAM3,
   estimates depth and camera poses with Depth Anything 3, and generates a metric,
   colored mesh per object with ShapeR. Progress streams live (frames → masks → point
   cloud → per-object mesh → room shell).
3. **Inspect** — every detected item lands in a Three.js scene viewer with an estimated
   weight and reconstructed dimensions.
4. **Pack** — choose a vehicle (RAV4, F-150, Transit 250, Isuzu NPR-HD), pick a packing
   strategy (balanced / weight / footprint), and HonkPack bin-packs the real meshes into
   the cargo envelope. Drag, rotate, and reassign items across trips; overlaps and payload
   limits are flagged.
5. **Route** — enter pickup and drop-off addresses (Google Places autocomplete), get a
   driving route and duration, and compare per-vehicle quotes
   (`trips × round-trip hours × hourly rate`).

## Repository layout

```
hackmit2026/
├── web/                     # Next.js 16 + React 19 frontend (Vercel / vinext)
│   ├── app/studio.tsx       #   landing, upload, live pipeline view, scene viewer
│   ├── app/truck-fitting.tsx#   vehicle packing, weight estimates, quotes, route map
│   ├── app/mobile-flow.tsx  #   QR phone pairing + mobile capture page
│   ├── app/api/shaper/      #   same-origin HTTP proxy to the backend (fallback)
│   └── public/vehicles/     #   vehicle GLBs used as cargo envelopes
├── backend/
│   ├── app.py               # FastAPI: jobs, SSE/WebSocket progress, pairing, maps
│   └── pipeline_service.py  # Stage-by-stage driver of the reconstruction pipeline
├── video_pipeline.py        # Frames → SAM3 → DA3 → ShapeR → colored GLBs
├── video_to_3d.py           # CLI entry point for the same pipeline
├── scene_postprocessing.py  # Room shell (floor/walls) TSDF fusion, Y-up conversion
├── shaper_runtime.py        # Cached ShapeR model loader shared across jobs
├── infer_shape.py           # Upstream ShapeR inference from preprocessed PKLs
├── model/ dataset/ preprocessing/ postprocessing/ evaluation/   # upstream ShapeR
├── scripts/                 # start_backend.sh, start_frontend.sh, public_proxy.mjs
├── tests/                   # pipeline unit tests (pytest); web/tests for the frontend
└── docs/                    # ShapeR project page assets
```

## Running locally

### Requirements

- A CUDA GPU host for the backend (Python 3.10, CUDA 12.8 — see [INSTALL.md](INSTALL.md)
  for the ShapeR environment, plus SAM3 and Depth Anything 3 weights).
- Node.js ≥ 22.13 for the frontend.
- Hugging Face access to the gated SAM3 checkpoint.

### Backend (GPU)

```bash
./scripts/start_backend.sh            # uvicorn backend.app:app on :8000
```

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SHAPER_JOB_ROOT` | `/tmp/shaper-video-jobs` | Where uploads and artifacts live |
| `SHAPER_JOB_TTL_SECONDS` | `604800` | Job expiry (7 days) |
| `SHAPER_MAX_UPLOAD_BYTES` | `1 GiB` | Upload cap |
| `HONKPACK_PAIRING_TTL_SECONDS` | `1200` | Phone QR link lifetime |
| `SHAPER_CORS_ORIGINS` | localhost + `https://honkpack.vercel.app` | Allowed frontend origins |
| `GOOGLE_MAPS_API_KEY` | — | Server key with **Routes API** |
| `GOOGLE_MAPS_PLACES_API_KEY` | falls back to above | Server key with **Places API (New)** |

There is no database or object store; everything is on local disk under the job root and
only one reconstruction runs on the GPU at a time. The loaded ShapeR runtime is reused
between jobs, so only the first request pays for model load and `torch.compile` warm-up.

### Frontend

```bash
./scripts/start_frontend.sh           # npm run dev in web/ on :3000
```

Set `NEXT_PUBLIC_SHAPER_API_URL` to point at a remote GPU backend (required for the
WebSocket progress and phone-pairing channels) and `NEXT_PUBLIC_GOOGLE_MAPS_API_KEY` to a
browser-restricted Maps JavaScript key. See [`web/.env.example`](web/.env.example).

### Command-line pipeline (no UI)

```bash
python video_to_3d.py input.mp4 --output-dir output/my_video \
  --prompt chair,table,lamp --max-frames 16 --max-objects 8 --preset balance
```

Produces `objects/*.glb`, `room_shell.glb`, `scene.glb`, and `work/manifest.json`.
Details in [VIDEO_PIPELINE.md](VIDEO_PIPELINE.md).

## API overview

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/jobs` | Upload a video (`prompts`, `max_frames`, `max_objects`, `preset`) |
| `GET` | `/api/jobs/{id}` · `/result` · `/files/{path}` | Status, final scene manifest, GLB/preview assets |
| `GET` / `WS` | `/api/jobs/{id}/events` · `/events/ws` | Live pipeline progress (SSE or WebSocket) |
| `DELETE` | `/api/jobs/{id}` | Cancel and remove artifacts |
| `POST` | `/api/pairings` → `/{id}/jobs` | Create a phone QR pairing; phone uploads through it |
| `GET` / `WS` | `/api/pairings/{id}/events` · `/events/ws` | Desktop waits for the phone upload |
| `GET` | `/api/maps/autocomplete?q=` | Address suggestions |
| `POST` | `/api/maps/route` | Driving distance, duration, encoded polyline |
| `GET` | `/api/health` | Backend and GPU queue status |

Interactive docs are served at `http://localhost:8000/docs`. Full notes in
[FULLSTACK.md](FULLSTACK.md).

## Deployment

The frontend deploys to Vercel with **Root Directory** set to `web`; the backend runs on
any GPU host reachable over HTTPS. See [`web/VERCEL.md`](web/VERCEL.md) for the required
environment variables and CORS setup.

## Tests

```bash
pytest tests/                 # pipeline refinement tests
cd web && npm test            # builds and checks the rendered app shell
cd web && npm run lint
```

## Built with

- [ShapeR](https://github.com/facebookresearch/ShapeR) — conditional 3D shape generation
- [SAM3](https://huggingface.co/facebook/sam3) — open-vocabulary detection and tracking
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) — depth, intrinsics and pose estimation
- FastAPI · Next.js · React Three Fiber · Google Maps Platform

## License and attribution

This project is a fork of [facebookresearch/ShapeR](https://github.com/facebookresearch/ShapeR).
The ShapeR code and weights are licensed under **CC-BY-NC**; see [LICENSE](LICENSE) and
[NOTICE](NOTICE). ShapeR and the selected Depth Anything 3 checkpoint are non-commercial
models. Vehicle assets in `web/public/vehicles/` carry their own license
(`web/public/vehicles/License.txt`).

If you build on the shape model, please cite the ShapeR paper:

```bibtex
@misc{siddiqui2026shaperrobustconditional3d,
      title={ShapeR: Robust Conditional 3D Shape Generation from Casual Captures},
      author={Yawar Siddiqui and Duncan Frost and Samir Aroudj and Armen Avetisyan and Henry Howard-Jenkins and Daniel DeTone and Pierre Moulon and Qirui Wu and Zhengqin Li and Julian Straub and Richard Newcombe and Jakob Engel},
      year={2026},
      eprint={2601.11514},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.11514},
}
```
