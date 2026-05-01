# Deployment Guide

Free deployment using **Hugging Face Spaces** (backend) + **Vercel** (frontend).

## 1. Backend → Hugging Face Spaces (Docker, free CPU tier)

1. Create a new Space at https://huggingface.co/new-space
   - **SDK**: Docker
   - **Hardware**: CPU basic (free) — 2 vCPU, 16 GB RAM
   - **Visibility**: Public (required for free tier)
2. Clone your new empty Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name> hf-space
   cd hf-space
   ```
3. Copy these files from this repo into the Space repo, preserving paths:
   - `Dockerfile`
   - `.dockerignore`
   - `src/backend/` (entire folder)
   - `deepfake_detection/src/` (entire folder)
   - `deepfake_detection/configs/`
   - `deepfake_detection/requirements.txt`
   - `HUGGINGFACE_SPACE_README.md` → rename to `README.md` at the Space root
4. Set Space secrets / variables (Settings → Variables and secrets):
   - `CORS_ALLOWED_ORIGINS` = `https://<your-vercel-app>.vercel.app` (set this after Vercel is up)
5. Commit and push:
   ```bash
   git add . && git commit -m "Initial deploy" && git push
   ```
6. Wait for the Space to build (~5–10 min on first build, downloads PyTorch + the HF model).
7. Test:
   ```bash
   curl https://<your-username>-<space-name>.hf.space/api/health
   ```
   Should return `{"status":"ok","mode":"hf",...}`.

> **Heads-up**: Free Spaces sleep after ~48 hr of inactivity. First request after sleep takes ~30 s.

## 2. Frontend → Vercel (free Hobby tier)

1. Push your repo to GitHub (already done).
2. Go to https://vercel.com/new and import the GitHub repo.
3. Vercel auto-detects Vite. No build command changes needed.
4. Under **Environment Variables**, add:
   - `VITE_API_BASE` = `https://<your-username>-<space-name>.hf.space`
5. Deploy. You'll get a `<project>.vercel.app` URL.
6. Go back to your HF Space and update `CORS_ALLOWED_ORIGINS` to that URL, then restart the Space.

## 3. Local development

Unchanged:
```bash
npm run server   # terminal 1 — uses deepfake_detection/.venv
npm run dev      # terminal 2 — Vite proxies /api → 127.0.0.1:8000
```
The frontend uses Vite's dev proxy when `VITE_API_BASE` is unset, so local dev still works.

## Troubleshooting

- **Space build fails on PyTorch wheel** → confirm Dockerfile uses `python:3.11-slim` (not 3.14).
- **CORS errors in browser** → make sure the Vercel origin is in `CORS_ALLOWED_ORIGINS` on the Space.
- **OOM on Space** → already mitigated (lazy load + frame cap). Lower `MAX_DETECT_FRAMES` env var if needed.
- **Slow first request** → cold start, expected. Subsequent requests are fast.
