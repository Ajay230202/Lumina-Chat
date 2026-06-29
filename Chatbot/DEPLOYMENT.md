# Lumina deployment checklist

## Frontend on Vercel

The repository includes a root `vercel.json` with conditional commands that work whether Vercel starts from the repository root or from `Chatbot/frontend`. If you configure Vercel manually, set the Vercel project root to `Chatbot/frontend` and use the default Next.js build command:

```bash
npm run build
```

Add this Vercel environment variable before deploying:

```bash
NEXT_PUBLIC_API_URL=https://your-deployed-backend-url
```

Do not leave `NEXT_PUBLIC_API_URL` unset in production. The frontend calls the FastAPI backend endpoints under `/api/*`, so a deployed frontend must point at a publicly reachable backend URL. If your deployed frontend shows a Vercel 404, confirm Vercel is building the Next.js app in `Chatbot/frontend` and not an empty repository root.

## Backend hosting

The backend is a FastAPI application and is not deployed by the Vercel Next.js build. Deploy `Chatbot/backend` to a Python-capable host such as Render, Railway, Fly.io, a VPS, or another ASGI-compatible platform.

Install dependencies from:

```bash
pip install -r requirements.txt
```

Start the API with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Required backend environment variables are listed in `backend/env.example`. Set real secrets in the hosting provider dashboard; do not commit real API keys to the repository.

## Where the real API keys live

The real keys are not stored in this repository. Files named `env.example` are templates only; they show the variable names the app expects, but the values should be entered as secrets in the relevant hosting dashboard.

For local development, create `Chatbot/backend/.env` from `Chatbot/backend/env.example` and fill in your real backend keys there. This `.env` file is ignored by git and must not be committed.

For the deployed backend, add the backend secrets in your Python hosting provider's environment-variable settings:

- `NVIDIA_API_KEY`
- `NVIDIA_BASE_URL`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_KEY`
- `GROQ_API_KEY` if you use the optional Groq path
- `CORS_ORIGINS`

For the deployed Vercel frontend, only add public frontend configuration in Vercel's environment-variable settings, especially:

- `NEXT_PUBLIC_API_URL`

Do not put private backend API keys in Vercel frontend variables. Any variable prefixed with `NEXT_PUBLIC_` is bundled for browser-side use and is visible to users.

## CORS

Set the backend `CORS_ORIGINS` value to include your Vercel URL, for example:

```bash
CORS_ORIGINS=https://your-app.vercel.app,http://localhost:3000
```

## Health check

After deploying the backend, verify:

```bash
curl https://your-deployed-backend-url/health
```

Expected response:

```json
{"status":"ok","message":"Multimodal RAG API is healthy"}
```
