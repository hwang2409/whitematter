# Whitematter Frontend (Next.js)

This is the Next.js frontend for the Whitematter platform. It talks to the FastAPI backend for auth, datasets, training, and models.

## How to run

**`npm run dev` runs only the frontend** (Next.js on http://localhost:3000). It does **not** start the backend API.

You need **both** running to use login, sign up, and the rest of the app:

1. **Backend** (in another terminal):
   ```bash
   cd platform
   source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
   python server.py
   ```
   API will be at http://localhost:8080.

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   App will be at http://localhost:3000. API requests are proxied to the backend via Next.js rewrites.

If the backend is not running, **Sign up** and **Sign in** will fail (e.g. connection refused or "Registration failed").

## Scripts

- `npm run dev` – start Next.js dev server (port 3000)
- `npm run build` – production build
- `npm run start` – run production build (after `npm run build`)
- `npm run lint` – run ESLint

## Tech

- Next.js 15 (App Router)
- React 19
- TypeScript
