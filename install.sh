#!/usr/bin/env bash
# One-line platform boot: run the full Whitematter ML platform with Docker.
# Usage: curl -sSL https://raw.githubusercontent.com/OWNER/whitematter/main/install.sh | bash
# Or from repo: ./install.sh

set -e

# Check Docker
if ! command -v docker &>/dev/null; then
  echo "Docker is required. Install it from https://docs.docker.com/get-docker/"
  exit 1
fi

# Prefer "docker compose" (v2) over "docker-compose" (v1)
COMPOSE_CMD="docker compose"
if ! docker compose version &>/dev/null 2>&1; then
  if command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="docker-compose"
  else
    echo "Docker Compose is required. Install it or use Docker Desktop."
    exit 1
  fi
fi

# Run from repo root if we're inside the repo (have docker-compose.yml)
REPO_ROOT=""
if [ -f "docker-compose.yml" ]; then
  REPO_ROOT="$(pwd)"
elif [ -f "install.sh" ]; then
  REPO_ROOT="$(pwd)"
fi

if [ -n "$REPO_ROOT" ]; then
  cd "$REPO_ROOT"
  echo "Starting Whitematter platform from $(pwd)..."
  export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
  $COMPOSE_CMD up -d
  echo "Platform is starting. API: http://localhost:8080  Frontend: http://localhost:5173"
  echo "To pass your API key: ANTHROPIC_API_KEY=sk-... ./install.sh"
  exit 0
fi

# Not in repo: clone and run
REPO_URL="${WHITEMATTER_REPO_URL:-https://github.com/hwang2409/whitematter}"
CLONE_DIR="./whitematter"
if [ ! -d "$CLONE_DIR/.git" ]; then
  echo "Cloning whitematter into $CLONE_DIR..."
  git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
fi
cd "$CLONE_DIR"
echo "Starting Whitematter platform from $(pwd)..."
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
$COMPOSE_CMD up -d
echo "Platform is starting. API: http://localhost:8080  Frontend: http://localhost:5173"
echo "To pass your API key: ANTHROPIC_API_KEY=sk-... ./install.sh"
