# ==============================================================================
# Whitematter ML Platform - Multi-stage Dockerfile
# ==============================================================================
#
# Build:   docker build -t whitematter .
# Run:     docker run -p 8080:8080 -p 5173:5173 -v whitematter-db:/data whitematter
#
# Environment variables:
#   WHITEMATTER_DATA_DIR  - Directory for SQLite database (default: /data)
#   WORKERS               - Number of training workers (default: 2)
#   ANTHROPIC_API_KEY     - API key for LLM features (optional)
#
# Storage:
#   All data (models, datasets, blobs) is stored in a single SQLite database.
#   Only the database file is persisted - no other files are written to disk.
#   Mount /data to persist the database between container restarts.
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build C++ core library and examples
# ------------------------------------------------------------------------------
FROM ubuntu:22.04 AS cpp-builder

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy C++ source files
COPY core/ core/
COPY datasets/ datasets/
COPY examples/ examples/
COPY tests/ tests/
COPY Makefile .

# Build the C++ library and training executables
RUN make -j$(nproc) all

# ------------------------------------------------------------------------------
# Stage 2: Build Python bindings
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS python-builder

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pybind11 first
RUN pip install --no-cache-dir pybind11 wheel setuptools

# Copy C++ source for Python bindings
COPY core/*.cpp core/*.h ./
COPY bindings/*.cpp ./
COPY platform/setup.py .

# Build the Python extension
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# ------------------------------------------------------------------------------
# Stage 3: Build frontend
# ------------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy source files
COPY frontend/ .

# Build for production
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 4: Final runtime image
# ------------------------------------------------------------------------------
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    make \
    libomp-dev \
    libgomp1 \
    nginx \
    supervisor \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built C++ binaries from cpp-builder
COPY --from=cpp-builder /app/build/ /app/build/

# Copy Python wheel and install
COPY --from=python-builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy built frontend from frontend-builder
COPY --from=frontend-builder /app/frontend/dist/ /app/frontend/dist/

# Copy platform code
COPY platform/ platform/
COPY core/ core/

# Install Python dependencies
COPY platform/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir sqlalchemy httpx

# Create data directory (only for SQLite database file and temp files)
RUN mkdir -p /data

# Configure nginx for frontend
RUN rm /etc/nginx/sites-enabled/default
COPY <<'EOF' /etc/nginx/sites-available/whitematter
server {
    listen 5173;
    root /app/frontend/dist;
    index index.html;

    # Serve static files
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://127.0.0.1:8080/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
    }

    # Proxy other backend routes
    location ~ ^/(datasets|models|train|predict|design|config|health|workers)(.*)$ {
        proxy_pass http://127.0.0.1:8080/$1$2;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
    }
}
EOF
RUN ln -s /etc/nginx/sites-available/whitematter /etc/nginx/sites-enabled/

# Configure supervisor to manage all processes
COPY <<'EOF' /etc/supervisor/conf.d/whitematter.conf
[supervisord]
nodaemon=true
user=root

[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:api]
command=python -u server.py --port 8080 --host 0.0.0.0
directory=/app/platform
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
environment=WHITEMATTER_DATA_DIR="/data"

[program:worker]
command=python -u run_worker.py --count %(ENV_WORKERS)s
directory=/app/platform
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
environment=WHITEMATTER_DATA_DIR="/data"
EOF

# Environment variables
ENV WHITEMATTER_DATA_DIR=/data
ENV WORKERS=2
ENV PYTHONUNBUFFERED=1

# Expose ports
# 8080 - API server
# 5173 - Frontend (via nginx)
EXPOSE 8080 5173

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start supervisor (manages nginx, api server, and workers)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
