# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder
#   Install all Python dependencies into an isolated prefix so the final image
#   only copies what is needed (no pip cache, no wheel debris).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
#   Thin final image: only the installed packages + application code.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Roitelet LLM" \
      org.opencontainers.image.description="Local-first adaptive LLM router" \
      org.opencontainers.image.authors="Warith Harchaoui <warith@deraison.ai>" \
      org.opencontainers.image.source="https://github.com/warithharchaoui/roitelet-llm"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/install/bin:$PATH"

# Copy installed packages from builder stage.
COPY --from=builder /install /usr/local

# Create a non-root user for security.
RUN groupadd --gid 1001 roitelet \
 && useradd --uid 1001 --gid roitelet --shell /bin/bash --create-home roitelet

WORKDIR /app

# Copy only the application source (data volume is mounted at runtime).
COPY --chown=roitelet:roitelet app/           ./app/
COPY --chown=roitelet:roitelet data/bootstrap ./data/bootstrap/
COPY --chown=roitelet:roitelet web/           ./web/
COPY --chown=roitelet:roitelet start.sh .
COPY --chown=roitelet:roitelet .env.example .

RUN chmod +x /app/start.sh \
 && mkdir -p /app/data/conversations /app/data/telemetry /app/data/runtime \
 && chown -R roitelet:roitelet /app/data

USER roitelet

EXPOSE 8000

# Health check — lightweight ping of the dedicated health endpoint.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz').read()" \
  || exit 1

CMD ["/app/start.sh"]
