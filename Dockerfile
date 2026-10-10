# Execution service image. Build:  docker build -t trading-engine .
# Run:    docker run --env-file .env -v "$PWD/config:/app/config:ro" trading-engine run --config config/local.toml
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# Never run as root; the service needs no filesystem writes.
RUN useradd --system --no-create-home engine
USER engine

EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8081/healthz', timeout=4).status == 200 else 1)"

ENTRYPOINT ["python", "-m", "engine.service"]
CMD ["run", "--config", "config/local.toml"]
