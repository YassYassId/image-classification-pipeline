# CPU-only Dockerfile for the CIFAR-10 DVC pipeline (with DVC GDrive support)
FROM python:3.10-slim

# Avoid interactive prompts and reduce noise
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DVC_NO_ANALYTICS=1 \
    DVC_NO_BROWSER=1

# System deps (git for DVC, build tools for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list and install first (better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Ensure directories exist (mount will override /app/data when using -v)
RUN mkdir -p data/raw/cifar10 data/processed models metrics/plots

# Lightweight entrypoint to optionally configure DVC and pull data on start
RUN printf '#!/usr/bin/env bash\nset -e\n\n# If a service account is provided, switch DVC remote to service-account mode.\nif [ -n "${GDRIVE_CREDENTIALS_DATA}" ]; then\n  echo "[entrypoint] Using service account for DVC GDrive"\n  dvc remote modify gdrive-remote gdrive_use_service_account true || true\nelse\n  dvc remote modify gdrive-remote gdrive_use_service_account false || true\nfi\n\n# Attempt to pull data unless skipped\nif [ -f dvc.yaml ] && [ "${SKIP_DVC_PULL}" != "1" ]; then\n  echo "[entrypoint] Running dvc pull (set SKIP_DVC_PULL=1 to skip)"\n  dvc pull || echo "[entrypoint] dvc pull failed (likely auth). Continue without cached data."\nfi\n\nexec "$@"\n' > /usr/local/bin/entrypoint.sh \
 && chmod +x /usr/local/bin/entrypoint.sh

# Use entrypoint so commands like `... image dvc repro` still work
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command prints help; typical usage mounts data and runs `dvc repro`
CMD ["bash", "-lc", "echo 'Container ready. Mount your data/ as /app/data and run: dvc repro'"]
