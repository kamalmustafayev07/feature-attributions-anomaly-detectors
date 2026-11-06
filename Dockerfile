# CPU-only reference environment. Data is mounted at run time (never baked in),
# so the image builds without any access to Zenodo.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/  # TODO(verify): uv image tag matches the uv version used to produce uv.lock

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install locked third-party dependencies first (cached layer), project last.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN uv sync --frozen --no-dev

VOLUME ["/app/data", "/app/results"]

CMD ["uv", "run", "python", "scripts/run_all.py", "--level", "1"]
