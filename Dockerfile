# Multi-stage production container for CortexRuntime
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt* ./
RUN pip install --no-cache-dir --user --upgrade pip \
    && if [ -f requirements.txt ]; then pip install --no-cache-dir --user -r requirements.txt; fi

# Runtime Stage
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -u 10001 -m cortexuser

COPY --from=builder /root/.local /home/cortexuser/.local
COPY --chown=cortexuser:cortexuser cortex/ /app/cortex/
COPY --chown=cortexuser:cortexuser cortex.yaml /app/cortex.yaml

USER cortexuser
ENV PATH=/home/cortexuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000 9090

ENTRYPOINT ["python", "-m", "cortex"]
CMD ["serve", "--port", "8000"]
