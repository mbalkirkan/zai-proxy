FROM python:3.12-slim AS pow-builder

WORKDIR /build
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY zai_proxy/deepseek_pow_solver.c .
RUN gcc -O3 -std=c11 deepseek_pow_solver.c -o deepseek-pow-solver

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=pow-builder /build/deepseek-pow-solver /usr/local/bin/deepseek-pow-solver

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/').read()"

CMD ["sh", "-c", "uvicorn zai_proxy.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]
