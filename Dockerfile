FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV STRUCTURE_DATA_DIR=/data/ds-structure
ENV STRUCTURE_MAX_JOBS=3
ENV PORT=8080
EXPOSE 8080
VOLUME ["/data/ds-structure"]

CMD ["uvicorn", "ds_structure_server.main:app", "--host", "0.0.0.0", "--port", "8080"]
