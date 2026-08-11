from __future__ import annotations

import datetime
import os
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger
from minio import Minio


def build_minio_client_from_env() -> Minio:
    endpoint = os.getenv("S3_COMPATIBLE_ENDPOINT", "localhost:9000").strip()
    access_key = os.getenv("S3_COMPATIBLE_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("S3_COMPATIBLE_SECRET_KEY", "minioadmin")
    region = os.getenv("S3_REGION", "us-east-1")
    secure = os.getenv("S3_SECURE", "true").lower() not in {"0", "false", "no"}
    if "://" in endpoint:
        parsed = urlparse(endpoint)
        host = parsed.netloc or parsed.path
        if parsed.scheme == "http":
            secure = False
        elif parsed.scheme == "https":
            secure = True
    else:
        host = endpoint
    return Minio(host, access_key=access_key, secret_key=secret_key, secure=secure, region=region)


def upload_file_to_minio(file_path: str, bucket_name: str, object_name: str) -> str | None:
    """Sync upload + presigned GET URL (safe to call from worker threads)."""
    client = build_minio_client_from_env()
    try:
        client.fput_object(bucket_name, object_name, file_path)
    except Exception as exc:
        logger.exception(f"Failed uploading {file_path} to s3://{bucket_name}/{object_name}: {exc}")
        return None
    try:
        return client.presigned_get_object(
            bucket_name,
            object_name,
            expires=datetime.timedelta(seconds=604800),
        )
    except Exception as exc:
        logger.exception(f"Failed creating presigned URL for s3://{bucket_name}/{object_name}: {exc}")
        return None


def upload_bytes_to_minio(
    data: bytes,
    bucket_name: str,
    object_name: str,
    *,
    content_type: str = "application/octet-stream",
) -> str | None:
    import tempfile

    _ = content_type
    suffix = Path(object_name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        tmp_path = handle.name
    try:
        return upload_file_to_minio(tmp_path, bucket_name, object_name)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
