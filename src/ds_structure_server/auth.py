from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status


def require_service_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.getenv("STRUCTURE_SERVICE_TOKEN", "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="STRUCTURE_SERVICE_TOKEN is not configured",
        )
    raw = (authorization or "").strip()
    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()
    if raw != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


ServiceAuth = Annotated[None, Depends(require_service_token)]
