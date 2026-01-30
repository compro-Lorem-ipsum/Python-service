from __future__ import annotations

from fastapi.responses import JSONResponse


def failure_to_response(payload: dict) -> JSONResponse:
    error_text = str(payload.get("error") or payload.get("message") or "").strip().lower()

    if "no face detected" in error_text or "multiple faces detected" in error_text:
        status_code = 422
    elif "failed to decode" in error_text:
        status_code = 400
    elif "collection not found" in error_text:
        status_code = 503
    elif "payload too large" in error_text:
        status_code = 413
    elif "not found" in error_text:
        status_code = 404
    else:
        status_code = 400

    return JSONResponse(status_code=status_code, content=payload)
