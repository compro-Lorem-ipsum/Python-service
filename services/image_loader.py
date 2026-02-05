import re

import httpx
from fastapi import HTTPException

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        timeout = httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0)
        _client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
        )
    return _client


async def download_image_from_url(url: str, max_bytes: int) -> bytes:
    try:
        if "drive.google.com" in url:
            match = re.search(r"/d/([^/]+)", url)
            if match:
                file_id = match.group(1)
                url = f"https://drive.google.com/uc?id={file_id}"

        client = _get_client()

        async with client.stream("GET", url) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail="URL does not point to a valid image"
                )

            data = bytearray()
            total = 0

            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue

                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Image payload too large"
                    )

                data.extend(chunk)

        return bytes(data)

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=408,
            detail="Timeout while downloading image"
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image: {exc}"
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image: HTTP {exc.response.status_code}"
        )
