import re
import requests
from fastapi import UploadFile, HTTPException

# async def read_image_with_limit(upload: UploadFile, limit_bytes: int):
#     data = await upload.read()
#     if len(data) > limit_bytes:
#         return None, {"success": False, "error": "Image payload too large"}
#     return data, None

async def download_image_from_url(url: str, max_bytes: int) -> bytes:
    try:
        if "drive.google.com" in url:
            match = re.search(r"/d/([^/]+)", url)
            if match:
                file_id = match.group(1)
                url = f"https://drive.google.com/uc?id={file_id}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        if not response.headers.get("Content-Type", "").startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="URL does not point to a valid image"
            )

        if len(response.content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="Image payload too large"
            )

        return response.content

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download image: {str(e)}"
        )

