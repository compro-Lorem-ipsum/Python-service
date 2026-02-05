import anyio
from fastapi import APIRouter, Form, HTTPException
from api.responses import failure_to_response
from core.container import container
from core.config import settings
from services.image_loader import download_image_from_url

router = APIRouter()

@router.post("/enroll", tags=["Enroll"])
async def enroll_employee(
    employee_id: str = Form(...),
    image_url: str = Form(...)
):
    try:
        face = container.face_service
        db = container.milvus_db

        bytes_img = await download_image_from_url(image_url, settings.MAX_IMAGE_BYTES)

        async with container.infer_semaphore:
            result = await anyio.to_thread.run_sync(face.extract_embedding_from_bytes, bytes_img)

        if not result.get("success", False):
            return failure_to_response(result)

        insert_result = await anyio.to_thread.run_sync(db.insert_embedding, employee_id, result["embedding"])

        return {
            "success": True,
            "employee_id": employee_id,
            "message": "Employee enrolled successfully",
            "detection_score": result.get("det_score"),
            "insert_result": insert_result
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify", tags=["Verify"])
async def verify_face(
    image_url: str = Form(...),
    threshold: float = Form(None),
):
    try:
        face = container.face_service
        db = container.milvus_db

        bytes_img = await download_image_from_url(image_url,settings.MAX_IMAGE_BYTES)

        async with container.infer_semaphore:
            extract_result = await anyio.to_thread.run_sync(face.extract_embedding_from_bytes, bytes_img)

        if not extract_result.get("success", False):
            return failure_to_response(extract_result)

        embedding = extract_result["embedding"]
        threshold = threshold if threshold else settings.SIMILARITY_THRESHOLD

        search_result = await anyio.to_thread.run_sync(db.search_similar, embedding, threshold)

        if not search_result.get("success", False):
            return failure_to_response(search_result)

        if not search_result.get("matched", False):
            return {
                "success": True,
                "matched": False,
                "similarity": search_result.get("similarity", 0.0),
                "threshold": threshold,
                "message": "No match found"
            }

        return {
            "success": True,
            "matched": True,
            "employee_id": search_result["employee_id"],
            "similarity": search_result["similarity"],
            "threshold": threshold,
            "detection_score": extract_result.get("det_score")
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{employee_id}", tags=["Delete"])
async def delete_employee(employee_id: str):
    try:
        db = container.milvus_db
        delete_result = await anyio.to_thread.run_sync(db.delete_by_employee_id, employee_id)

        if not delete_result.get("success", False):  
            return failure_to_response(delete_result)

        return {
            "success": True,
            "employee_id": employee_id,
            "message": "Employee data deleted successfully"
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/extract/embedding", tags=["Extract"])
async def extract_embedding(image_url: str = Form(...)):
    try:
        face = container.face_service
        bytes_img = await download_image_from_url(image_url, settings.MAX_IMAGE_BYTES)

        async with container.infer_semaphore:
            result = face.extract_embedding_from_bytes(bytes_img)

        if not result.get("success", False):
            return failure_to_response(result)

        # Convert numpy to list
        result["embedding"] = result["embedding"].tolist()
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/employees", tags=["List"])
async def list_employees():
    db = container.milvus_db
    result = await anyio.to_thread.run_sync(db.list_employee_ids)
    if not result.get("success", False):
        return failure_to_response(result)
    return result
