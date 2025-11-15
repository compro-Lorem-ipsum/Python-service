from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.container import container
from core.config import settings

router = APIRouter()

@router.post("/enroll", tags=["Enroll"])
async def enroll_employee(
    employee_id: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        face = container.face_service
        db = container.milvus_db

        bytes_img = await image.read()
        result = face.extract_embedding_from_bytes(bytes_img)

        if not result.get("success", False):
            return result

        insert_result = db.insert_embedding(employee_id, result["embedding"])

        return {
            "success": True,
            "employee_id": employee_id,
            "message": "Employee enrolled successfully",
            "detection_score": result.get("det_score"),
            "insert_result": insert_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify", tags=["Verify"])
async def verify_face(
    image: UploadFile = File(...),
    threshold: float = Form(None),
):
    try:
        face = container.face_service
        db = container.milvus_db

        bytes_img = await image.read()
        extract_result = face.extract_embedding_from_bytes(bytes_img)

        if not extract_result.get("success", False):
            return extract_result

        embedding = extract_result["embedding"]
        threshold = threshold if threshold else settings.SIMILARITY_THRESHOLD

        search_result = db.search_similar(embedding, threshold)

        if not search_result.get("success", False):
            return search_result

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{employee_id}", tags=["Delete"])
async def delete_employee(employee_id: str):
    try:
        db = container.milvus_db
        delete_result = db.delete_by_employee_id(employee_id)

        if not delete_result.get("success", False):  
            if "not found" in delete_result.get("error", "").lower():
                raise HTTPException(status_code=404, detail=delete_result["error"])
            else:
                raise HTTPException(status_code=500, detail=delete_result.get("error", "Unknown error"))

        return {
            "success": True,
            "employee_id": employee_id,
            "message": "Employee data deleted successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/extract/embedding", tags=["Extract"])
async def extract_embedding(image: UploadFile = File(...)):
    try:
        face = container.face_service
        bytes_img = await image.read()
        result = face.extract_embedding_from_bytes(bytes_img)

        if not result.get("success", False):
            return result

        # Convert numpy to list
        result["embedding"] = result["embedding"].tolist()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/employees", tags=["List"])
async def list_employees():
    db = container.milvus_db
    result = db.list_employee_ids()
    return result
