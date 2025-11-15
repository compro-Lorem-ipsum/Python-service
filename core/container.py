from services.face_service import FaceRecognitionService
from services.milvus_db import MilvusDB
from core.config import settings

class Container:
    def __init__(self):
        self.face_service = FaceRecognitionService()
        self.milvus_db = MilvusDB()

        if not self.milvus_db.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT):
            print("Warning: Failed to connect to Milvus.")
        else:
            self.milvus_db.create_collection()

container = Container()
