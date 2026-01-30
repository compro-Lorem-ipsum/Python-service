import asyncio

from services.face_service import FaceRecognitionService
from services.milvus_db import MilvusDB
from core.config import settings


class Container:
    def __init__(self):
        self.face_service = FaceRecognitionService(
            model_name=settings.MODEL_NAME,
            providers=settings.FACE_PROVIDERS,
            det_size=settings.DET_SIZE,
        )
        self.milvus_db = MilvusDB()
        self.infer_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_INFERENCE)

    def startup(self) -> None:
        self.face_service.warmup()

        if not self.milvus_db.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT):
            print("Warning: Failed to connect to Milvus.")
            return

        created = self.milvus_db.create_collection()
        if created is None:
            print("Warning: Milvus collection not ready.")

    def health(self) -> dict:
        return self.milvus_db.health()


container = Container()
